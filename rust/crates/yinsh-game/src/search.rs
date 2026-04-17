//! Self-play, battle, and best-move orchestration for YINSH.
//!
//! All tree search uses `core_game::mcts::MctsSearch<YinshBoard>` directly —
//! no game-specific MCTS code. Yinsh's policy is a single flat 7-channel tensor
//! (847 entries) consumed via the `PolicyIndex::Sum` mechanism that
//! `core_game::mcts` already supports.

use rand::RngExt;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;

use core_game::game::{Game, NNGame, Outcome, Player};
use core_game::mcts::arena::NodeId;
use core_game::mcts::search::{
    CpuctStrategy, ForcedExploration, MctsSearch, RootNoise, SearchParams,
};

use crate::board::{Phase, YinshBoard, YinshMove};
use crate::board_encoding::{NUM_CHANNELS, RESERVE_SIZE};
use crate::hex::GRID_SIZE;
use crate::move_encoding::POLICY_SIZE;

const BOARD_FLAT: usize = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;

/// Eval callback: `(boards_flat[N*BOARD_FLAT], reserves_flat[N*RESERVE_SIZE], n) ->
/// (policy[N*POLICY_SIZE], value[N])`.
pub type EvalFn = Box<
    dyn Fn(&[f32], &[f32], usize) -> Result<(Vec<f32>, Vec<f32>), String> + Send + Sync,
>;

/// Progress callback: `(finished, total, active, total_moves)`.
pub type ProgressFn = Box<dyn Fn(u32, u32, u32, u32) + Send + Sync>;

/// Phase of a turn — useful as a categorical training-data flag.
#[inline]
fn phase_code(p: Phase) -> u8 {
    match p {
        Phase::Setup => 0,
        Phase::Normal => 1,
        Phase::RemoveRow => 2,
        Phase::RemoveRing => 3,
    }
}

#[inline]
fn make_search(simulations: usize, c_puct: f32) -> MctsSearch<YinshBoard> {
    let mut s = MctsSearch::<YinshBoard>::new(simulations + 64);
    s.params = SearchParams::new(
        CpuctStrategy::Constant { c_puct },
        ForcedExploration::None,
        RootNoise::None,
    );
    s
}

/// Run an MCTS search to its simulation budget on a single game.
/// Stashes leaves, calls `eval_fn` in batches of `play_batch_size`,
/// expands+backprops until each game has reached its simulation cap.
fn run_simulations_single(
    search: &mut MctsSearch<YinshBoard>,
    sim_cap: usize,
    play_batch_size: usize,
    eval_fn: &EvalFn,
) -> Result<(), String> {
    let mut done = 0usize;
    let mut flat_boards = vec![0f32; play_batch_size * BOARD_FLAT];
    let mut flat_reserves = vec![0f32; play_batch_size * RESERVE_SIZE];
    while done < sim_cap {
        let want = (sim_cap - done).min(play_batch_size);
        let leaves = search.select_leaves(want);
        if leaves.is_empty() {
            break;
        }
        let nl = leaves.len();
        for (k, &leaf) in leaves.iter().enumerate() {
            let (b, r) = search.encode_leaf(leaf);
            flat_boards[k * BOARD_FLAT..(k + 1) * BOARD_FLAT].copy_from_slice(&b);
            flat_reserves[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE].copy_from_slice(&r);
        }
        let (policy_flat, values) = eval_fn(
            &flat_boards[..nl * BOARD_FLAT],
            &flat_reserves[..nl * RESERVE_SIZE],
            nl,
        )?;
        let policies: Vec<Vec<f32>> = (0..nl)
            .map(|i| policy_flat[i * POLICY_SIZE..(i + 1) * POLICY_SIZE].to_vec())
            .collect();
        search.expand_and_backprop(&policies, &values);
        done += nl;
    }
    Ok(())
}

/// Best move for a single position. Used by the interactive `PyYinshGame.best_move`.
pub fn best_move_core(
    board: &YinshBoard,
    simulations: usize,
    c_puct: f32,
    eval_fn: EvalFn,
) -> Result<YinshMove, String> {
    if board.is_game_over() {
        return Err("Game is already over".to_string());
    }

    let mut search = make_search(simulations, c_puct);

    // Initial root eval.
    let mut root_board = vec![0f32; BOARD_FLAT];
    let mut root_reserve = vec![0f32; RESERVE_SIZE];
    board.encode_board(&mut root_board, &mut root_reserve);
    let (root_policy, _root_value) = eval_fn(&root_board, &root_reserve, 1)?;
    search.init(board, &root_policy);

    run_simulations_single(&mut search, simulations, 8, &eval_fn)?;

    search
        .best_move()
        .ok_or_else(|| "No legal moves".to_string())
}

// ---------------------------------------------------------------------------
// Battle
// ---------------------------------------------------------------------------

/// Pure-Rust battle result (no PyO3 types).
#[derive(Clone, Debug, Default)]
pub struct BattleResult {
    pub wins_model1: u32,
    pub wins_model2: u32,
    pub draws: u32,
    pub wins_white: u32,
    pub wins_black: u32,
    pub game_lengths: Vec<u32>,
}

/// Play `num_games` between two evaluation functions. Games `0..num_games/2`
/// have model1 as Player1 (white), the rest are reversed. Returns stats from
/// model1's perspective.
pub fn play_battle_core(
    num_games: usize,
    simulations: usize,
    max_moves: u32,
    c_puct: f32,
    play_batch_size: usize,
    eval_fn1: EvalFn,
    eval_fn2: EvalFn,
    progress_fn: Option<ProgressFn>,
) -> Result<BattleResult, String> {
    let half = num_games / 2;

    // Per-game state.
    let mut boards: Vec<YinshBoard> = (0..num_games).map(|_| YinshBoard::new()).collect();
    let mut active = vec![true; num_games];
    let mut move_counts = vec![0u32; num_games];

    let mut result = BattleResult::default();
    let mut total_moves: u32 = 0;
    let mut finished: u32 = 0;

    let model1_for_p1 = |gi: usize| gi < half;
    let pick_eval =
        |gi: usize, p: Player| -> bool { model1_for_p1(gi) == (p == Player::Player1) };

    while active.iter().any(|&a| a) {
        // Step every active game by one move.
        for gi in 0..num_games {
            if !active[gi] {
                continue;
            }

            let use_fn1 = pick_eval(gi, boards[gi].next_player());
            let eval_ref: &EvalFn = if use_fn1 { &eval_fn1 } else { &eval_fn2 };

            let mut search = make_search(simulations, c_puct);

            let mut root_board = vec![0f32; BOARD_FLAT];
            let mut root_reserve = vec![0f32; RESERVE_SIZE];
            boards[gi].encode_board(&mut root_board, &mut root_reserve);
            let (root_policy, _) = eval_ref(&root_board, &root_reserve, 1)?;
            search.init(&boards[gi], &root_policy);

            run_simulations_single(&mut search, simulations, play_batch_size.max(1), eval_ref)?;

            let mv = search.best_move().unwrap_or_else(YinshBoard::pass_move);
            boards[gi]
                .play_move(&mv)
                .map_err(|e| format!("battle game {} illegal move: {}", gi, e))?;
            move_counts[gi] += 1;
            total_moves += 1;

            if boards[gi].is_game_over() || move_counts[gi] >= max_moves {
                active[gi] = false;
                finished += 1;
                result.game_lengths.push(move_counts[gi]);
                match boards[gi].outcome() {
                    Outcome::WonBy(winner) => {
                        let m1_won = model1_for_p1(gi) == (winner == Player::Player1);
                        if m1_won {
                            result.wins_model1 += 1;
                        } else {
                            result.wins_model2 += 1;
                        }
                        if winner == Player::Player1 {
                            result.wins_white += 1;
                        } else {
                            result.wins_black += 1;
                        }
                    }
                    _ => result.draws += 1,
                }
            }
        }

        if let Some(pfn) = &progress_fn {
            let active_count = active.iter().filter(|&&a| a).count() as u32;
            pfn(finished, num_games as u32, active_count, total_moves);
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Self-play
// ---------------------------------------------------------------------------

/// Pure-Rust self-play result. All training samples are concatenated flat.
#[derive(Clone, Debug, Default)]
pub struct SelfPlayResult {
    pub board_data: Vec<f32>,
    pub reserve_data: Vec<f32>,
    pub policy_data: Vec<f32>,
    pub value_targets: Vec<f32>,
    pub value_only_flags: Vec<bool>,
    pub phase_flags: Vec<u8>,
    pub num_samples: usize,

    pub wins_p1: u32,
    pub wins_p2: u32,
    pub draws: u32,
    pub total_moves: u32,
    pub game_lengths: Vec<u32>,
    pub decisive_lengths: Vec<u32>,

    pub full_search_turns: u32,
    pub total_turns: u32,

    /// Up to 3 short text labels summarizing decisive games (no full board render).
    pub sample_summaries: Vec<String>,
}

/// One training record per turn, accumulated per game.
struct TurnRecord {
    board: Vec<f32>,
    reserve: Vec<f32>,
    policy: Vec<f32>,
    player: Player,
    value_only: bool,
    phase: u8,
}

/// Run `num_games` of self-play in parallel. Each game uses its own
/// `MctsSearch<YinshBoard>`; leaves are batched across games via
/// `play_batch_size` for better GPU utilization. Returns a flat `SelfPlayResult`
/// suitable for direct ingestion into the Python replay buffer.
pub fn play_selfplay_core(
    num_games: usize,
    simulations: usize,
    max_moves: u32,
    temperature: f32,
    temp_threshold: u32,
    c_puct: f32,
    dir_alpha: f32,
    dir_epsilon: f32,
    play_batch_size: usize,
    playout_cap_p: f32,
    fast_cap: usize,
    eval_fn: EvalFn,
    progress_fn: Option<ProgressFn>,
) -> Result<SelfPlayResult, String> {
    let use_playout_cap = playout_cap_p > 0.0;

    let mut boards: Vec<YinshBoard> = (0..num_games).map(|_| YinshBoard::new()).collect();
    let mut searches: Vec<MctsSearch<YinshBoard>> =
        (0..num_games).map(|_| make_search(simulations, c_puct)).collect();
    let mut active = vec![true; num_games];
    let mut move_counts = vec![0u32; num_games];
    let mut histories: Vec<Vec<TurnRecord>> = (0..num_games).map(|_| Vec::new()).collect();

    let mut result = SelfPlayResult::default();
    let mut finished_count: u32 = 0;
    let mut rng = rand::rng();

    while active.iter().any(|&a| a) {
        let active_games: Vec<usize> = (0..num_games).filter(|&gi| active[gi]).collect();
        if active_games.is_empty() {
            break;
        }
        let n = active_games.len();
        result.total_turns += n as u32;

        // Decide full vs fast search per active game.
        let is_full: Vec<bool> = if use_playout_cap {
            (0..n).map(|_| rng.random::<f32>() < playout_cap_p).collect()
        } else {
            vec![true; n]
        };
        let sim_caps: Vec<usize> = is_full
            .iter()
            .map(|&f| if f { simulations } else { fast_cap })
            .collect();
        result.full_search_turns += is_full.iter().filter(|&&f| f).count() as u32;

        // Initial root NN eval — batched across all active games.
        let mut root_boards = vec![0f32; n * BOARD_FLAT];
        let mut root_reserves = vec![0f32; n * RESERVE_SIZE];
        for (i, &gi) in active_games.iter().enumerate() {
            boards[gi].encode_board(
                &mut root_boards[i * BOARD_FLAT..(i + 1) * BOARD_FLAT],
                &mut root_reserves[i * RESERVE_SIZE..(i + 1) * RESERVE_SIZE],
            );
        }
        let (init_policy, _) = eval_fn(&root_boards, &root_reserves, n)?;
        for (i, &gi) in active_games.iter().enumerate() {
            let policy_slice = &init_policy[i * POLICY_SIZE..(i + 1) * POLICY_SIZE];
            searches[gi].init(&boards[gi], policy_slice);
            if is_full[i] {
                searches[gi].apply_root_dirichlet(dir_alpha, dir_epsilon);
            }
        }

        // Cross-game batched simulations.
        let mut game_sims = vec![0usize; n];
        loop {
            let mut leaf_ids: Vec<NodeId> = Vec::new();
            let mut leaf_game_idx: Vec<usize> = Vec::new();

            // Collect up to `play_batch_size` leaves per round, one leaf at a time per game.
            for _round in 0..play_batch_size {
                let mut any_collected = false;
                for (i, &gi) in active_games.iter().enumerate() {
                    if game_sims[i] >= sim_caps[i] {
                        continue;
                    }
                    let leaves = searches[gi].select_leaves(1);
                    let count = leaves.len();
                    if count > 0 {
                        any_collected = true;
                    }
                    for leaf in leaves {
                        leaf_ids.push(leaf);
                        leaf_game_idx.push(i);
                    }
                    game_sims[i] += count;
                }
                if !any_collected {
                    break;
                }
            }
            if leaf_ids.is_empty() {
                break;
            }

            let nl = leaf_ids.len();
            let mut leaf_boards = vec![0f32; nl * BOARD_FLAT];
            let mut leaf_reserves = vec![0f32; nl * RESERVE_SIZE];
            for (k, (&leaf, &i)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
                let gi = active_games[i];
                let (b, r) = searches[gi].encode_leaf(leaf);
                leaf_boards[k * BOARD_FLAT..(k + 1) * BOARD_FLAT].copy_from_slice(&b);
                leaf_reserves[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE].copy_from_slice(&r);
            }
            let (leaf_policy, leaf_values) = eval_fn(&leaf_boards, &leaf_reserves, nl)?;

            // Group results back per game and call expand_and_backprop in stash order.
            let mut per_game_policies: Vec<Vec<Vec<f32>>> = (0..n).map(|_| Vec::new()).collect();
            let mut per_game_values: Vec<Vec<f32>> = (0..n).map(|_| Vec::new()).collect();
            for (k, &i) in leaf_game_idx.iter().enumerate() {
                per_game_policies[i].push(
                    leaf_policy[k * POLICY_SIZE..(k + 1) * POLICY_SIZE].to_vec(),
                );
                per_game_values[i].push(leaf_values[k]);
            }
            for (i, &gi) in active_games.iter().enumerate() {
                if per_game_policies[i].is_empty() {
                    continue;
                }
                searches[gi].expand_and_backprop(&per_game_policies[i], &per_game_values[i]);
            }

            if game_sims.iter().zip(sim_caps.iter()).all(|(s, c)| *s >= *c) {
                break;
            }
        }

        // Pick & apply moves; record training samples.
        for (i, &gi) in active_games.iter().enumerate() {
            let dist = searches[gi].get_pruned_visit_distribution();

            // Build dense policy target over the full POLICY_SIZE space.
            let mut policy_vec = vec![0.0f32; POLICY_SIZE];
            for (mv, prob) in &dist {
                use core_game::game::PolicyIndex;
                match crate::move_encoding::encode_move(mv) {
                    PolicyIndex::Single(idx) => policy_vec[idx] = *prob,
                    PolicyIndex::Sum(a, b) => {
                        // Split the visit probability equally between the two factor cells
                        // so the network learns both `from` and `to` channels.
                        let half = *prob * 0.5;
                        policy_vec[a] += half;
                        policy_vec[b] += half;
                    }
                    PolicyIndex::DotProduct { .. } => {} // unused for yinsh
                }
            }

            // Snapshot the position BEFORE playing the chosen move.
            let mut board_snap = vec![0f32; BOARD_FLAT];
            let mut reserve_snap = vec![0f32; RESERVE_SIZE];
            boards[gi].encode_board(&mut board_snap, &mut reserve_snap);
            let phase = phase_code(boards[gi].phase);
            let player = boards[gi].next_player();
            histories[gi].push(TurnRecord {
                board: board_snap,
                reserve: reserve_snap,
                policy: policy_vec,
                player,
                value_only: !is_full[i],
                phase,
            });

            // Sample the move (tempered for first `temp_threshold` moves).
            let mv = if dist.is_empty() {
                YinshBoard::pass_move()
            } else if move_counts[gi] < temp_threshold && temperature > 0.01 {
                let weights: Vec<f32> = dist.iter().map(|(_, p)| p.powf(1.0 / temperature)).collect();
                let wi = WeightedIndex::new(&weights).map_err(|e| e.to_string())?;
                dist[wi.sample(&mut rng)].0
            } else {
                dist.iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
                    .0
            };

            boards[gi].play_move(&mv).map_err(|e| e.to_string())?;
            move_counts[gi] += 1;
            result.total_moves += 1;

            if boards[gi].is_game_over() || move_counts[gi] >= max_moves {
                active[gi] = false;
                finished_count += 1;
                let len = move_counts[gi];
                result.game_lengths.push(len);
                match boards[gi].outcome() {
                    Outcome::WonBy(winner) => {
                        if winner == Player::Player1 {
                            result.wins_p1 += 1;
                        } else {
                            result.wins_p2 += 1;
                        }
                        result.decisive_lengths.push(len);
                        if result.sample_summaries.len() < 3 {
                            let label = format!(
                                "{} wins {}-{} ({} moves)",
                                if winner == Player::Player1 { "white" } else { "black" },
                                boards[gi].white_score,
                                boards[gi].black_score,
                                len,
                            );
                            result.sample_summaries.push(label);
                        }
                    }
                    _ => {
                        result.draws += 1;
                    }
                }
            }
        }

        if let Some(pfn) = &progress_fn {
            let active_count = active.iter().filter(|&&a| a).count() as u32;
            pfn(finished_count, num_games as u32, active_count, result.total_moves);
        }
    }

    // Flatten histories into the result, attaching final value targets.
    let total_samples: usize = histories.iter().map(|h| h.len()).sum();
    result.board_data.reserve(total_samples * BOARD_FLAT);
    result.reserve_data.reserve(total_samples * RESERVE_SIZE);
    result.policy_data.reserve(total_samples * POLICY_SIZE);
    result.value_targets.reserve(total_samples);
    result.value_only_flags.reserve(total_samples);
    result.phase_flags.reserve(total_samples);

    for (gi, history) in histories.iter().enumerate() {
        let outcome = boards[gi].outcome();
        for record in history {
            result.board_data.extend_from_slice(&record.board);
            result.reserve_data.extend_from_slice(&record.reserve);
            result.policy_data.extend_from_slice(&record.policy);
            let value = match outcome {
                Outcome::WonBy(winner) => {
                    if winner == record.player { 1.0 } else { -1.0 }
                }
                _ => 0.0,
            };
            result.value_targets.push(value);
            result.value_only_flags.push(record.value_only);
            result.phase_flags.push(record.phase);
        }
    }
    result.num_samples = total_samples;

    Ok(result)
}
