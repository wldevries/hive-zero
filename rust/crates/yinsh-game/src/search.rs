//! Self-play, battle, and best-move orchestration for YINSH.
//!
//! All tree search uses `core_game::mcts::MctsSearch<YinshBoard>` directly —
//! no game-specific MCTS code. Yinsh's policy is a single flat 7-channel tensor
//! (847 entries) consumed via the `PolicyIndex::Sum` mechanism that
//! `core_game::mcts` already supports.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use rand::RngExt;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;

use core_game::game::{Game, NNGame, Outcome, Player};
use core_game::mcts::arena::NodeId;
use core_game::mcts::search::{
    CpuctStrategy, ForcedExploration, MctsSearch, RootNoise, SearchParams,
};
use core_game::selfplay_config::{MctsConfig, OpeningRandomConfig, PlayoutCapConfig};

use crate::board::{Phase, YinshBoard, YinshMove};
use crate::board_encoding::{NUM_CHANNELS, RESERVE_SIZE};
use crate::hex::GRID_SIZE;
use crate::move_encoding::POLICY_SIZE;

const BOARD_FLAT: usize = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;

/// Eval callback: `(boards_flat[N*BOARD_FLAT], reserves_flat[N*RESERVE_SIZE], n) ->
/// (policy[N*POLICY_SIZE], values[N], draws[N])`.
/// `values[i]` = W−L (zero-sum); `draws[i]` = D probability (symmetric).
pub type EvalFn = Box<
    dyn Fn(&[f32], &[f32], usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> + Send + Sync,
>;

/// Progress callback: `(finished, total, active, total_moves)`.
pub type ProgressFn = Box<dyn Fn(u32, u32, u32, u32) + Send + Sync>;

/// Opaque ticket returned by `BatchEvaluator::submit` and consumed by
/// `collect`. The implementation chooses what it means: the synchronous
/// evaluator uses it as an index into a result stash; the pipelined ORT
/// evaluator maps it to a oneshot receiver from the worker thread.
pub type RequestId = u64;

/// Two-phase batch inference interface. `submit` posts a request and returns
/// immediately; `collect` blocks until that request's response is ready.
///
/// The split lets `play_selfplay_core` overlap leaf selection for batch N+1
/// with the GPU forward for batch N when paired with a worker-thread
/// implementation. The default synchronous wrapper (used for the Python
/// callback path, where the GIL precludes overlap) does the work inside
/// `submit` and trivially returns it in `collect`.
///
/// Boards and reserves are taken by value so a pipelined implementation can
/// ship the owned buffers across a channel to a worker thread without an
/// additional memcpy. The synchronous wrapper just borrows from the Vec
/// before dropping it.
pub trait BatchEvaluator {
    fn submit(
        &mut self,
        boards: Vec<f32>,
        reserves: Vec<f32>,
        n: usize,
    ) -> Result<RequestId, String>;

    fn collect(
        &mut self,
        id: RequestId,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String>;
}

/// Default `BatchEvaluator` that runs the closure synchronously inside
/// `submit` and stashes the result for `collect`. Behaviour is identical to
/// calling `eval_fn` inline — no pipelining benefit, but lets the inner loop
/// speak a single API regardless of backend.
pub struct SyncEvaluator {
    eval_fn: EvalFn,
    next_id: RequestId,
    pending: HashMap<RequestId, Result<(Vec<f32>, Vec<f32>, Vec<f32>), String>>,
}

impl SyncEvaluator {
    pub fn new(eval_fn: EvalFn) -> Self {
        Self {
            eval_fn,
            next_id: 0,
            pending: HashMap::new(),
        }
    }
}

impl BatchEvaluator for SyncEvaluator {
    fn submit(
        &mut self,
        boards: Vec<f32>,
        reserves: Vec<f32>,
        n: usize,
    ) -> Result<RequestId, String> {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let result = (self.eval_fn)(&boards, &reserves, n);
        self.pending.insert(id, result);
        Ok(id)
    }

    fn collect(
        &mut self,
        id: RequestId,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        self.pending
            .remove(&id)
            .ok_or_else(|| format!("SyncEvaluator: unknown request id {id}"))?
    }
}

/// Phase of a turn — useful as a categorical training-data flag.
#[inline]
fn phase_code(p: Phase) -> u8 {
    match p {
        Phase::Setup => 0,
        Phase::Normal => 1,
        Phase::ClaimRow => 2,
    }
}

#[inline]
fn make_search(
    simulations: usize,
    c_puct: f32,
    draw_contempt: f32,
    forced_playouts: bool,
) -> MctsSearch<YinshBoard> {
    let mut s = MctsSearch::<YinshBoard>::new(simulations + 64);
    let forced = if forced_playouts {
        ForcedExploration::Soft { selection_k: 0.5, pruning_k: 2.0 }
    } else {
        ForcedExploration::None
    };
    s.params = SearchParams::new(
        CpuctStrategy::Constant { c_puct },
        forced,
        RootNoise::None,
    );
    s.params.draw_contempt = draw_contempt;
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
        let (policy_flat, values, draws) = eval_fn(
            &flat_boards[..nl * BOARD_FLAT],
            &flat_reserves[..nl * RESERVE_SIZE],
            nl,
        )?;
        let policies: Vec<Vec<f32>> = (0..nl)
            .map(|i| policy_flat[i * POLICY_SIZE..(i + 1) * POLICY_SIZE].to_vec())
            .collect();
        search.expand_and_backprop(&policies, &values, &draws);
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

    let mut search = make_search(simulations, c_puct, 0.0, false);

    // Initial root eval.
    let mut root_board = vec![0f32; BOARD_FLAT];
    let mut root_reserve = vec![0f32; RESERVE_SIZE];
    board.encode_board(&mut root_board, &mut root_reserve);
    let (root_policy, _, _) = eval_fn(&root_board, &root_reserve, 1)?;
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

            let mut search = make_search(simulations, c_puct, 0.0, false);

            let mut root_board = vec![0f32; BOARD_FLAT];
            let mut root_reserve = vec![0f32; RESERVE_SIZE];
            boards[gi].encode_board(&mut root_board, &mut root_reserve);
            let (root_policy, _, _) = eval_ref(&root_board, &root_reserve, 1)?;
            search.init(&boards[gi], &root_policy);

            run_simulations_single(&mut search, simulations, play_batch_size.max(1), eval_ref)?;

            let mv = search.best_move().unwrap_or_else(YinshBoard::pass_move);
            boards[gi]
                .play_move(&mv)
                .map_err(|e| format!("battle game {} illegal move: {}", gi, e))?;
            move_counts[gi] += 1;
            total_moves += 1;

            if boards[gi].is_game_over() {
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

/// Battle the NN model (MCTS) against the heuristic alpha-beta bot, in
/// parallel across `num_games` games. Each ply, active games are partitioned
/// by whose turn it is:
///
/// - **Model-to-move** games batch their MCTS leaf evaluations through a
///   single `eval_fn` call (leaves across games are concatenated, and
///   `play_batch_size` controls how many leaf-collection rounds run before
///   each NN call).
/// - **Bot-to-move** games are resolved synchronously via
///   `alphabeta::alphabeta_best_move`, which does not need the NN.
///
/// Half the games have the model as P1 (white), the other half as P2 (black),
/// so `wins_model1` counts NN wins regardless of color. The model's MCTS tree
/// is rerooted after every move (model or bot) so the next model-turn round
/// can warm-start.
pub fn play_battle_vs_bot_core(
    num_games: usize,
    simulations: usize,
    c_puct: f32,
    play_batch_size: usize,
    bot_depth: u32,
    eval_fn: EvalFn,
    progress_fn: Option<ProgressFn>,
) -> Result<BattleResult, String> {
    let half = num_games / 2;
    let model_is_p1 = |gi: usize| gi < half;

    let mut boards: Vec<YinshBoard> = (0..num_games).map(|_| YinshBoard::new()).collect();
    let mut searches: Vec<MctsSearch<YinshBoard>> =
        (0..num_games).map(|_| make_search(simulations, c_puct, 0.0, false)).collect();
    let mut active = vec![true; num_games];
    let mut move_counts = vec![0u32; num_games];
    // Search trees are warm only after a successful reroot. Bot moves also
    // reroot, so consecutive bot turns keep the tree valid for the next model
    // turn.
    let mut search_warm = vec![false; num_games];

    let mut result = BattleResult::default();
    let mut total_moves: u32 = 0;
    let mut finished: u32 = 0;

    while active.iter().any(|&a| a) {
        // Partition active games by who's to move this ply.
        let mut mcts_games: Vec<usize> = Vec::new();
        let mut bot_games: Vec<usize> = Vec::new();
        for gi in 0..num_games {
            if !active[gi] {
                continue;
            }
            let model_to_move =
                (boards[gi].next_player() == Player::Player1) == model_is_p1(gi);
            if model_to_move {
                mcts_games.push(gi);
            } else {
                bot_games.push(gi);
            }
        }

        // Bot moves first — synchronous and cheap. Result list is built up
        // before any board mutation so the partition can't shift mid-ply.
        let mut chosen_moves: Vec<(usize, YinshMove)> = Vec::with_capacity(num_games);
        for &gi in &bot_games {
            let mv = crate::alphabeta::alphabeta_best_move(&boards[gi], bot_depth);
            chosen_moves.push((gi, mv));
        }

        // Model moves: batched MCTS across mcts_games (single eval_fn).
        if !mcts_games.is_empty() {
            let n = mcts_games.len();

            // Cold init for any game whose tree isn't warm.
            let cold: Vec<usize> =
                (0..n).filter(|&i| !search_warm[mcts_games[i]]).collect();
            if !cold.is_empty() {
                let nc = cold.len();
                let mut flat_boards = vec![0f32; nc * BOARD_FLAT];
                let mut flat_reserves = vec![0f32; nc * RESERVE_SIZE];
                for (k, &ci) in cold.iter().enumerate() {
                    let gi = mcts_games[ci];
                    boards[gi].encode_board(
                        &mut flat_boards[k * BOARD_FLAT..(k + 1) * BOARD_FLAT],
                        &mut flat_reserves[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE],
                    );
                }
                let (init_policy, _, _) = eval_fn(&flat_boards, &flat_reserves, nc)?;
                for (k, &ci) in cold.iter().enumerate() {
                    let gi = mcts_games[ci];
                    searches[gi].init(
                        &boards[gi],
                        &init_policy[k * POLICY_SIZE..(k + 1) * POLICY_SIZE],
                    );
                }
            }

            // Simulation rounds: select leaves across games, batch-eval, expand.
            let mut game_sims = vec![0usize; n];
            loop {
                let mut leaf_ids: Vec<NodeId> = Vec::new();
                let mut leaf_game_idx: Vec<usize> = Vec::new();
                for _round in 0..play_batch_size.max(1) {
                    let mut any = false;
                    for (i, _) in mcts_games.iter().enumerate() {
                        if game_sims[i] >= simulations {
                            continue;
                        }
                        let gi = mcts_games[i];
                        let leaves = searches[gi].select_leaves(1);
                        let count = leaves.len();
                        if count > 0 {
                            any = true;
                        }
                        for leaf in leaves {
                            leaf_ids.push(leaf);
                            leaf_game_idx.push(i);
                        }
                        game_sims[i] += count;
                    }
                    if !any {
                        break;
                    }
                }
                if leaf_ids.is_empty() {
                    break;
                }

                let nl = leaf_ids.len();
                let mut leaf_boards_flat = vec![0f32; nl * BOARD_FLAT];
                let mut leaf_reserves_flat = vec![0f32; nl * RESERVE_SIZE];
                for (k, (&leaf, &i)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
                    let gi = mcts_games[i];
                    let (b, r) = searches[gi].encode_leaf(leaf);
                    leaf_boards_flat[k * BOARD_FLAT..(k + 1) * BOARD_FLAT]
                        .copy_from_slice(&b);
                    leaf_reserves_flat[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE]
                        .copy_from_slice(&r);
                }

                let (leaf_policy, leaf_values, leaf_draws) =
                    eval_fn(&leaf_boards_flat, &leaf_reserves_flat, nl)?;

                let mut per_game_policies: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n];
                let mut per_game_values: Vec<Vec<f32>> = (0..n).map(|_| Vec::new()).collect();
                let mut per_game_draws: Vec<Vec<f32>> = (0..n).map(|_| Vec::new()).collect();
                for (k, &i) in leaf_game_idx.iter().enumerate() {
                    per_game_policies[i].push(
                        leaf_policy[k * POLICY_SIZE..(k + 1) * POLICY_SIZE].to_vec(),
                    );
                    per_game_values[i].push(leaf_values[k]);
                    per_game_draws[i].push(leaf_draws[k]);
                }
                for (i, _) in mcts_games.iter().enumerate() {
                    if per_game_policies[i].is_empty() {
                        continue;
                    }
                    let gi = mcts_games[i];
                    searches[gi].expand_and_backprop(
                        &per_game_policies[i],
                        &per_game_values[i],
                        &per_game_draws[i],
                    );
                }
                if game_sims.iter().all(|&s| s >= simulations) {
                    break;
                }
            }

            // Pick best move per game by visit count.
            for &gi in &mcts_games {
                let mv = searches[gi].best_move().unwrap_or_else(YinshBoard::pass_move);
                chosen_moves.push((gi, mv));
            }
        }

        // Apply all moves and update game state.
        for (gi, mv) in chosen_moves {
            boards[gi]
                .play_move(&mv)
                .map_err(|e| format!("battle game {} illegal move: {}", gi, e))?;
            move_counts[gi] += 1;
            total_moves += 1;

            if boards[gi].is_game_over() {
                active[gi] = false;
                finished += 1;
                result.game_lengths.push(move_counts[gi]);
                search_warm[gi] = false;
                match boards[gi].outcome() {
                    Outcome::WonBy(winner) => {
                        let m1_won = model_is_p1(gi) == (winner == Player::Player1);
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
            } else {
                // Reroot the model's tree to the move that was actually played
                // so the next model-turn round can warm-start.
                search_warm[gi] = searches[gi].reroot(mv);
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

fn mean_std(sum: f64, sum_sq: f64, count: u64) -> (f32, f32) {
    if count == 0 { return (0.0, 0.0); }
    let n = count as f64;
    let mean = sum / n;
    let var = (sum_sq / n) - mean * mean;
    (mean as f32, var.max(0.0).sqrt() as f32)
}

/// Pure-Rust self-play result. All training samples are concatenated flat.
#[derive(Clone, Debug, Default)]
pub struct SelfPlayResult {
    pub board_data: Vec<f32>,
    pub reserve_data: Vec<f32>,
    pub policy_data: Vec<f32>,
    pub value_targets: Vec<f32>,
    /// Per-sample MCTS root value (W−L scalar, no contempt) captured at the
    /// turn the position was played. Used as the q-target source for value
    /// target mixing: `target = (1-λ)·z + λ·q`.
    pub root_q_targets: Vec<f32>,
    pub value_only_flags: Vec<bool>,
    pub phase_flags: Vec<u8>,
    pub num_samples: usize,

    pub wins_p1: u32,
    pub wins_p2: u32,
    /// Real draws — game ended naturally with equal scores (marker exhaustion at 0-0,
    /// 1-1, etc.). Distinguished from `timeouts`.
    pub draws: u32,
    /// Games aborted because they hit the move cap (`max_moves`). These are NOT
    /// real Yinsh draws and should be treated separately when tuning.
    pub timeouts: u32,
    pub total_moves: u32,
    pub game_lengths: Vec<u32>,
    pub decisive_lengths: Vec<u32>,

    pub full_search_turns: u32,
    pub total_turns: u32,

    /// Up to 2 (label, board_string) pairs from decisive games for display.
    pub sample_board_data: Vec<(String, String)>,

    pub top1_visit_fraction_mean: f32,
    pub top1_visit_fraction_std: f32,
    pub search_depth_mean: f32,
    pub search_depth_std: f32,
    pub valid_moves_mean: f32,
    pub valid_moves_std: f32,

    /// Per-phase wall-clock breakdown of the self-play inner loop. Optional
    /// to consume — see `SelfPlayTiming` for the measured phases.
    pub timing: SelfPlayTiming,
}

/// Wall-clock accumulators for one self-play session, broken out by phase.
/// `select`/`encode`/`eval`/`expand` are measured around the four phases of
/// the synchronous inner sim loop on the main thread. `eval_input/run/extract`
/// are a finer subdivision of `eval` populated only by the ORT engine path —
/// they're set externally after self-play completes (see
/// `yinsh_python::play_games`) and stay zero for the Python-callback path.
#[derive(Clone, Debug, Default)]
pub struct SelfPlayTiming {
    pub select: Duration,
    pub encode: Duration,
    pub eval: Duration,
    pub expand: Duration,

    /// Tensor construction + boards/reserves clone-into-Tensor.
    pub eval_input: Duration,
    /// ort::Session::run (H2D + GPU compute + D2H).
    pub eval_run: Duration,
    /// try_extract_tensor + output policy.to_vec.
    pub eval_extract: Duration,
}

/// One training record per turn, accumulated per game.
struct TurnRecord {
    board: Vec<f32>,
    reserve: Vec<f32>,
    policy: Vec<f32>,
    player: Player,
    value_only: bool,
    phase: u8,
    /// MCTS root value (W−L scalar, no contempt) at this turn, in `player`'s
    /// perspective. Used as the q-target for training-time target mixing.
    root_q: f32,
}

// ---------------------------------------------------------------------------
// Self-play inner-loop helpers
// ---------------------------------------------------------------------------

/// Identifies which active game each leaf in a batch belongs to. Needed by
/// `expand_from_resp` to regroup the flat policy/values/draws back per-game
/// in stash order.
struct BatchMeta {
    leaf_game_idx: Vec<usize>,
}

/// One inference batch's worth of data: leaf metadata plus flat tensors
/// ready for the eval callback.
struct PreparedBatch {
    meta: BatchMeta,
    boards: Vec<f32>,
    reserves: Vec<f32>,
    nl: usize,
}

/// Select up to `play_batch_size` leaves per still-running active game and
/// encode them into flat tensors. Mirrors what the synchronous loop used to do
/// inline. Updates `game_sims[i]` in place. Returns `None` if no leaves were
/// produced (every game has hit its sim cap or all trees are exhausted).
fn prepare_batch(
    searches: &mut [MctsSearch<YinshBoard>],
    active_games: &[usize],
    play_batch_size: usize,
    sim_caps: &[usize],
    game_sims: &mut [usize],
    t_select: &mut Duration,
    t_encode: &mut Duration,
) -> Option<PreparedBatch> {
    let t0 = Instant::now();
    let mut leaf_ids: Vec<NodeId> = Vec::new();
    let mut leaf_game_idx: Vec<usize> = Vec::new();
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
    *t_select += t0.elapsed();

    if leaf_ids.is_empty() {
        return None;
    }

    let t1 = Instant::now();
    let nl = leaf_ids.len();
    let mut leaf_boards = vec![0f32; nl * BOARD_FLAT];
    let mut leaf_reserves = vec![0f32; nl * RESERVE_SIZE];
    for (k, (&leaf, &i)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
        let gi = active_games[i];
        let (b, r) = searches[gi].encode_leaf(leaf);
        leaf_boards[k * BOARD_FLAT..(k + 1) * BOARD_FLAT].copy_from_slice(&b);
        leaf_reserves[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE].copy_from_slice(&r);
    }
    *t_encode += t1.elapsed();

    let _ = leaf_ids;
    Some(PreparedBatch {
        meta: BatchMeta { leaf_game_idx },
        boards: leaf_boards,
        reserves: leaf_reserves,
        nl,
    })
}

/// Regroup an inference batch's results back per-game and call
/// `expand_and_backprop_with_stash` on each game's portion. Operates on
/// caller-owned stashes (previously taken from each search via `take_stash`)
/// so two batches can be in flight simultaneously on the pipelined path —
/// the next `prepare_batch` is free to refill each search's internal stash
/// without clobbering this batch's leaves.
fn expand_from_resp_with_stashes(
    searches: &mut [MctsSearch<YinshBoard>],
    active_games: &[usize],
    stashes: Vec<Vec<(NodeId, YinshBoard)>>,
    meta: &BatchMeta,
    policy: &[f32],
    values: &[f32],
    draws: &[f32],
) {
    let n = stashes.len();
    let mut per_game_policies: Vec<Vec<Vec<f32>>> = (0..n).map(|_| Vec::new()).collect();
    let mut per_game_values: Vec<Vec<f32>> = (0..n).map(|_| Vec::new()).collect();
    let mut per_game_draws: Vec<Vec<f32>> = (0..n).map(|_| Vec::new()).collect();
    for (k, &i) in meta.leaf_game_idx.iter().enumerate() {
        per_game_policies[i].push(
            policy[k * POLICY_SIZE..(k + 1) * POLICY_SIZE].to_vec(),
        );
        per_game_values[i].push(values[k]);
        per_game_draws[i].push(draws[k]);
    }
    for (i, stash) in stashes.into_iter().enumerate() {
        if stash.is_empty() {
            continue;
        }
        let gi = active_games[i];
        searches[gi].expand_and_backprop_with_stash(
            stash,
            &per_game_policies[i],
            &per_game_values[i],
            &per_game_draws[i],
        );
    }
}

/// Run `num_games` of self-play in parallel. Each game uses its own
/// `MctsSearch<YinshBoard>`; leaves are batched across games via
/// `play_batch_size` for better GPU utilization. Returns a flat `SelfPlayResult`
/// suitable for direct ingestion into the Python replay buffer.
pub fn play_selfplay_core(
    num_games: usize,
    mcts: MctsConfig,
    playout_cap: PlayoutCapConfig,
    opening: OpeningRandomConfig,
    evaluator: &mut dyn BatchEvaluator,
    progress_fn: Option<ProgressFn>,
    opening_sequences: Vec<Vec<String>>,
) -> Result<SelfPlayResult, String> {
    let use_playout_cap = playout_cap.enabled();

    let mut boards: Vec<YinshBoard> = (0..num_games).map(|_| YinshBoard::new()).collect();
    let mut searches: Vec<MctsSearch<YinshBoard>> = (0..num_games)
        .map(|_| make_search(mcts.simulations, mcts.c_puct, mcts.draw_contempt, mcts.forced_playouts))
        .collect();

    // Per-game asymmetric contempt assignment. When enabled each game randomly
    // picks one side as the contempt side and bakes that into its SearchParams,
    // so the MCTS UCB only applies `draw_contempt` at nodes where the contempt
    // side chose the move. Yinsh's `searches[gi].params` is not re-cloned during
    // the run, so setting it once here is sufficient.
    if mcts.asymmetric_contempt {
        let mut crng = rand::rng();
        for s in searches.iter_mut() {
            s.params.contempt_side = Some(if crng.random::<bool>() {
                Player::Player1
            } else {
                Player::Player2
            });
        }
    }
    let mut active = vec![true; num_games];
    let mut move_counts = vec![0u32; num_games];
    let mut histories: Vec<Vec<TurnRecord>> = (0..num_games).map(|_| Vec::new()).collect();
    // True once a game's tree has been rerooted and its root is already expanded with
    // priors from the previous search; such games skip the root NN eval + init().
    let mut search_warm = vec![false; num_games];

    let mut result = SelfPlayResult::default();
    let mut finished_count: u32 = 0;
    let mut rng = rand::rng();

    // Per-game random opening move counts: each game plays this many random moves
    // (sampled from valid_moves) before MCTS takes over. Diversifies openings.
    // Games that have a non-empty `opening_sequences[gi]` use the book branch
    // instead — random opening moves only apply to non-book games.
    let game_random_opening_moves: Vec<u32> = (0..num_games)
        .map(|_| {
            if opening.max > opening.min {
                rng.random_range(opening.min..=opening.max)
            } else {
                opening.min
            }
        })
        .collect();

    // Tracks games whose book replay aborted (illegal/unparseable move). Such
    // games skip remaining book moves and fall straight through to MCTS.
    let mut opening_done = vec![false; num_games];

    // Cached per-game flag: true iff `opening_sequences[gi]` is non-empty.
    // Used to decide whether the game's opening window is book- or random-driven.
    let has_book: Vec<bool> = (0..num_games)
        .map(|gi| opening_sequences.get(gi).is_some_and(|s| !s.is_empty()))
        .collect();

    let mut session_top1_sum = 0f64;
    let mut session_top1_sum_sq = 0f64;
    let mut session_top1_count = 0u64;
    let mut session_depth_sum = 0f64;
    let mut session_depth_sum_sq = 0f64;
    let mut session_depth_count = 0u64;
    let mut session_moves_sum = 0f64;
    let mut session_moves_sum_sq = 0f64;
    let mut session_moves_count = 0u64;

    // Per-phase wall-clock accumulators. See `SelfPlayTiming` for the
    // semantics of each phase.
    let mut t_select = Duration::ZERO;
    let mut t_encode = Duration::ZERO;
    let mut t_eval = Duration::ZERO;
    let mut t_expand = Duration::ZERO;

    while active.iter().any(|&a| a) {
        // Phase 1: play opening moves (book if assigned, else random) for any
        // active games still in their opening window. These moves do NOT produce
        // training samples and are not counted in `total_turns` or per-iteration
        // MCTS stats.
        for gi in 0..num_games {
            if !active[gi] {
                continue;
            }

            // Resolve which opening branch this game is on for this move.
            let in_book = has_book[gi]
                && !opening_done[gi]
                && (move_counts[gi] as usize) < opening_sequences[gi].len();
            let in_random = !has_book[gi] && move_counts[gi] < game_random_opening_moves[gi];
            if !in_book && !in_random {
                continue;
            }

            let mv = if in_book {
                let move_str = &opening_sequences[gi][move_counts[gi] as usize];
                match crate::notation::str_to_move(move_str) {
                    Ok(parsed) => {
                        let valid = boards[gi].valid_moves();
                        if valid.iter().any(|v| *v == parsed) {
                            parsed
                        } else {
                            // Move not legal at this position — abort book
                            // replay and let MCTS take over from here.
                            opening_done[gi] = true;
                            continue;
                        }
                    }
                    Err(_) => {
                        opening_done[gi] = true;
                        continue;
                    }
                }
            } else {
                let valid = boards[gi].valid_moves();
                if valid.is_empty() {
                    YinshBoard::pass_move()
                } else {
                    let idx = rng.random_range(0..valid.len());
                    valid[idx]
                }
            };

            boards[gi].play_move(&mv).map_err(|e| e.to_string())?;
            move_counts[gi] += 1;
            result.total_moves += 1;
            if boards[gi].is_game_over() {
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
                    }
                    Outcome::Draw => result.draws += 1,
                    Outcome::Ongoing => {}
                }
            }
        }

        // Phase 2: MCTS for games that have completed their opening window.
        // A book game is past its window once the sequence is exhausted (or
        // aborted via `opening_done`); a random-opening game is past its window
        // once `move_counts` reaches its per-game random count.
        let active_games: Vec<usize> = (0..num_games)
            .filter(|&gi| {
                if !active[gi] {
                    return false;
                }
                if has_book[gi] {
                    opening_done[gi]
                        || (move_counts[gi] as usize) >= opening_sequences[gi].len()
                } else {
                    move_counts[gi] >= game_random_opening_moves[gi]
                }
            })
            .collect();
        if active_games.is_empty() {
            if !active.iter().any(|&a| a) {
                break;
            }
            if let Some(pfn) = &progress_fn {
                let active_count = active.iter().filter(|&&a| a).count() as u32;
                pfn(finished_count, num_games as u32, active_count, result.total_moves);
            }
            continue;
        }
        let n = active_games.len();
        result.total_turns += n as u32;

        // Decide full vs fast search per active game.
        let is_full: Vec<bool> = if use_playout_cap {
            (0..n).map(|_| rng.random::<f32>() < playout_cap.p).collect()
        } else {
            vec![true; n]
        };
        let sim_caps: Vec<usize> = is_full
            .iter()
            .map(|&f| if f { mcts.simulations } else { playout_cap.fast_cap })
            .collect();
        result.full_search_turns += is_full.iter().filter(|&&f| f).count() as u32;

        // Only cold games need a root NN eval + init(). Warm games already have
        // an expanded root with priors from the previous ply's reroot().
        let cold: Vec<usize> = (0..n)
            .filter(|&i| !search_warm[active_games[i]])
            .collect();

        if !cold.is_empty() {
            let nc = cold.len();
            let mut cold_boards = vec![0f32; nc * BOARD_FLAT];
            let mut cold_reserves = vec![0f32; nc * RESERVE_SIZE];
            for (k, &i) in cold.iter().enumerate() {
                let gi = active_games[i];
                boards[gi].encode_board(
                    &mut cold_boards[k * BOARD_FLAT..(k + 1) * BOARD_FLAT],
                    &mut cold_reserves[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE],
                );
            }
            let init_id = evaluator.submit(cold_boards, cold_reserves, nc)?;
            let (init_policy, _, _) = evaluator.collect(init_id)?;
            for (k, &i) in cold.iter().enumerate() {
                let gi = active_games[i];
                let policy_slice = &init_policy[k * POLICY_SIZE..(k + 1) * POLICY_SIZE];
                searches[gi].init(&boards[gi], policy_slice);
            }
        }

        // Apply fresh Dirichlet noise to every full-search root (warm and cold).
        for (i, &gi) in active_games.iter().enumerate() {
            if is_full[i] {
                searches[gi].apply_root_dirichlet(mcts.dir_alpha, mcts.dir_epsilon);
            }
        }

        // Disable forced playouts on fast turns: their policy target is
        // discarded (value-only training), and the tied-visit pruning quirk
        // can mis-select the played move from a low-N distribution.
        if mcts.forced_playouts {
            for (i, &gi) in active_games.iter().enumerate() {
                searches[gi].params.forced_exploration = if is_full[i] {
                    ForcedExploration::Soft { selection_k: 0.5, pruning_k: 2.0 }
                } else {
                    ForcedExploration::None
                };
            }
        }

        // Cross-game batched simulations, pipelined: at any time at most one
        // batch is in flight on the evaluator. Each iteration builds + submits
        // batch N+1 and then waits on + expands batch N, so the GPU forward
        // for the previous batch overlaps with the main thread's leaf
        // selection for the next one. For the synchronous `SyncEvaluator`
        // this loop is equivalent to the old "build → submit → collect →
        // expand" pattern (no overlap), since `submit` does the work inline.
        let mut game_sims = vec![0usize; n];
        let mut pending: Option<(RequestId, Vec<Vec<(NodeId, YinshBoard)>>, BatchMeta)> = None;
        loop {
            // Phase A: prepare next batch's leaves (select + encode). Returns
            // None when every game has hit its sim cap or trees are exhausted.
            let next_batch = prepare_batch(
                &mut searches,
                &active_games,
                mcts.play_batch_size,
                &sim_caps,
                &mut game_sims,
                &mut t_select,
                &mut t_encode,
            );

            // Phase B: snapshot per-game stashes BEFORE the next iteration's
            // prepare_batch can re-fill them. Without this take_stash, the
            // next select_leaves would append onto the same Vec and clobber
            // this batch's leaves when expand_and_backprop later consumes
            // the merged stash via mem::take.
            let new_stashes: Vec<Vec<(NodeId, YinshBoard)>> = if next_batch.is_some() {
                active_games
                    .iter()
                    .map(|&gi| searches[gi].take_stash())
                    .collect()
            } else {
                Vec::new()
            };

            // Phase C: submit new batch. On the pipelined path this returns
            // immediately and the GPU starts batch N+1; on the sync path it
            // runs the eval inline.
            let new_pending = if let Some(batch) = next_batch {
                let t = Instant::now();
                let id = evaluator.submit(batch.boards, batch.reserves, batch.nl)?;
                t_eval += t.elapsed();
                Some((id, new_stashes, batch.meta))
            } else {
                None
            };

            // Phase D: collect + expand the PREVIOUS batch (if any). The
            // pipelined evaluator blocks here on the worker's response —
            // but the new batch is already in flight from phase C, so the
            // GPU stays busy across this boundary.
            if let Some((prev_id, prev_stashes, prev_meta)) = pending.take() {
                let t = Instant::now();
                let (policy, values, draws) = evaluator.collect(prev_id)?;
                t_eval += t.elapsed();
                let te = Instant::now();
                expand_from_resp_with_stashes(
                    &mut searches,
                    &active_games,
                    prev_stashes,
                    &prev_meta,
                    &policy,
                    &values,
                    &draws,
                );
                t_expand += te.elapsed();
            }

            pending = new_pending;

            // Termination: nothing more to submit and nothing in flight.
            if pending.is_none() {
                break;
            }
        }

        // Collect per-turn MCTS stats for full-search turns.
        for (i, &gi) in active_games.iter().enumerate() {
            let (ds, dss, dc) = searches[gi].take_depth_stats();
            if is_full[i] {
                let top1 = searches[gi].root_top1_visit_fraction() as f64;
                session_top1_sum += top1;
                session_top1_sum_sq += top1 * top1;
                session_top1_count += 1;
                session_depth_sum += ds;
                session_depth_sum_sq += dss;
                session_depth_count += dc;
                let moves = searches[gi].root_child_count() as f64;
                session_moves_sum += moves;
                session_moves_sum_sq += moves * moves;
                session_moves_count += 1;
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
            // Capture the search-improved root value (W−L, no contempt) for
            // training-time q-target mixing. Reads from the just-completed
            // search; must happen before any reroot below.
            let root_q = searches[gi].root_value_raw();
            histories[gi].push(TurnRecord {
                board: board_snap,
                reserve: reserve_snap,
                policy: policy_vec,
                player,
                value_only: !is_full[i],
                phase,
                root_q,
            });

            // Sample the move (tempered for first `temp_threshold` moves).
            let mv = if dist.is_empty() {
                YinshBoard::pass_move()
            } else if move_counts[gi] < mcts.temp_threshold && mcts.temperature > 0.01 {
                let weights: Vec<f32> = dist.iter().map(|(_, p)| p.powf(1.0 / mcts.temperature)).collect();
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

            if boards[gi].is_game_over() {
                active[gi] = false;
                finished_count += 1;
                search_warm[gi] = false;
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
                        if result.sample_board_data.len() < 2 {
                            let label = format!(
                                "{} wins {}-{} ({} moves)",
                                if winner == Player::Player1 { "white" } else { "black" },
                                boards[gi].white_score,
                                boards[gi].black_score,
                                len,
                            );
                            result.sample_board_data.push((label, format!("{}", boards[gi])));
                        }
                    }
                    Outcome::Draw => result.draws += 1,
                    Outcome::Ongoing => {} // cannot happen: game is naturally bounded
                }
            } else {
                // Reroot to preserve the subtree for the chosen move.
                // Falls back to a cold init next ply if the move wasn't expanded.
                search_warm[gi] = searches[gi].reroot(mv);
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
    result.root_q_targets.reserve(total_samples);
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
            result.root_q_targets.push(record.root_q);
            result.value_only_flags.push(record.value_only);
            result.phase_flags.push(record.phase);
        }
    }
    result.num_samples = total_samples;

    (result.top1_visit_fraction_mean, result.top1_visit_fraction_std) =
        mean_std(session_top1_sum, session_top1_sum_sq, session_top1_count);
    (result.search_depth_mean, result.search_depth_std) =
        mean_std(session_depth_sum, session_depth_sum_sq, session_depth_count);
    (result.valid_moves_mean, result.valid_moves_std) =
        mean_std(session_moves_sum, session_moves_sum_sq, session_moves_count);

    result.timing.select = t_select;
    result.timing.encode = t_encode;
    result.timing.eval = t_eval;
    result.timing.expand = t_expand;
    // `eval_input/run/extract` are populated by the caller (see
    // `yinsh_python::play_games`) once the ORT engine is drained.

    Ok(result)
}
