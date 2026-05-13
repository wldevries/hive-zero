use std::time::Duration;

use rand::RngExt;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::board_encoding::{encode_board, GRID_SIZE, NUM_CHANNELS, RESERVE_SIZE};
use crate::zertz::{ZertzBoard, ZertzMove, classify_win, WinType};
use crate::move_encoding::{encode_distribution_nn, NN_POLICY_SIZE};
use core_game::game::{Game, Outcome, Player};
use core_game::mcts::arena::NodeId;
use core_game::mcts::search::{MctsSearch, CpuctStrategy, ForcedExploration};
use core_game::selfplay_config::{MctsConfig, PlayoutCapConfig};

const BOARD_FLAT: usize = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;

/// Pure-Rust battle result (no PyO3 types).
#[derive(Clone, Debug)]
pub struct BattleResult {
    pub wins_model1: u32,
    pub wins_model2: u32,
    pub draws: u32,
    pub wins_white: u32,
    pub wins_grey: u32,
    pub wins_black: u32,
    pub wins_combo: u32,
    pub game_lengths: Vec<u32>,
}

/// Eval callback type: (boards_flat, reserves_flat, n) -> (flat_policy_490, value)
pub type EvalFn = Box<dyn Fn(&[f32], &[f32], usize) -> Result<(Vec<f32>, Vec<f32>), String> + Send + Sync>;

/// Progress callback: finished, total, active, total_moves
pub type ProgressFn = Box<dyn Fn(u32, u32, u32, u32) + Send + Sync>;

/// Core best-move search for a single position.
pub fn best_move_core(
    board: &ZertzBoard,
    simulations: usize,
    c_puct: f32,
    eval_fn: EvalFn,
) -> Result<ZertzMove, String> {
    if board.outcome() != Outcome::Ongoing {
        return Err("Game is already over".to_string());
    }

    let mut search = MctsSearch::<ZertzBoard>::new(simulations + 64);
    search.params.cpuct_strategy = CpuctStrategy::Constant { c_puct };

    // Initial NN eval on root position.
    let mut board_buf = vec![0f32; BOARD_FLAT];
    let mut reserve_buf = vec![0f32; RESERVE_SIZE];
    encode_board(board, &mut board_buf, &mut reserve_buf);
    let (root_policy, _root_val) = eval_fn(&board_buf, &reserve_buf, 1)?;
    search.init(board, &root_policy);
    // No Dirichlet noise at inference time — Dirichlet is for self-play exploration only.

    // Simulation rounds (batch_size=8)
    let batch = 8usize;
    let mut done = 0usize;
    let mut flat = vec![0f32; batch * BOARD_FLAT];
    let mut flat_res = vec![0f32; batch * RESERVE_SIZE];
    while done < simulations {
        let leaves = search.select_leaves(batch.min(simulations - done));
        if leaves.is_empty() {
            break;
        }
        let nl = leaves.len();

        for (k, &leaf) in leaves.iter().enumerate() {
            let (board_enc, reserve_enc) = search.encode_leaf(leaf);
            flat[k * BOARD_FLAT..(k + 1) * BOARD_FLAT].copy_from_slice(&board_enc);
            flat_res[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE].copy_from_slice(&reserve_enc);
        }

        let (flat_policy, leaf_values) = eval_fn(
            &flat[..nl * BOARD_FLAT],
            &flat_res[..nl * RESERVE_SIZE],
            nl,
        )?;

        let policies: Vec<Vec<f32>> = (0..nl)
            .map(|k| flat_policy[k * NN_POLICY_SIZE..(k + 1) * NN_POLICY_SIZE].to_vec())
            .collect();

        search.expand_and_backprop(&policies, &leaf_values, &[]);
        done += nl;
    }

    let dist = search.get_pruned_visit_distribution();
    let best = dist
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(mv, _)| *mv)
        .unwrap_or(ZertzMove::Pass);
    Ok(best)
}

/// Core play_battle implementation that contains business logic only. It accepts
/// boxed callbacks for evaluations and progress so bindings can adapt platform
/// specific callables (Python, JS, native engine, etc.).
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

    let mut boards: Vec<ZertzBoard> = (0..num_games).map(|_| ZertzBoard::default()).collect();
    let arena_capacity = simulations + 64;
    let mut searches: Vec<MctsSearch<ZertzBoard>> = (0..num_games).map(|_| {
        let mut s = MctsSearch::new(arena_capacity);
        s.params.cpuct_strategy = CpuctStrategy::Constant { c_puct };
        s.params.max_children = simulations;
        s
    }).collect();
    let mut active = vec![true; num_games];
    let mut move_counts = vec![0u32; num_games];
    let mut finished_count = 0u32;
    // True once a game's tree has been rerooted and its root is already expanded with
    // priors from the previous search; such games skip the root NN eval + init().
    let mut search_warm: Vec<bool> = vec![false; num_games];

    let mut total_moves = 0u32;
    let mut wins_model1 = 0u32;
    let mut wins_model2 = 0u32;
    let mut draws = 0u32;
    let mut wins_white = 0u32;
    let mut wins_grey = 0u32;
    let mut wins_black = 0u32;
    let mut wins_combo = 0u32;
    let mut game_lengths: Vec<u32> = Vec::new();

    let use_fn1_for = |gi: usize, player: Player| -> bool {
        (gi < half) == (player == Player::Player1)
    };

    let call_evals = |flat_boards: &[f32], flat_reserves: &[f32], fn1_flags: &[bool], n: usize|
     -> Result<(Vec<f32>, Vec<f32>), String> {
        let (fp1, va1) = eval_fn1(flat_boards, flat_reserves, n)?;
        let (fp2, va2) = eval_fn2(flat_boards, flat_reserves, n)?;
        let mut flat = vec![0.0f32; n * NN_POLICY_SIZE];
        let mut value = vec![0.0f32; n];
        for i in 0..n {
            if fn1_flags[i] {
                flat[i * NN_POLICY_SIZE..(i + 1) * NN_POLICY_SIZE].copy_from_slice(&fp1[i * NN_POLICY_SIZE..(i + 1) * NN_POLICY_SIZE]);
                value[i] = va1[i];
            } else {
                flat[i * NN_POLICY_SIZE..(i + 1) * NN_POLICY_SIZE].copy_from_slice(&fp2[i * NN_POLICY_SIZE..(i + 1) * NN_POLICY_SIZE]);
                value[i] = va2[i];
            }
        }
        Ok((flat, value))
    };

    while active.iter().any(|&a| a) {
        let mcts_games: Vec<usize> = (0..num_games).filter(|&gi| active[gi]).collect();
        if mcts_games.is_empty() { break; }
        let n = mcts_games.len();

        // Only cold games need a root NN eval + init(). Warm games already have
        // an expanded root from the previous ply's reroot().
        let cold: Vec<usize> = (0..n)
            .filter(|&i| !search_warm[mcts_games[i]])
            .collect();

        if !cold.is_empty() {
            let nc = cold.len();
            let mut flat_boards = vec![0f32; nc * BOARD_FLAT];
            let mut flat_reserves = vec![0f32; nc * RESERVE_SIZE];
            let mut fn1_flags: Vec<bool> = Vec::with_capacity(nc);
            for (k, &ci) in cold.iter().enumerate() {
                let gi = mcts_games[ci];
                encode_board(&boards[gi], &mut flat_boards[k * BOARD_FLAT..(k + 1) * BOARD_FLAT], &mut flat_reserves[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE]);
                fn1_flags.push(use_fn1_for(gi, boards[gi].next_player()));
            }

            let (init_policy, _) = call_evals(&flat_boards, &flat_reserves, &fn1_flags, nc)?;
            for (k, &ci) in cold.iter().enumerate() {
                let gi = mcts_games[ci];
                searches[gi].init(&boards[gi], &init_policy[k * NN_POLICY_SIZE..(k + 1) * NN_POLICY_SIZE]);
            }
        }

        let mut game_sims = vec![0usize; n];
        loop {
            let mut leaf_ids: Vec<NodeId> = Vec::new();
            let mut leaf_game_idx: Vec<usize> = Vec::new();
            for _round in 0..play_batch_size {
                let mut any = false;
                for (i, &gi) in mcts_games.iter().enumerate() {
                    if game_sims[i] >= simulations { continue; }
                    let leaves = searches[gi].select_leaves(1);
                    let count = leaves.len();
                    if count > 0 { any = true; }
                    for leaf in leaves { leaf_ids.push(leaf); leaf_game_idx.push(i); }
                    game_sims[i] += count;
                }
                if !any { break; }
            }
            if leaf_ids.is_empty() { break; }

            let nl = leaf_ids.len();
            let mut leaf_boards_flat = vec![0f32; nl * BOARD_FLAT];
            let mut leaf_reserves_flat = vec![0f32; nl * RESERVE_SIZE];
            let mut leaf_fn1_flags: Vec<bool> = Vec::with_capacity(nl);
            for (k, (&leaf, &i)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
                let gi = mcts_games[i];
                let (board_enc, reserve_enc) = searches[gi].encode_leaf(leaf);
                leaf_boards_flat[k * BOARD_FLAT..(k + 1) * BOARD_FLAT].copy_from_slice(&board_enc);
                leaf_reserves_flat[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE].copy_from_slice(&reserve_enc);
                let leaf_player = searches[gi].get_leaf_player(leaf);
                leaf_fn1_flags.push(use_fn1_for(gi, leaf_player));
            }

            let (leaf_policy, leaf_values) = call_evals(&leaf_boards_flat, &leaf_reserves_flat, &leaf_fn1_flags, nl)?;

            let mut per_game_policies: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n];
            let mut per_game_values: Vec<Vec<f32>> = (0..n).map(|_| Vec::new()).collect();
            for (k, &i) in leaf_game_idx.iter().enumerate() {
                per_game_policies[i].push(leaf_policy[k * NN_POLICY_SIZE..(k + 1) * NN_POLICY_SIZE].to_vec());
                per_game_values[i].push(leaf_values[k]);
            }
            for (i, &gi) in mcts_games.iter().enumerate() {
                if per_game_policies[i].is_empty() { continue; }
                searches[gi].expand_and_backprop(&per_game_policies[i], &per_game_values[i], &[]);
            }
            if game_sims.iter().all(|&s| s >= simulations) { break; }
        }

        for (_i, &gi) in mcts_games.iter().enumerate() {
            let dist = searches[gi].get_pruned_visit_distribution();
            let mv = if dist.is_empty() { ZertzMove::Pass } else { dist.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0 };
            boards[gi].play(mv).expect("battle selected illegal move");
            move_counts[gi] += 1;
            total_moves += 1;

            if boards[gi].outcome() != Outcome::Ongoing {
                active[gi] = false;
                finished_count += 1;
                game_lengths.push(move_counts[gi]);
                search_warm[gi] = false;
                match boards[gi].outcome() {
                    Outcome::WonBy(winner) => {
                        let model1_won = (gi < half) == (winner == Player::Player1);
                        if model1_won { wins_model1 += 1; } else { wins_model2 += 1; }
                        match classify_win(&boards[gi], winner) {
                            WinType::FourWhite  => wins_white += 1,
                            WinType::FiveGrey   => wins_grey  += 1,
                            WinType::SixBlack   => wins_black += 1,
                            WinType::ThreeEach  => wins_combo += 1,
                            WinType::Draw       => {}
                        }
                    }
                    _ => { draws += 1; }
                }
            } else {
                // Reroot to preserve the subtree for the chosen move.
                // Falls back to a cold init next ply if the move wasn't expanded.
                search_warm[gi] = searches[gi].reroot(mv);
            }
        }

        if let Some(pfn) = &progress_fn {
            let active_count = active.iter().filter(|&&a| a).count() as u32;
            pfn(finished_count, num_games as u32, active_count, total_moves);
        }
    }

    Ok(BattleResult { wins_model1, wins_model2, draws, wins_white, wins_grey, wins_black, wins_combo, game_lengths })
}

/// Battle the NN model (MCTS) against the heuristic alpha-beta bot, in
/// parallel across `num_games` games. At each ply, active games are
/// partitioned by whose turn it is:
///
/// - **Model-to-move** games batch their MCTS leaf evaluations through a
///   single `eval_fn` call (same shape as `play_battle_core` — leaves
///   across games are concatenated, and `play_batch_size` controls how
///   many leaf-collection rounds run before each NN call).
/// - **Bot-to-move** games are resolved via
///   `alphabeta::alphabeta_best_move_with_budget`, run **in parallel via
///   rayon** since each call is independent. `bot_time_budget` is a
///   per-call wall-clock cap; `None` means depth-only.
///
/// Half the games have the model as P1, the other half as P2, so the
/// `wins_model1` field counts NN wins regardless of color. The model's
/// MCTS tree is rerooted after every move (including bot moves) so the
/// next model-turn round can warm-start.
///
/// `progress_fn` ticks once per ply round and reports
/// `(finished_games, total_games, active_games, total_moves)`.
pub fn play_battle_vs_bot_core(
    num_games: usize,
    simulations: usize,
    c_puct: f32,
    play_batch_size: usize,
    bot_depth: u32,
    bot_time_budget: Option<Duration>,
    eval_fn: EvalFn,
    progress_fn: Option<ProgressFn>,
) -> Result<BattleResult, String> {
    let half = num_games / 2;
    // Model is P1 for the first half, P2 for the second half. This
    // pairing cancels color advantage out of the model's score.
    let model_is_p1 = |gi: usize| gi < half;

    let mut boards: Vec<ZertzBoard> = (0..num_games).map(|_| ZertzBoard::default()).collect();
    let arena_capacity = simulations + 64;
    let mut searches: Vec<MctsSearch<ZertzBoard>> = (0..num_games)
        .map(|_| {
            let mut s = MctsSearch::new(arena_capacity);
            s.params.cpuct_strategy = CpuctStrategy::Constant { c_puct };
            s.params.max_children = simulations;
            s
        })
        .collect();
    let mut active = vec![true; num_games];
    let mut move_counts = vec![0u32; num_games];
    let mut finished_count = 0u32;
    // Search trees are warm only after a successful reroot. Bot moves
    // also reroot, so consecutive bot turns (mid-capture) keep the tree
    // valid for the next model turn.
    let mut search_warm: Vec<bool> = vec![false; num_games];

    let mut total_moves = 0u32;
    let mut wins_model1 = 0u32;
    let mut wins_model2 = 0u32;
    let mut draws = 0u32;
    let mut wins_white = 0u32;
    let mut wins_grey = 0u32;
    let mut wins_black = 0u32;
    let mut wins_combo = 0u32;
    let mut game_lengths: Vec<u32> = Vec::new();

    while active.iter().any(|&a| a) {
        // Partition active games by who's to move this ply.
        let mut mcts_games: Vec<usize> = Vec::new();
        let mut bot_games: Vec<usize> = Vec::new();
        for gi in 0..num_games {
            if !active[gi] { continue; }
            let model_to_move = (boards[gi].next_player() == Player::Player1) == model_is_p1(gi);
            if model_to_move {
                mcts_games.push(gi);
            } else {
                bot_games.push(gi);
            }
        }

        // Bot moves: each alphabeta call is fully independent (own
        // `SearchContext`, own TT, reads `boards[gi]` immutably), so on
        // native builds we run them in parallel via rayon. With ~20
        // bot-turn games at depth 3 this is the only practical way to
        // keep the per-ply wall-clock under control on a multi-core box.
        // The wasm crate disables the `parallel` feature (rayon doesn't
        // compile on wasm32) and falls back to a sequential loop, which
        // is fine because wasm doesn't run battle-vs-bot anyway. Result
        // list is built up before any board mutation so the partition
        // can't shift mid-ply.
        #[cfg(feature = "parallel")]
        let bot_chosen: Vec<(usize, ZertzMove)> = bot_games
            .par_iter()
            .map(|&gi| {
                let mv = crate::alphabeta::alphabeta_best_move_with_budget(
                    &boards[gi],
                    bot_depth,
                    bot_time_budget,
                );
                (gi, mv)
            })
            .collect();
        #[cfg(not(feature = "parallel"))]
        let bot_chosen: Vec<(usize, ZertzMove)> = bot_games
            .iter()
            .map(|&gi| {
                let mv = crate::alphabeta::alphabeta_best_move_with_budget(
                    &boards[gi],
                    bot_depth,
                    bot_time_budget,
                );
                (gi, mv)
            })
            .collect();
        let mut chosen_moves: Vec<(usize, ZertzMove)> = Vec::with_capacity(num_games);
        chosen_moves.extend(bot_chosen);

        // Model moves: batched MCTS across mcts_games (single eval_fn).
        if !mcts_games.is_empty() {
            let n = mcts_games.len();

            // Cold init for any game whose tree isn't warm. Warm games
            // already have an expanded root with priors from the
            // previous reroot.
            let cold: Vec<usize> = (0..n).filter(|&i| !search_warm[mcts_games[i]]).collect();
            if !cold.is_empty() {
                let nc = cold.len();
                let mut flat_boards = vec![0f32; nc * BOARD_FLAT];
                let mut flat_reserves = vec![0f32; nc * RESERVE_SIZE];
                for (k, &ci) in cold.iter().enumerate() {
                    let gi = mcts_games[ci];
                    encode_board(
                        &boards[gi],
                        &mut flat_boards[k * BOARD_FLAT..(k + 1) * BOARD_FLAT],
                        &mut flat_reserves[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE],
                    );
                }
                let (init_policy, _) = eval_fn(&flat_boards, &flat_reserves, nc)?;
                for (k, &ci) in cold.iter().enumerate() {
                    let gi = mcts_games[ci];
                    searches[gi].init(
                        &boards[gi],
                        &init_policy[k * NN_POLICY_SIZE..(k + 1) * NN_POLICY_SIZE],
                    );
                }
            }

            // Simulation rounds: select leaves across games, batch-eval, expand.
            let mut game_sims = vec![0usize; n];
            loop {
                let mut leaf_ids: Vec<NodeId> = Vec::new();
                let mut leaf_game_idx: Vec<usize> = Vec::new();
                for _round in 0..play_batch_size {
                    let mut any = false;
                    for (i, _) in mcts_games.iter().enumerate() {
                        if game_sims[i] >= simulations { continue; }
                        let gi = mcts_games[i];
                        let leaves = searches[gi].select_leaves(1);
                        let count = leaves.len();
                        if count > 0 { any = true; }
                        for leaf in leaves { leaf_ids.push(leaf); leaf_game_idx.push(i); }
                        game_sims[i] += count;
                    }
                    if !any { break; }
                }
                if leaf_ids.is_empty() { break; }

                let nl = leaf_ids.len();
                let mut leaf_boards_flat = vec![0f32; nl * BOARD_FLAT];
                let mut leaf_reserves_flat = vec![0f32; nl * RESERVE_SIZE];
                for (k, (&leaf, &i)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
                    let gi = mcts_games[i];
                    let (board_enc, reserve_enc) = searches[gi].encode_leaf(leaf);
                    leaf_boards_flat[k * BOARD_FLAT..(k + 1) * BOARD_FLAT].copy_from_slice(&board_enc);
                    leaf_reserves_flat[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE].copy_from_slice(&reserve_enc);
                }

                let (leaf_policy, leaf_values) =
                    eval_fn(&leaf_boards_flat, &leaf_reserves_flat, nl)?;

                let mut per_game_policies: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n];
                let mut per_game_values: Vec<Vec<f32>> = (0..n).map(|_| Vec::new()).collect();
                for (k, &i) in leaf_game_idx.iter().enumerate() {
                    per_game_policies[i].push(
                        leaf_policy[k * NN_POLICY_SIZE..(k + 1) * NN_POLICY_SIZE].to_vec(),
                    );
                    per_game_values[i].push(leaf_values[k]);
                }
                for (i, _) in mcts_games.iter().enumerate() {
                    if per_game_policies[i].is_empty() { continue; }
                    let gi = mcts_games[i];
                    searches[gi].expand_and_backprop(&per_game_policies[i], &per_game_values[i], &[]);
                }
                if game_sims.iter().all(|&s| s >= simulations) { break; }
            }

            // Pick best move per game by visit count.
            for &gi in &mcts_games {
                let dist = searches[gi].get_pruned_visit_distribution();
                let mv = if dist.is_empty() {
                    ZertzMove::Pass
                } else {
                    dist.iter()
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .unwrap()
                        .0
                };
                chosen_moves.push((gi, mv));
            }
        }

        // Apply all moves and update game state.
        for (gi, mv) in chosen_moves {
            boards[gi].play(mv).expect("battle selected illegal move");
            move_counts[gi] += 1;
            total_moves += 1;

            if boards[gi].outcome() != Outcome::Ongoing {
                active[gi] = false;
                finished_count += 1;
                game_lengths.push(move_counts[gi]);
                search_warm[gi] = false;
                match boards[gi].outcome() {
                    Outcome::WonBy(winner) => {
                        let model_won = model_is_p1(gi) == (winner == Player::Player1);
                        if model_won { wins_model1 += 1; } else { wins_model2 += 1; }
                        match classify_win(&boards[gi], winner) {
                            WinType::FourWhite => wins_white += 1,
                            WinType::FiveGrey => wins_grey += 1,
                            WinType::SixBlack => wins_black += 1,
                            WinType::ThreeEach => wins_combo += 1,
                            WinType::Draw => {}
                        }
                    }
                    _ => { draws += 1; }
                }
            } else {
                // Reroot the model's tree to the move that was actually played
                // (model or bot) so the next model-turn round can warm-start.
                search_warm[gi] = searches[gi].reroot(mv);
            }
        }

        if let Some(pfn) = &progress_fn {
            let active_count = active.iter().filter(|&&a| a).count() as u32;
            pfn(finished_count, num_games as u32, active_count, total_moves);
        }
    }

    Ok(BattleResult {
        wins_model1, wins_model2, draws,
        wins_white, wins_grey, wins_black, wins_combo,
        game_lengths,
    })
}

fn mean_std(sum: f64, sum_sq: f64, count: u64) -> (f32, f32) {
    if count == 0 { return (0.0, 0.0); }
    let n = count as f64;
    let mean = sum / n;
    let var = (sum_sq / n) - mean * mean;
    (mean as f32, var.max(0.0).sqrt() as f32)
}

/// Result of self-play (pure Rust)
#[derive(Clone, Debug)]
pub struct SelfPlayResult {
    pub board_data: Vec<f32>,
    pub reserve_data: Vec<f32>,
    pub policy_data: Vec<f32>,
    pub value_targets: Vec<f32>,
    pub value_only_flags: Vec<bool>,
    pub capture_turn_flags: Vec<bool>,
    pub mid_capture_turn_flags: Vec<bool>,
    pub num_samples: usize,
    pub wins_p1: u32,
    pub wins_p2: u32,
    pub draws: u32,
    pub wins_white: u32,
    pub wins_grey: u32,
    pub wins_black: u32,
    pub wins_combo: u32,
    pub total_moves: u32,
    pub game_lengths: Vec<u32>,
    pub decisive_lengths: Vec<u32>,
    pub full_search_turns: u32,
    pub total_turns: u32,
    pub isolation_captures: u32,
    pub jump_captures: u32,
    pub sample_board_data: Vec<(String, String)>,

    pub top1_visit_fraction_mean: f32,
    pub top1_visit_fraction_std: f32,
    pub search_depth_mean: f32,
    pub search_depth_std: f32,
    pub valid_moves_mean: f32,
    pub valid_moves_std: f32,
}

/// Core self-play implementation factoring out business logic from Python bindings.
pub fn play_selfplay_core(
    num_games: usize,
    mcts: MctsConfig,
    playout_cap: PlayoutCapConfig,
    eval_fn: EvalFn,
    progress_fn: Option<ProgressFn>,
) -> Result<SelfPlayResult, String> {
    let use_playout_cap = playout_cap.enabled();

    let mut boards: Vec<ZertzBoard> = (0..num_games).map(|_| ZertzBoard::default()).collect();
    let arena_capacity = mcts.simulations + 64;
    let mut searches: Vec<MctsSearch<ZertzBoard>> = (0..num_games).map(|_| {
        let mut s = MctsSearch::new(arena_capacity);
        s.params.cpuct_strategy = CpuctStrategy::Constant { c_puct: mcts.c_puct };
        s.params.max_children = mcts.simulations;
        s.params.draw_contempt = mcts.draw_contempt;
        if mcts.forced_playouts {
            s.params.forced_exploration = ForcedExploration::Soft { selection_k: 0.5, pruning_k: 2.0 };
        }
        s
    }).collect();
    let mut move_counts: Vec<u32> = vec![0; num_games];
    let mut active: Vec<bool> = vec![true; num_games];
    let mut finished_count: u32 = 0;
    // True once a game's tree has been rerooted and its root is already expanded with
    // priors from the previous search; such games skip the root NN eval + init().
    let mut search_warm: Vec<bool> = vec![false; num_games];

    let mut histories: Vec<Vec<(usize, usize, Player, bool, bool, bool, Vec<f32>)>> = (0..num_games).map(|_| Vec::new()).collect();
    let mut board_buf: Vec<f32> = Vec::new();
    let mut reserve_buf: Vec<f32> = Vec::new();

    let mut rng = rand::rng();
    let mut wins_p1 = 0u32;
    let mut wins_p2 = 0u32;
    let mut draws = 0u32;
    let mut wins_white = 0u32;
    let mut wins_grey = 0u32;
    let mut wins_black = 0u32;
    let mut wins_combo = 0u32;
    let mut total_moves = 0u32;
    let mut game_lengths: Vec<u32> = Vec::new();
    let mut decisive_lengths: Vec<u32> = Vec::new();
    let mut full_search_turns: u32 = 0;
    let mut total_turns: u32 = 0;
    let mut isolation_captures: u32 = 0;
    let mut jump_captures: u32 = 0;
    let mut sample_board_data: Vec<(String, String)> = Vec::new();

    let mut session_top1_sum = 0f64;
    let mut session_top1_sum_sq = 0f64;
    let mut session_top1_count = 0u64;
    let mut session_depth_sum = 0f64;
    let mut session_depth_sum_sq = 0f64;
    let mut session_depth_count = 0u64;
    let mut session_moves_sum = 0f64;
    let mut session_moves_sum_sq = 0f64;
    let mut session_moves_count = 0u64;

    // Main loop
    while active.iter().any(|&a| a) {
        let mcts_games: Vec<usize> = (0..num_games).filter(|&gi| active[gi]).collect();
        if mcts_games.is_empty() { break; }

        let n = mcts_games.len();
        total_turns += n as u32;

        // Decide full vs fast per game
        let is_full: Vec<bool> = if use_playout_cap {
            (0..n).map(|_| rng.random::<f32>() < playout_cap.p).collect()
        } else { vec![true; n] };
        let sim_caps: Vec<usize> = is_full.iter().map(|&f| if f { mcts.simulations } else { playout_cap.fast_cap }).collect();
        full_search_turns += is_full.iter().filter(|&&f| f).count() as u32;

        // Encode positions into the training buffer for ALL active games (warm or cold).
        let mut turn_board_offsets: Vec<usize> = Vec::with_capacity(n);
        let mut turn_reserve_offsets: Vec<usize> = Vec::with_capacity(n);
        for &gi in mcts_games.iter() {
            let boff = board_buf.len();
            board_buf.resize(boff + BOARD_FLAT, 0.0);
            let roff = reserve_buf.len();
            reserve_buf.resize(roff + RESERVE_SIZE, 0.0);
            encode_board(&boards[gi], &mut board_buf[boff..boff + BOARD_FLAT], &mut reserve_buf[roff..roff + RESERVE_SIZE]);
            turn_board_offsets.push(boff);
            turn_reserve_offsets.push(roff);
        }

        // Only cold games need a root NN eval + init(). Warm games already have
        // an expanded root with priors from the previous ply's reroot().
        let cold: Vec<usize> = (0..n)
            .filter(|&i| !search_warm[mcts_games[i]])
            .collect();

        if !cold.is_empty() {
            let nc = cold.len();
            let mut flat_boards = vec![0f32; nc * BOARD_FLAT];
            let mut flat_reserves = vec![0f32; nc * RESERVE_SIZE];
            for (k, &ci) in cold.iter().enumerate() {
                let gi = mcts_games[ci];
                encode_board(&boards[gi], &mut flat_boards[k * BOARD_FLAT..(k + 1) * BOARD_FLAT], &mut flat_reserves[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE]);
            }

            // Initial NN eval for cold roots
            let (init_policy, _) = eval_fn(&flat_boards, &flat_reserves, nc)?;

            for (k, &ci) in cold.iter().enumerate() {
                let gi = mcts_games[ci];
                searches[gi].init(&boards[gi], &init_policy[k * NN_POLICY_SIZE..(k + 1) * NN_POLICY_SIZE]);
            }
        }

        // Apply fresh Dirichlet noise to every full-search root (both warm and cold).
        for (i, &gi) in mcts_games.iter().enumerate() {
            if is_full[i] {
                searches[gi].apply_root_dirichlet(mcts.dir_alpha, mcts.dir_epsilon);
            }
        }

        // Disable forced playouts on fast turns: their policy target is
        // discarded (value-only training), and the tied-visit pruning quirk
        // can mis-select the played move from a low-N distribution.
        if mcts.forced_playouts {
            for (i, &gi) in mcts_games.iter().enumerate() {
                searches[gi].params.forced_exploration = if is_full[i] {
                    ForcedExploration::Soft { selection_k: 0.5, pruning_k: 2.0 }
                } else {
                    ForcedExploration::None
                };
            }
        }

        // Simulation rounds
        let mut game_sims: Vec<usize> = vec![0; n];
        loop {
            let mut leaf_ids: Vec<NodeId> = Vec::new();
            let mut leaf_game_idx: Vec<usize> = Vec::new();

            for _round in 0..mcts.play_batch_size {
                let mut any_collected = false;
                for (i, &gi) in mcts_games.iter().enumerate() {
                    if game_sims[i] >= sim_caps[i] { continue; }
                    let leaves = searches[gi].select_leaves(1);
                    let count = leaves.len();
                    if count > 0 { any_collected = true; }
                    for leaf in leaves { leaf_ids.push(leaf); leaf_game_idx.push(i); }
                    game_sims[i] += count;
                }
                if !any_collected { break; }
            }

            if leaf_ids.is_empty() { break; }

            let nl = leaf_ids.len();
            let mut leaf_boards_flat = vec![0f32; nl * BOARD_FLAT];
            let mut leaf_reserves_flat = vec![0f32; nl * RESERVE_SIZE];
            for (k, (&leaf, &i)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
                let gi = mcts_games[i];
                let (board_enc, reserve_enc) = searches[gi].encode_leaf(leaf);
                leaf_boards_flat[k * BOARD_FLAT..(k + 1) * BOARD_FLAT].copy_from_slice(&board_enc);
                leaf_reserves_flat[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE].copy_from_slice(&reserve_enc);
            }

            let (leaf_policy, leaf_values) = eval_fn(&leaf_boards_flat, &leaf_reserves_flat, nl)?;

            let mut per_game_policies: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n];
            let mut per_game_values: Vec<Vec<f32>> = (0..n).map(|_| Vec::new()).collect();
            for (k, &i) in leaf_game_idx.iter().enumerate() {
                per_game_policies[i].push(leaf_policy[k * NN_POLICY_SIZE..(k + 1) * NN_POLICY_SIZE].to_vec());
                per_game_values[i].push(leaf_values[k]);
            }
            for (i, &gi) in mcts_games.iter().enumerate() {
                if per_game_policies[i].is_empty() { continue; }
                searches[gi].expand_and_backprop(&per_game_policies[i], &per_game_values[i], &[]);
            }

            if game_sims.iter().zip(sim_caps.iter()).all(|(s, c)| *s >= *c) { break; }
        }

        // Collect per-turn MCTS stats for full-search turns. We always drain
        // `take_depth_stats` to reset the per-search counter regardless.
        for (i, &gi) in mcts_games.iter().enumerate() {
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

        // Select and apply moves
        for (i, &gi) in mcts_games.iter().enumerate() {
            let dist = searches[gi].get_pruned_visit_distribution();
            let policy_vec = encode_distribution_nn(&dist);

            let is_capture_turn = dist.first().map_or(false, |(mv, _)| matches!(mv, ZertzMove::Capture { .. }));
            let is_mid_capture_turn = boards[gi].is_mid_capture();
            histories[gi].push((turn_board_offsets[i], turn_reserve_offsets[i], boards[gi].next_player(), !is_full[i], is_capture_turn, is_mid_capture_turn, policy_vec));

            let mv = if dist.is_empty() {
                ZertzMove::Pass
            } else if move_counts[gi] < mcts.temp_threshold && mcts.temperature > 0.01 {
                let weights: Vec<f32> = dist.iter().map(|(_, p)| p.powf(1.0 / mcts.temperature)).collect();
                let wi = WeightedIndex::new(&weights).map_err(|e| e.to_string())?;
                dist[wi.sample(&mut rng)].0
            } else {
                dist.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0
            };

            boards[gi].play(mv).map_err(|e| e.to_string())?;
            move_counts[gi] += 1;
            total_moves += 1;

            if boards[gi].outcome() != Outcome::Ongoing {
                active[gi] = false;
                finished_count += 1;
                search_warm[gi] = false;
                let len = move_counts[gi];
                game_lengths.push(len);
                isolation_captures += boards[gi].isolation_captures.iter().flat_map(|p| p.iter()).map(|&c| c as u32).sum::<u32>();
                jump_captures += boards[gi].jump_captures.iter().flat_map(|p| p.iter()).map(|&c| c as u32).sum::<u32>();
                match boards[gi].outcome() {
                    Outcome::WonBy(winner) => {
                        if winner == Player::Player1 { wins_p1 += 1; } else { wins_p2 += 1; }
                        decisive_lengths.push(len);
                        let win_type = classify_win(&boards[gi], winner);
                        match win_type {
                            WinType::FourWhite => wins_white += 1,
                            WinType::FiveGrey  => wins_grey += 1,
                            WinType::SixBlack  => wins_black += 1,
                            WinType::ThreeEach => wins_combo += 1,
                            WinType::Draw      => {}
                        }
                        let label = format!("{} wins ({} moves)", if winner == Player::Player1 { "P1" } else { "P2" }, len);
                        sample_board_data.push((label, format!("{}", boards[gi])));
                    }
                    _ => { draws += 1; }
                }
            } else {
                // Reroot to preserve the subtree for the chosen move.
                // Falls back to a cold init next ply if the move wasn't expanded.
                search_warm[gi] = searches[gi].reroot(mv);
            }
        }

        if let Some(pfn) = &progress_fn {
            let active_count = active.iter().filter(|&&a| a).count() as u32;
            pfn(finished_count, num_games as u32, active_count, total_moves);
        }
    }

    // Build training data
    let total_samples: usize = histories.iter().map(|h| h.len()).sum();
    let mut board_data = Vec::with_capacity(total_samples * BOARD_FLAT);
    let mut reserve_data = Vec::with_capacity(total_samples * RESERVE_SIZE);
    let mut policy_data = Vec::with_capacity(total_samples * NN_POLICY_SIZE);
    let mut value_targets = Vec::with_capacity(total_samples);
    let mut value_only_flags = Vec::with_capacity(total_samples);
    let mut capture_turn_flags = Vec::with_capacity(total_samples);
    let mut mid_capture_turn_flags = Vec::with_capacity(total_samples);

    for (gi, history) in histories.iter().enumerate() {
        let outcome = boards[gi].outcome();
        for record in history {
            let (boff, roff, player, is_value_only, is_capture_turn, is_mid_capture_turn, policy_vec) = record;
            board_data.extend_from_slice(&board_buf[*boff..*boff + BOARD_FLAT]);
            reserve_data.extend_from_slice(&reserve_buf[*roff..*roff + RESERVE_SIZE]);
            policy_data.extend_from_slice(&policy_vec);
            let value = match outcome {
                Outcome::WonBy(winner) => if winner == *player { 1.0f32 } else { -1.0f32 },
                _ => 0.0f32,
            };
            value_targets.push(value);
            value_only_flags.push(*is_value_only);
            capture_turn_flags.push(*is_capture_turn);
            mid_capture_turn_flags.push(*is_mid_capture_turn);
        }
    }

    let (top1_visit_fraction_mean, top1_visit_fraction_std) =
        mean_std(session_top1_sum, session_top1_sum_sq, session_top1_count);
    let (search_depth_mean, search_depth_std) =
        mean_std(session_depth_sum, session_depth_sum_sq, session_depth_count);
    let (valid_moves_mean, valid_moves_std) =
        mean_std(session_moves_sum, session_moves_sum_sq, session_moves_count);

    Ok(SelfPlayResult {
        board_data,
        reserve_data,
        policy_data,
        value_targets,
        value_only_flags,
        capture_turn_flags,
        mid_capture_turn_flags,
        num_samples: total_samples,
        wins_p1,
        wins_p2,
        draws,
        wins_white,
        wins_grey,
        wins_black,
        wins_combo,
        total_moves,
        game_lengths,
        decisive_lengths,
        full_search_turns,
        total_turns,
        isolation_captures,
        jump_captures,
        sample_board_data,
        top1_visit_fraction_mean,
        top1_visit_fraction_std,
        search_depth_mean,
        search_depth_std,
        valid_moves_mean,
        valid_moves_std,
    })
}
