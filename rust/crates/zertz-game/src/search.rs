use rand::RngExt;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;

use crate::board_encoding::{encode_board, GRID_SIZE, NUM_CHANNELS, RESERVE_SIZE};
use crate::zertz::{ZertzBoard, ZertzMove, classify_win, WinType};
use crate::move_encoding::{encode_distribution_nn, NN_POLICY_SIZE};
use core_game::game::{Game, Outcome, Player};
use core_game::mcts::arena::NodeId;
use core_game::mcts::search::{MctsSearch, CpuctStrategy};

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
    max_moves: u32,
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

            if boards[gi].outcome() != Outcome::Ongoing || move_counts[gi] >= max_moves {
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
}

/// Core self-play implementation factoring out business logic from Python bindings.
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

    let mut boards: Vec<ZertzBoard> = (0..num_games).map(|_| ZertzBoard::default()).collect();
    let arena_capacity = simulations + 64;
    let mut searches: Vec<MctsSearch<ZertzBoard>> = (0..num_games).map(|_| {
        let mut s = MctsSearch::new(arena_capacity);
        s.params.cpuct_strategy = CpuctStrategy::Constant { c_puct };
        s.params.max_children = simulations;
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

    // Main loop
    while active.iter().any(|&a| a) {
        let mcts_games: Vec<usize> = (0..num_games).filter(|&gi| active[gi]).collect();
        if mcts_games.is_empty() { break; }

        let n = mcts_games.len();
        total_turns += n as u32;

        // Decide full vs fast per game
        let is_full: Vec<bool> = if use_playout_cap {
            (0..n).map(|_| rng.random::<f32>() < playout_cap_p).collect()
        } else { vec![true; n] };
        let sim_caps: Vec<usize> = is_full.iter().map(|&f| if f { simulations } else { fast_cap }).collect();
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
                searches[gi].apply_root_dirichlet(dir_alpha, dir_epsilon);
            }
        }

        // Simulation rounds
        let mut game_sims: Vec<usize> = vec![0; n];
        loop {
            let mut leaf_ids: Vec<NodeId> = Vec::new();
            let mut leaf_game_idx: Vec<usize> = Vec::new();

            for _round in 0..play_batch_size {
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

        // Select and apply moves
        for (i, &gi) in mcts_games.iter().enumerate() {
            let dist = searches[gi].get_pruned_visit_distribution();
            let policy_vec = encode_distribution_nn(&dist);

            let is_capture_turn = dist.first().map_or(false, |(mv, _)| matches!(mv, ZertzMove::Capture { .. }));
            let is_mid_capture_turn = boards[gi].is_mid_capture();
            histories[gi].push((turn_board_offsets[i], turn_reserve_offsets[i], boards[gi].next_player(), !is_full[i], is_capture_turn, is_mid_capture_turn, policy_vec));

            let mv = if dist.is_empty() {
                ZertzMove::Pass
            } else if move_counts[gi] < temp_threshold && temperature > 0.01 {
                let weights: Vec<f32> = dist.iter().map(|(_, p)| p.powf(1.0 / temperature)).collect();
                let wi = WeightedIndex::new(&weights).map_err(|e| e.to_string())?;
                dist[wi.sample(&mut rng)].0
            } else {
                dist.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0
            };

            boards[gi].play(mv).map_err(|e| e.to_string())?;
            move_counts[gi] += 1;
            total_moves += 1;

            if boards[gi].outcome() != Outcome::Ongoing || move_counts[gi] >= max_moves {
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
    })
}
