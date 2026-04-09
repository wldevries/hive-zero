use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;

use zertz_game::board_encoding::{encode_board, GRID_SIZE, NUM_CHANNELS, RESERVE_SIZE};
use zertz_game::mcts::arena::NodeId;
use zertz_game::mcts::search::{MctsSearch, PolicyHeads, PLACE_HEAD_SIZE, CAP_HEAD_SIZE};
use zertz_game::random_play::{classify_win, WinType};
use zertz_game::zertz::{ZertzBoard, ZertzMove};
use zertz_game::move_encoding::{encode_move, POLICY_SIZE};
use core_game::game::{Game, Outcome, Player};
use core_game::symmetry::{D6Symmetry, Symmetry, apply_d6_sym_spatial};

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

/// Eval callback type: (boards_flat, reserves_flat, n) -> (place, src, dst, value)
pub type EvalFn = Box<dyn Fn(&[f32], &[f32], usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> + Send + Sync>;

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

    let mut search = MctsSearch::new(simulations + 64);
    search.c_puct = c_puct;

    // Initial NN eval on root position.
    let mut board_buf = vec![0f32; BOARD_FLAT];
    let mut reserve_buf = vec![0f32; RESERVE_SIZE];
    encode_board(board, &mut board_buf, &mut reserve_buf);
    let (root_place, root_src, root_dst, _root_val) = eval_fn(&board_buf, &reserve_buf, 1)?;
    let root_heads = PolicyHeads {
        place: &root_place,
        cap_source: &root_src,
        cap_dest: &root_dst,
    };

    search.init(board, &root_heads);
    search.apply_root_dirichlet(0.3, 0.25);

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

        let (lp_place_data, lp_src_data, lp_dst_data, leaf_values) = eval_fn(
            &flat[..nl * BOARD_FLAT],
            &flat_res[..nl * RESERVE_SIZE],
            nl,
        )?;

        let leaf_heads: Vec<PolicyHeads> = (0..nl)
            .map(|k| PolicyHeads {
                place: &lp_place_data[k * PLACE_HEAD_SIZE..(k + 1) * PLACE_HEAD_SIZE],
                cap_source: &lp_src_data[k * CAP_HEAD_SIZE..(k + 1) * CAP_HEAD_SIZE],
                cap_dest: &lp_dst_data[k * CAP_HEAD_SIZE..(k + 1) * CAP_HEAD_SIZE],
            })
            .collect();

        search.expand_and_backprop(&leaves, &leaf_heads, &leaf_values);
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
    let mut searches: Vec<MctsSearch> = (0..num_games).map(|_| {
        let mut s = MctsSearch::new(arena_capacity);
        s.c_puct = c_puct;
        s
    }).collect();
    let mut active = vec![true; num_games];
    let mut move_counts = vec![0u32; num_games];
    let mut finished_count = 0u32;

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
     -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let (pl1, sr1, ds1, va1) = eval_fn1(flat_boards, flat_reserves, n)?;
        let (pl2, sr2, ds2, va2) = eval_fn2(flat_boards, flat_reserves, n)?;
        let mut place = vec![0.0f32; n * PLACE_HEAD_SIZE];
        let mut src = vec![0.0f32; n * CAP_HEAD_SIZE];
        let mut dst = vec![0.0f32; n * CAP_HEAD_SIZE];
        let mut value = vec![0.0f32; n];
        for i in 0..n {
            if fn1_flags[i] {
                place[i * PLACE_HEAD_SIZE..(i + 1) * PLACE_HEAD_SIZE].copy_from_slice(&pl1[i * PLACE_HEAD_SIZE..(i + 1) * PLACE_HEAD_SIZE]);
                src[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE].copy_from_slice(&sr1[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE]);
                dst[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE].copy_from_slice(&ds1[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE]);
                value[i] = va1[i];
            } else {
                place[i * PLACE_HEAD_SIZE..(i + 1) * PLACE_HEAD_SIZE].copy_from_slice(&pl2[i * PLACE_HEAD_SIZE..(i + 1) * PLACE_HEAD_SIZE]);
                src[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE].copy_from_slice(&sr2[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE]);
                dst[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE].copy_from_slice(&ds2[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE]);
                value[i] = va2[i];
            }
        }
        Ok((place, src, dst, value))
    };

    let mut rng = rand::thread_rng();
    let place_channels = PLACE_HEAD_SIZE / (GRID_SIZE * GRID_SIZE);

    while active.iter().any(|&a| a) {
        let mcts_games: Vec<usize> = (0..num_games).filter(|&gi| active[gi]).collect();
        if mcts_games.is_empty() { break; }
        let n = mcts_games.len();

        let root_syms: Vec<D6Symmetry> = (0..n).map(|_| D6Symmetry::random(&mut rng)).collect();
        let mut flat_boards = vec![0f32; n * BOARD_FLAT];
        let mut flat_reserves = vec![0f32; n * RESERVE_SIZE];
        let mut fn1_flags: Vec<bool> = Vec::with_capacity(n);
        for (i, &gi) in mcts_games.iter().enumerate() {
            encode_board(&boards[gi], &mut flat_boards[i * BOARD_FLAT..(i + 1) * BOARD_FLAT], &mut flat_reserves[i * RESERVE_SIZE..(i + 1) * RESERVE_SIZE]);
            apply_d6_sym_spatial(&mut flat_boards[i * BOARD_FLAT..(i + 1) * BOARD_FLAT], root_syms[i], NUM_CHANNELS, GRID_SIZE);
            fn1_flags.push(use_fn1_for(gi, boards[gi].next_player()));
        }

        let (mut init_place, mut init_src, mut init_dst, _) = call_evals(&flat_boards, &flat_reserves, &fn1_flags, n)?;
        for i in 0..n {
            let inv = root_syms[i].inverse();
            apply_d6_sym_spatial(&mut init_place[i * PLACE_HEAD_SIZE..(i + 1) * PLACE_HEAD_SIZE], inv, place_channels, GRID_SIZE);
            apply_d6_sym_spatial(&mut init_src[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE], inv, 1, GRID_SIZE);
            apply_d6_sym_spatial(&mut init_dst[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE], inv, 1, GRID_SIZE);
        }
        for (i, &gi) in mcts_games.iter().enumerate() {
            let heads = PolicyHeads {
                place: &init_place[i * PLACE_HEAD_SIZE..(i + 1) * PLACE_HEAD_SIZE],
                cap_source: &init_src[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE],
                cap_dest: &init_dst[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE],
            };
            searches[gi].init(&boards[gi], &heads);
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
            let mut leaf_syms: Vec<D6Symmetry> = Vec::with_capacity(nl);
            for (k, (&leaf, &i)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
                let gi = mcts_games[i];
                let (board_enc, reserve_enc) = searches[gi].encode_leaf(leaf);
                leaf_boards_flat[k * BOARD_FLAT..(k + 1) * BOARD_FLAT].copy_from_slice(&board_enc);
                leaf_reserves_flat[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE].copy_from_slice(&reserve_enc);
                let sym = D6Symmetry::random(&mut rng);
                apply_d6_sym_spatial(&mut leaf_boards_flat[k * BOARD_FLAT..(k + 1) * BOARD_FLAT], sym, NUM_CHANNELS, GRID_SIZE);
                leaf_syms.push(sym);
                let leaf_player = searches[gi].get_leaf_player(leaf);
                leaf_fn1_flags.push(use_fn1_for(gi, leaf_player));
            }

            let (leaf_place, leaf_src, leaf_dst, leaf_values) = call_evals(&leaf_boards_flat, &leaf_reserves_flat, &leaf_fn1_flags, nl)?;

            struct LeafHeadData { place: Vec<f32>, src: Vec<f32>, dst: Vec<f32> }
            let mut per_game_leaves: Vec<Vec<NodeId>> = vec![Vec::new(); n];
            let mut per_game_head_data: Vec<Vec<LeafHeadData>> = (0..n).map(|_| Vec::new()).collect();
            let mut per_game_values: Vec<Vec<f32>> = (0..n).map(|_| Vec::new()).collect();
            for (k, (&leaf, &i)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
                let inv = leaf_syms[k].inverse();
                let mut place = leaf_place[k * PLACE_HEAD_SIZE..(k + 1) * PLACE_HEAD_SIZE].to_vec();
                let mut src = leaf_src[k * CAP_HEAD_SIZE..(k + 1) * CAP_HEAD_SIZE].to_vec();
                let mut dst = leaf_dst[k * CAP_HEAD_SIZE..(k + 1) * CAP_HEAD_SIZE].to_vec();
                apply_d6_sym_spatial(&mut place, inv, place_channels, GRID_SIZE);
                apply_d6_sym_spatial(&mut src, inv, 1, GRID_SIZE);
                apply_d6_sym_spatial(&mut dst, inv, 1, GRID_SIZE);
                per_game_leaves[i].push(leaf);
                per_game_head_data[i].push(LeafHeadData { place, src, dst });
                per_game_values[i].push(leaf_values[k]);
            }
            for (i, &gi) in mcts_games.iter().enumerate() {
                if per_game_leaves[i].is_empty() { continue; }
                let heads: Vec<PolicyHeads> = per_game_head_data[i].iter().map(|d| PolicyHeads {
                    place: &d.place,
                    cap_source: &d.src,
                    cap_dest: &d.dst,
                }).collect();
                searches[gi].expand_and_backprop(&per_game_leaves[i], &heads, &per_game_values[i]);
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
    let mut searches: Vec<MctsSearch> = (0..num_games).map(|_| {
        let mut s = MctsSearch::new(arena_capacity);
        s.c_puct = c_puct;
        s
    }).collect();
    let mut move_counts: Vec<u32> = vec![0; num_games];
    let mut active: Vec<bool> = vec![true; num_games];
    let mut finished_count: u32 = 0;

    let mut histories: Vec<Vec<(usize, usize, Player, bool, bool, bool, Vec<f32>)>> = (0..num_games).map(|_| Vec::new()).collect();
    let mut board_buf: Vec<f32> = Vec::new();
    let mut reserve_buf: Vec<f32> = Vec::new();

    let mut rng = rand::thread_rng();

    // Stats
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
            (0..n).map(|_| rng.gen::<f32>() < playout_cap_p).collect()
        } else { vec![true; n] };
        let sim_caps: Vec<usize> = is_full.iter().map(|&f| if f { simulations } else { fast_cap }).collect();
        full_search_turns += is_full.iter().filter(|&&f| f).count() as u32;

        // Encode positions with random D6 symmetry
        let root_syms: Vec<D6Symmetry> = (0..n).map(|_| D6Symmetry::random(&mut rng)).collect();
        let mut turn_board_offsets: Vec<usize> = Vec::with_capacity(n);
        let mut turn_reserve_offsets: Vec<usize> = Vec::with_capacity(n);
        let mut flat_boards = vec![0f32; n * BOARD_FLAT];
        let mut flat_reserves = vec![0f32; n * RESERVE_SIZE];
        for (i, &gi) in mcts_games.iter().enumerate() {
            let boff = board_buf.len();
            board_buf.resize(boff + BOARD_FLAT, 0.0);
            let roff = reserve_buf.len();
            reserve_buf.resize(roff + RESERVE_SIZE, 0.0);
            encode_board(&boards[gi], &mut board_buf[boff..boff + BOARD_FLAT], &mut reserve_buf[roff..roff + RESERVE_SIZE]);
            encode_board(&boards[gi], &mut flat_boards[i * BOARD_FLAT..(i + 1) * BOARD_FLAT], &mut flat_reserves[i * RESERVE_SIZE..(i + 1) * RESERVE_SIZE]);
            apply_d6_sym_spatial(&mut flat_boards[i * BOARD_FLAT..(i + 1) * BOARD_FLAT], root_syms[i], NUM_CHANNELS, GRID_SIZE);
            turn_board_offsets.push(boff);
            turn_reserve_offsets.push(roff);
        }

        // Initial NN eval for roots
        let (mut init_place, mut init_src, mut init_dst, _) = eval_fn(&flat_boards, &flat_reserves, n)?;
        for i in 0..n {
            let inv = root_syms[i].inverse();
            apply_d6_sym_spatial(&mut init_place[i * PLACE_HEAD_SIZE..(i + 1) * PLACE_HEAD_SIZE], inv, PLACE_HEAD_SIZE / (GRID_SIZE * GRID_SIZE), GRID_SIZE);
            apply_d6_sym_spatial(&mut init_src[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE], inv, 1, GRID_SIZE);
            apply_d6_sym_spatial(&mut init_dst[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE], inv, 1, GRID_SIZE);
        }

        for (i, &gi) in mcts_games.iter().enumerate() {
            let heads = PolicyHeads { place: &init_place[i * PLACE_HEAD_SIZE..(i + 1) * PLACE_HEAD_SIZE], cap_source: &init_src[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE], cap_dest: &init_dst[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE] };
            searches[gi].init(&boards[gi], &heads);
            if is_full[i] { searches[gi].apply_root_dirichlet(dir_alpha, dir_epsilon); }
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
            let mut leaf_syms: Vec<D6Symmetry> = Vec::with_capacity(nl);
            for (k, (&leaf, &i)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
                let gi = mcts_games[i];
                let (board_enc, reserve_enc) = searches[gi].encode_leaf(leaf);
                leaf_boards_flat[k * BOARD_FLAT..(k + 1) * BOARD_FLAT].copy_from_slice(&board_enc);
                leaf_reserves_flat[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE].copy_from_slice(&reserve_enc);
                let sym = D6Symmetry::random(&mut rng);
                apply_d6_sym_spatial(&mut leaf_boards_flat[k * BOARD_FLAT..(k + 1) * BOARD_FLAT], sym, NUM_CHANNELS, GRID_SIZE);
                leaf_syms.push(sym);
            }

            let (leaf_place, leaf_src, leaf_dst, leaf_values) = eval_fn(&leaf_boards_flat, &leaf_reserves_flat, nl)?;

            let place_channels = PLACE_HEAD_SIZE / (GRID_SIZE * GRID_SIZE);
            let mut per_game_leaves: Vec<Vec<NodeId>> = vec![Vec::new(); n];
            let mut per_game_values: Vec<Vec<f32>> = (0..n).map(|_| Vec::new()).collect();
            struct LeafHeadData { place: Vec<f32>, src: Vec<f32>, dst: Vec<f32> }
            let mut per_game_head_data: Vec<Vec<LeafHeadData>> = (0..n).map(|_| Vec::new()).collect();
            for (k, (&leaf, &i)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
                let inv = leaf_syms[k].inverse();
                let mut place = leaf_place[k * PLACE_HEAD_SIZE..(k + 1) * PLACE_HEAD_SIZE].to_vec();
                let mut src = leaf_src[k * CAP_HEAD_SIZE..(k + 1) * CAP_HEAD_SIZE].to_vec();
                let mut dst = leaf_dst[k * CAP_HEAD_SIZE..(k + 1) * CAP_HEAD_SIZE].to_vec();
                apply_d6_sym_spatial(&mut place, inv, place_channels, GRID_SIZE);
                apply_d6_sym_spatial(&mut src, inv, 1, GRID_SIZE);
                apply_d6_sym_spatial(&mut dst, inv, 1, GRID_SIZE);
                per_game_leaves[i].push(leaf);
                per_game_head_data[i].push(LeafHeadData { place, src, dst });
                per_game_values[i].push(leaf_values[k]);
            }
            for (i, &gi) in mcts_games.iter().enumerate() {
                if per_game_leaves[i].is_empty() { continue; }
                let heads: Vec<PolicyHeads> = per_game_head_data[i].iter().map(|d| PolicyHeads {
                    place: &d.place,
                    cap_source: &d.src,
                    cap_dest: &d.dst,
                }).collect();
                searches[gi].expand_and_backprop(&per_game_leaves[i], &heads, &per_game_values[i]);
            }

            if game_sims.iter().zip(sim_caps.iter()).all(|(s, c)| *s >= *c) { break; }
        }

        // Select and apply moves
        for (i, &gi) in mcts_games.iter().enumerate() {
            let dist = searches[gi].get_pruned_visit_distribution();
            let mut policy_vec = vec![0.0f32; POLICY_SIZE];
            for (mv, prob) in &dist {
                policy_vec[encode_move(mv)] = *prob;
            }

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
    let mut policy_data = Vec::with_capacity(total_samples * POLICY_SIZE);
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
