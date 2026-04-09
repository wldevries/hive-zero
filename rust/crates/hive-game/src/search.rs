use rand::RngExt;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;

use core_game::game::{Game as GameTrait, Outcome, Player, PolicyIndex};
use core_game::mcts::arena::NodeId;
use core_game::mcts::search::MctsSearch;
use core_game::symmetry::{D6Symmetry, Symmetry, apply_d6_sym_spatial};

use crate::board_encoding::{self, NUM_CHANNELS, RESERVE_SIZE};
use crate::game::{Game, GameState, Move};
use crate::hex::hex_neighbors;
use crate::move_encoding::{self, encode_game_move};
use crate::piece::{Piece, PieceColor, PieceType};

pub type EvalFn<'a> = Box<dyn FnMut(&[f32], &[f32], usize) -> Result<(Vec<f32>, Vec<f32>), String> + 'a>;
pub type SelfPlayProgressFn<'a> = Box<dyn FnMut(u32, u32, u32, u32, u32, u32) + 'a>;
pub type BattleProgressFn<'a> = Box<dyn FnMut(u32, u32, u32, u32) + 'a>;

#[derive(Clone)]
pub struct BattleResult {
    pub wins_model1: u32,
    pub wins_model2: u32,
    pub draws: u32,
    pub game_lengths: Vec<u32>,
}

#[derive(Clone)]
pub struct SelfPlayResult {
    pub grid_size: usize,
    pub board_data: Vec<f32>,
    pub reserve_data: Vec<f32>,
    pub policy_data: Vec<f32>,
    pub value_targets: Vec<f32>,
    pub value_only_flags: Vec<bool>,
    pub policy_only_flags: Vec<bool>,
    pub my_queen_danger: Vec<f32>,
    pub opp_queen_danger: Vec<f32>,
    pub my_queen_escape: Vec<f32>,
    pub opp_queen_escape: Vec<f32>,
    pub my_mobility: Vec<f32>,
    pub opp_mobility: Vec<f32>,
    pub num_samples: usize,
    pub wins_w: u32,
    pub wins_b: u32,
    pub draws: u32,
    pub resignations: u32,
    pub total_moves: u32,
    pub full_search_turns: u32,
    pub total_turns: u32,
    pub calibration_total: u32,
    pub calibration_would_resign: u32,
    pub calibration_false_positives: u32,
    pub use_playout_cap: bool,
    pub final_games: Vec<Game>,
}

struct TurnRecord {
    board_offset: usize,
    reserve_offset: usize,
    turn_color: PieceColor,
    is_value_only: bool,
    policy_vector: Vec<f32>,
    my_queen_danger: f32,
    opp_queen_danger: f32,
    my_queen_escape: f32,
    opp_queen_escape: f32,
    my_mobility: f32,
    opp_mobility: f32,
}

fn opposite_color(color: PieceColor) -> PieceColor {
    match color {
        PieceColor::White => PieceColor::Black,
        PieceColor::Black => PieceColor::White,
    }
}

fn queen_danger(game: &Game, color: PieceColor) -> f32 {
    let queen = Piece::new(color, PieceType::Queen, 1);
    match game.board.piece_position(queen) {
        None => 0.0,
        Some(pos) => {
            let neighbors = hex_neighbors(pos)
                .iter()
                .filter(|&&neighbor| game.board.is_occupied(neighbor))
                .count() as f32;
            (neighbors / 6.0).min(1.0)
        }
    }
}

fn queen_escape(game: &Game, color: PieceColor) -> f32 {
    let queen = Piece::new(color, PieceType::Queen, 1);
    match game.board.piece_position(queen) {
        None => 0.0,
        Some(pos) => {
            if game.board.top_piece(pos) != Some(queen) {
                return 0.0;
            }
            if game.board.stack_height(pos) == 1 {
                let articulation_points = game.board.articulation_points();
                if articulation_points.contains(&pos) {
                    return 0.0;
                }
            }

            let mut count = 0u32;
            for &neighbor in hex_neighbors(pos).iter() {
                if !game.board.is_occupied(neighbor) && game.board.can_slide(pos, neighbor) {
                    if hex_neighbors(neighbor)
                        .iter()
                        .any(|&adjacent| adjacent != pos && game.board.is_occupied(adjacent))
                    {
                        count += 1;
                    }
                }
            }
            count as f32 / 6.0
        }
    }
}

fn piece_mobility(game: &mut Game, color: PieceColor) -> f32 {
    let on_board = game.board.pieces_on_board(color);
    if on_board.is_empty() {
        return 0.0;
    }
    if !game.queen_placed(color) {
        return 0.0;
    }

    let articulation_points = game.board.articulation_points();
    let mut mobile = 0u32;
    for &piece in &on_board {
        let moves = crate::rules::get_moves(piece, &mut game.board, &articulation_points);
        if !moves.is_empty() {
            mobile += 1;
        }
    }
    mobile as f32 / on_board.len() as f32
}

pub fn best_move_core(
    game: &Game,
    simulations: usize,
    c_puct: f32,
    mut eval_fn: EvalFn<'_>,
) -> Result<Move, String> {
    if game.is_game_over() {
        return Err("Game is already over".to_string());
    }

    let mut root_game = game.clone();
    if root_game.valid_moves().is_empty() {
        return Ok(Move::pass());
    }

    let grid_size = game.nn_grid_size;
    let board_size = NUM_CHANNELS * grid_size * grid_size;
    let policy_size = move_encoding::policy_size(grid_size);
    let batch_size = 8usize;

    let mut search = MctsSearch::<Game>::new(simulations + 64);
    search.c_puct = c_puct;

    let mut board_buf = vec![0.0f32; board_size];
    let mut reserve_buf = vec![0.0f32; RESERVE_SIZE];
    board_encoding::encode_board(game, &mut board_buf, &mut reserve_buf, grid_size);
    let (root_policy, _root_value) = eval_fn(&board_buf, &reserve_buf, 1)?;

    search.init(game, &root_policy);
    search.apply_root_dirichlet(0.3, 0.25);

    let mut done = 0usize;
    while done < simulations {
        let leaf_ids = search.select_leaves(batch_size.min(simulations - done));
        if leaf_ids.is_empty() {
            break;
        }
        let num_leaves = leaf_ids.len();

        let mut boards = vec![0.0f32; num_leaves * board_size];
        let mut reserves = vec![0.0f32; num_leaves * RESERVE_SIZE];
        for (index, &leaf_id) in leaf_ids.iter().enumerate() {
            let (board, reserve) = search.encode_leaf(leaf_id);
            boards[index * board_size..(index + 1) * board_size].copy_from_slice(&board);
            reserves[index * RESERVE_SIZE..(index + 1) * RESERVE_SIZE].copy_from_slice(&reserve);
        }

        let (leaf_policy_data, leaf_values) = eval_fn(&boards, &reserves, num_leaves)?;
        let policies: Vec<Vec<f32>> = (0..num_leaves)
            .map(|index| leaf_policy_data[index * policy_size..(index + 1) * policy_size].to_vec())
            .collect();
        search.expand_and_backprop(&policies, &leaf_values);
        done += num_leaves;
    }

    let best = search
        .get_pruned_visit_distribution()
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(mv, _)| mv)
        .unwrap_or_else(Move::pass);
    Ok(best)
}

pub fn play_selfplay_core(
    num_games: usize,
    simulations: usize,
    max_moves: u32,
    temperature: f32,
    temp_threshold: u32,
    playout_cap_p: f32,
    fast_cap: usize,
    c_puct: f32,
    dir_alpha: f32,
    dir_epsilon: f32,
    leaf_batch_size: usize,
    resign_threshold: Option<f32>,
    resign_moves: u32,
    resign_min_moves: u32,
    calibration_frac: f32,
    random_opening_moves_min: u32,
    random_opening_moves_max: u32,
    skip_timeout_games: bool,
    grid_size: usize,
    mut eval_fn: EvalFn<'_>,
    mut progress_fn: Option<SelfPlayProgressFn<'_>>,
    opening_sequences: Vec<Vec<String>>,
) -> Result<SelfPlayResult, String> {
    let use_playout_cap = playout_cap_p > 0.0;
    let board_size = NUM_CHANNELS * grid_size * grid_size;
    let policy_size = move_encoding::policy_size(grid_size);

    let mut games: Vec<Game> = (0..num_games)
        .map(|_| Game::new_with_grid_size(grid_size))
        .collect();
    let mut searches: Vec<MctsSearch<Game>> = (0..num_games)
        .map(|_| {
            let mut search = MctsSearch::<Game>::new(100_000);
            search.c_puct = c_puct;
            search.use_forced_playouts = true;
            search
        })
        .collect();
    let mut move_counts = vec![0u32; num_games];
    let mut active = vec![true; num_games];
    let mut finished_count = 0u32;

    let mut histories: Vec<Vec<TurnRecord>> = (0..num_games).map(|_| Vec::new()).collect();
    let mut board_buf: Vec<f32> = Vec::new();
    let mut reserve_buf: Vec<f32> = Vec::new();

    let game_random_opening_moves: Vec<u32> = {
        let mut rng = rand::rng();
        (0..num_games)
            .map(|_| {
                if random_opening_moves_max > random_opening_moves_min {
                    rng.random_range(random_opening_moves_min..=random_opening_moves_max)
                } else {
                    random_opening_moves_min
                }
            })
            .collect()
    };

    let mut opening_done = vec![false; num_games];
    let opening_syms: Vec<D6Symmetry> = {
        let mut rng = rand::rng();
        (0..num_games)
            .map(|_| D6Symmetry::random(&mut rng))
            .collect()
    };

    let mut resign_counters = vec![0u32; num_games];
    let mut resigned_as: Vec<Option<PieceColor>> = vec![None; num_games];

    let mut calibration = vec![false; num_games];
    let mut calibration_would_resign: Vec<Option<PieceColor>> = vec![None; num_games];
    if resign_threshold.is_some() {
        let num_calibration = (num_games as f32 * calibration_frac).ceil().max(1.0) as usize;
        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..num_games).collect();
        for index in 0..num_calibration.min(num_games) {
            let swap_index = rng.random_range(index..num_games);
            indices.swap(index, swap_index);
            calibration[indices[index]] = true;
        }
    }

    let mut full_search_turns = 0u32;
    let mut total_turns = 0u32;

    while active.iter().any(|&is_active| is_active) {
        let mut rng = rand::rng();
        let mut mcts_games: Vec<usize> = Vec::new();

        for game_index in 0..num_games {
            if !active[game_index] {
                continue;
            }

            let opening_sequence = opening_sequences.get(game_index).filter(|sequence| !sequence.is_empty());
            if let Some(sequence) = opening_sequence {
                if !opening_done[game_index] && (move_counts[game_index] as usize) < sequence.len() {
                    let move_str = &sequence[move_counts[game_index] as usize];
                    let transformed = crate::uhp::transform_uhp_move(move_str, opening_syms[game_index]);
                    let valid_moves = games[game_index].valid_moves();
                    if let Some(mv) = valid_moves
                        .iter()
                        .find(|mv| crate::uhp::format_move_uhp(&games[game_index], mv) == transformed)
                    {
                        games[game_index]
                            .play_move(mv)
                            .map_err(|e| e.to_string())?;
                        move_counts[game_index] += 1;
                        if games[game_index].is_game_over() || move_counts[game_index] >= max_moves {
                            active[game_index] = false;
                            finished_count += 1;
                        }
                    } else {
                        opening_done[game_index] = true;
                    }
                    continue;
                }
            } else if move_counts[game_index] < game_random_opening_moves[game_index] {
                let valid_moves = games[game_index].valid_moves();
                if valid_moves.is_empty() {
                    games[game_index].play_pass();
                } else {
                    let move_index = rng.random_range(0..valid_moves.len());
                    games[game_index]
                        .play_move(&valid_moves[move_index])
                        .map_err(|e| e.to_string())?;
                }
                move_counts[game_index] += 1;
                if games[game_index].is_game_over() || move_counts[game_index] >= max_moves {
                    active[game_index] = false;
                    finished_count += 1;
                }
                continue;
            }

            if games[game_index].valid_moves().is_empty() {
                games[game_index].play_pass();
                move_counts[game_index] += 1;
                if games[game_index].is_game_over() || move_counts[game_index] >= max_moves {
                    active[game_index] = false;
                    finished_count += 1;
                }
            } else {
                mcts_games.push(game_index);
            }
        }

        if mcts_games.is_empty() {
            continue;
        }

        let num_search_games = mcts_games.len();
        let is_full: Vec<bool> = if use_playout_cap {
            (0..num_search_games)
                .map(|_| rng.random::<f32>() < playout_cap_p)
                .collect()
        } else {
            vec![true; num_search_games]
        };

        let mut turn_board_offsets = Vec::with_capacity(num_search_games);
        let mut turn_reserve_offsets = Vec::with_capacity(num_search_games);
        for &game_index in &mcts_games {
            let board_offset = board_buf.len();
            let reserve_offset = reserve_buf.len();
            board_buf.resize(board_offset + board_size, 0.0);
            reserve_buf.resize(reserve_offset + RESERVE_SIZE, 0.0);
            board_encoding::encode_board(
                &games[game_index],
                &mut board_buf[board_offset..board_offset + board_size],
                &mut reserve_buf[reserve_offset..reserve_offset + RESERVE_SIZE],
                grid_size,
            );
            turn_board_offsets.push(board_offset);
            turn_reserve_offsets.push(reserve_offset);
        }

        let root_syms: Vec<D6Symmetry> = (0..num_search_games)
            .map(|_| D6Symmetry::random(&mut rng))
            .collect();
        let mut flat_boards = Vec::with_capacity(num_search_games * board_size);
        let mut flat_reserves = Vec::with_capacity(num_search_games * RESERVE_SIZE);
        for index in 0..num_search_games {
            flat_boards.extend_from_slice(
                &board_buf[turn_board_offsets[index]..turn_board_offsets[index] + board_size],
            );
            flat_reserves.extend_from_slice(
                &reserve_buf[turn_reserve_offsets[index]..turn_reserve_offsets[index] + RESERVE_SIZE],
            );
            apply_d6_sym_spatial(
                &mut flat_boards[index * board_size..(index + 1) * board_size],
                root_syms[index],
                NUM_CHANNELS,
                grid_size,
            );
        }
        let (mut init_policies, init_values) = eval_fn(&flat_boards, &flat_reserves, num_search_games)?;
        for index in 0..num_search_games {
            apply_d6_sym_spatial(
                &mut init_policies[index * policy_size..(index + 1) * policy_size],
                root_syms[index].inverse(),
                move_encoding::NUM_POLICY_CHANNELS,
                grid_size,
            );
        }

        for (index, &game_index) in mcts_games.iter().enumerate() {
            let search = &mut searches[game_index];
            search.c_puct = c_puct;
            search.use_forced_playouts = true;
            let policy = &init_policies[index * policy_size..(index + 1) * policy_size];
            search.init(&games[game_index], policy);
            if is_full[index] {
                search.apply_root_dirichlet(dir_alpha, dir_epsilon);
            }
        }

        let per_game_caps: Vec<usize> = is_full
            .iter()
            .map(|&full| if full { simulations } else { fast_cap })
            .collect();
        let child_counts: Vec<u16> = mcts_games
            .iter()
            .map(|&game_index| searches[game_index].root_child_count())
            .collect();

        let mut searching: Vec<usize> = Vec::new();
        let mut sim_caps: Vec<usize> = Vec::new();
        for (index, &child_count) in child_counts.iter().enumerate() {
            if child_count > 0 {
                searching.push(index);
                sim_caps.push(per_game_caps[index]);
            } else {
                let game_index = mcts_games[index];
                games[game_index].play_pass();
                move_counts[game_index] += 1;
                if games[game_index].is_game_over() || move_counts[game_index] >= max_moves {
                    active[game_index] = false;
                    finished_count += 1;
                }
            }
        }

        if !searching.is_empty() {
            let mut sims_done = vec![0usize; searching.len()];
            let mut still_searching: Vec<usize> = (0..searching.len()).collect();
            let mut per_game_leaf_ids: Vec<Vec<NodeId>> = vec![Vec::new(); searching.len()];
            let mut per_game_syms: Vec<Vec<D6Symmetry>> = vec![Vec::new(); searching.len()];

            while !still_searching.is_empty() {
                for index in 0..searching.len() {
                    per_game_leaf_ids[index].clear();
                    per_game_syms[index].clear();
                }

                let mut flat_boards: Vec<f32> = Vec::new();
                let mut flat_reserves: Vec<f32> = Vec::new();
                let mut any_leaves = false;

                for _ in 0..leaf_batch_size {
                    let mut any_this_round = false;
                    for &search_index in &still_searching {
                        let game_index = mcts_games[searching[search_index]];
                        let leaf_ids = searches[game_index].select_leaves(1);
                        sims_done[search_index] += leaf_ids.len().max(1);
                        for &leaf_id in &leaf_ids {
                            let (board, reserve) = searches[game_index].encode_leaf(leaf_id);
                            let sym = D6Symmetry::random(&mut rng);
                            let start = flat_boards.len();
                            flat_boards.extend_from_slice(&board);
                            apply_d6_sym_spatial(
                                &mut flat_boards[start..],
                                sym,
                                NUM_CHANNELS,
                                grid_size,
                            );
                            flat_reserves.extend_from_slice(&reserve);
                            per_game_leaf_ids[search_index].push(leaf_id);
                            per_game_syms[search_index].push(sym);
                            any_this_round = true;
                            any_leaves = true;
                        }
                    }
                    still_searching.retain(|&search_index| sims_done[search_index] < sim_caps[search_index]);
                    if !any_this_round {
                        break;
                    }
                }

                if !any_leaves {
                    break;
                }

                let total_leaves = flat_boards.len() / board_size;
                let (policy_data, value_data) = eval_fn(&flat_boards, &flat_reserves, total_leaves)?;

                let mut offset = 0usize;
                for search_index in 0..searching.len() {
                    let num_leaves = per_game_leaf_ids[search_index].len();
                    if num_leaves == 0 {
                        continue;
                    }
                    let game_index = mcts_games[searching[search_index]];
                    let policies: Vec<Vec<f32>> = (0..num_leaves)
                        .map(|leaf_offset| {
                            let mut policy = policy_data
                                [(offset + leaf_offset) * policy_size..(offset + leaf_offset + 1) * policy_size]
                                .to_vec();
                            apply_d6_sym_spatial(
                                &mut policy,
                                per_game_syms[search_index][leaf_offset].inverse(),
                                move_encoding::NUM_POLICY_CHANNELS,
                                grid_size,
                            );
                            policy
                        })
                        .collect();
                    let values = value_data[offset..offset + num_leaves].to_vec();
                    searches[game_index].expand_and_backprop(&policies, &values);
                    offset += num_leaves;
                }
            }
        }

        for (index, &game_index) in mcts_games.iter().enumerate() {
            if child_counts[index] == 0 {
                continue;
            }

            if let Some(threshold) = resign_threshold {
                let value = init_values[index];
                if value < threshold && move_counts[game_index] >= resign_min_moves {
                    resign_counters[game_index] += 1;
                } else {
                    resign_counters[game_index] = 0;
                }
                if resign_counters[game_index] >= resign_moves {
                    let color = games[game_index].turn_color;
                    if calibration[game_index] {
                        if calibration_would_resign[game_index].is_none() {
                            calibration_would_resign[game_index] = Some(color);
                        }
                    } else {
                        resigned_as[game_index] = Some(color);
                        active[game_index] = false;
                        finished_count += 1;
                        continue;
                    }
                }
            }

            let search = &searches[game_index];
            let dist = if search.use_forced_playouts {
                search.get_pruned_visit_distribution()
            } else {
                search.get_visit_distribution()
            };

            if dist.is_empty() {
                games[game_index].play_pass();
                move_counts[game_index] += 1;
                if games[game_index].is_game_over() || move_counts[game_index] >= max_moves {
                    active[game_index] = false;
                    finished_count += 1;
                }
                continue;
            }

            let move_num = move_counts[game_index];
            let temp = if move_num < temp_threshold { temperature } else { 0.0 };
            let mut probs: Vec<f32> = dist.iter().map(|(_, prob)| *prob).collect();

            if temp == 0.0 || !is_full[index] {
                let best_index = probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(best_index, _)| best_index)
                    .unwrap();
                for prob in &mut probs {
                    *prob = 0.0;
                }
                probs[best_index] = 1.0;
            } else {
                for prob in &mut probs {
                    *prob = prob.powf(1.0 / temp);
                }
                let total_prob: f32 = probs.iter().sum();
                if total_prob > 0.0 {
                    for prob in &mut probs {
                        *prob /= total_prob;
                    }
                } else {
                    let uniform = 1.0 / probs.len() as f32;
                    for prob in &mut probs {
                        *prob = uniform;
                    }
                }
            }

            let mut policy_vector = vec![0.0f32; policy_size];
            for (move_index, (mv, _)) in dist.iter().enumerate() {
                if mv.piece.is_some() {
                    match encode_game_move(mv, grid_size) {
                        Some(PolicyIndex::Single(index)) => {
                            if index < policy_size {
                                policy_vector[index] = probs[move_index];
                            }
                        }
                        Some(PolicyIndex::Sum(a, b)) => {
                            if a < policy_size {
                                policy_vector[a] += probs[move_index];
                            }
                            if b < policy_size {
                                policy_vector[b] += probs[move_index];
                            }
                        }
                        None => {}
                    }
                }
            }

            let turn_color = games[game_index].turn_color;
            let opp_color = opposite_color(turn_color);
            let is_value_only = !is_full[index];
            if is_full[index] {
                full_search_turns += 1;
            }
            total_turns += 1;

            histories[game_index].push(TurnRecord {
                board_offset: turn_board_offsets[index],
                reserve_offset: turn_reserve_offsets[index],
                turn_color,
                is_value_only,
                policy_vector,
                my_queen_danger: queen_danger(&games[game_index], turn_color),
                opp_queen_danger: queen_danger(&games[game_index], opp_color),
                my_queen_escape: queen_escape(&games[game_index], turn_color),
                opp_queen_escape: queen_escape(&games[game_index], opp_color),
                my_mobility: piece_mobility(&mut games[game_index], turn_color),
                opp_mobility: piece_mobility(&mut games[game_index], opp_color),
            });

            let move_index = if !is_full[index] || temp == 0.0 {
                probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(best_index, _)| best_index)
                    .unwrap()
            } else {
                let weighted = WeightedIndex::new(&probs).map_err(|e| e.to_string())?;
                weighted.sample(&mut rng)
            };

            let (mv, _) = &dist[move_index];
            if mv.is_pass() {
                games[game_index].play_pass();
            } else {
                games[game_index]
                    .play_move(mv)
                    .map_err(|e| e.to_string())?;
            }

            move_counts[game_index] += 1;
            if games[game_index].is_game_over() || move_counts[game_index] >= max_moves {
                active[game_index] = false;
                finished_count += 1;
            }
        }

        if let Some(progress) = progress_fn.as_mut() {
            let total_moves: u32 = move_counts.iter().sum();
            let num_resigned = resigned_as.iter().filter(|entry| entry.is_some()).count() as u32;
            let num_active = active.iter().filter(|&&entry| entry).count() as u32;
            let max_turn = if num_active > 0 {
                move_counts
                    .iter()
                    .zip(active.iter())
                    .filter(|(_, is_active)| **is_active)
                    .map(|(&move_count, _)| move_count)
                    .max()
                    .unwrap_or(0)
            } else {
                move_counts.iter().copied().max().unwrap_or(0)
            };
            progress(
                finished_count,
                num_games as u32,
                num_active,
                total_moves,
                num_resigned,
                max_turn,
            );
        }
    }

    let mut result_board_data = Vec::new();
    let mut result_reserve_data = Vec::new();
    let mut result_policy_data = Vec::new();
    let mut result_value_targets = Vec::new();
    let mut result_value_only = Vec::new();
    let mut result_policy_only = Vec::new();
    let mut result_my_queen_danger = Vec::new();
    let mut result_opp_queen_danger = Vec::new();
    let mut result_my_queen_escape = Vec::new();
    let mut result_opp_queen_escape = Vec::new();
    let mut result_my_mobility = Vec::new();
    let mut result_opp_mobility = Vec::new();
    let mut num_samples = 0usize;

    let mut wins_w = 0u32;
    let mut wins_b = 0u32;
    let mut draws = 0u32;
    let mut resignations = 0u32;

    for game_index in 0..num_games {
        let (outcome_w, outcome_b, decisive) = if let Some(color) = resigned_as[game_index] {
            resignations += 1;
            match color {
                PieceColor::White => {
                    wins_b += 1;
                    (-1.0f32, 1.0f32, true)
                }
                PieceColor::Black => {
                    wins_w += 1;
                    (1.0f32, -1.0f32, true)
                }
            }
        } else {
            match games[game_index].state {
                GameState::WhiteWins => {
                    wins_w += 1;
                    (1.0f32, -1.0f32, true)
                }
                GameState::BlackWins => {
                    wins_b += 1;
                    (-1.0f32, 1.0f32, true)
                }
                _ => {
                    draws += 1;
                    let (white_score, black_score) = games[game_index].heuristic_value();
                    (white_score, black_score, false)
                }
            }
        };

        if !decisive && skip_timeout_games {
            continue;
        }

        let policy_only = !decisive && outcome_w == 0.0;
        for record in &histories[game_index] {
            let board = &board_buf[record.board_offset..record.board_offset + board_size];
            let reserve = &reserve_buf[record.reserve_offset..record.reserve_offset + RESERVE_SIZE];
            let value = match record.turn_color {
                PieceColor::White => outcome_w,
                PieceColor::Black => outcome_b,
            };

            result_board_data.extend_from_slice(board);
            result_reserve_data.extend_from_slice(reserve);
            result_policy_data.extend_from_slice(&record.policy_vector);
            result_value_targets.push(value);
            result_value_only.push(record.is_value_only);
            result_policy_only.push(policy_only);
            result_my_queen_danger.push(record.my_queen_danger);
            result_opp_queen_danger.push(record.opp_queen_danger);
            result_my_queen_escape.push(record.my_queen_escape);
            result_opp_queen_escape.push(record.opp_queen_escape);
            result_my_mobility.push(record.my_mobility);
            result_opp_mobility.push(record.opp_mobility);
            num_samples += 1;
        }
    }

    let calibration_total = calibration.iter().filter(|&&entry| entry).count() as u32;
    let calibration_would_resign_count = calibration_would_resign
        .iter()
        .filter(|entry| entry.is_some())
        .count() as u32;
    let mut calibration_false_positives = 0u32;
    for game_index in 0..num_games {
        if let Some(resign_color) = calibration_would_resign[game_index] {
            let won = match resign_color {
                PieceColor::White => games[game_index].state == GameState::WhiteWins,
                PieceColor::Black => games[game_index].state == GameState::BlackWins,
            };
            if won {
                calibration_false_positives += 1;
            }
        }
    }

    Ok(SelfPlayResult {
        grid_size,
        board_data: result_board_data,
        reserve_data: result_reserve_data,
        policy_data: result_policy_data,
        value_targets: result_value_targets,
        value_only_flags: result_value_only,
        policy_only_flags: result_policy_only,
        my_queen_danger: result_my_queen_danger,
        opp_queen_danger: result_opp_queen_danger,
        my_queen_escape: result_my_queen_escape,
        opp_queen_escape: result_opp_queen_escape,
        my_mobility: result_my_mobility,
        opp_mobility: result_opp_mobility,
        num_samples,
        wins_w,
        wins_b,
        draws,
        resignations,
        total_moves: move_counts.iter().sum(),
        full_search_turns,
        total_turns,
        calibration_total,
        calibration_would_resign: calibration_would_resign_count,
        calibration_false_positives,
        use_playout_cap,
        final_games: games,
    })
}

pub fn play_battle_core(
    num_games: usize,
    simulations: usize,
    max_moves: u32,
    c_puct: f32,
    leaf_batch_size: usize,
    grid_size: usize,
    mut eval_fn1: EvalFn<'_>,
    mut eval_fn2: EvalFn<'_>,
    mut progress_fn: Option<BattleProgressFn<'_>>,
) -> Result<BattleResult, String> {
    let half = num_games / 2;
    let board_size = NUM_CHANNELS * grid_size * grid_size;
    let policy_size = move_encoding::policy_size(grid_size);

    let mut games: Vec<Game> = (0..num_games)
        .map(|_| Game::new_with_grid_size(grid_size))
        .collect();
    let mut searches: Vec<MctsSearch<Game>> = (0..num_games)
        .map(|_| {
            let mut search = MctsSearch::<Game>::new(simulations + 64);
            search.c_puct = c_puct;
            search
        })
        .collect();
    let mut move_counts = vec![0u32; num_games];
    let mut active = vec![true; num_games];
    let mut finished_count = 0u32;
    let mut total_moves = 0u32;
    let mut wins_model1 = 0u32;
    let mut wins_model2 = 0u32;
    let mut draws = 0u32;
    let mut game_lengths = Vec::new();

    let use_fn1 = |game_index: usize, player: Player| (game_index < half) == (player == Player::Player1);

    let mut rng = rand::rng();
    while active.iter().any(|&is_active| is_active) {
        let active_games: Vec<usize> = (0..num_games).filter(|&game_index| active[game_index]).collect();
        let mut mcts_games = Vec::new();
        for game_index in active_games {
            if games[game_index].valid_moves().is_empty() {
                games[game_index].play_pass();
                move_counts[game_index] += 1;
                total_moves += 1;
                if games[game_index].is_game_over() || move_counts[game_index] >= max_moves {
                    active[game_index] = false;
                    finished_count += 1;
                    game_lengths.push(move_counts[game_index]);
                    match games[game_index].outcome() {
                        Outcome::WonBy(winner) => {
                            if use_fn1(game_index, winner) {
                                wins_model1 += 1;
                            } else {
                                wins_model2 += 1;
                            }
                        }
                        _ => draws += 1,
                    }
                }
            } else {
                mcts_games.push(game_index);
            }
        }
        if mcts_games.is_empty() {
            continue;
        }

        let num_search_games = mcts_games.len();
        let root_syms: Vec<D6Symmetry> = (0..num_search_games)
            .map(|_| D6Symmetry::random(&mut rng))
            .collect();
        let mut flat_boards = vec![0.0f32; num_search_games * board_size];
        let mut flat_reserves = vec![0.0f32; num_search_games * RESERVE_SIZE];
        let mut fn1_flags = Vec::with_capacity(num_search_games);
        for (index, &game_index) in mcts_games.iter().enumerate() {
            board_encoding::encode_board(
                &games[game_index],
                &mut flat_boards[index * board_size..(index + 1) * board_size],
                &mut flat_reserves[index * RESERVE_SIZE..(index + 1) * RESERVE_SIZE],
                grid_size,
            );
            apply_d6_sym_spatial(
                &mut flat_boards[index * board_size..(index + 1) * board_size],
                root_syms[index],
                NUM_CHANNELS,
                grid_size,
            );
            fn1_flags.push(use_fn1(game_index, games[game_index].next_player()));
        }

        let (policy1, _value1) = eval_fn1(&flat_boards, &flat_reserves, num_search_games)?;
        let (policy2, _value2) = eval_fn2(&flat_boards, &flat_reserves, num_search_games)?;

        let mut init_policies = vec![0.0f32; num_search_games * policy_size];
        for index in 0..num_search_games {
            let source = if fn1_flags[index] { &policy1 } else { &policy2 };
            init_policies[index * policy_size..(index + 1) * policy_size].copy_from_slice(
                &source[index * policy_size..(index + 1) * policy_size],
            );
            apply_d6_sym_spatial(
                &mut init_policies[index * policy_size..(index + 1) * policy_size],
                root_syms[index].inverse(),
                move_encoding::NUM_POLICY_CHANNELS,
                grid_size,
            );
        }

        for (index, &game_index) in mcts_games.iter().enumerate() {
            searches[game_index].init(
                &games[game_index],
                &init_policies[index * policy_size..(index + 1) * policy_size],
            );
        }

        let mut game_sims = vec![0usize; num_search_games];
        loop {
            let mut leaf_ids = Vec::new();
            let mut leaf_game_idx = Vec::new();
            for _ in 0..leaf_batch_size {
                let mut any = false;
                for (index, &game_index) in mcts_games.iter().enumerate() {
                    if game_sims[index] >= simulations {
                        continue;
                    }
                    let leaves = searches[game_index].select_leaves(1);
                    let count = leaves.len();
                    if count > 0 {
                        any = true;
                    }
                    for leaf in leaves {
                        leaf_ids.push(leaf);
                        leaf_game_idx.push(index);
                    }
                    game_sims[index] += count;
                }
                if !any {
                    break;
                }
            }

            if leaf_ids.is_empty() {
                break;
            }

            let num_leaves = leaf_ids.len();
            let mut leaf_boards = vec![0.0f32; num_leaves * board_size];
            let mut leaf_reserves = vec![0.0f32; num_leaves * RESERVE_SIZE];
            let mut leaf_syms = Vec::with_capacity(num_leaves);
            let mut leaf_fn1_flags = Vec::with_capacity(num_leaves);

            for (index, (&leaf_id, &search_index)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
                let game_index = mcts_games[search_index];
                let (board_enc, reserve_enc) = searches[game_index].encode_leaf(leaf_id);
                leaf_boards[index * board_size..(index + 1) * board_size].copy_from_slice(&board_enc);
                leaf_reserves[index * RESERVE_SIZE..(index + 1) * RESERVE_SIZE].copy_from_slice(&reserve_enc);
                let sym = D6Symmetry::random(&mut rng);
                apply_d6_sym_spatial(
                    &mut leaf_boards[index * board_size..(index + 1) * board_size],
                    sym,
                    NUM_CHANNELS,
                    grid_size,
                );
                leaf_syms.push(sym);
                let leaf_player = searches[game_index].get_leaf_player(leaf_id);
                leaf_fn1_flags.push(use_fn1(game_index, leaf_player));
            }

            let (leaf_policy1, leaf_value1) = eval_fn1(&leaf_boards, &leaf_reserves, num_leaves)?;
            let (leaf_policy2, leaf_value2) = eval_fn2(&leaf_boards, &leaf_reserves, num_leaves)?;

            let mut per_game_leaves: Vec<Vec<NodeId>> = vec![Vec::new(); num_search_games];
            let mut per_game_data: Vec<Vec<(Vec<f32>, f32)>> = (0..num_search_games).map(|_| Vec::new()).collect();

            for (index, (&leaf_id, &search_index)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
                let use_first_model = leaf_fn1_flags[index];
                let policy_source = if use_first_model { &leaf_policy1 } else { &leaf_policy2 };
                let value = if use_first_model { leaf_value1[index] } else { leaf_value2[index] };
                let mut policy = policy_source[index * policy_size..(index + 1) * policy_size].to_vec();
                apply_d6_sym_spatial(
                    &mut policy,
                    leaf_syms[index].inverse(),
                    move_encoding::NUM_POLICY_CHANNELS,
                    grid_size,
                );
                per_game_leaves[search_index].push(leaf_id);
                per_game_data[search_index].push((policy, value));
            }

            for (search_index, &game_index) in mcts_games.iter().enumerate() {
                if per_game_leaves[search_index].is_empty() {
                    continue;
                }
                let policies: Vec<Vec<f32>> = per_game_data[search_index]
                    .iter()
                    .map(|(policy, _)| policy.clone())
                    .collect();
                let values: Vec<f32> = per_game_data[search_index]
                    .iter()
                    .map(|(_, value)| *value)
                    .collect();
                searches[game_index].expand_and_backprop(&policies, &values);
            }

            if game_sims.iter().all(|&sims| sims >= simulations) {
                break;
            }
        }

        for &game_index in &mcts_games {
            let dist = searches[game_index].get_pruned_visit_distribution();
            let mv = if dist.is_empty() {
                Move::pass()
            } else {
                dist.iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(mv, _)| *mv)
                    .unwrap()
            };

            if mv.is_pass() {
                games[game_index].play_pass();
            } else {
                games[game_index]
                    .play_move(&mv)
                    .map_err(|e| e.to_string())?;
            }

            move_counts[game_index] += 1;
            total_moves += 1;
            if games[game_index].is_game_over() || move_counts[game_index] >= max_moves {
                active[game_index] = false;
                finished_count += 1;
                game_lengths.push(move_counts[game_index]);
                match games[game_index].outcome() {
                    Outcome::WonBy(winner) => {
                        if use_fn1(game_index, winner) {
                            wins_model1 += 1;
                        } else {
                            wins_model2 += 1;
                        }
                    }
                    _ => draws += 1,
                }
            }
        }

        if let Some(progress) = progress_fn.as_mut() {
            let active_count = active.iter().filter(|&&entry| entry).count() as u32;
            progress(finished_count, num_games as u32, active_count, total_moves);
        }
    }

    Ok(BattleResult {
        wins_model1,
        wins_model2,
        draws,
        game_lengths,
    })
}