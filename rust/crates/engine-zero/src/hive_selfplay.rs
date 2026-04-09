
/// Self-play game loop in Rust.
/// Plays all games to completion, only calling back to Python for GPU NN inference.
/// Returns training data as contiguous arrays.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use hive_game::board_encoding::{self, NUM_CHANNELS, RESERVE_SIZE, f32_to_bf16};
use crate::inference::HiveInference;
use hive_game::game::{Game, GameState};
use hive_game::hex::hex_neighbors;
use core_game::game::{Game as GameTrait, PolicyIndex, Outcome, Player};
use core_game::mcts::search::MctsSearch;
use core_game::mcts::arena::NodeId;
use hive_game::move_encoding::{self, encode_game_move};
use hive_game::piece::{Piece, PieceColor, PieceType};
use core_game::symmetry::{Symmetry, D6Symmetry, apply_d6_sym_spatial};

use rand::RngExt;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;


/// Wraps either a Python eval callback or a native engine (ORT, tract, …).
enum InferenceBackend<'py> {
    Python {
        py: Python<'py>,
        eval_fn: &'py Bound<'py, PyAny>,
    },
    Native {
        engine: Box<dyn HiveInference>,
    },
}

impl HiveInference for InferenceBackend<'_> {
    fn infer_batch(
        &mut self,
        boards: &[f32],
        reserves: &[f32],
        batch_size: usize,
        num_channels: usize,
        grid_size: usize,
        reserve_size: usize,
    ) -> Result<crate::inference::HiveInferenceResult, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            InferenceBackend::Python { py, eval_fn } => {
                let board_flat = boards.len() / batch_size;
                let boards_bf16: Vec<u16> = boards.iter().map(|&x| f32_to_bf16(x)).collect();
                let reserves_bf16: Vec<u16> = reserves.iter().map(|&x| f32_to_bf16(x)).collect();

                let board_arr = numpy::ndarray::Array2::from_shape_vec(
                    (batch_size, board_flat), boards_bf16,
                ).unwrap();
                let board_np = PyArray2::from_owned_array_bound(*py, board_arr);
                let board_4d = board_np.reshape([batch_size, num_channels, grid_size, grid_size])
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
                let reserve_arr = numpy::ndarray::Array2::from_shape_vec(
                    (batch_size, reserve_size), reserves_bf16,
                ).unwrap();
                let reserve_np = PyArray2::from_owned_array_bound(*py, reserve_arr);

                let result = eval_fn.call1((board_4d, reserve_np))?;
                let tuple = result.downcast::<PyTuple>()
                    .map_err(|_| "eval_fn must return (policy, value) tuple")?;
                let policy_arr: PyReadonlyArray2<f32> = tuple.get_item(0)?.extract()?;
                let value_arr: PyReadonlyArray1<f32> = tuple.get_item(1)?.extract()?;
                Ok(crate::inference::HiveInferenceResult {
                    policy: policy_arr.as_slice().unwrap().to_vec(),
                    value: value_arr.as_slice().unwrap().to_vec(),
                })
            },
            InferenceBackend::Native { engine } => {
                engine.infer_batch(boards, reserves, batch_size, num_channels, grid_size, reserve_size)
            },
        }
    }
}

/// Per-turn training record stored during play.
struct TurnRecord {
    board_offset: usize,   // offset into board_data buffer
    reserve_offset: usize, // offset into reserve_data buffer
    turn_color: PieceColor,
    is_value_only: bool,
    policy_vector: Vec<f32>, // POLICY_SIZE, zeroed for value-only
    my_queen_danger: f32,    // current player's queen danger [0, 1]
    opp_queen_danger: f32,   // opponent's queen danger [0, 1]
    my_queen_escape: f32,    // current player's queen escape [0, 1]
    opp_queen_escape: f32,   // opponent's queen escape [0, 1]
    my_mobility: f32,        // current player's piece mobility ratio [0, 1]
    opp_mobility: f32,       // opponent's piece mobility ratio [0, 1]
}

/// Compute queen danger score for a given color.
/// Returns value in [0, 1]: neighbor count / 6, plus bonus if any piece is on top of queen.
/// Returns 0.0 if queen is not on the board.
fn queen_danger(game: &Game, color: PieceColor) -> f32 {
    let queen = Piece::new(color, PieceType::Queen, 1);
    match game.board.piece_position(queen) {
        None => 0.0,
        Some(pos) => {
            let neighbors = hex_neighbors(pos).iter()
                .filter(|&&n| game.board.is_occupied(n))
                .count() as f32;
            (neighbors / 6.0).min(1.0)
        }
    }
}

/// Compute queen escape score for a given color.
/// Returns the number of legal slide destinations for the queen / 6.
/// Returns 0.0 if queen is not on the board.
fn queen_escape(game: &Game, color: PieceColor) -> f32 {
    let queen = Piece::new(color, PieceType::Queen, 1);
    match game.board.piece_position(queen) {
        None => 0.0,
        Some(pos) => {
            // Queen must be on top to move
            if game.board.top_piece(pos) != Some(queen) {
                return 0.0;
            }
            // Check one-hive: if queen is an articulation point, she can't move
            if game.board.stack_height(pos) == 1 {
                let aps = game.board.articulation_points();
                if aps.contains(&pos) {
                    return 0.0;
                }
            }
            // Count empty neighbors the queen can slide to
            let mut count = 0u32;
            for &n in hex_neighbors(pos).iter() {
                if !game.board.is_occupied(n) && game.board.can_slide(pos, n) {
                    // Must remain adjacent to hive after moving
                    if hex_neighbors(n).iter().any(|&adj| adj != pos && game.board.is_occupied(adj)) {
                        count += 1;
                    }
                }
            }
            count as f32 / 6.0
        }
    }
}

/// Compute piece mobility ratio for a given color.
/// Returns fraction of pieces on board that have at least one legal move.
/// Returns 0.0 if no pieces are placed.
fn piece_mobility(game: &mut Game, color: PieceColor) -> f32 {
    let on_board = game.board.pieces_on_board(color);
    if on_board.is_empty() {
        return 0.0;
    }
    // Only check mobility if queen is placed (pieces can't move without queen)
    if !game.queen_placed(color) {
        return 0.0;
    }
    let aps = game.board.articulation_points();
    let mut mobile = 0u32;
    for &piece in &on_board {
        let moves = hive_game::rules::get_moves(piece, &mut game.board, &aps);
        if !moves.is_empty() {
            mobile += 1;
        }
    }
    mobile as f32 / on_board.len() as f32
}

/// Configuration for self-play session.
struct SelfPlayConfig {
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
    /// If set, every inference call is padded to exactly this many positions.
    /// Keeps the batch shape constant for QNN/NPU backends that require fixed shapes.
    fixed_batch_size: Option<usize>,
}

/// Result of a self-play session, returned to Python.
#[pyclass(name = "SelfPlayResult")]
pub struct PySelfPlayResult {
    grid_size: usize,
    // Contiguous training data
    board_data: Vec<f32>,   // [total_samples * board_size]
    reserve_data: Vec<f32>, // [total_samples * RESERVE_SIZE]
    policy_data: Vec<f32>,  // [total_samples * policy_size]
    value_targets: Vec<f32>,
    value_only_flags: Vec<bool>,
    policy_only_flags: Vec<bool>,
    my_queen_danger: Vec<f32>,   // [total_samples] auxiliary target
    opp_queen_danger: Vec<f32>,  // [total_samples] auxiliary target
    my_queen_escape: Vec<f32>,   // [total_samples] auxiliary target
    opp_queen_escape: Vec<f32>,  // [total_samples] auxiliary target
    my_mobility: Vec<f32>,       // [total_samples] auxiliary target
    opp_mobility: Vec<f32>,      // [total_samples] auxiliary target
    num_samples: usize,
    // Stats
    wins_w: u32,
    wins_b: u32,
    draws: u32,
    resignations: u32,
    total_moves: u32,
    full_search_turns: u32,
    total_turns: u32,
    // Calibration
    calibration_total: u32,
    calibration_would_resign: u32,
    calibration_false_positives: u32,
    // Playout cap enabled
    use_playout_cap: bool,
    // Final games for board rendering
    final_games: Vec<Game>,
}

#[pymethods]
impl PySelfPlayResult {
    /// Get training data as numpy arrays.
    /// Returns (boards[N,C,H,W], reserves[N,10], policies[N,5819],
    ///          values[N], value_only[N], policy_only[N], aux_targets[N,6])
    /// aux_targets columns: [my_qd, opp_qd, my_qe, opp_qe, my_mob, opp_mob]
    fn training_data<'py>(&self, py: Python<'py>) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Vec<bool>,
        Vec<bool>,
        Bound<'py, PyArray2<f32>>,
    ) {
        let n = self.num_samples;
        let board_size = NUM_CHANNELS * self.grid_size * self.grid_size;
        let ps = move_encoding::policy_size(self.grid_size);
        let boards = numpy::ndarray::Array2::from_shape_vec(
            (n, board_size), self.board_data.clone(),
        ).unwrap();
        let reserves = numpy::ndarray::Array2::from_shape_vec(
            (n, RESERVE_SIZE), self.reserve_data.clone(),
        ).unwrap();
        let policies = numpy::ndarray::Array2::from_shape_vec(
            (n, ps), self.policy_data.clone(),
        ).unwrap();
        let values = numpy::ndarray::Array1::from(self.value_targets.clone());

        // Pack all auxiliary targets into a single [N, 6] array
        let mut aux_data = Vec::with_capacity(n * 6);
        for i in 0..n {
            aux_data.push(self.my_queen_danger[i]);
            aux_data.push(self.opp_queen_danger[i]);
            aux_data.push(self.my_queen_escape[i]);
            aux_data.push(self.opp_queen_escape[i]);
            aux_data.push(self.my_mobility[i]);
            aux_data.push(self.opp_mobility[i]);
        }
        let aux = numpy::ndarray::Array2::from_shape_vec((n, 6), aux_data).unwrap();

        (
            PyArray2::from_owned_array_bound(py, boards),
            PyArray2::from_owned_array_bound(py, reserves),
            PyArray2::from_owned_array_bound(py, policies),
            PyArray1::from_owned_array_bound(py, values),
            self.value_only_flags.clone(),
            self.policy_only_flags.clone(),
            PyArray2::from_owned_array_bound(py, aux),
        )
    }

    #[getter]
    fn num_samples(&self) -> usize { self.num_samples }
    #[getter]
    fn wins_w(&self) -> u32 { self.wins_w }
    #[getter]
    fn wins_b(&self) -> u32 { self.wins_b }
    #[getter]
    fn draws(&self) -> u32 { self.draws }
    #[getter]
    fn resignations(&self) -> u32 { self.resignations }
    #[getter]
    fn total_moves(&self) -> u32 { self.total_moves }
    #[getter]
    fn full_search_turns(&self) -> u32 { self.full_search_turns }
    #[getter]
    fn total_turns(&self) -> u32 { self.total_turns }
    #[getter]
    fn use_playout_cap(&self) -> bool { self.use_playout_cap }
    #[getter]
    fn calibration_total(&self) -> u32 { self.calibration_total }
    #[getter]
    fn calibration_would_resign(&self) -> u32 { self.calibration_would_resign }
    #[getter]
    fn calibration_false_positives(&self) -> u32 { self.calibration_false_positives }

    /// Get final game states as HiveGame objects (for board rendering).
    fn final_games(&self) -> Vec<crate::hive_python::PyGame> {
        self.final_games.iter().map(|g| crate::hive_python::PyGame { game: g.clone() }).collect()
    }
}

/// Self-play session that runs entire games in Rust.
#[pyclass(name = "RustSelfPlaySession")]
pub struct PySelfPlaySession {
    config: SelfPlayConfig,
}

#[pymethods]
impl PySelfPlaySession {
    #[new]
    #[pyo3(signature = (
        num_games,
        simulations = 100,
        max_moves = 200,
        temperature = 1.0,
        temp_threshold = 30,
        playout_cap_p = 0.0,
        fast_cap = 20,
        c_puct = 1.5,
        dir_alpha = 0.3,
        dir_epsilon = 0.25,
        leaf_batch_size = 1,
        resign_threshold = None,
        resign_moves = 5,
        resign_min_moves = 20,
        calibration_frac = 0.1,
        random_opening_moves_min = 0,
        random_opening_moves_max = 0,
        skip_timeout_games = false,
        grid_size = 23,
        fixed_batch_size = None,
    ))]
    fn new(
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
        fixed_batch_size: Option<usize>,
    ) -> Self {
        PySelfPlaySession {
            config: SelfPlayConfig {
                num_games, simulations, max_moves, temperature, temp_threshold,
                playout_cap_p, fast_cap, c_puct, dir_alpha, dir_epsilon, leaf_batch_size,
                resign_threshold, resign_moves, resign_min_moves, calibration_frac,
                random_opening_moves_min, random_opening_moves_max, skip_timeout_games,
                grid_size, fixed_batch_size,
            },
        }
    }

    /// Play all games to completion.
    /// Uses eval_fn (Python callback) or onnx_path (Rust-native ORT) for GPU inference.
    /// progress_fn(finished, total, active, total_moves, resigned) is called each turn.
    /// opening_sequences: per-game UHP move lists to replay before MCTS (empty list = use random_opening_moves).
    #[pyo3(signature = (eval_fn=None, progress_fn=None, opening_sequences=None, onnx_path=None))]
    fn play_games(
        &self,
        py: Python<'_>,
        eval_fn: Option<&Bound<'_, PyAny>>,
        progress_fn: Option<&Bound<'_, PyAny>>,
        opening_sequences: Option<Vec<Vec<String>>>,
        onnx_path: Option<String>,
    ) -> PyResult<PySelfPlayResult> {
        let opening_sequences = opening_sequences.unwrap_or_default();
        let mut backend = if let Some(ref path) = onnx_path {
            InferenceBackend::Native {
                engine: Box::new(crate::inference::HiveOrtEngine::load(path)
                    .expect("Failed to load ONNX model")),
            }
        } else {
            let ef = eval_fn.expect("eval_fn is required when onnx_path is not provided");
            InferenceBackend::Python { py, eval_fn: ef }
        };
        let cfg = &self.config;
        let num_games = cfg.num_games;
        let use_playout_cap = cfg.playout_cap_p > 0.0;
        let grid_size = cfg.grid_size;
        let board_size = NUM_CHANNELS * grid_size * grid_size;
        let policy_size = move_encoding::policy_size(grid_size);

        // --- Initialize state ---
        let mut games: Vec<Game> = (0..num_games).map(|_| Game::new_with_grid_size(grid_size)).collect();
        let mut searches: Vec<MctsSearch<Game>> = (0..num_games).map(|_| {
            let mut s = MctsSearch::<Game>::new(100_000);
            s.c_puct = cfg.c_puct;
            s.use_forced_playouts = true;
            s
        }).collect();
        let mut move_counts: Vec<u32> = vec![0; num_games];
        let mut active: Vec<bool> = vec![true; num_games];
        let mut finished_count: u32 = 0;

        // Per-game training history: offsets into contiguous buffers
        let mut histories: Vec<Vec<TurnRecord>> = (0..num_games).map(|_| Vec::new()).collect();

        // Contiguous training data buffers (grow as we go)
        let mut board_buf: Vec<f32> = Vec::new();
        let mut reserve_buf: Vec<f32> = Vec::new();

        // Per-game random opening move counts (sampled once per game from [min, max])
        let game_random_opening_moves: Vec<u32> = {
            let mut rng = rand::rng();
            (0..num_games).map(|_| {
                if cfg.random_opening_moves_max > cfg.random_opening_moves_min {
                    rng.random_range(cfg.random_opening_moves_min..=cfg.random_opening_moves_max)
                } else {
                    cfg.random_opening_moves_min
                }
            }).collect()
        };

        // Opening sequence state: tracks games that have abandoned their sequence early
        let mut opening_done: Vec<bool> = vec![false; num_games];

        // Per-game D6 symmetry for opening book moves
        let opening_syms: Vec<core_game::symmetry::D6Symmetry> = {
            let mut rng = rand::rng();
            (0..num_games).map(|_| core_game::symmetry::D6Symmetry::random(&mut rng)).collect()
        };

        // Resignation state
        let mut resign_counters: Vec<u32> = vec![0; num_games];
        let mut resigned_as: Vec<Option<PieceColor>> = vec![None; num_games];

        // Calibration games (ignore resignation to measure false positive rate)
        let mut calibration: Vec<bool> = vec![false; num_games];
        let mut calibration_would_resign: Vec<Option<PieceColor>> = vec![None; num_games];
        if cfg.resign_threshold.is_some() {
            let num_cal = (num_games as f32 * cfg.calibration_frac).ceil().max(1.0) as usize;
            let mut rng = rand::rng();
            let mut indices: Vec<usize> = (0..num_games).collect();
            // Fisher-Yates partial shuffle
            for i in 0..num_cal.min(num_games) {
                let j = rng.random_range(i..num_games);
                indices.swap(i, j);
                calibration[indices[i]] = true;
            }
        }

        // Stats
        let mut full_search_turns: u32 = 0;
        let mut total_turns: u32 = 0;

        // --- Main game loop ---
        while active.iter().any(|&a| a) {
            let mut rng = rand::rng();

            // Collect games that need MCTS search this turn
            let mut mcts_games: Vec<usize> = Vec::new();
            for gi in 0..num_games {
                if !active[gi] { continue; }

                // Opening phase: boardspace sequence or random moves (not recorded to history)
                let game_seq = opening_sequences.get(gi).filter(|s| !s.is_empty());
                if let Some(seq) = game_seq {
                    // Boardspace opening: replay next move from sequence
                    if !opening_done[gi] && (move_counts[gi] as usize) < seq.len() {
                        let move_str = &seq[move_counts[gi] as usize];
                        let transformed = hive_game::uhp::transform_uhp_move(move_str, opening_syms[gi]);
                        let valid = games[gi].valid_moves();
                        if let Some(mv) = valid.iter().find(|m| hive_game::uhp::format_move_uhp(&games[gi], m) == transformed) {
                            games[gi].play_move(mv).unwrap();
                            move_counts[gi] += 1;
                            if games[gi].is_game_over() || move_counts[gi] >= cfg.max_moves {
                                active[gi] = false;
                                finished_count += 1;
                            }
                        } else {
                            // Move not found in valid moves — sequence desync, switch to MCTS
                            opening_done[gi] = true;
                        }
                        continue;
                    }
                } else if move_counts[gi] < game_random_opening_moves[gi] {
                    // Random opening phase: play a single random move
                    let valid = games[gi].valid_moves();
                    if valid.is_empty() {
                        games[gi].play_pass();
                    } else {
                        let idx = rng.random_range(0..valid.len());
                        games[gi].play_move(&valid[idx]).unwrap();
                    }
                    move_counts[gi] += 1;
                    if games[gi].is_game_over() || move_counts[gi] >= cfg.max_moves {
                        active[gi] = false;
                        finished_count += 1;
                    }
                    continue;
                }

                if games[gi].valid_moves().is_empty() {
                    games[gi].play_pass();
                    move_counts[gi] += 1;
                    if games[gi].is_game_over() || move_counts[gi] >= cfg.max_moves {
                        active[gi] = false;
                        finished_count += 1;
                    }
                } else {
                    mcts_games.push(gi);
                }
            }

            if mcts_games.is_empty() { continue; }

            let n = mcts_games.len();

            // --- Decide fast vs full per game ---
            let is_full: Vec<bool> = if use_playout_cap {
                (0..n).map(|_| rng.random::<f32>() < cfg.playout_cap_p).collect()
            } else {
                vec![true; n]
            };

            // --- Encode positions into training buffer (f32) ---
            let mut turn_board_offsets: Vec<usize> = Vec::with_capacity(n);
            let mut turn_reserve_offsets: Vec<usize> = Vec::with_capacity(n);
            for &gi in mcts_games.iter() {
                let board_off = board_buf.len();
                let reserve_off = reserve_buf.len();
                board_buf.resize(board_off + board_size, 0.0);
                reserve_buf.resize(reserve_off + RESERVE_SIZE, 0.0);
                board_encoding::encode_board(
                    &games[gi],
                    &mut board_buf[board_off..board_off + board_size],
                    &mut reserve_buf[reserve_off..reserve_off + RESERVE_SIZE],
                    grid_size,
                );
                turn_board_offsets.push(board_off);
                turn_reserve_offsets.push(reserve_off);
            }

            // --- Initial policy eval (random D6 symmetry per game to avoid orientation bias) ---
            let root_syms: Vec<D6Symmetry> = (0..n).map(|_| D6Symmetry::random(&mut rng)).collect();
            let mut flat_boards = Vec::with_capacity(n * board_size);
            let mut flat_reserves = Vec::with_capacity(n * RESERVE_SIZE);
            for i in 0..n {
                flat_boards.extend_from_slice(
                    &board_buf[turn_board_offsets[i]..turn_board_offsets[i] + board_size]
                );
                flat_reserves.extend_from_slice(
                    &reserve_buf[turn_reserve_offsets[i]..turn_reserve_offsets[i] + RESERVE_SIZE]
                );
                apply_d6_sym_spatial(
                    &mut flat_boards[i * board_size..(i + 1) * board_size],
                    root_syms[i], NUM_CHANNELS, grid_size,
                );
            }
            let target = cfg.fixed_batch_size.unwrap_or(n);
            let r = infer_padded(&mut backend, flat_boards, flat_reserves, n, target, NUM_CHANNELS, grid_size);
            let (mut init_policies, init_values) = (r.policy, r.value);
            // Inverse-transform policy back to original orientation
            let num_policy_channels = move_encoding::NUM_POLICY_CHANNELS;
            for i in 0..n {
                apply_d6_sym_spatial(
                    &mut init_policies[i * policy_size..(i + 1) * policy_size],
                    root_syms[i].inverse(), num_policy_channels, grid_size,
                );
            }

            // --- Init MCTS trees ---
            for (i, &gi) in mcts_games.iter().enumerate() {
                let search = &mut searches[gi];
                search.c_puct = cfg.c_puct;
                search.use_forced_playouts = true;
                let policy = &init_policies[i * policy_size..(i + 1) * policy_size];
                search.init(&games[gi], policy);
            }

            // Apply Dirichlet noise to full-search games only
            for (i, &gi) in mcts_games.iter().enumerate() {
                if is_full[i] {
                    searches[gi].apply_root_dirichlet(cfg.dir_alpha, cfg.dir_epsilon);
                }
            }

            // --- Build per-game sim caps and run simulations ---
            let per_game_caps: Vec<usize> = is_full.iter().map(|&full| {
                if full { cfg.simulations } else { cfg.fast_cap }
            }).collect();

            // Check which games have moves (child_count > 0)
            let child_counts: Vec<u16> = mcts_games.iter()
                .map(|&gi| searches[gi].root_child_count())
                .collect();

            let mut searching: Vec<usize> = Vec::new();  // indices into mcts_games
            let mut sim_caps: Vec<usize> = Vec::new();
            for (i, &cc) in child_counts.iter().enumerate() {
                if cc > 0 {
                    searching.push(i);
                    sim_caps.push(per_game_caps[i]);
                } else {
                    let gi = mcts_games[i];
                    games[gi].play_pass();
                    move_counts[gi] += 1;
                    if games[gi].is_game_over() || move_counts[gi] >= cfg.max_moves {
                        active[gi] = false;
                        finished_count += 1;
                    }
                }
            }

            // --- Simulation loop ---
            if !searching.is_empty() {
                let num_policy_channels = move_encoding::NUM_POLICY_CHANNELS;
                let mut sims_done: Vec<usize> = vec![0; searching.len()];
                let mut still_searching: Vec<usize> = (0..searching.len()).collect();
                // per-game leaf tracking (reused across batches)
                let mut per_game_leaf_ids: Vec<Vec<NodeId>> = vec![Vec::new(); searching.len()];
                let mut per_game_syms: Vec<Vec<D6Symmetry>> = vec![Vec::new(); searching.len()];

                while !still_searching.is_empty() {
                    for k in 0..searching.len() {
                        per_game_leaf_ids[k].clear();
                        per_game_syms[k].clear();
                    }
                    let mut flat_boards: Vec<f32> = Vec::new();
                    let mut flat_reserves: Vec<f32> = Vec::new();
                    let mut any_leaves = false;

                    // leaf_batch_size rounds: select 1 leaf per still-searching game per round
                    for _round in 0..cfg.leaf_batch_size {
                        let mut any_this_round = false;
                        for &k in &still_searching {
                            let gi = mcts_games[searching[k]];
                            let leaf_ids = searches[gi].select_leaves(1);
                            sims_done[k] += leaf_ids.len().max(1);
                            for &lid in &leaf_ids {
                                let (board, reserve) = searches[gi].encode_leaf(lid);
                                let sym = D6Symmetry::random(&mut rng);
                                let start = flat_boards.len();
                                flat_boards.extend_from_slice(&board);
                                apply_d6_sym_spatial(&mut flat_boards[start..], sym, NUM_CHANNELS, grid_size);
                                flat_reserves.extend_from_slice(&reserve);
                                per_game_leaf_ids[k].push(lid);
                                per_game_syms[k].push(sym);
                                any_this_round = true;
                                any_leaves = true;
                            }
                        }
                        still_searching.retain(|&k| sims_done[k] < sim_caps[k]);
                        if !any_this_round { break; }
                    }

                    if !any_leaves { break; }

                    let total = flat_boards.len() / board_size;
                    let target = cfg.fixed_batch_size.unwrap_or(total);
                    let result = infer_padded(&mut backend, flat_boards, flat_reserves, total, target, NUM_CHANNELS, grid_size);

                    let mut offset = 0;
                    for k in 0..searching.len() {
                        let n = per_game_leaf_ids[k].len();
                        if n == 0 { continue; }
                        let gi = mcts_games[searching[k]];
                        let policies: Vec<Vec<f32>> = (0..n).map(|j| {
                            let mut p = result.policy[(offset + j) * policy_size..(offset + j + 1) * policy_size].to_vec();
                            apply_d6_sym_spatial(&mut p, per_game_syms[k][j].inverse(), num_policy_channels, grid_size);
                            p
                        }).collect();
                        let values: Vec<f32> = result.value[offset..offset + n].to_vec();
                        searches[gi].expand_and_backprop(&policies, &values);
                        offset += n;
                    }
                }
            }

            // --- Collect results and play moves ---
            for (i, &gi) in mcts_games.iter().enumerate() {
                if child_counts[i] == 0 { continue; }

                // Resignation check
                if let Some(threshold) = cfg.resign_threshold {
                    let val = init_values[i];
                    if val < threshold && move_counts[gi] >= cfg.resign_min_moves {
                        resign_counters[gi] += 1;
                    } else {
                        resign_counters[gi] = 0;
                    }
                    if resign_counters[gi] >= cfg.resign_moves {
                        let color = games[gi].turn_color;
                        if calibration[gi] {
                            if calibration_would_resign[gi].is_none() {
                                calibration_would_resign[gi] = Some(color);
                            }
                        } else {
                            resigned_as[gi] = Some(color);
                            active[gi] = false;
                            finished_count += 1;
                            continue;
                        }
                    }
                }

                let search = &searches[gi];
                let dist = if search.use_forced_playouts {
                    search.get_pruned_visit_distribution()
                } else {
                    search.get_visit_distribution()
                };

                if dist.is_empty() {
                    games[gi].play_pass();
                    move_counts[gi] += 1;
                    if games[gi].is_game_over() || move_counts[gi] >= cfg.max_moves {
                        active[gi] = false;
                        finished_count += 1;
                    }
                    continue;
                }

                // Apply temperature
                let move_num = move_counts[gi];
                let temp = if move_num < cfg.temp_threshold { cfg.temperature } else { 0.0 };
                let mut probs: Vec<f32> = dist.iter().map(|(_, p)| *p).collect();

                if temp == 0.0 || !is_full[i] {
                    // Argmax
                    let best = probs.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(idx, _)| idx).unwrap();
                    for p in probs.iter_mut() { *p = 0.0; }
                    probs[best] = 1.0;
                } else {
                    // Temperature sampling
                    for p in probs.iter_mut() { *p = p.powf(1.0 / temp); }
                    let total: f32 = probs.iter().sum();
                    if total > 0.0 {
                        for p in probs.iter_mut() { *p /= total; }
                    } else {
                        let uniform = 1.0 / probs.len() as f32;
                        for p in probs.iter_mut() { *p = uniform; }
                    }
                }

                // Build policy vector: placements → single slot; movements → marginals (src + dst)
                let mut policy_vector = vec![0.0f32; policy_size];
                for (j, (mv, _)) in dist.iter().enumerate() {
                    if mv.piece.is_some() {
                        match encode_game_move(mv, grid_size) {
                            Some(PolicyIndex::Single(idx)) => {
                                if idx < policy_size { policy_vector[idx] = probs[j]; }
                            }
                            Some(PolicyIndex::Sum(a, b)) => {
                                if a < policy_size { policy_vector[a] += probs[j]; }
                                if b < policy_size { policy_vector[b] += probs[j]; }
                            }
                            None => {}
                        }
                    }
                }

                // Record training data
                let turn_color = games[gi].turn_color;
                let opp_color = match turn_color {
                    PieceColor::White => PieceColor::Black,
                    PieceColor::Black => PieceColor::White,
                };
                let is_value_only = !is_full[i];
                if is_full[i] { full_search_turns += 1; }
                total_turns += 1;

                histories[gi].push(TurnRecord {
                    board_offset: turn_board_offsets[i],
                    reserve_offset: turn_reserve_offsets[i],
                    turn_color,
                    is_value_only,
                    policy_vector,
                    my_queen_danger: queen_danger(&games[gi], turn_color),
                    opp_queen_danger: queen_danger(&games[gi], opp_color),
                    my_queen_escape: queen_escape(&games[gi], turn_color),
                    opp_queen_escape: queen_escape(&games[gi], opp_color),
                    my_mobility: piece_mobility(&mut games[gi], turn_color),
                    opp_mobility: piece_mobility(&mut games[gi], opp_color),
                });

                // Sample and play move
                let move_idx = if !is_full[i] || temp == 0.0 {
                    probs.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(idx, _)| idx).unwrap()
                } else {
                    let weighted = WeightedIndex::new(&probs).unwrap();
                    weighted.sample(&mut rng)
                };

                let (mv, _) = &dist[move_idx];
                if mv.piece.is_none() {
                    games[gi].play_pass();
                } else {
                    games[gi].play_move(mv).unwrap();
                }

                move_counts[gi] += 1;
                if games[gi].is_game_over() || move_counts[gi] >= cfg.max_moves {
                    active[gi] = false;
                    finished_count += 1;
                }
            }

            // Progress callback
            if let Some(pfn) = progress_fn {
                let total_m: u32 = move_counts.iter().sum();
                let num_resigned = resigned_as.iter().filter(|r| r.is_some()).count() as u32;
                let num_active = active.iter().filter(|&&a| a).count() as u32;
                let max_turn: u32 = if num_active > 0 {
                    move_counts.iter().zip(active.iter())
                        .filter(|(_, a)| **a)
                        .map(|(&m, _)| m)
                        .max()
                        .unwrap_or(0)
                } else {
                    move_counts.iter().copied().max().unwrap_or(0)
                };
                let _ = pfn.call1((finished_count, num_games as u32, num_active, total_m, num_resigned, max_turn));
            }
            py.check_signals()?;
        }

        // --- Build training samples with outcomes ---
        let mut result_board_data: Vec<f32> = Vec::new();
        let mut result_reserve_data: Vec<f32> = Vec::new();
        let mut result_policy_data: Vec<f32> = Vec::new();
        let mut result_value_targets: Vec<f32> = Vec::new();
        let mut result_value_only: Vec<bool> = Vec::new();
        let mut result_policy_only: Vec<bool> = Vec::new();
        let mut result_my_queen_danger: Vec<f32> = Vec::new();
        let mut result_opp_queen_danger: Vec<f32> = Vec::new();
        let mut result_my_queen_escape: Vec<f32> = Vec::new();
        let mut result_opp_queen_escape: Vec<f32> = Vec::new();
        let mut result_my_mobility: Vec<f32> = Vec::new();
        let mut result_opp_mobility: Vec<f32> = Vec::new();
        let mut num_samples = 0usize;

        let mut wins_w: u32 = 0;
        let mut wins_b: u32 = 0;
        let mut draws: u32 = 0;
        let mut resignations: u32 = 0;

        for gi in 0..num_games {
            // Determine outcome
            let (outcome_w, outcome_b, decisive) = if let Some(color) = resigned_as[gi] {
                resignations += 1;
                match color {
                    PieceColor::White => { wins_b += 1; (-1.0f32, 1.0f32, true) },
                    PieceColor::Black => { wins_w += 1; (1.0f32, -1.0f32, true) },
                }
            } else {
                match games[gi].state {
                    GameState::WhiteWins => { wins_w += 1; (1.0, -1.0, true) },
                    GameState::BlackWins => { wins_b += 1; (-1.0, 1.0, true) },
                    _ => {
                        draws += 1;
                        let (w, b) = games[gi].heuristic_value();
                        (w, b, false)
                    },
                }
            };

            // Optionally skip all training data from timeout games
            if !decisive && cfg.skip_timeout_games {
                continue;
            }

            // Skip value training for timeout games with zero heuristic (balanced position,
            // no meaningful signal for the value head).
            let policy_only = !decisive && outcome_w == 0.0;

            for record in &histories[gi] {
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

        // Calibration stats
        let calibration_total = calibration.iter().filter(|&&c| c).count() as u32;
        let cal_would_resign = calibration_would_resign.iter().filter(|r| r.is_some()).count() as u32;
        let mut cal_false_pos: u32 = 0;
        for gi in 0..num_games {
            if let Some(resign_color) = calibration_would_resign[gi] {
                // False positive: would have resigned but actually won
                let won = match resign_color {
                    PieceColor::White => games[gi].state == GameState::WhiteWins,
                    PieceColor::Black => games[gi].state == GameState::BlackWins,
                };
                if won { cal_false_pos += 1; }
            }
        }

        Ok(PySelfPlayResult {
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
            wins_w, wins_b, draws, resignations,
            total_moves: move_counts.iter().sum(),
            full_search_turns,
            total_turns,
            use_playout_cap,
            calibration_total,
            calibration_would_resign: cal_would_resign,
            calibration_false_positives: cal_false_pos,
            final_games: games,
        })
    }

    /// Play a battle between two models. Games 0..N/2 have model1 as White, N/2..N reversed.
    /// Each model is called with the same bfloat16-encoded batch as `play_games` uses.
    /// eval_fn(boards[N,C,H,W], reserves[N,R]) -> (policy[N,P], value[N])
    /// progress_fn(finished, total, active, total_moves) called after each round.
    #[pyo3(signature = (eval_fn1, eval_fn2, progress_fn=None))]
    fn play_battle(
        &self,
        py: Python<'_>,
        eval_fn1: &Bound<'_, PyAny>,
        eval_fn2: &Bound<'_, PyAny>,
        progress_fn: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyHiveBattleResult> {
        let cfg = &self.config;
        let num_games = cfg.num_games;
        let half = num_games / 2;
        let grid_size = cfg.grid_size;
        let board_size = NUM_CHANNELS * grid_size * grid_size;
        let policy_size = move_encoding::policy_size(grid_size);
        let num_policy_channels = move_encoding::NUM_POLICY_CHANNELS;

        let mut games: Vec<Game> = (0..num_games)
            .map(|_| Game::new_with_grid_size(grid_size))
            .collect();
        let mut searches: Vec<MctsSearch<Game>> = (0..num_games).map(|_| {
            let mut s = MctsSearch::<Game>::new(cfg.simulations + 64);
            s.c_puct = cfg.c_puct;
            s
        }).collect();
        let mut move_counts = vec![0u32; num_games];
        let mut active = vec![true; num_games];
        let mut finished_count = 0u32;
        let mut total_moves = 0u32;
        let mut wins_model1 = 0u32;
        let mut wins_model2 = 0u32;
        let mut draws = 0u32;
        let mut game_lengths: Vec<u32> = Vec::new();

        // model1 = White (Player1) for games 0..half; model1 = Black (Player2) for half..num_games.
        let mut rng = rand::rng();
        let use_fn1 = |gi: usize, player: Player| -> bool {
            (gi < half) == (player == Player::Player1)
        };

        // Call a single eval_fn on a bfloat16-encoded batch, return (policies, values).
        let call_eval = |ef: &Bound<'_, PyAny>, b_bf16: &[u16], r_bf16: &[u16], n: usize|
            -> PyResult<(Vec<f32>, Vec<f32>)>
        {
            let ba = numpy::ndarray::Array2::from_shape_vec(
                (n, board_size), b_bf16.to_vec()
            ).unwrap();
            let bnp = PyArray2::from_owned_array_bound(py, ba);
            let b4d = bnp.reshape([n, NUM_CHANNELS, grid_size, grid_size])
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let ra = numpy::ndarray::Array2::from_shape_vec(
                (n, RESERVE_SIZE), r_bf16.to_vec()
            ).unwrap();
            let rnp = PyArray2::from_owned_array_bound(py, ra);
            let result = ef.call1((b4d, rnp))?;
            let tup = result.downcast::<PyTuple>()
                .map_err(|_| pyo3::exceptions::PyTypeError::new_err("eval_fn must return (policy, value)"))?;
            let p: PyReadonlyArray2<f32> = tup.get_item(0)?.extract()?;
            let v: PyReadonlyArray1<f32> = tup.get_item(1)?.extract()?;
            Ok((p.as_slice()?.to_vec(), v.as_slice()?.to_vec()))
        };

        while active.iter().any(|&a| a) {
            // Collect games needing MCTS; handle passes inline.
            let active_games: Vec<usize> = (0..num_games).filter(|&gi| active[gi]).collect();
            let mut mcts_games: Vec<usize> = Vec::new();
            for gi in active_games {
                if games[gi].valid_moves().is_empty() {
                    games[gi].play_pass();
                    move_counts[gi] += 1;
                    total_moves += 1;
                    if games[gi].is_game_over() || move_counts[gi] >= cfg.max_moves {
                        active[gi] = false;
                        finished_count += 1;
                        game_lengths.push(move_counts[gi]);
                        match games[gi].outcome() {
                            Outcome::WonBy(winner) => {
                                if use_fn1(gi, winner) { wins_model1 += 1; } else { wins_model2 += 1; }
                            }
                            _ => { draws += 1; }
                        }
                    }
                } else {
                    mcts_games.push(gi);
                }
            }
            if mcts_games.is_empty() { continue; }
            let n = mcts_games.len();

            // Encode root positions with random D6 symmetry.
            let root_syms: Vec<D6Symmetry> = (0..n).map(|_| D6Symmetry::random(&mut rng)).collect();
            let mut flat_boards = vec![0f32; n * board_size];
            let mut flat_reserves = vec![0f32; n * RESERVE_SIZE];
            let mut fn1_flags: Vec<bool> = Vec::with_capacity(n);
            for (i, &gi) in mcts_games.iter().enumerate() {
                board_encoding::encode_board(
                    &games[gi],
                    &mut flat_boards[i * board_size..(i + 1) * board_size],
                    &mut flat_reserves[i * RESERVE_SIZE..(i + 1) * RESERVE_SIZE],
                    grid_size,
                );
                apply_d6_sym_spatial(&mut flat_boards[i * board_size..(i + 1) * board_size], root_syms[i], NUM_CHANNELS, grid_size);
                fn1_flags.push(use_fn1(gi, games[gi].next_player()));
            }

            // Call both eval_fns and merge per-game.
            let b_bf16: Vec<u16> = flat_boards.iter().map(|&x| f32_to_bf16(x)).collect();
            let r_bf16: Vec<u16> = flat_reserves.iter().map(|&x| f32_to_bf16(x)).collect();
            let (p1, _v1) = call_eval(eval_fn1, &b_bf16, &r_bf16, n)?;
            let (p2, _v2) = call_eval(eval_fn2, &b_bf16, &r_bf16, n)?;

            let mut init_policies = vec![0f32; n * policy_size];
            for i in 0..n {
                let src = if fn1_flags[i] { &p1 } else { &p2 };
                init_policies[i * policy_size..(i + 1) * policy_size]
                    .copy_from_slice(&src[i * policy_size..(i + 1) * policy_size]);
                apply_d6_sym_spatial(
                    &mut init_policies[i * policy_size..(i + 1) * policy_size],
                    root_syms[i].inverse(), num_policy_channels, grid_size,
                );
            }

            // Init MCTS trees (no Dirichlet for eval).
            for (i, &gi) in mcts_games.iter().enumerate() {
                searches[gi].init(&games[gi], &init_policies[i * policy_size..(i + 1) * policy_size]);
            }

            // Simulation rounds.
            let mut game_sims = vec![0usize; n];
            loop {
                let mut leaf_ids: Vec<NodeId> = Vec::new();
                let mut leaf_game_idx: Vec<usize> = Vec::new();
                for _round in 0..cfg.leaf_batch_size {
                    let mut any = false;
                    for (i, &gi) in mcts_games.iter().enumerate() {
                        if game_sims[i] >= cfg.simulations { continue; }
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
                let mut leaf_boards = vec![0f32; nl * board_size];
                let mut leaf_reserves = vec![0f32; nl * RESERVE_SIZE];
                let mut leaf_syms: Vec<D6Symmetry> = Vec::with_capacity(nl);
                let mut leaf_fn1_flags: Vec<bool> = Vec::with_capacity(nl);

                for (k, (&leaf, &i)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
                    let gi = mcts_games[i];
                    let (board_enc, reserve_enc) = searches[gi].encode_leaf(leaf);
                    leaf_boards[k * board_size..(k + 1) * board_size].copy_from_slice(&board_enc);
                    leaf_reserves[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE].copy_from_slice(&reserve_enc);
                    let sym = D6Symmetry::random(&mut rng);
                    apply_d6_sym_spatial(&mut leaf_boards[k * board_size..(k + 1) * board_size], sym, NUM_CHANNELS, grid_size);
                    leaf_syms.push(sym);
                    let leaf_player = searches[gi].get_leaf_player(leaf);
                    leaf_fn1_flags.push(use_fn1(gi, leaf_player));
                }

                let lb_bf16: Vec<u16> = leaf_boards.iter().map(|&x| f32_to_bf16(x)).collect();
                let lr_bf16: Vec<u16> = leaf_reserves.iter().map(|&x| f32_to_bf16(x)).collect();
                let (lp1, lv1) = call_eval(eval_fn1, &lb_bf16, &lr_bf16, nl)?;
                let (lp2, lv2) = call_eval(eval_fn2, &lb_bf16, &lr_bf16, nl)?;

                struct LeafData { policy: Vec<f32>, value: f32 }
                let mut per_game_leaves: Vec<Vec<NodeId>> = vec![Vec::new(); n];
                let mut per_game_data: Vec<Vec<LeafData>> = (0..n).map(|_| Vec::new()).collect();

                for (k, (&leaf, &i)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
                    let use1 = leaf_fn1_flags[k];
                    let p_src = if use1 { &lp1 } else { &lp2 };
                    let v = if use1 { lv1[k] } else { lv2[k] };
                    let mut policy = p_src[k * policy_size..(k + 1) * policy_size].to_vec();
                    apply_d6_sym_spatial(&mut policy, leaf_syms[k].inverse(), num_policy_channels, grid_size);
                    per_game_leaves[i].push(leaf);
                    per_game_data[i].push(LeafData { policy, value: v });
                }

                for (i, &gi) in mcts_games.iter().enumerate() {
                    if per_game_leaves[i].is_empty() { continue; }
                    let policies: Vec<Vec<f32>> = per_game_data[i].iter().map(|d| d.policy.clone()).collect();
                    let values: Vec<f32> = per_game_data[i].iter().map(|d| d.value).collect();
                    searches[gi].expand_and_backprop(&policies, &values);
                }

                if game_sims.iter().all(|&s| s >= cfg.simulations) { break; }
            }

            // Select best move greedy.
            for (_, &gi) in mcts_games.iter().enumerate() {
                let dist = searches[gi].get_pruned_visit_distribution();
                let mv = if dist.is_empty() {
                    hive_game::game::Move::pass()
                } else {
                    dist.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|(mv, _)| *mv).unwrap()
                };
                let _ = games[gi].play_move(&mv);
                move_counts[gi] += 1;
                total_moves += 1;

                if games[gi].is_game_over() || move_counts[gi] >= cfg.max_moves {
                    active[gi] = false;
                    finished_count += 1;
                    game_lengths.push(move_counts[gi]);
                    match games[gi].outcome() {
                        Outcome::WonBy(winner) => {
                            if use_fn1(gi, winner) { wins_model1 += 1; } else { wins_model2 += 1; }
                        }
                        _ => { draws += 1; }
                    }
                }
            }

            if let Some(pfn) = progress_fn {
                let active_count = active.iter().filter(|&&a| a).count() as u32;
                pfn.call1((finished_count, num_games as u32, active_count, total_moves)).ok();
            }
            py.check_signals()?;
        }

        Ok(PyHiveBattleResult { wins_model1, wins_model2, draws, game_lengths })
    }
}

/// Result of a battle (eval match) between two models.
#[pyclass(name = "HiveBattleResult")]
pub struct PyHiveBattleResult {
    wins_model1: u32,
    wins_model2: u32,
    draws: u32,
    game_lengths: Vec<u32>,
}

#[pymethods]
impl PyHiveBattleResult {
    #[getter] fn wins_model1(&self) -> u32 { self.wins_model1 }
    #[getter] fn wins_model2(&self) -> u32 { self.wins_model2 }
    #[getter] fn draws(&self) -> u32 { self.draws }
    #[getter] fn game_lengths(&self) -> Vec<u32> { self.game_lengths.clone() }
}

/// Simulation loop with per-game caps. Each game selects 1 leaf per round;
/// leaves are flushed to the inference backend every `rounds_per_flush` rounds
/// (or when all games finish). Works for any backend via `HiveInference`.
/// Run inference in chunks of exactly `target` positions, padding the last chunk with zeros.
/// When `actual <= target`: one call, padded up. When `actual > target`: ceil(actual/target) calls.
/// The returned policy/value cover exactly `actual` positions in order.
/// When `target == actual` no allocation overhead occurs.
fn infer_padded(
    backend: &mut dyn HiveInference,
    boards: Vec<f32>,
    reserves: Vec<f32>,
    actual: usize,
    target: usize,
    num_channels: usize,
    grid_size: usize,
) -> crate::inference::HiveInferenceResult {
    let policy_size = move_encoding::policy_size(grid_size);
    let board_size = num_channels * grid_size * grid_size;

    let mut out_policy = Vec::with_capacity(actual * policy_size);
    let mut out_value  = Vec::with_capacity(actual);

    let mut offset = 0;
    while offset < actual {
        let chunk = (actual - offset).min(target);
        let b_slice = &boards[offset * board_size..(offset + chunk) * board_size];
        let r_slice = &reserves[offset * RESERVE_SIZE..(offset + chunk) * RESERVE_SIZE];

        let result = if chunk == target {
            backend.infer_batch(b_slice, r_slice, target, num_channels, grid_size, RESERVE_SIZE)
                .expect("inference failed")
        } else {
            // Last partial chunk — pad to target
            let mut pb = b_slice.to_vec();
            pb.resize(target * board_size, 0.0);
            let mut pr = r_slice.to_vec();
            pr.resize(target * RESERVE_SIZE, 0.0);
            let mut r = backend.infer_batch(&pb, &pr, target, num_channels, grid_size, RESERVE_SIZE)
                .expect("inference failed");
            r.policy.truncate(chunk * policy_size);
            r.value.truncate(chunk);
            r
        };

        out_policy.extend_from_slice(&result.policy);
        out_value.extend_from_slice(&result.value);
        offset += chunk;
    }

    crate::inference::HiveInferenceResult { policy: out_policy, value: out_value }
}


/// Register self-play classes with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySelfPlaySession>()?;
    m.add_class::<PySelfPlayResult>()?;
    m.add_class::<PyHiveBattleResult>()?;
    Ok(())
}
