/// Self-play game loop in Rust.
/// Plays all games to completion, only calling back to Python for GPU NN inference.
/// Returns training data as contiguous arrays.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;

use crate::board::GRID_SIZE;
use crate::board_encoding::{self, NUM_CHANNELS, RESERVE_SIZE};
use crate::game::{Game, GameState};
use crate::mcts::search::MctsSearch;
use crate::move_encoding::{self, POLICY_SIZE, encode_game_move};
use crate::piece::PieceColor;

use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;

const BOARD_SIZE: usize = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;
const DECISIVE_WEIGHT: f32 = 10.0;

/// Wrapper to send raw pointers across threads (safe when indices are unique).
struct SendPtr(*mut MctsSearch);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// Per-turn training record stored during play.
struct TurnRecord {
    board_offset: usize,   // offset into board_data buffer
    reserve_offset: usize, // offset into reserve_data buffer
    turn_color: PieceColor,
    is_value_only: bool,
    policy_vector: Vec<f32>, // POLICY_SIZE, zeroed for value-only
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
    leaf_batch_size: usize,
    resign_threshold: Option<f32>,
    resign_moves: u32,
    resign_min_moves: u32,
    calibration_frac: f32,
}

/// Result of a self-play session, returned to Python.
#[pyclass(name = "SelfPlayResult")]
pub struct PySelfPlayResult {
    // Contiguous training data
    board_data: Vec<f32>,   // [total_samples * BOARD_SIZE]
    reserve_data: Vec<f32>, // [total_samples * RESERVE_SIZE]
    policy_data: Vec<f32>,  // [total_samples * POLICY_SIZE]
    value_targets: Vec<f32>,
    weights: Vec<f32>,
    value_only_flags: Vec<bool>,
    policy_only_flags: Vec<bool>,
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
    /// Returns (boards[N,23,23,23], reserves[N,10], policies[N,6348],
    ///          values[N], weights[N], value_only[N], policy_only[N])
    fn training_data<'py>(&self, py: Python<'py>) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Vec<bool>,
        Vec<bool>,
    ) {
        let n = self.num_samples;
        let boards = numpy::ndarray::Array2::from_shape_vec(
            (n, BOARD_SIZE), self.board_data.clone(),
        ).unwrap();
        let reserves = numpy::ndarray::Array2::from_shape_vec(
            (n, RESERVE_SIZE), self.reserve_data.clone(),
        ).unwrap();
        let policies = numpy::ndarray::Array2::from_shape_vec(
            (n, POLICY_SIZE), self.policy_data.clone(),
        ).unwrap();
        let values = numpy::ndarray::Array1::from(self.value_targets.clone());
        let weights = numpy::ndarray::Array1::from(self.weights.clone());

        (
            PyArray2::from_owned_array_bound(py, boards),
            PyArray2::from_owned_array_bound(py, reserves),
            PyArray2::from_owned_array_bound(py, policies),
            PyArray1::from_owned_array_bound(py, values),
            PyArray1::from_owned_array_bound(py, weights),
            self.value_only_flags.clone(),
            self.policy_only_flags.clone(),
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

    /// Get final game states as RustGame objects (for board rendering).
    fn final_games(&self) -> Vec<crate::python::PyGame> {
        self.final_games.iter().map(|g| crate::python::PyGame { game: g.clone() }).collect()
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
        leaf_batch_size = 512,
        resign_threshold = None,
        resign_moves = 5,
        resign_min_moves = 20,
        calibration_frac = 0.1,
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
        leaf_batch_size: usize,
        resign_threshold: Option<f32>,
        resign_moves: u32,
        resign_min_moves: u32,
        calibration_frac: f32,
    ) -> Self {
        PySelfPlaySession {
            config: SelfPlayConfig {
                num_games, simulations, max_moves, temperature, temp_threshold,
                playout_cap_p, fast_cap, c_puct, leaf_batch_size,
                resign_threshold, resign_moves, resign_min_moves, calibration_frac,
            },
        }
    }

    /// Play all games to completion. Only calls eval_fn for GPU inference.
    /// progress_fn(finished, total, active, total_moves, resigned) is called each turn.
    #[pyo3(signature = (eval_fn, progress_fn=None))]
    fn play_games(
        &self,
        py: Python<'_>,
        eval_fn: &Bound<'_, PyAny>,
        progress_fn: Option<&Bound<'_, PyAny>>,
    ) -> PySelfPlayResult {
        let cfg = &self.config;
        let num_games = cfg.num_games;
        let use_playout_cap = cfg.playout_cap_p > 0.0;

        // --- Initialize state ---
        let mut games: Vec<Game> = (0..num_games).map(|_| Game::new()).collect();
        let mut searches: Vec<MctsSearch> = (0..num_games).map(|_| {
            let mut s = MctsSearch::new(100_000);
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

        // Resignation state
        let mut resign_counters: Vec<u32> = vec![0; num_games];
        let mut resigned_as: Vec<Option<PieceColor>> = vec![None; num_games];

        // Calibration games (ignore resignation to measure false positive rate)
        let mut calibration: Vec<bool> = vec![false; num_games];
        let mut calibration_would_resign: Vec<Option<PieceColor>> = vec![None; num_games];
        if cfg.resign_threshold.is_some() {
            let num_cal = (num_games as f32 * cfg.calibration_frac).ceil().max(1.0) as usize;
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..num_games).collect();
            // Fisher-Yates partial shuffle
            for i in 0..num_cal.min(num_games) {
                let j = rng.gen_range(i..num_games);
                indices.swap(i, j);
                calibration[indices[i]] = true;
            }
        }

        // Stats
        let mut full_search_turns: u32 = 0;
        let mut total_turns: u32 = 0;

        // --- Main game loop ---
        while active.iter().any(|&a| a) {
            // Collect games that need MCTS search this turn
            let mut mcts_games: Vec<usize> = Vec::new();
            for gi in 0..num_games {
                if !active[gi] { continue; }
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
            let mut rng = rand::thread_rng();
            let is_full: Vec<bool> = if use_playout_cap {
                (0..n).map(|_| rng.gen::<f32>() < cfg.playout_cap_p).collect()
            } else {
                vec![true; n]
            };

            // --- Encode positions (rayon parallel) ---
            let board_offsets: Vec<usize> = mcts_games.iter().map(|_| {
                // Will be set after parallel encoding
                0usize
            }).collect();

            // Encode all boards in parallel
            let encoded: Vec<(Vec<f32>, Vec<f32>)> = mcts_games.par_iter().map(|&gi| {
                let mut board = vec![0.0f32; BOARD_SIZE];
                let mut reserve = vec![0.0f32; RESERVE_SIZE];
                board_encoding::encode_board(&games[gi], &mut board, &mut reserve);
                (board, reserve)
            }).collect();

            // Store in contiguous buffers and record offsets
            let mut turn_board_offsets: Vec<usize> = Vec::with_capacity(n);
            let mut turn_reserve_offsets: Vec<usize> = Vec::with_capacity(n);
            for (board, reserve) in &encoded {
                turn_board_offsets.push(board_buf.len());
                turn_reserve_offsets.push(reserve_buf.len());
                board_buf.extend_from_slice(board);
                reserve_buf.extend_from_slice(reserve);
            }

            // --- Build flat arrays for GPU eval ---
            let mut flat_boards = vec![0.0f32; n * BOARD_SIZE];
            let mut flat_reserves = vec![0.0f32; n * RESERVE_SIZE];
            for (i, (board, reserve)) in encoded.iter().enumerate() {
                flat_boards[i * BOARD_SIZE..(i + 1) * BOARD_SIZE].copy_from_slice(board);
                flat_reserves[i * RESERVE_SIZE..(i + 1) * RESERVE_SIZE].copy_from_slice(reserve);
            }

            // --- GPU: initial policy eval ---
            let board_arr = numpy::ndarray::Array2::from_shape_vec(
                (n, BOARD_SIZE), flat_boards,
            ).unwrap();
            let board_np = PyArray2::from_owned_array_bound(py, board_arr);
            let board_4d = board_np.reshape([n, NUM_CHANNELS, GRID_SIZE, GRID_SIZE]).unwrap();
            let reserve_arr = numpy::ndarray::Array2::from_shape_vec(
                (n, RESERVE_SIZE), flat_reserves,
            ).unwrap();
            let reserve_np = PyArray2::from_owned_array_bound(py, reserve_arr);

            let result = eval_fn.call1((board_4d, reserve_np)).expect("eval_fn call failed");
            let tuple = result.downcast::<PyTuple>().expect("eval_fn must return tuple");
            let policy_arr: PyReadonlyArray2<f32> = tuple.get_item(0).unwrap().extract().unwrap();
            let value_arr: PyReadonlyArray1<f32> = tuple.get_item(1).unwrap().extract().unwrap();
            let init_policies = policy_arr.as_slice().unwrap().to_vec();
            let init_values = value_arr.as_slice().unwrap().to_vec();

            // --- Init MCTS trees (rayon parallel) ---
            let game_clones: Vec<Game> = mcts_games.iter().map(|&gi| games[gi].clone()).collect();
            {
                let c_puct = cfg.c_puct;
                let search_ptrs: Vec<SendPtr> = mcts_games.iter()
                    .map(|&gi| SendPtr(&mut searches[gi] as *mut MctsSearch))
                    .collect();

                search_ptrs.par_iter().enumerate().for_each(|(i, sp)| {
                    let search = unsafe { &mut *sp.0 };
                    search.c_puct = c_puct;
                    search.use_forced_playouts = true;
                    let policy = &init_policies[i * POLICY_SIZE..(i + 1) * POLICY_SIZE];
                    search.init(&game_clones[i], policy);
                });
            }

            // Apply Dirichlet noise to full-search games only
            for (i, &gi) in mcts_games.iter().enumerate() {
                if is_full[i] {
                    searches[gi].apply_root_dirichlet(0.3, 0.25);
                }
            }

            // --- Build per-game sim caps and run simulations ---
            let per_game_caps: Vec<usize> = is_full.iter().map(|&full| {
                if full { cfg.simulations } else { cfg.fast_cap }
            }).collect();

            // Check which games have moves (child_count > 0)
            let child_counts: Vec<u16> = mcts_games.iter()
                .map(|&gi| searches[gi].arena.get(searches[gi].root).child_count)
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

            // Run interleaved simulations
            if !searching.is_empty() {
                run_simulations_internal(
                    py,
                    &mut searches,
                    &mcts_games,
                    &searching,
                    &sim_caps,
                    cfg.leaf_batch_size,
                    eval_fn,
                );
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

                // Build policy vector
                let mut policy_vector = vec![0.0f32; POLICY_SIZE];
                for (j, (mv, _)) in dist.iter().enumerate() {
                    if mv.piece.is_some() {
                        if let Some(idx) = encode_game_move(mv) {
                            if idx < POLICY_SIZE {
                                policy_vector[idx] = probs[j];
                            }
                        }
                    }
                }

                // Record training data
                let turn_color = games[gi].turn_color;
                let is_value_only = !is_full[i];
                if is_full[i] { full_search_turns += 1; }
                total_turns += 1;

                histories[gi].push(TurnRecord {
                    board_offset: turn_board_offsets[i],
                    reserve_offset: turn_reserve_offsets[i],
                    turn_color,
                    is_value_only,
                    policy_vector,
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
                    games[gi].play_move(mv);
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
                        .filter(|(_, &a)| a)
                        .map(|(&m, _)| m)
                        .max()
                        .unwrap_or(0)
                } else {
                    move_counts.iter().copied().max().unwrap_or(0)
                };
                let _ = pfn.call1((finished_count, num_games as u32, num_active, total_m, num_resigned, max_turn));
            }
        }

        // --- Build training samples with outcomes ---
        let mut result_board_data: Vec<f32> = Vec::new();
        let mut result_reserve_data: Vec<f32> = Vec::new();
        let mut result_policy_data: Vec<f32> = Vec::new();
        let mut result_value_targets: Vec<f32> = Vec::new();
        let mut result_weights: Vec<f32> = Vec::new();
        let mut result_value_only: Vec<bool> = Vec::new();
        let mut result_policy_only: Vec<bool> = Vec::new();
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

            let weight = if decisive { DECISIVE_WEIGHT } else { 1.0 };
            // Skip value training for timeout games with zero heuristic (balanced position,
            // no meaningful signal for the value head).
            let policy_only = !decisive && outcome_w == 0.0;

            for record in &histories[gi] {
                let board = &board_buf[record.board_offset..record.board_offset + BOARD_SIZE];
                let reserve = &reserve_buf[record.reserve_offset..record.reserve_offset + RESERVE_SIZE];
                let value = match record.turn_color {
                    PieceColor::White => outcome_w,
                    PieceColor::Black => outcome_b,
                };

                result_board_data.extend_from_slice(board);
                result_reserve_data.extend_from_slice(reserve);
                result_policy_data.extend_from_slice(&record.policy_vector);
                result_value_targets.push(value);
                result_weights.push(weight);
                result_value_only.push(record.is_value_only);
                result_policy_only.push(policy_only);
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

        PySelfPlayResult {
            board_data: result_board_data,
            reserve_data: result_reserve_data,
            policy_data: result_policy_data,
            value_targets: result_value_targets,
            weights: result_weights,
            value_only_flags: result_value_only,
            policy_only_flags: result_policy_only,
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
        }
    }
}

/// Internal simulation loop with per-game caps (not exposed to Python).
///
/// Each game selects 1 leaf per round. Leaves are flushed to GPU every
/// `rounds_per_flush` rounds (or when all games finish). The effective GPU batch
/// size is `rounds_per_flush × active_game_count`, which naturally scales down as
/// games complete — unlike a fixed leaf-count threshold.
fn run_simulations_internal(
    py: Python<'_>,
    searches: &mut Vec<MctsSearch>,
    mcts_games: &[usize],       // game indices
    searching: &[usize],        // indices into mcts_games that have moves
    per_game_caps: &[usize],    // cap for each entry in `searching`
    rounds_per_flush: usize,    // GPU call every N rounds; 1 = flush every round
    eval_fn: &Bound<'_, PyAny>,
) {
    let board_size = BOARD_SIZE;
    let mut sims_done: Vec<usize> = vec![0; searching.len()];
    let mut still_searching: Vec<usize> = (0..searching.len()).collect();
    let mut round: usize = 0;

    // pending_gi[i] is the game index (into `searches`) for the i-th pending leaf.
    let mut pending_gi: Vec<usize> = Vec::new();
    let mut pending_leaf_ids: Vec<u32> = Vec::new();
    let mut pending_boards: Vec<f32> = Vec::new();
    let mut pending_reserves: Vec<f32> = Vec::new();

    while !still_searching.is_empty() || !pending_leaf_ids.is_empty() {
        // --- CPU (rayon): each active game selects 1 leaf and encodes it ---
        if !still_searching.is_empty() {
            let active_gis: Vec<usize> = still_searching.iter()
                .map(|&si| mcts_games[searching[si]])
                .collect();

            let search_ptrs: Vec<SendPtr> = active_gis.iter()
                .map(|&gi| SendPtr(&mut searches[gi] as *mut MctsSearch))
                .collect();

            let results: Vec<(Vec<u32>, Vec<f32>, Vec<f32>)> = search_ptrs.par_iter()
                .map(|sp| {
                    let search = unsafe { &mut *sp.0 };
                    let leaves = search.select_leaves(1);
                    let mut boards = vec![0.0f32; leaves.len() * board_size];
                    let mut reserves = vec![0.0f32; leaves.len() * RESERVE_SIZE];
                    for (j, &leaf) in leaves.iter().enumerate() {
                        let game_state = &search.arena.get(leaf).game;
                        board_encoding::encode_board(
                            game_state,
                            &mut boards[j * board_size..(j + 1) * board_size],
                            &mut reserves[j * RESERVE_SIZE..(j + 1) * RESERVE_SIZE],
                        );
                    }
                    (leaves, boards, reserves)
                })
                .collect();

            for (idx, &si) in still_searching.iter().enumerate() {
                let gi = active_gis[idx];
                let (ref leaves, ref boards, ref reserves) = results[idx];
                sims_done[si] += leaves.len().max(1);
                for &leaf_id in leaves {
                    pending_gi.push(gi);
                    pending_leaf_ids.push(leaf_id);
                }
                pending_boards.extend_from_slice(boards);
                pending_reserves.extend_from_slice(reserves);
            }

            still_searching.retain(|&si| sims_done[si] < per_game_caps[si]);
            round += 1;
        }

        // Flush every rounds_per_flush rounds, or on the final round.
        let should_flush = (!pending_leaf_ids.is_empty())
            && (round % rounds_per_flush == 0 || still_searching.is_empty());

        if should_flush {
            let total = pending_leaf_ids.len();

            // --- GPU: single inference call ---
            let boards_data = std::mem::take(&mut pending_boards);
            let reserves_data = std::mem::take(&mut pending_reserves);

            let board_arr = numpy::ndarray::Array2::from_shape_vec(
                (total, board_size), boards_data,
            ).unwrap();
            let board_np = PyArray2::from_owned_array_bound(py, board_arr);
            let board_4d = board_np.reshape([total, NUM_CHANNELS, GRID_SIZE, GRID_SIZE]).unwrap();
            let reserve_arr = numpy::ndarray::Array2::from_shape_vec(
                (total, RESERVE_SIZE), reserves_data,
            ).unwrap();
            let reserve_np = PyArray2::from_owned_array_bound(py, reserve_arr);

            let result = eval_fn.call1((board_4d, reserve_np)).expect("eval_fn call failed");
            let tuple = result.downcast::<PyTuple>().expect("eval_fn must return tuple");
            let policy_arr: PyReadonlyArray2<f32> = tuple.get_item(0).unwrap().extract().unwrap();
            let value_arr: PyReadonlyArray1<f32> = tuple.get_item(1).unwrap().extract().unwrap();
            let policy_data = policy_arr.as_slice().unwrap();
            let value_data = value_arr.as_slice().unwrap();

            // --- Group leaves by game for expand/backprop ---
            // Keys are unique game indices; values are (leaf_ids, policies, values).
            let mut gi_groups: std::collections::HashMap<
                usize, (Vec<u32>, Vec<Vec<f32>>, Vec<f32>)
            > = std::collections::HashMap::new();
            for (i, &gi) in pending_gi.iter().enumerate() {
                let entry = gi_groups.entry(gi).or_default();
                entry.0.push(pending_leaf_ids[i]);
                entry.1.push(policy_data[i * POLICY_SIZE..(i + 1) * POLICY_SIZE].to_vec());
                entry.2.push(value_data[i]);
            }

            // --- CPU (rayon): expand + backprop per game ---
            let unique_gis: Vec<usize> = gi_groups.keys().copied().collect();
            let expand_ptrs: Vec<SendPtr> = unique_gis.iter()
                .map(|&gi| SendPtr(&mut searches[gi] as *mut MctsSearch))
                .collect();

            expand_ptrs.par_iter().zip(unique_gis.par_iter()).for_each(|(sp, &gi)| {
                let search = unsafe { &mut *sp.0 };
                let (leaf_ids, policies, values) = &gi_groups[&gi];
                search.expand_and_backprop(leaf_ids, policies, values);
            });

            pending_gi.clear();
            pending_leaf_ids.clear();
            // pending_boards / pending_reserves already emptied via take()
        }
    }
}

/// Register self-play classes with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySelfPlaySession>()?;
    m.add_class::<PySelfPlayResult>()?;
    Ok(())
}
