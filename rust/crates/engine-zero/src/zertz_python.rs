/// PyO3 Python bindings for Zertz self-play. v2

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;

use zertz_game::board_encoding::{encode_board, GRID_SIZE, NUM_CHANNELS, RESERVE_SIZE};
use zertz_game::hex::{is_valid, Hex};
use zertz_game::mcts::arena::NodeId;
use zertz_game::mcts::search::{MctsSearch, PolicyHeads, PLACE_HEAD_SIZE, CAP_HEAD_SIZE};
use zertz_game::move_encoding::{encode_move, POLICY_SIZE};
use zertz_game::random_play::{classify_win, WinType};
use zertz_game::zertz::{Marble, ZertzBoard, ZertzMove, MAX_CAPTURE_JUMPS};
use core_game::game::{Game, Outcome, Player};
use core_game::symmetry::{D6Symmetry, Symmetry, apply_d6_sym_spatial};
use crate::inference::ZertzInference;

const BOARD_FLAT: usize = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;

// ---------------------------------------------------------------------------
// Inference backend
// ---------------------------------------------------------------------------

/// Wraps either a Python eval callback or a native engine (ORT, tract, …).
enum InferenceBackend<'py> {
    Python {
        py: Python<'py>,
        eval_fn: &'py Bound<'py, PyAny>,
    },
    Native {
        engine: Box<dyn crate::inference::ZertzInference>,
    },
}

/// Implement the trait so callers only need `backend.infer_batch(…)` with no
/// match arms at each call site.
impl crate::inference::ZertzInference for InferenceBackend<'_> {
    fn infer_batch(
        &mut self,
        boards: &[f32],
        reserves: &[f32],
        batch_size: usize,
        num_channels: usize,
        grid_size: usize,
        reserve_size: usize,
    ) -> Result<crate::inference::ZertzInferenceResult, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            InferenceBackend::Python { py, eval_fn } => {
                let board_flat = boards.len() / batch_size;
                let board_arr = numpy::ndarray::Array2::from_shape_vec(
                    (batch_size, board_flat), boards.to_vec()
                ).unwrap();
                let board_np = PyArray2::from_owned_array_bound(*py, board_arr);
                let board_4d = board_np.reshape([batch_size, num_channels, grid_size, grid_size])
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
                let reserve_arr = numpy::ndarray::Array2::from_shape_vec(
                    (batch_size, reserve_size), reserves.to_vec()
                ).unwrap();
                let reserve_np = PyArray2::from_owned_array_bound(*py, reserve_arr);

                let result = eval_fn.call1((board_4d, reserve_np))?;
                let tuple = result.downcast::<PyTuple>().map_err(|_|
                    "eval_fn must return (place, cap_source, cap_dest, value) tuple"
                )?;
                let place: PyReadonlyArray2<f32> = tuple.get_item(0)?.extract()?;
                let cap_source: PyReadonlyArray2<f32> = tuple.get_item(1)?.extract()?;
                let cap_dest: PyReadonlyArray2<f32> = tuple.get_item(2)?.extract()?;
                let value: PyReadonlyArray1<f32> = tuple.get_item(3)?.extract()?;
                Ok(crate::inference::ZertzInferenceResult {
                    place: place.as_slice().unwrap().to_vec(),
                    cap_source: cap_source.as_slice().unwrap().to_vec(),
                    cap_dest: cap_dest.as_slice().unwrap().to_vec(),
                    value: value.as_slice().unwrap().to_vec(),
                })
            },
            InferenceBackend::Native { engine } => {
                engine.infer_batch(boards, reserves, batch_size, num_channels, grid_size, reserve_size)
            },
        }
    }
}
// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

#[pyclass(name = "ZertzSelfPlayResult")]
pub struct PyZertzSelfPlayResult {
    board_data: Vec<f32>,
    reserve_data: Vec<f32>,
    policy_data: Vec<f32>,
    value_targets: Vec<f32>,
    weights: Vec<f32>,
    value_only_flags: Vec<bool>,
    capture_turn_flags: Vec<bool>,
    mid_capture_turn_flags: Vec<bool>,
    num_samples: usize,
    wins_p1: u32,
    wins_p2: u32,
    draws: u32,
    wins_white: u32,
    wins_grey: u32,
    wins_black: u32,
    wins_combo: u32,
    total_moves: u32,
    game_lengths: Vec<u32>,
    decisive_lengths: Vec<u32>,
    full_search_turns: u32,
    total_turns: u32,
    isolation_captures: u32,
    jump_captures: u32,
    /// Up to 3 sample boards: (label, board_string) pairs for display.
    sample_board_data: Vec<(String, String)>,
}

#[pymethods]
impl PyZertzSelfPlayResult {
    /// Returns (boards, reserves, policies, values, weights, value_only, capture_turn, mid_capture_turn)
    fn training_data<'py>(&self, py: Python<'py>) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Vec<bool>,
        Vec<bool>,
        Vec<bool>,
    ) {
        let n = self.num_samples;
        let boards = numpy::ndarray::Array2::from_shape_vec(
            (n, BOARD_FLAT), self.board_data.clone(),
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
            self.capture_turn_flags.clone(),
            self.mid_capture_turn_flags.clone(),
        )
    }

    #[getter] fn num_samples(&self) -> usize { self.num_samples }
    #[getter] fn wins_p1(&self) -> u32 { self.wins_p1 }
    #[getter] fn wins_p2(&self) -> u32 { self.wins_p2 }
    #[getter] fn draws(&self) -> u32 { self.draws }
    #[getter] fn wins_white(&self) -> u32 { self.wins_white }
    #[getter] fn wins_grey(&self) -> u32 { self.wins_grey }
    #[getter] fn wins_black(&self) -> u32 { self.wins_black }
    #[getter] fn wins_combo(&self) -> u32 { self.wins_combo }
    #[getter] fn total_moves(&self) -> u32 { self.total_moves }
    #[getter] fn game_lengths(&self) -> Vec<u32> { self.game_lengths.clone() }
    #[getter] fn decisive_lengths(&self) -> Vec<u32> { self.decisive_lengths.clone() }
    #[getter] fn full_search_turns(&self) -> u32 { self.full_search_turns }
    #[getter] fn total_turns(&self) -> u32 { self.total_turns }
    #[getter] fn isolation_captures(&self) -> u32 { self.isolation_captures }
    #[getter] fn jump_captures(&self) -> u32 { self.jump_captures }
    /// Returns list of (label, board_string) for up to 3 decisive games.
    fn sample_boards(&self) -> Vec<(String, String)> { self.sample_board_data.clone() }
}

// ---------------------------------------------------------------------------
// Battle result
// ---------------------------------------------------------------------------

#[pyclass(name = "ZertzBattleResult")]
pub struct PyZertzBattleResult {
    wins_model1: u32,
    wins_model2: u32,
    draws: u32,
    wins_white: u32,
    wins_grey: u32,
    wins_black: u32,
    wins_combo: u32,
    game_lengths: Vec<u32>,
}

#[pymethods]
impl PyZertzBattleResult {
    #[getter] fn wins_model1(&self) -> u32 { self.wins_model1 }
    #[getter] fn wins_model2(&self) -> u32 { self.wins_model2 }
    #[getter] fn draws(&self) -> u32 { self.draws }
    #[getter] fn wins_white(&self) -> u32 { self.wins_white }
    #[getter] fn wins_grey(&self) -> u32 { self.wins_grey }
    #[getter] fn wins_black(&self) -> u32 { self.wins_black }
    #[getter] fn wins_combo(&self) -> u32 { self.wins_combo }
    #[getter] fn game_lengths(&self) -> Vec<u32> { self.game_lengths.clone() }
}

// ---------------------------------------------------------------------------
// Per-turn record
// ---------------------------------------------------------------------------

struct TurnRecord {
    board_offset: usize,
    reserve_offset: usize,
    player: Player,
    is_value_only: bool,
    is_capture_turn: bool,
    is_mid_capture_turn: bool,
    policy_vector: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

#[pyclass(name = "ZertzSelfPlaySession")]
pub struct PyZertzSelfPlaySession {
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
}

#[pymethods]
impl PyZertzSelfPlaySession {
    #[new]
    #[pyo3(signature = (
        num_games,
        simulations = 100,
        max_moves = 200,
        temperature = 1.0,
        temp_threshold = 10,
        c_puct = 1.5,
        dir_alpha = 0.3,
        dir_epsilon = 0.25,
        play_batch_size = 2,
        playout_cap_p = 0.0,
        fast_cap = 20,
    ))]
    fn new(
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
    ) -> Self {
        PyZertzSelfPlaySession {
            num_games, simulations, max_moves, temperature, temp_threshold,
            c_puct, dir_alpha, dir_epsilon, play_batch_size, playout_cap_p, fast_cap,
        }
    }

    /// Play all games to completion.
    /// eval_fn(boards[N,C,H,W]) -> (policy[N,P], value[N])
    /// progress_fn(finished, total, active, total_moves) called after each turn.
    #[pyo3(signature = (eval_fn=None, progress_fn=None, onnx_path=None))]
    fn play_games(
        &self,
        py: Python<'_>,
        eval_fn: Option<&Bound<'_, PyAny>>,
        progress_fn: Option<&Bound<'_, PyAny>>,
        onnx_path: Option<String>,
    ) -> PyResult<PyZertzSelfPlayResult> {
        let mut backend = if let Some(ref path) = onnx_path {
            InferenceBackend::Native {
                engine: Box::new(crate::inference::ZertzOrtEngine::load(path)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?),
            }
        } else {
            let ef = eval_fn.expect("eval_fn is required when onnx_path is not provided");
            InferenceBackend::Python { py, eval_fn: ef }
        };

        let num_games = self.num_games;
        let use_playout_cap = self.playout_cap_p > 0.0;

        let mut boards: Vec<ZertzBoard> = (0..num_games).map(|_| ZertzBoard::default()).collect();
        // Size arena to just cover the simulations needed. With lazy child creation
        // at most ~simulations nodes are ever live per game per turn.
        let arena_capacity = self.simulations + 64;
        let mut searches: Vec<MctsSearch> = (0..num_games).map(|_| {
            let mut s = MctsSearch::new(arena_capacity);
            s.c_puct = self.c_puct;
            s
        }).collect();
        let mut move_counts: Vec<u32> = vec![0; num_games];
        let mut active: Vec<bool> = vec![true; num_games];
        let mut finished_count: u32 = 0;

        let mut histories: Vec<Vec<TurnRecord>> = (0..num_games).map(|_| Vec::new()).collect();
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

        // --- Main game loop ---
        while active.iter().any(|&a| a) {
            let mcts_games: Vec<usize> = (0..num_games).filter(|&gi| active[gi]).collect();
            if mcts_games.is_empty() { break; }

            let n = mcts_games.len();
            total_turns += n as u32;

            // Decide fast vs full search per game
            let is_full: Vec<bool> = if use_playout_cap {
                (0..n).map(|_| rng.gen::<f32>() < self.playout_cap_p).collect()
            } else {
                vec![true; n]
            };
            let sim_caps: Vec<usize> = is_full.iter()
                .map(|&f| if f { self.simulations } else { self.fast_cap })
                .collect();
            full_search_turns += is_full.iter().filter(|&&f| f).count() as u32;

            // Encode current positions (with random D6 symmetry per game to avoid orientation bias)
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

            // Initial NN eval for MCTS root policy
            let r = backend.infer_batch(&flat_boards, &flat_reserves, n, NUM_CHANNELS, GRID_SIZE, RESERVE_SIZE)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let (mut init_place, mut init_src, mut init_dst) = (r.place, r.cap_source, r.cap_dest);
            // Inverse-transform policy heads back to original orientation
            for i in 0..n {
                let inv = root_syms[i].inverse();
                apply_d6_sym_spatial(&mut init_place[i * PLACE_HEAD_SIZE..(i + 1) * PLACE_HEAD_SIZE], inv, PLACE_HEAD_SIZE / (GRID_SIZE * GRID_SIZE), GRID_SIZE);
                apply_d6_sym_spatial(&mut init_src[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE], inv, 1, GRID_SIZE);
                apply_d6_sym_spatial(&mut init_dst[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE], inv, 1, GRID_SIZE);
            }

            // Init MCTS trees
            for (i, &gi) in mcts_games.iter().enumerate() {
                let heads = PolicyHeads {
                    place: &init_place[i * PLACE_HEAD_SIZE..(i + 1) * PLACE_HEAD_SIZE],
                    cap_source: &init_src[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE],
                    cap_dest: &init_dst[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE],
                };
                searches[gi].init(&boards[gi], &heads);
                if is_full[i] {
                    searches[gi].apply_root_dirichlet(self.dir_alpha, self.dir_epsilon);
                }
            }

            // --- Simulation rounds ---
            // play_batch_size = number of selection rounds per NN call (same semantics as Hive).
            // Each round selects 1 leaf per active game, so the actual NN batch is
            // play_batch_size × active_games.
            let mut game_sims: Vec<usize> = vec![0; n];
            loop {
                let mut leaf_ids: Vec<NodeId> = Vec::new();
                let mut leaf_game_idx: Vec<usize> = Vec::new();

                // Run play_batch_size rounds; each round = 1 leaf per active game.
                for _round in 0..self.play_batch_size {
                    let mut any_collected = false;
                    for (i, &gi) in mcts_games.iter().enumerate() {
                        if game_sims[i] >= sim_caps[i] { continue; }
                        let leaves = searches[gi].select_leaves(1);
                        let count = leaves.len();
                        if count > 0 { any_collected = true; }
                        for leaf in leaves {
                            leaf_ids.push(leaf);
                            leaf_game_idx.push(i);
                        }
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

                let lr = backend.infer_batch(&leaf_boards_flat, &leaf_reserves_flat, nl, NUM_CHANNELS, GRID_SIZE, RESERVE_SIZE)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                let (leaf_place, leaf_src, leaf_dst, leaf_values) =
                    (lr.place, lr.cap_source, lr.cap_dest, lr.value);

                let place_channels = PLACE_HEAD_SIZE / (GRID_SIZE * GRID_SIZE);
                let mut per_game_leaves: Vec<Vec<NodeId>> = vec![Vec::new(); n];
                let mut per_game_values: Vec<Vec<f32>> = vec![Vec::new(); n];
                struct LeafHeadData {
                    place: Vec<f32>,
                    src: Vec<f32>,
                    dst: Vec<f32>,
                }
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
                    searches[gi].expand_and_backprop(
                        &per_game_leaves[i],
                        &heads,
                        &per_game_values[i],
                    );
                }

                if game_sims.iter().zip(sim_caps.iter()).all(|(s, c)| *s >= *c) { break; }
            }

            // --- Select and apply moves ---
            for (i, &gi) in mcts_games.iter().enumerate() {
                let dist = searches[gi].get_pruned_visit_distribution();
                let mut policy_vec = vec![0.0f32; POLICY_SIZE];
                for (mv, prob) in &dist {
                    policy_vec[encode_move(mv)] = *prob;
                }

                let is_capture_turn = dist.first().map_or(false, |(mv, _)| matches!(mv, ZertzMove::Capture { .. }));
                let is_mid_capture_turn = boards[gi].is_mid_capture();
                histories[gi].push(TurnRecord {
                    board_offset: turn_board_offsets[i],
                    reserve_offset: turn_reserve_offsets[i],
                    player: boards[gi].next_player(),
                    is_value_only: !is_full[i],
                    is_capture_turn,
                    is_mid_capture_turn,
                    policy_vector: policy_vec,
                });

                let mv = if dist.is_empty() {
                    ZertzMove::Pass
                } else if move_counts[gi] < self.temp_threshold && self.temperature > 0.01 {
                    let weights: Vec<f32> = dist.iter()
                        .map(|(_, p)| p.powf(1.0 / self.temperature))
                        .collect();
                    let wi = WeightedIndex::new(&weights).unwrap();
                    dist[wi.sample(&mut rng)].0
                } else {
                    dist.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0
                };

                boards[gi].play(mv).expect("MCTS selected illegal move");
                move_counts[gi] += 1;
                total_moves += 1;

                if boards[gi].outcome() != Outcome::Ongoing || move_counts[gi] >= self.max_moves {
                    active[gi] = false;
                    finished_count += 1;
                    let len = move_counts[gi];
                    game_lengths.push(len);
                    isolation_captures += boards[gi].isolation_captures.iter()
                        .flat_map(|p| p.iter())
                        .map(|&c| c as u32)
                        .sum::<u32>();
                    jump_captures += boards[gi].jump_captures.iter()
                        .flat_map(|p| p.iter())
                        .map(|&c| c as u32)
                        .sum::<u32>();
                    match boards[gi].outcome() {
                        Outcome::WonBy(winner) => {
                            if winner == Player::Player1 { wins_p1 += 1; } else { wins_p2 += 1; }
                            decisive_lengths.push(len);
                            let win_type = classify_win(&boards[gi], winner);
                            const CW: &str = "\x1b[38;2;255;160;50m";
                            const CG: &str = "\x1b[38;2;100;180;255m";
                            const CB: &str = "\x1b[38;2;255;60;180m";
                            const CC: &str = "\x1b[38;2;80;220;80m";
                            const CR: &str = "\x1b[0m";
                            let win_type_str = match &win_type {
                                WinType::FourWhite => format!("{CW}white{CR}"),
                                WinType::FiveGrey  => format!("{CG}grey{CR}"),
                                WinType::SixBlack  => format!("{CB}black{CR}"),
                                WinType::ThreeEach => format!("{CC}combo{CR}"),
                                WinType::Draw      => String::new(),
                            };
                            match win_type {
                                WinType::FourWhite  => wins_white += 1,
                                WinType::FiveGrey   => wins_grey  += 1,
                                WinType::SixBlack   => wins_black += 1,
                                WinType::ThreeEach  => wins_combo += 1,
                                WinType::Draw       => {}
                            }
                            let label = format!("{} wins ({} moves, {})", if winner == Player::Player1 { "P1" } else { "P2" }, len, win_type_str);
                            sample_board_data.push((label, format!("{}", boards[gi])));
                        }
                        _ => { draws += 1; }
                    }
                }
            }

            // Progress callback
            if let Some(pfn) = progress_fn {
                let active_count = active.iter().filter(|&&a| a).count() as u32;
                pfn.call1((finished_count, num_games as u32, active_count, total_moves)).ok();
            }
            py.check_signals()?;
        }

        // --- Build training data ---
        let total_samples: usize = histories.iter().map(|h| h.len()).sum();
        let mut board_data = Vec::with_capacity(total_samples * BOARD_FLAT);
        let mut reserve_data = Vec::with_capacity(total_samples * RESERVE_SIZE);
        let mut policy_data = Vec::with_capacity(total_samples * POLICY_SIZE);
        let mut value_targets = Vec::with_capacity(total_samples);
        let mut weights = Vec::with_capacity(total_samples);
        let mut value_only_flags = Vec::with_capacity(total_samples);
        let mut capture_turn_flags = Vec::with_capacity(total_samples);
        let mut mid_capture_turn_flags = Vec::with_capacity(total_samples);

        for (gi, history) in histories.iter().enumerate() {
            let outcome = boards[gi].outcome();
            for record in history {
                board_data.extend_from_slice(
                    &board_buf[record.board_offset..record.board_offset + BOARD_FLAT],
                );
                reserve_data.extend_from_slice(
                    &reserve_buf[record.reserve_offset..record.reserve_offset + RESERVE_SIZE],
                );
                policy_data.extend_from_slice(&record.policy_vector);
                let value = match outcome {
                    Outcome::WonBy(winner) => {
                        if winner == record.player { 1.0f32 } else { -1.0f32 }
                    }
                    _ => 0.0f32,
                };
                value_targets.push(value);
                weights.push(1.0f32);
                value_only_flags.push(record.is_value_only);
                capture_turn_flags.push(record.is_capture_turn);
                mid_capture_turn_flags.push(record.is_mid_capture_turn);
            }
        }

        Ok(PyZertzSelfPlayResult {
            board_data,
            reserve_data,
            policy_data,
            value_targets,
            weights,
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

    /// Run a battle between two models. Games 0..N/2 have model1 as P1, model2 as P2.
    /// Games N/2..N are reversed. Returns win/draw counts from model1's perspective.
    fn play_battle(
        &self,
        py: Python,
        eval_fn1: PyObject,
        eval_fn2: PyObject,
        progress_fn: Option<PyObject>,
    ) -> PyResult<PyZertzBattleResult> {
        let num_games = self.num_games;
        let half = num_games / 2;

        let mut boards: Vec<ZertzBoard> = (0..num_games).map(|_| ZertzBoard::default()).collect();
        let arena_capacity = self.simulations + 64;
        let mut searches: Vec<MctsSearch> = (0..num_games).map(|_| {
            let mut s = MctsSearch::new(arena_capacity);
            s.c_puct = self.c_puct;
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

        // Helper: is model1 the evaluator for game gi at player p?
        let use_fn1_for = |gi: usize, player: Player| -> bool {
            (gi < half) == (player == Player::Player1)
        };

        // Helper: call both eval_fns on a batch, return merged heads by fn1_flags.
        // Returns (place_data, src_data, dst_data, value) with per-sample selection.
        let call_evals = |py: Python,
                          flat_boards: &[f32],
                          flat_reserves: &[f32],
                          fn1_flags: &[bool],
                          n: usize|
         -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
            let mk_arrays = |boards: &[f32], reserves: &[f32]| -> PyResult<(_, _)> {
                let ba = numpy::ndarray::Array2::from_shape_vec((n, BOARD_FLAT), boards.to_vec()).unwrap();
                let bnp = PyArray2::from_owned_array_bound(py, ba);
                let b4d = bnp.reshape([n, NUM_CHANNELS, GRID_SIZE, GRID_SIZE]).unwrap();
                let ra = numpy::ndarray::Array2::from_shape_vec((n, RESERVE_SIZE), reserves.to_vec()).unwrap();
                let rnp = PyArray2::from_owned_array_bound(py, ra);
                Ok((b4d, rnp))
            };
            let extract_heads = |res: &Bound<'_, PyAny>| -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
                let t = res.downcast::<PyTuple>().map_err(|_| pyo3::exceptions::PyTypeError::new_err("eval_fn must return (place, cap_source, cap_dest, value)"))?;
                let pl: PyReadonlyArray2<f32> = t.get_item(0)?.extract()?;
                let sr: PyReadonlyArray2<f32> = t.get_item(1)?.extract()?;
                let ds: PyReadonlyArray2<f32> = t.get_item(2)?.extract()?;
                let va: PyReadonlyArray1<f32> = t.get_item(3)?.extract()?;
                Ok((pl.as_slice()?.to_vec(), sr.as_slice()?.to_vec(), ds.as_slice()?.to_vec(), va.as_slice()?.to_vec()))
            };

            let (b4d1, rnp1) = mk_arrays(flat_boards, flat_reserves)?;
            let res1 = eval_fn1.call1(py, (b4d1, rnp1))?;
            let (pl1, sr1, ds1, va1) = extract_heads(res1.bind(py))?;

            let (b4d2, rnp2) = mk_arrays(flat_boards, flat_reserves)?;
            let res2 = eval_fn2.call1(py, (b4d2, rnp2))?;
            let (pl2, sr2, ds2, va2) = extract_heads(res2.bind(py))?;

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

            let (mut init_place, mut init_src, mut init_dst, _) = call_evals(py, &flat_boards, &flat_reserves, &fn1_flags, n)?;
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

            // --- Simulation rounds ---
            let mut game_sims = vec![0usize; n];
            loop {
                let mut leaf_ids: Vec<NodeId> = Vec::new();
                let mut leaf_game_idx: Vec<usize> = Vec::new();
                for _round in 0..self.play_batch_size {
                    let mut any = false;
                    for (i, &gi) in mcts_games.iter().enumerate() {
                        if game_sims[i] >= self.simulations { continue; }
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

                let (leaf_place, leaf_src, leaf_dst, leaf_values) = call_evals(py, &leaf_boards_flat, &leaf_reserves_flat, &leaf_fn1_flags, nl)?;

                struct LeafHeadData {
                    place: Vec<f32>,
                    src: Vec<f32>,
                    dst: Vec<f32>,
                }
                let place_channels = PLACE_HEAD_SIZE / (GRID_SIZE * GRID_SIZE);
                let mut per_game_leaves: Vec<Vec<NodeId>> = vec![Vec::new(); n];
                let mut per_game_head_data: Vec<Vec<LeafHeadData>> = (0..n).map(|_| Vec::new()).collect();
                let mut per_game_values: Vec<Vec<f32>> = vec![Vec::new(); n];
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
                if game_sims.iter().all(|&s| s >= self.simulations) { break; }
            }

            // --- Select and apply moves (greedy) ---
            for (_i, &gi) in mcts_games.iter().enumerate() {
                let dist = searches[gi].get_pruned_visit_distribution();
                let mv = if dist.is_empty() {
                    ZertzMove::Pass
                } else {
                    dist.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0
                };
                boards[gi].play(mv).expect("battle selected illegal move");
                move_counts[gi] += 1;
                total_moves += 1;

                if boards[gi].outcome() != Outcome::Ongoing || move_counts[gi] >= self.max_moves {
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
                pfn.call1(py, (finished_count, num_games as u32, active_count, total_moves)).ok();
            }
            py.check_signals()?;
        }

        Ok(PyZertzBattleResult { wins_model1, wins_model2, draws, wins_white, wins_grey, wins_black, wins_combo, game_lengths })
    }
}

// ---------------------------------------------------------------------------
// Move string utilities  (A-G columns, 1-7 rows, matching board display)
// ---------------------------------------------------------------------------

fn hex_to_coord(h: Hex) -> String {
    let (q, r) = h;
    let col = (b'A' + (q + 3) as u8) as char;
    let row = 4 - r;  // r=-3→7, r=0→4, r=3→1
    format!("{}{}", col, row)
}

fn coord_to_hex(s: &str) -> Result<Hex, String> {
    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return Err(format!("Expected cell like D4, got '{}'", s));
    }
    let col_b = bytes[0].to_ascii_uppercase();
    let row_b = bytes[1];
    if !(b'A'..=b'G').contains(&col_b) {
        return Err(format!("Column must be A-G, got '{}'", col_b as char));
    }
    if !(b'1'..=b'7').contains(&row_b) {
        return Err(format!("Row must be 1-7, got '{}'", row_b as char));
    }
    let q = col_b as i8 - b'A' as i8 - 3;
    let r = 4 - (row_b - b'0') as i8;
    let h = (q, r);
    if !is_valid(h) {
        return Err(format!("'{}' is not on the Zertz board", s));
    }
    Ok(h)
}

pub fn move_to_str(mv: ZertzMove) -> String {
    match mv {
        ZertzMove::Place { color, place_at, remove } =>
            format!("{} {} {}", color, hex_to_coord(place_at), hex_to_coord(remove)),
        ZertzMove::PlaceOnly { color, place_at } =>
            format!("{} {}", color, hex_to_coord(place_at)),
        ZertzMove::Capture { jumps, len } => {
            let mut s = format!("CAP {}", hex_to_coord(jumps[0].0));
            for i in 0..len as usize {
                s.push(' ');
                s.push_str(&hex_to_coord(jumps[i].2));
            }
            s
        }
        ZertzMove::Pass => "pass".to_string(),
    }
}

fn str_to_move(s: &str) -> Result<ZertzMove, String> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.is_empty() {
        return Err("Empty move string".to_string());
    }
    let first = parts[0].to_ascii_uppercase();
    match first.as_str() {
        "CAP" => {
            if parts.len() < 3 {
                return Err("CAP needs at least from + one landing: CAP D4 D6".to_string());
            }
            let positions: Result<Vec<Hex>, _> = parts[1..].iter().map(|s| coord_to_hex(s)).collect();
            let positions = positions?;
            let n_hops = positions.len() - 1;
            if n_hops > MAX_CAPTURE_JUMPS {
                return Err(format!("Too many hops (max {})", MAX_CAPTURE_JUMPS));
            }
            let mut jumps = [((0i8, 0i8), (0i8, 0i8), (0i8, 0i8)); MAX_CAPTURE_JUMPS];
            for i in 0..n_hops {
                let (fq, fr) = positions[i];
                let (tq, tr) = positions[i + 1];
                let dq = tq - fq;
                let dr = tr - fr;
                if dq % 2 != 0 || dr % 2 != 0 {
                    return Err(format!(
                        "Hop from {} to {} is not a valid 2-step jump",
                        hex_to_coord(positions[i]), hex_to_coord(positions[i + 1])
                    ));
                }
                jumps[i] = (positions[i], (fq + dq / 2, fr + dr / 2), positions[i + 1]);
            }
            Ok(ZertzMove::Capture { jumps, len: n_hops as u8 })
        }
        "PASS" => Ok(ZertzMove::Pass),
        "W" | "G" | "B" => {
            let color = match first.as_str() {
                "W" => Marble::White,
                "G" => Marble::Grey,
                _   => Marble::Black,
            };
            match parts.len() {
                2 => Ok(ZertzMove::PlaceOnly { color, place_at: coord_to_hex(parts[1])? }),
                3 => Ok(ZertzMove::Place { color, place_at: coord_to_hex(parts[1])?, remove: coord_to_hex(parts[2])? }),
                _ => Err(format!("Expected 'W/G/B place [remove]', got '{}'", s)),
            }
        }
        _ => Err(format!("Unknown move format '{}': expected CAP/W/G/B/pass", s)),
    }
}

// ---------------------------------------------------------------------------
// ZertzGame — interactive game state exposed to Python
// ---------------------------------------------------------------------------

#[pyclass(name = "ZertzGame")]
struct PyZertzGame {
    board: ZertzBoard,
}

#[pymethods]
impl PyZertzGame {
    #[new]
    fn new() -> Self {
        PyZertzGame { board: ZertzBoard::default() }
    }

    fn valid_moves(&self) -> Vec<String> {
        self.board.legal_moves().into_iter().map(move_to_str).collect()
    }

    fn play(&mut self, move_str: &str) -> PyResult<()> {
        let mv = str_to_move(move_str)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        self.board.play(mv)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn board_str(&self) -> String {
        format!("{}", self.board)
    }

    fn outcome(&self) -> &str {
        match self.board.outcome() {
            Outcome::Ongoing              => "ongoing",
            Outcome::WonBy(Player::Player1) => "p1",
            Outcome::WonBy(Player::Player2) => "p2",
            Outcome::Draw                 => "draw",
        }
    }

    fn next_player(&self) -> u8 {
        match self.board.next_player() {
            Player::Player1 => 0,
            Player::Player2 => 1,
        }
    }

    /// Returns (board[C*H*W], reserve[RESERVE_SIZE])
    fn encode<'py>(&self, py: Python<'py>) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>) {
        let mut board_buf = vec![0f32; BOARD_FLAT];
        let mut reserve_buf = vec![0f32; RESERVE_SIZE];
        encode_board(&self.board, &mut board_buf, &mut reserve_buf);
        (PyArray1::from_vec_bound(py, board_buf), PyArray1::from_vec_bound(py, reserve_buf))
    }

    /// Run MCTS for `simulations` sims and return the best move string.
    /// eval_fn(boards[N, C, H, W]) -> (policy[N, POLICY_SIZE], value[N])
    #[pyo3(signature = (eval_fn, simulations=200, c_puct=1.5))]
    fn best_move(
        &self,
        py: Python<'_>,
        eval_fn: &Bound<'_, PyAny>,
        simulations: usize,
        c_puct: f32,
    ) -> PyResult<String> {
        if self.board.outcome() != Outcome::Ongoing {
            return Err(pyo3::exceptions::PyValueError::new_err("Game is already over"));
        }

        let mut search = MctsSearch::new(simulations + 64);
        search.c_puct = c_puct;

        // Initial NN eval on root position
        let mut board_buf = vec![0f32; BOARD_FLAT];
        let mut reserve_buf = vec![0f32; RESERVE_SIZE];
        encode_board(&self.board, &mut board_buf, &mut reserve_buf);
        let root_arr = numpy::ndarray::Array2::from_shape_vec((1, BOARD_FLAT), board_buf).unwrap();
        let root_np = PyArray2::from_owned_array_bound(py, root_arr);
        let root_4d = root_np.reshape([1, NUM_CHANNELS, GRID_SIZE, GRID_SIZE]).unwrap();
        let root_res_arr = numpy::ndarray::Array2::from_shape_vec((1, RESERVE_SIZE), reserve_buf).unwrap();
        let root_res_np = PyArray2::from_owned_array_bound(py, root_res_arr);

        let result = eval_fn.call1((root_4d, root_res_np))?;
        let tuple = result.downcast::<PyTuple>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("eval_fn must return (place, cap_source, cap_dest, value)")
        })?;
        let rp_place: PyReadonlyArray2<f32> = tuple.get_item(0)?.extract()?;
        let rp_src: PyReadonlyArray2<f32> = tuple.get_item(1)?.extract()?;
        let rp_dst: PyReadonlyArray2<f32> = tuple.get_item(2)?.extract()?;
        let _rp_val: PyReadonlyArray1<f32> = tuple.get_item(3)?.extract()?;
        let root_place = rp_place.as_slice()?.to_vec();
        let root_src = rp_src.as_slice()?.to_vec();
        let root_dst = rp_dst.as_slice()?.to_vec();
        let root_heads = PolicyHeads {
            place: &root_place,
            cap_source: &root_src,
            cap_dest: &root_dst,
        };

        search.init(&self.board, &root_heads);
        search.apply_root_dirichlet(0.3, 0.25);

        // Simulation rounds (batch_size=8)
        let batch = 8usize;
        let mut done = 0usize;
        while done < simulations {
            let leaves = search.select_leaves(batch.min(simulations - done));
            if leaves.is_empty() { break; }
            let nl = leaves.len();

            let mut flat = vec![0f32; nl * BOARD_FLAT];
            let mut flat_res = vec![0f32; nl * RESERVE_SIZE];
            for (k, &leaf) in leaves.iter().enumerate() {
                let (board_enc, reserve_enc) = search.encode_leaf(leaf);
                flat[k * BOARD_FLAT..(k + 1) * BOARD_FLAT].copy_from_slice(&board_enc);
                flat_res[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE].copy_from_slice(&reserve_enc);
            }

            let leaf_arr = numpy::ndarray::Array2::from_shape_vec((nl, BOARD_FLAT), flat).unwrap();
            let leaf_np = PyArray2::from_owned_array_bound(py, leaf_arr);
            let leaf_4d = leaf_np.reshape([nl, NUM_CHANNELS, GRID_SIZE, GRID_SIZE]).unwrap();
            let leaf_res_arr = numpy::ndarray::Array2::from_shape_vec((nl, RESERVE_SIZE), flat_res).unwrap();
            let leaf_res_np = PyArray2::from_owned_array_bound(py, leaf_res_arr);

            let res = eval_fn.call1((leaf_4d, leaf_res_np))?;
            let tup = res.downcast::<PyTuple>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("eval_fn must return (place, cap_source, cap_dest, value)")
            })?;
            let lp_place: PyReadonlyArray2<f32> = tup.get_item(0)?.extract()?;
            let lp_src: PyReadonlyArray2<f32> = tup.get_item(1)?.extract()?;
            let lp_dst: PyReadonlyArray2<f32> = tup.get_item(2)?.extract()?;
            let lv: PyReadonlyArray1<f32> = tup.get_item(3)?.extract()?;
            let lp_place_data = lp_place.as_slice()?.to_vec();
            let lp_src_data = lp_src.as_slice()?.to_vec();
            let lp_dst_data = lp_dst.as_slice()?.to_vec();
            let leaf_values: Vec<f32> = lv.as_slice()?.to_vec();

            struct LeafHeadData { place: Vec<f32>, src: Vec<f32>, dst: Vec<f32> }
            let leaf_head_data: Vec<LeafHeadData> = (0..nl).map(|k| LeafHeadData {
                place: lp_place_data[k * PLACE_HEAD_SIZE..(k + 1) * PLACE_HEAD_SIZE].to_vec(),
                src: lp_src_data[k * CAP_HEAD_SIZE..(k + 1) * CAP_HEAD_SIZE].to_vec(),
                dst: lp_dst_data[k * CAP_HEAD_SIZE..(k + 1) * CAP_HEAD_SIZE].to_vec(),
            }).collect();
            let leaf_heads: Vec<PolicyHeads> = leaf_head_data.iter().map(|d| PolicyHeads {
                place: &d.place,
                cap_source: &d.src,
                cap_dest: &d.dst,
            }).collect();

            search.expand_and_backprop(&leaves, &leaf_heads, &leaf_values);
            done += nl;
        }

        let dist = search.get_pruned_visit_distribution();
        let best = dist.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(mv, _)| move_to_str(*mv))
            .unwrap_or_else(|| "pass".to_string());
        Ok(best)
    }
}

// ---------------------------------------------------------------------------
// D6 symmetry permutation tables
// ---------------------------------------------------------------------------

/// Compute the 12 D6 symmetry gather-permutation tables for the Zertz 7x7 board grid.
///
/// Returns a list of 12 numpy arrays, each of shape (49,) = 7*7.
/// perm[new_cell] = old_cell, or 49 (sentinel for invalid/out-of-bounds cells → zero).
#[pyfunction]
fn zertz_d6_grid_permutations<'py>(py: Python<'py>) -> Vec<Bound<'py, PyArray1<i64>>> {
    use core_game::symmetry::{D6Symmetry, Symmetry};
    use zertz_game::hex::{all_hexes, hex_to_grid, GRID_SIZE};

    let num_cells = GRID_SIZE * GRID_SIZE; // 49

    D6Symmetry::all().iter().map(|sym| {
        let mut perm = vec![num_cells as i64; num_cells];
        for &h in all_hexes().iter() {
            let (q, r) = h;
            let (q2, r2) = sym.transform_hex(q as i32, r as i32);
            let h2 = (q2 as i8, r2 as i8);
            let (src_row, src_col) = hex_to_grid(h);
            let (dst_row, dst_col) = hex_to_grid(h2);
            let src = src_row * GRID_SIZE + src_col;
            let dst = dst_row * GRID_SIZE + dst_col;
            perm[dst] = src as i64;
        }
        PyArray1::from_owned_array_bound(py, numpy::ndarray::Array1::from(perm))
    }).collect()
}

/// Compute the 12 D6 symmetry gather-permutation tables in hex-index space (0..37).
///
/// Returns a list of 12 numpy arrays, each of shape (37,).
/// perm[new_hex_idx] = old_hex_idx: the new position new_hex_idx gets its value from old_hex_idx.
#[pyfunction]
fn zertz_d6_hex_permutations<'py>(py: Python<'py>) -> Vec<Bound<'py, PyArray1<i64>>> {
    use core_game::symmetry::{D6Symmetry, Symmetry};
    use zertz_game::hex::{all_hexes, hex_to_index, BOARD_SIZE};

    D6Symmetry::all().iter().map(|sym| {
        let mut perm = vec![0i64; BOARD_SIZE];
        for (i, &h) in all_hexes().iter().enumerate() {
            let (q, r) = h;
            let (q2, r2) = sym.transform_hex(q as i32, r as i32);
            let h2 = (q2 as i8, r2 as i8);
            let dst = hex_to_index(h2);
            perm[dst] = i as i64;
        }
        PyArray1::from_owned_array_bound(py, numpy::ndarray::Array1::from(perm))
    }).collect()
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyZertzSelfPlaySession>()?;
    m.add_class::<PyZertzSelfPlayResult>()?;
    m.add_class::<PyZertzBattleResult>()?;
    m.add_class::<PyZertzGame>()?;
    m.add("ZERTZ_POLICY_SIZE", POLICY_SIZE)?;
    m.add("ZERTZ_NUM_CHANNELS", NUM_CHANNELS)?;
    m.add("ZERTZ_GRID_SIZE", GRID_SIZE)?;
    m.add("ZERTZ_RESERVE_SIZE", RESERVE_SIZE)?;
    m.add_function(wrap_pyfunction!(zertz_d6_grid_permutations, m)?)?;
    m.add_function(wrap_pyfunction!(zertz_d6_hex_permutations, m)?)?;
    Ok(())
}
