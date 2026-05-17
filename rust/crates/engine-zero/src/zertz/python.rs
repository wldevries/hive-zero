/// PyO3 Python bindings for Zertz self-play. v2

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use std::collections::HashMap;
use std::sync::mpsc::Receiver;

use zertz_game::board_encoding::{encode_board, GRID_SIZE, NUM_CHANNELS, RESERVE_SIZE};
use zertz_game::move_encoding::{NN_POLICY_SIZE, PLACE_HEAD_SIZE, CAP_HEAD_SIZE};
use zertz_game::notation::{move_to_str, str_to_move};
use zertz_game::zertz::{classify_win, WinType, ZertzBoard};
use core_game::game::{Game, Outcome, Player};
use zertz_game::search::{
    best_move_core, play_battle_core, play_battle_versioned_core, play_battle_vs_bot_core,
    play_selfplay_core, BatchEvaluator, EvalFn, ModelStyle, RequestId, SyncEvaluator,
};

use super::inference::{ZertzInferenceResult, ZertzInferenceWorker};

const BOARD_FLAT: usize = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;

fn call_python_eval(
    eval_fn: &Py<PyAny>,
    boards: &[f32],
    reserves: &[f32],
    batch_size: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    call_python_eval_n(eval_fn, boards, reserves, batch_size, NUM_CHANNELS)
}

/// Channel-parameterised variant. Lets the V1 (joint) side of versioned
/// battle reshape its 6-channel board tensors before handing them to the
/// Python callback (which is wrapping a `ZertzNetV1` checkpoint).
fn call_python_eval_n(
    eval_fn: &Py<PyAny>,
    boards: &[f32],
    reserves: &[f32],
    batch_size: usize,
    board_channels: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let board_flat = board_channels * GRID_SIZE * GRID_SIZE;
    Python::attach(|py| {
        let board_arr = numpy::ndarray::Array2::from_shape_vec(
            (batch_size, board_flat),
            boards.to_vec(),
        ).map_err(|e| e.to_string())?;
        let board_np = PyArray2::from_owned_array(py, board_arr);
        let board_4d = board_np
            .reshape([batch_size, board_channels, GRID_SIZE, GRID_SIZE])
            .map_err(|e| e.to_string())?;

        let reserve_arr = numpy::ndarray::Array2::from_shape_vec(
            (batch_size, RESERVE_SIZE),
            reserves.to_vec(),
        ).map_err(|e| e.to_string())?;
        let reserve_np = PyArray2::from_owned_array(py, reserve_arr);

        let result = eval_fn
            .bind(py)
            .call1((board_4d, reserve_np))
            .map_err(|e| e.to_string())?;
        let tuple = result
            .cast::<PyTuple>()
            .map_err(|_| "eval_fn must return (place, cap_dir, value) tuple".to_string())?;

        let place = tuple
            .get_item(0)
            .map_err(|e| e.to_string())?
            .cast::<PyArray2<f32>>()
            .map_err(|e| e.to_string())?
            .readonly();
        let cap_dir = tuple
            .get_item(1)
            .map_err(|e| e.to_string())?
            .cast::<PyArray2<f32>>()
            .map_err(|e| e.to_string())?
            .readonly();
        let value = tuple
            .get_item(2)
            .map_err(|e| e.to_string())?
            .cast::<PyArray1<f32>>()
            .map_err(|e| e.to_string())?
            .readonly();

        let place_slice = place.as_slice().map_err(|e| e.to_string())?;
        let cap_dir_slice = cap_dir.as_slice().map_err(|e| e.to_string())?;
        // Concatenate per-sample: [place(196), cap_dir(294)] = flat 490
        let mut flat = Vec::with_capacity(batch_size * NN_POLICY_SIZE);
        for i in 0..batch_size {
            flat.extend_from_slice(&place_slice[i * PLACE_HEAD_SIZE..(i + 1) * PLACE_HEAD_SIZE]);
            flat.extend_from_slice(&cap_dir_slice[i * CAP_HEAD_SIZE..(i + 1) * CAP_HEAD_SIZE]);
        }
        Ok((flat, value.as_slice().map_err(|e| e.to_string())?.to_vec()))
    })
}

// ---------------------------------------------------------------------------
// Pipelined evaluator — hands batches off to a ZertzInferenceWorker so the
// GPU forward for batch N runs while the main thread selects leaves for
// batch N+1.
// ---------------------------------------------------------------------------

/// `BatchEvaluator` impl that hands batches off to a `ZertzInferenceWorker`
/// running on a dedicated thread. `submit` returns immediately after posting
/// the request (one batch in flight at a time, bounded by the worker's
/// channel capacity); `collect` blocks on the per-request response channel.
struct PipelinedEvaluator<'w> {
    worker: &'w ZertzInferenceWorker,
    next_id: RequestId,
    pending: HashMap<RequestId, Receiver<Result<ZertzInferenceResult, String>>>,
}

impl<'w> PipelinedEvaluator<'w> {
    fn new(worker: &'w ZertzInferenceWorker) -> Self {
        Self {
            worker,
            next_id: 0,
            pending: HashMap::new(),
        }
    }
}

impl<'w> BatchEvaluator for PipelinedEvaluator<'w> {
    fn submit(
        &mut self,
        boards: Vec<f32>,
        reserves: Vec<f32>,
        n: usize,
    ) -> Result<RequestId, String> {
        let recv = self.worker.submit(boards, reserves, n)?;
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.pending.insert(id, recv);
        Ok(id)
    }

    fn collect(
        &mut self,
        id: RequestId,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let recv = self
            .pending
            .remove(&id)
            .ok_or_else(|| format!("PipelinedEvaluator: unknown request {id}"))?;
        let result = recv
            .recv()
            .map_err(|_| "zertz-ort-worker died before responding".to_string())??;
        Ok((result.policy, result.value))
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

#[pyclass(name = "ZertzSearchStats", skip_from_py_object)]
#[derive(Clone)]
pub struct PyZertzSearchStats {
    #[pyo3(get)] pub top1_mean: f32,
    #[pyo3(get)] pub top1_std: f32,
    #[pyo3(get)] pub depth_mean: f32,
    #[pyo3(get)] pub depth_std: f32,
    #[pyo3(get)] pub valid_moves_mean: f32,
    #[pyo3(get)] pub valid_moves_std: f32,
}

#[pyclass(name = "ZertzSelfPlayResult")]
pub struct PyZertzSelfPlayResult {
    board_data: Vec<f32>,
    reserve_data: Vec<f32>,
    policy_data: Vec<f32>,
    value_targets: Vec<f32>,
    value_only_flags: Vec<bool>,
    capture_turn_flags: Vec<bool>,
    mid_capture_turn_flags: Vec<bool>,
    mid_placement_turn_flags: Vec<bool>,
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
    top1_visit_fraction_mean: f32,
    top1_visit_fraction_std: f32,
    search_depth_mean: f32,
    search_depth_std: f32,
    valid_moves_mean: f32,
    valid_moves_std: f32,
}

#[pymethods]
impl PyZertzSelfPlayResult {
    /// Returns (boards, reserves, policies, values,
    ///          value_only, capture_turn, mid_capture_turn, mid_placement_turn)
    fn training_data<'py>(&self, py: Python<'py>) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Vec<bool>,
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
            (n, NN_POLICY_SIZE), self.policy_data.clone(),
        ).unwrap();
        let values = numpy::ndarray::Array1::from(self.value_targets.clone());
        (
            PyArray2::from_owned_array(py, boards),
            PyArray2::from_owned_array(py, reserves),
            PyArray2::from_owned_array(py, policies),
            PyArray1::from_owned_array(py, values),
            self.value_only_flags.clone(),
            self.capture_turn_flags.clone(),
            self.mid_capture_turn_flags.clone(),
            self.mid_placement_turn_flags.clone(),
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

    #[getter]
    fn search_stats(&self) -> PyZertzSearchStats {
        PyZertzSearchStats {
            top1_mean: self.top1_visit_fraction_mean,
            top1_std: self.top1_visit_fraction_std,
            depth_mean: self.search_depth_mean,
            depth_std: self.search_depth_std,
            valid_moves_mean: self.valid_moves_mean,
            valid_moves_std: self.valid_moves_std,
        }
    }
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
// Session
// ---------------------------------------------------------------------------

#[pyclass(name = "ZertzSelfPlaySession")]
pub struct PyZertzSelfPlaySession {
    num_games: usize,
    simulations: usize,
    temperature: f32,
    temp_threshold: u32,
    c_puct: f32,
    dir_alpha: f32,
    dir_epsilon: f32,
    play_batch_size: usize,
    playout_cap_p: f32,
    fast_cap: usize,
    forced_playouts: bool,
    draw_contempt: f32,
}

#[pymethods]
impl PyZertzSelfPlaySession {
    #[new]
    #[pyo3(signature = (
        num_games,
        simulations = 100,
        temperature = 1.0,
        temp_threshold = 10,
        c_puct = 1.5,
        dir_alpha = 0.3,
        dir_epsilon = 0.25,
        play_batch_size = 2,
        playout_cap_p = 0.0,
        fast_cap = 20,
        forced_playouts = false,
        draw_contempt = 0.0,
        asymmetric_contempt = false,
    ))]
    fn new(
        num_games: usize,
        simulations: usize,
        temperature: f32,
        temp_threshold: u32,
        c_puct: f32,
        dir_alpha: f32,
        dir_epsilon: f32,
        play_batch_size: usize,
        playout_cap_p: f32,
        fast_cap: usize,
        forced_playouts: bool,
        draw_contempt: f32,
        // Accepted for kwarg compatibility with the shared MctsConfig — Zertz
        // does not currently use asymmetric contempt (no draw-convergence
        // problem because games are finite by construction).
        asymmetric_contempt: bool,
    ) -> Self {
        let _ = asymmetric_contempt;
        PyZertzSelfPlaySession {
            num_games, simulations, temperature, temp_threshold,
            c_puct, dir_alpha, dir_epsilon, play_batch_size, playout_cap_p, fast_cap,
            forced_playouts, draw_contempt,
        }
    }

    /// Play all games to completion.
    /// eval_fn(boards[N,C,H,W]) -> (policy[N,P], value[N])
    /// progress_fn(finished, total, active, total_moves) called after each turn.
    #[pyo3(signature = (eval_fn=None, progress_fn=None, onnx_path=None))]
    fn play_games(
        &self,
        _py: Python<'_>,
        eval_fn: Option<&Bound<'_, PyAny>>,
        progress_fn: Option<&Bound<'_, PyAny>>,
        onnx_path: Option<String>,
    ) -> PyResult<PyZertzSelfPlayResult> {
        let progress_core = progress_fn.map(|pfn| {
            let pfn = pfn.clone().unbind();
            Box::new(move |finished: u32, total: u32, active: u32, total_moves: u32| {
                Python::attach(|py| {
                    pfn.bind(py).call1((finished, total, active, total_moves)).ok();
                    py.check_signals().ok();
                });
            }) as Box<dyn Fn(u32, u32, u32, u32) + Send + Sync>
        });

        let mut mcts = core_game::selfplay_config::MctsConfig {
            simulations: self.simulations,
            c_puct: self.c_puct,
            dir_alpha: self.dir_alpha,
            dir_epsilon: self.dir_epsilon,
            forced_playouts: self.forced_playouts,
            draw_contempt: self.draw_contempt,
            asymmetric_contempt: false,
            play_batch_size: self.play_batch_size,
            temperature: self.temperature,
            temp_threshold: self.temp_threshold,
        };
        let playout_cap = core_game::selfplay_config::PlayoutCapConfig {
            p: self.playout_cap_p,
            fast_cap: self.fast_cap,
        };

        let r = if let Some(path) = onnx_path {
            // Pipelined ORT path: a worker thread owns the ZertzOrtEngine;
            // play_selfplay_core overlaps leaf selection for batch N+1 with
            // the GPU forward for batch N.
            //
            // Halve play_batch_size so per-game in-flight virtual-loss leaves
            // stay constant: the pipeline holds two batches alive at once
            // (the one running on GPU + the next being selected), so VL
            // accumulates over `2 × play_batch_size` leaves per game vs.
            // `play_batch_size` on the sync path.
            let original_pbs = mcts.play_batch_size;
            if original_pbs > 1 {
                mcts.play_batch_size = original_pbs / 2;
            }

            let worker = ZertzInferenceWorker::spawn(
                path,
                NUM_CHANNELS,
                GRID_SIZE,
                RESERVE_SIZE,
            )
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            let mut evaluator = PipelinedEvaluator::new(&worker);

            let result = play_selfplay_core(
                self.num_games,
                mcts,
                playout_cap,
                &mut evaluator,
                progress_core,
            );

            // Release the evaluator's borrow before taking ownership of the
            // worker for shutdown.
            drop(evaluator);
            let _timing = worker.shutdown_and_drain_timing();

            result.map_err(pyo3::exceptions::PyRuntimeError::new_err)?
        } else {
            // Sync path: Python eval callback wrapped in SyncEvaluator. The
            // GIL precludes pipelining; this preserves the previous
            // behaviour exactly.
            let py_eval = eval_fn
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("eval_fn is required when onnx_path is not provided"))?
                .clone()
                .unbind();
            let core_eval: EvalFn = Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
                call_python_eval(&py_eval, boards, reserves, n)
            });
            let mut evaluator = SyncEvaluator::new(core_eval);
            play_selfplay_core(
                self.num_games,
                mcts,
                playout_cap,
                &mut evaluator,
                progress_core,
            )
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
        };

        Ok(PyZertzSelfPlayResult {
            board_data: r.board_data,
            reserve_data: r.reserve_data,
            policy_data: r.policy_data,
            value_targets: r.value_targets,
            value_only_flags: r.value_only_flags,
            capture_turn_flags: r.capture_turn_flags,
            mid_capture_turn_flags: r.mid_capture_turn_flags,
            mid_placement_turn_flags: r.mid_placement_turn_flags,
            num_samples: r.num_samples,
            wins_p1: r.wins_p1,
            wins_p2: r.wins_p2,
            draws: r.draws,
            wins_white: r.wins_white,
            wins_grey: r.wins_grey,
            wins_black: r.wins_black,
            wins_combo: r.wins_combo,
            total_moves: r.total_moves,
            game_lengths: r.game_lengths,
            decisive_lengths: r.decisive_lengths,
            full_search_turns: r.full_search_turns,
            total_turns: r.total_turns,
            isolation_captures: r.isolation_captures,
            jump_captures: r.jump_captures,
            sample_board_data: r.sample_board_data,
            top1_visit_fraction_mean: r.top1_visit_fraction_mean,
            top1_visit_fraction_std: r.top1_visit_fraction_std,
            search_depth_mean: r.search_depth_mean,
            search_depth_std: r.search_depth_std,
            valid_moves_mean: r.valid_moves_mean,
            valid_moves_std: r.valid_moves_std,
        })
    }

    /// Run a battle between two models. Games 0..N/2 have model1 as P1, model2 as P2.
    /// Games N/2..N are reversed. Returns win/draw counts from model1's perspective.
    ///
    /// Each model can independently use the Python eval_fn (sync) or a Rust-native
    /// pipelined ORT worker (when onnx_path is set). With both models on the
    /// pipelined path, their forwards run concurrently on independent CUDA streams,
    /// and play_batch_size is halved to keep per-game in-flight virtual-loss
    /// leaves constant across the two pipeline stages.
    #[pyo3(signature = (
        eval_fn1=None, eval_fn2=None,
        progress_fn=None,
        onnx_path1=None, onnx_path2=None,
    ))]
    fn play_battle(
        &self,
        _py: Python,
        eval_fn1: Option<Py<PyAny>>,
        eval_fn2: Option<Py<PyAny>>,
        progress_fn: Option<Py<PyAny>>,
        onnx_path1: Option<String>,
        onnx_path2: Option<String>,
    ) -> PyResult<PyZertzBattleResult> {
        let progress_core = progress_fn.map(|pfn| {
            let pfn: Py<PyAny> = pfn;
            Box::new(move |finished: u32, total: u32, active: u32, total_moves: u32| {
                Python::attach(|py| {
                    pfn.bind(py).call1((finished, total, active, total_moves)).ok();
                    py.check_signals().ok();
                });
            }) as Box<dyn Fn(u32, u32, u32, u32) + Send + Sync>
        });

        // Spawn ORT workers up front for whichever side(s) opted into the
        // pipelined path. Workers must outlive their PipelinedEvaluator,
        // hence the explicit drop-order at the end.
        let worker1 = onnx_path1
            .map(|path| ZertzInferenceWorker::spawn(path, NUM_CHANNELS, GRID_SIZE, RESERVE_SIZE))
            .transpose()
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        let worker2 = onnx_path2
            .map(|path| ZertzInferenceWorker::spawn(path, NUM_CHANNELS, GRID_SIZE, RESERVE_SIZE))
            .transpose()
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        // Halve play_batch_size if either side is pipelined — same rationale
        // as play_games: two batches alive per game across the pipeline stages.
        let mut play_batch_size = self.play_batch_size;
        if (worker1.is_some() || worker2.is_some()) && play_batch_size > 1 {
            play_batch_size /= 2;
        }

        // Per-side sync evaluators (only built if no worker for that side).
        let mut sync_eval1: Option<SyncEvaluator> = if worker1.is_none() {
            let py_eval = eval_fn1.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "either eval_fn1 or onnx_path1 must be provided for model1",
                )
            })?;
            let core_eval: EvalFn = Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
                call_python_eval(&py_eval, boards, reserves, n)
            });
            Some(SyncEvaluator::new(core_eval))
        } else { None };
        let mut sync_eval2: Option<SyncEvaluator> = if worker2.is_none() {
            let py_eval = eval_fn2.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "either eval_fn2 or onnx_path2 must be provided for model2",
                )
            })?;
            let core_eval: EvalFn = Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
                call_python_eval(&py_eval, boards, reserves, n)
            });
            Some(SyncEvaluator::new(core_eval))
        } else { None };

        let mut pipelined_eval1 = worker1.as_ref().map(PipelinedEvaluator::new);
        let mut pipelined_eval2 = worker2.as_ref().map(PipelinedEvaluator::new);

        let result = {
            let evaluator1: &mut dyn BatchEvaluator = match (&mut pipelined_eval1, &mut sync_eval1) {
                (Some(p), _) => p,
                (None, Some(s)) => s,
                _ => unreachable!("missing evaluator for model1"),
            };
            let evaluator2: &mut dyn BatchEvaluator = match (&mut pipelined_eval2, &mut sync_eval2) {
                (Some(p), _) => p,
                (None, Some(s)) => s,
                _ => unreachable!("missing evaluator for model2"),
            };

            play_battle_core(
                self.num_games,
                self.simulations,
                self.c_puct,
                play_batch_size,
                self.temperature,
                self.temp_threshold,
                evaluator1,
                evaluator2,
                progress_core,
            )
        };

        drop(pipelined_eval1);
        drop(pipelined_eval2);
        let _t1 = worker1.map(|w| w.shutdown_and_drain_timing());
        let _t2 = worker2.map(|w| w.shutdown_and_drain_timing());

        let r = result.map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(PyZertzBattleResult {
            wins_model1: r.wins_model1,
            wins_model2: r.wins_model2,
            draws: r.draws,
            wins_white: r.wins_white,
            wins_grey: r.wins_grey,
            wins_black: r.wins_black,
            wins_combo: r.wins_combo,
            game_lengths: r.game_lengths,
        })
    }

    /// Battle the model against the heuristic alpha-beta bot in parallel.
    /// Each ply, model-turn games batch their MCTS leaves while bot-turn
    /// games resolve via alphabeta in parallel across rayon workers. Half
    /// the games have the model as P1, half as P2 — `wins_model1` counts
    /// the model's wins regardless of color.
    ///
    /// When `onnx_path` is set the model's NN runs through a dedicated ORT
    /// worker thread (pipelined: batch N+1 is submitted while the GPU
    /// finishes batch N), and `eval_fn` is ignored. Otherwise the Python
    /// `eval_fn` callback is wrapped in a `SyncEvaluator` and called inline.
    ///
    /// `bot_time_ms` is an optional per-call wall-clock budget for each
    /// alphabeta search (0 or None = depth-only). When set, iterative
    /// deepening polls the deadline inside the search and falls back to
    /// the deepest fully-completed iteration if the budget expires
    /// mid-iteration. `bot_depth` is still a hard upper bound.
    #[pyo3(signature = (eval_fn=None, bot_depth=3, progress_fn=None, bot_time_ms=None, onnx_path=None))]
    fn play_battle_vs_bot(
        &self,
        _py: Python,
        eval_fn: Option<Py<PyAny>>,
        bot_depth: u32,
        progress_fn: Option<Py<PyAny>>,
        bot_time_ms: Option<u64>,
        onnx_path: Option<String>,
    ) -> PyResult<PyZertzBattleResult> {
        let progress_core = progress_fn.map(|pfn| {
            let pfn: Py<PyAny> = pfn;
            Box::new(move |finished: u32, total: u32, active: u32, total_moves: u32| {
                Python::attach(|py| {
                    pfn.bind(py).call1((finished, total, active, total_moves)).ok();
                    py.check_signals().ok();
                });
            }) as Box<dyn Fn(u32, u32, u32, u32) + Send + Sync>
        });

        let bot_time_budget = bot_time_ms
            .filter(|&ms| ms > 0)
            .map(std::time::Duration::from_millis);

        let r = if let Some(path) = onnx_path {
            // Pipelined ORT path: a worker thread owns the engine; the inner
            // sim loop overlaps leaf selection for batch N+1 with the GPU
            // forward for batch N. Halve play_batch_size so per-game
            // in-flight virtual-loss leaves stay constant across the two
            // pipeline stages.
            let mut play_batch_size = self.play_batch_size;
            if play_batch_size > 1 {
                play_batch_size /= 2;
            }
            let worker = ZertzInferenceWorker::spawn(
                path,
                NUM_CHANNELS,
                GRID_SIZE,
                RESERVE_SIZE,
            )
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            let mut evaluator = PipelinedEvaluator::new(&worker);

            let result = play_battle_vs_bot_core(
                self.num_games,
                self.simulations,
                self.c_puct,
                play_batch_size,
                self.temperature,
                self.temp_threshold,
                bot_depth,
                bot_time_budget,
                &mut evaluator,
                progress_core,
            );

            drop(evaluator);
            let _timing = worker.shutdown_and_drain_timing();
            result.map_err(pyo3::exceptions::PyRuntimeError::new_err)?
        } else {
            let py_eval = eval_fn.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "eval_fn is required when onnx_path is not provided",
                )
            })?;
            let core_eval: EvalFn = Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
                call_python_eval(&py_eval, boards, reserves, n)
            });
            let mut evaluator = SyncEvaluator::new(core_eval);
            play_battle_vs_bot_core(
                self.num_games,
                self.simulations,
                self.c_puct,
                self.play_batch_size,
                self.temperature,
                self.temp_threshold,
                bot_depth,
                bot_time_budget,
                &mut evaluator,
                progress_core,
            )
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?
        };

        Ok(PyZertzBattleResult {
            wins_model1: r.wins_model1,
            wins_model2: r.wins_model2,
            draws: r.draws,
            wins_white: r.wins_white,
            wins_grey: r.wins_grey,
            wins_black: r.wins_black,
            wins_combo: r.wins_combo,
            game_lengths: r.game_lengths,
        })
    }

    /// Battle two models of possibly-different architectures. `style1` and
    /// `style2` are `"split"` (post-split-ply, 7-channel input) or
    /// `"joint"` (V1, pre-split-ply, 6-channel input). Each side may use a
    /// Python `eval_fn` (sync callback) or a Rust-native ORT worker driven
    /// by `onnx_path` — independent per side.
    ///
    /// Half the games run with model1 as P1, half with model1 as P2;
    /// `wins_model1` is the model1 count regardless of color.
    ///
    /// Compared to `play_battle`, this entry point trades pipelined ORT
    /// and warm tree rerooting for simplicity — every player turn cold-
    /// inits its MCTS. Acceptable because cross-version battle isn't a
    /// training hot path.
    #[pyo3(signature = (
        style1, style2,
        eval_fn1=None, eval_fn2=None,
        progress_fn=None,
        onnx_path1=None, onnx_path2=None,
    ))]
    fn play_battle_versioned(
        &self,
        _py: Python,
        style1: &str,
        style2: &str,
        eval_fn1: Option<Py<PyAny>>,
        eval_fn2: Option<Py<PyAny>>,
        progress_fn: Option<Py<PyAny>>,
        onnx_path1: Option<String>,
        onnx_path2: Option<String>,
    ) -> PyResult<PyZertzBattleResult> {
        fn parse_style(s: &str) -> PyResult<ModelStyle> {
            match s.to_ascii_lowercase().as_str() {
                "split" | "v2" | "new" => Ok(ModelStyle::Split),
                "joint" | "v1" | "old" => Ok(ModelStyle::Joint),
                other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown model style '{other}' — expected 'split' or 'joint'"
                ))),
            }
        }
        let s1 = parse_style(style1)?;
        let s2 = parse_style(style2)?;

        let progress_core = progress_fn.map(|pfn| {
            let pfn: Py<PyAny> = pfn;
            Box::new(move |finished: u32, total: u32, active: u32, total_moves: u32| {
                Python::attach(|py| {
                    pfn.bind(py).call1((finished, total, active, total_moves)).ok();
                    py.check_signals().ok();
                });
            }) as Box<dyn Fn(u32, u32, u32, u32) + Send + Sync>
        });

        // Spawn ORT workers per side that requested one. Channel count is
        // dictated by style (6 for V1, 7 for V2) — must match the .onnx.
        let worker1 = onnx_path1
            .map(|path| ZertzInferenceWorker::spawn(path, s1.board_channels(), GRID_SIZE, RESERVE_SIZE))
            .transpose()
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        let worker2 = onnx_path2
            .map(|path| ZertzInferenceWorker::spawn(path, s2.board_channels(), GRID_SIZE, RESERVE_SIZE))
            .transpose()
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        // Per-side sync evaluator (built only if the worker wasn't).
        let s1_channels = s1.board_channels();
        let s2_channels = s2.board_channels();
        let mut sync_eval1: Option<SyncEvaluator> = if worker1.is_none() {
            let py_eval = eval_fn1.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "either eval_fn1 or onnx_path1 must be provided for model1",
                )
            })?;
            let core_eval: EvalFn = Box::new(move |boards, reserves, n| {
                call_python_eval_n(&py_eval, boards, reserves, n, s1_channels)
            });
            Some(SyncEvaluator::new(core_eval))
        } else { None };
        let mut sync_eval2: Option<SyncEvaluator> = if worker2.is_none() {
            let py_eval = eval_fn2.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "either eval_fn2 or onnx_path2 must be provided for model2",
                )
            })?;
            let core_eval: EvalFn = Box::new(move |boards, reserves, n| {
                call_python_eval_n(&py_eval, boards, reserves, n, s2_channels)
            });
            Some(SyncEvaluator::new(core_eval))
        } else { None };

        let mut pipelined_eval1 = worker1.as_ref().map(PipelinedEvaluator::new);
        let mut pipelined_eval2 = worker2.as_ref().map(PipelinedEvaluator::new);

        let result = {
            let evaluator1: &mut dyn BatchEvaluator = match (&mut pipelined_eval1, &mut sync_eval1) {
                (Some(p), _) => p,
                (None, Some(s)) => s,
                _ => unreachable!("missing evaluator for model1"),
            };
            let evaluator2: &mut dyn BatchEvaluator = match (&mut pipelined_eval2, &mut sync_eval2) {
                (Some(p), _) => p,
                (None, Some(s)) => s,
                _ => unreachable!("missing evaluator for model2"),
            };
            play_battle_versioned_core(
                self.num_games,
                self.simulations,
                self.c_puct,
                self.play_batch_size,
                self.temperature,
                self.temp_threshold,
                s1,
                s2,
                evaluator1,
                evaluator2,
                progress_core,
            )
        };

        drop(pipelined_eval1);
        drop(pipelined_eval2);
        let _t1 = worker1.map(|w| w.shutdown_and_drain_timing());
        let _t2 = worker2.map(|w| w.shutdown_and_drain_timing());

        let r = result.map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(PyZertzBattleResult {
            wins_model1: r.wins_model1,
            wins_model2: r.wins_model2,
            draws: r.draws,
            wins_white: r.wins_white,
            wins_grey: r.wins_grey,
            wins_black: r.wins_black,
            wins_combo: r.wins_combo,
            game_lengths: r.game_lengths,
        })
    }
}

// Move string utilities live in zertz_game::notation.

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
        (PyArray1::from_vec(py, board_buf), PyArray1::from_vec(py, reserve_buf))
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
        let py_eval = eval_fn.clone().unbind();
        let core_eval = Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
            call_python_eval(&py_eval, boards, reserves, n)
        });

        let best = best_move_core(&self.board, simulations, c_puct, core_eval)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        py.check_signals()?;
        Ok(move_to_str(best))
    }

    /// Heuristic-only alpha-beta search to `depth` plies. No NN required.
    /// Returns the chosen move as a notation string.
    #[pyo3(signature = (depth=3))]
    fn best_move_alphabeta(&self, depth: u32) -> PyResult<String> {
        let mv = zertz_game::alphabeta::alphabeta_best_move(&self.board, depth);
        Ok(move_to_str(mv))
    }

    /// Win classification for the current outcome — one of "white", "grey",
    /// "black", "combo" if a player has won, or `None` if the game is
    /// ongoing or drawn. Mirrors `WinType` produced by the self-play /
    /// battle loops so battle scripts can report the same color split.
    fn win_type(&self) -> Option<&'static str> {
        let winner = match self.board.outcome() {
            Outcome::WonBy(p) => p,
            _ => return None,
        };
        Some(match classify_win(&self.board, winner) {
            WinType::FourWhite => "white",
            WinType::FiveGrey => "grey",
            WinType::SixBlack => "black",
            WinType::ThreeEach => "combo",
            WinType::Draw => "draw",
        })
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
        PyArray1::from_owned_array(py, numpy::ndarray::Array1::from(perm))
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
        PyArray1::from_owned_array(py, numpy::ndarray::Array1::from(perm))
    }).collect()
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyZertzSelfPlaySession>()?;
    m.add_class::<PyZertzSelfPlayResult>()?;
    m.add_class::<PyZertzSearchStats>()?;
    m.add_class::<PyZertzBattleResult>()?;
    m.add_class::<PyZertzGame>()?;
    m.add("ZERTZ_POLICY_SIZE", NN_POLICY_SIZE)?;
    m.add("ZERTZ_NUM_CHANNELS", NUM_CHANNELS)?;
    m.add("ZERTZ_GRID_SIZE", GRID_SIZE)?;
    m.add("ZERTZ_RESERVE_SIZE", RESERVE_SIZE)?;
    m.add_function(wrap_pyfunction!(zertz_d6_grid_permutations, m)?)?;
    m.add_function(wrap_pyfunction!(zertz_d6_hex_permutations, m)?)?;
    Ok(())
}
