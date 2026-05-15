//! PyO3 bindings for YINSH: interactive game, parallel self-play session,
//! battle harness, and D6 symmetry permutation tables for Python-side
//! data augmentation.

use std::collections::HashMap;
use std::sync::mpsc::Receiver;
use std::time::Instant;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use core_game::game::{Game, NNGame, Outcome, Player};
use core_game::symmetry::{D6Symmetry, Symmetry};

use yinsh_game::board::{YinshBoard, YinshMove};
use yinsh_game::board_encoding::{NUM_CHANNELS, RESERVE_SIZE};
use yinsh_game::hex::{ALL_CELLS, BOARD_SIZE, DIRECTIONS, GRID_SIZE, ROW_DIRS, is_valid_i8};
use yinsh_game::move_encoding::{NUM_POLICY_CHANNELS, POLICY_SIZE, encode_move as encode_move_to_policy, get_legal_move_mask};
use core_game::game::PolicyIndex;
use yinsh_game::notation::{move_to_str, str_to_move};
use yinsh_game::sgf::parse_sgf_move_strs;
use yinsh_game::search::{
    BatchEvaluator, BattleResult, EvalFn, ProgressFn, RequestId, SelfPlayResult, SyncEvaluator,
    best_move_core, play_battle_core, play_battle_vs_bot_core, play_selfplay_core,
};

const BOARD_FLAT: usize = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;
const SENTINEL: i64 = (GRID_SIZE * GRID_SIZE) as i64; // out-of-board grid index

// ---------------------------------------------------------------------------
// Python eval callback adapter
// ---------------------------------------------------------------------------

/// Call `eval_fn(boards[N, C, H, W], reserves[N, R]) -> (policy[N, P], wdl[N, 3])`.
/// Returns `(policy, values, draws)` where `values[i] = W-L` and `draws[i] = D`.
fn call_python_eval(
    eval_fn: &Py<PyAny>,
    boards: &[f32],
    reserves: &[f32],
    batch_size: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    Python::attach(|py| {
        let board_arr = numpy::ndarray::Array2::from_shape_vec(
            (batch_size, BOARD_FLAT),
            boards.to_vec(),
        )
        .map_err(|e| e.to_string())?;
        let board_np = PyArray2::from_owned_array(py, board_arr);
        let board_4d = board_np
            .reshape([batch_size, NUM_CHANNELS, GRID_SIZE, GRID_SIZE])
            .map_err(|e| e.to_string())?;

        let reserve_arr = numpy::ndarray::Array2::from_shape_vec(
            (batch_size, RESERVE_SIZE),
            reserves.to_vec(),
        )
        .map_err(|e| e.to_string())?;
        let reserve_np = PyArray2::from_owned_array(py, reserve_arr);

        let result = eval_fn
            .bind(py)
            .call1((board_4d, reserve_np))
            .map_err(|e| e.to_string())?;
        let tuple = result
            .cast::<PyTuple>()
            .map_err(|_| "eval_fn must return (policy, wdl) tuple".to_string())?;

        let policy = tuple
            .get_item(0)
            .map_err(|e| e.to_string())?
            .cast::<PyArray2<f32>>()
            .map_err(|e| e.to_string())?
            .readonly();
        let wdl = tuple
            .get_item(1)
            .map_err(|e| e.to_string())?
            .cast::<PyArray2<f32>>()
            .map_err(|e| e.to_string())?
            .readonly();
        let wdl_slice = wdl.as_slice().map_err(|e| e.to_string())?;

        let mut values = Vec::with_capacity(batch_size);
        let mut draws = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            values.push(wdl_slice[i * 3] - wdl_slice[i * 3 + 2]); // W - L
            draws.push(wdl_slice[i * 3 + 1]);                       // D
        }

        Ok((
            policy.as_slice().map_err(|e| e.to_string())?.to_vec(),
            values,
            draws,
        ))
    })
}

// ---------------------------------------------------------------------------
// Pipelined evaluator — hands batches off to a YinshInferenceWorker so the
// GPU forward for batch N runs while the main thread selects leaves for
// batch N+1.
// ---------------------------------------------------------------------------

/// `BatchEvaluator` impl that hands batches off to a `YinshInferenceWorker`
/// running on a dedicated thread. `submit` returns immediately after posting
/// the request (one batch in flight at a time, bounded by the worker's
/// channel capacity); `collect` blocks on the per-request response channel.
struct PipelinedEvaluator<'w> {
    worker: &'w super::inference::YinshInferenceWorker,
    next_id: RequestId,
    pending: HashMap<RequestId, Receiver<Result<(Vec<f32>, Vec<f32>, Vec<f32>), String>>>,
}

impl<'w> PipelinedEvaluator<'w> {
    fn new(worker: &'w super::inference::YinshInferenceWorker) -> Self {
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
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let recv = self
            .pending
            .remove(&id)
            .ok_or_else(|| format!("PipelinedEvaluator: unknown request {id}"))?;
        recv
            .recv()
            .map_err(|_| "yinsh-ort-worker died before responding".to_string())?
    }
}

// ---------------------------------------------------------------------------
// Self-play result
// ---------------------------------------------------------------------------

#[pyclass(name = "YinshSearchStats", skip_from_py_object)]
#[derive(Clone)]
pub struct PyYinshSearchStats {
    #[pyo3(get)] pub top1_mean: f32,
    #[pyo3(get)] pub top1_std: f32,
    #[pyo3(get)] pub depth_mean: f32,
    #[pyo3(get)] pub depth_std: f32,
    #[pyo3(get)] pub valid_moves_mean: f32,
    #[pyo3(get)] pub valid_moves_std: f32,
}

/// Per-phase wall-clock breakdown of one self-play session, in seconds.
/// `eval` covers the eval_fn boundary; on the ORT path it splits into
/// `eval_input` + `eval_run` + `eval_extract`, all zero on the Python-callback
/// path. Diagnostic only — gated by `--show-timing` on the Python side.
#[pyclass(name = "YinshTiming", skip_from_py_object)]
#[derive(Clone)]
pub struct PyYinshTiming {
    #[pyo3(get)] pub select_s: f64,
    #[pyo3(get)] pub encode_s: f64,
    #[pyo3(get)] pub eval_s: f64,
    #[pyo3(get)] pub expand_s: f64,
    #[pyo3(get)] pub eval_input_s: f64,
    #[pyo3(get)] pub eval_run_s: f64,
    #[pyo3(get)] pub eval_extract_s: f64,
}

#[pyclass(name = "YinshSelfPlayResult")]
pub struct PyYinshSelfPlayResult {
    inner: SelfPlayResult,
}

#[pymethods]
impl PyYinshSelfPlayResult {
    /// Returns `(boards[N, C*H*W], reserves[N, R], policies[N, P], values[N],
    /// root_q[N], value_only_flags, phase_flags)`. `root_q[i]` is the MCTS
    /// root W−L (no contempt) at the turn position `i` was played, in that
    /// turn's player's perspective. Used for q-target value mixing.
    fn training_data<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Vec<bool>,
        Vec<u8>,
    ) {
        let n = self.inner.num_samples;
        let boards = numpy::ndarray::Array2::from_shape_vec(
            (n, BOARD_FLAT),
            self.inner.board_data.clone(),
        )
        .unwrap();
        let reserves = numpy::ndarray::Array2::from_shape_vec(
            (n, RESERVE_SIZE),
            self.inner.reserve_data.clone(),
        )
        .unwrap();
        let policies = numpy::ndarray::Array2::from_shape_vec(
            (n, POLICY_SIZE),
            self.inner.policy_data.clone(),
        )
        .unwrap();
        let values = numpy::ndarray::Array1::from(self.inner.value_targets.clone());
        let root_q = numpy::ndarray::Array1::from(self.inner.root_q_targets.clone());
        (
            PyArray2::from_owned_array(py, boards),
            PyArray2::from_owned_array(py, reserves),
            PyArray2::from_owned_array(py, policies),
            PyArray1::from_owned_array(py, values),
            PyArray1::from_owned_array(py, root_q),
            self.inner.value_only_flags.clone(),
            self.inner.phase_flags.clone(),
        )
    }

    #[getter] fn num_samples(&self) -> usize { self.inner.num_samples }
    #[getter] fn wins_p1(&self) -> u32 { self.inner.wins_p1 }
    #[getter] fn wins_p2(&self) -> u32 { self.inner.wins_p2 }
    #[getter] fn draws(&self) -> u32 { self.inner.draws }
    #[getter] fn timeouts(&self) -> u32 { self.inner.timeouts }
    #[getter] fn total_moves(&self) -> u32 { self.inner.total_moves }
    #[getter] fn game_lengths(&self) -> Vec<u32> { self.inner.game_lengths.clone() }
    #[getter] fn decisive_lengths(&self) -> Vec<u32> { self.inner.decisive_lengths.clone() }
    #[getter] fn full_search_turns(&self) -> u32 { self.inner.full_search_turns }
    #[getter] fn total_turns(&self) -> u32 { self.inner.total_turns }

    /// Per-phase wall-clock breakdown of the inner sim loop. See
    /// `YinshTiming` for individual fields. Diagnostic only.
    #[getter]
    fn timing(&self) -> PyYinshTiming {
        let t = &self.inner.timing;
        PyYinshTiming {
            select_s: t.select.as_secs_f64(),
            encode_s: t.encode.as_secs_f64(),
            eval_s: t.eval.as_secs_f64(),
            expand_s: t.expand.as_secs_f64(),
            eval_input_s: t.eval_input.as_secs_f64(),
            eval_run_s: t.eval_run.as_secs_f64(),
            eval_extract_s: t.eval_extract.as_secs_f64(),
        }
    }

    /// Returns list of (label, board_string) for up to 2 decisive games.
    fn sample_boards(&self) -> Vec<(String, String)> { self.inner.sample_board_data.clone() }
    #[getter]
    fn search_stats(&self) -> PyYinshSearchStats {
        PyYinshSearchStats {
            top1_mean: self.inner.top1_visit_fraction_mean,
            top1_std: self.inner.top1_visit_fraction_std,
            depth_mean: self.inner.search_depth_mean,
            depth_std: self.inner.search_depth_std,
            valid_moves_mean: self.inner.valid_moves_mean,
            valid_moves_std: self.inner.valid_moves_std,
        }
    }
}

// ---------------------------------------------------------------------------
// Battle result
// ---------------------------------------------------------------------------

#[pyclass(name = "YinshBattleResult")]
pub struct PyYinshBattleResult {
    inner: BattleResult,
}

#[pymethods]
impl PyYinshBattleResult {
    #[getter] fn wins_model1(&self) -> u32 { self.inner.wins_model1 }
    #[getter] fn wins_model2(&self) -> u32 { self.inner.wins_model2 }
    #[getter] fn draws(&self) -> u32 { self.inner.draws }
    #[getter] fn wins_white(&self) -> u32 { self.inner.wins_white }
    #[getter] fn wins_black(&self) -> u32 { self.inner.wins_black }
    #[getter] fn game_lengths(&self) -> Vec<u32> { self.inner.game_lengths.clone() }
}

// ---------------------------------------------------------------------------
// Self-play session
// ---------------------------------------------------------------------------

#[pyclass(name = "YinshSelfPlaySession")]
pub struct PyYinshSelfPlaySession {
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
    draw_contempt: f32,
    asymmetric_contempt: bool,
    random_opening_moves_min: u32,
    random_opening_moves_max: u32,
    forced_playouts: bool,
}

#[pymethods]
impl PyYinshSelfPlaySession {
    #[new]
    #[pyo3(signature = (
        num_games,
        simulations = 200,
        temperature = 1.0,
        temp_threshold = 20,
        c_puct = 1.5,
        dir_alpha = 0.3,
        dir_epsilon = 0.25,
        play_batch_size = 8,
        playout_cap_p = 0.0,
        fast_cap = 30,
        draw_contempt = 0.0,
        asymmetric_contempt = false,
        random_opening_moves_min = 0,
        random_opening_moves_max = 0,
        forced_playouts = false,
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
        draw_contempt: f32,
        asymmetric_contempt: bool,
        random_opening_moves_min: u32,
        random_opening_moves_max: u32,
        forced_playouts: bool,
    ) -> Self {
        Self {
            num_games,
            simulations,
            temperature,
            temp_threshold,
            c_puct,
            dir_alpha,
            dir_epsilon,
            play_batch_size,
            playout_cap_p,
            fast_cap,
            draw_contempt,
            asymmetric_contempt,
            random_opening_moves_min,
            random_opening_moves_max,
            forced_playouts,
        }
    }

    /// Run self-play with either a Python eval callback or an ONNX model file.
    ///
    /// `opening_sequences`, if provided, is a per-game list of canonical Yinsh
    /// notation strings replayed verbatim before MCTS takes over. An empty
    /// inner list (or a missing entry) falls back to `random_opening_moves`
    /// for that game.
    #[pyo3(signature = (eval_fn=None, progress_fn=None, onnx_path=None, opening_sequences=None))]
    fn play_games(
        &self,
        _py: Python<'_>,
        eval_fn: Option<&Bound<'_, PyAny>>,
        progress_fn: Option<&Bound<'_, PyAny>>,
        onnx_path: Option<String>,
        opening_sequences: Option<Vec<Vec<String>>>,
    ) -> PyResult<PyYinshSelfPlayResult> {
        let progress_core: Option<ProgressFn> = progress_fn.map(|pfn| {
            let pfn = pfn.clone().unbind();
            Box::new(move |finished: u32, total: u32, active: u32, total_moves: u32| {
                Python::attach(|py| {
                    pfn.bind(py).call1((finished, total, active, total_moves)).ok();
                    py.check_signals().ok();
                });
            }) as ProgressFn
        });

        let mut mcts = core_game::selfplay_config::MctsConfig {
            simulations: self.simulations,
            c_puct: self.c_puct,
            dir_alpha: self.dir_alpha,
            dir_epsilon: self.dir_epsilon,
            forced_playouts: self.forced_playouts,
            draw_contempt: self.draw_contempt,
            asymmetric_contempt: self.asymmetric_contempt,
            play_batch_size: self.play_batch_size,
            temperature: self.temperature,
            temp_threshold: self.temp_threshold,
        };
        let playout_cap = core_game::selfplay_config::PlayoutCapConfig {
            p: self.playout_cap_p,
            fast_cap: self.fast_cap,
        };
        let opening = core_game::selfplay_config::OpeningRandomConfig {
            min: self.random_opening_moves_min,
            max: self.random_opening_moves_max,
        };
        let opening_sequences = opening_sequences.unwrap_or_default();

        if let Some(path) = onnx_path {
            // Pipelined ORT path: a worker thread owns the YinshOrtEngine;
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

            let worker = super::inference::YinshInferenceWorker::spawn(path)
                .map_err(PyRuntimeError::new_err)?;
            let mut evaluator = PipelinedEvaluator::new(&worker);

            let t_total_start = Instant::now();
            let core_result = play_selfplay_core(
                self.num_games,
                mcts,
                playout_cap,
                opening,
                &mut evaluator,
                progress_core,
                opening_sequences,
            );
            let total = t_total_start.elapsed();

            // Release the evaluator's borrow before taking ownership of the
            // worker for shutdown.
            drop(evaluator);
            let timing = worker.shutdown_and_drain_timing();

            let mut result = core_result.map_err(PyRuntimeError::new_err)?;
            result.timing.eval = total;
            result.timing.eval_input = timing.t_input;
            result.timing.eval_run = timing.t_run;
            result.timing.eval_extract = timing.t_extract;

            Ok(PyYinshSelfPlayResult { inner: result })
        } else {
            // Sync path: Python eval callback wrapped in SyncEvaluator. The
            // GIL precludes pipelining; this preserves the previous
            // behaviour exactly.
            let py_eval = eval_fn
                .ok_or_else(|| PyValueError::new_err("eval_fn is required when onnx_path is not provided"))?
                .clone()
                .unbind();
            let core_eval: EvalFn = Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
                call_python_eval(&py_eval, boards, reserves, n)
            });
            let mut evaluator = SyncEvaluator::new(core_eval);
            let result = play_selfplay_core(
                self.num_games,
                mcts,
                playout_cap,
                opening,
                &mut evaluator,
                progress_core,
                opening_sequences,
            )
            .map_err(PyRuntimeError::new_err)?;
            Ok(PyYinshSelfPlayResult { inner: result })
        }
    }

    /// Run a battle between two Python eval functions.
    #[pyo3(signature = (eval_fn1, eval_fn2, progress_fn=None))]
    fn play_battle(
        &self,
        _py: Python<'_>,
        eval_fn1: Py<PyAny>,
        eval_fn2: Py<PyAny>,
        progress_fn: Option<Py<PyAny>>,
    ) -> PyResult<PyYinshBattleResult> {
        let core_eval1: EvalFn = Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
            call_python_eval(&eval_fn1, boards, reserves, n)
        });
        let core_eval2: EvalFn = Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
            call_python_eval(&eval_fn2, boards, reserves, n)
        });

        let progress_core: Option<ProgressFn> = progress_fn.map(|pfn| {
            Box::new(move |finished: u32, total: u32, active: u32, total_moves: u32| {
                Python::attach(|py| {
                    pfn.bind(py).call1((finished, total, active, total_moves)).ok();
                    py.check_signals().ok();
                });
            }) as ProgressFn
        });

        let result = play_battle_core(
            self.num_games,
            self.simulations,
            self.c_puct,
            self.play_batch_size,
            self.temperature,
            self.temp_threshold,
            core_eval1,
            core_eval2,
            progress_core,
        )
        .map_err(PyRuntimeError::new_err)?;

        Ok(PyYinshBattleResult { inner: result })
    }

    /// Battle the model against the heuristic alpha-beta bot in parallel.
    /// Each ply, model-turn games batch their MCTS leaves through `eval_fn`
    /// while bot-turn games resolve synchronously via alphabeta. Half the
    /// games have the model as P1, half as P2 — `wins_model1` counts the
    /// model's wins regardless of color.
    #[pyo3(signature = (eval_fn, bot_depth, progress_fn=None))]
    fn play_battle_vs_bot(
        &self,
        _py: Python<'_>,
        eval_fn: Py<PyAny>,
        bot_depth: u32,
        progress_fn: Option<Py<PyAny>>,
    ) -> PyResult<PyYinshBattleResult> {
        let core_eval: EvalFn = Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
            call_python_eval(&eval_fn, boards, reserves, n)
        });

        let progress_core: Option<ProgressFn> = progress_fn.map(|pfn| {
            Box::new(move |finished: u32, total: u32, active: u32, total_moves: u32| {
                Python::attach(|py| {
                    pfn.bind(py).call1((finished, total, active, total_moves)).ok();
                    py.check_signals().ok();
                });
            }) as ProgressFn
        });

        let result = play_battle_vs_bot_core(
            self.num_games,
            self.simulations,
            self.c_puct,
            self.play_batch_size,
            self.temperature,
            self.temp_threshold,
            bot_depth,
            core_eval,
            progress_core,
        )
        .map_err(PyRuntimeError::new_err)?;

        Ok(PyYinshBattleResult { inner: result })
    }
}

// ---------------------------------------------------------------------------
// Interactive game (used by REPL `yinsh play` and tests)
// ---------------------------------------------------------------------------

#[pyclass(name = "YinshGame")]
pub struct PyYinshGame {
    board: YinshBoard,
}

#[pymethods]
impl PyYinshGame {
    #[new]
    fn new() -> Self {
        Self { board: YinshBoard::new() }
    }

    fn valid_moves(&mut self) -> Vec<String> {
        self.board.valid_moves().into_iter().map(|mv| move_to_str(&mv)).collect()
    }

    fn play(&mut self, move_str: &str) -> PyResult<()> {
        let mv: YinshMove = str_to_move(move_str).map_err(PyValueError::new_err)?;
        self.board.play_move(&mv).map_err(PyValueError::new_err)
    }

    fn outcome(&self) -> &'static str {
        match self.board.outcome() {
            Outcome::Ongoing => "ongoing",
            Outcome::WonBy(Player::Player1) => "white",
            Outcome::WonBy(Player::Player2) => "black",
            Outcome::Draw => "draw",
        }
    }

    fn current_player(&self) -> &'static str {
        match self.board.next_player() {
            Player::Player1 => "white",
            Player::Player2 => "black",
        }
    }

    fn phase(&self) -> &'static str {
        use yinsh_game::board::Phase;
        match self.board.phase {
            Phase::Setup => "setup",
            Phase::Normal => "normal",
            Phase::ClaimRow => "claim_row",
        }
    }

    fn white_score(&self) -> u8 { self.board.white_score }
    fn black_score(&self) -> u8 { self.board.black_score }
    fn markers_in_pool(&self) -> u8 { self.board.markers_in_pool }

    /// Encode current position as `(board[C, H*W], reserve[R])` numpy arrays.
    fn encode<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>) {
        let mut board_buf = vec![0.0f32; BOARD_FLAT];
        let mut reserve_buf = vec![0.0f32; RESERVE_SIZE];
        self.board.encode_board(&mut board_buf, &mut reserve_buf);
        let board_arr = numpy::ndarray::Array2::from_shape_vec(
            (NUM_CHANNELS, GRID_SIZE * GRID_SIZE),
            board_buf,
        )
        .unwrap();
        let reserve_arr = numpy::ndarray::Array1::from(reserve_buf);
        (
            PyArray2::from_owned_array(py, board_arr),
            PyArray1::from_owned_array(py, reserve_arr),
        )
    }

    /// Return the 7139-cell legal-policy mask: 1.0 at every flat policy index
    /// referenced by a legal move at the current position, 0.0 elsewhere.
    /// Used by supervised pretrain to mark the legal action set in policy targets.
    fn legal_policy_mask<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let (mask, _) = get_legal_move_mask(&mut self.board);
        PyArray1::from_owned_array(py, numpy::ndarray::Array1::from(mask))
    }

    /// Encode a move-notation string into its flat policy indices.
    /// Returns 1 index for `PlaceRing`/`MoveRing`, 2 for `ClaimRow` (joint Sum).
    /// Errors if the string isn't a recognized move at the current position.
    fn encode_move(&mut self, move_str: &str) -> PyResult<Vec<usize>> {
        let mv = str_to_move(move_str).map_err(PyValueError::new_err)?;
        Ok(match encode_move_to_policy(&mv) {
            PolicyIndex::Single(i) => vec![i],
            PolicyIndex::Sum(a, b) => vec![a, b],
            PolicyIndex::DotProduct { .. } => Vec::new(),
        })
    }

    /// Run MCTS and return `(best_move_str, root_value)`.
    #[pyo3(signature = (eval_fn, simulations=400, c_puct=1.5))]
    fn best_move(
        &self,
        py: Python<'_>,
        eval_fn: &Bound<'_, PyAny>,
        simulations: usize,
        c_puct: f32,
    ) -> PyResult<String> {
        let py_eval = eval_fn.clone().unbind();
        let core_eval: EvalFn = Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
            call_python_eval(&py_eval, boards, reserves, n)
        });
        let best = best_move_core(&self.board, simulations, c_puct, core_eval)
            .map_err(PyRuntimeError::new_err)?;
        py.check_signals()?;
        Ok(move_to_str(&best))
    }

    /// Heuristic-only alpha-beta search to `depth` plies. No NN required.
    /// Returns the chosen move as a notation string.
    #[pyo3(signature = (depth=3))]
    fn best_move_alphabeta(&self, depth: u32) -> PyResult<String> {
        let mv = yinsh_game::alphabeta::alphabeta_best_move(&self.board, depth);
        Ok(move_to_str(&mv))
    }

    /// Return `[(move_str, score), ...]` for every legal root move at the
    /// given depth. Useful for inspecting what the bot is thinking about.
    #[pyo3(signature = (depth=2))]
    fn alphabeta_root_scores(&self, depth: u32) -> PyResult<Vec<(String, f32)>> {
        let scores = yinsh_game::alphabeta::alphabeta_root_scores(&self.board, depth);
        Ok(scores
            .into_iter()
            .map(|(mv, s)| (move_to_str(&mv), s))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// D6 symmetry permutation tables for Python-side data augmentation
// ---------------------------------------------------------------------------

/// Yinsh's `(col, row)` axes are non-orthogonal in a way that differs from the
/// standard `D6Symmetry::transform_hex` axial convention. Specifically, Yinsh's
/// 6 hex neighbor directions are
///   ±(0, 1), ±(1, 0), ±(1, 1),
/// whereas standard axial uses
///   ±(1, 0), ±(0, 1), ±(1, -1).
///
/// Convert via the basis change `q = col, r = row - col` so that `transform_hex`
/// can be applied directly, then convert back via `col = q, row = q + r`.
#[inline]
fn yinsh_to_axial(col: i32, row: i32) -> (i32, i32) {
    (col, row - col)
}

#[inline]
fn axial_to_yinsh(q: i32, r: i32) -> (i32, i32) {
    (q, q + r)
}

/// 12 D6 gather-permutation tables for the `GRID_SIZE x GRID_SIZE` flat grid.
/// Returns a list of 12 numpy arrays of length `GRID_SIZE * GRID_SIZE`.
/// `perm[new_grid_idx] = old_grid_idx`, or sentinel `GRID_SIZE * GRID_SIZE`
/// for new positions that are outside the 85-cell board (filled with zero).
#[pyfunction]
fn yinsh_d6_grid_permutations<'py>(py: Python<'py>) -> Vec<Bound<'py, PyArray1<i64>>> {
    let n = GRID_SIZE * GRID_SIZE;
    let center = GRID_SIZE as i32 / 2;

    D6Symmetry::all()
        .iter()
        .map(|sym| {
            let mut perm = vec![SENTINEL; n];
            for &(col, row) in ALL_CELLS.iter() {
                let (q, r) = yinsh_to_axial(col as i32 - center, row as i32 - center);
                let (q2, r2) = sym.transform_hex(q, r);
                let (col2_c, row2_c) = axial_to_yinsh(q2, r2);
                let col2 = col2_c + center;
                let row2 = row2_c + center;
                if col2 < 0 || col2 >= GRID_SIZE as i32 || row2 < 0 || row2 >= GRID_SIZE as i32 {
                    continue;
                }
                let src = row as usize * GRID_SIZE + col as usize;
                let dst = row2 as usize * GRID_SIZE + col2 as usize;
                perm[dst] = src as i64;
            }
            PyArray1::from_owned_array(py, numpy::ndarray::Array1::from(perm))
        })
        .collect()
}

/// 12 D6 row-direction permutation tables for Yinsh's 3 unsigned line directions
/// (used by the `ClaimRow` row-direction policy channels 1..=3).
/// Returns a list of 12 numpy arrays of length 3, where `perm[new_dir] = old_dir`.
#[pyfunction]
fn yinsh_d6_dir_permutations<'py>(py: Python<'py>) -> Vec<Bound<'py, PyArray1<i64>>> {
    // Build axial vectors for the 3 row dirs once.
    let row_axials: [(i32, i32); 3] = {
        let mut v = [(0i32, 0i32); 3];
        for (i, &(dc, dr)) in ROW_DIRS.iter().enumerate() {
            v[i] = yinsh_to_axial(dc as i32, dr as i32);
        }
        v
    };

    D6Symmetry::all()
        .iter()
        .map(|sym| {
            // For each output direction `new_d`, find the input `old_d` whose
            // transform under `sym` produces `new_d` (or its negation).
            let mut perm = vec![0i64; 3];
            // Compute forward map: old_d -> new_d.
            let mut forward = [0usize; 3];
            for old_d in 0..3 {
                let (q, r) = row_axials[old_d];
                let (q2, r2) = sym.transform_hex(q, r);
                let target = (q2, r2);
                let neg_target = (-q2, -r2);
                let mut found = false;
                for new_d in 0..3 {
                    if row_axials[new_d] == target || row_axials[new_d] == neg_target {
                        forward[old_d] = new_d;
                        found = true;
                        break;
                    }
                }
                debug_assert!(found, "yinsh row dir {} not preserved by sym {:?}", old_d, sym);
            }
            // Invert: perm[new_d] = old_d
            for old_d in 0..3 {
                perm[forward[old_d]] = old_d as i64;
            }
            PyArray1::from_owned_array(py, numpy::ndarray::Array1::from(perm))
        })
        .collect()
}

/// 12 D6 direction-permutation tables for all 6 DIRECTIONS (used by ring movement
/// policy channels). Returns 12 arrays of length 6: `perm[new_dir] = old_dir`.
#[pyfunction]
fn yinsh_d6_movement_dir_permutations<'py>(py: Python<'py>) -> Vec<Bound<'py, PyArray1<i64>>> {
    let dir_axials: [(i32, i32); 6] = {
        let mut v = [(0i32, 0i32); 6];
        for (i, &(dc, dr)) in DIRECTIONS.iter().enumerate() {
            v[i] = yinsh_to_axial(dc as i32, dr as i32);
        }
        v
    };

    D6Symmetry::all()
        .iter()
        .map(|sym| {
            // Forward map: old_d -> new_d
            let mut forward = [0usize; 6];
            for old_d in 0..6 {
                let (q, r) = dir_axials[old_d];
                let (q2, r2) = sym.transform_hex(q, r);
                let target = (q2, r2);
                let new_d = dir_axials.iter().position(|&d| d == target)
                    .expect("direction not preserved by D6 symmetry");
                forward[old_d] = new_d;
            }
            // Invert: perm[new_d] = old_d
            let mut perm = vec![0i64; 6];
            for old_d in 0..6 {
                perm[forward[old_d]] = old_d as i64;
            }
            PyArray1::from_owned_array(py, numpy::ndarray::Array1::from(perm))
        })
        .collect()
}

/// Parse a Boardspace YINSH SGF and return the move sequence as canonical
/// notation strings ("E5", "E5 G5", "Claim E5 E6 E7 E8 E9 D3"). Pairs each
/// `RemoveRow` action with its following `RemoveRing` into a single `Claim`.
/// Mirrors `parse_sgf_moves` for Hive.
#[pyfunction]
fn parse_yinsh_sgf_moves(content: &str) -> PyResult<Vec<String>> {
    parse_sgf_move_strs(content).map_err(PyValueError::new_err)
}

/// Subset of D6 symmetries that map every Yinsh valid cell to another valid cell.
/// Returns the indices (0..12) into `D6Symmetry::all()` of the preserving symmetries.
/// Used by Python-side augmentation to skip transforms that would inject phantom
/// off-board positions.
#[pyfunction]
fn yinsh_valid_d6_indices() -> Vec<usize> {
    let center = GRID_SIZE as i32 / 2;
    let mut valid: Vec<usize> = Vec::new();
    for (i, sym) in D6Symmetry::all().iter().enumerate() {
        let mut all_ok = true;
        for &(col, row) in ALL_CELLS.iter() {
            let (q, r) = yinsh_to_axial(col as i32 - center, row as i32 - center);
            let (q2, r2) = sym.transform_hex(q, r);
            let (col2_c, row2_c) = axial_to_yinsh(q2, r2);
            let col2 = col2_c + center;
            let row2 = row2_c + center;
            if !is_valid_i8(col2 as i8, row2 as i8) {
                all_ok = false;
                break;
            }
        }
        if all_ok {
            valid.push(i);
        }
    }
    valid
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyYinshGame>()?;
    m.add_class::<PyYinshSelfPlaySession>()?;
    m.add_class::<PyYinshSelfPlayResult>()?;
    m.add_class::<PyYinshSearchStats>()?;
    m.add_class::<PyYinshTiming>()?;
    m.add_class::<PyYinshBattleResult>()?;
    m.add("YINSH_BOARD_SIZE", BOARD_SIZE)?;
    m.add("YINSH_GRID_SIZE", GRID_SIZE)?;
    m.add("YINSH_NUM_CHANNELS", NUM_CHANNELS)?;
    m.add("YINSH_RESERVE_SIZE", RESERVE_SIZE)?;
    m.add("YINSH_POLICY_SIZE", POLICY_SIZE)?;
    m.add("YINSH_NUM_POLICY_CHANNELS", NUM_POLICY_CHANNELS)?;
    m.add_function(wrap_pyfunction!(yinsh_d6_grid_permutations, m)?)?;
    m.add_function(wrap_pyfunction!(yinsh_d6_dir_permutations, m)?)?;
    m.add_function(wrap_pyfunction!(yinsh_d6_movement_dir_permutations, m)?)?;
    m.add_function(wrap_pyfunction!(yinsh_valid_d6_indices, m)?)?;
    m.add_function(wrap_pyfunction!(parse_yinsh_sgf_moves, m)?)?;
    Ok(())
}
