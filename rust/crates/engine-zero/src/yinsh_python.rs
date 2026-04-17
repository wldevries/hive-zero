//! PyO3 bindings for YINSH: interactive game, parallel self-play session,
//! battle harness, and D6 symmetry permutation tables for Python-side
//! data augmentation.

use std::sync::{Arc, Mutex};

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use core_game::game::{Game, NNGame, Outcome, Player};
use core_game::symmetry::{D6Symmetry, Symmetry};

use yinsh_game::board::{YinshBoard, YinshMove};
use yinsh_game::board_encoding::{NUM_CHANNELS, RESERVE_SIZE};
use yinsh_game::hex::{ALL_CELLS, BOARD_SIZE, GRID_SIZE, ROW_DIRS, is_valid_i8};
use yinsh_game::move_encoding::{NUM_POLICY_CHANNELS, POLICY_SIZE};
use yinsh_game::notation::{move_to_str, str_to_move};
use yinsh_game::search::{
    BattleResult, EvalFn, ProgressFn, SelfPlayResult, best_move_core, play_battle_core,
    play_selfplay_core,
};

const BOARD_FLAT: usize = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;
const SENTINEL: i64 = (GRID_SIZE * GRID_SIZE) as i64; // out-of-board grid index

// ---------------------------------------------------------------------------
// Python eval callback adapter
// ---------------------------------------------------------------------------

/// Call `eval_fn(boards[N, C, H, W], reserves[N, R]) -> (policy[N, P], value[N])`.
fn call_python_eval(
    eval_fn: &Py<PyAny>,
    boards: &[f32],
    reserves: &[f32],
    batch_size: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
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
            .map_err(|_| "eval_fn must return (policy, value) tuple".to_string())?;

        let policy = tuple
            .get_item(0)
            .map_err(|e| e.to_string())?
            .cast::<PyArray2<f32>>()
            .map_err(|e| e.to_string())?
            .readonly();
        let value = tuple
            .get_item(1)
            .map_err(|e| e.to_string())?
            .cast::<PyArray1<f32>>()
            .map_err(|e| e.to_string())?
            .readonly();

        Ok((
            policy.as_slice().map_err(|e| e.to_string())?.to_vec(),
            value.as_slice().map_err(|e| e.to_string())?.to_vec(),
        ))
    })
}

// ---------------------------------------------------------------------------
// Self-play result
// ---------------------------------------------------------------------------

#[pyclass(name = "YinshSelfPlayResult")]
pub struct PyYinshSelfPlayResult {
    inner: SelfPlayResult,
}

#[pymethods]
impl PyYinshSelfPlayResult {
    /// Returns `(boards[N, C*H*W], reserves[N, R], policies[N, P], values[N],
    /// value_only_flags, phase_flags)`.
    fn training_data<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
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
        (
            PyArray2::from_owned_array(py, boards),
            PyArray2::from_owned_array(py, reserves),
            PyArray2::from_owned_array(py, policies),
            PyArray1::from_owned_array(py, values),
            self.inner.value_only_flags.clone(),
            self.inner.phase_flags.clone(),
        )
    }

    #[getter] fn num_samples(&self) -> usize { self.inner.num_samples }
    #[getter] fn wins_p1(&self) -> u32 { self.inner.wins_p1 }
    #[getter] fn wins_p2(&self) -> u32 { self.inner.wins_p2 }
    #[getter] fn draws(&self) -> u32 { self.inner.draws }
    #[getter] fn total_moves(&self) -> u32 { self.inner.total_moves }
    #[getter] fn game_lengths(&self) -> Vec<u32> { self.inner.game_lengths.clone() }
    #[getter] fn decisive_lengths(&self) -> Vec<u32> { self.inner.decisive_lengths.clone() }
    #[getter] fn full_search_turns(&self) -> u32 { self.inner.full_search_turns }
    #[getter] fn total_turns(&self) -> u32 { self.inner.total_turns }
    fn sample_summaries(&self) -> Vec<String> { self.inner.sample_summaries.clone() }
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
impl PyYinshSelfPlaySession {
    #[new]
    #[pyo3(signature = (
        num_games,
        simulations = 200,
        max_moves = 400,
        temperature = 1.0,
        temp_threshold = 20,
        c_puct = 1.5,
        dir_alpha = 0.3,
        dir_epsilon = 0.25,
        play_batch_size = 8,
        playout_cap_p = 0.0,
        fast_cap = 30,
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
        Self {
            num_games,
            simulations,
            max_moves,
            temperature,
            temp_threshold,
            c_puct,
            dir_alpha,
            dir_epsilon,
            play_batch_size,
            playout_cap_p,
            fast_cap,
        }
    }

    /// Run self-play with either a Python eval callback or an ONNX model file.
    #[pyo3(signature = (eval_fn=None, progress_fn=None, onnx_path=None))]
    fn play_games(
        &self,
        _py: Python<'_>,
        eval_fn: Option<&Bound<'_, PyAny>>,
        progress_fn: Option<&Bound<'_, PyAny>>,
        onnx_path: Option<String>,
    ) -> PyResult<PyYinshSelfPlayResult> {
        let core_eval: EvalFn = if let Some(path) = onnx_path {
            let engine = crate::inference::YinshOrtEngine::load(&path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let shared = Arc::new(Mutex::new(engine));
            Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
                let mut engine = shared.lock().map_err(|_| "ONNX engine mutex poisoned".to_string())?;
                engine.infer_batch(boards, reserves, n)
            })
        } else {
            let py_eval = eval_fn
                .ok_or_else(|| PyValueError::new_err("eval_fn is required when onnx_path is not provided"))?
                .clone()
                .unbind();
            Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
                call_python_eval(&py_eval, boards, reserves, n)
            })
        };

        let progress_core: Option<ProgressFn> = progress_fn.map(|pfn| {
            let pfn = pfn.clone().unbind();
            Box::new(move |finished: u32, total: u32, active: u32, total_moves: u32| {
                Python::attach(|py| {
                    pfn.bind(py).call1((finished, total, active, total_moves)).ok();
                    py.check_signals().ok();
                });
            }) as ProgressFn
        });

        let result = play_selfplay_core(
            self.num_games,
            self.simulations,
            self.max_moves,
            self.temperature,
            self.temp_threshold,
            self.c_puct,
            self.dir_alpha,
            self.dir_epsilon,
            self.play_batch_size,
            self.playout_cap_p,
            self.fast_cap,
            core_eval,
            progress_core,
        )
        .map_err(PyRuntimeError::new_err)?;

        Ok(PyYinshSelfPlayResult { inner: result })
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
            self.max_moves,
            self.c_puct,
            self.play_batch_size,
            core_eval1,
            core_eval2,
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
            Phase::RemoveRow => "remove_row",
            Phase::RemoveRing => "remove_ring",
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
/// (used by the `RemoveRow` policy channels 3..6).
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
    m.add_class::<PyYinshBattleResult>()?;
    m.add("YINSH_BOARD_SIZE", BOARD_SIZE)?;
    m.add("YINSH_GRID_SIZE", GRID_SIZE)?;
    m.add("YINSH_NUM_CHANNELS", NUM_CHANNELS)?;
    m.add("YINSH_RESERVE_SIZE", RESERVE_SIZE)?;
    m.add("YINSH_POLICY_SIZE", POLICY_SIZE)?;
    m.add("YINSH_NUM_POLICY_CHANNELS", NUM_POLICY_CHANNELS)?;
    m.add_function(wrap_pyfunction!(yinsh_d6_grid_permutations, m)?)?;
    m.add_function(wrap_pyfunction!(yinsh_d6_dir_permutations, m)?)?;
    m.add_function(wrap_pyfunction!(yinsh_valid_d6_indices, m)?)?;
    Ok(())
}
