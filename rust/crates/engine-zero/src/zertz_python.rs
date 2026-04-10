/// PyO3 Python bindings for Zertz self-play. v2

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use std::sync::{Arc, Mutex};

use zertz_game::board_encoding::{encode_board, GRID_SIZE, NUM_CHANNELS, RESERVE_SIZE};
use zertz_game::move_encoding::POLICY_SIZE;
use zertz_game::notation::{move_to_str, str_to_move};
use zertz_game::zertz::ZertzBoard;
use core_game::game::{Game, Outcome, Player};
use crate::inference::ZertzInference;
use zertz_game::search::{best_move_core, play_battle_core, play_selfplay_core};

const BOARD_FLAT: usize = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;

fn call_python_eval(
    eval_fn: &Py<PyAny>,
    boards: &[f32],
    reserves: &[f32],
    batch_size: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
    Python::attach(|py| {
        let board_arr = numpy::ndarray::Array2::from_shape_vec(
            (batch_size, BOARD_FLAT),
            boards.to_vec(),
        ).map_err(|e| e.to_string())?;
        let board_np = PyArray2::from_owned_array(py, board_arr);
        let board_4d = board_np
            .reshape([batch_size, NUM_CHANNELS, GRID_SIZE, GRID_SIZE])
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
            .map_err(|_| "eval_fn must return (place, cap_source, cap_dest, value) tuple".to_string())?;

        let place = tuple
            .get_item(0)
            .map_err(|e| e.to_string())?
            .cast::<PyArray2<f32>>()
            .map_err(|e| e.to_string())?
            .readonly();
        let cap_source = tuple
            .get_item(1)
            .map_err(|e| e.to_string())?
            .cast::<PyArray2<f32>>()
            .map_err(|e| e.to_string())?
            .readonly();
        let cap_dest = tuple
            .get_item(2)
            .map_err(|e| e.to_string())?
            .cast::<PyArray2<f32>>()
            .map_err(|e| e.to_string())?
            .readonly();
        let value = tuple
            .get_item(3)
            .map_err(|e| e.to_string())?
            .cast::<PyArray1<f32>>()
            .map_err(|e| e.to_string())?
            .readonly();

        Ok((
            place.as_slice().map_err(|e| e.to_string())?.to_vec(),
            cap_source.as_slice().map_err(|e| e.to_string())?.to_vec(),
            cap_dest.as_slice().map_err(|e| e.to_string())?.to_vec(),
            value.as_slice().map_err(|e| e.to_string())?.to_vec(),
        ))
    })
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
    /// Returns (boards, reserves, policies, values, value_only, capture_turn, mid_capture_turn)
    fn training_data<'py>(&self, py: Python<'py>) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
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
        (
            PyArray2::from_owned_array(py, boards),
            PyArray2::from_owned_array(py, reserves),
            PyArray2::from_owned_array(py, policies),
            PyArray1::from_owned_array(py, values),
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
        _py: Python<'_>,
        eval_fn: Option<&Bound<'_, PyAny>>,
        progress_fn: Option<&Bound<'_, PyAny>>,
        onnx_path: Option<String>,
    ) -> PyResult<PyZertzSelfPlayResult> {
        let eval_core: zertz_game::search::EvalFn = if let Some(path) = onnx_path {
            let engine = crate::inference::ZertzOrtEngine::load(&path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let shared = Arc::new(Mutex::new(engine));
            Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
                let mut engine = shared.lock().map_err(|_| "ONNX engine mutex poisoned".to_string())?;
                let r = engine
                    .infer_batch(boards, reserves, n, NUM_CHANNELS, GRID_SIZE, RESERVE_SIZE)
                    .map_err(|e| e.to_string())?;
                Ok((r.place, r.cap_source, r.cap_dest, r.value))
            })
        } else {
            let py_eval = eval_fn
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("eval_fn is required when onnx_path is not provided"))?
                .clone()
                .unbind();
            Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
                call_python_eval(&py_eval, boards, reserves, n)
            })
        };

        let progress_core = progress_fn.map(|pfn| {
            let pfn = pfn.clone().unbind();
            Box::new(move |finished: u32, total: u32, active: u32, total_moves: u32| {
                Python::attach(|py| {
                    pfn.bind(py).call1((finished, total, active, total_moves)).ok();
                    py.check_signals().ok();
                });
            }) as Box<dyn Fn(u32, u32, u32, u32) + Send + Sync>
        });

        let r = play_selfplay_core(
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
            eval_core,
            progress_core,
        ).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        Ok(PyZertzSelfPlayResult {
            board_data: r.board_data,
            reserve_data: r.reserve_data,
            policy_data: r.policy_data,
            value_targets: r.value_targets,
            value_only_flags: r.value_only_flags,
            capture_turn_flags: r.capture_turn_flags,
            mid_capture_turn_flags: r.mid_capture_turn_flags,
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
        })
    }

    /// Run a battle between two models. Games 0..N/2 have model1 as P1, model2 as P2.
    /// Games N/2..N are reversed. Returns win/draw counts from model1's perspective.
    #[pyo3(signature = (eval_fn1, eval_fn2, progress_fn=None))]
    fn play_battle(
        &self,
        _py: Python,
        eval_fn1: Py<PyAny>,
        eval_fn2: Py<PyAny>,
        progress_fn: Option<Py<PyAny>>,
    ) -> PyResult<PyZertzBattleResult> {
        let core_eval1 = Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
            call_python_eval(&eval_fn1, boards, reserves, n)
        });
        let core_eval2 = Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
            call_python_eval(&eval_fn2, boards, reserves, n)
        });

        let progress_core = progress_fn.map(|pfn| {
            let pfn: Py<PyAny> = pfn;
            Box::new(move |finished: u32, total: u32, active: u32, total_moves: u32| {
                Python::attach(|py| {
                    pfn.bind(py).call1((finished, total, active, total_moves)).ok();
                    py.check_signals().ok();
                });
            }) as Box<dyn Fn(u32, u32, u32, u32) + Send + Sync>
        });

        let r = play_battle_core(
            self.num_games,
            self.simulations,
            self.max_moves,
            self.c_puct,
            self.play_batch_size,
            core_eval1,
            core_eval2,
            progress_core,
        ).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

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
