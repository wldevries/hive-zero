/// PyO3 Python bindings for YINSH game logic + MCTS best_move.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};

use core_game::game::{Game, NNGame, Outcome, Player};
use core_game::mcts::search::{CpuctStrategy, ForcedExploration, MctsSearch, RootNoise, SearchParams};
use yinsh_game::board::{YinshBoard, YinshMove};
use yinsh_game::board_encoding::{NUM_CHANNELS, RESERVE_SIZE};
use yinsh_game::hex::GRID_SIZE;
use yinsh_game::move_encoding::POLICY_SIZE;
use yinsh_game::notation::{move_to_str, str_to_move};

const BOARD_FLAT: usize = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;

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
        self.board.valid_moves()
            .into_iter()
            .map(|mv| move_to_str(&mv))
            .collect()
    }

    fn play(&mut self, move_str: &str) -> PyResult<()> {
        let mv: YinshMove = str_to_move(move_str)
            .map_err(|e| PyValueError::new_err(e))?;
        self.board.play_move(&mv)
            .map_err(|e| PyValueError::new_err(e))
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

    /// Encode current position as (board_tensor[9, 11, 11], reserve[6]) numpy arrays.
    fn encode<'py>(&self, py: Python<'py>) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>) {
        let mut board_buf = vec![0.0f32; BOARD_FLAT];
        let mut reserve_buf = vec![0.0f32; RESERVE_SIZE];
        self.board.encode_board(&mut board_buf, &mut reserve_buf);
        let board_arr = numpy::ndarray::Array2::from_shape_vec(
            (NUM_CHANNELS, GRID_SIZE * GRID_SIZE), board_buf,
        ).unwrap();
        let reserve_arr = numpy::ndarray::Array1::from(reserve_buf);
        (PyArray2::from_owned_array(py, board_arr), PyArray1::from_owned_array(py, reserve_arr))
    }

    /// Run MCTS and return (best_move_str, root_value).
    ///
    /// Each MCTS decision corresponds to a single sub-phase move:
    /// placing a ring, moving a ring, removing a row, or removing a ring.
    /// Same-player consecutive turns (row removal → ring removal, or deferred
    /// opponent rows) are handled by the shared MCTS backprop's player-boundary
    /// sign rule, so no game-specific logic is required here.
    #[pyo3(signature = (eval_fn, simulations=400, c_puct=1.5))]
    fn best_move(
        &mut self,
        py: Python<'_>,
        eval_fn: &Bound<'_, PyAny>,
        simulations: usize,
        c_puct: f32,
    ) -> PyResult<(String, f32)> {
        if self.board.is_game_over() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Game is already over"));
        }

        let params = SearchParams::new(
            CpuctStrategy::Constant { c_puct },
            ForcedExploration::None,
            RootNoise::None,
        );
        let mut search = MctsSearch::<YinshBoard>::new(simulations.max(1024) * 4);
        search.params = params;

        // Initial NN eval at root
        let mut board_enc = vec![0.0f32; BOARD_FLAT];
        let mut reserve_enc = vec![0.0f32; RESERVE_SIZE];
        self.board.encode_board(&mut board_enc, &mut reserve_enc);
        let (init_policy, _init_value) =
            infer_batch(py, eval_fn, &board_enc, &reserve_enc, 1)?;
        search.init(&self.board, &init_policy);

        // Simulation loop
        let mut sims_done = 0usize;
        while sims_done < simulations {
            let leaves = search.select_leaves(1);
            if leaves.is_empty() {
                sims_done += 1;
                continue;
            }
            let count = leaves.len();
            let mut leaf_boards = vec![0.0f32; count * BOARD_FLAT];
            let mut leaf_reserves = vec![0.0f32; count * RESERVE_SIZE];
            for (k, &leaf) in leaves.iter().enumerate() {
                let (b, r) = search.encode_leaf(leaf);
                leaf_boards[k * BOARD_FLAT..(k + 1) * BOARD_FLAT].copy_from_slice(&b);
                leaf_reserves[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE].copy_from_slice(&r);
            }
            let (policies_flat, values) =
                infer_batch(py, eval_fn, &leaf_boards, &leaf_reserves, count)?;
            let policies: Vec<Vec<f32>> = (0..count)
                .map(|i| policies_flat[i * POLICY_SIZE..(i + 1) * POLICY_SIZE].to_vec())
                .collect();
            search.expand_and_backprop(&policies, &values);
            sims_done += count;
        }

        let value = search.root_value();
        let best = search.best_move()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No legal moves"))?;
        Ok((move_to_str(&best), value))
    }
}

/// Call Python eval_fn(board[N, C, G, G], reserve[N, R]) → (policy[N, P], value[N]).
fn infer_batch(
    py: Python<'_>,
    eval_fn: &Bound<'_, PyAny>,
    boards_flat: &[f32],
    reserves_flat: &[f32],
    batch_size: usize,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let board_arr = numpy::ndarray::Array2::from_shape_vec(
        (batch_size, BOARD_FLAT), boards_flat.to_vec(),
    ).unwrap();
    let board_np = PyArray2::from_owned_array(py, board_arr);
    let board_4d = board_np.reshape([batch_size, NUM_CHANNELS, GRID_SIZE, GRID_SIZE])?;

    let reserve_arr = numpy::ndarray::Array2::from_shape_vec(
        (batch_size, RESERVE_SIZE), reserves_flat.to_vec(),
    ).unwrap();
    let reserve_np = PyArray2::from_owned_array(py, reserve_arr);

    let result = eval_fn.call1((board_4d, reserve_np))?;
    let tuple = result.cast::<PyTuple>().map_err(|_|
        pyo3::exceptions::PyRuntimeError::new_err("eval_fn must return (policy, value) tuple")
    )?;
    let policy: PyReadonlyArray2<f32> = tuple.get_item(0)?.extract()?;
    let value: PyReadonlyArray1<f32> = tuple.get_item(1)?.extract()?;
    Ok((
        policy.as_slice().unwrap().to_vec(),
        value.as_slice().unwrap().to_vec(),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyYinshGame>()?;
    m.add("YINSH_BOARD_SIZE", yinsh_game::hex::BOARD_SIZE)?;
    m.add("YINSH_GRID_SIZE", GRID_SIZE)?;
    m.add("YINSH_NUM_CHANNELS", NUM_CHANNELS)?;
    m.add("YINSH_RESERVE_SIZE", RESERVE_SIZE)?;
    m.add("YINSH_POLICY_SIZE", POLICY_SIZE)?;
    m.add("YINSH_NUM_POLICY_CHANNELS", yinsh_game::move_encoding::NUM_POLICY_CHANNELS)?;
    Ok(())
}
