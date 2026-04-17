/// PyO3 Python bindings for YINSH (game logic only; no MCTS/self-play yet).

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use core_game::game::{Game, Outcome, Player};
use yinsh_game::board::{YinshBoard, YinshMove};
use yinsh_game::notation::{move_to_str, str_to_move};

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
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyYinshGame>()?;
    m.add("YINSH_BOARD_SIZE", yinsh_game::hex::BOARD_SIZE)?;
    m.add("YINSH_GRID_SIZE", yinsh_game::hex::GRID_SIZE)?;
    Ok(())
}
