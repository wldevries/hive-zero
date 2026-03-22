pub mod hex;
pub mod piece;
pub mod board;
pub mod rules;
pub mod game;
pub mod uhp;
pub mod board_encoding;
pub mod move_encoding;
pub mod mcts;
pub mod sgf;
pub mod python;
pub mod selfplay;

use pyo3::prelude::*;

/// Python module: hive_engine
#[pymodule]
fn hive_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)?;
    selfplay::register(m)?;
    Ok(())
}
