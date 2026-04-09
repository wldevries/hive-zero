pub mod inference;
pub mod hive_python;
pub mod hive_selfplay;
pub mod zertz_python;
pub mod zertz_core;
pub mod tictactoe_python;

use pyo3::prelude::*;

/// Python module: engine_zero
#[pymodule]
fn engine_zero(m: &Bound<'_, PyModule>) -> PyResult<()> {
    hive_python::register(m)?;
    hive_selfplay::register(m)?;
    zertz_python::register(m)?;
    tictactoe_python::register(m)?;
    Ok(())
}
