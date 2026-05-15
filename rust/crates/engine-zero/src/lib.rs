pub mod ort_common;

pub mod hive;
pub mod tictactoe;
pub mod yinsh;
pub mod zertz;

use pyo3::prelude::*;

/// Python module: engine_zero
#[pymodule]
fn engine_zero(m: &Bound<'_, PyModule>) -> PyResult<()> {
    hive::python::register(m)?;
    hive::selfplay::register(m)?;
    zertz::python::register(m)?;
    tictactoe::python::register(m)?;
    yinsh::python::register(m)?;
    Ok(())
}
