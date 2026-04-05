pub mod inference;
pub mod python;
pub mod hive_selfplay;
pub mod zertz_python;

use pyo3::prelude::*;

/// Python module: hive_engine
#[pymodule]
fn hive_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)?;
    hive_selfplay::register(m)?;
    zertz_python::register(m)?;
    Ok(())
}
