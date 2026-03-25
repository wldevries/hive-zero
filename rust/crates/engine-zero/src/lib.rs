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
