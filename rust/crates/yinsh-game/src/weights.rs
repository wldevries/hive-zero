//! Plain-text load/save of the 6-element alphabeta weight vector.
//!
//! Format (one feature per line, comments starting with `#` ignored):
//!
//! ```text
//! # Yinsh alphabeta linear eval weights
//! SCORE     340.0784
//! ROW5      204.6750
//! ROW4       34.6309
//! ROW3       -7.5237
//! MOBILITY    3.9147
//! MARKER      3.4497
//! ```
//!
//! Lines must appear in the order above; the leading label is ignored on
//! parse (only the numeric value matters), but `save_weights` writes labels
//! for human readability.

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use yinsh_game::alphabeta::N_FEATURES;

const FEATURE_NAMES: [&str; N_FEATURES] = [
    "SCORE", "ROW5", "ROW4", "ROW3", "MOBILITY", "MARKER",
];

pub fn save_weights(path: &Path, weights: &[f32; N_FEATURES]) -> std::io::Result<()> {
    let mut f = BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "# Yinsh alphabeta linear eval weights")?;
    for (i, w) in weights.iter().enumerate() {
        writeln!(f, "{:<9} {:>12.4}", FEATURE_NAMES[i], w)?;
    }
    Ok(())
}

pub fn load_weights(path: &Path) -> Result<[f32; N_FEATURES], String> {
    let f = BufReader::new(
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?,
    );
    let mut values: Vec<f32> = Vec::new();
    for line in f.lines() {
        let line = line.map_err(|e| format!("read {}: {e}", path.display()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') { continue; }
        // Take the last whitespace-separated token as the float, so both
        // bare numbers and `LABEL value` lines parse the same.
        let tok = trimmed.split_whitespace().last().unwrap();
        let v: f32 = tok.parse().map_err(|_| format!(
            "{}: cannot parse '{}' as float", path.display(), tok
        ))?;
        values.push(v);
    }
    if values.len() != N_FEATURES {
        return Err(format!(
            "{}: expected {} weights, got {}",
            path.display(), N_FEATURES, values.len()
        ));
    }
    let mut out = [0f32; N_FEATURES];
    out.copy_from_slice(&values);
    Ok(out)
}
