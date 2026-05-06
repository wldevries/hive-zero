//! Shared configuration structs for AlphaZero-style self-play across games.
//!
//! These types cover the knobs that every game's `play_selfplay_core` consumes
//! identically — MCTS settings, playout-cap randomization, and opening-move
//! randomization. Game-specific extras (resignation, calibration, max-moves
//! caps, heuristic eval, …) stay in each game's own session config struct.
//!
//! At the pyo3 boundary the Python signature stays flat (individual kwargs),
//! and the binding layer unpacks them into these structs.

/// MCTS-related knobs that every game's self-play loop uses identically.
#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    pub simulations: usize,
    pub c_puct: f32,
    pub dir_alpha: f32,
    pub dir_epsilon: f32,
    pub forced_playouts: bool,
    pub draw_contempt: f32,
    /// When true, each self-play game randomly designates one side as the
    /// "contempt side"; `draw_contempt` only applies on that side's turns,
    /// while the other side searches with contempt 0. When false, contempt
    /// is applied symmetrically on every turn (the historical default).
    pub asymmetric_contempt: bool,
    /// Number of leaves collected across games per inference batch.
    pub play_batch_size: usize,
    /// Visit-count temperature applied for the first `temp_threshold` plies.
    pub temperature: f32,
    pub temp_threshold: u32,
}

/// KataGo-style playout-cap randomization. `p == 0.0` disables it; otherwise
/// each turn flips a coin: with probability `p`, run the full `simulations`
/// budget (full-search turn → trains policy + value); else run `fast_cap`
/// sims (value-only training).
#[derive(Clone, Copy, Debug, Default)]
pub struct PlayoutCapConfig {
    pub p: f32,
    pub fast_cap: usize,
}

impl PlayoutCapConfig {
    pub fn enabled(&self) -> bool {
        self.p > 0.0
    }
}

/// Opening-move randomization: each game plays a uniform random count
/// in `[min, max]` of random moves before MCTS takes over. `max == 0`
/// disables it.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpeningRandomConfig {
    pub min: u32,
    pub max: u32,
}

impl OpeningRandomConfig {
    pub fn enabled(&self) -> bool {
        self.max > 0
    }
}
