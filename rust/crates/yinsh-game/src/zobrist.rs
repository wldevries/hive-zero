//! Zobrist hashing tables for `YinshBoard`.
//!
//! All tables are `const`, generated at compile time from a fixed seed via a
//! splitmix64 stream. This keeps `tt_key()` a single `u64` field read in the
//! hot path while making the table values deterministic across builds (so a
//! cached TT can be compared between runs of the same binary).
//!
//! Layout / convention:
//! - One u64 per (cell, cell-state) pair. `Cell::Empty` (discriminant 0) is
//!   forced to 0 so empty cells contribute nothing — the standard Zobrist
//!   convention. Updates: XOR out the old contribution, XOR in the new.
//! - A single `TURN_DIFF` u64. Toggling the side-to-move is `hash ^= TURN_DIFF`.
//! - Per-value tables for the small scalars (phase, markers_in_pool, scores,
//!   rings-placed counts) so each (field, value) pair has its own random.
//! - Pending / deferred row queues use distinct (start, dir) tables — the same
//!   logical row contributes a different hash bit depending on whether it sits
//!   in the current player's pending queue or the opponent's deferred queue.
//! - `Option<Player>` fields hash `None` as no contribution and `Some(p)` as
//!   the corresponding random.

use crate::hex::BOARD_SIZE;

/// Generate one splitmix64 value from `state`. State is updated by adding the
/// SplitMix64 constant before mixing — call this once per output you need,
/// always passing the *previous returned state*.
const fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const fn make_table_1d<const N: usize>(seed: u64) -> [u64; N] {
    let mut out = [0u64; N];
    let mut state = seed;
    let mut i = 0;
    while i < N {
        out[i] = splitmix64_next(&mut state);
        i += 1;
    }
    out
}

const fn make_table_2d<const M: usize, const N: usize>(seed: u64) -> [[u64; N]; M] {
    let mut out = [[0u64; N]; M];
    let mut state = seed;
    let mut i = 0;
    while i < M {
        let mut j = 0;
        while j < N {
            out[i][j] = splitmix64_next(&mut state);
            j += 1;
        }
        i += 1;
    }
    out
}

// Distinct seeds → independent streams. Values are arbitrary; pinned so the
// generated tables don't shift with unrelated edits.
const SEED_CELL: u64 = 0x1234_5678_9ABC_DEF0;
const SEED_PHASE: u64 = 0x0F0F_0F0F_F0F0_F0F0;
const SEED_TURN: u64 = 0x5555_AAAA_5555_AAAA;
const SEED_MARKERS: u64 = 0x1111_2222_3333_4444;
const SEED_WHITE_SCORE: u64 = 0xAAAA_BBBB_CCCC_DDDD;
const SEED_BLACK_SCORE: u64 = 0xDDDD_CCCC_BBBB_AAAA;
const SEED_WHITE_RINGS_PLACED: u64 = 0x9999_8888_7777_6666;
const SEED_BLACK_RINGS_PLACED: u64 = 0x6666_7777_8888_9999;
const SEED_PENDING_ROW: u64 = 0xFEED_FACE_DEAD_BEEF;
const SEED_DEFERRED_ROW: u64 = 0xCAFE_BABE_FEED_DAD0;
const SEED_DEFERRED_PLAYER: u64 = 0x1357_9BDF_2468_ACE0;
const SEED_ORIGINAL_MOVER: u64 = 0x0ECA_8642_FDB9_7531;

/// Per-cell table. `[idx][cell_state]`. `Cell::Empty` (state 0) is zeroed so
/// empty cells contribute nothing.
pub const CELL: [[u64; 5]; BOARD_SIZE] = {
    let mut t = make_table_2d::<BOARD_SIZE, 5>(SEED_CELL);
    let mut i = 0;
    while i < BOARD_SIZE {
        t[i][0] = 0;
        i += 1;
    }
    t
};

pub const PHASE: [u64; 3] = make_table_1d::<3>(SEED_PHASE);

/// XOR this single value to flip the side to move.
pub const TURN_DIFF: u64 = {
    let mut state = SEED_TURN;
    splitmix64_next(&mut state)
};

pub const MARKERS_IN_POOL: [u64; 52] = make_table_1d::<52>(SEED_MARKERS);
pub const WHITE_SCORE: [u64; 4] = make_table_1d::<4>(SEED_WHITE_SCORE);
pub const BLACK_SCORE: [u64; 4] = make_table_1d::<4>(SEED_BLACK_SCORE);
pub const WHITE_RINGS_PLACED: [u64; 6] = make_table_1d::<6>(SEED_WHITE_RINGS_PLACED);
pub const BLACK_RINGS_PLACED: [u64; 6] = make_table_1d::<6>(SEED_BLACK_RINGS_PLACED);

/// Pending and deferred row tables. `[start][dir]` where `start` indexes the
/// row's first cell and `dir` ∈ 0..3.
pub const PENDING_ROW: [[u64; 3]; BOARD_SIZE] =
    make_table_2d::<BOARD_SIZE, 3>(SEED_PENDING_ROW);
pub const DEFERRED_ROW: [[u64; 3]; BOARD_SIZE] =
    make_table_2d::<BOARD_SIZE, 3>(SEED_DEFERRED_ROW);

/// `Option<Player>` fields. Indexed by `Player as usize`. `None` contributes 0.
pub const DEFERRED_PLAYER: [u64; 2] = make_table_1d::<2>(SEED_DEFERRED_PLAYER);
pub const ORIGINAL_MOVER: [u64; 2] = make_table_1d::<2>(SEED_ORIGINAL_MOVER);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cell_contribution_is_zero() {
        for i in 0..BOARD_SIZE {
            assert_eq!(CELL[i][0], 0, "cell {i} empty contribution must be 0");
        }
    }

    #[test]
    fn nonzero_entries_distinct_within_each_table() {
        // Sanity check: no collisions inside the cell table for non-empty
        // states. (A handful of duplicates would survive a mediocre PRNG.)
        let mut seen = std::collections::HashSet::new();
        for i in 0..BOARD_SIZE {
            for s in 1..5 {
                assert!(seen.insert(CELL[i][s]), "duplicate Zobrist random at CELL[{i}][{s}]");
            }
        }
    }

    #[test]
    fn turn_diff_is_nonzero() {
        assert_ne!(TURN_DIFF, 0);
    }
}
