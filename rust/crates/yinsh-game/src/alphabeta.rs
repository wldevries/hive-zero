//! Yinsh-specific glue for the generic alphabeta driver in `core_game::alphabeta`.
//!
//! Provides:
//! - `Undoable` (snapshot/restore via `YinshBoard::history`)
//! - `TranspositionKey` (SipHash over all state that affects the score)
//! - `MoveOrdering` (flat history-table index per (move-type, params))
//! - `evaluate` — the static heuristic, current-player-relative
//!
//! The public `alphabeta_best_move` and `alphabeta_root_scores` entry points
//! now go through the core driver, picking up iterative deepening, TT, and
//! killer/history move ordering for free.

use core_game::alphabeta::{
    self, AlphaBetaParams, MoveOrdering, SearchContext, TranspositionKey, WIN_SCORE,
};
use core_game::game::{Outcome, Player, Undoable};

use crate::board::{Cell, YinshBoard, YinshMove};
use crate::hex::{BOARD_SIZE, ROW_DIRS, cell_index_i8, index_to_cell, is_valid_i8};

/// Number of linear features in the eval. Feature order:
/// 0: score_diff      (rings captured)
/// 1: potential5_diff (5-cell windows containing all 5 of my markers — formed 5-rows)
/// 2: potential4_diff (5-cell windows with exactly 4 of my markers)
/// 3: potential3_diff (5-cell windows with exactly 3 of my markers)
/// 4: potential2_diff (5-cell windows with exactly 2 of my markers)
/// 5: mobility_diff   (sum of legal ring destinations)
/// 6: marker_diff
///
/// POTENTIAL_N counts overlap deliberately: a solid 4-in-a-row with both
/// flanks empty appears in 2 POTENTIAL4 windows, while a split-4 like
/// `xxxox` appears in only 1 — so the more tactically valuable shapes
/// score higher automatically. Ring-blocking is ignored (a ring inside a
/// window doesn't disqualify it); we may add that as a separate feature
/// later if it matters.
pub const N_FEATURES: usize = 7;

/// Linear eval weights, current-player-relative. First-pass guesses; the
/// `tune` subcommand on `yinsh-tools` fits these from boardspace outcomes.
pub const DEFAULT_WEIGHTS: [f32; N_FEATURES] = [
    1000.0, // SCORE
    500.0,  // POTENTIAL5
    10.0,   // POTENTIAL4
    1.0,    // POTENTIAL3
    0.1,    // POTENTIAL2
    1.0,    // MOBILITY (per ring; range ~0..20)
    0.5,    // MARKER
];

#[inline]
fn score_of(board: &YinshBoard, player: Player) -> u8 {
    match player {
        Player::Player1 => board.white_score,
        Player::Player2 => board.black_score,
    }
}

#[inline]
fn marker_of(player: Player) -> Cell {
    match player {
        Player::Player1 => Cell::WhiteMarker,
        Player::Player2 => Cell::BlackMarker,
    }
}

#[inline]
fn ring_of(player: Player) -> Cell {
    match player {
        Player::Player1 => Cell::WhiteRing,
        Player::Player2 => Cell::BlackRing,
    }
}

/// Per-window histograms for both players in a single pass over the board.
/// `out.0[k]` counts windows containing exactly k of `me`'s markers;
/// `out.1[k]` does the same for `opp`. Each window is scanned once and both
/// counts are tallied in lockstep — twice as fast as calling the
/// single-player version twice.
///
/// A 4-in-a-row with both flanks on-board appears in 2 windows of `hist[4]`;
/// a split-four like `xxxox` appears in 1. Solid shapes therefore naturally
/// outscore split ones, and shapes in the middle of the board outscore
/// edge-pinned shapes (more overlapping windows fit).
fn potential_histograms_both(
    board: &YinshBoard,
    me: Player,
    opp: Player,
) -> ([usize; 6], [usize; 6]) {
    let me_marker = marker_of(me);
    let opp_marker = marker_of(opp);
    let mut my_hist = [0usize; 6];
    let mut opp_hist = [0usize; 6];

    for &(dc, dr) in &ROW_DIRS {
        for idx in 0..BOARD_SIZE {
            let (c, r) = index_to_cell(idx);
            let mut my_count = 0usize;
            let mut opp_count = 0usize;
            let mut on_board = true;
            for k in 0..5i8 {
                let cc = c as i8 + k * dc;
                let rr = r as i8 + k * dr;
                if !is_valid_i8(cc, rr) {
                    on_board = false;
                    break;
                }
                let cell = board.cells[cell_index_i8(cc, rr)];
                if cell == me_marker {
                    my_count += 1;
                } else if cell == opp_marker {
                    opp_count += 1;
                }
            }
            if on_board {
                my_hist[my_count] += 1;
                opp_hist[opp_count] += 1;
            }
        }
    }
    (my_hist, opp_hist)
}

/// Sum of legal ring destinations across all of `player`'s rings on the board.
/// Setup-phase rings (none yet on board) contribute zero; that's fine — eval
/// during Setup is dominated by the (zero) positional terms anyway.
fn ring_mobility(board: &YinshBoard, player: Player) -> usize {
    let ring = ring_of(player);
    (0..BOARD_SIZE)
        .filter(|&i| board.cells[i] == ring)
        .map(|i| board.ring_destinations(i).len())
        .sum()
}

fn count_markers(board: &YinshBoard, player: Player) -> usize {
    let marker = marker_of(player);
    board.cells.iter().filter(|&&c| c == marker).count()
}

fn count_rings(board: &YinshBoard, player: Player) -> usize {
    let ring = ring_of(player);
    board.cells.iter().filter(|&&c| c == ring).count()
}

/// Extract the linear feature vector for `board`, current-player-relative.
/// Caller must check that `board.outcome == Ongoing` first — terminal
/// positions are handled by `evaluate` / `evaluate_with_weights`.
pub fn extract_features(board: &YinshBoard) -> [f32; N_FEATURES] {
    let me = board.next_player;
    let opp = me.opposite();

    let score_diff = score_of(board, me) as f32 - score_of(board, opp) as f32;
    let (my_hist, opp_hist) = potential_histograms_both(board, me, opp);
    // Mobility-per-ring rather than raw mobility: capturing a 5-row removes
    // a ring (and its destinations from the total) without any drop in the
    // remaining rings' quality. The per-ring metric stays comparable across
    // game phases. max(1) avoids div-by-zero for empty boards / Setup phase.
    let my_rings = count_rings(board, me).max(1) as f32;
    let opp_rings = count_rings(board, opp).max(1) as f32;
    let my_mob = ring_mobility(board, me) as f32 / my_rings;
    let opp_mob = ring_mobility(board, opp) as f32 / opp_rings;
    let my_markers = count_markers(board, me) as f32;
    let opp_markers = count_markers(board, opp) as f32;

    [
        score_diff,
        my_hist[5] as f32 - opp_hist[5] as f32,
        my_hist[4] as f32 - opp_hist[4] as f32,
        my_hist[3] as f32 - opp_hist[3] as f32,
        my_hist[2] as f32 - opp_hist[2] as f32,
        my_mob - opp_mob,
        my_markers - opp_markers,
    ]
}

/// Evaluate `board` from `board.next_player`'s perspective using `weights`.
/// Terminal positions return ±`WIN_SCORE` (or 0 for a draw) regardless of
/// weights.
pub fn evaluate_with_weights(board: &YinshBoard, weights: &[f32; N_FEATURES]) -> f32 {
    match board.outcome {
        Outcome::WonBy(p) if p == board.next_player => return WIN_SCORE,
        Outcome::WonBy(_) => return -WIN_SCORE,
        Outcome::Draw => return 0.0,
        Outcome::Ongoing => {}
    }
    let f = extract_features(board);
    let mut sum = 0.0;
    for i in 0..N_FEATURES {
        sum += weights[i] * f[i];
    }
    sum
}

/// Evaluate `board` from `board.next_player`'s perspective using the default
/// weights. Positive = good for whoever is about to move.
pub fn evaluate(board: &YinshBoard) -> f32 {
    evaluate_with_weights(board, &DEFAULT_WEIGHTS)
}

// ---------------------------------------------------------------------------
// Core-driver trait impls
// ---------------------------------------------------------------------------

impl Undoable for YinshBoard {
    fn undo(&mut self) {
        YinshBoard::undo(self);
    }
}

impl TranspositionKey for YinshBoard {
    /// Read the incrementally maintained Zobrist hash. O(1) on every probe.
    /// Order-independent set semantics for `pending_rows` / `deferred_rows`
    /// fall out of XOR's commutativity, so two queues differing only in
    /// push order map to the same key.
    fn tt_key(&self) -> u64 {
        self.hash
    }
}

// History-table layout: one slot per (move-type, parameters) combination.
// PlaceRing(idx)                              [0 .. 85)
// MoveRing { from, to }                       [85 .. 85 + 85*85)
// ClaimRow { start, dir, ring }               [next .. next + 85*3*85)
// Pass                                        no slot
const PLACE_BASE: usize = 0;
const MOVE_BASE: usize = PLACE_BASE + BOARD_SIZE;
const MOVE_SLOTS: usize = BOARD_SIZE * BOARD_SIZE;
const CLAIM_BASE: usize = MOVE_BASE + MOVE_SLOTS;
const CLAIM_SLOTS: usize = BOARD_SIZE * 3 * BOARD_SIZE;
const TOTAL_HISTORY_SLOTS: usize = CLAIM_BASE + CLAIM_SLOTS;

impl MoveOrdering for YinshMove {
    const HISTORY_SIZE: usize = TOTAL_HISTORY_SLOTS;

    fn history_index(&self) -> Option<usize> {
        match *self {
            YinshMove::PlaceRing(idx) => Some(PLACE_BASE + idx),
            YinshMove::MoveRing { from, to } => Some(MOVE_BASE + from * BOARD_SIZE + to),
            YinshMove::ClaimRow { start, dir, ring } => {
                Some(CLAIM_BASE + start * 3 * BOARD_SIZE + dir * BOARD_SIZE + ring)
            }
            YinshMove::Pass => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Run alphabeta from a clone of `board`, returning the best move at the given
/// depth (or `Pass` if the position has no legal moves). Operates on a clone
/// — the caller's board is never mutated.
pub fn alphabeta_best_move(board: &YinshBoard, depth: u32) -> YinshMove {
    alphabeta_best_move_with_weights(board, depth, &DEFAULT_WEIGHTS)
}

/// Like `alphabeta_best_move`, but evaluates positions using the supplied
/// linear weight vector instead of `DEFAULT_WEIGHTS`. Used by the tournament
/// harness to A/B-test fitted weights against the defaults.
pub fn alphabeta_best_move_with_weights(
    board: &YinshBoard,
    depth: u32,
    weights: &[f32; N_FEATURES],
) -> YinshMove {
    let mut work = board.clone();
    work.history.clear();
    let mut ctx = SearchContext::new(AlphaBetaParams::new(depth));
    let mut eval = |b: &YinshBoard| evaluate_with_weights(b, weights);
    let (mv, _score) = alphabeta::alphabeta_best_move(&mut work, &mut eval, &mut ctx);
    mv.unwrap_or(YinshMove::Pass)
}

/// Like `alphabeta_best_move`, but returns exact minimax scores for *every*
/// root move at the given depth. Operates on a clone — the caller's board is
/// never mutated.
pub fn alphabeta_root_scores(board: &YinshBoard, depth: u32) -> Vec<(YinshMove, f32)> {
    let mut work = board.clone();
    work.history.clear();
    let mut ctx = SearchContext::new(AlphaBetaParams::new(depth));
    let mut eval = evaluate;
    alphabeta::alphabeta_root_scores(&mut work, &mut eval, &mut ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hex::cell_index;

    #[test]
    fn alphabeta_does_not_mutate_caller() {
        let mut board = YinshBoard::new();
        // Play one Setup move so the position isn't trivially symmetric.
        let opening = board.legal_moves()[0];
        board.apply_move(opening).unwrap();

        let cells_before = board.cells;
        let next_before = board.next_player;
        let phase_before = board.phase;
        let history_len_before = board.history.len();

        let _ = alphabeta_best_move(&board, 2);

        assert_eq!(board.cells, cells_before, "cells changed");
        assert_eq!(board.next_player, next_before, "next_player changed");
        assert_eq!(board.phase, phase_before, "phase changed");
        assert_eq!(
            board.history.len(),
            history_len_before,
            "history length changed",
        );
    }

    #[test]
    fn alphabeta_returns_legal_move() {
        let mut board = YinshBoard::new();
        // Place all 10 rings to enter Normal phase.
        for _ in 0..10 {
            let mv = board.legal_moves()[0];
            board.apply_move(mv).unwrap();
        }
        let mv = alphabeta_best_move(&board, 2);
        let mut probe = board.clone();
        let legal = probe.legal_moves();
        assert!(
            legal.iter().any(|m| *m == mv),
            "alphabeta returned non-legal move {:?}",
            mv
        );
    }

    #[test]
    fn alphabeta_returns_legal_move_in_setup() {
        // First-move test: Setup phase, every empty cell is legal.
        let board = YinshBoard::new();
        let mv = alphabeta_best_move(&board, 2);
        assert!(matches!(mv, YinshMove::PlaceRing(_)));
    }

    #[test]
    fn evaluate_terminal_positions() {
        let mut board = YinshBoard::new();
        board.outcome = Outcome::WonBy(Player::Player1);
        board.next_player = Player::Player1;
        assert_eq!(evaluate(&board), WIN_SCORE);
        board.next_player = Player::Player2;
        assert_eq!(evaluate(&board), -WIN_SCORE);
        board.outcome = Outcome::Draw;
        assert_eq!(evaluate(&board), 0.0);
    }

    #[test]
    fn evaluate_score_diff_dominates() {
        let mut board = YinshBoard::new();
        // White has captured 2 rings, Black 0. White to move.
        board.white_score = 2;
        board.black_score = 0;
        let v_white_to_move = evaluate(&board);
        board.next_player = Player::Player2;
        let v_black_to_move = evaluate(&board);
        assert!(v_white_to_move > 0.0);
        assert!(v_black_to_move < 0.0);
        // Symmetric to within positional noise (no markers/rings on board).
        assert!((v_white_to_move + v_black_to_move).abs() < 1e-3);
    }

    #[test]
    fn potential_histogram_counts_overlapping_windows() {
        let mut board = YinshBoard::new();
        // 4 white markers in a row at (4, 4..7) along the (0, 1) direction.
        // Both flanks (rows 3 and 8) are on-board for column 4, so the 4-run
        // sits inside two distinct 5-windows: rows [3..7] and rows [4..8].
        for k in 0..4u8 {
            board.cells[cell_index(4, 4 + k)] = Cell::WhiteMarker;
        }
        let (my_hist, opp_hist) =
            potential_histograms_both(&board, Player::Player1, Player::Player2);
        assert_eq!(my_hist[4], 2, "solid-4 in mid-board sits in 2 windows: {my_hist:?}");
        assert_eq!(my_hist[5], 0, "no 5-row formed");
        assert_eq!(opp_hist[4], 0, "opponent has no markers");
    }

    #[test]
    fn potential_histogram_counts_split_pattern() {
        let mut board = YinshBoard::new();
        // Split pattern x.x.x at (4, 3), (4, 5), (4, 7) — 3 markers within a
        // single 5-cell window starting at row 3.
        board.cells[cell_index(4, 3)] = Cell::WhiteMarker;
        board.cells[cell_index(4, 5)] = Cell::WhiteMarker;
        board.cells[cell_index(4, 7)] = Cell::WhiteMarker;
        let (my_hist, _opp_hist) =
            potential_histograms_both(&board, Player::Player1, Player::Player2);
        assert!(my_hist[3] >= 1, "x.x.x should yield at least one P3 window: {my_hist:?}");
        assert_eq!(my_hist[5], 0);
        assert_eq!(my_hist[4], 0);
    }

    #[test]
    fn root_scores_one_per_legal_move() {
        let mut board = YinshBoard::new();
        for _ in 0..10 {
            let mv = board.legal_moves()[0];
            board.apply_move(mv).unwrap();
        }
        let mut probe = board.clone();
        let legal = probe.legal_moves();
        let scores = alphabeta_root_scores(&board, 2);
        assert_eq!(scores.len(), legal.len());
        for (mv, _) in &scores {
            assert!(legal.iter().any(|m| m == mv));
        }
    }

    #[test]
    fn undo_restores_state_through_search() {
        // After a search on `work`, the work board should equal `start` again
        // — the snapshot stack is fully drained.
        let mut start = YinshBoard::new();
        for _ in 0..6 {
            let mv = start.legal_moves()[0];
            start.apply_move(mv).unwrap();
        }
        let mut work = start.clone();
        work.history.clear();
        let snapshot_cells = work.cells;
        let snapshot_next = work.next_player;
        let snapshot_phase = work.phase;

        let mut ctx = SearchContext::new(AlphaBetaParams::new(3));
        let mut eval = evaluate;
        let _ = alphabeta::alphabeta_best_move(&mut work, &mut eval, &mut ctx);

        assert_eq!(work.cells, snapshot_cells, "cells diverged after search");
        assert_eq!(work.next_player, snapshot_next);
        assert_eq!(work.phase, snapshot_phase);
        assert_eq!(work.history.len(), 0, "history should be drained");
    }

    #[test]
    fn tt_does_not_change_search_result() {
        // TT-enabled search must reach the same (best_move, score) as a
        // search with the TT bypassed. Depth 2 in setup is fast and exercises
        // the standard probe/store paths.
        let mut start = YinshBoard::new();
        for _ in 0..3 {
            let mv = start.legal_moves()[0];
            start.apply_move(mv).unwrap();
        }
        let mut eval = evaluate;

        let mut work_no_tt = start.clone();
        work_no_tt.history.clear();
        let mut ctx_no_tt = SearchContext::new(AlphaBetaParams::new(2));
        ctx_no_tt.tt_enabled = false;
        let (mv_no_tt, score_no_tt) =
            alphabeta::alphabeta_best_move(&mut work_no_tt, &mut eval, &mut ctx_no_tt);

        let mut work_tt = start.clone();
        work_tt.history.clear();
        let mut ctx_tt = SearchContext::new(AlphaBetaParams::new(2));
        let (mv_tt, score_tt) =
            alphabeta::alphabeta_best_move(&mut work_tt, &mut eval, &mut ctx_tt);

        assert_eq!(score_no_tt, score_tt, "TT changed best score");
        assert_eq!(mv_no_tt, mv_tt, "TT changed best move");
        assert!(
            ctx_tt.nodes_visited <= ctx_no_tt.nodes_visited,
            "TT visited more nodes: {} vs {}",
            ctx_tt.nodes_visited, ctx_no_tt.nodes_visited,
        );
    }

    #[test]
    fn ordering_does_not_change_search_result() {
        // With killers + history disabled, the search must reach the same
        // (best_move, score) as with the full ordering stack.
        let mut start = YinshBoard::new();
        for _ in 0..3 {
            let mv = start.legal_moves()[0];
            start.apply_move(mv).unwrap();
        }
        let mut eval = evaluate;

        let mut work_off = start.clone();
        work_off.history.clear();
        let mut ctx_off = SearchContext::new(AlphaBetaParams::new(2));
        ctx_off.ordering_enabled = false;
        let (mv_off, score_off) =
            alphabeta::alphabeta_best_move(&mut work_off, &mut eval, &mut ctx_off);

        let mut work_on = start.clone();
        work_on.history.clear();
        let mut ctx_on = SearchContext::new(AlphaBetaParams::new(2));
        let (mv_on, score_on) =
            alphabeta::alphabeta_best_move(&mut work_on, &mut eval, &mut ctx_on);

        assert_eq!(score_off, score_on, "ordering changed best score");
        assert_eq!(mv_off, mv_on, "ordering changed best move");
        assert!(
            ctx_on.nodes_visited <= ctx_off.nodes_visited,
            "ordering visited more nodes: {} vs {}",
            ctx_on.nodes_visited, ctx_off.nodes_visited,
        );
    }

    #[test]
    fn undo_round_trip_through_claimrow() {
        // Force a ClaimRow situation, apply the claim, then undo and verify
        // the pre-claim state is fully restored. ClaimRow has the messiest
        // mutations (5 marker erasures, ring removal, score++, phase moves),
        // so this is the strictest undo test.
        let mut board = YinshBoard::new();
        // Place all 10 rings.
        for _ in 0..10 {
            let mv = board.legal_moves()[0];
            board.apply_move(mv).unwrap();
        }
        // Manually set up a 5-row of white markers along (0,1) starting at A1.
        // Place a ring nearby so we have something to remove. Decrement the
        // marker pool to keep state internally consistent (5 markers placed →
        // pool drops by 5), so the post-claim `set_markers_in_pool(+5)` lands
        // on a valid Zobrist table index.
        let row_start_col = 1u8;
        let row_start_row = 0u8;
        for k in 0u8..5 {
            let idx = crate::hex::cell_index(row_start_col, row_start_row + k);
            board.cells[idx] = Cell::WhiteMarker;
        }
        board.markers_in_pool -= 5;
        // Find an existing white ring to remove.
        let ring_idx = (0..BOARD_SIZE)
            .find(|&i| board.cells[i] == Cell::WhiteRing)
            .expect("a white ring must exist after setup");
        // Force the phase as if a row scan had just fired.
        board.phase = crate::board::Phase::ClaimRow;
        board.next_player = Player::Player1;
        board.original_mover = Some(Player::Player1);
        let row_start_idx = crate::hex::cell_index(row_start_col, row_start_row);
        board.pending_rows = vec![(row_start_idx, 0)];
        board.deferred_rows = vec![];
        board.deferred_player = None;

        let snapshot_cells = board.cells;
        let snapshot_white_score = board.white_score;
        let snapshot_phase = board.phase;
        let snapshot_pending = board.pending_rows.clone();
        // Drop the placement-phase snapshots so the post-undo `is_empty`
        // assertion checks only the claim's own undo frame.
        board.history.clear();

        let claim = YinshMove::ClaimRow {
            start: row_start_idx,
            dir: 0,
            ring: ring_idx,
        };
        board.apply_move(claim).unwrap();
        assert_eq!(board.white_score, snapshot_white_score + 1);
        assert_ne!(board.cells, snapshot_cells);

        board.undo();
        assert_eq!(board.cells, snapshot_cells, "claimrow undo missed cells");
        assert_eq!(board.white_score, snapshot_white_score);
        assert_eq!(board.phase, snapshot_phase);
        assert_eq!(board.pending_rows, snapshot_pending);
        assert!(board.history.is_empty());
    }
}
