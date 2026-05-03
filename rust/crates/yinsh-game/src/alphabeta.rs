//! Heuristic-only alpha-beta minimax bot for Yinsh.
//!
//! Self-contained negamax that clones the board per node (Yinsh state is
//! ~150 bytes — cheaper than implementing a correct undo for `MoveRing`'s
//! marker flips and `ClaimRow`'s 5-marker + ring removal). If/when we want
//! transposition tables and the killer/history move ordering from
//! `core_game::alphabeta`, we'll need to implement `Undoable` for `YinshBoard`.

use core_game::game::{Outcome, Player};

use crate::board::{Cell, YinshBoard, YinshMove};
use crate::hex::{BOARD_SIZE, ROW_DIRS, cell_index_i8, index_to_cell, is_valid_i8};

/// Score returned for a position where the side-to-move has already won (or
/// lost). Large enough that no positional term can outweigh it.
pub const WIN_SCORE: f32 = 1_000_000.0;

// Eval weights. Tune by play; current numbers are first-pass guesses.
const SCORE_W: f32 = 1000.0;
const ROW5_W: f32 = 500.0;
const ROW4_W: f32 = 50.0;
const ROW3_W: f32 = 5.0;
const MOBILITY_W: f32 = 0.1;
const MARKER_W: f32 = 0.5;

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

/// Histogram of maximal contiguous runs of `player`'s markers along the three
/// row directions. `hist[k]` (for `k = 1..=5`) counts runs of exactly that
/// length; `hist[5]` collects 5 or longer (5+ shouldn't occur in practice
/// because rule G triggers a row claim immediately, but the clamp is safe).
fn run_length_histogram(board: &YinshBoard, player: Player) -> [usize; 6] {
    let marker = marker_of(player);
    let mut hist = [0usize; 6];

    for &(dc, dr) in &ROW_DIRS {
        for idx in 0..BOARD_SIZE {
            if board.cells[idx] != marker {
                continue;
            }
            // Only count from the start of a maximal run: previous cell in
            // this direction is off-board or not our marker.
            let (c, r) = index_to_cell(idx);
            let pc = c as i8 - dc;
            let pr = r as i8 - dr;
            if is_valid_i8(pc, pr) && board.cells[cell_index_i8(pc, pr)] == marker {
                continue;
            }
            // Walk forward to find the run length.
            let mut len = 0usize;
            let mut cc = c as i8;
            let mut rr = r as i8;
            while is_valid_i8(cc, rr) && board.cells[cell_index_i8(cc, rr)] == marker {
                len += 1;
                cc += dc;
                rr += dr;
            }
            hist[len.min(5)] += 1;
        }
    }
    hist
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

/// Evaluate `board` from `board.next_player`'s perspective. Positive = good
/// for whoever is about to move. Terminal positions return ±`WIN_SCORE` (or 0
/// for a draw).
pub fn evaluate(board: &YinshBoard) -> f32 {
    let me = board.next_player;
    let opp = me.opposite();

    match board.outcome {
        Outcome::WonBy(p) if p == me => return WIN_SCORE,
        Outcome::WonBy(_) => return -WIN_SCORE,
        Outcome::Draw => return 0.0,
        Outcome::Ongoing => {}
    }

    let score_diff = score_of(board, me) as f32 - score_of(board, opp) as f32;

    // Formed (unclaimed) 5-rows. Outside ClaimRow phase this is 0 — rows are
    // either claimed immediately or don't exist yet.
    let my_5 = board.find_rows(me).len() as f32;
    let opp_5 = board.find_rows(opp).len() as f32;

    let my_hist = run_length_histogram(board, me);
    let opp_hist = run_length_histogram(board, opp);
    let my_4 = my_hist[4] as f32;
    let opp_4 = opp_hist[4] as f32;
    let my_3 = my_hist[3] as f32;
    let opp_3 = opp_hist[3] as f32;

    let my_mob = ring_mobility(board, me) as f32;
    let opp_mob = ring_mobility(board, opp) as f32;

    let my_markers = count_markers(board, me) as f32;
    let opp_markers = count_markers(board, opp) as f32;

    SCORE_W * score_diff
        + ROW5_W * (my_5 - opp_5)
        + ROW4_W * (my_4 - opp_4)
        + ROW3_W * (my_3 - opp_3)
        + MOBILITY_W * (my_mob - opp_mob)
        + MARKER_W * (my_markers - opp_markers)
}

/// Negamax with alpha-beta pruning. Score is from the perspective of whoever
/// is about to move at `board`.
///
/// ClaimRow phase keeps `next_player` the same across a move (Yinsh rule G.3
/// pairs row claims with ring removal, then continues — possibly with another
/// claim — for the same player). On a same-player edge we don't flip the sign
/// of the recursive score; on a normal edge we do.
fn negamax(board: &mut YinshBoard, depth: u32, mut alpha: f32, beta: f32) -> f32 {
    if depth == 0 || board.outcome != Outcome::Ongoing {
        return evaluate(board);
    }

    let player_before = board.next_player;
    let moves = board.legal_moves();
    if moves.is_empty() {
        return evaluate(board);
    }

    let mut best = f32::NEG_INFINITY;
    for mv in moves {
        let mut child = board.clone();
        child
            .apply_move(mv)
            .expect("legal_moves yielded an illegal move");
        let score = if child.next_player == player_before {
            negamax(&mut child, depth - 1, alpha, beta)
        } else {
            -negamax(&mut child, depth - 1, -beta, -alpha)
        };
        if score > best {
            best = score;
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            break;
        }
    }
    best
}

/// Pick the best move at `depth` plies. Operates on a clone — the caller's
/// `board` is never mutated. Returns `Pass` if there are no legal moves
/// (which shouldn't happen in Yinsh — included for safety).
pub fn alphabeta_best_move(board: &YinshBoard, depth: u32) -> YinshMove {
    let mut work = board.clone();
    let moves = work.legal_moves();
    if moves.is_empty() {
        return YinshMove::Pass;
    }
    if depth == 0 {
        return moves[0];
    }

    let player_before = work.next_player;
    let mut alpha = f32::NEG_INFINITY;
    let beta = f32::INFINITY;
    let mut best_mv = moves[0];
    let mut best_score = f32::NEG_INFINITY;

    for mv in moves {
        let mut child = board.clone();
        child
            .apply_move(mv)
            .expect("legal_moves yielded an illegal move");
        let score = if child.next_player == player_before {
            negamax(&mut child, depth - 1, alpha, beta)
        } else {
            -negamax(&mut child, depth - 1, -beta, -alpha)
        };
        if score > best_score {
            best_score = score;
            best_mv = mv;
        }
        if score > alpha {
            alpha = score;
        }
    }
    best_mv
}

/// Return `(move, score)` for every legal root move at `depth`. Each move is
/// searched with a full alpha-beta window so the score is exact (not an
/// alpha-bounded estimate). Useful for debugging the eval and for showing the
/// top-N moves in the REPL.
///
/// O(N · search) where N is the legal move count — slower than `best_move`
/// but the right semantics for "show me the score of every move".
pub fn alphabeta_root_scores(board: &YinshBoard, depth: u32) -> Vec<(YinshMove, f32)> {
    let mut work = board.clone();
    let moves = work.legal_moves();
    let player_before = work.next_player;
    let mut out = Vec::with_capacity(moves.len());
    for mv in moves {
        let mut child = board.clone();
        child
            .apply_move(mv)
            .expect("legal_moves yielded an illegal move");
        let score = if depth == 0 {
            // No search — score the resulting position with the static eval.
            // Flip sign if the move passed the turn so the score is from
            // `player_before`'s perspective like the depth>0 branch.
            let raw = evaluate(&child);
            if child.next_player == player_before { raw } else { -raw }
        } else if child.next_player == player_before {
            negamax(&mut child, depth - 1, f32::NEG_INFINITY, f32::INFINITY)
        } else {
            -negamax(&mut child, depth - 1, f32::NEG_INFINITY, f32::INFINITY)
        };
        out.push((mv, score));
    }
    out
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

        let before = board.clone();
        let _ = alphabeta_best_move(&board, 2);

        assert_eq!(board.cells, before.cells, "cells changed");
        assert_eq!(board.next_player, before.next_player, "next_player changed");
        assert_eq!(board.phase, before.phase, "phase changed");
        assert_eq!(board.white_score, before.white_score);
        assert_eq!(board.black_score, before.black_score);
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
    fn run_length_histogram_counts_runs() {
        let mut board = YinshBoard::new();
        // Build a horizontal run of 4 white markers along the (0, 1) direction
        // starting at E5 — cells E5, E6, E7, E8.
        for k in 0..4u8 {
            board.cells[cell_index(4, 4 + k)] = Cell::WhiteMarker;
        }
        let hist = run_length_histogram(&board, Player::Player1);
        assert_eq!(hist[4], 1, "expected exactly one run of 4: {:?}", hist);
        assert_eq!(hist[3], 0);
        assert_eq!(hist[5], 0);
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
}
