//! Zertz-specific glue for the generic alphabeta driver in `core_game::alphabeta`.
//!
//! Provides:
//! - `Undoable` (already implemented on `ZertzBoard` via the snapshot stack)
//! - `TranspositionKey` (SipHash over rings/supply/captures/player/mid_capture)
//! - `MoveOrdering` (flat history-table index per move-shape + parameters)
//! - `evaluate` — a small linear heuristic, current-player-relative
//!
//! Public entry points `alphabeta_best_move` and `alphabeta_root_scores`
//! delegate to the core driver and inherit iterative deepening, transposition
//! table, killer moves, and history ordering.

use std::time::Duration;

use core_game::alphabeta::{
    self, AlphaBetaParams, MoveOrdering, SearchContext, TranspositionKey, WIN_SCORE,
};
use core_game::game::{Game, Outcome, Player};

use crate::hex::{BOARD_SIZE, DIRECTIONS, Hex, hex_to_index, is_valid};
use crate::zertz::{Marble, Ring, ZertzBoard, ZertzMove};

// ---------------------------------------------------------------------------
// Linear evaluation
// ---------------------------------------------------------------------------

/// Number of linear features in the eval. All features are
/// current-player-relative — positive = good for the side about to move.
///
/// 0: WEIGHTED_CAPS — sum over colors of `caps[c] / win_threshold[c]`,
///    diff. A captured white is worth 1/4 (you need 4 to win the white path),
///    grey 1/5, black 1/6. This reflects the marginal value of each marble
///    along its single-color win condition.
/// 1: BEST_PROGRESS — fractional progress toward the *closest* of the four
///    win paths (4W, 5G, 6B, 3-of-each). Diff. Captures matter most when
///    they advance the path you're already winning down.
/// 2: RAW_CAPS — total capture count diff. Small tiebreaker that nudges the
///    eval to prefer extra material even when the marginal-progress features
///    are flat (e.g. once you have 3+ of each color and are sitting at a
///    won-but-not-yet-claimed position).
/// 3: MID_CAPTURE — +1.0 if the side to move is currently mid-capture chain
///    (free continuation hops with no opponent reply). A hop is forced for
///    the player on move, but the *fact* of being mid-chain is good for them.
pub const N_FEATURES: usize = 4;

pub const F_WEIGHTED_CAPS: usize = 0;
pub const F_BEST_PROGRESS: usize = 1;
pub const F_RAW_CAPS: usize = 2;
pub const F_MID_CAPTURE: usize = 3;

/// Single-color win thresholds: 4 white, 5 grey, 6 black.
const WIN_THRESH: [f32; 3] = [4.0, 5.0, 6.0];

/// Linear eval weights, current-player-relative. Hand-picked first pass —
/// tuning from boardspace games (analogous to Yinsh's Texel pipeline) can
/// refine these later.
pub const DEFAULT_WEIGHTS: [f32; N_FEATURES] = [
    100.0, // WEIGHTED_CAPS  — diff is roughly in [-3, 3]
    100.0, // BEST_PROGRESS  — diff in [-1, 1]
    5.0,   // RAW_CAPS       — diff in [-N, N], small weight as a tiebreaker
    30.0,  // MID_CAPTURE    — binary; rewards being on a free-hop streak
];

#[inline]
fn player_index(p: Player) -> usize {
    match p {
        Player::Player1 => 0,
        Player::Player2 => 1,
    }
}

#[inline]
fn weighted_caps(c: &[u8; 3]) -> f32 {
    c[0] as f32 / WIN_THRESH[0]
        + c[1] as f32 / WIN_THRESH[1]
        + c[2] as f32 / WIN_THRESH[2]
}

#[inline]
fn raw_caps(c: &[u8; 3]) -> f32 {
    (c[0] as u16 + c[1] as u16 + c[2] as u16) as f32
}

/// Best fractional progress toward any win condition for one player.
/// Returns the max over (W/4, G/5, B/6, three-of-each-progress), where
/// three-of-each-progress = (min(W,3) + min(G,3) + min(B,3)) / 9. So
/// a balanced 3-3-3 returns 1.0, lopsided 6-0-0 returns max(6/4, ..) clamped
/// at 1.0 below — but we keep it unclamped so terminal-adjacent positions
/// (e.g. having 4 of a color but not yet flagged as won by the rules) still
/// score above ongoing positions. The `Outcome::WonBy` short-circuit in
/// `evaluate_with_weights` handles actual wins anyway.
fn best_progress(c: &[u8; 3]) -> f32 {
    let p_w = c[0] as f32 / WIN_THRESH[0];
    let p_g = c[1] as f32 / WIN_THRESH[1];
    let p_b = c[2] as f32 / WIN_THRESH[2];
    let p_each = (c[0].min(3) as u16 + c[1].min(3) as u16 + c[2].min(3) as u16) as f32 / 9.0;
    p_w.max(p_g).max(p_b).max(p_each)
}

/// Extract the linear feature vector for `board`, current-player-relative.
/// Caller must check that `board.outcome() == Ongoing` first — terminal
/// positions are handled by `evaluate_with_weights`.
pub fn extract_features(board: &ZertzBoard) -> [f32; N_FEATURES] {
    let me = player_index(board.next_player());
    let opp = 1 - me;
    let caps = board.captures();
    let my = &caps[me];
    let op = &caps[opp];

    let mid_cap = if board.is_mid_capture() { 1.0 } else { 0.0 };

    [
        weighted_caps(my) - weighted_caps(op),
        best_progress(my) - best_progress(op),
        raw_caps(my) - raw_caps(op),
        mid_cap,
    ]
}

/// Evaluate `board` from `board.next_player()`'s perspective using `weights`.
/// Terminal positions return ±`WIN_SCORE` (or 0 for a draw) regardless of
/// the linear weights.
pub fn evaluate_with_weights(board: &ZertzBoard, weights: &[f32; N_FEATURES]) -> f32 {
    match board.outcome() {
        Outcome::WonBy(p) if p == board.next_player() => return WIN_SCORE,
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

/// Evaluate `board` from `board.next_player()`'s perspective using the
/// default weights. Positive = good for whoever is about to move.
pub fn evaluate(board: &ZertzBoard) -> f32 {
    evaluate_with_weights(board, &DEFAULT_WEIGHTS)
}

// ---------------------------------------------------------------------------
// Core-driver trait impls
// ---------------------------------------------------------------------------

impl TranspositionKey for ZertzBoard {
    fn tt_key(&self) -> u64 {
        ZertzBoard::tt_key(self)
    }
}

// ---------------------------------------------------------------------------
// MovePriority — Zertz-specific static move ordering
// ---------------------------------------------------------------------------
//
// Zertz tactics live almost entirely in the placement decision: a placement
// can hand the opponent a forced capture, expose a marble I'd love to
// recapture, or progress toward isolating a vulnerable group. The standard
// killer/history machinery is poorly tuned for these (capture turns are
// forced and have no choice; placement subtree is too wide for killers to
// recur). So we provide a static priority that classifies each Place /
// PlaceOnly move tactically, and let the driver use it as the primary
// sort key (killers/history fall back to within-bucket tiebreakers).
//
// Cost budget: per-move computation must stay well under 1 µs because this
// runs inside the inner sort loop. The two helpers below do at most ~12 cell
// reads (capture exposure) + ~36 cell reads (isolation) per move.

/// Marginal value (in [0, ~0.4]) of capturing one more `color` for the
/// player whose captures are `caps`. Combines two paths:
/// - Single-color path: `1 / threshold[color]`. A captured white nudges
///   you `1/4` of the way to the 4-white win; grey is `1/5`; black `1/6`.
/// - 3-of-each path: `1/9` extra if you don't yet have 3 of this color.
///   Each color carries 1/3 of the 3-each progress, distributed evenly.
fn marginal_value(caps: &[u8; 3], color: Marble) -> f32 {
    let c = color.index();
    let single = 1.0 / WIN_THRESH[c];
    let three_each = if caps[c] < 3 { 1.0 / 9.0 } else { 0.0 };
    single + three_each
}

/// Recapture-upside discount: a forced capture leaves the jumper sitting on
/// a new cell where it *might* be capturable next turn, but verifying that
/// statically would defeat the cheapness budget. 0.5 is a rough average
/// over "actually recapturable" (≈1) and "out of reach" (0).
const RECAPTURE_DISCOUNT: f32 = 0.5;

/// Cost (positive = bad for the side to move) of the captures that the
/// placement at `place_at` (with optional ring removal at `remove_at`)
/// exposes to the opponent's next turn.
///
/// Key insight: pre-move there are no jumps available (otherwise captures
/// would be mandatory and we wouldn't be placing). Removing a ring can
/// only kill landing spots, never create new ones. So every newly-exposed
/// jump must involve the just-placed marble at `H = place_at` either as
/// the jumper or as the jumped-over piece. That collapses the check to
/// 12 directional probes per move.
fn capture_exposure(
    board: &ZertzBoard,
    placed_color: Marble,
    place_at: Hex,
    remove_at: Option<Hex>,
) -> f32 {
    let me = match board.next_player() {
        Player::Player1 => 0,
        Player::Player2 => 1,
    };
    let opp = 1 - me;
    let my_caps = &board.captures()[me];
    let opp_caps = &board.captures()[opp];
    let rings = board.rings();

    // View the post-move board: H now holds `placed_color`, R is Removed,
    // off-board cells act as Removed for landing/jumping purposes.
    let cell_at = |h: Hex| -> Ring {
        if !is_valid(h) {
            return Ring::Removed;
        }
        if h == place_at {
            return Ring::Occupied(placed_color);
        }
        if Some(h) == remove_at {
            return Ring::Removed;
        }
        rings[hex_to_index(h)]
    };

    let mut score = 0.0_f32;
    for &dir in &DIRECTIONS {
        // Case A: H acts as the "over" — opponent's jumper at H-dir jumps
        // over H to land at H+dir. Opp captures our just-placed marble;
        // their jumper survives at H+dir.
        let from = (place_at.0 - dir.0, place_at.1 - dir.1);
        let to_a = (place_at.0 + dir.0, place_at.1 + dir.1);
        if let Ring::Occupied(jumper_color) = cell_at(from) {
            if cell_at(to_a) == Ring::Empty {
                score += marginal_value(opp_caps, placed_color);
                score -= RECAPTURE_DISCOUNT * marginal_value(my_caps, jumper_color);
            }
        }

        // Case B: H acts as the "from" — opp jumps using H as the jumper,
        // capturing the marble at H+dir. Captures aren't player-specific in
        // Zertz, so opp is happy to use our marble. The jumper (= our
        // placed marble) survives at H+2dir, where we *might* recapture.
        let over = (place_at.0 + dir.0, place_at.1 + dir.1);
        let to_b = (place_at.0 + 2 * dir.0, place_at.1 + 2 * dir.1);
        if let Ring::Occupied(victim_color) = cell_at(over) {
            if cell_at(to_b) == Ring::Empty {
                score += marginal_value(opp_caps, victim_color);
                score -= RECAPTURE_DISCOUNT * marginal_value(my_caps, placed_color);
            }
        }
    }
    score
}

/// Bonus (positive = good for the side to move) for ring removals that
/// progress toward isolating a marble I'd want. Cheap proxy: a marble M
/// neighboring `remove_at` with at most one *other* marble neighbor is
/// in a loose group, and removing one of M's surrounding rings makes
/// isolation more likely on subsequent turns. Isolated marbles are
/// captured by the *current* player (the one removing the ring), so they
/// score by my marginal value, weighted by color.
fn isolation_bonus(board: &ZertzBoard, remove_at: Hex) -> f32 {
    let me = match board.next_player() {
        Player::Player1 => 0,
        Player::Player2 => 1,
    };
    let my_caps = &board.captures()[me];
    let rings = board.rings();

    let mut bonus = 0.0_f32;
    for &dir in &DIRECTIONS {
        let n = (remove_at.0 + dir.0, remove_at.1 + dir.1);
        if !is_valid(n) {
            continue;
        }
        let m_color = match rings[hex_to_index(n)] {
            Ring::Occupied(c) => c,
            _ => continue,
        };
        // Count M's currently-Occupied neighbors (including remove_at — we
        // want the pre-removal count; the +1 in the threshold below
        // accounts for losing the remove_at connection).
        let mut occupied_neighbors = 0u8;
        for &dir2 in &DIRECTIONS {
            let nn = (n.0 + dir2.0, n.1 + dir2.1);
            if is_valid(nn) && matches!(rings[hex_to_index(nn)], Ring::Occupied(_)) {
                occupied_neighbors += 1;
            }
        }
        // Loose group proxy: M has 0 or 1 marble friend. Removing remove_at
        // is suggestive of impending isolation. (More precise: full
        // connectivity check, but we're staying cheap.)
        if occupied_neighbors <= 1 {
            bonus += marginal_value(my_caps, m_color);
        }
    }
    bonus
}

/// Multiplier converting the f32 priority score to i32 for the sort key.
/// 1000 gives us 3 decimal digits of resolution on values that live in
/// roughly [-3, 3].
const ORDERING_SCALE: f32 = 1000.0;

// History-table layout: one slot per (move-shape, parameters) tuple.
//   PlaceOnly(color, place_at)            : 3 * BOARD_SIZE
//   Place(color, place_at, remove)        : 3 * BOARD_SIZE * BOARD_SIZE
//   Capture (single hop, by from + dir)   : BOARD_SIZE * 6
// Pass and any move whose direction can't be matched against `DIRECTIONS`
// (multi-hop captures collapse to first-hop direction) get None.
const PLACE_ONLY_BASE: usize = 0;
const PLACE_ONLY_SLOTS: usize = 3 * BOARD_SIZE;
const PLACE_BASE: usize = PLACE_ONLY_BASE + PLACE_ONLY_SLOTS;
const PLACE_SLOTS: usize = 3 * BOARD_SIZE * BOARD_SIZE;
const CAPTURE_BASE: usize = PLACE_BASE + PLACE_SLOTS;
const CAPTURE_SLOTS: usize = BOARD_SIZE * 6;
const TOTAL_HISTORY_SLOTS: usize = CAPTURE_BASE + CAPTURE_SLOTS;

#[inline]
fn marble_idx(m: Marble) -> usize {
    m as usize
}

/// Index of `dir` in `DIRECTIONS`, or `None` if `dir` is not a unit hex
/// direction (shouldn't happen for legal capture moves, but we'd rather
/// silently skip ordering for an unrecognised direction than panic).
#[inline]
fn direction_index(from: Hex, over: Hex) -> Option<usize> {
    let d = (over.0 - from.0, over.1 - from.1);
    DIRECTIONS.iter().position(|&u| u == d)
}

impl MoveOrdering<ZertzBoard> for ZertzMove {
    const HISTORY_SIZE: usize = TOTAL_HISTORY_SLOTS;

    fn history_index(&self) -> Option<usize> {
        match *self {
            ZertzMove::PlaceOnly { color, place_at } => Some(
                PLACE_ONLY_BASE + marble_idx(color) * BOARD_SIZE + hex_to_index(place_at),
            ),
            ZertzMove::Place {
                color,
                place_at,
                remove,
            } => Some(
                PLACE_BASE
                    + marble_idx(color) * BOARD_SIZE * BOARD_SIZE
                    + hex_to_index(place_at) * BOARD_SIZE
                    + hex_to_index(remove),
            ),
            ZertzMove::Capture { jumps, len } if len > 0 => {
                let (from, over, _to) = jumps[0];
                let dir = direction_index(from, over)?;
                Some(CAPTURE_BASE + hex_to_index(from) * 6 + dir)
            }
            ZertzMove::Capture { .. } | ZertzMove::Pass => None,
        }
    }

    fn priority(&self, board: &ZertzBoard) -> i32 {
        match *self {
            ZertzMove::Place {
                color,
                place_at,
                remove,
            } => {
                let exposure = capture_exposure(board, color, place_at, Some(remove));
                let isolation = isolation_bonus(board, remove);
                ((exposure - isolation) * ORDERING_SCALE) as i32
            }
            ZertzMove::PlaceOnly { color, place_at } => {
                let exposure = capture_exposure(board, color, place_at, None);
                (exposure * ORDERING_SCALE) as i32
            }
            // Capture turns: small fan-out (every legal capture must be
            // searched anyway) and no real ordering payoff. Skip.
            // Pass: never produced by Zertz's valid_moves.
            ZertzMove::Capture { .. } | ZertzMove::Pass => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Run alphabeta from a clone of `board`, returning the best move at the
/// given depth (or `Pass` if no legal moves). Operates on a clone — the
/// caller's board is never mutated.
pub fn alphabeta_best_move(board: &ZertzBoard, depth: u32) -> ZertzMove {
    alphabeta_best_move_with_budget(board, depth, None)
}

/// Like `alphabeta_best_move`, but caps the search by wall-clock time.
/// `depth` is still a hard upper bound; whichever expires first wins.
/// On a deadline abort the function returns the best move from the
/// deepest fully-completed iteration (always at least depth 1 in
/// practice — depth 1 on Zertz is cheap).
pub fn alphabeta_best_move_with_budget(
    board: &ZertzBoard,
    depth: u32,
    time_budget: Option<Duration>,
) -> ZertzMove {
    alphabeta_best_move_with_weights_and_budget(board, depth, &DEFAULT_WEIGHTS, time_budget)
}

/// Like `alphabeta_best_move`, but evaluates positions using a supplied
/// linear weight vector. Useful for tournament A/B testing of fitted
/// weights against the defaults.
pub fn alphabeta_best_move_with_weights(
    board: &ZertzBoard,
    depth: u32,
    weights: &[f32; N_FEATURES],
) -> ZertzMove {
    alphabeta_best_move_with_weights_and_budget(board, depth, weights, None)
}

/// Combines `with_weights` and `with_budget`. The two flags are orthogonal
/// — pass `None` for the budget to disable the wall-clock cap.
pub fn alphabeta_best_move_with_weights_and_budget(
    board: &ZertzBoard,
    depth: u32,
    weights: &[f32; N_FEATURES],
    time_budget: Option<Duration>,
) -> ZertzMove {
    let mut work = board.clone();
    work.apply_history.clear();
    let params = AlphaBetaParams::new(depth).with_time_budget(time_budget);
    let mut ctx = SearchContext::new(params);
    let mut eval = |b: &ZertzBoard| evaluate_with_weights(b, weights);
    let (mv, _score) = alphabeta::alphabeta_best_move(&mut work, &mut eval, &mut ctx);
    mv.unwrap_or(ZertzMove::Pass)
}

/// Like `alphabeta_best_move`, but returns exact minimax scores for *every*
/// root move at the given depth — handy for deriving a softmax policy
/// distribution or inspecting the search's view of every option. Operates
/// on a clone — the caller's board is never mutated.
pub fn alphabeta_root_scores(board: &ZertzBoard, depth: u32) -> Vec<(ZertzMove, f32)> {
    let mut work = board.clone();
    work.apply_history.clear();
    let mut ctx = SearchContext::new(AlphaBetaParams::new(depth));
    let mut eval = evaluate;
    alphabeta::alphabeta_root_scores(&mut work, &mut eval, &mut ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_game::game::Game;

    #[test]
    fn alphabeta_does_not_mutate_caller() {
        let mut board = ZertzBoard::default();
        // Play one opening move so the position isn't trivial.
        let opening = board.legal_moves()[0];
        board.play(opening).unwrap();

        let rings_before = *board.rings();
        let captures_before = *board.captures();
        let supply_before = *board.supply();
        let next_before = board.next_player();
        let history_len_before = board.apply_history.len();

        let _ = alphabeta_best_move(&board, 2);

        assert_eq!(*board.rings(), rings_before, "rings changed");
        assert_eq!(*board.captures(), captures_before, "captures changed");
        assert_eq!(*board.supply(), supply_before, "supply changed");
        assert_eq!(board.next_player(), next_before, "next_player changed");
        assert_eq!(
            board.apply_history.len(),
            history_len_before,
            "apply_history length changed",
        );
    }

    #[test]
    fn alphabeta_returns_legal_move() {
        // Open game, no captures forced — search must return a placement.
        let board = ZertzBoard::default();
        let mv = alphabeta_best_move(&board, 2);
        let legal = board.legal_moves();
        assert!(
            legal.iter().any(|m| *m == mv),
            "alphabeta returned non-legal move {:?}",
            mv,
        );
    }

    #[test]
    fn starting_position_is_balanced() {
        // No marbles captured, no mid-capture: every feature should be zero
        // and the eval should be exactly zero.
        let board = ZertzBoard::default();
        let f = extract_features(&board);
        for v in &f {
            assert_eq!(*v, 0.0, "expected starting features all zero, got {f:?}");
        }
        assert_eq!(evaluate(&board), 0.0);
    }

    #[test]
    fn root_scores_one_per_legal_move() {
        let mut board = ZertzBoard::default();
        // Play a few moves to create asymmetry.
        for _ in 0..3 {
            let mv = board.legal_moves()[0];
            board.play(mv).unwrap();
        }
        let legal = board.legal_moves();
        let scores = alphabeta_root_scores(&board, 1);
        assert_eq!(scores.len(), legal.len(), "one score per legal move expected");
        for (mv, _) in &scores {
            assert!(legal.iter().any(|m| m == mv), "scored non-legal move {:?}", mv);
        }
    }

    #[test]
    fn undo_round_trip_through_search() {
        // After a search on a clone of `start`, the cloned board's apply
        // history should be fully drained (all undo frames popped).
        let mut start = ZertzBoard::default();
        for _ in 0..3 {
            let mv = start.legal_moves()[0];
            start.play(mv).unwrap();
        }

        let mut work = start.clone();
        work.apply_history.clear();
        let snapshot_rings = *work.rings();
        let snapshot_caps = *work.captures();
        let snapshot_next = work.next_player();

        let mut ctx = SearchContext::new(AlphaBetaParams::new(2));
        let mut eval = evaluate;
        let _ = alphabeta::alphabeta_best_move(&mut work, &mut eval, &mut ctx);

        assert_eq!(*work.rings(), snapshot_rings, "rings diverged after search");
        assert_eq!(*work.captures(), snapshot_caps, "captures diverged");
        assert_eq!(work.next_player(), snapshot_next, "player diverged");
        assert_eq!(
            work.apply_history.len(),
            0,
            "apply_history should be drained",
        );
    }

    #[test]
    fn tt_does_not_change_search_result() {
        // A TT-enabled search must reach the same (best_move, score) as a
        // search with the TT bypassed.
        let mut start = ZertzBoard::default();
        for _ in 0..3 {
            let mv = start.legal_moves()[0];
            start.play(mv).unwrap();
        }
        let mut eval = evaluate;

        let mut work_no_tt = start.clone();
        work_no_tt.apply_history.clear();
        let mut ctx_no_tt = SearchContext::new(AlphaBetaParams::new(2));
        ctx_no_tt.tt_enabled = false;
        let (mv_no_tt, score_no_tt) =
            alphabeta::alphabeta_best_move(&mut work_no_tt, &mut eval, &mut ctx_no_tt);

        let mut work_tt = start.clone();
        work_tt.apply_history.clear();
        let mut ctx_tt = SearchContext::new(AlphaBetaParams::new(2));
        let (mv_tt, score_tt) =
            alphabeta::alphabeta_best_move(&mut work_tt, &mut eval, &mut ctx_tt);

        assert_eq!(score_no_tt, score_tt, "TT changed best score");
        assert_eq!(mv_no_tt, mv_tt, "TT changed best move");
        assert!(
            ctx_tt.nodes_visited <= ctx_no_tt.nodes_visited,
            "TT visited more nodes: {} vs {}",
            ctx_tt.nodes_visited,
            ctx_no_tt.nodes_visited,
        );
    }

    #[test]
    fn ordering_does_not_change_search_result() {
        let mut start = ZertzBoard::default();
        for _ in 0..3 {
            let mv = start.legal_moves()[0];
            start.play(mv).unwrap();
        }
        let mut eval = evaluate;

        let mut work_off = start.clone();
        work_off.apply_history.clear();
        let mut ctx_off = SearchContext::new(AlphaBetaParams::new(2));
        ctx_off.ordering_enabled = false;
        let (mv_off, score_off) =
            alphabeta::alphabeta_best_move(&mut work_off, &mut eval, &mut ctx_off);

        let mut work_on = start.clone();
        work_on.apply_history.clear();
        let mut ctx_on = SearchContext::new(AlphaBetaParams::new(2));
        let (mv_on, score_on) =
            alphabeta::alphabeta_best_move(&mut work_on, &mut eval, &mut ctx_on);

        assert_eq!(score_off, score_on, "ordering changed best score");
        assert_eq!(mv_off, mv_on, "ordering changed best move");
        assert!(
            ctx_on.nodes_visited <= ctx_off.nodes_visited,
            "ordering visited more nodes: {} vs {}",
            ctx_on.nodes_visited,
            ctx_off.nodes_visited,
        );
    }

    #[test]
    fn move_ordering_indices_are_in_range() {
        // Every history index produced by a legal-move shape must fall
        // within HISTORY_SIZE — out-of-bounds indices would silently
        // corrupt the killer/history tables.
        let board = ZertzBoard::default();
        for mv in board.legal_moves() {
            if let Some(idx) = mv.history_index() {
                assert!(
                    idx < <ZertzMove as MoveOrdering<ZertzBoard>>::HISTORY_SIZE,
                    "history_index {idx} out of range for move {mv:?}",
                );
            }
        }
    }
}
