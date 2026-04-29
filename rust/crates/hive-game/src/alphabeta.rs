//! Hive-specific glue for the generic alphabeta driver in `core_game::alphabeta`.

use core_game::alphabeta::{self, AlphaBetaParams, SearchContext, TranspositionKey, WIN_SCORE};
use core_game::game::Undoable;

use crate::game::{Game, GameState, Move};
use crate::piece::PieceColor;

/// Penalty returned for `DrawByRepetition` so the bot avoids stalling. Strong
/// enough to dominate any positional consideration but well below the win/loss
/// magnitude so a real loss still trumps a repetition draw.
const REPETITION_PENALTY: f32 = -WIN_SCORE / 4.0;

impl Undoable for Game {
    fn undo(&mut self) {
        Game::undo(self);
    }
}

impl TranspositionKey for Game {
    fn tt_key(&self) -> u64 {
        // position_hash is a Zobrist over board layout + side-to-move; it
        // does NOT encode repetition history. Caching is gated below.
        self.position_hash()
    }

    fn tt_cacheable(&self) -> bool {
        // The score at a `DrawByRepetition` node is the repetition penalty,
        // which only applies on paths that actually triggered the repetition.
        // The same Zobrist key reached via a non-repeating path would score
        // very differently, so do not cache these.
        self.state != GameState::DrawByRepetition
    }
}

/// Heuristic evaluation wrapped for the negamax driver: returns a value from
/// the perspective of the player about to move (current-player-relative).
///
/// Distinguishes `DrawByRepetition` from a "real" draw (mutual queen surround)
/// because the bot's role in self-play is to drive games to a decisive result
/// — accepting repetition defeats that goal.
pub fn heuristic_eval(game: &Game) -> f32 {
    if game.state == GameState::DrawByRepetition {
        return REPETITION_PENALTY;
    }
    let (white, black) = game.heuristic_value();
    match game.turn_color {
        PieceColor::White => white,
        PieceColor::Black => black,
    }
}

/// Run alphabeta from a clone of `game`, returning the best move at the given
/// depth (or `Move::pass()` if the position has no legal moves).
///
/// Operates on a clone — the caller's `game` is never mutated.
pub fn alphabeta_best_move(game: &Game, depth: u32) -> Move {
    let mut work = game.clone();
    let mut ctx = SearchContext::new(AlphaBetaParams::new(depth));
    let mut eval = heuristic_eval;
    let (mv, _score) = alphabeta::alphabeta_best_move(&mut work, &mut eval, &mut ctx);
    mv.unwrap_or_else(Move::pass)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    use core_game::alphabeta::BoundFlag;

    #[test]
    fn alphabeta_does_not_mutate_caller_game() {
        let mut canonical = Game::new();
        // Play one opening move so the position hash isn't trivial zero.
        let opening = canonical.valid_moves()[0];
        canonical.play_move(&opening).unwrap();
        let before_hash = canonical.position_hash();
        let before_count = canonical.move_count;

        let _mv = alphabeta_best_move(&canonical, 2);

        assert_eq!(canonical.position_hash(), before_hash, "canonical hash changed");
        assert_eq!(canonical.move_count, before_count, "canonical move_count changed");
    }

    #[test]
    fn alphabeta_returns_legal_move() {
        let mut game = Game::new();
        let opening = game.valid_moves()[0];
        game.play_move(&opening).unwrap();

        let mv = alphabeta_best_move(&game, 2);
        let legal: Vec<_> = game.valid_moves();
        assert!(legal.iter().any(|m| *m == mv), "alphabeta returned non-legal move {:?}", mv);
    }

    #[test]
    fn tt_does_not_change_search_result() {
        // Correctness invariant: a TT-enabled search must return the same
        // (best_move, score) as a search with the TT bypassed, and it must
        // not visit more nodes. Depth 3 is fast (~1s debug) and exercises
        // both store paths (Exact and Bounded entries asserted below).
        // Phase 3's move-ordering win is exercised in the ignored
        // `tt_phase3_ordering_saves_nodes` benchmark.
        let mut start = Game::new();
        for _ in 0..3 {
            let mv = start.valid_moves()[0];
            start.play_move(&mv).unwrap();
        }

        let mut eval = heuristic_eval;

        let mut work_no_tt = start.clone();
        let mut ctx_no_tt = SearchContext::new(AlphaBetaParams::new(3));
        ctx_no_tt.tt_enabled = false;
        let (mv_no_tt, score_no_tt) =
            alphabeta::alphabeta_best_move(&mut work_no_tt, &mut eval, &mut ctx_no_tt);

        let mut work_tt = start.clone();
        let mut ctx_tt = SearchContext::new(AlphaBetaParams::new(3));
        let (mv_tt, score_tt) =
            alphabeta::alphabeta_best_move(&mut work_tt, &mut eval, &mut ctx_tt);

        assert_eq!(score_no_tt, score_tt, "TT changed best score");
        assert_eq!(mv_no_tt, mv_tt, "TT changed best move");
        assert!(
            ctx_tt.nodes_visited <= ctx_no_tt.nodes_visited,
            "TT visited more nodes than no-TT: {} vs {}",
            ctx_tt.nodes_visited, ctx_no_tt.nodes_visited,
        );

        // Phase 2 sanity: the TT must actually contain at least one
        // non-Exact entry from a non-trivial search, otherwise the bound-
        // classification code is silently dead.
        let mut exact = 0usize;
        let mut lower = 0usize;
        let mut upper = 0usize;
        for (_k, entry) in ctx_tt.tt.entries() {
            match entry.flag {
                BoundFlag::Exact => exact += 1,
                BoundFlag::LowerBound => lower += 1,
                BoundFlag::UpperBound => upper += 1,
            }
        }
        assert!(exact > 0, "no Exact TT entries stored");
        assert!(
            lower + upper > 0,
            "no bounded TT entries stored — phase 2 code path not exercised \
             (exact={exact}, lower={lower}, upper={upper})",
        );
    }

    /// Benchmark: depth 4 search demonstrates the combined Phase 1-4 win.
    /// Ignored by default; run with `cargo test -- --ignored --nocapture`.
    /// With iterative deepening, shallow TT entries rarely satisfy deeper
    /// queries (so `tt_hits` ≈ 0), but their best_moves drive aggressive
    /// move ordering which is the dominant source of savings here.
    #[test]
    #[ignore]
    fn tt_iterative_deepening_saves_nodes() {
        let mut start = Game::new();
        for _ in 0..3 {
            let mv = start.valid_moves()[0];
            start.play_move(&mv).unwrap();
        }

        let mut eval = heuristic_eval;

        let mut work_no_tt = start.clone();
        let mut ctx_no_tt = SearchContext::new(AlphaBetaParams::new(4));
        ctx_no_tt.tt_enabled = false;
        let (_, score_no_tt) =
            alphabeta::alphabeta_best_move(&mut work_no_tt, &mut eval, &mut ctx_no_tt);

        let mut work_tt = start.clone();
        let mut ctx_tt = SearchContext::new(AlphaBetaParams::new(4));
        let (_, score_tt) =
            alphabeta::alphabeta_best_move(&mut work_tt, &mut eval, &mut ctx_tt);

        assert_eq!(score_no_tt, score_tt, "TT changed best score at depth 4");

        let saved = ctx_no_tt.nodes_visited.saturating_sub(ctx_tt.nodes_visited);
        let pct = 100.0 * saved as f64 / ctx_no_tt.nodes_visited as f64;
        eprintln!(
            "TT depth 4: nodes {} -> {} ({:.1}% saved), tt_hits={}, \
             tt_order_hits={}, tt_size={}",
            ctx_no_tt.nodes_visited, ctx_tt.nodes_visited, pct,
            ctx_tt.tt_hits, ctx_tt.tt_order_hits, ctx_tt.tt.len(),
        );

        // ID's win is move ordering, not score reuse: assert ordering is
        // actually firing, and that the resulting node savings clear a
        // meaningful threshold.
        assert!(
            ctx_tt.tt_order_hits > 0,
            "ID move-ordering never fired at depth 4",
        );
        assert!(
            pct > 50.0,
            "expected ID+TT to save >50% of nodes at depth 4, got {pct:.1}%",
        );
    }

    /// Wall-clock timing benchmark: how long does a single bot move take
    /// at each depth with all four phases of TT enabled? Run in release:
    /// `cargo test -p hive-game --release alphabeta_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn alphabeta_timing() {
        let mut start = Game::new();
        for _ in 0..6 {
            let mv = start.valid_moves()[0];
            start.play_move(&mv).unwrap();
        }
        let mut eval = heuristic_eval;

        for depth in [2u32, 3, 4, 5] {
            let mut work = start.clone();
            let mut ctx = SearchContext::new(AlphaBetaParams::new(depth));
            let t0 = std::time::Instant::now();
            let _ = alphabeta::alphabeta_best_move(&mut work, &mut eval, &mut ctx);
            let elapsed = t0.elapsed();
            eprintln!(
                "depth {}: {:>8.2} ms  ({:>7} nodes, {:>5} order_hits, tt_size={})",
                depth, elapsed.as_secs_f64() * 1000.0,
                ctx.nodes_visited, ctx.tt_order_hits, ctx.tt.len(),
            );
        }
    }

    #[test]
    fn undo_restores_state_through_search() {
        // Run alphabeta on `work`, then verify hash matches the original.
        let mut start = Game::new();
        for _ in 0..3 {
            let mv = start.valid_moves()[0];
            start.play_move(&mv).unwrap();
        }
        let snapshot = start.clone();
        let snapshot_hash = snapshot.position_hash();

        let mut work = start.clone();
        let mut ctx = SearchContext::new(AlphaBetaParams::new(2));
        let mut eval = heuristic_eval;
        let _ = alphabeta::alphabeta_best_move(&mut work, &mut eval, &mut ctx);

        assert_eq!(work.position_hash(), snapshot_hash,
            "work game hash diverged from snapshot after search — undo path is incomplete");
        assert_eq!(work.move_count, snapshot.move_count);
        assert_eq!(work.turn_color, snapshot.turn_color);
    }
}
