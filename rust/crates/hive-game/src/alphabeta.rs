//! Hive-specific glue for the generic alphabeta driver in `core_game::alphabeta`.

use core_game::alphabeta::{
    self, AlphaBetaParams, MoveOrdering, SearchContext, TranspositionKey, WIN_SCORE,
};
use core_game::game::Undoable;

use crate::board::{GRID_SIZE, hex_to_grid};
use crate::game::{Game, GameState, Move};
use crate::piece::{PIECES_PER_PLAYER, PieceColor};

/// Penalty returned for `DrawByRepetition` so the bot avoids stalling. Strong
/// enough to dominate any positional consideration but well below the win/loss
/// magnitude so a real loss still trumps a repetition draw.
const REPETITION_PENALTY: f32 = -WIN_SCORE / 4.0;

impl Undoable for Game {
    fn undo(&mut self) {
        Game::undo(self);
    }
}

impl MoveOrdering<Game> for Move {
    // 22 distinct piece linear indices (white 0..10, black 11..21) times the
    // grid area give every (piece, destination) pair its own slot. Pass moves
    // and any out-of-range destination map to None and are not tracked.
    const HISTORY_SIZE: usize = 2 * PIECES_PER_PLAYER * GRID_SIZE * GRID_SIZE;

    fn history_index(&self) -> Option<usize> {
        let piece = self.piece?;
        let to = self.to?;
        let (row, col) = hex_to_grid(to)?;
        Some(piece.linear_index() * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col)
    }
    // priority defaults to 0 — Hive eval already encodes queen-danger /
    // queen-escape tactics, so the recursion finds them naturally. Worth
    // revisiting if Hive search ever becomes a measured bottleneck.
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
pub fn evaluate(game: &Game) -> f32 {
    if game.state == GameState::DrawByRepetition {
        return REPETITION_PENALTY;
    }
    let (white, black) = game.evaluate();
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
    let mut eval = evaluate;
    let (mv, _score) = alphabeta::alphabeta_best_move(&mut work, &mut eval, &mut ctx);
    mv.unwrap_or_else(Move::pass)
}

/// Like `alphabeta_best_move`, but returns exact minimax scores for *every*
/// root move at the given depth — the data needed to derive a softmax policy
/// distribution for training the policy head on bot turns. Operates on a
/// clone — the caller's `game` is never mutated.
pub fn alphabeta_root_scores(game: &Game, depth: u32) -> Vec<(Move, f32)> {
    let mut work = game.clone();
    let mut ctx = SearchContext::new(AlphaBetaParams::new(depth));
    let mut eval = evaluate;
    alphabeta::alphabeta_root_scores(&mut work, &mut eval, &mut ctx)
}

/// Deepened static evaluation: run alphabeta to the given depth and return
/// the best-line value as `(white_score, black_score)`, both in `[-1, 1]`.
///
/// At depth 0 this matches `Game::evaluate()`. At depth ≥ 1 the score
/// reflects the minimax outcome `depth` plies ahead with the same static
/// heuristic at the leaves, so tactical patterns inside the horizon (e.g.
/// a queen-surround threat one or two moves away) are baked into the score
/// instead of being invisible to the static eval.
///
/// Terminal positions (`Draw`, `DrawByRepetition`, `WhiteWins`, `BlackWins`)
/// bypass the search and return the static heuristic unchanged — there is
/// nothing to search and the previous training pipeline already used the
/// static eval for these states.
///
/// Wins/losses found inside the search horizon collapse to `±1.0` after
/// clamping, so the result stays in the same range as the static heuristic
/// and remains a valid value-head training target.
pub fn evaluate_alphabeta(game: &Game, depth: u32) -> (f32, f32) {
    if game.state != GameState::InProgress {
        return game.evaluate();
    }
    let mut work = game.clone();
    let mut ctx = SearchContext::new(AlphaBetaParams::new(depth));
    let mut eval = evaluate;
    let (_mv, score) = alphabeta::alphabeta_best_move(&mut work, &mut eval, &mut ctx);
    let cur = score.clamp(-1.0, 1.0);
    match game.turn_color {
        PieceColor::White => (cur, -cur),
        PieceColor::Black => (-cur, cur),
    }
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
    fn root_scores_cover_all_legal_moves() {
        let mut game = Game::new();
        for _ in 0..3 {
            let mv = game.valid_moves()[0];
            game.play_move(&mv).unwrap();
        }
        let scores = alphabeta_root_scores(&game, 2);
        let legal = game.valid_moves();
        assert_eq!(scores.len(), legal.len(), "one score per legal move expected");
        for (mv, _) in &scores {
            assert!(legal.iter().any(|m| m == mv), "scored non-legal move {:?}", mv);
        }
    }

    #[test]
    fn root_scores_best_is_legal_and_does_not_mutate_caller() {
        let mut canonical = Game::new();
        let opening = canonical.valid_moves()[0];
        canonical.play_move(&opening).unwrap();
        let before_hash = canonical.position_hash();
        let before_count = canonical.move_count;

        let scores = alphabeta_root_scores(&canonical, 2);
        let (best_mv, _) = scores
            .iter()
            .copied()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .expect("non-empty");
        let legal = canonical.valid_moves();
        assert!(legal.iter().any(|m| *m == best_mv), "best is not legal: {best_mv:?}");

        assert_eq!(canonical.position_hash(), before_hash);
        assert_eq!(canonical.move_count, before_count);
    }

    #[test]
    fn evaluate_alphabeta_depth0_matches_static() {
        // Depth 0: alphabeta short-circuits to eval(game) for InProgress
        // positions, so the deepened heuristic must match the static one.
        let mut game = Game::new();
        for _ in 0..3 {
            let mv = game.valid_moves()[0];
            game.play_move(&mv).unwrap();
        }
        let (sw, sb) = game.evaluate();
        let (dw, db) = evaluate_alphabeta(&game, 0);
        assert!((sw - dw).abs() < 1e-6, "white score diverged: static={sw}, deep={dw}");
        assert!((sb - db).abs() < 1e-6, "black score diverged: static={sb}, deep={db}");
    }

    #[test]
    fn evaluate_alphabeta_stays_in_range() {
        // Depth 3 should keep the result in [-1, 1] even when the search
        // discovers terminal lines (which would otherwise score ±WIN_SCORE).
        let mut game = Game::new();
        for _ in 0..6 {
            let mv = game.valid_moves()[0];
            game.play_move(&mv).unwrap();
        }
        let (w, b) = evaluate_alphabeta(&game, 3);
        assert!((-1.0..=1.0).contains(&w), "white score out of range: {w}");
        assert!((-1.0..=1.0).contains(&b), "black score out of range: {b}");
        // Zero-sum invariant: the two sides' scores must be exact negatives.
        assert!((w + b).abs() < 1e-6, "scores not zero-sum: w={w}, b={b}");
    }

    #[test]
    fn evaluate_alphabeta_does_not_mutate_caller() {
        let mut canonical = Game::new();
        let opening = canonical.valid_moves()[0];
        canonical.play_move(&opening).unwrap();
        let before_hash = canonical.position_hash();
        let before_count = canonical.move_count;

        let _ = evaluate_alphabeta(&canonical, 2);

        assert_eq!(canonical.position_hash(), before_hash, "canonical hash changed");
        assert_eq!(canonical.move_count, before_count, "canonical move_count changed");
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

        let mut eval = evaluate;

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

        let mut eval = evaluate;

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
    /// at each depth, comparing TT-only ordering (killers + history off)
    /// to the full TT + killers + history ordering. Run in release:
    /// `cargo test -p hive-game --release alphabeta_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn alphabeta_timing() {
        let mut start = Game::new();
        for _ in 0..6 {
            let mv = start.valid_moves()[0];
            start.play_move(&mv).unwrap();
        }
        let mut eval = evaluate;

        for depth in [2u32, 3, 4, 5] {
            let mut tt_only = start.clone();
            let mut ctx_tt_only = SearchContext::new(AlphaBetaParams::new(depth));
            ctx_tt_only.ordering_enabled = false;
            let t0 = std::time::Instant::now();
            let _ = alphabeta::alphabeta_best_move(&mut tt_only, &mut eval, &mut ctx_tt_only);
            let t_tt_only = t0.elapsed();

            let mut full = start.clone();
            let mut ctx_full = SearchContext::new(AlphaBetaParams::new(depth));
            let t0 = std::time::Instant::now();
            let _ = alphabeta::alphabeta_best_move(&mut full, &mut eval, &mut ctx_full);
            let t_full = t0.elapsed();

            let saved_nodes = ctx_tt_only.nodes_visited.saturating_sub(ctx_full.nodes_visited);
            let saved_pct = 100.0 * saved_nodes as f64 / ctx_tt_only.nodes_visited as f64;
            eprintln!(
                "depth {}: TT-only {:>8.2} ms ({:>7} nodes) -> +k+h {:>8.2} ms ({:>7} nodes)  \
                 saved {:>5.1}%, killer_hits={}, order_hits={}",
                depth,
                t_tt_only.as_secs_f64() * 1000.0, ctx_tt_only.nodes_visited,
                t_full.as_secs_f64() * 1000.0, ctx_full.nodes_visited,
                saved_pct, ctx_full.killer_hits, ctx_full.tt_order_hits,
            );
        }
    }

    #[test]
    fn ordering_does_not_change_search_result() {
        // With killers + history disabled, the search must reach the same
        // (best_move, score) as with the full ordering stack. Different node
        // counts are expected (and fewer with ordering on), but the minimax
        // value cannot move.
        let mut start = Game::new();
        for _ in 0..3 {
            let mv = start.valid_moves()[0];
            start.play_move(&mv).unwrap();
        }
        let mut eval = evaluate;

        let mut work_off = start.clone();
        let mut ctx_off = SearchContext::new(AlphaBetaParams::new(3));
        ctx_off.ordering_enabled = false;
        let (mv_off, score_off) =
            alphabeta::alphabeta_best_move(&mut work_off, &mut eval, &mut ctx_off);

        let mut work_on = start.clone();
        let mut ctx_on = SearchContext::new(AlphaBetaParams::new(3));
        let (mv_on, score_on) =
            alphabeta::alphabeta_best_move(&mut work_on, &mut eval, &mut ctx_on);

        assert_eq!(score_off, score_on, "ordering changed best score");
        assert_eq!(mv_off, mv_on, "ordering changed best move");
        assert!(
            ctx_on.nodes_visited <= ctx_off.nodes_visited,
            "ordering visited more nodes than no-ordering: {} vs {}",
            ctx_on.nodes_visited, ctx_off.nodes_visited,
        );
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
        let mut eval = evaluate;
        let _ = alphabeta::alphabeta_best_move(&mut work, &mut eval, &mut ctx);

        assert_eq!(work.position_hash(), snapshot_hash,
            "work game hash diverged from snapshot after search — undo path is incomplete");
        assert_eq!(work.move_count, snapshot.move_count);
        assert_eq!(work.turn_color, snapshot.turn_color);
    }
}
