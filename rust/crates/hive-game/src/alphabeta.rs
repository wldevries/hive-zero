//! Hive-specific glue for the generic alphabeta driver in `core_game::alphabeta`.

use core_game::alphabeta::{self, AlphaBetaParams, SearchContext};
use core_game::game::Undoable;

use crate::game::{Game, Move};
use crate::piece::PieceColor;

impl Undoable for Game {
    fn undo(&mut self) {
        Game::undo(self);
    }
}

/// Heuristic evaluation wrapped for the negamax driver: returns a value from
/// the perspective of the player about to move (current-player-relative).
///
/// `Game::heuristic_value()` returns `(white_score, black_score)`. Hive is
/// zero-sum so `black_score == -white_score`, but we read the side-correct
/// component explicitly to keep this robust if the eval ever changes.
pub fn heuristic_eval(game: &Game) -> f32 {
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
