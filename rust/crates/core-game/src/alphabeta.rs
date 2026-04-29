//! Generic negamax alphabeta search.
//!
//! Caller supplies an evaluation function that scores a position from the
//! perspective of the player about to move (current-player-relative). The
//! driver itself is game-agnostic — it relies only on the `Game` and
//! `Undoable` traits.
//!
//! Mzinga's `GameAI` adds iterative deepening, a transposition table, and
//! killer-move heuristics on top of the same alphabeta core. This module
//! exposes a `SearchContext` so those features can be plugged in later
//! without changing the recursion signature.
//!
//! Value convention:
//! - Terminal win for the player to move → `WIN_SCORE` (≈ +1e6).
//! - Terminal loss → `-WIN_SCORE`.
//! - Draw → 0.0.
//! - Heuristic eval is expected to live in (−1, 1) for ongoing positions,
//!   so it never collides with terminal magnitudes.

use crate::game::{Game, Outcome, Undoable};

/// Scalar used for terminal win/loss positions. Larger than any reasonable
/// heuristic eval so that win-finding always dominates positional scoring.
pub const WIN_SCORE: f32 = 1_000_000.0;

#[derive(Clone, Copy, Debug)]
pub struct AlphaBetaParams {
    pub max_depth: u32,
}

impl AlphaBetaParams {
    pub fn new(max_depth: u32) -> Self {
        Self { max_depth }
    }
}

/// Mutable per-search state. v1 only carries params; future variants will add
/// a transposition table, killer-move slots, node counts, etc.
pub struct SearchContext {
    pub params: AlphaBetaParams,
    pub nodes_visited: u64,
}

impl SearchContext {
    pub fn new(params: AlphaBetaParams) -> Self {
        Self { params, nodes_visited: 0 }
    }
}

/// Map an `Outcome` to a value from the current player's perspective.
/// Returns `None` if the game is still ongoing.
fn terminal_score<G: Game>(game: &G) -> Option<f32> {
    match game.outcome() {
        Outcome::Ongoing => None,
        Outcome::Draw => Some(0.0),
        Outcome::WonBy(winner) => {
            if winner == game.next_player() {
                Some(WIN_SCORE)
            } else {
                Some(-WIN_SCORE)
            }
        }
    }
}

/// Negamax alphabeta on a mutable game state. Mutates `game` during recursion
/// (via `play_move`/`undo`) but always restores it before returning.
fn negamax<G, F>(
    game: &mut G,
    depth: u32,
    mut alpha: f32,
    beta: f32,
    eval: &mut F,
    ctx: &mut SearchContext,
) -> f32
where
    G: Game + Undoable,
    F: FnMut(&G) -> f32,
{
    ctx.nodes_visited += 1;

    if let Some(score) = terminal_score(game) {
        return score;
    }
    if depth == 0 {
        return eval(game);
    }

    let moves = game.valid_moves();
    if moves.is_empty() {
        // Forced pass — recurse with one ply consumed and flipped sign.
        let pass = G::pass_move();
        game.play_move(&pass).expect("pass must be playable");
        let score = -negamax(game, depth.saturating_sub(1), -beta, -alpha, eval, ctx);
        game.undo();
        return score;
    }

    let mut best = f32::NEG_INFINITY;
    for mv in moves.iter() {
        game.play_move(mv).expect("valid_moves returned an unplayable move");
        let score = -negamax(game, depth - 1, -beta, -alpha, eval, ctx);
        game.undo();

        if score > best {
            best = score;
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            break; // beta cutoff
        }
    }

    best
}

/// Find the best move for the player to move, searching to the given depth.
/// Returns `(best_move, score)` from the current player's perspective.
///
/// Mutates `game` during the search but restores it before returning, so the
/// caller can pass `&mut canonical` if they own it. Most callers will pass a
/// clone to keep canonical state untouched.
pub fn alphabeta_best_move<G, F>(
    game: &mut G,
    eval: &mut F,
    ctx: &mut SearchContext,
) -> (Option<G::Move>, f32)
where
    G: Game + Undoable,
    F: FnMut(&G) -> f32,
{
    if game.is_game_over() {
        return (None, terminal_score(game).unwrap_or(0.0));
    }

    let moves = game.valid_moves();
    if moves.is_empty() {
        return (Some(G::pass_move()), 0.0);
    }

    let depth = ctx.params.max_depth;
    let mut alpha = f32::NEG_INFINITY;
    let beta = f32::INFINITY;
    let mut best_move = moves[0];
    let mut best_score = f32::NEG_INFINITY;

    for mv in moves.iter() {
        game.play_move(mv).expect("valid_moves returned an unplayable move");
        let score = -negamax(game, depth.saturating_sub(1), -beta, -alpha, eval, ctx);
        game.undo();

        if score > best_score {
            best_score = score;
            best_move = *mv;
        }
        if score > alpha {
            alpha = score;
        }
    }

    (Some(best_move), best_score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Player;
    use crate::symmetry::UnitSymmetry;

    /// A tiny deterministic game for unit-testing the alphabeta driver:
    /// players take turns picking a number 1..=3 to subtract from a counter.
    /// Whoever takes the counter to exactly 0 loses (analog of misère Nim).
    #[derive(Clone, Debug)]
    struct CountdownGame {
        counter: u8,
        history: Vec<u8>,
        player: Player,
    }

    impl CountdownGame {
        fn new(start: u8) -> Self {
            Self { counter: start, history: Vec::new(), player: Player::Player1 }
        }
    }

    impl Game for CountdownGame {
        type Move = u8;
        type Symmetry = UnitSymmetry;

        fn next_player(&self) -> Player { self.player }
        fn outcome(&self) -> Outcome {
            if self.counter == 0 {
                // Whoever just moved took it to 0; that player loses.
                Outcome::WonBy(self.player)
            } else {
                Outcome::Ongoing
            }
        }
        fn valid_moves(&mut self) -> Vec<u8> {
            if self.counter == 0 { return vec![]; }
            (1..=3).filter(|&n| n <= self.counter).collect()
        }
        fn play_move(&mut self, mv: &u8) -> Result<(), String> {
            self.history.push(self.counter);
            self.counter -= mv;
            self.player = self.player.opposite();
            Ok(())
        }
        fn pass_move() -> u8 { 0 }
        fn is_pass(mv: &u8) -> bool { *mv == 0 }
    }

    impl Undoable for CountdownGame {
        fn undo(&mut self) {
            self.counter = self.history.pop().expect("undo with empty history");
            self.player = self.player.opposite();
        }
    }

    #[test]
    fn negamax_finds_winning_move_at_counter_4() {
        // counter=4: the player to move wants to leave counter=1 for the opponent
        // (forcing them to take the last 1 and lose), so the winning move is 3.
        let mut game = CountdownGame::new(4);
        let mut ctx = SearchContext::new(AlphaBetaParams::new(6));
        let mut eval = |_: &CountdownGame| 0.0;
        let (mv, score) = alphabeta_best_move(&mut game, &mut eval, &mut ctx);
        assert_eq!(mv, Some(3));
        assert!(score > 0.0, "expected winning score, got {score}");
    }

    #[test]
    fn undo_restores_state() {
        let mut game = CountdownGame::new(5);
        let mut ctx = SearchContext::new(AlphaBetaParams::new(4));
        let mut eval = |_: &CountdownGame| 0.0;
        let snapshot_counter = game.counter;
        let snapshot_player = game.player;
        let _ = alphabeta_best_move(&mut game, &mut eval, &mut ctx);
        assert_eq!(game.counter, snapshot_counter);
        assert_eq!(game.player, snapshot_player);
        assert!(game.history.is_empty());
    }
}
