/// Generic game traits for AlphaZero-style training.

use crate::symmetry::Symmetry;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    Player1,
    Player2,
}

impl Player {
    #[inline]
    pub fn opposite(self) -> Player {
        match self {
            Player::Player1 => Player::Player2,
            Player::Player2 => Player::Player1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Ongoing,
    Draw,
    WonBy(Player),
}

/// Core game rules trait — play moves, check outcome, enumerate legal moves.
pub trait Game: Clone + Send {
    type Move: Copy + Clone + Send + Sync + std::fmt::Debug;

    /// The symmetry group of this game's board geometry.
    /// Use `UnitSymmetry` for games with no exploitable symmetries.
    type Symmetry: Symmetry;

    /// Whose turn is it?
    fn current_player(&self) -> Player;

    /// Current game outcome.
    fn outcome(&self) -> Outcome;

    /// Is the game over?
    fn is_game_over(&self) -> bool {
        self.outcome() != Outcome::Ongoing
    }

    /// Generate all legal moves (empty if pass required or game over).
    fn valid_moves(&mut self) -> Vec<Self::Move>;

    /// Apply a move (including pass).
    fn play_move(&mut self, mv: &Self::Move) -> Result<(), String>;

    /// Create a pass move.
    fn pass_move() -> Self::Move;

    /// Check if a move is a pass.
    fn is_pass(mv: &Self::Move) -> bool;
}

/// Neural network encoding trait — tensor encoding and policy masks for AlphaZero training.
pub trait NNGame: Game {
    /// Number of channels in the board tensor encoding.
    const BOARD_CHANNELS: usize;
    /// Spatial grid dimension (board is GRID_SIZE x GRID_SIZE).
    const GRID_SIZE: usize;
    /// Size of the reserve/auxiliary input vector.
    const RESERVE_SIZE: usize;
    /// Total size of the policy output vector.
    const POLICY_SIZE: usize;

    /// Encode the board state into a flat tensor and reserve vector.
    fn encode_board(&self, board_out: &mut [f32], reserve_out: &mut [f32]);

    /// Get the legal move mask (POLICY_SIZE) and indexed moves.
    fn get_legal_move_mask(&mut self) -> (Vec<f32>, Vec<(usize, Self::Move)>);
}

// Keep the old name as an alias during migration.
pub trait GameEngine: NNGame {}
impl<T: NNGame> GameEngine for T {}
