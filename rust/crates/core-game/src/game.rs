/// Generic game engine trait for AlphaZero-style training.

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

/// Trait that a game must implement to be used with MCTS and AlphaZero training.
pub trait GameEngine: Clone + Send {
    type Move: Copy + Clone + Send + Sync + std::fmt::Debug;

    /// Number of channels in the board tensor encoding.
    const BOARD_CHANNELS: usize;
    /// Spatial grid dimension (board is GRID_SIZE x GRID_SIZE).
    const GRID_SIZE: usize;
    /// Size of the reserve/auxiliary input vector.
    const RESERVE_SIZE: usize;
    /// Total size of the policy output vector.
    const POLICY_SIZE: usize;

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

    /// Encode the board state into a flat tensor and reserve vector.
    fn encode_board(&self, board_out: &mut [f32], reserve_out: &mut [f32]);

    /// Get the legal move mask (POLICY_SIZE) and indexed moves.
    fn get_legal_move_mask(&mut self) -> (Vec<f32>, Vec<(usize, Self::Move)>);

    /// Create a pass move.
    fn pass_move() -> Self::Move;

    /// Check if a move is a pass.
    fn is_pass(mv: &Self::Move) -> bool;
}
