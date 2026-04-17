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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    fn next_player(&self) -> Player;

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

/// How to extract a scalar prior from the flat policy vector.
///
/// `Single(idx)`:    `prior = policy[idx]`
/// `Sum(a, b)`:      `prior = policy[a] + policy[b]` (factorized logits, legacy)
/// `DotProduct`:     `prior = Q[src] · K[dst] / sqrt(embed_dim)` (bilinear head)
///   Q[src] = policy[q_offset + 0*g2 + src_cell .. q_offset + (embed_dim-1)*g2 + src_cell]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyIndex {
    Single(usize),
    Sum(usize, usize),
    DotProduct {
        q_offset: usize,
        k_offset: usize,
        src_cell: usize,
        dst_cell: usize,
        embed_dim: usize,
        g2: usize,
    },
}

/// Neural network encoding trait — tensor encoding and policy masks for AlphaZero training.
pub trait NNGame: Game {
    /// Number of channels in the board tensor encoding.
    const BOARD_CHANNELS: usize;
    /// Size of the reserve/auxiliary input vector.
    const RESERVE_SIZE: usize;
    /// Number of policy channels. For factorized policies this is the conceptual
    /// channel count used to compute policy_size = NUM_POLICY_CHANNELS * G * G.
    const NUM_POLICY_CHANNELS: usize;

    /// Spatial grid dimension for NN encoding (runtime, may differ from physical board size).
    fn grid_size(&self) -> usize;

    /// Total size of the board tensor: BOARD_CHANNELS * grid_size * grid_size.
    fn board_tensor_size(&self) -> usize {
        Self::BOARD_CHANNELS * self.grid_size() * self.grid_size()
    }

    /// Total size of the policy output vector: NUM_POLICY_CHANNELS * grid_size * grid_size.
    fn policy_size(&self) -> usize {
        Self::NUM_POLICY_CHANNELS * self.grid_size() * self.grid_size()
    }

    /// Encode the board state into a flat tensor and reserve vector.
    fn encode_board(&self, board_out: &mut [f32], reserve_out: &mut [f32]);

    /// Get the legal move mask (policy_size()) and indexed moves.
    /// Each move is paired with a `PolicyIndex` describing how to compute its prior
    /// from the flat policy vector.
    fn get_legal_move_mask(&mut self) -> (Vec<f32>, Vec<(PolicyIndex, Self::Move)>);
}

// Keep the old name as an alias during migration.
pub trait GameEngine: NNGame {}
impl<T: NNGame> GameEngine for T {}
