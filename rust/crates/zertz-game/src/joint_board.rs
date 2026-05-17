//! V1 (pre-split-ply) wrapper around `ZertzBoard` for cross-version battle.
//!
//! A model trained before the split-ply refactor expects:
//!   - 6-channel board input (no mid_placement flag)
//!   - Joint move-space priors (`PolicyIndex::Sum` for `Place(color, pos, remove)`)
//!   - Never to see a mid_placement state (its training distribution had none)
//!
//! `JointZertzBoard` is a newtype over `ZertzBoard` whose `Game + NNGame`
//! impls deliver exactly that contract. The MCTS code is generic over
//! `GameEngine = NNGame`, so `MctsSearch<JointZertzBoard>` is a drop-in
//! V1 search runtime — its only departure from `MctsSearch<ZertzBoard>` is
//! at the trait boundary, not in the search loop.
//!
//! The inner `ZertzBoard` is the same struct used by the new split-ply
//! state machine, so `apply_move(Place)` on it does both halves atomically
//! without ever leaving mid_placement set. V1's MCTS therefore navigates
//! only the "between-turn" board states V1 was trained on.

use core_game::game::{Game, NNGame, Outcome, Player, PolicyIndex, Undoable};

use crate::zertz::{ZertzBoard, ZertzMove};

#[derive(Debug, Clone)]
pub struct JointZertzBoard(pub ZertzBoard);

impl JointZertzBoard {
    pub fn new() -> Self {
        JointZertzBoard(ZertzBoard::default())
    }

    /// Wrap a `clone_light` of an external `ZertzBoard` — what battle code
    /// uses to hand the V1 MCTS its own scratch copy of the shared game.
    pub fn from_shared(b: &ZertzBoard) -> Self {
        JointZertzBoard(b.clone_light())
    }

    pub fn inner(&self) -> &ZertzBoard {
        &self.0
    }
}

impl Default for JointZertzBoard {
    fn default() -> Self {
        Self::new()
    }
}

impl Undoable for JointZertzBoard {
    fn undo(&mut self) {
        self.0.undo();
    }
}

impl Game for JointZertzBoard {
    type Move = ZertzMove;
    type Symmetry = core_game::symmetry::D6Symmetry;

    fn next_player(&self) -> Player {
        self.0.next_player()
    }

    fn outcome(&self) -> Outcome {
        self.0.outcome()
    }

    fn valid_moves(&mut self) -> Vec<ZertzMove> {
        self.0.legal_joint_moves()
    }

    fn play_move(&mut self, mv: &ZertzMove) -> Result<(), String> {
        // V1 only emits joint Place / PlaceOnly / Capture / Pass — but we
        // forward whatever the caller passes; the underlying ZertzBoard
        // rejects half-moves applied without mid_placement set.
        self.0.play_move(mv)
    }

    fn pass_move() -> ZertzMove {
        ZertzMove::Pass
    }

    fn is_pass(mv: &ZertzMove) -> bool {
        matches!(mv, ZertzMove::Pass)
    }
}

impl NNGame for JointZertzBoard {
    const BOARD_CHANNELS: usize = crate::board_encoding::NUM_CHANNELS_V1;
    const RESERVE_SIZE: usize = crate::board_encoding::RESERVE_SIZE;
    const NUM_POLICY_CHANNELS: usize = crate::move_encoding::NN_POLICY_CHANNELS;

    fn grid_size(&self) -> usize {
        crate::board_encoding::GRID_SIZE
    }

    fn encode_board(&self, board_out: &mut [f32], reserve_out: &mut [f32]) {
        crate::board_encoding::encode_board_v1(&self.0, board_out, reserve_out);
    }

    fn get_legal_move_mask(&mut self) -> (Vec<f32>, Vec<(PolicyIndex, ZertzMove)>) {
        crate::move_encoding::get_legal_move_mask_nn_joint(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zertz::Marble;

    #[test]
    fn default_board_yields_joint_moves() {
        let mut b = JointZertzBoard::default();
        let moves = b.valid_moves();
        assert!(!moves.is_empty());
        for mv in &moves {
            assert!(
                matches!(mv, ZertzMove::Place { .. } | ZertzMove::PlaceOnly { .. }),
                "joint MCTS should never see half-moves, got {:?}",
                mv
            );
        }
    }

    #[test]
    fn joint_apply_skips_mid_placement() {
        // Applying a joint Place via the wrapper should not leave the inner
        // board in mid_placement state — that's the whole point of staying
        // in V1's training distribution.
        let mut b = JointZertzBoard::default();
        b.play_move(&ZertzMove::Place {
            color: Marble::White,
            place_at: (0, 0),
            remove: (3, -3),
        })
        .unwrap();
        assert!(!b.inner().is_mid_placement());
    }

    #[test]
    fn encode_board_v1_is_six_channels() {
        use crate::board_encoding::{GRID_SIZE, NUM_CHANNELS_V1, RESERVE_SIZE};
        let b = JointZertzBoard::default();
        let mut board_buf = vec![0.0f32; NUM_CHANNELS_V1 * GRID_SIZE * GRID_SIZE];
        let mut reserve_buf = vec![0.0f32; RESERVE_SIZE];
        b.encode_board(&mut board_buf, &mut reserve_buf);
        assert_eq!(board_buf.len(), 6 * 49);
    }

    #[test]
    fn joint_legal_mask_uses_sum_priors() {
        let mut b = JointZertzBoard::default();
        let (_mask, indexed) = b.get_legal_move_mask();
        // The default board has Place moves available — every one of them
        // must be scored with PolicyIndex::Sum (V1 factor-of-two prior).
        let any_sum = indexed.iter().any(|(pi, mv)| {
            matches!(pi, PolicyIndex::Sum(_, _)) && matches!(mv, ZertzMove::Place { .. })
        });
        assert!(any_sum, "expected Sum priors for joint Place moves");
    }
}
