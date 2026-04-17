/// YINSH move → policy index encoding.
///
/// Policy channels (7, all on the 11x11 grid):
///   0: PlaceRing destination               (Setup phase)
///   1: MoveRing from                       (Normal phase, paired with ch 2)
///   2: MoveRing to                         (Normal phase, paired with ch 1)
///   3: RemoveRow start, dir = 0 (vertical)
///   4: RemoveRow start, dir = 1 (horizontal)
///   5: RemoveRow start, dir = 2 (diagonal)
///   6: RemoveRing target                   (RemoveRing phase)
///
/// MoveRing uses `PolicyIndex::Sum` — the prior for (from, to) is
/// `policy[from_idx] + policy[to_idx]` where each index is in its own channel.
/// All other moves use `PolicyIndex::Single`.

use core_game::game::PolicyIndex;

use crate::board::{YinshBoard, YinshMove};
use crate::board_encoding::cell_to_grid;
use crate::hex::GRID_SIZE;

pub const NUM_POLICY_CHANNELS: usize = 7;
pub const POLICY_SIZE: usize = NUM_POLICY_CHANNELS * GRID_SIZE * GRID_SIZE;

const CH_PLACE_RING: usize = 0;
const CH_MOVE_FROM: usize = 1;
const CH_MOVE_TO: usize = 2;
const CH_REMOVE_ROW_BASE: usize = 3; // +dir (0..3)
const CH_REMOVE_RING: usize = 6;

#[inline]
fn ch_offset(ch: usize) -> usize {
    ch * GRID_SIZE * GRID_SIZE
}

/// Encode a YinshMove as a `PolicyIndex` for MCTS prior lookup.
pub fn encode_move(mv: &YinshMove) -> PolicyIndex {
    match *mv {
        YinshMove::PlaceRing(idx) => {
            PolicyIndex::Single(ch_offset(CH_PLACE_RING) + cell_to_grid(idx))
        }
        YinshMove::MoveRing { from, to } => {
            PolicyIndex::Sum(
                ch_offset(CH_MOVE_FROM) + cell_to_grid(from),
                ch_offset(CH_MOVE_TO) + cell_to_grid(to),
            )
        }
        YinshMove::RemoveRow { start, dir } => {
            PolicyIndex::Single(ch_offset(CH_REMOVE_ROW_BASE + dir) + cell_to_grid(start))
        }
        YinshMove::RemoveRing(idx) => {
            PolicyIndex::Single(ch_offset(CH_REMOVE_RING) + cell_to_grid(idx))
        }
        YinshMove::Pass => PolicyIndex::Single(0),
    }
}

/// Legal move mask and indexed moves for a board position.
/// The mask has 1.0 at every policy index referenced by a legal move
/// (both halves of Sum-encoded moves).
pub fn get_legal_move_mask(board: &mut YinshBoard) -> (Vec<f32>, Vec<(PolicyIndex, YinshMove)>) {
    let moves = board.legal_moves();
    let mut mask = vec![0.0f32; POLICY_SIZE];
    let mut indexed = Vec::with_capacity(moves.len());

    for mv in moves {
        let policy_idx = encode_move(&mv);
        match policy_idx {
            PolicyIndex::Single(i) => mask[i] = 1.0,
            PolicyIndex::Sum(a, b) => {
                mask[a] = 1.0;
                mask[b] = 1.0;
            }
            PolicyIndex::DotProduct { .. } => {} // not used for yinsh
        }
        indexed.push((policy_idx, mv));
    }

    (mask, indexed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hex::cell_index;

    #[test]
    fn test_place_ring_encoding() {
        let mv = YinshMove::PlaceRing(cell_index(4, 4));
        match encode_move(&mv) {
            PolicyIndex::Single(i) => {
                assert_eq!(i, ch_offset(CH_PLACE_RING) + cell_to_grid(cell_index(4, 4)));
            }
            _ => panic!("expected Single"),
        }
    }

    #[test]
    fn test_move_ring_encoding() {
        let mv = YinshMove::MoveRing { from: cell_index(4, 4), to: cell_index(4, 7) };
        match encode_move(&mv) {
            PolicyIndex::Sum(a, b) => {
                assert_eq!(a, ch_offset(CH_MOVE_FROM) + cell_to_grid(cell_index(4, 4)));
                assert_eq!(b, ch_offset(CH_MOVE_TO) + cell_to_grid(cell_index(4, 7)));
            }
            _ => panic!("expected Sum"),
        }
    }

    #[test]
    fn test_remove_row_encoding() {
        let mv = YinshMove::RemoveRow { start: cell_index(4, 4), dir: 2 };
        match encode_move(&mv) {
            PolicyIndex::Single(i) => {
                assert_eq!(i, ch_offset(CH_REMOVE_ROW_BASE + 2) + cell_to_grid(cell_index(4, 4)));
            }
            _ => panic!("expected Single"),
        }
    }

    /// A run of 6+ markers offers multiple start positions; each must encode
    /// to a distinct policy index so the network can choose between them.
    #[test]
    fn test_remove_row_six_markers_distinct_indices() {
        use crate::board::{Cell, Phase, YinshBoard};
        use crate::hex::ROW_DIRS;
        use core_game::game::Player;

        let mut board = YinshBoard::new();
        // Manually stage: white to move in RemoveRow phase, with 6 white markers
        // vertically from E3..E8.  dir index for (0,1) is 0.
        let dir = ROW_DIRS.iter().position(|&d| d == (0, 1)).unwrap();
        let col = 4u8;
        for row in 2..=7u8 {
            board.cells[crate::hex::cell_index(col, row)] = Cell::WhiteMarker;
        }
        board.next_player = Player::Player1;
        board.phase = Phase::RemoveRow;
        board.original_mover = Some(Player::Player1);
        board.pending_rows = board.find_rows(Player::Player1);

        // Two valid 5-slice starts: E3 (row=2) and E4 (row=3)
        assert_eq!(board.pending_rows.len(), 2);
        let moves = board.legal_moves();
        assert_eq!(moves.len(), 2);

        // Encode both — indices must differ, same channel, different grid cells
        let idx_a = match encode_move(&moves[0]) {
            PolicyIndex::Single(i) => i,
            _ => panic!(),
        };
        let idx_b = match encode_move(&moves[1]) {
            PolicyIndex::Single(i) => i,
            _ => panic!(),
        };
        assert_ne!(idx_a, idx_b);
        let ch_a = idx_a / (GRID_SIZE * GRID_SIZE);
        let ch_b = idx_b / (GRID_SIZE * GRID_SIZE);
        assert_eq!(ch_a, CH_REMOVE_ROW_BASE + dir);
        assert_eq!(ch_b, CH_REMOVE_ROW_BASE + dir);
    }

    #[test]
    fn test_legal_mask_initial() {
        let mut board = YinshBoard::new();
        let (mask, indexed) = get_legal_move_mask(&mut board);
        // In Setup, every empty cell is a legal PlaceRing
        assert_eq!(indexed.len(), crate::hex::BOARD_SIZE);
        let ones: usize = mask.iter().filter(|&&v| v == 1.0).count();
        assert_eq!(ones, crate::hex::BOARD_SIZE);
    }
}
