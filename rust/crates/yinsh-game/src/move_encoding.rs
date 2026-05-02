/// YINSH move → policy index encoding.
///
/// Policy channels (59, all on the 11x11 grid):
///   0:     PlaceRing destination                 (Setup phase)
///   1-3:   ClaimRow row start, dir = 0,1,2       (ClaimRow phase, partner A)
///   4:     ClaimRow ring target                  (ClaimRow phase, partner B)
///   5-58:  MoveRing encoding                     (Normal phase)
///           Channel = CH_MOVE_BASE + dir_idx * MAX_RING_DIST + (dist - 1)
///           Value at source cell; dir_idx in 0..6 (DIRECTIONS order), dist in 1..=9
///
/// `ClaimRow` is a joint move: its prior is `Sum(row_idx, ring_idx)` — the
/// network's row-channel logit at the row's start cell plus the ring-channel
/// logit at the chosen ring cell. PlaceRing and MoveRing use `Single`.

use core_game::game::PolicyIndex;

use crate::board::{YinshBoard, YinshMove};
use crate::board_encoding::cell_to_grid;
use crate::hex::{DIRECTIONS, GRID_SIZE, index_to_cell};

pub const MAX_RING_DIST: usize = 9;
pub const NUM_DIRS: usize = 6; // matches DIRECTIONS.len()
pub const NUM_MOVE_CHANNELS: usize = NUM_DIRS * MAX_RING_DIST; // 54
pub const NUM_SINGLE_CHANNELS: usize = 5; // PlaceRing + 3×row + ring
pub const NUM_POLICY_CHANNELS: usize = NUM_SINGLE_CHANNELS + NUM_MOVE_CHANNELS; // 59
pub const POLICY_SIZE: usize = NUM_POLICY_CHANNELS * GRID_SIZE * GRID_SIZE; // 7139

const CH_PLACE_RING: usize = 0;
const CH_CLAIM_ROW_BASE: usize = 1; // +dir (0..3) for row start cell
const CH_CLAIM_RING: usize = 4;     // ring partner channel
pub const CH_MOVE_BASE: usize = 5;  // + dir_idx*MAX_RING_DIST + (dist-1)

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
            let (fc, fr) = index_to_cell(from);
            let (tc, tr) = index_to_cell(to);
            let dc = (tc as i8 - fc as i8).signum();
            let dr = (tr as i8 - fr as i8).signum();
            let dir_idx = DIRECTIONS.iter().position(|&d| d == (dc, dr))
                .expect("MoveRing: direction not in DIRECTIONS");
            let dist = (tc as i8 - fc as i8).abs().max((tr as i8 - fr as i8).abs()) as usize;
            debug_assert!(dist >= 1 && dist <= MAX_RING_DIST, "dist {} out of range", dist);
            PolicyIndex::Single(
                ch_offset(CH_MOVE_BASE + dir_idx * MAX_RING_DIST + (dist - 1)) + cell_to_grid(from)
            )
        }
        YinshMove::ClaimRow { start, dir, ring } => {
            // Joint prior: row-channel logit at `start` + ring-channel logit at `ring`.
            PolicyIndex::Sum(
                ch_offset(CH_CLAIM_ROW_BASE + dir) + cell_to_grid(start),
                ch_offset(CH_CLAIM_RING) + cell_to_grid(ring),
            )
        }
        YinshMove::Pass => PolicyIndex::Single(0),
    }
}

/// Legal move mask and indexed moves for a board position.
/// The mask has 1.0 at every policy index referenced by a legal move.
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
            PolicyIndex::DotProduct { .. } => {}
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
        // E5 (4,4) → E8 (4,7): direction (0,1), dist=3
        let mv = YinshMove::MoveRing { from: cell_index(4, 4), to: cell_index(4, 7) };
        let dir_idx = DIRECTIONS.iter().position(|&d| d == (0, 1)).unwrap();
        let dist = 3usize;
        match encode_move(&mv) {
            PolicyIndex::Single(i) => {
                let expected = ch_offset(CH_MOVE_BASE + dir_idx * MAX_RING_DIST + (dist - 1))
                    + cell_to_grid(cell_index(4, 4));
                assert_eq!(i, expected);
            }
            _ => panic!("expected Single"),
        }
    }

    #[test]
    fn test_move_ring_different_dirs_differ() {
        // Same source, different directions → different channels
        let from = cell_index(4, 4);
        let mv1 = YinshMove::MoveRing { from, to: cell_index(4, 5) }; // dir (0,1), dist 1
        let mv2 = YinshMove::MoveRing { from, to: cell_index(5, 4) }; // dir (1,0), dist 1
        let PolicyIndex::Single(i1) = encode_move(&mv1) else { panic!() };
        let PolicyIndex::Single(i2) = encode_move(&mv2) else { panic!() };
        assert_ne!(i1, i2);
        // Both at same source grid cell, but different channels
        let g2 = GRID_SIZE * GRID_SIZE;
        assert_eq!(i1 % g2, i2 % g2); // same grid cell
        assert_ne!(i1 / g2, i2 / g2); // different channels
    }

    #[test]
    fn test_move_ring_different_dists_differ() {
        // Same source and direction, different distances → different channels
        let from = cell_index(4, 4);
        let mv1 = YinshMove::MoveRing { from, to: cell_index(4, 5) }; // dist 1
        let mv2 = YinshMove::MoveRing { from, to: cell_index(4, 6) }; // dist 2
        let PolicyIndex::Single(i1) = encode_move(&mv1) else { panic!() };
        let PolicyIndex::Single(i2) = encode_move(&mv2) else { panic!() };
        assert_ne!(i1, i2);
    }

    #[test]
    fn test_claim_row_encoding() {
        // ClaimRow returns Sum(row_offset, ring_offset).
        let mv = YinshMove::ClaimRow {
            start: cell_index(4, 4),
            dir: 2,
            ring: cell_index(3, 2),
        };
        match encode_move(&mv) {
            PolicyIndex::Sum(a, b) => {
                assert_eq!(a, ch_offset(CH_CLAIM_ROW_BASE + 2) + cell_to_grid(cell_index(4, 4)));
                assert_eq!(b, ch_offset(CH_CLAIM_RING) + cell_to_grid(cell_index(3, 2)));
            }
            _ => panic!("expected Sum"),
        }
    }

    /// A run of 6+ markers offers two row start positions, paired with each
    /// of the player's rings — each combination must encode to a distinct
    /// (row, ring) pair so the network can score them jointly.
    #[test]
    fn test_claim_row_six_markers_distinct_pairs() {
        use crate::board::{Cell, Phase, YinshBoard};
        use crate::hex::ROW_DIRS;
        use core_game::game::Player;

        let mut board = YinshBoard::new();
        let dir = ROW_DIRS.iter().position(|&d| d == (0, 1)).unwrap();
        let col = 4u8;
        for row in 2..=7u8 {
            board.cells[crate::hex::cell_index(col, row)] = Cell::WhiteMarker;
        }
        // Place a couple of white rings so ClaimRow has ring options.
        let r1 = crate::hex::cell_index(0, 2);
        let r2 = crate::hex::cell_index(0, 3);
        board.cells[r1] = Cell::WhiteRing;
        board.cells[r2] = Cell::WhiteRing;
        board.next_player = Player::Player1;
        board.phase = Phase::ClaimRow;
        board.original_mover = Some(Player::Player1);
        board.pending_rows = board.find_rows(Player::Player1);

        assert_eq!(board.pending_rows.len(), 2);
        let moves = board.legal_moves();
        // 2 rows × 2 rings = 4 joint claims.
        assert_eq!(moves.len(), 4);

        // All four must encode to distinct (row_idx, ring_idx) pairs.
        let mut pairs = Vec::new();
        for mv in &moves {
            let YinshMove::ClaimRow { .. } = mv else { panic!() };
            let PolicyIndex::Sum(a, b) = encode_move(mv) else { panic!("expected Sum") };
            assert_eq!(a / (GRID_SIZE * GRID_SIZE), CH_CLAIM_ROW_BASE + dir);
            assert_eq!(b / (GRID_SIZE * GRID_SIZE), CH_CLAIM_RING);
            pairs.push((a, b));
        }
        pairs.sort();
        pairs.dedup();
        assert_eq!(pairs.len(), 4);
    }

    #[test]
    fn test_legal_mask_initial() {
        let mut board = YinshBoard::new();
        let (mask, indexed) = get_legal_move_mask(&mut board);
        assert_eq!(indexed.len(), crate::hex::BOARD_SIZE);
        let ones: usize = mask.iter().filter(|&&v| v == 1.0).count();
        assert_eq!(ones, crate::hex::BOARD_SIZE);
    }
}
