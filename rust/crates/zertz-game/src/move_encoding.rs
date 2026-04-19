/// Move encoding: maps ZertzMove to/from a flat policy index.
///
/// Layout:
///   Place:     color(3) * 37 * 37 + place_at(37) * 37 + remove(37)   → [0, 4107)
///   PlaceOnly: 4107 + color(3) * 37 + place_at(37)                   → [4107, 4218)
///   Capture:   4218 + direction(6) * 37 + from(37)                   → [4218, 4440)
///
/// Direction indices match DIRECTIONS: E=0, NE=1, NW=2, W=3, SW=4, SE=5.
/// All Zertz captures jump exactly 2 cells in one of these 6 directions.

use core_game::game::PolicyIndex;

use crate::hex::{self, hex_to_index, hex_to_grid, DIRECTIONS, GRID_SIZE};
use crate::zertz::{ZertzBoard, ZertzMove};
#[cfg(test)]
use crate::zertz::Marble;

const BOARD_SIZE: usize = hex::BOARD_SIZE;
const NUM_DIRECTIONS: usize = 6;

pub const NN_POLICY_CHANNELS: usize = 10; // 4 place + 6 cap_dir
pub const NN_POLICY_SIZE: usize = NN_POLICY_CHANNELS * GRID_SIZE * GRID_SIZE; // 490
pub const PLACE_HEAD_SIZE: usize = 4 * GRID_SIZE * GRID_SIZE; // 196
pub const CAP_HEAD_SIZE: usize = 6 * GRID_SIZE * GRID_SIZE;   // 294

#[inline]
fn hex_to_grid_cell(h: crate::hex::Hex) -> usize {
    let (row, col) = hex_to_grid(h);
    row * GRID_SIZE + col
}

const PLACE_OFFSET: usize = 0;
const PLACE_ONLY_OFFSET: usize = 3 * BOARD_SIZE * BOARD_SIZE; // 4107
const CAPTURE_OFFSET: usize = PLACE_ONLY_OFFSET + 3 * BOARD_SIZE; // 4218

pub const POLICY_SIZE: usize = CAPTURE_OFFSET + NUM_DIRECTIONS * BOARD_SIZE; // 4440

/// Find the direction index for a capture jump from `from` to `to`.
/// All Zertz captures jump exactly 2 cells in one of 6 directions.
pub fn capture_direction(from: crate::hex::Hex, to: crate::hex::Hex) -> usize {
    for (d, &dir) in DIRECTIONS.iter().enumerate() {
        if to.0 == from.0 + 2 * dir.0 && to.1 == from.1 + 2 * dir.1 {
            return d;
        }
    }
    panic!("invalid capture jump: no direction from {:?} to {:?}", from, to);
}

/// Encode a ZertzMove as a policy index.
pub fn encode_move(mv: &ZertzMove) -> usize {
    match *mv {
        ZertzMove::Place {
            color,
            place_at,
            remove,
        } => {
            PLACE_OFFSET
                + color.index() * BOARD_SIZE * BOARD_SIZE
                + hex_to_index(place_at) * BOARD_SIZE
                + hex_to_index(remove)
        }
        ZertzMove::PlaceOnly { color, place_at } => {
            PLACE_ONLY_OFFSET + color.index() * BOARD_SIZE + hex_to_index(place_at)
        }
        ZertzMove::Capture { jumps, .. } => {
            let from = jumps[0].0;
            let to = jumps[0].2;
            let d = capture_direction(from, to);
            CAPTURE_OFFSET + d * BOARD_SIZE + hex_to_index(from)
        }
        ZertzMove::Pass => 0, // Pass is never generated; return dummy index
    }
}

/// Get legal move mask and indexed moves for a board position.
/// Returns (mask[POLICY_SIZE], Vec<(policy_index, move)>).
pub fn get_legal_move_mask(board: &ZertzBoard) -> ([f32; POLICY_SIZE], Vec<(usize, ZertzMove)>) {
    let mut mask = [0.0f32; POLICY_SIZE];
    let moves = board.legal_moves();
    let mut indexed = Vec::with_capacity(moves.len());

    for mv in moves {
        let idx = encode_move(&mv);
        mask[idx] = 1.0;
        indexed.push((idx, mv));
    }

    (mask, indexed)
}

/// Encode a visit distribution as a flat NN_POLICY_SIZE=490 vector for training.
///
/// Layout: [place_W(49), place_G(49), place_B(49), remove(49),
///          cap_E(49), cap_NE(49), cap_NW(49), cap_W(49), cap_SW(49), cap_SE(49)]
///
/// For Place moves, probability is added to both the color/position cell and the
/// remove-ring cell (marginal distributions). For PlaceOnly and Capture, one cell only.
pub fn encode_distribution_nn(dist: &[(ZertzMove, f32)]) -> Vec<f32> {
    const G2: usize = GRID_SIZE * GRID_SIZE;
    let mut policy = vec![0.0f32; NN_POLICY_SIZE];
    for &(mv, prob) in dist {
        match mv {
            ZertzMove::Place { color, place_at, remove } => {
                let a = color.index() * G2 + hex_to_grid_cell(place_at);
                let b = 3 * G2 + hex_to_grid_cell(remove);
                policy[a] += prob;
                policy[b] += prob;
            }
            ZertzMove::PlaceOnly { color, place_at } => {
                let a = color.index() * G2 + hex_to_grid_cell(place_at);
                policy[a] += prob;
            }
            ZertzMove::Capture { jumps, .. } => {
                let from = jumps[0].0;
                let to = jumps[0].2;
                let d = capture_direction(from, to);
                let a = (4 + d) * G2 + hex_to_grid_cell(from);
                policy[a] += prob;
            }
            ZertzMove::Pass => {}
        }
    }
    policy
}

/// Get legal move mask and indexed moves for the main MCTS (NNGame interface).
/// Policy layout: [place_W(49), place_G(49), place_B(49), remove(49),
///                 cap_dir_E(49), cap_dir_NE(49), ..., cap_dir_SE(49)] = 490 total.
/// Returns (mask[NN_POLICY_SIZE], Vec<(PolicyIndex, move)>).
pub fn get_legal_move_mask_nn(board: &ZertzBoard) -> (Vec<f32>, Vec<(PolicyIndex, ZertzMove)>) {
    const G2: usize = GRID_SIZE * GRID_SIZE;
    let mut mask = vec![0.0f32; NN_POLICY_SIZE];
    let moves = board.legal_moves();
    let mut indexed = Vec::with_capacity(moves.len());

    for mv in moves {
        let pi = match mv {
            ZertzMove::Place { color, place_at, remove } => {
                let a = color.index() * G2 + hex_to_grid_cell(place_at);
                let b = 3 * G2 + hex_to_grid_cell(remove);
                mask[a] = 1.0;
                mask[b] = 1.0;
                PolicyIndex::Sum(a, b)
            }
            ZertzMove::PlaceOnly { color, place_at } => {
                let a = color.index() * G2 + hex_to_grid_cell(place_at);
                mask[a] = 1.0;
                PolicyIndex::Single(a)
            }
            ZertzMove::Capture { jumps, .. } => {
                let from = jumps[0].0;
                let to = jumps[0].2;
                let d = capture_direction(from, to);
                let a = (4 + d) * G2 + hex_to_grid_cell(from);
                mask[a] = 1.0;
                PolicyIndex::Single(a)
            }
            ZertzMove::Pass => continue,
        };
        indexed.push((pi, mv));
    }

    (mask, indexed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_place_encoding() {
        use crate::hex::index_to_hex;
        let place_hex = index_to_hex(5);
        let remove_hex = index_to_hex(10);
        let mv = ZertzMove::Place {
            color: Marble::White,
            place_at: place_hex,
            remove: remove_hex,
        };
        let idx = encode_move(&mv);
        assert!(idx < PLACE_ONLY_OFFSET);
        assert_eq!(idx, 0 * 37 * 37 + 5 * 37 + 10);
    }

    #[test]
    fn test_place_only_encoding() {
        use crate::hex::index_to_hex;
        let place_hex = index_to_hex(3);
        let mv = ZertzMove::PlaceOnly {
            color: Marble::Grey,
            place_at: place_hex,
        };
        let idx = encode_move(&mv);
        assert!(idx >= PLACE_ONLY_OFFSET && idx < CAPTURE_OFFSET);
        assert_eq!(idx, PLACE_ONLY_OFFSET + 1 * 37 + 3);
    }

    #[test]
    fn test_capture_encoding() {
        use crate::hex::{index_to_hex, hex_add};
        // Build a valid 2-step capture: from + 2*dir = to
        let from = index_to_hex(10);
        let dir = DIRECTIONS[0]; // E direction
        let mid = hex_add(from, dir);
        let to = hex_add(mid, dir);
        let mv = ZertzMove::capture_single(from, mid, to);
        let idx = encode_move(&mv);
        assert!(idx >= CAPTURE_OFFSET && idx < POLICY_SIZE);
        let expected_d = 0; // E direction
        assert_eq!(idx, CAPTURE_OFFSET + expected_d * 37 + 10);
    }

    #[test]
    fn test_legal_move_mask() {
        let board = ZertzBoard::default();
        let (mask, indexed) = get_legal_move_mask(&board);
        assert!(!indexed.is_empty());
        for &(idx, _) in &indexed {
            assert!(idx < POLICY_SIZE);
            assert_eq!(mask[idx], 1.0);
        }
    }
}
