/// Move encoding: maps ZertzMove to/from a flat policy index.
///
/// Layout:
///   Place:     color(3) * 37 * 37 + place_at(37) * 37 + remove(37)   → [0, 4107)
///   PlaceOnly: 4107 + color(3) * 37 + place_at(37)                   → [4107, 4218)
///   Capture:   4218 + from(37) * 37 + final_to(37)                   → [4218, 5587)

use crate::zertz::{BOARD_SIZE, ZertzBoard, ZertzMove};
#[cfg(test)]
use crate::zertz::Marble;

const PLACE_OFFSET: usize = 0;
const PLACE_ONLY_OFFSET: usize = 3 * BOARD_SIZE * BOARD_SIZE; // 4107
const CAPTURE_OFFSET: usize = PLACE_ONLY_OFFSET + 3 * BOARD_SIZE; // 4218

pub const POLICY_SIZE: usize = CAPTURE_OFFSET + BOARD_SIZE * BOARD_SIZE; // 5587

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
                + place_at as usize * BOARD_SIZE
                + remove as usize
        }
        ZertzMove::PlaceOnly { color, place_at } => {
            PLACE_ONLY_OFFSET + color.index() * BOARD_SIZE + place_at as usize
        }
        ZertzMove::Capture { jumps, len } => {
            let from = jumps[0].0 as usize;
            let final_to = jumps[len as usize - 1].2 as usize;
            CAPTURE_OFFSET + from * BOARD_SIZE + final_to
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_place_encoding() {
        let mv = ZertzMove::Place {
            color: Marble::White,
            place_at: 5,
            remove: 10,
        };
        let idx = encode_move(&mv);
        assert!(idx < PLACE_ONLY_OFFSET);
        assert_eq!(idx, 0 * 37 * 37 + 5 * 37 + 10);
    }

    #[test]
    fn test_place_only_encoding() {
        let mv = ZertzMove::PlaceOnly {
            color: Marble::Grey,
            place_at: 3,
        };
        let idx = encode_move(&mv);
        assert!(idx >= PLACE_ONLY_OFFSET && idx < CAPTURE_OFFSET);
        assert_eq!(idx, PLACE_ONLY_OFFSET + 1 * 37 + 3);
    }

    #[test]
    fn test_capture_encoding() {
        let mv = ZertzMove::capture_single(10, 15, 20);
        let idx = encode_move(&mv);
        assert!(idx >= CAPTURE_OFFSET && idx < POLICY_SIZE);
        assert_eq!(idx, CAPTURE_OFFSET + 10 * 37 + 20);
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
