/// Encode/decode Hive moves for neural network policy output.
/// Must produce identical indices to Python move_encoder.py.

use crate::hex::{Hex, DIRECTIONS, hex_sub};
use crate::board::{GRID_SIZE, hex_to_grid};
use crate::piece::Piece;
use crate::game::{Game, Move};

/// Number of policy channels.
pub const NUM_POLICY_CHANNELS: usize = 12;
/// Total policy size: 12 * 23 * 23 = 6348.
pub const POLICY_SIZE: usize = NUM_POLICY_CHANNELS * GRID_SIZE * GRID_SIZE;

/// Policy channel for placement by piece type.
const PIECE_TYPE_CHANNEL: [usize; 5] = [7, 8, 9, 10, 11]; // Q, S, B, G, A

/// Flat index into policy vector.
#[inline]
fn policy_index(channel: usize, row: usize, col: usize) -> usize {
    channel * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col
}

/// Encode a move as a flat policy index. Returns None if out of grid bounds.
pub fn encode_move_checked(piece: Piece, from: Option<Hex>, to: Hex) -> Option<usize> {
    let (dest_row, dest_col) = hex_to_grid(to)?;

    let channel = if from.is_none() {
        // Placement move
        PIECE_TYPE_CHANNEL[piece.piece_type().index()]
    } else {
        // Movement: direction from dest to source
        let diff = hex_sub(from.unwrap(), to);
        let mut ch = 6; // default: stacking (beetle)
        for (i, &d) in DIRECTIONS.iter().enumerate() {
            if diff == d {
                ch = i;
                break;
            }
        }
        ch
    };

    Some(policy_index(channel, dest_row, dest_col))
}

/// Encode a move as a flat policy index. Panics if out of grid bounds.
pub fn encode_move(piece: Piece, from: Option<Hex>, to: Hex) -> usize {
    encode_move_checked(piece, from, to).expect("destination out of grid")
}

/// Decode a flat policy index into (channel, row, col).
pub fn decode_move(index: usize) -> (usize, usize, usize) {
    let channel = index / (GRID_SIZE * GRID_SIZE);
    let remainder = index % (GRID_SIZE * GRID_SIZE);
    let row = remainder / GRID_SIZE;
    let col = remainder % GRID_SIZE;
    (channel, row, col)
}

/// Encode a Move struct as a policy index. Returns None if out of grid.
pub fn encode_game_move(mv: &Move) -> Option<usize> {
    let piece = mv.piece?;
    let to = mv.to?;
    encode_move_checked(piece, mv.from, to)
}

/// Create a binary mask over the policy space for legal moves.
/// Returns (mask, indexed_moves) where indexed_moves maps policy indices to moves.
pub fn get_legal_move_mask(game: &Game) -> (Vec<f32>, Vec<(usize, Move)>) {
    let mut mask = vec![0.0f32; POLICY_SIZE];
    let valid_moves = game.valid_moves();
    let mut indexed_moves = Vec::with_capacity(valid_moves.len());

    for mv in valid_moves {
        if let Some(idx) = encode_game_move(&mv) {
            if idx < POLICY_SIZE {
                mask[idx] = 1.0;
                indexed_moves.push((idx, mv));
            }
        }
    }

    (mask, indexed_moves)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::{PieceColor, PieceType};

    #[test]
    fn test_encode_placement() {
        let piece = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let idx = encode_move(piece, None, (0, 0));
        let (ch, row, col) = decode_move(idx);
        assert_eq!(ch, 7); // Queen placement channel
        assert_eq!(row, 11); // center
        assert_eq!(col, 11);
    }

    #[test]
    fn test_encode_movement_east() {
        let piece = Piece::new(PieceColor::White, PieceType::Queen, 1);
        // Move from (1,0) to (0,0) - piece came from the East
        let idx = encode_move(piece, Some((1, 0)), (0, 0));
        let (ch, row, col) = decode_move(idx);
        assert_eq!(ch, 0); // E direction
        assert_eq!(row, 11);
        assert_eq!(col, 11);
    }

    #[test]
    fn test_encode_stacking() {
        let piece = Piece::new(PieceColor::White, PieceType::Beetle, 1);
        // Move from (0,1) to (0,0) - SE to origin... diff is (0,1) which is SE direction
        // Wait, diff = from - to = (0,1) - (0,0) = (0,1) = SE
        let idx = encode_move(piece, Some((0, 1)), (0, 0));
        let (ch, _, _) = decode_move(idx);
        assert_eq!(ch, 5); // SE direction channel

        // If diff doesn't match any direction (e.g. grasshopper jump), use channel 6
        let idx2 = encode_move(piece, Some((3, 0)), (0, 0));
        let (ch2, _, _) = decode_move(idx2);
        assert_eq!(ch2, 6); // stacking/non-adjacent
    }

    #[test]
    fn test_policy_size() {
        assert_eq!(POLICY_SIZE, 6348);
    }

    #[test]
    fn test_roundtrip() {
        for ch in 0..NUM_POLICY_CHANNELS {
            for r in 0..GRID_SIZE {
                for c in 0..GRID_SIZE {
                    let idx = policy_index(ch, r, c);
                    let (ch2, r2, c2) = decode_move(idx);
                    assert_eq!((ch, r, c), (ch2, r2, c2));
                }
            }
        }
    }
}
