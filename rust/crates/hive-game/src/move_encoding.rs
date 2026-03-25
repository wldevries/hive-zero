/// Encode/decode Hive moves for neural network policy output.
/// Must produce identical indices to Python move_encoder.py.
///
/// Policy layout: 11 channels x 23 x 23 = 5,819 total policy logits.
/// Channel = piece index within current player (0-10), for both placement and movement.
/// Destination cell stores the logit. No direction encoding — piece identity is the channel.
///
/// Channel mapping (matches Piece::linear_index() % PIECES_PER_PLAYER):
///   0: Queen, 1: Spider1, 2: Spider2, 3: Beetle1, 4: Beetle2,
///   5: Grasshopper1, 6: Grasshopper2, 7: Grasshopper3,
///   8: Ant1, 9: Ant2, 10: Ant3

use crate::board::{GRID_SIZE, hex_to_grid};
use crate::hex::Hex;
use crate::piece::{Piece, PIECES_PER_PLAYER};
use crate::game::{Game, Move};

/// Number of policy channels.
pub const NUM_POLICY_CHANNELS: usize = 11;
/// Total policy size: 11 * 23 * 23 = 5819.
pub const POLICY_SIZE: usize = NUM_POLICY_CHANNELS * GRID_SIZE * GRID_SIZE;

/// Flat index into policy vector.
#[inline]
fn policy_index(channel: usize, row: usize, col: usize) -> usize {
    channel * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col
}

/// Channel for a piece: its index within the current player (0-10).
#[inline]
fn piece_channel(piece: Piece) -> usize {
    piece.linear_index() % PIECES_PER_PLAYER
}

/// Encode a move as a flat policy index. Returns None if out of grid bounds.
pub fn encode_move_checked(piece: Piece, from: Option<Hex>, to: Hex) -> Option<usize> {
    let (dest_row, dest_col) = hex_to_grid(to)?;
    let channel = piece_channel(piece);
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
pub fn get_legal_move_mask(game: &mut Game) -> (Vec<f32>, Vec<(usize, Move)>) {
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
        assert_eq!(ch, 0); // Queen = channel 0
        assert_eq!(row, 11); // center
        assert_eq!(col, 11);
    }

    #[test]
    fn test_encode_movement() {
        // Queen: channel 0, regardless of direction or distance
        let queen = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let idx = encode_move(queen, Some((1, 0)), (0, 0));
        let (ch, row, col) = decode_move(idx);
        assert_eq!(ch, 0);
        assert_eq!(row, 11);
        assert_eq!(col, 11);

        // Ant2: channel 9
        let ant2 = Piece::new(PieceColor::White, PieceType::Ant, 2);
        let idx2 = encode_move(ant2, Some((5, 3)), (0, 0)); // long-distance ant move
        let (ch2, _, _) = decode_move(idx2);
        assert_eq!(ch2, 9);

        // Grasshopper2: channel 6
        let gh2 = Piece::new(PieceColor::White, PieceType::Grasshopper, 2);
        let idx3 = encode_move(gh2, Some((3, 0)), (0, 0));
        let (ch3, _, _) = decode_move(idx3);
        assert_eq!(ch3, 6);
    }

    #[test]
    fn test_no_collision_same_dest() {
        // Two different pieces moving to same destination must get different indices
        let ant1 = Piece::new(PieceColor::White, PieceType::Ant, 1);
        let ant2 = Piece::new(PieceColor::White, PieceType::Ant, 2);
        let gh1 = Piece::new(PieceColor::White, PieceType::Grasshopper, 1);
        let dest = (0, 0);

        let idx_ant1 = encode_move(ant1, Some((5, 0)), dest);
        let idx_ant2 = encode_move(ant2, Some((3, 2)), dest);
        let idx_gh1 = encode_move(gh1, Some((3, 0)), dest);

        assert_ne!(idx_ant1, idx_ant2);
        assert_ne!(idx_ant1, idx_gh1);
        assert_ne!(idx_ant2, idx_gh1);
    }

    #[test]
    fn test_black_same_channel_as_white() {
        // Black piece should use same channel as equivalent white piece (% PIECES_PER_PLAYER)
        let white_ant1 = Piece::new(PieceColor::White, PieceType::Ant, 1);
        let black_ant1 = Piece::new(PieceColor::Black, PieceType::Ant, 1);
        let dest = (0, 0);
        let idx_w = encode_move(white_ant1, Some((1, 0)), dest);
        let idx_b = encode_move(black_ant1, Some((2, 0)), dest);
        assert_eq!(idx_w, idx_b); // same channel, same dest
    }

    #[test]
    fn test_policy_size() {
        assert_eq!(POLICY_SIZE, 5819);
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
