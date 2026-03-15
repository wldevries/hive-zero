/// Encode Hive board state as a fixed-size tensor for neural network input.
/// Must produce bitwise-identical output to Python board_encoder.py.

use crate::board::{GRID_SIZE, hex_to_grid};
use crate::game::Game;
use crate::piece::{PieceColor, ALL_PIECE_TYPES, PIECE_COUNTS};

/// Number of board encoding channels.
pub const NUM_CHANNELS: usize = 23;
/// Reserve vector size: 5 piece types x 2 colors.
pub const RESERVE_SIZE: usize = 10;

/// Channel layout:
/// 0-4:   White pieces on top (Q, S, B, G, A) - binary
/// 5-9:   Black pieces on top (Q, S, B, G, A) - binary
/// 10-14: White pieces in stack (count)
/// 15-19: Black pieces in stack (count)
/// 20:    Stack height (normalized /7)
/// 21:    Current player's pieces (1 = current player)
/// 22:    Is current player white (constant plane)

/// Encode a game state into board tensor and reserve vector.
/// Board tensor shape: (NUM_CHANNELS, GRID_SIZE, GRID_SIZE) = (23, 23, 23)
/// Reserve vector shape: (RESERVE_SIZE,) = (10,)
pub fn encode_board(game: &Game, board_out: &mut [f32], reserve_out: &mut [f32]) {
    debug_assert!(board_out.len() == NUM_CHANNELS * GRID_SIZE * GRID_SIZE);
    debug_assert!(reserve_out.len() == RESERVE_SIZE);

    // Zero out
    board_out.fill(0.0);
    reserve_out.fill(0.0);

    let is_white_turn = game.turn_color == PieceColor::White;

    // Iterate over occupied positions
    for (pos, stack) in game.board.iter_occupied() {
        let (row, col) = match hex_to_grid(pos) {
            Some(rc) => rc,
            None => continue,
        };

        // Top piece
        let top = stack.top().unwrap();
        let offset = if top.color() == PieceColor::White { 0 } else { 5 };
        let ch = top.piece_type().index();
        board_out[(offset + ch) * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col] = 1.0;

        // All pieces in stack
        for piece in stack.iter() {
            let offset2 = if piece.color() == PieceColor::White { 10 } else { 15 };
            let ch2 = piece.piece_type().index();
            board_out[(offset2 + ch2) * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col] += 1.0;
        }

        // Stack height
        board_out[20 * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col] =
            stack.height() as f32 / 7.0;

        // Current player's piece
        if (top.color() == PieceColor::White) == is_white_turn {
            board_out[21 * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col] = 1.0;
        }
    }

    // Current player plane
    if is_white_turn {
        let base = 22 * GRID_SIZE * GRID_SIZE;
        for i in 0..(GRID_SIZE * GRID_SIZE) {
            board_out[base + i] = 1.0;
        }
    }

    // Reserve vector
    for (i, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
        let max_count = PIECE_COUNTS[i] as f32;
        if max_count > 0.0 {
            let w_count = game.reserve_count(PieceColor::White, pt) as f32;
            reserve_out[i] = w_count / max_count;
            let b_count = game.reserve_count(PieceColor::Black, pt) as f32;
            reserve_out[5 + i] = b_count / max_count;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::{Piece, PieceType};
    use crate::game::Game;

    #[test]
    fn test_encode_empty_board() {
        let game = Game::new();
        let mut board_tensor = vec![0.0f32; NUM_CHANNELS * GRID_SIZE * GRID_SIZE];
        let mut reserve_vec = vec![0.0f32; RESERVE_SIZE];
        encode_board(&game, &mut board_tensor, &mut reserve_vec);

        // Channel 22 should be all 1.0 (white's turn)
        let base = 22 * GRID_SIZE * GRID_SIZE;
        for i in 0..(GRID_SIZE * GRID_SIZE) {
            assert_eq!(board_tensor[base + i], 1.0);
        }

        // Reserves should all be 1.0 (all pieces in reserve)
        for i in 0..RESERVE_SIZE {
            assert_eq!(reserve_vec[i], 1.0);
        }
    }

    #[test]
    fn test_encode_one_piece() {
        let mut game = Game::new();
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        game.play_move(&crate::game::Move::placement(wq, (0, 0)));
        // Now it's black's turn

        let mut board_tensor = vec![0.0f32; NUM_CHANNELS * GRID_SIZE * GRID_SIZE];
        let mut reserve_vec = vec![0.0f32; RESERVE_SIZE];
        encode_board(&game, &mut board_tensor, &mut reserve_vec);

        // White queen at grid center (11, 11)
        let row = 11;
        let col = 11;
        // Channel 0 = white queen on top
        assert_eq!(
            board_tensor[0 * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col],
            1.0
        );
        // Channel 10 = white queen in stack
        assert_eq!(
            board_tensor[10 * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col],
            1.0
        );
        // Stack height = 1/7
        let h = board_tensor[20 * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col];
        assert!((h - 1.0 / 7.0).abs() < 1e-6);

        // Channel 22 should be 0 (black's turn)
        let base = 22 * GRID_SIZE * GRID_SIZE;
        assert_eq!(board_tensor[base], 0.0);
    }
}
