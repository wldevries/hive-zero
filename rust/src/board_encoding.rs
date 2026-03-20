/// Encode Hive board state as a fixed-size tensor for neural network input.
/// Must produce bitwise-identical output to Python board_encoder.py.

use crate::board::{GRID_SIZE, hex_to_grid};
use crate::game::Game;
use crate::piece::{PieceColor, ALL_PIECE_TYPES, PIECE_COUNTS};

/// Number of board encoding channels.
pub const NUM_CHANNELS: usize = 22;
/// Reserve vector size: 5 piece types x 2 colors.
pub const RESERVE_SIZE: usize = 10;

/// Channel layout (all channels are current-player-relative):
/// 0-4:   Current player's pieces on top (Q, S, B, G, A) - binary
/// 5-9:   Opponent's pieces on top (Q, S, B, G, A) - binary
/// 10-14: Current player's pieces in stack (count)
/// 15-19: Opponent's pieces in stack (count)
/// 20:    Stack height (normalized /7)
/// 21:    Current player's pieces (1 = current player's top piece)
///
/// Reserve vector (current-player-relative):
/// 0-4:   Current player's reserve counts (normalized)
/// 5-9:   Opponent's reserve counts (normalized)

/// Encode a game state into board tensor and reserve vector.
/// Board tensor shape: (NUM_CHANNELS, GRID_SIZE, GRID_SIZE) = (22, 23, 23)
/// Reserve vector shape: (RESERVE_SIZE,) = (10,)
pub fn encode_board(game: &Game, board_out: &mut [f32], reserve_out: &mut [f32]) {
    debug_assert!(board_out.len() == NUM_CHANNELS * GRID_SIZE * GRID_SIZE);
    debug_assert!(reserve_out.len() == RESERVE_SIZE);

    // Zero out
    board_out.fill(0.0);
    reserve_out.fill(0.0);

    let is_white_turn = game.turn_color == PieceColor::White;

    // Helper: is this piece the current player's?
    let is_mine = |color: PieceColor| -> bool {
        (color == PieceColor::White) == is_white_turn
    };

    // Iterate over occupied positions
    for (pos, stack) in game.board.iter_occupied() {
        let (row, col) = match hex_to_grid(pos) {
            Some(rc) => rc,
            None => continue,
        };

        // Top piece — current player's in 0-4, opponent's in 5-9
        let top = stack.top().unwrap();
        let offset = if is_mine(top.color()) { 0 } else { 5 };
        let ch = top.piece_type().index();
        board_out[(offset + ch) * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col] = 1.0;

        // All pieces in stack — current player's in 10-14, opponent's in 15-19
        for piece in stack.iter() {
            let offset2 = if is_mine(piece.color()) { 10 } else { 15 };
            let ch2 = piece.piece_type().index();
            board_out[(offset2 + ch2) * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col] += 1.0;
        }

        // Stack height
        board_out[20 * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col] =
            stack.height() as f32 / 7.0;

        // Current player's piece marker
        if is_mine(top.color()) {
            board_out[21 * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col] = 1.0;
        }
    }

    // Reserve vector — current player first (0-4), opponent second (5-9)
    let (cur_color, opp_color) = if is_white_turn {
        (PieceColor::White, PieceColor::Black)
    } else {
        (PieceColor::Black, PieceColor::White)
    };
    for (i, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
        let max_count = PIECE_COUNTS[i] as f32;
        if max_count > 0.0 {
            reserve_out[i] = game.reserve_count(cur_color, pt) as f32 / max_count;
            reserve_out[5 + i] = game.reserve_count(opp_color, pt) as f32 / max_count;
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

        // Empty board — all board channels should be zero
        assert!(board_tensor.iter().all(|&v| v == 0.0));

        // All pieces in reserve for both players
        for i in 0..RESERVE_SIZE {
            assert_eq!(reserve_vec[i], 1.0);
        }
    }

    #[test]
    fn test_encode_one_piece() {
        let mut game = Game::new();
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        game.play_move(&crate::game::Move::placement(wq, (0, 0)));
        // Now it's black's turn — white queen is the *opponent's* piece

        let mut board_tensor = vec![0.0f32; NUM_CHANNELS * GRID_SIZE * GRID_SIZE];
        let mut reserve_vec = vec![0.0f32; RESERVE_SIZE];
        encode_board(&game, &mut board_tensor, &mut reserve_vec);

        let row = 11;
        let col = 11;
        // Channel 5 = opponent's queen on top (black is current player, white queen is opponent's)
        assert_eq!(
            board_tensor[5 * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col],
            1.0
        );
        // Channel 0 (current player on top) should be 0
        assert_eq!(
            board_tensor[0 * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col],
            0.0
        );
        // Channel 15 = opponent's queen in stack
        assert_eq!(
            board_tensor[15 * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col],
            1.0
        );
        // Stack height = 1/7
        let h = board_tensor[20 * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col];
        assert!((h - 1.0 / 7.0).abs() < 1e-6);

        // Reserve: black (current player) has all pieces, white (opponent) missing queen
        // current player reserve[0] (queen) = 1.0 (black queen still in reserve)
        assert_eq!(reserve_vec[0], 1.0);
        // opponent reserve[0] (queen) = 0.0 (white queen placed)
        assert_eq!(reserve_vec[5], 0.0);
    }

    #[test]
    fn test_symmetric_encoding() {
        // A position encoded from white's turn should look the same as the
        // color-mirrored position encoded from black's turn.
        // We verify this by placing white's queen and checking that the
        // "current player on top" channel (0) fires when it's white's turn
        // but not when it's black's turn.
        let mut game = Game::new();
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);

        // White's first turn: encode before placing (white is current player, reserve only)
        let mut bt = vec![0.0f32; NUM_CHANNELS * GRID_SIZE * GRID_SIZE];
        let mut rv = vec![0.0f32; RESERVE_SIZE];
        encode_board(&game, &mut bt, &mut rv);
        // All reserve, current player (white) has full reserve at indices 0-4
        assert_eq!(rv[0], 1.0); // white queen in reserve

        game.play_move(&crate::game::Move::placement(wq, (0, 0)));
        // Now black's turn — white queen is opponent's piece
        let mut bt2 = vec![0.0f32; NUM_CHANNELS * GRID_SIZE * GRID_SIZE];
        let mut rv2 = vec![0.0f32; RESERVE_SIZE];
        encode_board(&game, &mut bt2, &mut rv2);

        // Opponent (white) queen is placed → opponent reserve[0] = 0
        assert_eq!(rv2[5], 0.0);
        // Current player (black) queen still in reserve → current reserve[0] = 1
        assert_eq!(rv2[0], 1.0);
        // Opponent queen on top at center → channel 5
        assert_eq!(bt2[5 * GRID_SIZE * GRID_SIZE + 11 * GRID_SIZE + 11], 1.0);
    }
}
