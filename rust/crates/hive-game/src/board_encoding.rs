/// Encode Hive board state as a fixed-size tensor for neural network input.
/// Must produce bitwise-identical output to Python board_encoder.py.

use crate::hex::Hex;
use crate::game::Game;
use crate::piece::{PieceColor, ALL_PIECE_TYPES, PIECE_COUNTS, PIECES_PER_PLAYER};

/// Number of board encoding channels.
pub const NUM_CHANNELS: usize = 39;
/// Reserve vector size: 5 piece types x 2 colors.
pub const RESERVE_SIZE: usize = 10;

/// Channel layout (all channels current-player-relative):
///
/// Base layer — piece at depth 0 of each hex (binary, one per piece):
///   0-10:  Current player's pieces  (Q, S1, S2, B1, B2, G1, G2, G3, A1, A2, A3)
///   11-21: Opponent's pieces
///
/// Stacked beetles — beetle at depth D above the base (binary):
///   22-25: Current player's Beetle1 at depths 1-4
///   26-29: Current player's Beetle2 at depths 1-4
///   30-33: Opponent's Beetle1 at depths 1-4
///   34-37: Opponent's Beetle2 at depths 1-4
///
///   Channel = 22 + player_offset(0 or 8) + (beetle_number-1)*4 + (depth-1)
///
/// 38: Stack height (normalized /7)
///
/// Reserve vector (current-player-relative):
///   0-4:  Current player's reserve counts (normalized by max)
///   5-9:  Opponent's reserve counts

const STACKED_BEETLE_BASE: usize = 22;
const STACK_HEIGHT_CH: usize = 38;

/// Map hex coordinates to encoding grid indices, using grid_size for the encoding.
#[inline]
fn hex_to_encoding_grid(h: Hex, grid_size: usize) -> Option<(usize, usize)> {
    let center = (grid_size / 2) as i16;
    let col = h.0 as i16 + center;
    let row = h.1 as i16 + center;
    if col >= 0 && col < grid_size as i16 && row >= 0 && row < grid_size as i16 {
        Some((row as usize, col as usize))
    } else {
        None
    }
}

#[inline]
fn piece_idx(piece: crate::piece::Piece) -> usize {
    piece.linear_index() % PIECES_PER_PLAYER
}

/// Convert f32 to bfloat16, stored as u16 (truncation, matches PyTorch convention).
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// Encode a game state into board tensor and reserve vector as bfloat16.
/// Board tensor shape: (NUM_CHANNELS, grid_size, grid_size)
/// Reserve vector shape: (RESERVE_SIZE,) = (10,)
pub fn encode_board_bf16(game: &Game, board_out: &mut [u16], reserve_out: &mut [u16], grid_size: usize) {
    debug_assert!(board_out.len() == NUM_CHANNELS * grid_size * grid_size);
    debug_assert!(reserve_out.len() == RESERVE_SIZE);

    let gs2 = grid_size * grid_size;
    let bf16_zero = f32_to_bf16(0.0);
    let bf16_one = f32_to_bf16(1.0);
    board_out.fill(bf16_zero);
    reserve_out.fill(bf16_zero);

    let is_white_turn = game.turn_color == PieceColor::White;
    let is_mine = |color: PieceColor| (color == PieceColor::White) == is_white_turn;

    for (pos, stack) in game.board.iter_occupied() {
        let (row, col) = match hex_to_encoding_grid(pos, grid_size) {
            Some(rc) => rc,
            None => continue,
        };
        let cell = row * grid_size + col;

        board_out[STACK_HEIGHT_CH * gs2 + cell] =
            f32_to_bf16(stack.height() as f32 / 7.0);

        for (depth, piece) in stack.iter().enumerate() {
            let mine = is_mine(piece.color());
            let idx = piece_idx(piece);

            if depth == 0 {
                let ch = if mine { idx } else { 11 + idx };
                board_out[ch * gs2 + cell] = bf16_one;
            } else {
                let player_offset = if mine { 0 } else { 8 };
                let beetle_offset = (piece.number() as usize - 1) * 4;
                let depth_offset = (depth - 1).min(3);
                let ch = STACKED_BEETLE_BASE + player_offset + beetle_offset + depth_offset;
                board_out[ch * gs2 + cell] = bf16_one;
            }
        }
    }

    let (cur_color, opp_color) = if is_white_turn {
        (PieceColor::White, PieceColor::Black)
    } else {
        (PieceColor::Black, PieceColor::White)
    };
    for (i, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
        let max_count = PIECE_COUNTS[i] as f32;
        if max_count > 0.0 {
            reserve_out[i] = f32_to_bf16(game.reserve_count(cur_color, pt) as f32 / max_count);
            reserve_out[5 + i] = f32_to_bf16(game.reserve_count(opp_color, pt) as f32 / max_count);
        }
    }
}

/// Encode a game state into board tensor and reserve vector.
/// Board tensor shape: (NUM_CHANNELS, grid_size, grid_size)
/// Reserve vector shape: (RESERVE_SIZE,) = (10,)
pub fn encode_board(game: &Game, board_out: &mut [f32], reserve_out: &mut [f32], grid_size: usize) {
    debug_assert!(board_out.len() == NUM_CHANNELS * grid_size * grid_size);
    debug_assert!(reserve_out.len() == RESERVE_SIZE);

    let gs2 = grid_size * grid_size;
    board_out.fill(0.0);
    reserve_out.fill(0.0);

    let is_white_turn = game.turn_color == PieceColor::White;
    let is_mine = |color: PieceColor| (color == PieceColor::White) == is_white_turn;

    for (pos, stack) in game.board.iter_occupied() {
        let (row, col) = match hex_to_encoding_grid(pos, grid_size) {
            Some(rc) => rc,
            None => continue,
        };
        let cell = row * grid_size + col;

        // Stack height
        board_out[STACK_HEIGHT_CH * gs2 + cell] =
            stack.height() as f32 / 7.0;

        // stack.iter() goes bottom-to-top; depth 0 = base piece
        for (depth, piece) in stack.iter().enumerate() {
            let mine = is_mine(piece.color());
            let idx = piece_idx(piece);

            if depth == 0 {
                // Base layer: any piece type, current-player-relative
                let ch = if mine { idx } else { 11 + idx };
                board_out[ch * gs2 + cell] = 1.0;
            } else {
                // Stacked piece — must be a beetle
                let player_offset = if mine { 0 } else { 8 };
                let beetle_offset = (piece.number() as usize - 1) * 4;
                let depth_offset = (depth - 1).min(3); // depths 1-4 → offsets 0-3
                let ch = STACKED_BEETLE_BASE + player_offset + beetle_offset + depth_offset;
                board_out[ch * gs2 + cell] = 1.0;
            }
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
    use crate::board::GRID_SIZE;
    use crate::piece::{Piece, PieceType};
    use crate::game::{Game, Move};

    fn encode(game: &Game) -> (Vec<f32>, Vec<f32>) {
        let gs = game.nn_grid_size;
        let mut bt = vec![0.0f32; NUM_CHANNELS * gs * gs];
        let mut rv = vec![0.0f32; RESERVE_SIZE];
        encode_board(game, &mut bt, &mut rv, gs);
        (bt, rv)
    }

    fn at(bt: &[f32], ch: usize, row: usize, col: usize) -> f32 {
        let gs = GRID_SIZE; // tests use default grid size
        bt[ch * gs * gs + row * gs + col]
    }

    #[test]
    fn test_empty_board() {
        let game = Game::new();
        let (bt, rv) = encode(&game);
        assert!(bt.iter().all(|&v| v == 0.0));
        // All pieces in reserve for both players
        for i in 0..RESERVE_SIZE {
            assert_eq!(rv[i], 1.0);
        }
    }

    #[test]
    fn test_base_layer_current_player_relative() {
        let mut game = Game::new();
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        game.play_move(&Move::placement(wq, (0, 0)));
        // Black's turn — white queen is opponent's piece (channel 11 = opponent Q)

        let (bt, rv) = encode(&game);
        // White queen at center (11,11) — opponent's piece, index 0 → channel 11
        assert_eq!(at(&bt, 11, 11, 11), 1.0);
        // Not in current player's channels
        assert_eq!(at(&bt, 0, 11, 11), 0.0);
        // Opponent (white) queen used → reserve slot 5 (opponent Q) = 0
        assert_eq!(rv[5], 0.0);
        // Current player (black) queen still in reserve
        assert_eq!(rv[0], 1.0);
    }

    #[test]
    fn test_stacked_beetle() {
        let mut game = Game::new();
        let wq  = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let wb1 = Piece::new(PieceColor::White, PieceType::Beetle, 1);
        let bq  = Piece::new(PieceColor::Black, PieceType::Queen, 1);
        let bb1 = Piece::new(PieceColor::Black, PieceType::Beetle, 1);

        // Place pieces to set up a stack: wQ at (0,0), bQ adjacent, then move beetles on top
        game.play_move(&Move::placement(wq, (0, 0)));
        game.play_move(&Move::placement(bq, (1, 0)));
        game.play_move(&Move::placement(wb1, (-1, 0)));
        game.play_move(&Move::placement(bb1, (2, 0)));
        // Move wb1 onto wQ — stack at (0,0): [wQ, wB1], white's turn
        game.play_move(&Move::movement(wb1, (-1, 0), (0, 0)));
        // Now black's turn. Stack at (0,0) from black's perspective:
        // base = wQ (opponent's Q, channel 11), stacked = wB1 (opponent's B1 at depth 1)
        // Opponent's B1 at depth 1: channel = 22 + 8 + 0*4 + 0 = 30

        let (bt, _) = encode(&game);
        // base at (0,0) = opponent's queen → channel 11
        assert_eq!(at(&bt, 11, 11, 11), 1.0);
        // opponent's B1 at depth 1 → channel 30
        assert_eq!(at(&bt, 30, 11, 11), 1.0);
        // stack height = 2/7
        let h = at(&bt, 38, 11, 11);
        assert!((h - 2.0 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_symmetric_encoding() {
        // After placing wQ, encoding from black's turn should show it in opponent's channels.
        // After placing bQ (mirrored), encoding from white's turn should show it the same way.
        let mut game_a = Game::new();
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let bq = Piece::new(PieceColor::Black, PieceType::Queen, 1);

        game_a.play_move(&Move::placement(wq, (0, 0)));
        // Black's turn: wQ is opponent at center
        let (bt_a, _) = encode(&game_a);

        let mut game_b = Game::new();
        // Skip to black's first move by placing wQ somewhere else first
        let wa1 = Piece::new(PieceColor::White, PieceType::Ant, 1);
        game_b.play_move(&Move::placement(wa1, (0, 0)));
        game_b.play_move(&Move::placement(bq, (1, 0)));
        // White's turn: bQ is opponent at (1,0)
        let (bt_b, _) = encode(&game_b);

        // Both cases: opponent's queen in channel 11
        assert_eq!(at(&bt_a, 11, 11, 11), 1.0); // wQ at center, black's turn
        assert_eq!(at(&bt_b, 11, 11, 12), 1.0); // bQ at (1,0)=(row11,col12), white's turn
        // Neither appears in current player's channels (0-10)
        assert_eq!(at(&bt_a, 0, 11, 11), 0.0);
        assert_eq!(at(&bt_b, 0, 11, 12), 0.0);
    }
}
