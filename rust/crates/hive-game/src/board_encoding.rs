/// Encode Hive board state as a fixed-size tensor for neural network input.
/// Must produce bitwise-identical output to Python board_encoder.py.

use core_game::hex::{Hex, hex_distance, hex_neighbors};
use crate::game::Game;
use crate::piece::{Piece, PieceColor, PieceType, PIECE_COUNTS};

/// Number of board encoding channels.
pub const NUM_CHANNELS: usize = 24;
/// Reserve vector size: 5 piece types x 2 colors.
pub const RESERVE_SIZE: usize = 10;

const BASE_PIECE_TYPES: [PieceType; 5] = [
    PieceType::Queen,
    PieceType::Spider,
    PieceType::Beetle,
    PieceType::Grasshopper,
    PieceType::Ant,
];

/// Channel layout (all channels current-player-relative):
///
/// Base layer — piece type at depth 0 of each hex (binary, one per type):
///   0-4:  Current player's pieces  (Q=0, S=1, B=2, G=3, A=4)
///   5-9:  Opponent's pieces        (same type order)
///
/// Stacked pieces — generic "stacker" channels by depth (binary):
///   10-13: Current player's stacker at depths 1-4
///   14-17: Opponent's stacker at depths 1-4
///
/// 18: Hive edge (binary: 1 for empty cells adjacent to at least one occupied cell)
///
/// Queen geometry channels:
///   19: Hex distance to my queen (normalized by grid_size; 1.0 if queen not placed)
///   20: Hex distance to opponent's queen (normalized; 1.0 if not placed)
///   21: Adjacent to my queen (binary: 1 if cell is a neighbor of my queen)
///   22: Adjacent to opponent's queen (binary)
///
/// One Hive constraint:
///   23: Pinned piece (binary: 1 if occupied and removing would split the hive)
///
/// Reserve vector (current-player-relative):
///   0-4:  Current player's reserve counts (normalized by max)
///   5-9:  Opponent's reserve counts

const MY_PIECES_BASE: usize = 0;
const OPP_PIECES_BASE: usize = 5;
const MY_STACKER_BASE: usize = 10;
const OPP_STACKER_BASE: usize = 14;
const HIVE_EDGE_CH: usize = 18;
const MY_QUEEN_DIST_CH: usize = 19;
const OPP_QUEEN_DIST_CH: usize = 20;
const MY_QUEEN_ADJ_CH: usize = 21;
const OPP_QUEEN_ADJ_CH: usize = 22;
const PINNED_CH: usize = 23;

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
fn base_piece_type_idx(piece_type: PieceType) -> Option<usize> {
    match piece_type {
        PieceType::Queen => Some(0),
        PieceType::Spider => Some(1),
        PieceType::Beetle => Some(2),
        PieceType::Grasshopper => Some(3),
        PieceType::Ant => Some(4),
        PieceType::Mosquito | PieceType::Ladybug | PieceType::Pillbug => None,
    }
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

        for (depth, piece) in stack.iter().enumerate() {
            let mine = is_mine(piece.color());

            if depth == 0 {
                if let Some(type_idx) = base_piece_type_idx(piece.piece_type()) {
                    let ch = if mine { MY_PIECES_BASE + type_idx } else { OPP_PIECES_BASE + type_idx };
                    board_out[ch * gs2 + cell] = bf16_one;
                }
            } else {
                let stacker_base = if mine { MY_STACKER_BASE } else { OPP_STACKER_BASE };
                let depth_offset = (depth - 1).min(3);
                let ch = stacker_base + depth_offset;
                board_out[ch * gs2 + cell] = bf16_one;
            }
        }

        // Hive edge: mark empty neighbors of occupied cells
        for nb in hex_neighbors(pos) {
            if !game.board.is_occupied(nb) {
                if let Some((nr, nc)) = hex_to_encoding_grid(nb, grid_size) {
                    board_out[HIVE_EDGE_CH * gs2 + nr * grid_size + nc] = bf16_one;
                }
            }
        }
    }

    let (cur_color, opp_color) = if is_white_turn {
        (PieceColor::White, PieceColor::Black)
    } else {
        (PieceColor::Black, PieceColor::White)
    };
    for (i, &pt) in BASE_PIECE_TYPES.iter().enumerate() {
        let max_count = PIECE_COUNTS[i] as f32;
        if max_count > 0.0 {
            reserve_out[i] = f32_to_bf16(game.reserve_count(cur_color, pt) as f32 / max_count);
            reserve_out[5 + i] = f32_to_bf16(game.reserve_count(opp_color, pt) as f32 / max_count);
        }
    }

    // Queen geometry channels (19-22) and pinned channel (23).
    let my_queen = Piece::new(cur_color, PieceType::Queen, 1);
    let opp_queen = Piece::new(opp_color, PieceType::Queen, 1);
    let my_queen_pos = game.board.piece_position(my_queen);
    let opp_queen_pos = game.board.piece_position(opp_queen);
    let norm = grid_size as f32;
    let center = (grid_size / 2) as i16;

    for row in 0..grid_size {
        for col in 0..grid_size {
            let cell = row * grid_size + col;
            let h: Hex = ((col as i16 - center) as i8, (row as i16 - center) as i8);

            match my_queen_pos {
                Some(qpos) => {
                    let d = hex_distance(h, qpos);
                    board_out[MY_QUEEN_DIST_CH * gs2 + cell] = f32_to_bf16(d as f32 / norm);
                    if d == 1 {
                        board_out[MY_QUEEN_ADJ_CH * gs2 + cell] = bf16_one;
                    }
                }
                None => {
                    board_out[MY_QUEEN_DIST_CH * gs2 + cell] = bf16_one;
                }
            }

            match opp_queen_pos {
                Some(qpos) => {
                    let d = hex_distance(h, qpos);
                    board_out[OPP_QUEEN_DIST_CH * gs2 + cell] = f32_to_bf16(d as f32 / norm);
                    if d == 1 {
                        board_out[OPP_QUEEN_ADJ_CH * gs2 + cell] = bf16_one;
                    }
                }
                None => {
                    board_out[OPP_QUEEN_DIST_CH * gs2 + cell] = bf16_one;
                }
            }
        }
    }

    // Pinned pieces: articulation points of the hive graph.
    for ap in game.board.articulation_points() {
        if let Some((row, col)) = hex_to_encoding_grid(ap, grid_size) {
            board_out[PINNED_CH * gs2 + row * grid_size + col] = bf16_one;
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

        // stack.iter() goes bottom-to-top; depth 0 = base piece
        for (depth, piece) in stack.iter().enumerate() {
            let mine = is_mine(piece.color());

            if depth == 0 {
                // Base layer: piece type, current-player-relative
                if let Some(type_idx) = base_piece_type_idx(piece.piece_type()) {
                    let ch = if mine { MY_PIECES_BASE + type_idx } else { OPP_PIECES_BASE + type_idx };
                    board_out[ch * gs2 + cell] = 1.0;
                }
            } else {
                // Stacked piece — generic stacker channel by depth
                let stacker_base = if mine { MY_STACKER_BASE } else { OPP_STACKER_BASE };
                let depth_offset = (depth - 1).min(3);
                let ch = stacker_base + depth_offset;
                board_out[ch * gs2 + cell] = 1.0;
            }
        }

        // Hive edge: mark empty neighbors of occupied cells
        for nb in hex_neighbors(pos) {
            if !game.board.is_occupied(nb) {
                if let Some((nr, nc)) = hex_to_encoding_grid(nb, grid_size) {
                    board_out[HIVE_EDGE_CH * gs2 + nr * grid_size + nc] = 1.0;
                }
            }
        }
    }

    // Reserve vector — current player first (0-4), opponent second (5-9)
    let (cur_color, opp_color) = if is_white_turn {
        (PieceColor::White, PieceColor::Black)
    } else {
        (PieceColor::Black, PieceColor::White)
    };
    for (i, &pt) in BASE_PIECE_TYPES.iter().enumerate() {
        let max_count = PIECE_COUNTS[i] as f32;
        if max_count > 0.0 {
            reserve_out[i] = game.reserve_count(cur_color, pt) as f32 / max_count;
            reserve_out[5 + i] = game.reserve_count(opp_color, pt) as f32 / max_count;
        }
    }

    // Queen geometry channels (19-22) and pinned channel (23).
    let my_queen = Piece::new(cur_color, PieceType::Queen, 1);
    let opp_queen = Piece::new(opp_color, PieceType::Queen, 1);
    let my_queen_pos = game.board.piece_position(my_queen);
    let opp_queen_pos = game.board.piece_position(opp_queen);
    let norm = grid_size as f32;
    let center = (grid_size / 2) as i16;

    for row in 0..grid_size {
        for col in 0..grid_size {
            let cell = row * grid_size + col;
            let h: Hex = ((col as i16 - center) as i8, (row as i16 - center) as i8);

            match my_queen_pos {
                Some(qpos) => {
                    let d = hex_distance(h, qpos);
                    board_out[MY_QUEEN_DIST_CH * gs2 + cell] = d as f32 / norm;
                    if d == 1 {
                        board_out[MY_QUEEN_ADJ_CH * gs2 + cell] = 1.0;
                    }
                }
                None => {
                    board_out[MY_QUEEN_DIST_CH * gs2 + cell] = 1.0;
                }
            }

            match opp_queen_pos {
                Some(qpos) => {
                    let d = hex_distance(h, qpos);
                    board_out[OPP_QUEEN_DIST_CH * gs2 + cell] = d as f32 / norm;
                    if d == 1 {
                        board_out[OPP_QUEEN_ADJ_CH * gs2 + cell] = 1.0;
                    }
                }
                None => {
                    board_out[OPP_QUEEN_DIST_CH * gs2 + cell] = 1.0;
                }
            }
        }
    }

    // Pinned pieces: articulation points of the hive graph.
    for ap in game.board.articulation_points() {
        if let Some((row, col)) = hex_to_encoding_grid(ap, grid_size) {
            board_out[PINNED_CH * gs2 + row * grid_size + col] = 1.0;
        }
    }
}

#[cfg(test)]
#[allow(unused_must_use)]
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
        let gs = game.nn_grid_size;
        let (bt, rv) = encode(&game);
        // Channels 0-18 and 21-23 are zero on an empty board.
        // Channels 19-20 (queen distances) are 1.0 when queens are not placed.
        let gs2 = gs * gs;
        for ch in 0..NUM_CHANNELS {
            for cell in 0..gs2 {
                let v = bt[ch * gs2 + cell];
                if ch == MY_QUEEN_DIST_CH || ch == OPP_QUEEN_DIST_CH {
                    assert_eq!(v, 1.0, "ch={ch} cell={cell}");
                } else {
                    assert_eq!(v, 0.0, "ch={ch} cell={cell}");
                }
            }
        }
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
        // Black's turn — white queen is opponent's piece, type index 0 → channel OPP_PIECES_BASE+0 = 5

        let (bt, rv) = encode(&game);
        // White queen at center (11,11) — opponent's piece → channel 5
        assert_eq!(at(&bt, 5, 11, 11), 1.0);
        // Not in current player's channels
        assert_eq!(at(&bt, 0, 11, 11), 0.0);
        // Opponent (white) queen used → reserve slot 5 (opponent Q) = 0
        assert_eq!(rv[5], 0.0);
        // Current player (black) queen still in reserve
        assert_eq!(rv[0], 1.0);
    }

    #[test]
    fn test_stacked_beetle() {
        use crate::game::GameState;
        // Set up board directly to avoid recentering from play_move.
        // Position: wQ at (0,0) with wB1 stacked on top; bQ at (1,0); bB1 at (2,0). Black's turn.
        let mut game = Game::new();
        let wq  = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let wb1 = Piece::new(PieceColor::White, PieceType::Beetle, 1);
        let bq  = Piece::new(PieceColor::Black, PieceType::Queen, 1);
        let bb1 = Piece::new(PieceColor::Black, PieceType::Beetle, 1);

        game.board.place_piece(wq, (0, 0)).unwrap();
        game.board.place_piece(wb1, (0, 0)).unwrap(); // beetle stacks on wQ
        game.board.place_piece(bq, (1, 0)).unwrap();
        game.board.place_piece(bb1, (2, 0)).unwrap();
        game.state = GameState::InProgress;
        game.turn_color = PieceColor::Black;

        // From black's perspective: stack at (0,0) = opponent's pieces.
        // base = wQ (opponent's Q, type 0 → channel OPP_PIECES_BASE+0 = 5)
        // stacked = wB1 (opponent's stacker at depth 1 → channel OPP_STACKER_BASE+0 = 14)
        let (bt, _) = encode(&game);
        assert_eq!(at(&bt, 5, 11, 11), 1.0);
        assert_eq!(at(&bt, 14, 11, 11), 1.0);
        // Hive edge: occupied cell (0,0) is NOT on the edge channel (it's occupied)
        assert_eq!(at(&bt, HIVE_EDGE_CH, 11, 11), 0.0);
    }

    #[test]
    fn test_expansion_piece_does_not_alias_base_channels_or_reserve() {
        use crate::game::GameState;

        let mut game = Game::new();
        let wm = Piece::new(PieceColor::White, PieceType::Mosquito, 1);
        game.board.place_piece(wm, (0, 0)).unwrap();
        game.state = GameState::InProgress;
        game.turn_color = PieceColor::Black;

        let (bt, rv) = encode(&game);

        // Expansion piece at base depth should not be encoded into Q/S/B/G/A channels.
        for ch in 0..10 {
            assert_eq!(at(&bt, ch, 11, 11), 0.0, "ch={ch}");
        }

        // Reserve vector is fixed-size base-game layout and must stay in-bounds.
        assert_eq!(rv.len(), RESERVE_SIZE);
        // Opponent queen reserve slot remains unchanged (was not consumed by a mosquito).
        assert_eq!(rv[5], 1.0);
    }

    #[test]
    fn test_symmetric_encoding() {
        use crate::game::GameState;
        // After placing wQ, encoding from black's turn should show it in opponent's channels.
        // After placing bQ (mirrored), encoding from white's turn should show it the same way.
        // Use direct board setup to avoid recentering from play_move.

        // game_a: wQ at (0,0), black's turn.
        let mut game_a = Game::new();
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        game_a.board.place_piece(wq, (0, 0)).unwrap();
        game_a.state = GameState::InProgress;
        game_a.turn_color = PieceColor::Black;
        let (bt_a, _) = encode(&game_a);

        // game_b: wa1 at (0,0) and bQ at (1,0), white's turn.
        let mut game_b = Game::new();
        let bq  = Piece::new(PieceColor::Black, PieceType::Queen, 1);
        let wa1 = Piece::new(PieceColor::White, PieceType::Ant, 1);
        game_b.board.place_piece(wa1, (0, 0)).unwrap();
        game_b.board.place_piece(bq, (1, 0)).unwrap();
        game_b.state = GameState::InProgress;
        game_b.turn_color = PieceColor::White;
        let (bt_b, _) = encode(&game_b);

        // Both cases: opponent's queen in channel 5 (OPP_PIECES_BASE + Queen=0)
        assert_eq!(at(&bt_a, 5, 11, 11), 1.0); // wQ at (0,0)=grid(11,11), black's turn
        assert_eq!(at(&bt_b, 5, 11, 12), 1.0); // bQ at (1,0)=grid(11,12), white's turn
        // Neither appears in current player's channels (0-4)
        assert_eq!(at(&bt_a, 0, 11, 11), 0.0);
        assert_eq!(at(&bt_b, 0, 11, 12), 0.0);
    }
}
