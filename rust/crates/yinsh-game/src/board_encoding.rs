/// YINSH board tensor encoding.
///
/// Grid: 11x11 embedding of the 85-cell board. Cell (col, row) maps to
/// grid position `row * 11 + col`.
///
/// Board channels (4 spatial), all current-player-relative:
///   0: my rings
///   1: opponent rings
///   2: my markers
///   3: opponent markers
///
/// The valid-cell mask is a constant 11x11 plane and is baked into the model
/// as a non-trainable buffer rather than transferred per-sample. The phase
/// one-hot moved into the reserve vector for the same reason — phase is a
/// global scalar, the input conv recovers a spatial plane via reserve broadcast.
///
/// Reserve vector (9 floats):
///   [0]: markers_in_pool / 51
///   [1]: my_score / 3           (win threshold = 3)
///   [2]: opp_score / 3
///   [3]: my_rings_on_board / 5
///   [4]: opp_rings_on_board / 5
///   [5]: rings_placed_total / 10
///   [6]: phase == Setup
///   [7]: phase == Normal
///   [8]: phase == ClaimRow

use core_game::game::Player;

use crate::board::{Cell, Phase, YinshBoard, INITIAL_MARKERS, RINGS_PER_PLAYER, WIN_SCORE};
use crate::hex::{ALL_CELLS, BOARD_SIZE, GRID_SIZE};

pub const NUM_CHANNELS: usize = 4;
pub const RESERVE_SIZE: usize = 9;

#[inline]
pub fn cell_to_grid(cell_idx: usize) -> usize {
    let (col, row) = ALL_CELLS[cell_idx];
    row as usize * GRID_SIZE + col as usize
}

pub fn encode_board(board: &YinshBoard, board_out: &mut [f32], reserve_out: &mut [f32]) {
    debug_assert_eq!(board_out.len(), NUM_CHANNELS * GRID_SIZE * GRID_SIZE);
    debug_assert_eq!(reserve_out.len(), RESERVE_SIZE);

    board_out.fill(0.0);

    let me = board.next_player;
    let g2 = GRID_SIZE * GRID_SIZE;

    let (my_ring, opp_ring, my_marker, opp_marker) = match me {
        Player::Player1 => (Cell::WhiteRing, Cell::BlackRing, Cell::WhiteMarker, Cell::BlackMarker),
        Player::Player2 => (Cell::BlackRing, Cell::WhiteRing, Cell::BlackMarker, Cell::WhiteMarker),
    };

    let mut my_rings_on_board = 0u8;
    let mut opp_rings_on_board = 0u8;
    for i in 0..BOARD_SIZE {
        let g = cell_to_grid(i);
        match board.cells[i] {
            c if c == my_ring => {
                board_out[0 * g2 + g] = 1.0;
                my_rings_on_board += 1;
            }
            c if c == opp_ring => {
                board_out[1 * g2 + g] = 1.0;
                opp_rings_on_board += 1;
            }
            c if c == my_marker => board_out[2 * g2 + g] = 1.0,
            c if c == opp_marker => board_out[3 * g2 + g] = 1.0,
            _ => {}
        }
    }

    let (my_score, opp_score) = match me {
        Player::Player1 => (board.white_score, board.black_score),
        Player::Player2 => (board.black_score, board.white_score),
    };
    let rings_placed = board.white_rings_placed + board.black_rings_placed;

    reserve_out[0] = board.markers_in_pool as f32 / INITIAL_MARKERS as f32;
    reserve_out[1] = (my_score as f32 / WIN_SCORE as f32).min(1.0);
    reserve_out[2] = (opp_score as f32 / WIN_SCORE as f32).min(1.0);
    reserve_out[3] = my_rings_on_board as f32 / RINGS_PER_PLAYER as f32;
    reserve_out[4] = opp_rings_on_board as f32 / RINGS_PER_PLAYER as f32;
    reserve_out[5] = rings_placed as f32 / (2.0 * RINGS_PER_PLAYER as f32);
    reserve_out[6] = 0.0;
    reserve_out[7] = 0.0;
    reserve_out[8] = 0.0;
    let phase_idx = match board.phase {
        Phase::Setup => 6,
        Phase::Normal => 7,
        Phase::ClaimRow => 8,
    };
    reserve_out[phase_idx] = 1.0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_game::game::Game;

    #[test]
    fn test_encode_initial_state() {
        let board = YinshBoard::new();
        let mut buf = vec![0.0f32; NUM_CHANNELS * GRID_SIZE * GRID_SIZE];
        let mut reserve = vec![0.0f32; RESERVE_SIZE];
        encode_board(&board, &mut buf, &mut reserve);

        // Initial board: no rings, no markers — board tensor is all zeros.
        assert_eq!(buf.iter().filter(|&&v| v != 0.0).count(), 0);

        // Reserve: full marker pool, scores 0, no rings on board, Setup phase.
        assert!((reserve[0] - 1.0).abs() < 1e-6);
        assert_eq!(reserve[1], 0.0);
        assert_eq!(reserve[5], 0.0);
        assert_eq!(reserve[6], 1.0); // Setup
        assert_eq!(reserve[7], 0.0); // Normal
        assert_eq!(reserve[8], 0.0); // ClaimRow
    }

    #[test]
    fn test_encode_after_ring_placement() {
        let mut board = YinshBoard::new();
        let idx = crate::hex::cell_index(4, 4);
        board.play_move(&crate::board::YinshMove::PlaceRing(idx)).unwrap();

        let mut buf = vec![0.0f32; NUM_CHANNELS * GRID_SIZE * GRID_SIZE];
        let mut reserve = vec![0.0f32; RESERVE_SIZE];
        encode_board(&board, &mut buf, &mut reserve);

        let g2 = GRID_SIZE * GRID_SIZE;
        let g = cell_to_grid(idx);
        // After white places, next_player is black, so the white ring is the opponent's ring (channel 1)
        assert_eq!(buf[1 * g2 + g], 1.0);
        assert_eq!(buf[0 * g2 + g], 0.0);
        // Still in Setup until 10 rings are placed.
        assert_eq!(reserve[6], 1.0);
    }
}
