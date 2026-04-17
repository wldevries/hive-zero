/// YINSH board tensor encoding.
///
/// Grid: 11x11 embedding of the 85-cell board. Cell (col, row) maps to
/// grid position `row * 11 + col`.
///
/// Board channels (9 spatial), all current-player-relative:
///   0: my rings
///   1: opponent rings
///   2: my markers
///   3: opponent markers
///   4: valid cell mask
///   5: phase flag — Setup (broadcast on valid cells)
///   6: phase flag — Normal
///   7: phase flag — RemoveRow (row of 5 markers pending selection)
///   8: phase flag — RemoveRing
///
/// Reserve vector (6 floats):
///   [0]: markers_in_pool / 51
///   [1]: my_score / 3           (win threshold = 3)
///   [2]: opp_score / 3
///   [3]: my_rings_on_board / 5
///   [4]: opp_rings_on_board / 5
///   [5]: rings_placed_total / 10

use core_game::game::Player;

use crate::board::{Cell, Phase, YinshBoard, INITIAL_MARKERS, RINGS_PER_PLAYER, WIN_SCORE};
use crate::hex::{ALL_CELLS, BOARD_SIZE, GRID_SIZE};

pub const NUM_CHANNELS: usize = 9;
pub const RESERVE_SIZE: usize = 6;

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

    // Per-cell channels and valid mask
    for i in 0..BOARD_SIZE {
        let g = cell_to_grid(i);
        board_out[4 * g2 + g] = 1.0; // valid mask
        match board.cells[i] {
            c if c == my_ring => board_out[0 * g2 + g] = 1.0,
            c if c == opp_ring => board_out[1 * g2 + g] = 1.0,
            c if c == my_marker => board_out[2 * g2 + g] = 1.0,
            c if c == opp_marker => board_out[3 * g2 + g] = 1.0,
            _ => {}
        }
    }

    // Phase flag — broadcast over valid cells only
    let phase_ch = match board.phase {
        Phase::Setup => 5,
        Phase::Normal => 6,
        Phase::RemoveRow => 7,
        Phase::RemoveRing => 8,
    };
    for i in 0..BOARD_SIZE {
        let g = cell_to_grid(i);
        board_out[phase_ch * g2 + g] = 1.0;
    }

    // Rings on board (count my_ring / opp_ring in cells)
    let mut my_rings_on_board = 0u8;
    let mut opp_rings_on_board = 0u8;
    for i in 0..BOARD_SIZE {
        if board.cells[i] == my_ring {
            my_rings_on_board += 1;
        } else if board.cells[i] == opp_ring {
            opp_rings_on_board += 1;
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

        // Valid mask: exactly BOARD_SIZE cells set
        let g2 = GRID_SIZE * GRID_SIZE;
        let valid_count: usize = buf[4 * g2..5 * g2].iter().filter(|&&v| v == 1.0).count();
        assert_eq!(valid_count, BOARD_SIZE);

        // Setup phase flag set on valid cells
        let setup_count: usize = buf[5 * g2..6 * g2].iter().filter(|&&v| v == 1.0).count();
        assert_eq!(setup_count, BOARD_SIZE);

        // Reserve: full marker pool
        assert!((reserve[0] - 1.0).abs() < 1e-6);
        assert_eq!(reserve[1], 0.0);
        assert_eq!(reserve[5], 0.0);
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
    }
}
