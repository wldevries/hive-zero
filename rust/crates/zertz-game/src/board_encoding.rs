/// Board encoding: converts ZertzBoard into a tensor for neural network input.
///
/// Grid: 7x7 (hex board rows 4-5-6-7-6-5-4 embedded left-aligned).
///
/// Channels (13 total):
///   0: White marbles
///   1: Grey marbles
///   2: Black marbles
///   3: Empty rings (valid, unoccupied)
///   4-6: Supply (W, G, B) normalized by initial counts (6, 8, 10)
///   7-9: Current player captures (W, G, B) normalized by initial supply per color
///   10-12: Opponent captures (W, G, B) normalized by initial supply per color

use core_game::game::{Game, Player};

use crate::hex::{all_hexes, hex_to_grid, hex_to_index};
use crate::zertz::{Marble, Ring, ZertzBoard};

const BOARD_SIZE: usize = crate::hex::BOARD_SIZE;

pub const GRID_SIZE: usize = 7;
pub const NUM_CHANNELS: usize = 13;

/// Initial supply for normalization: [6, 8, 10].
const INITIAL_SUPPLY: [f32; 3] = [6.0, 8.0, 10.0];

/// Encode a ZertzBoard into flat tensor buffers.
///
/// `board_out` must have length NUM_CHANNELS * GRID_SIZE * GRID_SIZE (= 686).
pub fn encode_board(board: &ZertzBoard, board_out: &mut [f32]) {
    debug_assert_eq!(board_out.len(), NUM_CHANNELS * GRID_SIZE * GRID_SIZE);

    // Zero out
    board_out.fill(0.0);

    let rings = board.rings();
    let supply = board.supply();
    let captures = board.captures();
    let next = board.next_player();

    let (cur_pi, opp_pi) = match next {
        Player::Player1 => (0usize, 1usize),
        Player::Player2 => (1usize, 0usize),
    };

    // Per-cell channels: iterate over all hexes on the board
    for h in all_hexes() {
        let (row, col) = hex_to_grid(h);
        let base = row * GRID_SIZE + col;

        let ring = rings[hex_to_index(h)];
        match ring {
            Ring::Occupied(Marble::White) => board_out[0 * GRID_SIZE * GRID_SIZE + base] = 1.0,
            Ring::Occupied(Marble::Grey)  => board_out[1 * GRID_SIZE * GRID_SIZE + base] = 1.0,
            Ring::Occupied(Marble::Black) => board_out[2 * GRID_SIZE * GRID_SIZE + base] = 1.0,
            Ring::Empty   => board_out[3 * GRID_SIZE * GRID_SIZE + base] = 1.0,
            Ring::Removed => {}
        }
    }

    // Broadcast channels (constant across all valid cells): supply and captures
    let supply_norm: [f32; 3] = [
        supply[0] as f32 / INITIAL_SUPPLY[0],
        supply[1] as f32 / INITIAL_SUPPLY[1],
        supply[2] as f32 / INITIAL_SUPPLY[2],
    ];
    // Normalize captures by initial supply per color so each is in [0, 1]
    let cur_caps: [f32; 3] = [
        captures[cur_pi][0] as f32 / INITIAL_SUPPLY[0],
        captures[cur_pi][1] as f32 / INITIAL_SUPPLY[1],
        captures[cur_pi][2] as f32 / INITIAL_SUPPLY[2],
    ];
    let opp_caps: [f32; 3] = [
        captures[opp_pi][0] as f32 / INITIAL_SUPPLY[0],
        captures[opp_pi][1] as f32 / INITIAL_SUPPLY[1],
        captures[opp_pi][2] as f32 / INITIAL_SUPPLY[2],
    ];

    for h in all_hexes() {
        let (row, col) = hex_to_grid(h);
        let base = row * GRID_SIZE + col;

        if rings[hex_to_index(h)] != Ring::Removed {
            board_out[4 * GRID_SIZE * GRID_SIZE + base] = supply_norm[0];
            board_out[5 * GRID_SIZE * GRID_SIZE + base] = supply_norm[1];
            board_out[6 * GRID_SIZE * GRID_SIZE + base] = supply_norm[2];
            board_out[7 * GRID_SIZE * GRID_SIZE + base] = cur_caps[0];
            board_out[8 * GRID_SIZE * GRID_SIZE + base] = cur_caps[1];
            board_out[9 * GRID_SIZE * GRID_SIZE + base] = cur_caps[2];
            board_out[10 * GRID_SIZE * GRID_SIZE + base] = opp_caps[0];
            board_out[11 * GRID_SIZE * GRID_SIZE + base] = opp_caps[1];
            board_out[12 * GRID_SIZE * GRID_SIZE + base] = opp_caps[2];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_to_grid_bounds() {
        for h in all_hexes() {
            let (row, col) = hex_to_grid(h);
            assert!(row < GRID_SIZE);
            assert!(col < GRID_SIZE);
        }
    }

    #[test]
    fn test_encode_default_board() {
        let board = ZertzBoard::default();
        let mut buf = vec![0.0f32; NUM_CHANNELS * GRID_SIZE * GRID_SIZE];
        encode_board(&board, &mut buf);

        // Default board: all rings empty, supply full
        // Channel 3 (empty rings) should have 37 cells set to 1.0
        let ch3_start = 3 * GRID_SIZE * GRID_SIZE;
        let ch3_end = 4 * GRID_SIZE * GRID_SIZE;
        let empty_count: usize = buf[ch3_start..ch3_end]
            .iter()
            .filter(|&&v| v == 1.0)
            .count();
        assert_eq!(empty_count, BOARD_SIZE);

        // Channel 4 (supply white normalized) should be 1.0 on valid cells
        let ch4_start = 4 * GRID_SIZE * GRID_SIZE;
        let val = buf[ch4_start]; // cell 0 is valid
        assert!((val - 1.0).abs() < 1e-6);
    }
}
