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
pub const NUM_CHANNELS: usize = 4;

/// Reserve vector: [supply_W, supply_G, supply_B, cur_cap_W, cur_cap_G, cur_cap_B, opp_cap_W, opp_cap_G, opp_cap_B]
pub const RESERVE_SIZE: usize = 9;

/// Initial supply for normalization: [6, 8, 10].
const INITIAL_SUPPLY: [f32; 3] = [6.0, 8.0, 10.0];

/// Encode a ZertzBoard into flat tensor buffers.
///
/// `board_out` must have length NUM_CHANNELS * GRID_SIZE * GRID_SIZE (= 196).
/// `reserve_out` must have length RESERVE_SIZE (= 9).
pub fn encode_board(board: &ZertzBoard, board_out: &mut [f32], reserve_out: &mut [f32]) {
    debug_assert_eq!(board_out.len(), NUM_CHANNELS * GRID_SIZE * GRID_SIZE);
    debug_assert_eq!(reserve_out.len(), RESERVE_SIZE);

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

    // Reserve vector: supply and captures normalized by initial supply per color
    reserve_out[0] = supply[0] as f32 / INITIAL_SUPPLY[0];
    reserve_out[1] = supply[1] as f32 / INITIAL_SUPPLY[1];
    reserve_out[2] = supply[2] as f32 / INITIAL_SUPPLY[2];
    reserve_out[3] = captures[cur_pi][0] as f32 / INITIAL_SUPPLY[0];
    reserve_out[4] = captures[cur_pi][1] as f32 / INITIAL_SUPPLY[1];
    reserve_out[5] = captures[cur_pi][2] as f32 / INITIAL_SUPPLY[2];
    reserve_out[6] = captures[opp_pi][0] as f32 / INITIAL_SUPPLY[0];
    reserve_out[7] = captures[opp_pi][1] as f32 / INITIAL_SUPPLY[1];
    reserve_out[8] = captures[opp_pi][2] as f32 / INITIAL_SUPPLY[2];
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
        let mut reserve = vec![0.0f32; RESERVE_SIZE];
        encode_board(&board, &mut buf, &mut reserve);

        // Default board: all rings empty
        // Channel 3 (empty rings) should have 37 cells set to 1.0
        let ch3_start = 3 * GRID_SIZE * GRID_SIZE;
        let ch3_end = 4 * GRID_SIZE * GRID_SIZE;
        let empty_count: usize = buf[ch3_start..ch3_end]
            .iter()
            .filter(|&&v| v == 1.0)
            .count();
        assert_eq!(empty_count, BOARD_SIZE);

        // Reserve: supply fully stocked → all 1.0; captures → all 0.0
        assert!((reserve[0] - 1.0).abs() < 1e-6); // supply W = 6/6
        assert!((reserve[1] - 1.0).abs() < 1e-6); // supply G = 8/8
        assert!((reserve[2] - 1.0).abs() < 1e-6); // supply B = 10/10
        assert_eq!(reserve[3], 0.0);               // cur caps W
        assert_eq!(reserve[6], 0.0);               // opp caps W
    }
}
