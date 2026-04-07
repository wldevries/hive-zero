use core_game::game::{Game, NNGame, Outcome, Player, PolicyIndex};
use core_game::symmetry::UnitSymmetry;

pub const GRID_SIZE: usize = 3;
pub const NUM_CHANNELS: usize = 2;
pub const RESERVE_SIZE: usize = 0;
pub const NUM_POLICY_CHANNELS: usize = 1;
pub const POLICY_SIZE: usize = NUM_POLICY_CHANNELS * GRID_SIZE * GRID_SIZE; // 9

/// A tic-tac-toe move: cell index 0-8, or 255 for pass (unused but required by trait).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TTTMove(pub u8);

impl TTTMove {
    pub fn cell(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Cell {
    Empty,
    X, // Player1
    O, // Player2
}

const WIN_LINES: [[usize; 3]; 8] = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // cols
    [0, 4, 8], [2, 4, 6],             // diagonals
];

#[derive(Clone)]
pub struct TicTacToe {
    pub board: [Cell; 9],
    pub next_player: Player,
    pub outcome: Outcome,
    pub move_count: u8,
}

impl TicTacToe {
    pub fn new() -> Self {
        TicTacToe {
            board: [Cell::Empty; 9],
            next_player: Player::Player1,
            outcome: Outcome::Ongoing,
            move_count: 0,
        }
    }

    fn player_cell(player: Player) -> Cell {
        match player {
            Player::Player1 => Cell::X,
            Player::Player2 => Cell::O,
        }
    }

    fn check_winner(&self, cell: Cell) -> bool {
        WIN_LINES.iter().any(|line| line.iter().all(|&i| self.board[i] == cell))
    }

    /// Render the board as a string for display.
    pub fn render(&self) -> String {
        let mut s = String::new();
        for row in 0..3 {
            for col in 0..3 {
                let ch = match self.board[row * 3 + col] {
                    Cell::Empty => '.',
                    Cell::X => 'X',
                    Cell::O => 'O',
                };
                s.push(ch);
                if col < 2 { s.push(' '); }
            }
            if row < 2 { s.push('\n'); }
        }
        s
    }
}

impl Default for TicTacToe {
    fn default() -> Self {
        Self::new()
    }
}

impl Game for TicTacToe {
    type Move = TTTMove;
    type Symmetry = UnitSymmetry;

    fn next_player(&self) -> Player {
        self.next_player
    }

    fn outcome(&self) -> Outcome {
        self.outcome
    }

    fn valid_moves(&mut self) -> Vec<TTTMove> {
        if self.outcome != Outcome::Ongoing {
            return Vec::new();
        }
        (0..9u8)
            .filter(|&i| self.board[i as usize] == Cell::Empty)
            .map(TTTMove)
            .collect()
    }

    fn play_move(&mut self, mv: &TTTMove) -> Result<(), String> {
        let idx = mv.cell();
        if idx >= 9 {
            return Err("Invalid cell index".into());
        }
        if self.board[idx] != Cell::Empty {
            return Err(format!("Cell {} is already occupied", idx));
        }
        if self.outcome != Outcome::Ongoing {
            return Err("Game is already over".into());
        }

        let cell = Self::player_cell(self.next_player);
        self.board[idx] = cell;
        self.move_count += 1;

        if self.check_winner(cell) {
            self.outcome = Outcome::WonBy(self.next_player);
        } else if self.move_count == 9 {
            self.outcome = Outcome::Draw;
        }

        self.next_player = self.next_player.opposite();
        Ok(())
    }

    fn pass_move() -> TTTMove {
        TTTMove(255)
    }

    fn is_pass(mv: &TTTMove) -> bool {
        mv.0 == 255
    }
}

impl NNGame for TicTacToe {
    const BOARD_CHANNELS: usize = NUM_CHANNELS;
    const RESERVE_SIZE: usize = RESERVE_SIZE;
    const NUM_POLICY_CHANNELS: usize = NUM_POLICY_CHANNELS;

    fn grid_size(&self) -> usize {
        GRID_SIZE
    }

    fn encode_board(&self, board_out: &mut [f32], _reserve_out: &mut [f32]) {
        // Current-player-relative encoding:
        // Channel 0: current player's pieces
        // Channel 1: opponent's pieces
        let my_cell = Self::player_cell(self.next_player);
        let opp_cell = Self::player_cell(self.next_player.opposite());
        for i in 0..9 {
            board_out[i] = if self.board[i] == my_cell { 1.0 } else { 0.0 };
            board_out[9 + i] = if self.board[i] == opp_cell { 1.0 } else { 0.0 };
        }
    }

    fn get_legal_move_mask(&mut self) -> (Vec<f32>, Vec<(PolicyIndex, TTTMove)>) {
        let moves = self.valid_moves();
        let mut mask = vec![0.0f32; POLICY_SIZE];
        let mut indexed = Vec::with_capacity(moves.len());
        for mv in moves {
            let idx = mv.cell();
            mask[idx] = 1.0;
            indexed.push((PolicyIndex::Single(idx), mv));
        }
        (mask, indexed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game() {
        let game = TicTacToe::new();
        assert_eq!(game.next_player(), Player::Player1);
        assert_eq!(game.outcome(), Outcome::Ongoing);
    }

    #[test]
    fn test_play_and_win() {
        let mut game = TicTacToe::new();
        // X wins with top row: 0, 1, 2
        game.play_move(&TTTMove(0)).unwrap(); // X
        game.play_move(&TTTMove(3)).unwrap(); // O
        game.play_move(&TTTMove(1)).unwrap(); // X
        game.play_move(&TTTMove(4)).unwrap(); // O
        game.play_move(&TTTMove(2)).unwrap(); // X wins
        assert_eq!(game.outcome(), Outcome::WonBy(Player::Player1));
        assert!(game.valid_moves().is_empty());
    }

    #[test]
    fn test_draw() {
        let mut game = TicTacToe::new();
        // X O X
        // X X O
        // O X O
        for &cell in &[0, 1, 2, 4, 3, 5, 7, 6, 8] {
            game.play_move(&TTTMove(cell)).unwrap();
        }
        assert_eq!(game.outcome(), Outcome::Draw);
    }

    #[test]
    fn test_encode_board() {
        let mut game = TicTacToe::new();
        game.play_move(&TTTMove(0)).unwrap(); // X at 0
        game.play_move(&TTTMove(4)).unwrap(); // O at 4
        // Now it's X's turn (Player1)
        let mut board = [0.0f32; 18];
        let mut reserve = [0.0f32; 0];
        game.encode_board(&mut board, &mut reserve);
        // Channel 0 (current=X): cell 0 should be 1
        assert_eq!(board[0], 1.0);
        // Channel 1 (opponent=O): cell 4 should be 1
        assert_eq!(board[9 + 4], 1.0);
    }

    #[test]
    fn test_valid_moves_count() {
        let mut game = TicTacToe::new();
        assert_eq!(game.valid_moves().len(), 9);
        game.play_move(&TTTMove(0)).unwrap();
        assert_eq!(game.valid_moves().len(), 8);
    }
}
