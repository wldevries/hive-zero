/// Game state management for Hive.

use crate::hex::{Hex, hex_neighbors};
use crate::board::Board;
use crate::piece::{Piece, PieceColor, PieceType, PIECE_COUNTS, ALL_PIECE_TYPES, player_pieces};
use crate::rules::{get_moves, get_placements};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameState {
    NotStarted,
    InProgress,
    Draw,
    WhiteWins,
    BlackWins,
}

impl GameState {
    pub fn as_str(&self) -> &'static str {
        match self {
            GameState::NotStarted => "NotStarted",
            GameState::InProgress => "InProgress",
            GameState::Draw => "Draw",
            GameState::WhiteWins => "WhiteWins",
            GameState::BlackWins => "BlackWins",
        }
    }
}

/// A move in the game: (piece, from_pos, to_pos).
/// from_pos is None for placement moves.
/// piece is None for pass moves (to_pos is also meaningless).
#[derive(Debug, Clone, Copy)]
pub struct Move {
    pub piece: Option<Piece>,
    pub from: Option<Hex>,
    pub to: Option<Hex>,
}

impl Move {
    pub fn placement(piece: Piece, to: Hex) -> Self {
        Move { piece: Some(piece), from: None, to: Some(to) }
    }

    pub fn movement(piece: Piece, from: Hex, to: Hex) -> Self {
        Move { piece: Some(piece), from: Some(from), to: Some(to) }
    }

    pub fn pass() -> Self {
        Move { piece: None, from: None, to: None }
    }

    pub fn is_pass(&self) -> bool {
        self.piece.is_none()
    }
}

/// Reserve tracking using bitfields.
/// 11 pieces per player, tracked as a u16 bitmask.
#[derive(Clone)]
struct Reserves {
    /// Bitmask of pieces still in reserve (1 = in reserve).
    /// Bit layout matches piece linear_index within a color.
    white: u16,
    black: u16,
}

impl Reserves {
    fn new() -> Self {
        // All 11 pieces start in reserve
        let mask = (1u16 << 11) - 1; // bits 0..10 set
        Reserves {
            white: mask,
            black: mask,
        }
    }

    fn has(&self, piece: Piece) -> bool {
        let (mask, bit) = self.piece_bit(piece);
        (mask & (1 << bit)) != 0
    }

    fn remove(&mut self, piece: Piece) {
        let (_, bit) = self.piece_bit(piece);
        match piece.color() {
            PieceColor::White => self.white &= !(1 << bit),
            PieceColor::Black => self.black &= !(1 << bit),
        }
    }

    fn add(&mut self, piece: Piece) {
        let (_, bit) = self.piece_bit(piece);
        match piece.color() {
            PieceColor::White => self.white |= 1 << bit,
            PieceColor::Black => self.black |= 1 << bit,
        }
    }

    /// Count pieces of a specific type in reserve for a color.
    fn count_type(&self, color: PieceColor, piece_type: PieceType) -> u8 {
        let mask = match color {
            PieceColor::White => self.white,
            PieceColor::Black => self.black,
        };

        // Find bit range for this piece type
        let mut offset = 0usize;
        for (i, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
            let count = PIECE_COUNTS[i] as usize;
            if pt == piece_type {
                let mut total = 0u8;
                for j in 0..count {
                    if (mask & (1 << (offset + j))) != 0 {
                        total += 1;
                    }
                }
                return total;
            }
            offset += count;
        }
        0
    }

    /// Get pieces in reserve for iteration.
    fn pieces_in_reserve(&self, color: PieceColor) -> Vec<Piece> {
        let all = player_pieces(color);
        all.into_iter().filter(|p| self.has(*p)).collect()
    }

    fn piece_bit(&self, piece: Piece) -> (u16, usize) {
        // Compute bit position within color's bitmask
        let mut offset = 0usize;
        for (i, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
            if pt == piece.piece_type() {
                let bit = offset + (piece.number() as usize - 1);
                let mask = match piece.color() {
                    PieceColor::White => self.white,
                    PieceColor::Black => self.black,
                };
                return (mask, bit);
            }
            offset += PIECE_COUNTS[i] as usize;
        }
        unreachable!()
    }

    fn raw(&self) -> (u16, u16) {
        (self.white, self.black)
    }

    fn restore(&mut self, white: u16, black: u16) {
        self.white = white;
        self.black = black;
    }
}

/// Full Hive game state with move history and undo support.
#[derive(Clone)]
pub struct Game {
    pub board: Board,
    pub state: GameState,
    pub turn_color: PieceColor,
    pub turn_number: u16,
    pub move_count: u16,
    move_history: Vec<Move>,
    reserves: Reserves,
    /// Reserve snapshots for undo.
    history_reserves: Vec<(u16, u16)>,
}

impl Game {
    pub fn new() -> Self {
        Game {
            board: Board::new(),
            state: GameState::NotStarted,
            turn_color: PieceColor::White,
            turn_number: 1,
            move_count: 0,
            move_history: Vec::new(),
            reserves: Reserves::new(),
            history_reserves: Vec::new(),
        }
    }

    pub fn is_game_over(&self) -> bool {
        matches!(self.state, GameState::Draw | GameState::WhiteWins | GameState::BlackWins)
    }

    pub fn reserve_has(&self, piece: Piece) -> bool {
        self.reserves.has(piece)
    }

    pub fn reserve_count(&self, color: PieceColor, piece_type: PieceType) -> u8 {
        self.reserves.count_type(color, piece_type)
    }

    pub fn reserve_pieces(&self, color: PieceColor) -> Vec<Piece> {
        self.reserves.pieces_in_reserve(color)
    }

    fn queen_placed(&self, color: PieceColor) -> bool {
        let queen = Piece::new(color, PieceType::Queen, 1);
        !self.reserves.has(queen)
    }

    fn must_place_queen(&self) -> bool {
        let color = self.turn_color;
        let queen = Piece::new(color, PieceType::Queen, 1);
        if !self.reserves.has(queen) {
            return false; // already placed
        }

        let player_moves = if color == PieceColor::White {
            (self.move_count + 1) / 2
        } else {
            self.move_count / 2
        };

        player_moves >= 3
    }

    /// Return all valid moves as Vec<Move>.
    pub fn valid_moves(&self) -> Vec<Move> {
        if self.is_game_over() {
            return Vec::new();
        }

        let color = self.turn_color;
        let mut moves = Vec::new();
        let must_queen = self.must_place_queen();

        // Placement moves
        let placement_hexes = get_placements(color, &self.board);
        if !placement_hexes.is_empty() {
            let reserve = self.reserves.pieces_in_reserve(color);
            let mut placeable: Vec<Piece> = reserve
                .into_iter()
                .filter(|p| !must_queen || p.piece_type() == PieceType::Queen)
                .collect();
            placeable.sort_by_key(|p| p.raw());

            for piece in &placeable {
                let mut sorted_hexes = placement_hexes.clone();
                sorted_hexes.sort();
                for pos in sorted_hexes {
                    moves.push(Move::placement(*piece, pos));
                }
            }
        }

        // Movement moves (only if queen placed and not forced to place queen)
        if !must_queen && self.queen_placed(color) {
            let aps = self.board.articulation_points();
            let mut on_board = self.board.pieces_on_board(color);
            on_board.sort_by_key(|p| p.raw());

            for piece in &on_board {
                let mut destinations = get_moves(*piece, &self.board, &aps);
                destinations.sort();
                let pos = self.board.piece_position(*piece).unwrap();
                for dest in destinations {
                    moves.push(Move::movement(*piece, pos, dest));
                }
            }
        }

        moves
    }

    /// Execute a move.
    pub fn play_move(&mut self, mv: &Move) {
        // Save reserve state for undo
        let (wr, br) = self.reserves.raw();
        self.history_reserves.push((wr, br));

        if let Some(piece) = mv.piece {
            if mv.from.is_none() {
                // Placement
                let to = mv.to.unwrap();
                self.reserves.remove(piece);
                self.board.place_piece(piece, to);
            } else {
                // Movement
                let to = mv.to.unwrap();
                self.board.move_piece(piece, to);
            }
        }

        self.move_history.push(*mv);

        if self.state == GameState::NotStarted {
            self.state = GameState::InProgress;
        }

        self.move_count += 1;
        if self.turn_color == PieceColor::Black {
            self.turn_number += 1;
        }
        self.turn_color = self.turn_color.opposite();

        self.check_game_end();
    }

    /// Execute a pass.
    pub fn play_pass(&mut self) {
        self.play_move(&Move::pass());
    }

    /// Undo the last move.
    pub fn undo(&mut self) {
        if self.move_history.is_empty() {
            return;
        }

        let mv = self.move_history.pop().unwrap();
        let (wr, br) = self.history_reserves.pop().unwrap();
        self.reserves.restore(wr, br);

        if let Some(piece) = mv.piece {
            if mv.from.is_none() {
                // Undo placement
                self.board.remove_piece(piece);
            } else {
                // Undo movement
                let from = mv.from.unwrap();
                self.board.move_piece(piece, from);
            }
        }

        self.move_count -= 1;
        if self.turn_color == PieceColor::White {
            self.turn_number -= 1;
        }
        self.turn_color = self.turn_color.opposite();

        if self.move_count == 0 {
            self.state = GameState::NotStarted;
        } else {
            self.state = GameState::InProgress;
            self.check_game_end();
        }
    }

    fn check_game_end(&mut self) {
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let bq = Piece::new(PieceColor::Black, PieceType::Queen, 1);

        let w_pos = self.board.piece_position(wq);
        let b_pos = self.board.piece_position(bq);

        let w_surrounded = w_pos.map_or(false, |pos| {
            hex_neighbors(pos).iter().all(|&n| self.board.is_occupied(n))
        });
        let b_surrounded = b_pos.map_or(false, |pos| {
            hex_neighbors(pos).iter().all(|&n| self.board.is_occupied(n))
        });

        if w_surrounded && b_surrounded {
            self.state = GameState::Draw;
        } else if w_surrounded {
            self.state = GameState::BlackWins;
        } else if b_surrounded {
            self.state = GameState::WhiteWins;
        }
    }

    pub fn move_history(&self) -> &[Move] {
        &self.move_history
    }

    /// Turn string like "White[1]".
    pub fn turn_string(&self) -> String {
        let color_name = match self.turn_color {
            PieceColor::White => "White",
            PieceColor::Black => "Black",
        };
        format!("{}[{}]", color_name, self.turn_number)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game() {
        let game = Game::new();
        assert_eq!(game.state, GameState::NotStarted);
        assert_eq!(game.turn_color, PieceColor::White);
        assert_eq!(game.move_count, 0);
    }

    #[test]
    fn test_first_move() {
        let mut game = Game::new();
        let moves = game.valid_moves();
        // First move: all 5 piece types (11 pieces) at origin
        assert_eq!(moves.len(), 11);
        for mv in &moves {
            assert!(mv.from.is_none()); // all placements
            assert_eq!(mv.to, Some((0, 0)));
        }

        // Play first move
        game.play_move(&moves[0]);
        assert_eq!(game.state, GameState::InProgress);
        assert_eq!(game.turn_color, PieceColor::Black);
    }

    #[test]
    fn test_second_move() {
        let mut game = Game::new();
        // White places queen
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        game.play_move(&Move::placement(wq, (0, 0)));

        // Black should have 11 pieces * 6 positions = 66 moves
        let moves = game.valid_moves();
        assert_eq!(moves.len(), 11 * 6);
    }

    #[test]
    fn test_undo() {
        let mut game = Game::new();
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        game.play_move(&Move::placement(wq, (0, 0)));

        assert_eq!(game.turn_color, PieceColor::Black);
        assert_eq!(game.move_count, 1);

        game.undo();
        assert_eq!(game.turn_color, PieceColor::White);
        assert_eq!(game.move_count, 0);
        assert_eq!(game.state, GameState::NotStarted);
        assert!(game.reserve_has(wq));
    }

    #[test]
    fn test_queen_must_be_placed_by_turn_4() {
        let mut game = Game::new();

        // White turn 1: place ant
        let wa1 = Piece::new(PieceColor::White, PieceType::Ant, 1);
        game.play_move(&Move::placement(wa1, (0, 0)));

        // Black turn 1: place ant
        let ba1 = Piece::new(PieceColor::Black, PieceType::Ant, 1);
        game.play_move(&Move::placement(ba1, (1, 0)));

        // White turn 2: place spider
        let ws1 = Piece::new(PieceColor::White, PieceType::Spider, 1);
        game.play_move(&Move::placement(ws1, (-1, 0)));

        // Black turn 2: place spider
        let bs1 = Piece::new(PieceColor::Black, PieceType::Spider, 1);
        game.play_move(&Move::placement(bs1, (2, 0)));

        // White turn 3: place another
        let wa2 = Piece::new(PieceColor::White, PieceType::Ant, 2);
        game.play_move(&Move::placement(wa2, (-2, 0)));

        // Black turn 3: place another
        let ba2 = Piece::new(PieceColor::Black, PieceType::Ant, 2);
        game.play_move(&Move::placement(ba2, (3, 0)));

        // White turn 4: MUST place queen
        let moves = game.valid_moves();
        assert!(moves.iter().all(|m| {
            m.piece.unwrap().piece_type() == PieceType::Queen
        }));
    }

    #[test]
    fn test_game_end_queen_surrounded() {
        let mut game = Game::new();
        // Manually set up a position where white queen is surrounded
        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let bq = Piece::new(PieceColor::Black, PieceType::Queen, 1);

        // Place white queen and surround it
        game.board.place_piece(wq, (0, 0));
        game.reserves.remove(wq);
        game.board.place_piece(bq, (1, 0));
        game.reserves.remove(bq);

        // Surround white queen with pieces
        let pieces = [
            Piece::new(PieceColor::Black, PieceType::Ant, 1),
            Piece::new(PieceColor::Black, PieceType::Ant, 2),
            Piece::new(PieceColor::Black, PieceType::Spider, 1),
            Piece::new(PieceColor::Black, PieceType::Spider, 2),
            Piece::new(PieceColor::Black, PieceType::Beetle, 1),
        ];
        let surrounding = [(1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];

        for (piece, pos) in pieces.iter().zip(surrounding.iter()) {
            game.board.place_piece(*piece, *pos);
            game.reserves.remove(*piece);
        }

        game.state = GameState::InProgress;
        game.check_game_end();
        assert_eq!(game.state, GameState::BlackWins);
    }
}
