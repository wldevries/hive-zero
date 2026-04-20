/// Game state management for Hive.

use core_game::hex::{Hex, hex_distance, hex_neighbors};
use crate::board::Board;
use crate::piece::{Piece, PieceColor, PieceType, PIECE_COUNTS, ALL_PIECE_TYPES, player_pieces};
use crate::rules::{get_moves, get_placements};
use std::collections::HashMap;

use core_game::game::{Game as GameTrait, NNGame, Player, Outcome, PolicyIndex};
use core_game::symmetry::D6Symmetry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameState {
    NotStarted,
    InProgress,
    Draw,
    DrawByRepetition,
    WhiteWins,
    BlackWins,
}

impl GameState {
    pub fn as_str(&self) -> &'static str {
        match self {
            GameState::NotStarted => "NotStarted",
            GameState::InProgress => "InProgress",
            GameState::Draw => "Draw",
            GameState::DrawByRepetition => "DrawByRepetition",
            GameState::WhiteWins => "WhiteWins",
            GameState::BlackWins => "BlackWins",
        }
    }
}

/// A move in the game: (piece, from_pos, to_pos).
/// from_pos is None for placement moves.
/// piece is None for pass moves (to_pos is also meaningless).
#[derive(Debug, Clone, Copy, PartialEq)]
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

    #[allow(dead_code)]
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
    /// When true, queen cannot be placed on the first move of each player.
    pub tournament_mode: bool,
    /// Grid size for NN encoding (may be smaller than physical board GRID_SIZE).
    pub nn_grid_size: usize,
    move_history: Vec<Move>,
    reserves: Reserves,
    /// Reserve snapshots for undo.
    history_reserves: Vec<(u16, u16)>,
    /// Recentering shift applied after each move, for undo.
    history_shifts: Vec<(i8, i8)>,
    /// Position repetition counts keyed by Zobrist hash.
    repetition_counts: HashMap<u64, u8>,
    /// Position keys in move order (includes initial position at index 0).
    repetition_history: Vec<u64>,
}

// Exponential queen danger table (normalized boardspace queen_safety: 0,-5,-10,-15,-40,-65,-120 / 120)
const QUEEN_DANGER_TABLE: [f32; 7] = [0.0, 0.042, 0.083, 0.125, 0.333, 0.542, 1.0];

// Heuristic weights — sum = 4.05; TOTAL_SCALE normalizes to [-1, 1] without clamping.
// DANGER dominates (as in boardspace); ESCAPE is secondary and correlated with DANGER.
// LEGALMOVES subsumes pinned-piece immobility (pinned pieces contribute 0 moves).
const HEURISTIC_DANGER_WEIGHT:     f32 = 2.00;
const HEURISTIC_ESCAPE_WEIGHT:     f32 = 0.60;
const HEURISTIC_ATTACK_WEIGHT:     f32 = 0.50;
const HEURISTIC_SHUTOUT_WEIGHT:    f32 = 0.40;
const HEURISTIC_LEGALMOVES_WEIGHT: f32 = 0.25;
const HEURISTIC_DROP_WEIGHT:       f32 = 0.15;
const HEURISTIC_MOBILITY_WEIGHT:   f32 = 0.15;
const HEURISTIC_TOTAL_SCALE:       f32 = 1.0 / 4.05;

impl Game {
    const ZOBRIST_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

    #[inline]
    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    #[inline]
    fn zobrist_piece_key(piece_idx: usize, row: usize, col: usize, depth: usize) -> u64 {
        let idx = ((((piece_idx * crate::board::GRID_SIZE + row) * crate::board::GRID_SIZE + col)
            * crate::board::MAX_STACK)
            + depth) as u64;
        Self::splitmix64(Self::ZOBRIST_SEED ^ idx)
    }

    #[inline]
    fn zobrist_side_to_move_key() -> u64 {
        Self::splitmix64(Self::ZOBRIST_SEED ^ 0xFFFF_FFFF_FFFF_FFFF)
    }

    /// Zobrist position key including board occupancy, stack order, and side to move.
    fn position_key(&self) -> u64 {
        let mut key = 0u64;

        for row in 0..crate::board::GRID_SIZE {
            for col in 0..crate::board::GRID_SIZE {
                let stack = self.board.stack_at_grid(row, col);
                for (depth, piece) in stack.iter().enumerate() {
                    key ^= Self::zobrist_piece_key(piece.linear_index(), row, col, depth);
                }
            }
        }

        if self.turn_color == PieceColor::Black {
            key ^= Self::zobrist_side_to_move_key();
        }

        key
    }

    fn record_position_and_check_repetition(&mut self) {
        let key = self.position_key();
        self.repetition_history.push(key);
        let count = self.repetition_counts.entry(key).or_insert(0);
        *count = count.saturating_add(1);
        if *count >= 2 {
            self.state = GameState::DrawByRepetition;
        }
    }

    fn undo_repetition_for_last_position(&mut self) {
        if let Some(key) = self.repetition_history.pop() {
            if let Some(count) = self.repetition_counts.get_mut(&key) {
                if *count <= 1 {
                    self.repetition_counts.remove(&key);
                } else {
                    *count -= 1;
                }
            }
        }
    }

    /// Standard game — queen can be placed on the first move.
    pub fn new() -> Self {
        let mut game = Game {
            board: Board::new(),
            state: GameState::NotStarted,
            turn_color: PieceColor::White,
            turn_number: 1,
            move_count: 0,
            tournament_mode: false,
            nn_grid_size: crate::board::GRID_SIZE,
            move_history: Vec::new(),
            reserves: Reserves::new(),
            history_reserves: Vec::new(),
            history_shifts: Vec::new(),
            repetition_counts: HashMap::new(),
            repetition_history: Vec::new(),
        };

        let start_key = game.position_key();
        game.repetition_counts.insert(start_key, 1);
        game.repetition_history.push(start_key);

        game
    }

    /// Standard game with a custom NN encoding grid size.
    pub fn new_with_grid_size(nn_grid_size: usize) -> Self {
        assert!(nn_grid_size % 2 == 1, "nn_grid_size must be odd");
        assert!(nn_grid_size <= crate::board::GRID_SIZE,
                "nn_grid_size {} exceeds board GRID_SIZE {}", nn_grid_size, crate::board::GRID_SIZE);
        Game { nn_grid_size, ..Game::new() }
    }

    /// Tournament game — queen cannot be placed on the first move.
    pub fn new_tournament() -> Self {
        Game { tournament_mode: true, ..Game::new() }
    }

    /// Tournament game with a custom NN encoding grid size.
    pub fn new_tournament_with_grid_size(nn_grid_size: usize) -> Self {
        assert!(nn_grid_size % 2 == 1, "nn_grid_size must be odd");
        assert!(nn_grid_size <= crate::board::GRID_SIZE,
                "nn_grid_size {} exceeds board GRID_SIZE {}", nn_grid_size, crate::board::GRID_SIZE);
        Game { nn_grid_size, tournament_mode: true, ..Game::new() }
    }

    pub fn is_game_over(&self) -> bool {
        matches!(self.state, GameState::Draw | GameState::DrawByRepetition | GameState::WhiteWins | GameState::BlackWins)
    }

    /// Zobrist hash of the current position (board layout + side to move).
    pub fn position_hash(&self) -> u64 {
        self.position_key()
    }

    /// The hash that triggered `DrawByRepetition` (its second occurrence was the last move played).
    /// Returns `None` if the game did not end by position repetition.
    pub fn repetition_trigger_hash(&self) -> Option<u64> {
        if self.state == GameState::DrawByRepetition {
            self.repetition_history.last().copied()
        } else {
            None
        }
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

    pub fn queen_placed(&self, color: PieceColor) -> bool {
        let queen = Piece::new(color, PieceType::Queen, 1);
        !self.reserves.has(queen)
    }

    fn must_place_queen_for(&self, color: PieceColor) -> bool {
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

    fn must_place_queen(&self) -> bool {
        self.must_place_queen_for(self.turn_color)
    }

    /// Return all valid moves as Vec<Move>.
    pub fn valid_moves(&mut self) -> Vec<Move> {
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
            let is_first_move = self.tournament_mode
                && ((color == PieceColor::White && self.move_count == 0)
                    || (color == PieceColor::Black && self.move_count == 1));
            let mut placeable: Vec<Piece> = reserve
                .into_iter()
                .filter(|p| {
                    if must_queen {
                        p.piece_type() == PieceType::Queen
                    } else if is_first_move {
                        // Tournament rule: cannot place queen on first move
                        p.piece_type() != PieceType::Queen
                    } else {
                        true
                    }
                })
                .collect();
            placeable.sort_by_key(|p| p.raw());
            // UHP requires pieces of the same type to be played in order
            // (e.g., wA1 before wA2). Since pieces of the same type are
            // functionally identical for placement, only keep the lowest
            // numbered piece per type to enforce this.
            placeable.dedup_by_key(|p| p.piece_type());

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
                let mut destinations = get_moves(*piece, &mut self.board, &aps);
                destinations.retain(|&d| crate::board::hex_to_grid(d).is_some());
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
    pub fn play_move(&mut self, mv: &Move) -> Result<(), String> {
        // Save reserve state for undo
        let (wr, br) = self.reserves.raw();
        self.history_reserves.push((wr, br));

        if let Some(piece) = mv.piece {
            if mv.from.is_none() {
                // Placement
                let to = mv.to.unwrap();
                self.reserves.remove(piece);
                self.board.place_piece(piece, to)?;
            } else {
                // Movement
                let to = mv.to.unwrap();
                self.board.move_piece(piece, to)?;
            }
        }

        // Recenter the board after each non-pass move so pieces stay near (0,0).
        // Store the shift for undo.
        let shift = if mv.piece.is_some() {
            self.board.recenter(self.nn_grid_size) // recenter only when pieces approach NN view boundary
        } else {
            (0, 0)
        };
        self.history_shifts.push(shift);

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
        if self.state == GameState::InProgress {
            self.record_position_and_check_repetition();
        }
        Ok(())
    }

    /// Execute a pass.
    pub fn play_pass(&mut self) {
        self.play_move(&Move::pass()).unwrap();
    }

    /// Undo the last move.
    pub fn undo(&mut self) {
        if self.move_history.is_empty() {
            return;
        }

        // Remove repetition entry for the position after the move being undone.
        self.undo_repetition_for_last_position();

        let mv = self.move_history.pop().unwrap();
        let (wr, br) = self.history_reserves.pop().unwrap();
        let (dq, dr) = self.history_shifts.pop().unwrap();
        self.reserves.restore(wr, br);

        // Reverse the recentering shift so move coordinates are valid again.
        self.board.apply_shift(-dq, -dr);

        if let Some(piece) = mv.piece {
            if mv.from.is_none() {
                // Undo placement
                self.board.remove_piece(piece).expect("undo: piece must be on board");
            } else {
                // Undo movement
                let from = mv.from.unwrap();
                self.board.move_piece(piece, from)
                    .expect("undo coordinates must be in bounds");
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

    pub fn queen_danger(&self, color: PieceColor) -> f32 {
        let queen = Piece::new(color, PieceType::Queen, 1);
        match self.board.piece_position(queen) {
            None => 0.0,
            Some(pos) => {
                let neighbors = hex_neighbors(pos)
                    .iter()
                    .filter(|&&neighbor| self.board.is_occupied(neighbor))
                    .count();
                // Enemy beetle on top counts as one extra virtual neighbor (~0.75 in boardspace).
                let beetle_on_top = self.board.top_piece(pos) != Some(queen);
                let effective = (neighbors + usize::from(beetle_on_top)).min(6);
                QUEEN_DANGER_TABLE[effective]
            }
        }
    }

    pub fn queen_escape(&self, color: PieceColor) -> f32 {
        let queen = Piece::new(color, PieceType::Queen, 1);
        match self.board.piece_position(queen) {
            None => 0.0,
            Some(pos) => {
                if self.board.top_piece(pos) != Some(queen) {
                    return 0.0;
                }
                if self.board.stack_height(pos) == 1 {
                    let articulation_points = self.board.articulation_points();
                    if articulation_points.contains(&pos) {
                        return 0.0;
                    }
                }

                let mut count = 0u32;
                for &neighbor in hex_neighbors(pos).iter() {
                    if !self.board.is_occupied(neighbor) && self.board.can_slide(pos, neighbor) {
                        if hex_neighbors(neighbor)
                            .iter()
                            .any(|&adjacent| adjacent != pos && self.board.is_occupied(adjacent))
                        {
                            count += 1;
                        }
                    }
                }
                // Most perimeter pieces have exactly 2 slide options; 0 is "dead/pinned".
                (count as f32 / 2.0).min(1.0)
            }
        }
    }

    pub fn piece_mobility(&self, color: PieceColor) -> f32 {
        let mut mobile_count = 0u32;
        let on_board = self.board.pieces_on_board(color);
        let in_reserve = self.reserves.pieces_in_reserve(color);
        let total_pieces = (on_board.len() + in_reserve.len()) as f32;

        if total_pieces == 0.0 {
            return 0.0;
        }

        // 1. Check movement mobility for pieces already on board
        // (Only possible if queen is already placed and we're not forced to place queen)
        if self.queen_placed(color) && !self.must_place_queen_for(color) {
            let articulation_points = self.board.articulation_points();
            let mut board_clone = self.board.clone();
            for &piece in &on_board {
                let moves = crate::rules::get_moves(piece, &mut board_clone, &articulation_points);
                if !moves.is_empty() {
                    mobile_count += 1;
                }
            }
        }

        // 2. Check placement mobility for pieces in reserve
        if !in_reserve.is_empty() {
            let placement_hexes = crate::rules::get_placements(color, &self.board);
            if !placement_hexes.is_empty() {
                if self.must_place_queen_for(color) {
                    // Only the queen can be placed
                    mobile_count += 1;
                } else {
                    // All pieces in reserve are "mobile" (can be placed)
                    mobile_count += in_reserve.len() as u32;
                }
            }
        }

        // Normalize by total pieces available to this player
        (mobile_count as f32 / total_pieces).min(1.0)
    }

    // Own pieces near enemy queen, weighted by 1/distance (0-1).
    fn attack_pressure(&self, attacker: PieceColor) -> f32 {
        let defender = attacker.opposite();
        let queen = Piece::new(defender, PieceType::Queen, 1);
        let queen_pos = match self.board.piece_position(queen) {
            None => return 0.0,
            Some(p) => p,
        };
        let mut pressure = 0.0f32;
        let on_board = self.board.pieces_on_board(attacker);
        for &piece in &on_board {
            if let Some(pos) = self.board.piece_position(piece) {
                let dist = hex_distance(pos, queen_pos);
                if dist > 0 && dist <= 4 {
                    pressure += 1.0 / dist as f32;
                }
            }
        }
        (pressure / 6.0).min(1.0)
    }

    // Total legal moves available (normalized) and shutout flag (1.0 if zero moves).
    fn legal_moves_info(&self, color: PieceColor) -> (f32, f32) {
        let mut count: usize = 0;
        if self.queen_placed(color) && !self.must_place_queen_for(color) {
            let articulation_points = self.board.articulation_points();
            let mut board_clone = self.board.clone();
            let on_board = self.board.pieces_on_board(color);
            for &piece in &on_board {
                count += get_moves(piece, &mut board_clone, &articulation_points).len();
            }
        }
        let in_reserve = self.reserves.pieces_in_reserve(color);
        if !in_reserve.is_empty() {
            let placement_hexes = get_placements(color, &self.board);
            if !placement_hexes.is_empty() {
                let placeable = if self.must_place_queen_for(color) { 1 } else { in_reserve.len() };
                count += placeable * placement_hexes.len();
            }
        }
        let score = (count as f32 / 15.0).min(1.0);
        let shutout = if count == 0 { 1.0 } else { 0.0 };
        (score, shutout)
    }

    // Valid placement hexes available, normalized (0-1). Zero if reserve is empty.
    fn drop_count(&self, color: PieceColor) -> f32 {
        if self.reserves.pieces_in_reserve(color).is_empty() {
            return 0.0;
        }
        let placements = get_placements(color, &self.board);
        (placements.len() as f32 / 10.0).min(1.0)
    }

    /// Heuristic evaluation for unfinished games.
    /// Returns (white_score, black_score) in range [-1, 1].
    pub fn heuristic_value(&self) -> (f32, f32) {
        let w_danger  = self.queen_danger(PieceColor::White);
        let b_danger  = self.queen_danger(PieceColor::Black);
        let w_escape  = self.queen_escape(PieceColor::White);
        let b_escape  = self.queen_escape(PieceColor::Black);
        let w_attack  = self.attack_pressure(PieceColor::White);
        let b_attack  = self.attack_pressure(PieceColor::Black);
        let (w_legal, w_shutout) = self.legal_moves_info(PieceColor::White);
        let (b_legal, b_shutout) = self.legal_moves_info(PieceColor::Black);
        let w_drop    = self.drop_count(PieceColor::White);
        let b_drop    = self.drop_count(PieceColor::Black);
        let w_mob     = self.piece_mobility(PieceColor::White);
        let b_mob     = self.piece_mobility(PieceColor::Black);

        let w_score = (HEURISTIC_TOTAL_SCALE * (
            HEURISTIC_DANGER_WEIGHT     * (b_danger  - w_danger)  +
            HEURISTIC_ESCAPE_WEIGHT     * (w_escape  - b_escape)  +
            HEURISTIC_ATTACK_WEIGHT     * (w_attack  - b_attack)  +
            HEURISTIC_LEGALMOVES_WEIGHT * (w_legal   - b_legal)   +
            HEURISTIC_SHUTOUT_WEIGHT    * (b_shutout - w_shutout) +
            HEURISTIC_DROP_WEIGHT       * (w_drop    - b_drop)    +
            HEURISTIC_MOBILITY_WEIGHT   * (w_mob     - b_mob)
        )).clamp(-1.0, 1.0);

        (w_score, -w_score)
    }

    pub fn move_history(&self) -> &[Move] {
        &self.move_history
    }

    /// Return the recentering shift that was applied by the last played move.
    /// Returns (0, 0) if no moves have been played yet.
    pub fn last_recenter_shift(&self) -> (i8, i8) {
        *self.history_shifts.last().unwrap_or(&(0, 0))
    }

    /// Get the last move's source and destination adjusted for that move's recentering shift.
    /// Returns (source_adjusted, dest_adjusted) or (None, None) if no moves.
    pub fn last_move_display_coords(&self) -> (Option<Hex>, Option<Hex>) {
        if self.move_history.is_empty() {
            return (None, None);
        }

        // Move coordinates are recorded in the board frame before that move's recenter.
        // To display the latest move on the current board, apply only the latest shift.
        let (dq, dr) = *self.history_shifts.last().unwrap_or(&(0, 0));

        let last_move = self.move_history.last().unwrap();
        let source_adj = last_move.from.map(|(q, r)| (
            q.saturating_add(dq),
            r.saturating_add(dr),
        ));
        let dest_adj = last_move.to.map(|(q, r)| (
            q.saturating_add(dq),
            r.saturating_add(dr),
        ));

        (source_adj, dest_adj)
    }

    /// Generate UHP GameString: "Base;State;Turn;move1;move2;..."
    pub fn game_string(&self) -> String {
        let mut parts = vec![
            "Base".to_string(),
            self.state.as_str().to_string(),
            self.turn_string(),
        ];
        // Replay moves to format each in context
        let mut replay = if self.tournament_mode {
            Game::new_tournament_with_grid_size(self.nn_grid_size)
        } else {
            Game::new_with_grid_size(self.nn_grid_size)
        };
        for mv in &self.move_history {
            if mv.is_pass() {
                parts.push("pass".to_string());
                replay.play_pass();
            } else {
                let uhp = crate::uhp::format_move_uhp(&replay, mv);
                parts.push(uhp);
                replay.play_move(mv).unwrap();
            }
        }
        parts.join(";")
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

// --- GameEngine trait implementation ---

impl Game {
    /// Convert PieceColor to generic Player.
    pub fn color_to_player(color: PieceColor) -> Player {
        match color {
            PieceColor::White => Player::Player1,
            PieceColor::Black => Player::Player2,
        }
    }

    /// Convert generic Player to PieceColor.
    pub fn player_to_color(player: Player) -> PieceColor {
        match player {
            Player::Player1 => PieceColor::White,
            Player::Player2 => PieceColor::Black,
        }
    }
}

impl GameTrait for Game {
    type Move = Move;
    type Symmetry = D6Symmetry;

    fn next_player(&self) -> Player {
        Game::color_to_player(self.turn_color)
    }

    fn outcome(&self) -> Outcome {
        match self.state {
            GameState::WhiteWins => Outcome::WonBy(Player::Player1),
            GameState::BlackWins => Outcome::WonBy(Player::Player2),
            GameState::Draw | GameState::DrawByRepetition => Outcome::Draw,
            _ => Outcome::Ongoing,
        }
    }

    fn valid_moves(&mut self) -> Vec<Move> {
        Game::valid_moves(self)
    }

    fn play_move(&mut self, mv: &Move) -> Result<(), String> {
        Game::play_move(self, mv)
    }

    fn pass_move() -> Move {
        Move::pass()
    }

    fn is_pass(mv: &Move) -> bool {
        mv.is_pass()
    }
}

impl NNGame for Game {
    const BOARD_CHANNELS: usize = crate::board_encoding::NUM_CHANNELS;
    const RESERVE_SIZE: usize = crate::board_encoding::RESERVE_SIZE;
    const NUM_POLICY_CHANNELS: usize = crate::move_encoding::NUM_POLICY_CHANNELS;

    fn grid_size(&self) -> usize {
        self.nn_grid_size
    }

    fn encode_board(&self, board_out: &mut [f32], reserve_out: &mut [f32]) {
        crate::board_encoding::encode_board(self, board_out, reserve_out, self.nn_grid_size);
    }

    fn get_legal_move_mask(&mut self) -> (Vec<f32>, Vec<(PolicyIndex, Move)>) {
        crate::move_encoding::get_legal_move_mask(self, self.nn_grid_size)
    }
}

/// Test helpers: direct board construction that bypasses move validation.
/// Only compiled for `#[cfg(test)]` so private fields are accessible.
#[cfg(test)]
impl Game {
    /// Build a test position by placing pieces directly on the board, bypassing
    /// all move-validation rules. Useful for constructing near-terminal positions
    /// in MCTS tests without requiring a full legal game sequence.
    ///
    /// `pieces`: list of (Piece, hex) to place and remove from reserves.
    /// `turn`: which player is to move.
    /// `move_count`: sets game.move_count (non-zero marks the state as InProgress).
    /// `nn_grid_size`: must be odd and ≤ GRID_SIZE (use 7 for fast tests).
    pub(crate) fn test_position(
        pieces: &[(Piece, Hex)],
        turn: PieceColor,
        move_count: u16,
        nn_grid_size: usize,
    ) -> Self {
        let mut game = Game::new_with_grid_size(nn_grid_size);
        game.turn_color = turn;
        game.move_count = move_count;
        game.state = if move_count > 0 {
            GameState::InProgress
        } else {
            GameState::NotStarted
        };
        for &(piece, hex) in pieces {
            game.board.place_piece(piece, hex).expect("test_position: piece placement failed");
            game.reserves.remove(piece);
        }
        game
    }
}

#[cfg(test)]
#[allow(unused_must_use)]
mod tests {
    use super::*;
    use crate::uhp::{format_move_uhp, parse_and_play_uhp};

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
        // First move: 5 piece types at origin (all types, deduped by type)
        assert_eq!(moves.len(), 5);
        for mv in &moves {
            assert!(mv.from.is_none()); // all placements
            assert_eq!(mv.to, Some((0, 0)));
            assert_eq!(mv.piece.unwrap().number(), 1); // always #1 first
        }

        // Play first move
        game.play_move(&moves[0]);
        assert_eq!(game.state, GameState::InProgress);
        assert_eq!(game.turn_color, PieceColor::Black);
    }

    #[test]
    fn test_first_move_tournament() {
        let mut game = Game::new_tournament();
        let moves = game.valid_moves();
        // Tournament: 4 piece types at origin (queen excluded)
        assert_eq!(moves.len(), 4);
        for mv in &moves {
            assert_ne!(mv.piece.unwrap().piece_type(), PieceType::Queen);
        }
    }

    #[test]
    fn test_second_move() {
        let mut game = Game::new();
        let ws1 = Piece::new(PieceColor::White, PieceType::Spider, 1);
        game.play_move(&Move::placement(ws1, (0, 0)));

        // Black: 5 piece types * 6 positions = 30 moves
        let moves = game.valid_moves();
        assert_eq!(moves.len(), 5 * 6);
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

        // Play 6 moves (3 per player) without placing a queen, using legal moves from
        // valid_moves() so coordinates are always correct even after recentering.
        for _ in 0..6 {
            let moves = game.valid_moves();
            let mv = *moves.iter()
                .find(|m| m.from.is_none() && m.piece.map_or(false, |p| p.piece_type() != PieceType::Queen))
                .unwrap();
            game.play_move(&mv).unwrap();
        }

        // White turn 4: MUST place queen
        let moves = game.valid_moves();
        assert!(!moves.is_empty());
        assert!(moves.iter().all(|m| m.piece.unwrap().piece_type() == PieceType::Queen));
    }

    #[test]
    fn test_last_move_display_coords_uses_only_latest_shift() {
        let mut game = Game::new();
        game.move_history.push(Move::movement(
            Piece::new(PieceColor::White, PieceType::Ant, 1),
            (4, -2),
            (5, -1),
        ));
        game.history_shifts.push((3, 1));
        game.history_shifts.push((-2, 4));

        let (source, dest) = game.last_move_display_coords();
        assert_eq!(source, Some((2, 2)));
        assert_eq!(dest, Some((3, 3)));
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

    #[test]
    fn test_position_key_differs_by_side_to_move() {
        let mut game_white = Game::new();
        let mut game_black = game_white.clone();

        game_white.turn_color = PieceColor::White;
        game_black.turn_color = PieceColor::Black;

        assert_ne!(game_white.position_key(), game_black.position_key());
    }

    #[test]
    fn test_position_key_includes_stack_order() {
        let mut game_a = Game::new();
        let mut game_b = Game::new();

        let wq = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let bb1 = Piece::new(PieceColor::Black, PieceType::Beetle, 1);

        game_a.board.place_piece(wq, (0, 0)).unwrap();
        game_a.board.place_piece(bb1, (0, 0)).unwrap();

        game_b.board.place_piece(bb1, (0, 0)).unwrap();
        game_b.board.place_piece(wq, (0, 0)).unwrap();

        assert_ne!(game_a.position_key(), game_b.position_key());
    }

    #[test]
    fn test_twofold_repetition_draw() {
        let mut game = Game::new();

        // One full pass cycle repeats the initial side-to-move position for White 2 times:
        // initial + after move 2.
        game.play_pass();
        assert_eq!(game.state, GameState::InProgress);
        game.play_pass();
        assert_eq!(game.state, GameState::DrawByRepetition);
    }

    #[test]
    fn test_undo_clears_twofold_draw_state() {
        let mut game = Game::new();

        game.play_pass();
        game.play_pass();
        assert_eq!(game.state, GameState::DrawByRepetition);

        game.undo();
        assert_eq!(game.state, GameState::InProgress);

        game.play_pass();
        assert_eq!(game.state, GameState::DrawByRepetition);
    }

    #[test]
    fn test_game_string_roundtrip_preserves_small_grid_recenter_behavior() {
        let mut game = Game::new_with_grid_size(7);

        for _ in 0..5 {
            let mv = game.valid_moves()[0];
            let uhp = format_move_uhp(&game, &mv);
            assert!(parse_and_play_uhp(&mut game, &uhp), "failed to play live move {uhp}");
        }

        let game_string = game.game_string();
        assert_eq!(game_string, "Base;InProgress;Black[3];wQ;bQ -wQ;wS1 wQ\\;bS1 -bQ;wS2 /wS1");

        let mut replay = Game::new_with_grid_size(7);
        for move_str in game_string.split(';').skip(3) {
            assert!(parse_and_play_uhp(&mut replay, move_str), "roundtrip failed on {move_str}");
        }
    }
}
