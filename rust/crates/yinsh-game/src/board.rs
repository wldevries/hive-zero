/// YINSH game state and rules.

use core_game::game::{Game, Outcome, Player};
use core_game::symmetry::UnitSymmetry;

use crate::hex::{
    BOARD_SIZE, DIRECTIONS, ROW_DIRS, cell_index_i8, index_to_cell, is_valid_i8,
};

pub const INITIAL_MARKERS: u8 = 51;
pub const RINGS_PER_PLAYER: u8 = 5;
pub const WIN_SCORE: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Empty,
    WhiteRing,
    BlackRing,
    WhiteMarker,
    BlackMarker,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Setup,
    Normal,
    RemoveRow,
    RemoveRing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YinshMove {
    PlaceRing(usize),
    MoveRing { from: usize, to: usize },
    RemoveRow { start: usize, dir: usize },
    RemoveRing(usize),
    Pass,
}

#[derive(Clone)]
pub struct YinshBoard {
    pub cells: [Cell; BOARD_SIZE],
    pub white_rings_placed: u8,
    pub black_rings_placed: u8,
    pub white_score: u8,
    pub black_score: u8,
    pub markers_in_pool: u8,
    pub next_player: Player,
    pub outcome: Outcome,
    pub phase: Phase,
    pub pending_rows: Vec<(usize, usize)>,
    pub pending_rings: Vec<Player>,
    pub deferred_rows: Vec<(usize, usize)>,
    pub deferred_player: Option<Player>,
    pub original_mover: Option<Player>,
}

impl Default for YinshBoard {
    fn default() -> Self {
        Self::new()
    }
}

impl YinshBoard {
    pub fn new() -> Self {
        Self {
            cells: [Cell::Empty; BOARD_SIZE],
            white_rings_placed: 0,
            black_rings_placed: 0,
            white_score: 0,
            black_score: 0,
            markers_in_pool: INITIAL_MARKERS,
            next_player: Player::Player1,
            outcome: Outcome::Ongoing,
            phase: Phase::Setup,
            pending_rows: Vec::new(),
            pending_rings: Vec::new(),
            deferred_rows: Vec::new(),
            deferred_player: None,
            original_mover: None,
        }
    }

    fn my_ring(&self, p: Player) -> Cell {
        match p {
            Player::Player1 => Cell::WhiteRing,
            Player::Player2 => Cell::BlackRing,
        }
    }

    fn my_marker(&self, p: Player) -> Cell {
        match p {
            Player::Player1 => Cell::WhiteMarker,
            Player::Player2 => Cell::BlackMarker,
        }
    }

    /// All legal ring destinations from `from_idx`.
    pub fn ring_destinations(&self, from_idx: usize) -> Vec<usize> {
        let (fc, fr) = index_to_cell(from_idx);
        let mut dests = Vec::new();
        for &(dc, dr) in &DIRECTIONS {
            let mut c = fc as i8 + dc;
            let mut r = fr as i8 + dr;
            let mut jumping = false;
            while is_valid_i8(c, r) {
                let idx = cell_index_i8(c, r);
                match self.cells[idx] {
                    Cell::WhiteRing | Cell::BlackRing => break,
                    Cell::Empty => {
                        dests.push(idx);
                        if jumping { break; }
                    }
                    Cell::WhiteMarker | Cell::BlackMarker => {
                        jumping = true;
                    }
                }
                c += dc;
                r += dr;
            }
        }
        dests
    }

    fn is_my_ring(&self, idx: usize, p: Player) -> bool {
        self.cells[idx] == self.my_ring(p)
    }

    /// Check if the 5 cells starting at `start` in direction `dir` all have `player`'s marker.
    fn row_still_valid(&self, start: usize, dir: usize, player: Player) -> bool {
        let marker = self.my_marker(player);
        let (c, r) = index_to_cell(start);
        let (dc, dr) = ROW_DIRS[dir];
        for k in 0i8..5 {
            let nc = c as i8 + dc * k;
            let nr = r as i8 + dr * k;
            if !is_valid_i8(nc, nr) {
                return false;
            }
            if self.cells[cell_index_i8(nc, nr)] != marker {
                return false;
            }
        }
        true
    }

    /// Scan the board for all 5-in-a-row starting positions for `player`.
    pub fn find_rows(&self, player: Player) -> Vec<(usize, usize)> {
        let mut rows = Vec::new();
        for dir_idx in 0..3 {
            for start in 0..BOARD_SIZE {
                if self.row_still_valid(start, dir_idx, player) {
                    rows.push((start, dir_idx));
                }
            }
        }
        rows
    }

    /// Legal moves for the current phase / player.
    pub fn legal_moves(&mut self) -> Vec<YinshMove> {
        if self.outcome != Outcome::Ongoing {
            return Vec::new();
        }
        match self.phase {
            Phase::Setup => {
                let mut moves = Vec::with_capacity(BOARD_SIZE);
                for i in 0..BOARD_SIZE {
                    if self.cells[i] == Cell::Empty {
                        moves.push(YinshMove::PlaceRing(i));
                    }
                }
                moves
            }
            Phase::Normal => {
                let me = self.next_player;
                let my_ring = self.my_ring(me);
                let mut moves = Vec::new();
                for i in 0..BOARD_SIZE {
                    if self.cells[i] == my_ring {
                        for to in self.ring_destinations(i) {
                            moves.push(YinshMove::MoveRing { from: i, to });
                        }
                    }
                }
                moves
            }
            Phase::RemoveRow => {
                let me = self.next_player;
                self.pending_rows.iter().cloned()
                    .filter(|&(s, d)| self.row_still_valid(s, d, me))
                    .map(|(start, dir)| YinshMove::RemoveRow { start, dir })
                    .collect()
            }
            Phase::RemoveRing => {
                let me = self.next_player;
                let my_ring = self.my_ring(me);
                let mut moves = Vec::new();
                for i in 0..BOARD_SIZE {
                    if self.cells[i] == my_ring {
                        moves.push(YinshMove::RemoveRing(i));
                    }
                }
                moves
            }
        }
    }

    pub fn apply_move(&mut self, mv: YinshMove) -> Result<(), String> {
        if self.outcome != Outcome::Ongoing {
            return Err("game is over".into());
        }
        match (self.phase, mv) {
            (Phase::Setup, YinshMove::PlaceRing(idx)) => self.apply_place_ring(idx),
            (Phase::Normal, YinshMove::MoveRing { from, to }) => self.apply_move_ring(from, to),
            (Phase::RemoveRow, YinshMove::RemoveRow { start, dir }) => self.apply_remove_row(start, dir),
            (Phase::RemoveRing, YinshMove::RemoveRing(idx)) => self.apply_remove_ring(idx),
            _ => Err(format!("move {:?} invalid in phase {:?}", mv, self.phase)),
        }
    }

    fn apply_place_ring(&mut self, idx: usize) -> Result<(), String> {
        if self.cells[idx] != Cell::Empty {
            return Err("cell not empty".into());
        }
        let me = self.next_player;
        self.cells[idx] = self.my_ring(me);
        match me {
            Player::Player1 => self.white_rings_placed += 1,
            Player::Player2 => self.black_rings_placed += 1,
        }
        if self.white_rings_placed + self.black_rings_placed >= 2 * RINGS_PER_PLAYER {
            self.phase = Phase::Normal;
        }
        self.next_player = me.opposite();
        Ok(())
    }

    fn apply_move_ring(&mut self, from: usize, to: usize) -> Result<(), String> {
        let me = self.next_player;
        if !self.is_my_ring(from, me) {
            return Err("from cell is not your ring".into());
        }
        if !self.ring_destinations(from).contains(&to) {
            return Err("illegal ring destination".into());
        }

        // Compute direction
        let (fc, fr) = index_to_cell(from);
        let (tc, tr) = index_to_cell(to);
        let dc = (tc as i8 - fc as i8).signum();
        let dr = (tr as i8 - fr as i8).signum();

        // Place marker at from
        self.cells[from] = self.my_marker(me);
        self.markers_in_pool -= 1;

        // Flip markers between from+dir and to (exclusive of to)
        let mut c = fc as i8 + dc;
        let mut r = fr as i8 + dr;
        while (c, r) != (tc as i8, tr as i8) {
            let idx = cell_index_i8(c, r);
            self.cells[idx] = flip_marker(self.cells[idx]);
            c += dc;
            r += dr;
        }

        // Place ring at to
        self.cells[to] = self.my_ring(me);

        // Scan for rows
        let opp = me.opposite();
        let cur_rows = self.find_rows(me);
        let opp_rows = self.find_rows(opp);

        if !cur_rows.is_empty() {
            self.original_mover = Some(me);
            self.phase = Phase::RemoveRow;
            self.pending_rows = cur_rows;
            self.deferred_rows = opp_rows.clone();
            self.deferred_player = if opp_rows.is_empty() { None } else { Some(opp) };
            // next_player stays as me (cur player removes their own rows first)
        } else if !opp_rows.is_empty() {
            self.original_mover = Some(me);
            self.phase = Phase::RemoveRow;
            self.pending_rows = opp_rows;
            self.deferred_rows = Vec::new();
            self.deferred_player = None;
            self.next_player = opp;
        } else {
            // No rows: flip turn, check exhaustion
            self.next_player = opp;
            if self.markers_in_pool == 0 {
                self.check_marker_exhaustion();
            }
        }

        Ok(())
    }

    fn apply_remove_row(&mut self, start: usize, dir: usize) -> Result<(), String> {
        let me = self.next_player;
        if !self.row_still_valid(start, dir, me) {
            return Err("row no longer valid".into());
        }

        // Remove the 5 markers
        let (sc, sr) = index_to_cell(start);
        let (dc, dr) = ROW_DIRS[dir];
        for k in 0i8..5 {
            let nc = sc as i8 + dc * k;
            let nr = sr as i8 + dr * k;
            let idx = cell_index_i8(nc, nr);
            self.cells[idx] = Cell::Empty;
        }
        self.markers_in_pool += 5;

        // This player owes a ring removal
        self.pending_rings.push(me);

        // Re-filter pending_rows (some may no longer be valid after marker removal)
        let marker = self.my_marker(me);
        let cells_snapshot = self.cells;
        self.pending_rows.retain(|&(s, d)| {
            let (c, r) = index_to_cell(s);
            let (ddc, ddr) = ROW_DIRS[d];
            (0i8..5).all(|k| {
                let nc = c as i8 + ddc * k;
                let nr = r as i8 + ddr * k;
                is_valid_i8(nc, nr) && cells_snapshot[cell_index_i8(nc, nr)] == marker
            })
        });

        if self.pending_rows.is_empty() {
            // Move to RemoveRing phase
            self.phase = Phase::RemoveRing;
            self.next_player = self.pending_rings[0];
        }
        // Else stay in RemoveRow with same player
        Ok(())
    }

    fn apply_remove_ring(&mut self, idx: usize) -> Result<(), String> {
        let me = self.next_player;
        if !self.is_my_ring(idx, me) {
            return Err("not your ring".into());
        }
        self.cells[idx] = Cell::Empty;
        match me {
            Player::Player1 => self.white_score += 1,
            Player::Player2 => self.black_score += 1,
        }

        // Check win before transitioning
        let score = match me {
            Player::Player1 => self.white_score,
            Player::Player2 => self.black_score,
        };
        if score >= WIN_SCORE {
            self.outcome = Outcome::WonBy(me);
            return Ok(());
        }

        // Pop this player's pending ring removal
        self.pending_rings.remove(0);

        if !self.pending_rings.is_empty() {
            // Next ring removal
            self.next_player = self.pending_rings[0];
        } else if !self.deferred_rows.is_empty() {
            // Load opponent's deferred rows
            let opp = self.deferred_player.expect("deferred_rows without deferred_player");
            self.pending_rows = std::mem::take(&mut self.deferred_rows);
            self.deferred_player = None;
            self.phase = Phase::RemoveRow;
            self.next_player = opp;
        } else {
            // All removals done: back to Normal, opponent of original mover's turn
            let mover = self.original_mover.expect("no original_mover during removal");
            self.phase = Phase::Normal;
            self.next_player = mover.opposite();
            self.original_mover = None;
            if self.markers_in_pool == 0 {
                self.check_marker_exhaustion();
            }
        }
        Ok(())
    }

    fn check_marker_exhaustion(&mut self) {
        if self.white_score > self.black_score {
            self.outcome = Outcome::WonBy(Player::Player1);
        } else if self.black_score > self.white_score {
            self.outcome = Outcome::WonBy(Player::Player2);
        } else {
            self.outcome = Outcome::Draw;
        }
    }
}

fn flip_marker(cell: Cell) -> Cell {
    match cell {
        Cell::WhiteMarker => Cell::BlackMarker,
        Cell::BlackMarker => Cell::WhiteMarker,
        other => other,
    }
}

impl Game for YinshBoard {
    type Move = YinshMove;
    type Symmetry = UnitSymmetry;

    fn next_player(&self) -> Player { self.next_player }
    fn outcome(&self) -> Outcome { self.outcome }
    fn valid_moves(&mut self) -> Vec<YinshMove> { self.legal_moves() }
    fn play_move(&mut self, mv: &YinshMove) -> Result<(), String> { self.apply_move(*mv) }
    fn pass_move() -> YinshMove { YinshMove::Pass }
    fn is_pass(mv: &YinshMove) -> bool { matches!(mv, YinshMove::Pass) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let mut b = YinshBoard::new();
        assert_eq!(b.phase, Phase::Setup);
        assert_eq!(b.next_player, Player::Player1);
        assert_eq!(b.outcome, Outcome::Ongoing);
        let moves = b.legal_moves();
        assert_eq!(moves.len(), BOARD_SIZE);
        for mv in moves {
            assert!(matches!(mv, YinshMove::PlaceRing(_)));
        }
    }

    #[test]
    fn test_setup_completion() {
        let mut b = YinshBoard::new();
        // Place 10 rings
        for i in 0..10 {
            let mv = b.legal_moves()[0];
            b.apply_move(mv).unwrap();
            // After i+1 placements, phase should be Setup unless i+1 == 10
            if i + 1 < 10 {
                assert_eq!(b.phase, Phase::Setup);
            } else {
                assert_eq!(b.phase, Phase::Normal);
            }
        }
        assert_eq!(b.white_rings_placed, 5);
        assert_eq!(b.black_rings_placed, 5);
    }

    #[test]
    fn test_ring_destinations_empty_board() {
        let mut b = YinshBoard::new();
        // Place a white ring at E5 (center-ish)
        let idx = crate::hex::cell_index(4, 4);
        b.apply_move(YinshMove::PlaceRing(idx)).unwrap();
        // Place remaining 9 rings elsewhere
        for _ in 0..9 {
            let mv = b.legal_moves().into_iter().find(|mv| {
                if let YinshMove::PlaceRing(i) = mv {
                    *i != idx
                } else { false }
            }).unwrap();
            b.apply_move(mv).unwrap();
        }
        // In Normal phase now; E5 should have many ring destinations if white moves
        // (after setup, white moves first again)
        let dests = b.ring_destinations(idx);
        // All 6 directions yield some empty cells — expect more than 6 total empty destinations
        assert!(!dests.is_empty());
    }
}
