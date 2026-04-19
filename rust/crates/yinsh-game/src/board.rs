/// YINSH game state and rules.

use std::fmt;

use core_game::game::{Game, NNGame, Outcome, Player, PolicyIndex};
use core_game::symmetry::UnitSymmetry;

use crate::hex::{
    BOARD_SIZE, COL_ENDS, COL_STARTS, DIRECTIONS, GRID_SIZE, ROW_DIRS, cell_index, cell_index_i8,
    index_to_cell, is_valid, is_valid_i8,
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
            (_, YinshMove::Pass) => {
                self.next_player = self.next_player.opposite();
                Ok(())
            }
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

        // Re-filter pending_rows: the row we just claimed is now gone (its markers
        // are Empty), and other rows that overlapped with it may no longer be
        // valid. Per yinsh rule G.5, intersecting rows cost only one ring because
        // the intersecting row falls off after the first is claimed.
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

        // Each row claim is paired with a ring removal (yinsh rule G.3).
        self.phase = Phase::RemoveRing;
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
            // Per yinsh rule H.1, the game ends the instant a 3rd ring comes off,
            // even if pending rows remain unclaimed.
            self.outcome = Outcome::WonBy(me);
            return Ok(());
        }

        if !self.pending_rows.is_empty() {
            // More rows still pending for this player — claim the next one.
            self.phase = Phase::RemoveRow;
        } else if !self.deferred_rows.is_empty() {
            // Current player done; opponent now claims their own (deferred) rows.
            let opp = self.deferred_player.expect("deferred_rows without deferred_player");
            self.pending_rows = std::mem::take(&mut self.deferred_rows);
            self.deferred_player = None;
            self.phase = Phase::RemoveRow;
            self.next_player = opp;
        } else {
            // All removals done: back to Normal, opponent of original mover's turn.
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

impl fmt::Display for YinshBoard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const CW: &str = "\x1b[38;2;220;220;220m";
        const CB: &str = "\x1b[38;2;80;140;255m";
        const CR: &str = "\x1b[0m";
        const DIM: &str = "\x1b[90m";
        const STEP: usize = 3;
        const SHIFT: usize = 3;

        let outcome_str = match self.outcome {
            Outcome::Ongoing => {
                let next = match self.next_player {
                    Player::Player1 => "White",
                    Player::Player2 => "Black",
                };
                format!("next={next}")
            }
            Outcome::WonBy(Player::Player1) => "White wins".to_string(),
            Outcome::WonBy(Player::Player2) => "Black wins".to_string(),
            Outcome::Draw => "Draw".to_string(),
        };
        writeln!(
            f,
            "{CW}W{CR}={} {CB}B{CR}={} m={} {outcome_str}",
            self.white_score, self.black_score, self.markers_in_pool
        )?;

        // Staggered hex grid.
        // Display row: drow = 2*row - col + 5
        // Display pos:  pos  = col * STEP + SHIFT
        // Row 11 at top, row 1 at bottom. Column letters sit 2 drows below
        // each column's topmost cell (lowest row index). Iteration is reversed
        // so large drow values (high rows) print first (at top of screen).

        let cell_pos = |col: i32| -> usize { col as usize * STEP + SHIFT };

        let col_labels: Vec<(i32, usize, char)> = (0i32..11).map(|col| {
            let top_drow = 2 * COL_STARTS[col as usize] as i32 - col + 5;
            let extra = match col { 0 | 10 => -2, 5 => -1, _ => 0 };
            (top_drow - 2 + extra, cell_pos(col), (b'A' + col as u8) as char)
        }).collect();

        let row_labels: Vec<(i32, usize, String)> = (0i32..11).map(|r| {
            let leftmost = (0i32..11)
                .find(|&c| is_valid(c as u8, r as u8))
                .unwrap();
            // Rows 1 and 6 (r=0,5) shift up 2 and 3 left; all others shift up 1.
            let (drow_shift, pos_extra_left) = if r == 0 || r == 5 { (2, STEP) } else { (1, 0) };
            let drow = 2 * r - leftmost + 5 + drow_shift;
            let pos = cell_pos(leftmost) - STEP - pos_extra_left;
            (drow, pos, format!("{}", r + 1))
        }).collect();

        let min_drow = col_labels.iter().map(|(d, _, _)| *d).min().unwrap();
        let max_drow = row_labels.iter().map(|(d, _, _)| *d).max().unwrap();

        for drow in (min_drow..=max_drow).rev() {
            // (pos, rendered string, visible width)
            let mut items: Vec<(usize, String, usize)> = Vec::new();

            for &(d, pos, ch) in &col_labels {
                if d == drow {
                    items.push((pos, format!("{DIM}{ch}{CR}"), 1));
                }
            }

            for (d, pos, label) in &row_labels {
                if *d == drow {
                    let w = label.chars().count();
                    items.push((*pos, format!("{DIM}{label}{CR}"), w));
                }
            }

            for col in 0i32..11 {
                let num = drow - 5 + col;
                if num < 0 || num % 2 != 0 { continue; }
                let row = (num / 2) as u8;
                if row >= 11 { continue; }
                if !is_valid(col as u8, row) { continue; }
                let idx = cell_index(col as u8, row);
                let s = match self.cells[idx] {
                    Cell::Empty => "·".to_string(),
                    Cell::WhiteRing => format!("{CW}O{CR}"),
                    Cell::BlackRing => format!("{CB}O{CR}"),
                    Cell::WhiteMarker => format!("{CW}*{CR}"),
                    Cell::BlackMarker => format!("{CB}*{CR}"),
                };
                items.push((cell_pos(col), s, 1));
            }

            if !items.is_empty() {
                items.sort_by_key(|(p, _, _)| *p);
                let mut line = String::new();
                let mut cursor = 0usize;
                for (pos, cell, width) in &items {
                    while cursor < *pos { line.push(' '); cursor += 1; }
                    line.push_str(cell);
                    cursor += *width;
                }
                writeln!(f, "{line}")?;
            }
        }

        Ok(())
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

impl NNGame for YinshBoard {
    const BOARD_CHANNELS: usize = crate::board_encoding::NUM_CHANNELS;
    const RESERVE_SIZE: usize = crate::board_encoding::RESERVE_SIZE;
    const NUM_POLICY_CHANNELS: usize = crate::move_encoding::NUM_POLICY_CHANNELS;

    fn grid_size(&self) -> usize { GRID_SIZE }

    fn encode_board(&self, board_out: &mut [f32], reserve_out: &mut [f32]) {
        crate::board_encoding::encode_board(self, board_out, reserve_out);
    }

    fn get_legal_move_mask(&mut self) -> (Vec<f32>, Vec<(PolicyIndex, YinshMove)>) {
        crate::move_encoding::get_legal_move_mask(self)
    }
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
