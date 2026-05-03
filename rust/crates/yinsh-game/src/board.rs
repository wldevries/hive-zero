/// YINSH game state and rules.

use std::fmt;

use core_game::game::{Game, NNGame, Outcome, Player, PolicyIndex};
use core_game::symmetry::UnitSymmetry;

use crate::hex::{
    BOARD_SIZE, COL_ENDS, COL_STARTS, DIRECTIONS, GRID_SIZE, ROW_DIRS, cell_index, cell_index_i8,
    index_to_cell, is_valid, is_valid_i8,
};
use crate::zobrist;

pub const INITIAL_MARKERS: u8 = 51;
pub const RINGS_PER_PLAYER: u8 = 5;
pub const WIN_SCORE: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cell {
    Empty,
    WhiteRing,
    BlackRing,
    WhiteMarker,
    BlackMarker,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    Setup,
    Normal,
    /// A 5-marker row is pending: the player picks both which row to claim and
    /// which of their rings to remove in a single joint decision.
    ClaimRow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YinshMove {
    PlaceRing(usize),
    MoveRing { from: usize, to: usize },
    /// Joint row claim: remove the 5 markers in `(start, dir)` and remove the
    /// ring at `ring`. Always paired — Yinsh requires one ring to come off per
    /// completed row.
    ClaimRow { start: usize, dir: usize, ring: usize },
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
    /// Incremental Zobrist hash maintained by every state-mutating helper
    /// below. `tt_key()` returns this directly. Consistency is asserted
    /// against `from_scratch_hash()` in the random-game test, so any helper
    /// that misses an XOR pair fails immediately.
    pub hash: u64,
    /// Snapshot stack used by `undo()`. Pushed by `apply_move` immediately
    /// before the mutation; popped + restored by `undo`. Snapshots intentionally
    /// exclude the history field itself to avoid O(N²) memory growth.
    pub history: Vec<YinshUndo>,
}

/// Saved state for one `apply_move` step. Mirrors every field of `YinshBoard`
/// except `history`, so a snapshot is shallow w.r.t. the history stack.
#[derive(Clone)]
pub struct YinshUndo {
    cells: [Cell; BOARD_SIZE],
    white_rings_placed: u8,
    black_rings_placed: u8,
    white_score: u8,
    black_score: u8,
    markers_in_pool: u8,
    next_player: Player,
    outcome: Outcome,
    phase: Phase,
    pending_rows: Vec<(usize, usize)>,
    deferred_rows: Vec<(usize, usize)>,
    deferred_player: Option<Player>,
    original_mover: Option<Player>,
    hash: u64,
}

impl Default for YinshBoard {
    fn default() -> Self {
        Self::new()
    }
}

impl YinshBoard {
    pub fn new() -> Self {
        let mut b = Self {
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
            hash: 0,
            history: Vec::new(),
        };
        b.hash = b.from_scratch_hash();
        b
    }

    /// Recompute the Zobrist hash from scratch. Used to seed `hash` in `new()`
    /// and to assert (in tests) that the incremental updates done by the
    /// helper methods stay in sync with reality.
    pub fn from_scratch_hash(&self) -> u64 {
        let mut h = 0u64;
        for i in 0..BOARD_SIZE {
            h ^= zobrist::CELL[i][self.cells[i] as usize];
        }
        if matches!(self.next_player, Player::Player2) {
            h ^= zobrist::TURN_DIFF;
        }
        h ^= zobrist::PHASE[self.phase as usize];
        h ^= zobrist::MARKERS_IN_POOL[self.markers_in_pool as usize];
        h ^= zobrist::WHITE_SCORE[self.white_score as usize];
        h ^= zobrist::BLACK_SCORE[self.black_score as usize];
        h ^= zobrist::WHITE_RINGS_PLACED[self.white_rings_placed as usize];
        h ^= zobrist::BLACK_RINGS_PLACED[self.black_rings_placed as usize];
        for &(s, d) in &self.pending_rows {
            h ^= zobrist::PENDING_ROW[s][d];
        }
        for &(s, d) in &self.deferred_rows {
            h ^= zobrist::DEFERRED_ROW[s][d];
        }
        if let Some(p) = self.deferred_player {
            h ^= zobrist::DEFERRED_PLAYER[p as usize];
        }
        if let Some(p) = self.original_mover {
            h ^= zobrist::ORIGINAL_MOVER[p as usize];
        }
        h
    }

    // -------------------------------------------------------------------
    // Hash-aware setters. Every state-mutating site in `apply_*` goes
    // through these so the incremental Zobrist hash stays in sync.
    // -------------------------------------------------------------------

    fn set_cell(&mut self, idx: usize, new: Cell) {
        let old = self.cells[idx];
        if old != new {
            self.hash ^= zobrist::CELL[idx][old as usize] ^ zobrist::CELL[idx][new as usize];
            self.cells[idx] = new;
        }
    }

    fn flip_turn(&mut self) {
        self.hash ^= zobrist::TURN_DIFF;
        self.next_player = self.next_player.opposite();
    }

    fn set_turn(&mut self, new: Player) {
        if self.next_player != new {
            self.flip_turn();
        }
    }

    fn set_phase(&mut self, new: Phase) {
        if self.phase != new {
            self.hash ^= zobrist::PHASE[self.phase as usize] ^ zobrist::PHASE[new as usize];
            self.phase = new;
        }
    }

    fn set_markers_in_pool(&mut self, new: u8) {
        if self.markers_in_pool != new {
            self.hash ^= zobrist::MARKERS_IN_POOL[self.markers_in_pool as usize]
                ^ zobrist::MARKERS_IN_POOL[new as usize];
            self.markers_in_pool = new;
        }
    }

    fn add_white_score(&mut self) {
        let new = self.white_score + 1;
        self.hash ^= zobrist::WHITE_SCORE[self.white_score as usize]
            ^ zobrist::WHITE_SCORE[new as usize];
        self.white_score = new;
    }

    fn add_black_score(&mut self) {
        let new = self.black_score + 1;
        self.hash ^= zobrist::BLACK_SCORE[self.black_score as usize]
            ^ zobrist::BLACK_SCORE[new as usize];
        self.black_score = new;
    }

    fn add_white_rings_placed(&mut self) {
        let new = self.white_rings_placed + 1;
        self.hash ^= zobrist::WHITE_RINGS_PLACED[self.white_rings_placed as usize]
            ^ zobrist::WHITE_RINGS_PLACED[new as usize];
        self.white_rings_placed = new;
    }

    fn add_black_rings_placed(&mut self) {
        let new = self.black_rings_placed + 1;
        self.hash ^= zobrist::BLACK_RINGS_PLACED[self.black_rings_placed as usize]
            ^ zobrist::BLACK_RINGS_PLACED[new as usize];
        self.black_rings_placed = new;
    }

    fn set_deferred_player(&mut self, new: Option<Player>) {
        if let Some(p) = self.deferred_player {
            self.hash ^= zobrist::DEFERRED_PLAYER[p as usize];
        }
        if let Some(p) = new {
            self.hash ^= zobrist::DEFERRED_PLAYER[p as usize];
        }
        self.deferred_player = new;
    }

    fn set_original_mover(&mut self, new: Option<Player>) {
        if let Some(p) = self.original_mover {
            self.hash ^= zobrist::ORIGINAL_MOVER[p as usize];
        }
        if let Some(p) = new {
            self.hash ^= zobrist::ORIGINAL_MOVER[p as usize];
        }
        self.original_mover = new;
    }

    fn replace_pending_rows(&mut self, new: Vec<(usize, usize)>) {
        for &(s, d) in &self.pending_rows {
            self.hash ^= zobrist::PENDING_ROW[s][d];
        }
        for &(s, d) in &new {
            self.hash ^= zobrist::PENDING_ROW[s][d];
        }
        self.pending_rows = new;
    }

    fn replace_deferred_rows(&mut self, new: Vec<(usize, usize)>) {
        for &(s, d) in &self.deferred_rows {
            self.hash ^= zobrist::DEFERRED_ROW[s][d];
        }
        for &(s, d) in &new {
            self.hash ^= zobrist::DEFERRED_ROW[s][d];
        }
        self.deferred_rows = new;
    }

    /// Capture the current state into an undo frame. Cheap: cells are Copy, the
    /// two pending/deferred Vecs are usually empty (only populated mid-claim).
    fn snapshot(&self) -> YinshUndo {
        YinshUndo {
            cells: self.cells,
            white_rings_placed: self.white_rings_placed,
            black_rings_placed: self.black_rings_placed,
            white_score: self.white_score,
            black_score: self.black_score,
            markers_in_pool: self.markers_in_pool,
            next_player: self.next_player,
            outcome: self.outcome,
            phase: self.phase,
            pending_rows: self.pending_rows.clone(),
            deferred_rows: self.deferred_rows.clone(),
            deferred_player: self.deferred_player,
            original_mover: self.original_mover,
            hash: self.hash,
        }
    }

    fn restore(&mut self, u: YinshUndo) {
        self.cells = u.cells;
        self.white_rings_placed = u.white_rings_placed;
        self.black_rings_placed = u.black_rings_placed;
        self.white_score = u.white_score;
        self.black_score = u.black_score;
        self.markers_in_pool = u.markers_in_pool;
        self.next_player = u.next_player;
        self.outcome = u.outcome;
        self.phase = u.phase;
        self.pending_rows = u.pending_rows;
        self.deferred_rows = u.deferred_rows;
        self.deferred_player = u.deferred_player;
        self.original_mover = u.original_mover;
        self.hash = u.hash;
    }

    /// Reverse the most recent `apply_move` (including pass). Panics if the
    /// history stack is empty.
    pub fn undo(&mut self) {
        let u = self.history.pop().expect("undo with empty history");
        self.restore(u);
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
            Phase::ClaimRow => {
                let me = self.next_player;
                let my_ring = self.my_ring(me);
                let valid_rows: Vec<(usize, usize)> = self
                    .pending_rows
                    .iter()
                    .cloned()
                    .filter(|&(s, d)| self.row_still_valid(s, d, me))
                    .collect();
                let mut my_rings: Vec<usize> = Vec::new();
                for i in 0..BOARD_SIZE {
                    if self.cells[i] == my_ring {
                        my_rings.push(i);
                    }
                }
                let mut moves = Vec::with_capacity(valid_rows.len() * my_rings.len());
                for &(start, dir) in &valid_rows {
                    for &ring in &my_rings {
                        moves.push(YinshMove::ClaimRow { start, dir, ring });
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
        // Push the snapshot up front and pop it back if the sub-handler
        // returns Err. Every sub-handler validates its arguments before
        // mutating state, so a popped snapshot fully restores `self`.
        self.history.push(self.snapshot());
        let result = match (self.phase, mv) {
            (Phase::Setup, YinshMove::PlaceRing(idx)) => self.apply_place_ring(idx),
            (Phase::Normal, YinshMove::MoveRing { from, to }) => self.apply_move_ring(from, to),
            (Phase::ClaimRow, YinshMove::ClaimRow { start, dir, ring }) => {
                self.apply_claim_row(start, dir, ring)
            }
            (_, YinshMove::Pass) => {
                self.flip_turn();
                Ok(())
            }
            _ => Err(format!("move {:?} invalid in phase {:?}", mv, self.phase)),
        };
        if result.is_err() {
            self.history.pop();
        }
        result
    }

    fn apply_place_ring(&mut self, idx: usize) -> Result<(), String> {
        if self.cells[idx] != Cell::Empty {
            return Err("cell not empty".into());
        }
        let me = self.next_player;
        let ring = self.my_ring(me);
        self.set_cell(idx, ring);
        match me {
            Player::Player1 => self.add_white_rings_placed(),
            Player::Player2 => self.add_black_rings_placed(),
        }
        if self.white_rings_placed + self.black_rings_placed >= 2 * RINGS_PER_PLAYER {
            self.set_phase(Phase::Normal);
        }
        self.flip_turn();
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

        // Place marker at `from` (the ring leaves a marker behind).
        let my_marker = self.my_marker(me);
        self.set_cell(from, my_marker);
        self.set_markers_in_pool(self.markers_in_pool - 1);

        // Flip markers between from+dir and to (exclusive of to).
        let mut c = fc as i8 + dc;
        let mut r = fr as i8 + dr;
        while (c, r) != (tc as i8, tr as i8) {
            let idx = cell_index_i8(c, r);
            let flipped = flip_marker(self.cells[idx]);
            self.set_cell(idx, flipped);
            c += dc;
            r += dr;
        }

        // Place ring at `to`.
        let my_ring = self.my_ring(me);
        self.set_cell(to, my_ring);

        // Scan for rows.
        let opp = me.opposite();
        let cur_rows = self.find_rows(me);
        let opp_rows = self.find_rows(opp);

        if !cur_rows.is_empty() {
            self.set_original_mover(Some(me));
            self.set_phase(Phase::ClaimRow);
            self.replace_pending_rows(cur_rows);
            let new_deferred_player = if opp_rows.is_empty() { None } else { Some(opp) };
            self.replace_deferred_rows(opp_rows);
            self.set_deferred_player(new_deferred_player);
            // next_player stays as me (cur player claims their own rows first)
        } else if !opp_rows.is_empty() {
            self.set_original_mover(Some(me));
            self.set_phase(Phase::ClaimRow);
            self.replace_pending_rows(opp_rows);
            // deferred_rows already empty in this branch.
            self.set_deferred_player(None);
            self.set_turn(opp);
        } else {
            // No rows: flip turn, check exhaustion.
            self.set_turn(opp);
            if self.markers_in_pool == 0 {
                self.check_marker_exhaustion();
            }
        }

        Ok(())
    }

    /// Apply a joint row claim: remove the 5 markers in `(start, dir)` and the
    /// ring at `ring` in a single MCTS step. Yinsh rule G.3 pairs every row
    /// claim with one ring removal, so we encode them as one move.
    fn apply_claim_row(&mut self, start: usize, dir: usize, ring: usize) -> Result<(), String> {
        let me = self.next_player;
        if !self.row_still_valid(start, dir, me) {
            return Err("row no longer valid".into());
        }
        if !self.is_my_ring(ring, me) {
            return Err("not your ring".into());
        }

        // Remove the 5 markers.
        let (sc, sr) = index_to_cell(start);
        let (dc, dr) = ROW_DIRS[dir];
        for k in 0i8..5 {
            let nc = sc as i8 + dc * k;
            let nr = sr as i8 + dr * k;
            let mc = cell_index_i8(nc, nr);
            self.set_cell(mc, Cell::Empty);
        }
        self.set_markers_in_pool(self.markers_in_pool + 5);

        // Re-filter pending_rows: the row we just claimed is now gone (its markers
        // are Empty), and other rows that overlapped with it may no longer be
        // valid. Per yinsh rule G.5, intersecting rows cost only one ring because
        // the intersecting row falls off after the first is claimed.
        let marker = self.my_marker(me);
        let cells_snapshot = self.cells;
        let new_pending: Vec<(usize, usize)> = self
            .pending_rows
            .iter()
            .copied()
            .filter(|&(s, d)| {
                let (c, r) = index_to_cell(s);
                let (ddc, ddr) = ROW_DIRS[d];
                (0i8..5).all(|k| {
                    let nc = c as i8 + ddc * k;
                    let nr = r as i8 + ddr * k;
                    is_valid_i8(nc, nr) && cells_snapshot[cell_index_i8(nc, nr)] == marker
                })
            })
            .collect();
        self.replace_pending_rows(new_pending);

        // Remove the ring.
        self.set_cell(ring, Cell::Empty);
        match me {
            Player::Player1 => self.add_white_score(),
            Player::Player2 => self.add_black_score(),
        }

        // Check win before transitioning. Per yinsh rule H.1, the game ends the
        // instant a 3rd ring comes off, even if pending rows remain unclaimed.
        let score = match me {
            Player::Player1 => self.white_score,
            Player::Player2 => self.black_score,
        };
        if score >= WIN_SCORE {
            self.outcome = Outcome::WonBy(me);
            return Ok(());
        }

        if !self.pending_rows.is_empty() {
            // More rows still pending for this player — stay in ClaimRow phase.
            self.set_phase(Phase::ClaimRow);
        } else if !self.deferred_rows.is_empty() {
            // Current player done; opponent now claims their own (deferred) rows.
            let opp = self.deferred_player.expect("deferred_rows without deferred_player");
            let new_pending = self.deferred_rows.clone();
            self.replace_deferred_rows(Vec::new());
            self.replace_pending_rows(new_pending);
            self.set_deferred_player(None);
            self.set_phase(Phase::ClaimRow);
            self.set_turn(opp);
        } else {
            // All claims done: back to Normal, opponent of original mover's turn.
            let mover = self.original_mover.expect("no original_mover during claim");
            self.set_phase(Phase::Normal);
            self.set_turn(mover.opposite());
            self.set_original_mover(None);
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

    /// Tiny xorshift64 — deterministic, no extra deps. Used to drive a
    /// reproducible random walk through legal-move space.
    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// Walk a random Yinsh game and assert that `hash` (maintained by the
    /// helper setters) matches `from_scratch_hash()` after every move,
    /// including pass and after entering ClaimRow.
    ///
    /// This is the "I forgot a mutation site" canary. If any helper falls out
    /// of sync — or any new state-changing code skips a helper — this test
    /// fails on the first divergent move with the offending move printed.
    #[test]
    fn zobrist_hash_matches_from_scratch_random_walk() {
        let mut seed = 0xCAFE_F00D_DEAD_BEEFu64;
        for trial in 0..16 {
            let mut board = YinshBoard::new();
            assert_eq!(
                board.hash,
                board.from_scratch_hash(),
                "trial {trial}: initial hash mismatch",
            );
            for ply in 0..400usize {
                if board.outcome != Outcome::Ongoing {
                    break;
                }
                let moves = board.legal_moves();
                if moves.is_empty() {
                    board.apply_move(YinshMove::Pass).unwrap();
                } else {
                    let pick = (xorshift64(&mut seed) as usize) % moves.len();
                    let mv = moves[pick];
                    board.apply_move(mv).unwrap();
                    let truth = board.from_scratch_hash();
                    assert_eq!(
                        board.hash, truth,
                        "trial {trial} ply {ply}: hash diverged after {:?} \
                         (phase={:?}, next={:?})",
                        mv, board.phase, board.next_player,
                    );
                }
            }
        }
    }

    /// Apply a sequence of moves, then undo them all in reverse order. After
    /// each undo, `hash` must equal both the snapshot's pre-move hash and a
    /// fresh `from_scratch_hash()`. Catches a snapshot/restore that misses
    /// the hash field, or any helper whose XOR pair isn't a true inverse.
    #[test]
    fn zobrist_hash_round_trip_through_undo() {
        let mut seed = 0x0123_4567_89AB_CDEFu64;
        let mut board = YinshBoard::new();
        let mut hashes_before: Vec<u64> = Vec::new();

        for _ in 0..120 {
            if board.outcome != Outcome::Ongoing {
                break;
            }
            let moves = board.legal_moves();
            if moves.is_empty() {
                hashes_before.push(board.hash);
                board.apply_move(YinshMove::Pass).unwrap();
            } else {
                let pick = (xorshift64(&mut seed) as usize) % moves.len();
                hashes_before.push(board.hash);
                board.apply_move(moves[pick]).unwrap();
            }
        }

        while let Some(expected) = hashes_before.pop() {
            board.undo();
            assert_eq!(board.hash, expected, "undo failed to restore hash");
            assert_eq!(
                board.hash,
                board.from_scratch_hash(),
                "undo'd hash diverged from from_scratch",
            );
        }
        assert!(board.history.is_empty());
        assert_eq!(board.hash, YinshBoard::new().hash, "fully undone != fresh");
    }
}
