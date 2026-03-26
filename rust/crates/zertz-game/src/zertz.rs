use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{DefaultHasher, Hash, Hasher};

use core_game::game::{Game, Outcome, Player};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Standard Zertz uses a hex board with 37 rings arranged in rows of
/// 4-5-6-7-6-5-4.  We index them 0..36.
pub const BOARD_SIZE: usize = 37;

/// Maximum number of jumps in a single capture chain.
/// On a 37-cell board the theoretical max is higher, but 8 covers all
/// observed games (some have 5-6 hops).
pub const MAX_CAPTURE_JUMPS: usize = 8;

/// Row lengths for the hex board (top to bottom).
pub const ROW_LENGTHS: [usize; 7] = [4, 5, 6, 7, 6, 5, 4];

/// Starting marble supply: 6 white, 8 grey, 10 black.
const INITIAL_SUPPLY: [u8; 3] = [6, 8, 10];

/// Win thresholds per color: 4 white, 5 grey, or 6 black wins.
const WIN_SINGLE: [u8; 3] = [4, 5, 6];

/// 3 of each color also wins.
const WIN_EACH: u8 = 3;

// ---------------------------------------------------------------------------
// Board layout (for coordinate mapping and future tournament support)
// ---------------------------------------------------------------------------

/// Column lengths for standard (37-ring) and tournament (48-ring) boards.
pub const STANDARD_COL_LENGTHS: &[usize] = &[4, 5, 6, 7, 6, 5, 4];
pub const TOURNAMENT_COL_LENGTHS: &[usize] = &[5, 6, 7, 8, 7, 6, 5, 4];

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BoardLayout {
    col_lengths: Vec<usize>,
    col_starts: Vec<usize>,
    board_size: usize,
}

impl BoardLayout {
    pub fn new(col_lengths: &[usize]) -> Self {
        let mut col_starts = Vec::with_capacity(col_lengths.len());
        let mut acc = 0;
        for &len in col_lengths {
            col_starts.push(acc);
            acc += len;
        }
        BoardLayout {
            col_lengths: col_lengths.to_vec(),
            col_starts,
            board_size: acc,
        }
    }

    pub fn standard() -> Self {
        Self::new(STANDARD_COL_LENGTHS)
    }

    pub fn tournament() -> Self {
        Self::new(TOURNAMENT_COL_LENGTHS)
    }

    pub fn board_size(&self) -> usize {
        self.board_size
    }

    pub fn num_cols(&self) -> usize {
        self.col_lengths.len()
    }

    pub fn col_lengths(&self) -> &[usize] {
        &self.col_lengths
    }

    /// Convert boardspace coordinate (col: A=0, B=1, ...; row: 0-based) to cell index.
    pub fn coord_to_index(&self, col: u8, row: u8) -> Option<usize> {
        let c = col as usize;
        let r = row as usize;
        if c >= self.col_lengths.len() || r >= self.col_lengths[c] {
            return None;
        }
        Some(self.col_starts[c] + r)
    }

    /// Convert cell index to (col, row) pair (both 0-based).
    pub fn index_to_coord(&self, idx: usize) -> Option<(u8, u8)> {
        if idx >= self.board_size {
            return None;
        }
        for (c, &start) in self.col_starts.iter().enumerate() {
            if idx < start + self.col_lengths[c] {
                return Some((c as u8, (idx - start) as u8));
            }
        }
        None
    }
}

/// Find the intermediate cell between `from` and `to` for a 2-step hex hop,
/// using a given neighbour table.
pub fn find_intermediate(
    neighbours: &[[u8; NUM_DIRECTIONS]],
    from: usize,
    to: usize,
) -> Option<usize> {
    for dir in 0..NUM_DIRECTIONS {
        let mid = neighbours[from][dir];
        if mid == 255 {
            continue;
        }
        let end = neighbours[mid as usize][dir];
        if end != 255 && end as usize == to {
            return Some(mid as usize);
        }
    }
    None
}

/// Get a reference to the standard (37-ring) neighbour table.
pub fn standard_neighbours() -> &'static [[u8; NUM_DIRECTIONS]; BOARD_SIZE] {
    &NEIGHBOURS
}

/// Find a multi-hop capture path from `from` to `to` on the given board.
///
/// Used for old-format boardspace games where a multi-step capture is recorded
/// as a single BtoB(start, end). Returns a list of (from, over, to) triples,
/// or None if no valid path exists.
pub fn find_capture_path(
    neighbours: &[[u8; NUM_DIRECTIONS]],
    rings: &[Ring],
    from: usize,
    to: usize,
) -> Option<Vec<(usize, usize, usize)>> {
    // DFS: find a sequence of jumps from `from` to `to`.
    let mut path = Vec::new();
    let mut jumped_over = vec![false; rings.len()];
    if dfs_capture(neighbours, rings, from, to, &mut jumped_over, &mut path) {
        Some(path)
    } else {
        None
    }
}

fn dfs_capture(
    neighbours: &[[u8; NUM_DIRECTIONS]],
    rings: &[Ring],
    current: usize,
    target: usize,
    jumped_over: &mut Vec<bool>,
    path: &mut Vec<(usize, usize, usize)>,
) -> bool {
    if current == target && !path.is_empty() {
        return true;
    }
    for dir in 0..NUM_DIRECTIONS {
        let mid = neighbours[current][dir];
        if mid == 255 {
            continue;
        }
        let mid = mid as usize;
        // Must jump over an occupied cell that hasn't been jumped yet.
        if jumped_over[mid] || !matches!(rings[mid], Ring::Occupied(_)) {
            continue;
        }
        let end = neighbours[mid][dir];
        if end == 255 {
            continue;
        }
        let end = end as usize;
        // Landing cell must be empty, OR be the target if we're completing the path,
        // OR be where we started (passing through).
        if rings[end] != Ring::Empty && end != target {
            continue;
        }
        // For the landing cell: if it's occupied and not the target, skip.
        if matches!(rings[end], Ring::Occupied(_)) && end != current {
            continue;
        }

        jumped_over[mid] = true;
        path.push((current, mid, end));
        if dfs_capture(neighbours, rings, end, target, jumped_over, path) {
            return true;
        }
        path.pop();
        jumped_over[mid] = false;
    }
    false
}

// ---------------------------------------------------------------------------
// Marble color
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Marble {
    White = 0,
    Grey = 1,
    Black = 2,
}

impl Marble {
    pub const ALL: [Marble; 3] = [Marble::White, Marble::Grey, Marble::Black];

    pub fn index(self) -> usize {
        self as usize
    }
}

impl Display for Marble {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Marble::White => write!(f, "W"),
            Marble::Grey => write!(f, "G"),
            Marble::Black => write!(f, "B"),
        }
    }
}

// ---------------------------------------------------------------------------
// Ring state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ring {
    /// Ring exists and is empty.
    Empty,
    /// Ring exists and has a marble on it.
    Occupied(Marble),
    /// Ring has been removed from the board.
    Removed,
}

// ---------------------------------------------------------------------------
// Move
// ---------------------------------------------------------------------------

/// A Zertz move is either:
/// - Place a marble on an empty ring, then remove an edge ring.
/// - Capture: jump over an adjacent marble to land on an empty ring (may chain).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZertzMove {
    /// Place marble of given color at `place_at`, then remove ring `remove`.
    Place {
        color: Marble,
        place_at: u8,
        remove: u8,
    },
    /// Place marble when no free ring exists to remove.
    PlaceOnly {
        color: Marble,
        place_at: u8,
    },
    /// No-op pass move (Zertz never generates this, but the Game trait requires it).
    Pass,
    /// A capture sequence encoded as start position + sequence of directions.
    /// We encode captures as (from, over, to) triples, up to 4 jumps max
    /// (practically sufficient for standard Zertz).
    Capture {
        jumps: [(u8, u8, u8); MAX_CAPTURE_JUMPS],
        len: u8,
    },
}

impl ZertzMove {
    pub fn capture_single(from: u8, over: u8, to: u8) -> Self {
        ZertzMove::Capture {
            jumps: {
                let mut j = [(0u8, 0u8, 0u8); MAX_CAPTURE_JUMPS];
                j[0] = (from, over, to);
                j
            },
            len: 1,
        }
    }

    fn with_extra_jump(self, from: u8, over: u8, to: u8) -> Option<Self> {
        match self {
            ZertzMove::Capture { mut jumps, len } => {
                if (len as usize) >= MAX_CAPTURE_JUMPS {
                    return None;
                }
                jumps[len as usize] = (from, over, to);
                Some(ZertzMove::Capture {
                    jumps,
                    len: len + 1,
                })
            }
            _ => panic!("with_extra_jump called on Place move"),
        }
    }
}

impl Debug for ZertzMove {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ZertzMove::Place {
                color,
                place_at,
                remove,
            } => write!(f, "Place({color} @{place_at}, rm {remove})"),
            ZertzMove::PlaceOnly { color, place_at } => {
                write!(f, "Place({color} @{place_at})")
            }
            ZertzMove::Capture { jumps, len } => {
                write!(f, "Capture(")?;
                for i in 0..*len as usize {
                    if i > 0 {
                        write!(f, " -> ")?;
                    }
                    let (from, over, to) = jumps[i];
                    write!(f, "{from}x{over}->{to}")?;
                }
                write!(f, ")")
            }
            ZertzMove::Pass => write!(f, "Pass"),
        }
    }
}

impl Display for ZertzMove {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

// ---------------------------------------------------------------------------
// Hex grid neighbour table
// ---------------------------------------------------------------------------

/// For each of the 37 positions, the six hex neighbours (or 255 if off-board).
/// Hex directions: E, W, NE, NW, SE, SW.
const NUM_DIRECTIONS: usize = 6;

/// Pre-computed neighbour table.  Built at startup via `build_neighbours`.
fn build_neighbours() -> [[u8; NUM_DIRECTIONS]; BOARD_SIZE] {
    // Map each cell to (row, col) and back.
    let mut row_of = [0u8; BOARD_SIZE];
    let mut col_of = [0u8; BOARD_SIZE];
    let mut idx = 0usize;
    let mut r = 0u8;
    while r < 7 {
        let mut c = 0u8;
        while c < ROW_LENGTHS[r as usize] as u8 {
            row_of[idx] = r;
            col_of[idx] = c;
            idx += 1;
            c += 1;
        }
        r += 1;
    }

    // For hex grids with offset rows, neighbours depend on whether we are
    // in the "expanding" part (rows 0..3) or "shrinking" part (rows 3..6).
    // We use axial-style adjacency.
    //
    // Row start indices:
    let row_start: [usize; 7] = [0, 4, 9, 15, 22, 28, 33];

    let mut neighbours = [[255u8; NUM_DIRECTIONS]; BOARD_SIZE];
    let mut i = 0usize;
    while i < BOARD_SIZE {
        let r = row_of[i] as usize;
        let c = col_of[i] as usize;
        let rlen = ROW_LENGTHS[r];

        // East / West (same row)
        if c + 1 < rlen {
            neighbours[i][0] = (row_start[r] + c + 1) as u8;
        }
        if c > 0 {
            neighbours[i][1] = (row_start[r] + c - 1) as u8;
        }

        // NE, NW (row - 1)
        if r > 0 {
            let prev_len = ROW_LENGTHS[r - 1];
            if r <= 3 {
                // current row is longer than previous
                // NW: same col - 1 in prev row, NE: same col in prev row
                if c > 0 && c - 1 < prev_len {
                    neighbours[i][3] = (row_start[r - 1] + c - 1) as u8; // NW
                }
                if c < prev_len {
                    neighbours[i][2] = (row_start[r - 1] + c) as u8; // NE
                }
            } else {
                // current row is shorter than or equal to previous
                // NW: same col in prev row, NE: col+1 in prev row
                if c < prev_len {
                    neighbours[i][3] = (row_start[r - 1] + c) as u8; // NW
                }
                if c + 1 < prev_len {
                    neighbours[i][2] = (row_start[r - 1] + c + 1) as u8; // NE
                }
            }
        }

        // SE, SW (row + 1)
        if r < 6 {
            let next_len = ROW_LENGTHS[r + 1];
            if r < 3 {
                // next row is longer than current
                // SW: same col in next row, SE: col+1 in next row
                if c < next_len {
                    neighbours[i][5] = (row_start[r + 1] + c) as u8; // SW
                }
                if c + 1 < next_len {
                    neighbours[i][4] = (row_start[r + 1] + c + 1) as u8; // SE
                }
            } else {
                // next row is shorter than or equal to current
                // SW: col-1 in next row, SE: same col in next row
                if c > 0 && c - 1 < next_len {
                    neighbours[i][5] = (row_start[r + 1] + c - 1) as u8; // SW
                }
                if c < next_len {
                    neighbours[i][4] = (row_start[r + 1] + c) as u8; // SE
                }
            }
        }

        i += 1;
    }

    neighbours
}

// We use a lazily-initialized static for the neighbour table.
use std::sync::LazyLock;

static NEIGHBOURS: LazyLock<[[u8; NUM_DIRECTIONS]; BOARD_SIZE]> =
    LazyLock::new(build_neighbours);

// ---------------------------------------------------------------------------
// ZertzBoard
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ZertzBoard {
    rings: [Ring; BOARD_SIZE],
    /// Marble supply shared by both players: [white, grey, black].
    supply: [u8; 3],
    /// Captured marbles per player: captures[player_index][color_index].
    captures: [[u8; 3]; 2],
    next_player: Player,
    outcome: Outcome,
    /// Position repetition counts keyed by position hash (excludes history itself).
    history: HashMap<u64, u8>,
    /// Whether the game ended via the F2 board-full rule.
    pub board_full: bool,
    /// Marbles gained via jump captures per player.
    pub jump_captures: [[u8; 3]; 2],
    /// Marbles gained via isolation per player.
    pub isolation_captures: [[u8; 3]; 2],
}

impl ZertzBoard {
    fn position_key(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.rings.hash(&mut h);
        self.supply.hash(&mut h);
        self.captures.hash(&mut h);
        self.next_player.hash(&mut h);
        h.finish()
    }
}

impl PartialEq for ZertzBoard {
    fn eq(&self, other: &Self) -> bool {
        self.rings == other.rings
            && self.supply == other.supply
            && self.captures == other.captures
            && self.next_player == other.next_player
            && self.outcome == other.outcome
    }
}

impl Eq for ZertzBoard {}

impl Hash for ZertzBoard {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.rings.hash(state);
        self.supply.hash(state);
        self.captures.hash(state);
        self.next_player.hash(state);
        self.outcome.hash(state);
    }
}

impl Default for ZertzBoard {
    fn default() -> Self {
        let mut board = ZertzBoard {
            rings: [Ring::Empty; BOARD_SIZE],
            supply: INITIAL_SUPPLY,
            captures: [[0; 3]; 2],
            next_player: Player::Player1,
            outcome: Outcome::Ongoing,
            history: HashMap::new(),
            jump_captures: [[0; 3]; 2],
            isolation_captures: [[0; 3]; 2],
            board_full: false,
        };
        let key = board.position_key();
        board.history.insert(key, 1);
        board
    }
}

impl ZertzBoard {
    fn player_index(p: Player) -> usize {
        match p {
            Player::Player1 => 0,
            Player::Player2 => 1,
        }
    }

    /// Generate all capture moves (jumps) from the current position.
    /// Captures are mandatory if any exist.
    fn generate_captures(&self) -> Vec<ZertzMove> {
        let mut all_captures = Vec::new();

        for from in 0..BOARD_SIZE {
            if let Ring::Occupied(_) = self.rings[from] {
                let chain = ZertzMove::Capture {
                    jumps: [(0, 0, 0); MAX_CAPTURE_JUMPS],
                    len: 0,
                };
                let mut temp_rings = self.rings;
                self.find_capture_chains(from, &mut temp_rings, chain, &mut all_captures);
            }
        }

        // Also allow capturing any marble (not just your own) — in Zertz,
        // you jump any marble over any adjacent marble.
        // The above loop already starts from any occupied ring.
        all_captures
    }

    fn find_capture_chains(
        &self,
        from: usize,
        rings: &mut [Ring; BOARD_SIZE],
        current_chain: ZertzMove,
        results: &mut Vec<ZertzMove>,
    ) {
        let mut found_jump = false;

        for dir in 0..NUM_DIRECTIONS {
            let over = NEIGHBOURS[from][dir];
            if over == 255 {
                continue;
            }
            let over = over as usize;

            // Must jump over an occupied ring.
            if let Ring::Occupied(_) = rings[over] {
                // Find landing spot: next cell in same direction from `over`.
                let to = NEIGHBOURS[over][dir];
                if to == 255 {
                    continue;
                }
                let to = to as usize;

                // Landing spot must be empty.
                if rings[to] != Ring::Empty {
                    continue;
                }

                found_jump = true;

                // Execute jump temporarily.
                let orig_from = rings[from];
                let orig_over = rings[over];
                rings[from] = Ring::Empty;
                rings[over] = Ring::Empty;
                rings[to] = orig_from;

                let new_chain = if let ZertzMove::Capture { jumps: _, len } = current_chain {
                    if len == 0 {
                        Some(ZertzMove::capture_single(from as u8, over as u8, to as u8))
                    } else {
                        current_chain.with_extra_jump(from as u8, over as u8, to as u8)
                    }
                } else {
                    unreachable!()
                };

                // If chain is at max length, record it and stop recursing.
                let Some(new_chain) = new_chain else {
                    results.push(current_chain);
                    rings[from] = orig_from;
                    rings[over] = orig_over;
                    rings[to] = Ring::Empty;
                    continue;
                };

                // Recurse for multi-jumps.
                self.find_capture_chains(to, rings, new_chain, results);

                // Undo jump.
                rings[from] = orig_from;
                rings[over] = orig_over;
                rings[to] = Ring::Empty;
            }
        }

        // If no further jumps, and we've made at least one jump, record the chain.
        if !found_jump {
            if let ZertzMove::Capture { len, .. } = current_chain {
                if len > 0 {
                    results.push(current_chain);
                }
            }
        }
    }

    /// Colors available for placement (supply + captured marbles of current player
    /// when supply is depleted).
    fn available_colors(&self) -> Vec<Marble> {
        let supply_empty = self.supply.iter().all(|&s| s == 0);
        let mut colors = Vec::new();
        let pi = Self::player_index(self.next_player);
        for &color in &Marble::ALL {
            let count = if supply_empty {
                self.captures[pi][color.index()]
            } else {
                self.supply[color.index()]
            };
            if count > 0 {
                colors.push(color);
            }
        }
        colors
    }

    /// Generate all placement moves.
    fn generate_placements(&self) -> Vec<ZertzMove> {
        let mut moves = Vec::new();

        let colors = self.available_colors();
        if colors.is_empty() {
            return moves;
        }

        // Precompute empty positions and removable edges once.
        // Placing a marble doesn't change which rings are removable edges:
        // is_edge_static only checks for Removed/off-board neighbors, not Occupied.
        let mut empty_positions = Vec::new();
        let mut removable_edges = Vec::new();
        for i in 0..BOARD_SIZE {
            if self.rings[i] == Ring::Empty {
                empty_positions.push(i);
                if Self::is_edge_static(&self.rings, i) {
                    removable_edges.push(i);
                }
            }
        }

        for &color in &colors {
            for &place in &empty_positions {
                let mut found_removal = false;
                for &remove in &removable_edges {
                    if remove == place {
                        continue; // can't remove where we just placed
                    }
                    found_removal = true;
                    moves.push(ZertzMove::Place {
                        color,
                        place_at: place as u8,
                        remove: remove as u8,
                    });
                }

                // If no free ring exists to remove, placement still happens.
                if !found_removal {
                    moves.push(ZertzMove::PlaceOnly {
                        color,
                        place_at: place as u8,
                    });
                }
            }
        }

        moves
    }

    fn is_edge_static(rings: &[Ring; BOARD_SIZE], i: usize) -> bool {
        if rings[i] == Ring::Removed {
            return false;
        }
        let neighbours = &NEIGHBOURS[i];
        for &n in neighbours.iter() {
            if n == 255 || rings[n as usize] == Ring::Removed {
                return true;
            }
        }
        false
    }

    /// Check win conditions and update outcome.
    fn check_winner(&mut self) {
        // F2: if all remaining rings are occupied (none empty), award them to the
        // current player as an isolated group before checking win conditions.
        let no_empty_rings = !self.rings.iter().any(|r| *r == Ring::Empty);
        let marbles_on_board = self.rings.iter().any(|r| matches!(r, Ring::Occupied(_)));
        if no_empty_rings && marbles_on_board {
            self.board_full = true;
            let pi = Self::player_index(self.next_player);
            for ring in &mut self.rings {
                if let Ring::Occupied(m) = *ring {
                    self.captures[pi][m.index()] += 1;
                    *ring = Ring::Removed;
                }
            }
        }

        // Check win conditions.
        for p in [Player::Player1, Player::Player2] {
            let pi = Self::player_index(p);
            let caps = &self.captures[pi];

            // 4 white, 5 grey, or 6 black
            for (i, &threshold) in WIN_SINGLE.iter().enumerate() {
                if caps[i] >= threshold {
                    self.outcome = Outcome::WonBy(p);
                    return;
                }
            }

            // 3 of each color
            if caps.iter().all(|&c| c >= WIN_EACH) {
                self.outcome = Outcome::WonBy(p);
                return;
            }
        }

        // If no marbles remain on the board and supply is empty, all marbles are
        // in captures — someone must already have won above.
        let supply_empty = self.supply.iter().all(|&s| s == 0);
        let marbles_on_board = self.rings.iter().any(|r| matches!(r, Ring::Occupied(_)));
        debug_assert!(
            marbles_on_board || !supply_empty,
            "all marbles distributed but no winner detected\n  supply={:?}\n  captures A={:?}\n  captures B={:?}",
            self.supply, self.captures[0], self.captures[1],
        );
    }

    /// Take a marble from supply, or from own captures if supply is empty.
    fn take_marble(&mut self, color: Marble) -> Result<(), String> {
        let ci = color.index();
        if self.supply[ci] > 0 {
            self.supply[ci] -= 1;
        } else {
            let pi = Self::player_index(self.next_player);
            if self.captures[pi][ci] == 0 {
                return Err(format!("no {color} marble available to place"));
            }
            self.captures[pi][ci] -= 1;
        }
        Ok(())
    }

    pub fn captures(&self) -> &[[u8; 3]; 2] {
        &self.captures
    }

    pub fn supply(&self) -> &[u8; 3] {
        &self.supply
    }

    pub fn rings(&self) -> &[Ring] {
        &self.rings
    }

    /// Create a lightweight clone without history/stats — for use in MCTS tree nodes
    /// where repetition detection is not needed.
    pub fn clone_light(&self) -> Self {
        ZertzBoard {
            rings: self.rings,
            supply: self.supply,
            captures: self.captures,
            next_player: self.next_player,
            outcome: self.outcome,
            history: HashMap::new(),
            board_full: false,
            jump_captures: [[0; 3]; 2],
            isolation_captures: [[0; 3]; 2],
        }
    }

    /// Apply a move without checking legality. Use for trusted replay.
    pub fn play_unchecked(&mut self, mv: ZertzMove) -> Result<(), String> {
        self.apply_move(mv)
    }

    /// Core move execution: applies the move, checks winner, updates history.
    fn apply_move(&mut self, mv: ZertzMove) -> Result<(), String> {
        match mv {
            ZertzMove::Place {
                color,
                place_at,
                remove,
            } => {
                self.take_marble(color)?;
                self.rings[place_at as usize] = Ring::Occupied(color);
                self.rings[remove as usize] = Ring::Removed;
                self.resolve_isolation();
            }
            ZertzMove::PlaceOnly { color, place_at } => {
                self.take_marble(color)?;
                self.rings[place_at as usize] = Ring::Occupied(color);
                self.resolve_isolation();
            }
            ZertzMove::Capture { jumps, len } => {
                let pi = Self::player_index(self.next_player);
                for i in 0..len as usize {
                    let (from, over, to) = jumps[i];
                    let marble_from = self.rings[from as usize];
                    let captured = match self.rings[over as usize] {
                        Ring::Occupied(m) => m,
                        _ => {
                            return Err(format!(
                                "no marble to jump over at position {over} (hop {i})"
                            ))
                        }
                    };
                    self.rings[from as usize] = Ring::Empty;
                    self.rings[over as usize] = Ring::Empty;
                    self.rings[to as usize] = marble_from;
                    self.captures[pi][captured.index()] += 1;
                    self.jump_captures[pi][captured.index()] += 1;
                }
            }
            ZertzMove::Pass => {}
        }

        self.check_winner();
        self.next_player = self.next_player.opposite();

        if self.outcome == Outcome::Ongoing {
            let key = self.position_key();
            let count = self.history.entry(key).or_insert(0);
            *count += 1;
            if *count >= 2 {
                self.outcome = Outcome::Draw;
            }
        }
        Ok(())
    }

    /// Get all legal moves for the current position.
    pub fn legal_moves(&self) -> Vec<ZertzMove> {
        if self.outcome != Outcome::Ongoing {
            return Vec::new();
        }

        let captures = self.generate_captures();
        if !captures.is_empty() {
            // Captures are mandatory.
            return captures;
        }

        self.generate_placements()
    }
}

// ---------------------------------------------------------------------------
// Game trait implementation
// ---------------------------------------------------------------------------

impl Game for ZertzBoard {
    type Move = ZertzMove;
    type Symmetry = core_game::symmetry::D6Symmetry;

    fn next_player(&self) -> Player {
        self.next_player
    }

    fn outcome(&self) -> Outcome {
        self.outcome
    }

    fn valid_moves(&mut self) -> Vec<ZertzMove> {
        self.legal_moves()
    }

    fn play_move(&mut self, mv: &ZertzMove) -> Result<(), String> {
        self.apply_move(*mv)
    }

    fn pass_move() -> ZertzMove {
        ZertzMove::Pass
    }

    fn is_pass(mv: &ZertzMove) -> bool {
        matches!(mv, ZertzMove::Pass)
    }
}

impl ZertzBoard {
    /// Play a move using the board-game style API (takes move by value).
    pub fn play(&mut self, mv: ZertzMove) -> Result<(), String> {
        self.apply_move(mv)
    }

    /// After a ring removal, check if any components became isolated and
    /// award those marbles to the current player.
    fn resolve_isolation(&mut self) {
        let mut visited = [false; BOARD_SIZE];
        let mut components: Vec<Vec<usize>> = Vec::new();

        for start in 0..BOARD_SIZE {
            if self.rings[start] == Ring::Removed || visited[start] {
                continue;
            }
            let mut component = Vec::new();
            let mut queue = vec![start];
            visited[start] = true;
            while let Some(pos) = queue.pop() {
                component.push(pos);
                for &n in NEIGHBOURS[pos].iter() {
                    if n != 255 && !visited[n as usize] && self.rings[n as usize] != Ring::Removed {
                        visited[n as usize] = true;
                        queue.push(n as usize);
                    }
                }
            }
            components.push(component);
        }

        if components.len() <= 1 {
            return;
        }

        // Largest component stays; marbles on smaller ones go to current player,
        // but ONLY if the isolated group has no vacant rings (rule D.2).
        let largest_size = components.iter().map(|c| c.len()).max().unwrap_or(0);
        let pi = Self::player_index(self.next_player);

        for comp in &components {
            if comp.len() < largest_size {
                let has_vacant = comp.iter().any(|&pos| self.rings[pos] == Ring::Empty);
                if has_vacant {
                    // Isolated group still has empty rings — cannot claim (rule D.2).
                    // Rings and marbles stay on the board.
                    continue;
                }
                for &pos in comp {
                    if let Ring::Occupied(m) = self.rings[pos] {
                        self.captures[pi][m.index()] += 1;
                        self.isolation_captures[pi][m.index()] += 1;
                    }
                    self.rings[pos] = Ring::Removed;
                }
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl Display for ZertzBoard {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Supply: W={} G={} B={}", self.supply[0], self.supply[1], self.supply[2])?;
        writeln!(
            f,
            "Player A captures: W={} G={} B={}",
            self.captures[0][0], self.captures[0][1], self.captures[0][2]
        )?;
        writeln!(
            f,
            "Player B captures: W={} G={} B={}",
            self.captures[1][0], self.captures[1][1], self.captures[1][2]
        )?;
        writeln!(f, "Next: {:?}", self.next_player)?;

        let mut idx = 0;
        for (row_idx, &len) in ROW_LENGTHS.iter().enumerate() {
            // Indent for hex layout.
            let indent = (3usize).saturating_sub(row_idx).max(row_idx.saturating_sub(3));
            for _ in 0..indent {
                write!(f, " ")?;
            }
            for col in 0..len {
                if col > 0 {
                    write!(f, " ")?;
                }
                match self.rings[idx] {
                    Ring::Empty => write!(f, ".")?,
                    Ring::Occupied(m) => write!(f, "{m}")?,
                    Ring::Removed => write!(f, " ")?,
                }
                idx += 1;
            }
            writeln!(f)?;
        }

        if self.outcome != Outcome::Ongoing {
            writeln!(f, "Outcome: {:?}", self.outcome)?;
        }

        Ok(())
    }
}
