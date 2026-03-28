use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::hash::{DefaultHasher, Hash, Hasher};

use core_game::game::{Game, Outcome, Player};

use crate::hex::{
    self, hex_add, hex_neighbors, hex_to_index, index_to_hex, is_valid, Hex, DIRECTIONS,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Standard Zertz uses a hex board with 37 rings arranged in rows of
/// 4-5-6-7-6-5-4.
pub const BOARD_SIZE: usize = hex::BOARD_SIZE;

/// Maximum number of jumps in a single capture chain.
/// On a 37-cell board the theoretical max is higher, but 8 covers all
/// observed games (some have 5-6 hops).
pub const MAX_CAPTURE_JUMPS: usize = 8;

/// Starting marble supply: 6 white, 8 grey, 10 black.
const INITIAL_SUPPLY: [u8; 3] = [6, 8, 10];

/// Win thresholds per color: 4 white, 5 grey, or 6 black wins.
const WIN_SINGLE: [u8; 3] = [4, 5, 6];

/// 3 of each color also wins.
const WIN_EACH: u8 = 3;

// ---------------------------------------------------------------------------
// Hex-based helper functions
// ---------------------------------------------------------------------------

/// Find the intermediate cell between `from` and `to` for a 2-step hex hop.
/// The hop must be exactly 2 steps in one of the 6 hex directions.
pub fn find_intermediate(rings: &[Ring; BOARD_SIZE], from: Hex, to: Hex) -> Option<Hex> {
    for &dir in &DIRECTIONS {
        let mid = hex_add(from, dir);
        if !is_valid(mid) || rings[hex_to_index(mid)] == Ring::Removed {
            continue;
        }
        let end = hex_add(mid, dir);
        if end == to && is_valid(end) && rings[hex_to_index(end)] != Ring::Removed {
            return Some(mid);
        }
    }
    None
}

/// Find a multi-hop capture path from `from` to `to` on the given board.
///
/// Used for old-format boardspace games where a multi-step capture is recorded
/// as a single BtoB(start, end). Returns a list of (from, over, to) triples,
/// or None if no valid path exists.
pub fn find_capture_path(
    rings: &[Ring; BOARD_SIZE],
    from: Hex,
    to: Hex,
) -> Option<Vec<(Hex, Hex, Hex)>> {
    let mut path = Vec::new();
    let mut jumped_over = HashSet::new();
    if dfs_capture(rings, from, to, &mut jumped_over, &mut path) {
        Some(path)
    } else {
        None
    }
}

fn dfs_capture(
    rings: &[Ring; BOARD_SIZE],
    current: Hex,
    target: Hex,
    jumped_over: &mut HashSet<Hex>,
    path: &mut Vec<(Hex, Hex, Hex)>,
) -> bool {
    if current == target && !path.is_empty() {
        return true;
    }
    for &dir in &DIRECTIONS {
        let mid = hex_add(current, dir);
        if !is_valid(mid) {
            continue;
        }
        // Must jump over an occupied cell that hasn't been jumped yet.
        match rings[hex_to_index(mid)] {
            Ring::Occupied(_) if !jumped_over.contains(&mid) => {}
            _ => continue,
        }
        let end = hex_add(mid, dir);
        if !is_valid(end) {
            continue;
        }
        let end_ring = rings[hex_to_index(end)];
        if end_ring == Ring::Removed {
            continue;
        }
        // Landing cell must be empty, OR be the target if we're completing the path.
        if end_ring != Ring::Empty && end != target {
            continue;
        }
        // For the landing cell: if it's occupied and not the start, skip.
        if matches!(end_ring, Ring::Occupied(_)) && end != current {
            continue;
        }

        jumped_over.insert(mid);
        path.push((current, mid, end));
        if dfs_capture(rings, end, target, jumped_over, path) {
            return true;
        }
        path.pop();
        jumped_over.remove(&mid);
    }
    false
}

// ---------------------------------------------------------------------------
// Marble color
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
        place_at: Hex,
        remove: Hex,
    },
    /// Place marble when no free ring exists to remove.
    PlaceOnly {
        color: Marble,
        place_at: Hex,
    },
    /// No-op pass move (Zertz never generates this, but the Game trait requires it).
    Pass,
    /// A capture sequence encoded as (from, over, to) triples, up to MAX_CAPTURE_JUMPS.
    Capture {
        jumps: [(Hex, Hex, Hex); MAX_CAPTURE_JUMPS],
        len: u8,
    },
}

impl ZertzMove {
    pub fn capture_single(from: Hex, over: Hex, to: Hex) -> Self {
        ZertzMove::Capture {
            jumps: {
                let mut j = [((0i8, 0i8), (0i8, 0i8), (0i8, 0i8)); MAX_CAPTURE_JUMPS];
                j[0] = (from, over, to);
                j
            },
            len: 1,
        }
    }

    fn with_extra_jump(self, from: Hex, over: Hex, to: Hex) -> Option<Self> {
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
            } => write!(
                f,
                "Place({color} @({},{}), rm ({},{}))",
                place_at.0, place_at.1, remove.0, remove.1
            ),
            ZertzMove::PlaceOnly { color, place_at } => {
                write!(f, "Place({color} @({},{}))", place_at.0, place_at.1)
            }
            ZertzMove::Capture { jumps, len } => {
                write!(f, "Capture(")?;
                for i in 0..*len as usize {
                    if i > 0 {
                        write!(f, " -> ")?;
                    }
                    let (from, over, to) = jumps[i];
                    write!(
                        f,
                        "({},{})x({},{})->({},{})",
                        from.0, from.1, over.0, over.1, to.0, to.1
                    )?;
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
// Mid-capture state (sequential capture MCTS)
// ---------------------------------------------------------------------------

/// Tracks an in-progress capture chain. When set, only continuation hops
/// from `marble_pos` are legal — the same player keeps moving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MidCaptureState {
    pub marble_pos: Hex,
}

// ---------------------------------------------------------------------------
// ZertzBoard
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ZertzBoard {
    /// Fixed-size array indexed by hex_to_index(h). Ring::Removed means the
    /// ring has been taken off the board. Copy-friendly — no heap allocation.
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
    /// Mid-capture state: set when a capture chain is in progress.
    /// The same player continues moving until no more hops are available.
    mid_capture: Option<MidCaptureState>,
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
            mid_capture: None,
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

    /// Generate all single-hop capture moves from any occupied position.
    /// Used when NOT in mid_capture (first hop of a potential chain).
    fn generate_first_hops(&self) -> Vec<ZertzMove> {
        let mut hops = Vec::new();
        for (i, &ring) in self.rings.iter().enumerate() {
            if !matches!(ring, Ring::Occupied(_)) {
                continue;
            }
            let from = index_to_hex(i);
            Self::collect_single_hops(&self.rings, from, &mut hops);
        }
        hops
    }

    /// Generate single-hop continuation moves from a specific marble position.
    /// Used when mid_capture is active.
    fn generate_continuation_hops(&self, from: Hex) -> Vec<ZertzMove> {
        let mut hops = Vec::new();
        Self::collect_single_hops(&self.rings, from, &mut hops);
        hops
    }

    /// Collect all valid single-hop captures from `from` into `results`.
    fn collect_single_hops(rings: &[Ring; BOARD_SIZE], from: Hex, results: &mut Vec<ZertzMove>) {
        for &dir in &DIRECTIONS {
            let over = hex_add(from, dir);
            if !is_valid(over) {
                continue;
            }
            if !matches!(rings[hex_to_index(over)], Ring::Occupied(_)) {
                continue;
            }
            let to = hex_add(over, dir);
            if !is_valid(to) {
                continue;
            }
            if rings[hex_to_index(to)] != Ring::Empty {
                continue;
            }
            results.push(ZertzMove::capture_single(from, over, to));
        }
    }

    /// Generate all capture moves (full chains) from the current position.
    /// Used for replay/boardspace parsing — NOT used in MCTS (which uses
    /// sequential single-hop captures instead).
    fn generate_captures(&self) -> Vec<ZertzMove> {
        let mut all_captures = Vec::new();

        for (i, &ring) in self.rings.iter().enumerate() {
            if !matches!(ring, Ring::Occupied(_)) {
                continue;
            }
            let from = index_to_hex(i);
            let chain = ZertzMove::Capture {
                jumps: [((0, 0), (0, 0), (0, 0)); MAX_CAPTURE_JUMPS],
                len: 0,
            };
            // Stack copy — [Ring; 37] is Copy, no heap allocation.
            let mut temp_rings = self.rings;
            Self::find_capture_chains(from, &mut temp_rings, chain, &mut all_captures);
        }

        all_captures
    }

    fn find_capture_chains(
        from: Hex,
        rings: &mut [Ring; BOARD_SIZE],
        current_chain: ZertzMove,
        results: &mut Vec<ZertzMove>,
    ) {
        let mut found_jump = false;

        for &dir in &DIRECTIONS {
            let over = hex_add(from, dir);
            if !is_valid(over) {
                continue;
            }

            // Must jump over an occupied ring.
            let over_ring = rings[hex_to_index(over)];
            if !matches!(over_ring, Ring::Occupied(_)) {
                continue;
            }

            // Find landing spot: next cell in same direction from `over`.
            let to = hex_add(over, dir);
            if !is_valid(to) {
                continue;
            }

            // Landing spot must be empty.
            if rings[hex_to_index(to)] != Ring::Empty {
                continue;
            }

            found_jump = true;

            // Execute jump temporarily.
            let orig_from = rings[hex_to_index(from)];
            rings[hex_to_index(from)] = Ring::Empty;
            rings[hex_to_index(over)] = Ring::Empty;
            rings[hex_to_index(to)] = orig_from;

            let new_chain = if let ZertzMove::Capture { jumps: _, len } = current_chain {
                if len == 0 {
                    Some(ZertzMove::capture_single(from, over, to))
                } else {
                    current_chain.with_extra_jump(from, over, to)
                }
            } else {
                unreachable!()
            };

            // If chain is at max length, record it and stop recursing.
            let Some(new_chain) = new_chain else {
                results.push(current_chain);
                rings[hex_to_index(from)] = orig_from;
                rings[hex_to_index(over)] = over_ring;
                rings[hex_to_index(to)] = Ring::Empty;
                continue;
            };

            // Recurse for multi-jumps.
            Self::find_capture_chains(to, rings, new_chain, results);

            // Undo jump.
            rings[hex_to_index(from)] = orig_from;
            rings[hex_to_index(over)] = over_ring;
            rings[hex_to_index(to)] = Ring::Empty;
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
        let mut empty_positions = Vec::new();
        let mut removable_edges = Vec::new();
        for (i, &ring) in self.rings.iter().enumerate() {
            if ring == Ring::Empty {
                let pos = index_to_hex(i);
                empty_positions.push(pos);
                if Self::is_edge_static(&self.rings, pos) {
                    removable_edges.push(pos);
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
                        place_at: place,
                        remove,
                    });
                }

                // If no free ring exists to remove, placement still happens.
                if !found_removal {
                    moves.push(ZertzMove::PlaceOnly {
                        color,
                        place_at: place,
                    });
                }
            }
        }

        moves
    }

    /// A ring at `pos` is an edge if any of its 6 hex neighbors is absent
    /// (off the board or removed).
    fn is_edge_static(rings: &[Ring; BOARD_SIZE], pos: Hex) -> bool {
        for n in hex_neighbors(pos) {
            if !is_valid(n) || rings[hex_to_index(n)] == Ring::Removed {
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
            let to_capture: Vec<(usize, Marble)> = self
                .rings
                .iter()
                .enumerate()
                .filter_map(|(i, &r)| {
                    if let Ring::Occupied(m) = r {
                        Some((i, m))
                    } else {
                        None
                    }
                })
                .collect();
            for (i, m) in to_capture {
                self.captures[pi][m.index()] += 1;
                self.rings[i] = Ring::Removed;
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

    pub fn rings(&self) -> &[Ring; BOARD_SIZE] {
        &self.rings
    }

    /// Create a lightweight clone without history/stats — for use in MCTS tree nodes
    /// where repetition detection is not needed.
    /// Fast: [Ring; 37] is Copy — just a 37-byte stack copy, no heap allocation.
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
            mid_capture: self.mid_capture,
        }
    }

    /// Apply a move without checking legality. Use for trusted replay.
    pub fn play_unchecked(&mut self, mv: ZertzMove) -> Result<(), String> {
        self.apply_move(mv)
    }

    /// Apply a move for use in MCTS tree nodes: no history tracking, no
    /// repetition detection. Avoids the HashMap allocation that `play` incurs.
    pub fn play_mcts(&mut self, mv: ZertzMove) -> Result<(), String> {
        self.apply_move_no_history(mv)
    }

    fn apply_move_no_history(&mut self, mv: ZertzMove) -> Result<(), String> {
        match mv {
            ZertzMove::Place { color, place_at, remove } => {
                self.take_marble(color)?;
                self.rings[hex_to_index(place_at)] = Ring::Occupied(color);
                self.rings[hex_to_index(remove)] = Ring::Removed;
                self.resolve_isolation();
                self.check_winner();
                self.next_player = self.next_player.opposite();
            }
            ZertzMove::PlaceOnly { color, place_at } => {
                self.take_marble(color)?;
                self.rings[hex_to_index(place_at)] = Ring::Occupied(color);
                self.resolve_isolation();
                self.check_winner();
                self.next_player = self.next_player.opposite();
            }
            ZertzMove::Capture { jumps, len } => {
                if len == 1 {
                    // Single hop — used by MCTS sequential capture.
                    let (from, over, to) = jumps[0];
                    self.apply_single_hop(from, over, to)?;
                } else {
                    // Multi-hop — used by replay/boardspace parsing.
                    let pi = Self::player_index(self.next_player);
                    for i in 0..len as usize {
                        let (from, over, to) = jumps[i];
                        let marble_from = self.rings[hex_to_index(from)];
                        let captured = match self.rings[hex_to_index(over)] {
                            Ring::Occupied(m) => m,
                            _ => return Err(format!(
                                "no marble to jump over at position ({},{}) (hop {i})",
                                over.0, over.1
                            )),
                        };
                        self.rings[hex_to_index(from)] = Ring::Empty;
                        self.rings[hex_to_index(over)] = Ring::Empty;
                        self.rings[hex_to_index(to)] = marble_from;
                        self.captures[pi][captured.index()] += 1;
                        self.jump_captures[pi][captured.index()] += 1;
                    }
                    self.mid_capture = None;
                    self.check_winner();
                    self.next_player = self.next_player.opposite();
                }
            }
            ZertzMove::Pass => {
                self.check_winner();
                self.next_player = self.next_player.opposite();
            }
        }
        Ok(())
    }

    /// Execute a single capture hop and handle mid-capture state transitions.
    fn apply_single_hop(&mut self, from: Hex, over: Hex, to: Hex) -> Result<(), String> {
        let pi = Self::player_index(self.next_player);
        let marble_from = self.rings[hex_to_index(from)];
        let captured = match self.rings[hex_to_index(over)] {
            Ring::Occupied(m) => m,
            _ => return Err(format!(
                "no marble to jump over at position ({},{})",
                over.0, over.1
            )),
        };
        self.rings[hex_to_index(from)] = Ring::Empty;
        self.rings[hex_to_index(over)] = Ring::Empty;
        self.rings[hex_to_index(to)] = marble_from;
        self.captures[pi][captured.index()] += 1;
        self.jump_captures[pi][captured.index()] += 1;

        // Check if more hops are available from the landing position.
        let continuations = self.generate_continuation_hops(to);
        if continuations.is_empty() {
            // Chain complete — finalize turn.
            self.mid_capture = None;
            self.check_winner();
            self.next_player = self.next_player.opposite();
        } else {
            // More hops available — stay in mid-capture, same player's turn.
            self.mid_capture = Some(MidCaptureState { marble_pos: to });
        }
        Ok(())
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
                self.rings[hex_to_index(place_at)] = Ring::Occupied(color);
                self.rings[hex_to_index(remove)] = Ring::Removed;
                self.resolve_isolation();
                self.check_winner();
                self.next_player = self.next_player.opposite();
            }
            ZertzMove::PlaceOnly { color, place_at } => {
                self.take_marble(color)?;
                self.rings[hex_to_index(place_at)] = Ring::Occupied(color);
                self.resolve_isolation();
                self.check_winner();
                self.next_player = self.next_player.opposite();
            }
            ZertzMove::Capture { jumps, len } => {
                if len == 1 {
                    let (from, over, to) = jumps[0];
                    self.apply_single_hop(from, over, to)?;
                } else {
                    // Multi-hop — used by replay/boardspace parsing.
                    let pi = Self::player_index(self.next_player);
                    for i in 0..len as usize {
                        let (from, over, to) = jumps[i];
                        let marble_from = self.rings[hex_to_index(from)];
                        let captured = match self.rings[hex_to_index(over)] {
                            Ring::Occupied(m) => m,
                            _ => return Err(format!(
                                "no marble to jump over at position ({},{}) (hop {i})",
                                over.0, over.1
                            )),
                        };
                        self.rings[hex_to_index(from)] = Ring::Empty;
                        self.rings[hex_to_index(over)] = Ring::Empty;
                        self.rings[hex_to_index(to)] = marble_from;
                        self.captures[pi][captured.index()] += 1;
                        self.jump_captures[pi][captured.index()] += 1;
                    }
                    self.mid_capture = None;
                    self.check_winner();
                    self.next_player = self.next_player.opposite();
                }
            }
            ZertzMove::Pass => {
                self.check_winner();
                self.next_player = self.next_player.opposite();
            }
        }

        // History tracking: only record after a complete turn (not mid-capture).
        if self.mid_capture.is_none() && self.outcome == Outcome::Ongoing {
            let key = self.position_key();
            let count = self.history.entry(key).or_insert(0);
            *count += 1;
            if *count >= 2 {
                self.outcome = Outcome::Draw;
            }
        }
        Ok(())
    }

    /// Whether the board is in mid-capture state (a chain is in progress).
    pub fn is_mid_capture(&self) -> bool {
        self.mid_capture.is_some()
    }

    /// Get all legal moves for the current position.
    /// During mid-capture, only continuation hops from the active marble.
    pub fn legal_moves(&self) -> Vec<ZertzMove> {
        if self.outcome != Outcome::Ongoing {
            return Vec::new();
        }

        if let Some(mc) = self.mid_capture {
            return self.generate_continuation_hops(mc.marble_pos);
        }

        let hops = self.generate_first_hops();
        if !hops.is_empty() {
            // Captures are mandatory.
            return hops;
        }

        self.generate_placements()
    }

    /// Generate all full-chain captures. Used for replay/boardspace parsing only.
    pub fn legal_captures_full_chains(&self) -> Vec<ZertzMove> {
        if self.outcome != Outcome::Ongoing {
            return Vec::new();
        }
        self.generate_captures()
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

        for start_idx in 0..BOARD_SIZE {
            if self.rings[start_idx] == Ring::Removed || visited[start_idx] {
                continue;
            }
            let mut component = Vec::new();
            let mut queue = vec![start_idx];
            visited[start_idx] = true;
            while let Some(cur_idx) = queue.pop() {
                component.push(cur_idx);
                let cur_hex = index_to_hex(cur_idx);
                for n in hex_neighbors(cur_hex) {
                    if is_valid(n) {
                        let n_idx = hex_to_index(n);
                        if !visited[n_idx] && self.rings[n_idx] != Ring::Removed {
                            visited[n_idx] = true;
                            queue.push(n_idx);
                        }
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
                let has_vacant = comp.iter().any(|&i| self.rings[i] == Ring::Empty);
                if has_vacant {
                    // Isolated group still has empty rings — cannot claim (rule D.2).
                    continue;
                }
                for &i in comp {
                    if let Ring::Occupied(m) = self.rings[i] {
                        self.captures[pi][m.index()] += 1;
                        self.isolation_captures[pi][m.index()] += 1;
                    }
                    self.rings[i] = Ring::Removed;
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
        // ANSI color codes matching the board display.
        const CW: &str = "\x1b[38;2;255;160;50m";   // orange — White
        const CG: &str = "\x1b[38;2;100;180;255m"; // steel blue — Grey
        const CB: &str = "\x1b[38;2;255;60;180m";  // hot pink   — Black
        const CR: &str = "\x1b[0m";  // reset
        writeln!(
            f,
            "Supply:      W={CW}{}{CR} G={CG}{}{CR} B={CB}{}{CR}",
            self.supply[0], self.supply[1], self.supply[2]
        )?;
        writeln!(
            f,
            "Captures P1: W={CW}{}{CR} G={CG}{}{CR} B={CB}{}{CR}",
            self.captures[0][0], self.captures[0][1], self.captures[0][2]
        )?;
        writeln!(
            f,
            "Captures P2: W={CW}{}{CR} G={CG}{}{CR} B={CB}{}{CR}",
            self.captures[1][0], self.captures[1][1], self.captures[1][2]
        )?;

        // Staggered board display.
        //
        // Each cell (q, r) appears at display row  drow = 2r + q + 2*RADIUS
        // and visual column position              pos  = (q + RADIUS) * STEP + SHIFT
        //
        // SHIFT = STEP reserves one extra column-width on the left so that row
        // labels can sit at "outer-rim" positions (one step beyond the left
        // boundary of the board) and are always visible regardless of ring state.
        //
        // Row labels (7 at top, 1 at bottom):  outer-rim drow and pos come from
        //   q_outer = (leftmost q for that r) - 1
        //   drow    = 2r + q_outer + 2*RADIUS
        //   pos     = (q_outer + RADIUS) * STEP + SHIFT   (evaluates to 0..9)
        //
        // Column labels (A..G): outer-rim position one step below the last ring
        // of each column.  For columns A/G the outer row falls within the 13
        // board rows; for B/F one row below; C/D/E need one or two extra rows.
        const STEP: usize = 3;
        const SHIFT: usize = STEP;
        let rad = hex::RADIUS as i32; // 3
        let n_rows = (rad as usize) * 4 + 1; // 13

        // Visual position for a (possibly negative) col_idx.
        let vpos = |ci: i32| -> usize { ((ci + rad) * STEP as i32 + SHIFT as i32) as usize };

        // Gather all row-label items: (drow, pos, char).
        // drow = -1 is an extra row above the board (for label '7').
        let mut row_labels: Vec<(i32, usize, char)> = Vec::new();
        for r in -rad..=rad {
            let q_leftmost = (-rad).max(-rad - r);
            let q_outer    = q_leftmost - 1;
            let drow = 2 * r + q_outer + 2 * rad;
            let pos  = vpos(q_outer);
            let ch   = (b'0' + (4 - r) as u8) as char; // r=-3→'7', r=3→'1'
            row_labels.push((drow, pos, ch));
        }

        // Gather all column-label items: (drow, pos, char).
        let mut col_labels: Vec<(i32, usize, char)> = Vec::new();
        for q in -rad..=rad {
            let r_bottom = rad.min(rad - q);
            let r_outer  = r_bottom + 1;
            let drow = 2 * r_outer + q + 2 * rad;
            let pos  = vpos(q);
            let ch   = (b'A' + (q + rad) as u8) as char;
            col_labels.push((drow, pos, ch));
        }

        // Render one output line: place (pos, content) pairs left-to-right,
        // trimming trailing whitespace.
        let render = |items: &mut Vec<(usize, String)>| -> String {
            items.sort_by_key(|(p, _)| *p);
            // remove trailing " " (removed rings)
            while items.last().map_or(false, |(_, c)| c == " ") {
                items.pop();
            }
            let mut row = String::new();
            let mut cursor = 0usize;
            for (pos, cell) in items.iter() {
                while cursor < *pos { row.push(' '); cursor += 1; }
                row.push_str(cell);
                cursor += 1;
            }
            row
        };

        // Extra top row (row label '7' sits one drow above the board).
        if let Some((_, pos, ch)) = row_labels.iter().find(|(d, _, _)| *d == -1) {
            let mut items = vec![(*pos, format!("\x1b[90m{ch}\x1b[0m"))];
            writeln!(f, "{}", render(&mut items))?;
        }

        // Board rows drow = 0 .. n_rows-1.
        for drow in 0..n_rows {
            let mut items: Vec<(usize, String)> = Vec::new();

            // Row label for this drow (always shown).
            if let Some((_, pos, ch)) = row_labels.iter().find(|(d, _, _)| *d == drow as i32) {
                items.push((*pos, format!("\x1b[90m{ch}\x1b[0m")));
            }

            // Column labels whose outer-rim drow falls on this board row.
            for &(d, pos, ch) in &col_labels {
                if d == drow as i32 {
                    items.push((pos, format!("\x1b[90m{ch}\x1b[0m")));
                }
            }

            // Ring cells.
            for q in -rad..=rad {
                let two_r = drow as i32 - 2 * rad - q;
                if two_r % 2 != 0 { continue; }
                let r = two_r / 2;
                let h: Hex = (q as i8, r as i8);
                if !is_valid(h) { continue; }
                let pos = vpos(q);
                let cell = match self.rings[hex_to_index(h)] {
                    Ring::Occupied(Marble::White) => format!("{CW}W{CR}"),
                    Ring::Occupied(Marble::Grey)  => format!("{CG}G{CR}"),
                    Ring::Occupied(Marble::Black) => format!("{CB}B{CR}"),
                    Ring::Removed => " ".to_string(),
                    Ring::Empty   => "o".to_string(),
                };
                items.push((pos, cell));
            }

            writeln!(f, "{}", render(&mut items))?;
        }

        // Extra bottom rows for column labels C/E (drow=13) and D (drow=14).
        let mut extra_drows: Vec<i32> = col_labels.iter()
            .filter(|(d, _, _)| *d >= n_rows as i32)
            .map(|(d, _, _)| *d)
            .collect();
        extra_drows.sort();
        extra_drows.dedup();
        for ed in extra_drows {
            let mut items: Vec<(usize, String)> = col_labels.iter()
                .filter(|(d, _, _)| *d == ed)
                .map(|(_, pos, ch)| (*pos, format!("\x1b[90m{ch}\x1b[0m")))
                .collect();
            writeln!(f, "{}", render(&mut items))?;
        }

        Ok(())
    }
}
