use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::hash::{DefaultHasher, Hash, Hasher};

use core_game::game::{Game, Outcome, Player};

use crate::hex::{self, all_hexes, hex_add, hex_neighbors, Hex, DIRECTIONS};

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
// Hex-based helper functions (replacing old cell-index infrastructure)
// ---------------------------------------------------------------------------

/// Find the intermediate cell between `from` and `to` for a 2-step hex hop.
/// The hop must be exactly 2 steps in one of the 6 hex directions.
pub fn find_intermediate(
    rings: &BTreeMap<Hex, Ring>,
    from: Hex,
    to: Hex,
) -> Option<Hex> {
    for &dir in &DIRECTIONS {
        let mid = hex_add(from, dir);
        if !rings.contains_key(&mid) {
            continue;
        }
        let end = hex_add(mid, dir);
        if end == to && rings.contains_key(&end) {
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
    rings: &BTreeMap<Hex, Ring>,
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
    rings: &BTreeMap<Hex, Ring>,
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
        // Must jump over an occupied cell that hasn't been jumped yet.
        match rings.get(&mid) {
            Some(Ring::Occupied(_)) if !jumped_over.contains(&mid) => {}
            _ => continue,
        }
        let end = hex_add(mid, dir);
        // Landing cell must exist on the board.
        let end_ring = match rings.get(&end) {
            Some(r) => *r,
            None => continue,
        };
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
// ZertzBoard
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ZertzBoard {
    rings: BTreeMap<Hex, Ring>,
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
        // BTreeMap iterates in deterministic order, so this is stable.
        for (hex, ring) in &self.rings {
            hex.hash(&mut h);
            ring.hash(&mut h);
        }
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
        // BTreeMap iterates in deterministic order.
        for (hex, ring) in &self.rings {
            hex.hash(state);
            ring.hash(state);
        }
        self.supply.hash(state);
        self.captures.hash(state);
        self.next_player.hash(state);
        self.outcome.hash(state);
    }
}

impl Default for ZertzBoard {
    fn default() -> Self {
        let mut rings = BTreeMap::new();
        for h in all_hexes() {
            rings.insert(h, Ring::Empty);
        }
        let mut board = ZertzBoard {
            rings,
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

        let occupied: Vec<Hex> = self
            .rings
            .iter()
            .filter_map(|(&h, r)| {
                if matches!(r, Ring::Occupied(_)) {
                    Some(h)
                } else {
                    None
                }
            })
            .collect();

        for from in occupied {
            let chain = ZertzMove::Capture {
                jumps: [((0, 0), (0, 0), (0, 0)); MAX_CAPTURE_JUMPS],
                len: 0,
            };
            let mut temp_rings = self.rings.clone();
            self.find_capture_chains(from, &mut temp_rings, chain, &mut all_captures);
        }

        // Also allow capturing any marble (not just your own) — in Zertz,
        // you jump any marble over any adjacent marble.
        // The above loop already starts from any occupied ring.
        all_captures
    }

    fn find_capture_chains(
        &self,
        from: Hex,
        rings: &mut BTreeMap<Hex, Ring>,
        current_chain: ZertzMove,
        results: &mut Vec<ZertzMove>,
    ) {
        let mut found_jump = false;

        for &dir in &DIRECTIONS {
            let over = hex_add(from, dir);

            // Must jump over an occupied ring that is on the board.
            let over_ring = match rings.get(&over) {
                Some(Ring::Occupied(_)) => *rings.get(&over).unwrap(),
                _ => continue,
            };

            // Find landing spot: next cell in same direction from `over`.
            let to = hex_add(over, dir);

            // Landing spot must exist on the board.
            let to_ring = match rings.get(&to) {
                Some(r) => *r,
                None => continue,
            };

            // Landing spot must be empty.
            if to_ring != Ring::Empty {
                continue;
            }

            found_jump = true;

            // Execute jump temporarily.
            let orig_from = *rings.get(&from).unwrap();
            rings.insert(from, Ring::Empty);
            rings.insert(over, Ring::Empty);
            rings.insert(to, orig_from);

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
                rings.insert(from, orig_from);
                rings.insert(over, over_ring);
                rings.insert(to, Ring::Empty);
                continue;
            };

            // Recurse for multi-jumps.
            self.find_capture_chains(to, rings, new_chain, results);

            // Undo jump.
            rings.insert(from, orig_from);
            rings.insert(over, over_ring);
            rings.insert(to, Ring::Empty);
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
        // is_edge_static only checks for absent neighbors in the rings map.
        let mut empty_positions = Vec::new();
        let mut removable_edges = Vec::new();
        for (&pos, &ring) in &self.rings {
            if ring == Ring::Empty {
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

    /// A ring is an edge if any of its 6 hex neighbors is NOT in the rings map.
    fn is_edge_static(rings: &BTreeMap<Hex, Ring>, pos: Hex) -> bool {
        for n in hex_neighbors(pos) {
            if !rings.contains_key(&n) {
                return true;
            }
        }
        false
    }

    /// Check win conditions and update outcome.
    fn check_winner(&mut self) {
        // F2: if all remaining rings are occupied (none empty), award them to the
        // current player as an isolated group before checking win conditions.
        let no_empty_rings = !self.rings.values().any(|r| *r == Ring::Empty);
        let marbles_on_board = self.rings.values().any(|r| matches!(r, Ring::Occupied(_)));
        if no_empty_rings && marbles_on_board {
            self.board_full = true;
            let pi = Self::player_index(self.next_player);
            let to_capture: Vec<(Hex, Marble)> = self
                .rings
                .iter()
                .filter_map(|(&h, &r)| {
                    if let Ring::Occupied(m) = r {
                        Some((h, m))
                    } else {
                        None
                    }
                })
                .collect();
            for (h, m) in to_capture {
                self.captures[pi][m.index()] += 1;
                self.rings.remove(&h);
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
        let marbles_on_board = self.rings.values().any(|r| matches!(r, Ring::Occupied(_)));
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

    pub fn rings(&self) -> &BTreeMap<Hex, Ring> {
        &self.rings
    }

    /// Create a lightweight clone without history/stats — for use in MCTS tree nodes
    /// where repetition detection is not needed.
    pub fn clone_light(&self) -> Self {
        ZertzBoard {
            rings: self.rings.clone(),
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
                self.rings.insert(place_at, Ring::Occupied(color));
                self.rings.remove(&remove);
                self.resolve_isolation();
            }
            ZertzMove::PlaceOnly { color, place_at } => {
                self.take_marble(color)?;
                self.rings.insert(place_at, Ring::Occupied(color));
                self.resolve_isolation();
            }
            ZertzMove::Capture { jumps, len } => {
                let pi = Self::player_index(self.next_player);
                for i in 0..len as usize {
                    let (from, over, to) = jumps[i];
                    let marble_from = *self.rings.get(&from).unwrap_or(&Ring::Empty);
                    let captured = match self.rings.get(&over) {
                        Some(Ring::Occupied(m)) => *m,
                        _ => {
                            return Err(format!(
                                "no marble to jump over at position ({},{}) (hop {i})",
                                over.0, over.1
                            ))
                        }
                    };
                    self.rings.insert(from, Ring::Empty);
                    self.rings.insert(over, Ring::Empty);
                    self.rings.insert(to, marble_from);
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
        let mut visited: HashSet<Hex> = HashSet::new();
        let mut components: Vec<Vec<Hex>> = Vec::new();

        for &pos in self.rings.keys() {
            if visited.contains(&pos) {
                continue;
            }
            let mut component = Vec::new();
            let mut queue = vec![pos];
            visited.insert(pos);
            while let Some(cur) = queue.pop() {
                component.push(cur);
                for n in hex_neighbors(cur) {
                    if !visited.contains(&n) && self.rings.contains_key(&n) {
                        visited.insert(n);
                        queue.push(n);
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
                let has_vacant = comp
                    .iter()
                    .any(|pos| self.rings.get(pos) == Some(&Ring::Empty));
                if has_vacant {
                    // Isolated group still has empty rings — cannot claim (rule D.2).
                    // Rings and marbles stay on the board.
                    continue;
                }
                for &pos in comp {
                    if let Some(Ring::Occupied(m)) = self.rings.get(&pos) {
                        self.captures[pi][m.index()] += 1;
                        self.isolation_captures[pi][m.index()] += 1;
                    }
                    self.rings.remove(&pos);
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
        writeln!(
            f,
            "Supply: W={} G={} B={}",
            self.supply[0], self.supply[1], self.supply[2]
        )?;
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

        let radius = hex::RADIUS;
        for ri in 0..7 {
            let r = ri as i8 - radius;
            let q_min = (-radius).max(-radius - r);
            let q_max = radius.min(radius - r);
            // Indent for hex layout.
            let indent = (3usize).saturating_sub(ri).max(ri.saturating_sub(3));
            for _ in 0..indent {
                write!(f, " ")?;
            }
            let mut first = true;
            let mut q = q_min;
            while q <= q_max {
                if !first {
                    write!(f, " ")?;
                }
                first = false;
                let h: Hex = (q, r);
                match self.rings.get(&h) {
                    Some(Ring::Empty) => write!(f, ".")?,
                    Some(Ring::Occupied(m)) => write!(f, "{m}")?,
                    Some(Ring::Removed) | None => write!(f, " ")?,
                }
                q += 1;
            }
            writeln!(f)?;
        }

        if self.outcome != Outcome::Ongoing {
            writeln!(f, "Outcome: {:?}", self.outcome)?;
        }

        Ok(())
    }
}
