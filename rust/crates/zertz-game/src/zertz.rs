use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::hash::{DefaultHasher, Hash, Hasher};

use core_game::game::{Game, NNGame, Outcome, Player, PolicyIndex, Undoable};

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
// Win classification
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum WinType {
    FourWhite,
    FiveGrey,
    SixBlack,
    ThreeEach,
    Draw,
}

pub fn classify_win(board: &ZertzBoard, winner: Player) -> WinType {
    let pi = match winner {
        Player::Player1 => 0,
        Player::Player2 => 1,
    };
    let caps = &board.captures()[pi];
    if caps.iter().all(|&c| c >= 3) {
        WinType::ThreeEach
    } else if caps[0] >= 4 {
        WinType::FourWhite
    } else if caps[1] >= 5 {
        WinType::FiveGrey
    } else {
        WinType::SixBlack
    }
}

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

/// A Zertz move is one of:
/// - `Place`/`PlaceOnly`: joint placement (legacy, kept for SGF replay and alpha-beta).
/// - `PlaceMarbleHalf` + `RemoveRingHalf`: split placement into two same-player plies
///   so MCTS can search them independently. `apply_move` accepts the joint forms
///   too — they execute as the two halves chained atomically without ever leaving
///   the board in mid_placement state, so SGF replay and AB both keep working.
/// - `Capture`: jump over an adjacent marble to land on an empty ring (may chain).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZertzMove {
    /// Place marble of given color at `place_at`, then remove ring `remove`.
    /// Joint form: applies both halves atomically, never leaves mid_placement set.
    /// Used by SGF replay and alpha-beta.
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
    /// First half of a split placement: place marble and enter mid_placement state.
    /// Same player must follow up with `RemoveRingHalf`.
    PlaceMarbleHalf {
        color: Marble,
        place_at: Hex,
    },
    /// Second half of a split placement: remove a ring and end the turn.
    /// Only legal when board is in mid_placement state.
    RemoveRingHalf {
        remove: Hex,
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
            ZertzMove::PlaceMarbleHalf { color, place_at } => {
                write!(f, "PlaceHalf({color} @({},{}))", place_at.0, place_at.1)
            }
            ZertzMove::RemoveRingHalf { remove } => {
                write!(f, "RemoveHalf(({},{}))", remove.0, remove.1)
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
// Mid-placement state (sequential placement MCTS)
// ---------------------------------------------------------------------------

/// Tracks an in-progress placement that has placed a marble but not yet
/// removed a ring. When set, only `RemoveRingHalf` moves are legal — the same
/// player keeps moving. No payload is needed: the placed marble is already
/// on the board (just like any other Occupied ring), and the set of legal
/// ring removals is determined purely from current board topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MidPlacementState;

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
    /// Mid-placement state: set when a marble has been placed via
    /// `PlaceMarbleHalf` but the matching `RemoveRingHalf` has not yet been
    /// applied. The same player continues moving until the ring is removed.
    /// Always `None` after legacy joint `Place`/`PlaceOnly` apply because
    /// those execute both halves atomically.
    mid_placement: Option<MidPlacementState>,
    /// Undo stack: pushed by `apply_move` before mutation, popped + restored
    /// by `undo`. Empty for boards built via `clone_light` (MCTS hot path)
    /// since MCTS replays from root rather than undoing.
    pub apply_history: Vec<ZertzUndo>,
}

/// Snapshot of all `ZertzBoard` state needed to revert one `apply_move` step.
/// `repetition_key` records the position hash that was inserted into the
/// repetition `history` HashMap during this move (or `None` if this move did
/// not bump the repetition counter — e.g. mid-capture continuations).
#[derive(Clone, Debug)]
pub struct ZertzUndo {
    rings: [Ring; BOARD_SIZE],
    supply: [u8; 3],
    captures: [[u8; 3]; 2],
    next_player: Player,
    outcome: Outcome,
    board_full: bool,
    jump_captures: [[u8; 3]; 2],
    isolation_captures: [[u8; 3]; 2],
    mid_capture: Option<MidCaptureState>,
    mid_placement: Option<MidPlacementState>,
    repetition_key: Option<u64>,
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

    /// Hash key for transposition-table use. Like `position_key` plus the
    /// fields that distinguish tactically-equivalent states from
    /// search's perspective: `mid_capture` (legal moves differ) and
    /// `outcome` (terminal states must not collide with ongoing ones).
    /// Repetition history is intentionally excluded — alphabeta does not
    /// observe the path through the search subtree, so two visits with the
    /// same board produce the same score regardless of history.
    pub fn tt_key(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.rings.hash(&mut h);
        self.supply.hash(&mut h);
        self.captures.hash(&mut h);
        self.next_player.hash(&mut h);
        self.mid_capture.hash(&mut h);
        self.mid_placement.hash(&mut h);
        self.outcome.hash(&mut h);
        h.finish()
    }
}

impl Undoable for ZertzBoard {
    fn undo(&mut self) {
        ZertzBoard::undo(self);
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
            mid_placement: None,
            apply_history: Vec::new(),
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

    /// Generate joint placement moves (`Place` + `PlaceOnly`) — the legacy
    /// move space. Used by alpha-beta and any consumer that wants placement
    /// decided in a single ply. MCTS uses the split form via
    /// `generate_split_placements` / `generate_ring_removals` instead.
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

    /// Generate split-form placement first halves: one `PlaceMarbleHalf` per
    /// (color, empty-position) pair, plus `PlaceOnly` for positions whose
    /// placement leaves no removable ring (terminal one-ply placement). Used
    /// by MCTS — the matching `RemoveRingHalf` decision is searched as a
    /// child node once we're in mid_placement state.
    fn generate_split_placements(&self) -> Vec<ZertzMove> {
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
                // After placing at `place`, would any removable edge other
                // than `place` itself still exist? If yes, split into two
                // halves; otherwise the placement is terminal (`PlaceOnly`).
                let has_post_removal =
                    removable_edges.iter().any(|&r| r != place);
                if has_post_removal {
                    moves.push(ZertzMove::PlaceMarbleHalf { color, place_at: place });
                } else {
                    moves.push(ZertzMove::PlaceOnly { color, place_at: place });
                }
            }
        }

        moves
    }

    /// Generate `RemoveRingHalf` moves: one per removable empty ring on the
    /// current board. Used during mid_placement: the marble has already been
    /// placed (and is part of `self.rings`), so removability is computed
    /// directly off the current ring set.
    fn generate_ring_removals(&self) -> Vec<ZertzMove> {
        let mut moves = Vec::new();
        for (i, &ring) in self.rings.iter().enumerate() {
            if ring != Ring::Empty {
                continue;
            }
            let pos = index_to_hex(i);
            if Self::is_edge_static(&self.rings, pos) {
                moves.push(ZertzMove::RemoveRingHalf { remove: pos });
            }
        }
        moves
    }

    /// A ring at `pos` is an edge if it two neighbours that are adjacent to it are both missing
    /// a ring (either removed or off the board).
    fn is_edge_static(rings: &[Ring; BOARD_SIZE], pos: Hex) -> bool {
        let neighbors = hex_neighbors(pos);
        for i in 0..6 {
            let n1 = neighbors[i];
            let n2 = neighbors[(i + 1) % 6];
            let n1_missing = !is_valid(n1) || rings[hex_to_index(n1)] == Ring::Removed;
            let n2_missing = !is_valid(n2) || rings[hex_to_index(n2)] == Ring::Removed;
            if n1_missing && n2_missing {
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
            mid_placement: self.mid_placement,
            apply_history: Vec::new(),
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
            ZertzMove::PlaceMarbleHalf { color, place_at } => {
                // First half of split placement: marble lands, mid_placement
                // is set, same player must follow up with RemoveRingHalf.
                self.take_marble(color)?;
                self.rings[hex_to_index(place_at)] = Ring::Occupied(color);
                self.mid_placement = Some(MidPlacementState);
                // Do NOT swap player; do NOT resolve isolation or check_winner —
                // those happen after the ring is removed.
            }
            ZertzMove::RemoveRingHalf { remove } => {
                // Second half: ring removal finalises the turn.
                self.rings[hex_to_index(remove)] = Ring::Removed;
                self.mid_placement = None;
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
    /// Does not validate legality — callers that need legality checks must do
    /// so before calling (see `apply_move`).
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

    fn snapshot(&self) -> ZertzUndo {
        ZertzUndo {
            rings: self.rings,
            supply: self.supply,
            captures: self.captures,
            next_player: self.next_player,
            outcome: self.outcome,
            board_full: self.board_full,
            jump_captures: self.jump_captures,
            isolation_captures: self.isolation_captures,
            mid_capture: self.mid_capture,
            mid_placement: self.mid_placement,
            repetition_key: None,
        }
    }

    fn restore(&mut self, u: ZertzUndo) {
        self.rings = u.rings;
        self.supply = u.supply;
        self.captures = u.captures;
        self.next_player = u.next_player;
        self.outcome = u.outcome;
        self.board_full = u.board_full;
        self.jump_captures = u.jump_captures;
        self.isolation_captures = u.isolation_captures;
        self.mid_capture = u.mid_capture;
        self.mid_placement = u.mid_placement;
        if let Some(key) = u.repetition_key {
            if let Some(count) = self.history.get_mut(&key) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.history.remove(&key);
                }
            }
        }
    }

    /// Reverse the most recent `apply_move` (including pass and capture hops).
    /// Panics if the undo stack is empty — only `apply_move` pushes snapshots,
    /// so undo is balanced 1:1 against `apply_move` calls. `apply_move_no_history`
    /// (MCTS hot path) does not push snapshots and therefore cannot be undone.
    pub fn undo(&mut self) {
        let u = self
            .apply_history
            .pop()
            .expect("undo with empty apply_history");
        self.restore(u);
    }

    /// Core move execution: applies the move with legality checks, updates history,
    /// and pushes an undo snapshot so the move can be reversed via `undo()`.
    /// Use play_mcts / apply_move_no_history for the MCTS hot path (no checks,
    /// no snapshot — MCTS replays from the root rather than undoing).
    fn apply_move(&mut self, mv: ZertzMove) -> Result<(), String> {
        if self.outcome != Outcome::Ongoing {
            return Err("game is already over".to_string());
        }
        // Push the undo frame up front and pop it back if the move is rejected.
        // Each guard below validates *before* mutating, so the popped frame is
        // semantically equivalent to never having pushed in the first place.
        self.apply_history.push(self.snapshot());
        let result = self.apply_move_inner(mv);
        if result.is_err() {
            self.apply_history.pop();
        }
        result
    }

    fn apply_move_inner(&mut self, mv: ZertzMove) -> Result<(), String> {
        match mv {
            ZertzMove::Place { color, place_at, remove } => {
                if self.mid_capture.is_some() {
                    return Err("cannot place a marble during a capture chain".to_string());
                }
                if self.mid_placement.is_some() {
                    return Err("cannot start a new placement during a pending ring removal".to_string());
                }
                if !self.generate_first_hops().is_empty() {
                    return Err("captures are mandatory — must capture instead of placing".to_string());
                }
                if self.rings[hex_to_index(place_at)] != Ring::Empty {
                    return Err(format!("({},{}) is not an empty ring", place_at.0, place_at.1));
                }
                if remove == place_at {
                    return Err("cannot remove the ring you are placing on".to_string());
                }
                if self.rings[hex_to_index(remove)] != Ring::Empty {
                    return Err(format!("({},{}) is not an empty ring to remove", remove.0, remove.1));
                }
                if !Self::is_edge_static(&self.rings, remove) {
                    return Err(format!("({},{}) is not a removable edge ring", remove.0, remove.1));
                }
                self.take_marble(color)?;
                self.rings[hex_to_index(place_at)] = Ring::Occupied(color);
                self.rings[hex_to_index(remove)] = Ring::Removed;
                self.resolve_isolation();
                self.check_winner();
                self.next_player = self.next_player.opposite();
            }
            ZertzMove::PlaceOnly { color, place_at } => {
                if self.mid_capture.is_some() {
                    return Err("cannot place a marble during a capture chain".to_string());
                }
                if self.mid_placement.is_some() {
                    return Err("cannot start a new placement during a pending ring removal".to_string());
                }
                if !self.generate_first_hops().is_empty() {
                    return Err("captures are mandatory — must capture instead of placing".to_string());
                }
                if self.rings[hex_to_index(place_at)] != Ring::Empty {
                    return Err(format!("({},{}) is not an empty ring", place_at.0, place_at.1));
                }
                // PlaceOnly is only legal when no other ring can be removed.
                let has_removable = self.rings.iter().enumerate().any(|(i, &r)| {
                    r == Ring::Empty
                        && index_to_hex(i) != place_at
                        && Self::is_edge_static(&self.rings, index_to_hex(i))
                });
                if has_removable {
                    return Err("must remove a ring after placing — use Place instead of PlaceOnly".to_string());
                }
                self.take_marble(color)?;
                self.rings[hex_to_index(place_at)] = Ring::Occupied(color);
                self.resolve_isolation();
                self.check_winner();
                self.next_player = self.next_player.opposite();
            }
            ZertzMove::PlaceMarbleHalf { color, place_at } => {
                if self.mid_capture.is_some() {
                    return Err("cannot place a marble during a capture chain".to_string());
                }
                if self.mid_placement.is_some() {
                    return Err("placement first-half already in progress".to_string());
                }
                if !self.generate_first_hops().is_empty() {
                    return Err("captures are mandatory — must capture instead of placing".to_string());
                }
                if self.rings[hex_to_index(place_at)] != Ring::Empty {
                    return Err(format!("({},{}) is not an empty ring", place_at.0, place_at.1));
                }
                // Must be a state where some removable ring will still exist
                // after placement — otherwise PlaceOnly is the correct move.
                let has_post_removal = self.rings.iter().enumerate().any(|(i, &r)| {
                    r == Ring::Empty
                        && index_to_hex(i) != place_at
                        && Self::is_edge_static(&self.rings, index_to_hex(i))
                });
                if !has_post_removal {
                    return Err(
                        "no removable ring will exist after placement — use PlaceOnly".to_string()
                    );
                }
                self.take_marble(color)?;
                self.rings[hex_to_index(place_at)] = Ring::Occupied(color);
                self.mid_placement = Some(MidPlacementState);
                // No isolation/winner check yet — the turn is mid-flight.
            }
            ZertzMove::RemoveRingHalf { remove } => {
                if self.mid_placement.is_none() {
                    return Err("RemoveRingHalf is only legal during mid-placement".to_string());
                }
                if self.rings[hex_to_index(remove)] != Ring::Empty {
                    return Err(format!("({},{}) is not an empty ring to remove", remove.0, remove.1));
                }
                if !Self::is_edge_static(&self.rings, remove) {
                    return Err(format!("({},{}) is not a removable edge ring", remove.0, remove.1));
                }
                self.rings[hex_to_index(remove)] = Ring::Removed;
                self.mid_placement = None;
                self.resolve_isolation();
                self.check_winner();
                self.next_player = self.next_player.opposite();
            }
            ZertzMove::Capture { jumps, len } => {
                if self.mid_placement.is_some() {
                    return Err("cannot capture during a pending ring removal".to_string());
                }
                if len == 1 {
                    let (from, over, to) = jumps[0];
                    // During a capture chain, only the active marble may continue.
                    if let Some(mc) = self.mid_capture {
                        if from != mc.marble_pos {
                            return Err(format!(
                                "mid-capture continuation must use the marble at ({},{}), not ({},{})",
                                mc.marble_pos.0, mc.marble_pos.1, from.0, from.1
                            ));
                        }
                    }
                    if !matches!(self.rings[hex_to_index(from)], Ring::Occupied(_)) {
                        return Err(format!("no marble at ({},{}) to capture with", from.0, from.1));
                    }
                    if self.rings[hex_to_index(to)] != Ring::Empty {
                        return Err(format!("landing position ({},{}) is not empty", to.0, to.1));
                    }
                    self.apply_single_hop(from, over, to)?;
                } else {
                    // Multi-hop — used by replay/boardspace parsing (trusted).
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

        // History tracking: only record after a complete turn (not mid-capture
        // or mid-placement — those are same-player continuations).
        if self.mid_capture.is_none()
            && self.mid_placement.is_none()
            && self.outcome == Outcome::Ongoing
        {
            let key = self.position_key();
            let count = self.history.entry(key).or_insert(0);
            *count += 1;
            if *count >= 2 {
                self.outcome = Outcome::Draw;
            }
            // Record which key we bumped so `undo` can decrement (and remove
            // the entry if its count drops to zero).
            if let Some(top) = self.apply_history.last_mut() {
                top.repetition_key = Some(key);
            }
        }
        Ok(())
    }

    /// Whether the board is in mid-capture state (a chain is in progress).
    pub fn is_mid_capture(&self) -> bool {
        self.mid_capture.is_some()
    }

    /// Whether the board is mid-placement: a marble has been placed via
    /// `PlaceMarbleHalf` but no `RemoveRingHalf` has been applied yet.
    pub fn is_mid_placement(&self) -> bool {
        self.mid_placement.is_some()
    }

    /// Whether this is a capture turn (first-hop or mid-capture).
    /// Captures are mandatory in Zertz, so if any hop exists, it's a capture turn.
    pub fn is_capture_turn(&self) -> bool {
        self.mid_capture.is_some() || !self.generate_first_hops().is_empty()
    }

    /// Position of the active marble during mid-capture, if any.
    pub fn mid_capture_pos(&self) -> Option<Hex> {
        self.mid_capture.map(|mc| mc.marble_pos)
    }

    /// Get all legal moves for the current position in the **split** move
    /// space — what MCTS searches over. Placements are returned as
    /// `PlaceMarbleHalf` (followed in mid_placement by `RemoveRingHalf`), or
    /// as `PlaceOnly` for terminal placements with no removable ring left.
    /// During mid-capture/mid-placement the move space is restricted to the
    /// continuation moves.
    pub fn legal_moves(&self) -> Vec<ZertzMove> {
        if self.outcome != Outcome::Ongoing {
            return Vec::new();
        }

        if self.mid_placement.is_some() {
            return self.generate_ring_removals();
        }

        if let Some(mc) = self.mid_capture {
            return self.generate_continuation_hops(mc.marble_pos);
        }

        let hops = self.generate_first_hops();
        if !hops.is_empty() {
            // Captures are mandatory.
            return hops;
        }

        self.generate_split_placements()
    }

    /// Get all legal moves in the **joint** move space — placements appear as
    /// single-ply `Place` / `PlaceOnly` moves. Used by alpha-beta. The board
    /// must NOT be in mid_placement state (caller should never reach a
    /// mid_placement state when only joint applies are used).
    pub fn legal_joint_moves(&self) -> Vec<ZertzMove> {
        if self.outcome != Outcome::Ongoing {
            return Vec::new();
        }
        debug_assert!(
            self.mid_placement.is_none(),
            "legal_joint_moves called on a mid_placement board — joint callers should not visit this state",
        );

        if let Some(mc) = self.mid_capture {
            return self.generate_continuation_hops(mc.marble_pos);
        }

        let hops = self.generate_first_hops();
        if !hops.is_empty() {
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
        // Alpha-beta is the only Game-trait consumer of `valid_moves` (MCTS
        // goes through `NNGame::get_legal_move_mask`). AB wants the joint
        // placement move-space — searching split halves would double the
        // effective depth for the same game horizon.
        self.legal_joint_moves()
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

impl NNGame for ZertzBoard {
    const BOARD_CHANNELS: usize    = crate::board_encoding::NUM_CHANNELS;
    const RESERVE_SIZE: usize      = crate::board_encoding::RESERVE_SIZE;
    const NUM_POLICY_CHANNELS: usize = crate::move_encoding::NN_POLICY_CHANNELS;

    fn grid_size(&self) -> usize {
        crate::board_encoding::GRID_SIZE
    }

    fn encode_board(&self, board_out: &mut [f32], reserve_out: &mut [f32]) {
        crate::board_encoding::encode_board(self, board_out, reserve_out);
    }

    fn get_legal_move_mask(&mut self) -> (Vec<f32>, Vec<(PolicyIndex, ZertzMove)>) {
        crate::move_encoding::get_legal_move_mask_nn(self)
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

// ---------------------------------------------------------------------------
// Tests — split-placement state machine
// ---------------------------------------------------------------------------

#[cfg(test)]
mod split_placement_tests {
    use super::*;

    /// On a default board, `legal_moves` (split) yields PlaceMarbleHalf moves
    /// for every (color, empty cell) pair — same count as available_colors() *
    /// empty_cells. legal_joint_moves yields Place + PlaceOnly.
    #[test]
    fn legal_moves_returns_split_form_on_default() {
        let board = ZertzBoard::default();
        let split = board.legal_moves();
        let joint = board.legal_joint_moves();
        assert!(!split.is_empty());
        assert!(!joint.is_empty());
        // Every split move on a default board is PlaceMarbleHalf (no terminal
        // PlaceOnly because every cell leaves removable rings behind).
        for mv in &split {
            assert!(
                matches!(mv, ZertzMove::PlaceMarbleHalf { .. }),
                "expected PlaceMarbleHalf, got {:?}", mv
            );
        }
        // Joint move-space is strictly larger than split first-halves (one
        // half-move expands into many (color, place, remove) joint moves).
        assert!(joint.len() > split.len());
    }

    /// Applying PlaceMarbleHalf sets mid_placement, keeps the same player on
    /// move, and restricts legal_moves to RemoveRingHalf only.
    #[test]
    fn place_marble_half_sets_mid_placement() {
        let mut board = ZertzBoard::default();
        let before_player = board.next_player();
        let mv = ZertzMove::PlaceMarbleHalf {
            color: Marble::White,
            place_at: (0, 0),
        };
        board.play(mv).expect("PlaceMarbleHalf should be legal");
        assert!(board.is_mid_placement(), "should be mid_placement after PlaceMarbleHalf");
        assert_eq!(board.next_player(), before_player, "player must not switch mid-placement");

        // Now only ring removals are legal.
        let next = board.legal_moves();
        assert!(!next.is_empty());
        for mv in &next {
            assert!(
                matches!(mv, ZertzMove::RemoveRingHalf { .. }),
                "expected RemoveRingHalf during mid_placement, got {:?}", mv
            );
        }
    }

    /// PlaceMarbleHalf + RemoveRingHalf executed atomically equals the joint
    /// Place move in terms of resulting board state.
    #[test]
    fn split_equals_joint_after_both_halves() {
        let place = (0i8, 0i8);
        let remove = (3i8, -3i8); // a corner edge

        let mut split_board = ZertzBoard::default();
        split_board.play(ZertzMove::PlaceMarbleHalf {
            color: Marble::White,
            place_at: place,
        }).unwrap();
        split_board.play(ZertzMove::RemoveRingHalf { remove }).unwrap();

        let mut joint_board = ZertzBoard::default();
        joint_board.play(ZertzMove::Place {
            color: Marble::White,
            place_at: place,
            remove,
        }).unwrap();

        assert_eq!(split_board.rings(), joint_board.rings());
        assert_eq!(split_board.captures(), joint_board.captures());
        assert_eq!(split_board.supply(), joint_board.supply());
        assert_eq!(split_board.next_player(), joint_board.next_player());
        assert!(!split_board.is_mid_placement());
        assert!(!joint_board.is_mid_placement());
    }

    /// You can't start a new placement, capture, or PlaceOnly while a
    /// ring removal is pending.
    #[test]
    fn other_moves_rejected_during_mid_placement() {
        let mut board = ZertzBoard::default();
        board.play(ZertzMove::PlaceMarbleHalf {
            color: Marble::White,
            place_at: (0, 0),
        }).unwrap();

        // Place during mid_placement → error
        let err = board.play(ZertzMove::Place {
            color: Marble::White,
            place_at: (1, 0),
            remove: (3, -3),
        });
        assert!(err.is_err(), "Place should be rejected mid_placement");

        // PlaceOnly during mid_placement → error
        let err = board.play(ZertzMove::PlaceOnly {
            color: Marble::White,
            place_at: (1, 0),
        });
        assert!(err.is_err(), "PlaceOnly should be rejected mid_placement");

        // RemoveRingHalf without mid_placement → error
        let mut fresh = ZertzBoard::default();
        let err = fresh.play(ZertzMove::RemoveRingHalf { remove: (3, -3) });
        assert!(err.is_err(), "RemoveRingHalf should be rejected without mid_placement");
    }

    /// Undo a PlaceMarbleHalf restores both the board AND the cleared
    /// mid_placement state.
    #[test]
    fn undo_through_split_placement() {
        let mut board = ZertzBoard::default();
        let before_rings = *board.rings();
        let before_supply = *board.supply();
        let before_player = board.next_player();

        board.play(ZertzMove::PlaceMarbleHalf {
            color: Marble::White,
            place_at: (0, 0),
        }).unwrap();
        assert!(board.is_mid_placement());

        board.play(ZertzMove::RemoveRingHalf { remove: (3, -3) }).unwrap();
        assert!(!board.is_mid_placement());

        // Undo the ring removal — should re-enter mid_placement.
        board.undo();
        assert!(board.is_mid_placement());

        // Undo the marble placement — should be back to default.
        board.undo();
        assert!(!board.is_mid_placement());
        assert_eq!(*board.rings(), before_rings);
        assert_eq!(*board.supply(), before_supply);
        assert_eq!(board.next_player(), before_player);
    }

    /// The board encoder sets the new channel 6 only during mid_placement.
    #[test]
    fn encoder_channel_6_tracks_mid_placement() {
        use crate::board_encoding::{encode_board, GRID_SIZE, NUM_CHANNELS, RESERVE_SIZE};
        let g2 = GRID_SIZE * GRID_SIZE;
        let mut buf = vec![0.0f32; NUM_CHANNELS * g2];
        let mut reserve = vec![0.0f32; RESERVE_SIZE];

        let board = ZertzBoard::default();
        encode_board(&board, &mut buf, &mut reserve);
        // Channel 6 should be zero on a default board (not mid_placement).
        let ch6_sum: f32 = buf[6 * g2..7 * g2].iter().sum();
        assert_eq!(ch6_sum, 0.0);

        let mut board = ZertzBoard::default();
        board.play(ZertzMove::PlaceMarbleHalf {
            color: Marble::White,
            place_at: (0, 0),
        }).unwrap();
        encode_board(&board, &mut buf, &mut reserve);
        // Channel 6 broadcast — every cell is 1.0.
        for i in 0..g2 {
            assert_eq!(buf[6 * g2 + i], 1.0, "ch6 cell {} expected 1.0", i);
        }
    }
}
