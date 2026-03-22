/// Board representation for Hive with beetle stacking.
/// Uses a dense 23x23 grid (fits L1 cache) plus piece position lookup.

use crate::hex::{Hex, hex_neighbors};
use crate::piece::{Piece, PieceColor, PIECES_PER_PLAYER};

pub const GRID_SIZE: usize = 23;
pub const GRID_CENTER: i8 = 11;
pub const MAX_STACK: usize = 7;
const TOTAL_PIECES: usize = PIECES_PER_PLAYER * 2; // 22

/// A slot in the grid that can hold a stack of pieces.
#[derive(Clone, Copy)]
pub struct StackSlot {
    /// Pieces in this slot, bottom to top. 0 = empty sentinel.
    pieces: [u8; MAX_STACK],
    /// Number of pieces in stack.
    height: u8,
}

impl Default for StackSlot {
    fn default() -> Self {
        StackSlot {
            pieces: [0; MAX_STACK],
            height: 0,
        }
    }
}

impl StackSlot {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.height == 0
    }

    #[inline]
    pub fn height(&self) -> u8 {
        self.height
    }

    #[inline]
    pub fn top(&self) -> Option<Piece> {
        if self.height > 0 {
            Some(Piece::from_raw(self.pieces[self.height as usize - 1]))
        } else {
            None
        }
    }

    #[inline]
    pub fn push(&mut self, piece: Piece) {
        debug_assert!((self.height as usize) < MAX_STACK);
        self.pieces[self.height as usize] = piece.raw();
        self.height += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> Piece {
        debug_assert!(self.height > 0);
        self.height -= 1;
        Piece::from_raw(self.pieces[self.height as usize])
    }

    /// Iterator over pieces bottom to top.
    pub fn iter(&self) -> impl Iterator<Item = Piece> + '_ {
        self.pieces[..self.height as usize]
            .iter()
            .map(|&raw| Piece::from_raw(raw))
    }
}

/// Convert axial hex to grid indices. Returns None if out of bounds.
#[inline]
pub fn hex_to_grid(h: Hex) -> Option<(usize, usize)> {
    let col = h.0 as i16 + GRID_CENTER as i16;
    let row = h.1 as i16 + GRID_CENTER as i16;
    if col >= 0 && col < GRID_SIZE as i16 && row >= 0 && row < GRID_SIZE as i16 {
        Some((row as usize, col as usize))
    } else {
        None
    }
}

/// Convert grid indices to axial hex.
#[inline]
pub fn grid_to_hex(row: usize, col: usize) -> Hex {
    (col as i8 - GRID_CENTER, row as i8 - GRID_CENTER)
}

/// Dense board representation.
#[derive(Clone)]
pub struct Board {
    /// 23x23 grid of stack slots.
    grid: Box<[[StackSlot; GRID_SIZE]; GRID_SIZE]>,
    /// Reverse lookup: piece linear index -> position (row, col), or None.
    piece_positions: [Option<(u8, u8)>; TOTAL_PIECES],
    /// Number of occupied cells (positions with at least 1 piece).
    pub occupied_count: u16,
}

impl Board {
    pub fn new() -> Self {
        Board {
            grid: Box::new([[StackSlot::default(); GRID_SIZE]; GRID_SIZE]),
            piece_positions: [None; TOTAL_PIECES],
            occupied_count: 0,
        }
    }

    /// Get the stack at a hex position.
    #[inline]
    pub fn stack_at(&self, h: Hex) -> &StackSlot {
        let (r, c) = hex_to_grid(h).expect("hex out of grid bounds");
        &self.grid[r][c]
    }

    /// Get mutable stack at a hex position.
    #[inline]
    fn stack_at_mut(&mut self, h: Hex) -> &mut StackSlot {
        let (r, c) = hex_to_grid(h).expect("hex out of grid bounds");
        &mut self.grid[r][c]
    }

    /// Get stack at grid position.
    #[inline]
    pub fn stack_at_grid(&self, row: usize, col: usize) -> &StackSlot {
        &self.grid[row][col]
    }

    /// Check if a position is occupied.
    #[inline]
    pub fn is_occupied(&self, h: Hex) -> bool {
        if let Some((r, c)) = hex_to_grid(h) {
            !self.grid[r][c].is_empty()
        } else {
            false
        }
    }

    /// Get the top piece at a position.
    #[inline]
    pub fn top_piece(&self, h: Hex) -> Option<Piece> {
        if let Some((r, c)) = hex_to_grid(h) {
            self.grid[r][c].top()
        } else {
            None
        }
    }

    /// Stack height at a position.
    #[inline]
    pub fn stack_height(&self, h: Hex) -> u8 {
        if let Some((r, c)) = hex_to_grid(h) {
            self.grid[r][c].height()
        } else {
            0
        }
    }

    /// Get position of a piece (as hex).
    pub fn piece_position(&self, piece: Piece) -> Option<Hex> {
        let idx = piece.linear_index();
        self.piece_positions[idx].map(|(r, c)| grid_to_hex(r as usize, c as usize))
    }

    /// Place a piece on the board (on top of any existing stack).
    pub fn place_piece(&mut self, piece: Piece, pos: Hex) -> Result<(), String> {
        let (r, c) = hex_to_grid(pos)
            .ok_or_else(|| format!("hex ({}, {}) out of grid bounds", pos.0, pos.1))?;
        let was_empty = self.grid[r][c].is_empty();
        self.grid[r][c].push(piece);
        self.piece_positions[piece.linear_index()] = Some((r as u8, c as u8));
        if was_empty {
            self.occupied_count += 1;
        }
        Ok(())
    }

    /// Remove the top piece at a piece's position. Returns the position.
    pub fn remove_piece(&mut self, piece: Piece) -> Result<Hex, String> {
        let idx = piece.linear_index();
        let (r, c) = self.piece_positions[idx]
            .ok_or_else(|| format!("{} not on board", piece.to_uhp_string()))?;
        let slot = &mut self.grid[r as usize][c as usize];
        let top = slot.top()
            .ok_or_else(|| format!("{} position ({}, {}) has empty stack", piece.to_uhp_string(), r, c))?;
        if top != piece {
            return Err(format!("{} not on top of stack (top is {})",
                piece.to_uhp_string(), top.to_uhp_string()));
        }
        slot.pop();
        self.piece_positions[idx] = None;
        if slot.is_empty() {
            self.occupied_count -= 1;
        }
        Ok(grid_to_hex(r as usize, c as usize))
    }

    /// Move a piece from its current position to dest.
    pub fn move_piece(&mut self, piece: Piece, dest: Hex) -> Result<(), String> {
        self.remove_piece(piece)?;
        self.place_piece(piece, dest)
    }

    /// Return occupied hex neighbors.
    pub fn neighbors_of(&self, pos: Hex) -> Vec<Hex> {
        let mut result = Vec::new();
        for &n in hex_neighbors(pos).iter() {
            if self.is_occupied(n) {
                result.push(n);
            }
        }
        result
    }

    /// Return unoccupied hex neighbors.
    pub fn empty_neighbors(&self, pos: Hex) -> Vec<Hex> {
        let mut result = Vec::new();
        for &n in hex_neighbors(pos).iter() {
            if let Some((r, c)) = hex_to_grid(n) {
                if self.grid[r][c].is_empty() {
                    result.push(n);
                }
            }
        }
        result
    }

    /// Check if a ground-level piece can slide between two adjacent positions.
    /// Blocked if both common neighbors are occupied (gate blocking).
    pub fn can_slide(&self, from: Hex, to: Hex) -> bool {
        // Find the two common neighbors of from and to
        let from_ns = hex_neighbors(from);
        let to_ns = hex_neighbors(to);
        let mut blocked = 0;
        for &fn_ in &from_ns {
            if fn_ == to {
                continue;
            }
            for &tn in &to_ns {
                if fn_ == tn && fn_ != from {
                    // This is a common neighbor
                    if self.is_occupied(fn_) {
                        blocked += 1;
                    }
                }
            }
        }
        blocked < 2
    }

    /// Check if all pieces form a single connected group (BFS).
    pub fn is_connected(&self, exclude: Option<Piece>) -> bool {
        if self.occupied_count <= 1 {
            return true;
        }

        // Build set of occupied positions, possibly excluding one
        let mut exclude_pos: Option<Hex> = None;
        let mut exclude_removes_pos = false;

        if let Some(piece) = exclude {
            if let Some(pos) = self.piece_position(piece) {
                let height = self.stack_height(pos);
                if height == 1 {
                    exclude_pos = Some(pos);
                    exclude_removes_pos = true;
                }
            }
        }

        let total_positions = if exclude_removes_pos {
            self.occupied_count as usize - 1
        } else {
            self.occupied_count as usize
        };

        if total_positions <= 1 {
            return true;
        }

        // Find a start position
        let mut start = None;
        for r in 0..GRID_SIZE {
            for c in 0..GRID_SIZE {
                if !self.grid[r][c].is_empty() {
                    let h = grid_to_hex(r, c);
                    if exclude_pos != Some(h) {
                        start = Some(h);
                        break;
                    }
                }
            }
            if start.is_some() {
                break;
            }
        }

        let start = match start {
            Some(s) => s,
            None => return true,
        };

        // BFS
        let mut visited = vec![false; GRID_SIZE * GRID_SIZE];
        let mut queue = Vec::with_capacity(32);
        let (sr, sc) = hex_to_grid(start).unwrap();
        visited[sr * GRID_SIZE + sc] = true;
        queue.push(start);
        let mut count = 1usize;

        while let Some(current) = queue.pop() {
            for &n in hex_neighbors(current).iter() {
                if let Some((r, c)) = hex_to_grid(n) {
                    let idx = r * GRID_SIZE + c;
                    if !visited[idx] && !self.grid[r][c].is_empty() && exclude_pos != Some(n) {
                        visited[idx] = true;
                        count += 1;
                        queue.push(n);
                    }
                }
            }
        }

        count == total_positions
    }

    /// Find articulation points using Tarjan's algorithm.
    /// Only considers positions with stack height 1.
    pub fn articulation_points(&self) -> Vec<Hex> {
        if self.occupied_count <= 2 {
            return Vec::new();
        }

        let mut disc = [0u16; GRID_SIZE * GRID_SIZE];
        let mut low = [0u16; GRID_SIZE * GRID_SIZE];
        let mut parent = [u16::MAX; GRID_SIZE * GRID_SIZE]; // MAX = no parent
        let mut visited = [false; GRID_SIZE * GRID_SIZE];
        let mut aps = Vec::new();
        let mut timer = 1u16;

        // Find start position
        let mut start_idx = 0;
        for r in 0..GRID_SIZE {
            for c in 0..GRID_SIZE {
                if !self.grid[r][c].is_empty() {
                    start_idx = r * GRID_SIZE + c;
                    break;
                }
            }
            if start_idx != 0 || !self.grid[0][0].is_empty() {
                break;
            }
        }

        // Iterative Tarjan's to avoid stack overflow
        // Using explicit stack: (hex, neighbor_index, is_initial_call)
        struct Frame {
            hex: Hex,
            idx: usize,        // grid index
            neighbor_i: usize, // which neighbor we're processing
            children: u16,
        }

        let start_hex = grid_to_hex(start_idx / GRID_SIZE, start_idx % GRID_SIZE);
        visited[start_idx] = true;
        disc[start_idx] = timer;
        low[start_idx] = timer;
        timer += 1;

        let mut stack: Vec<Frame> = vec![Frame {
            hex: start_hex,
            idx: start_idx,
            neighbor_i: 0,
            children: 0,
        }];

        while let Some(frame) = stack.last_mut() {
            let neighbors = hex_neighbors(frame.hex);
            if frame.neighbor_i < 6 {
                let ni = frame.neighbor_i;
                frame.neighbor_i += 1;
                let v = neighbors[ni];

                if let Some((vr, vc)) = hex_to_grid(v) {
                    let vidx = vr * GRID_SIZE + vc;
                    if self.grid[vr][vc].is_empty() {
                        // Not an occupied position, skip
                    } else if !visited[vidx] {
                        // Tree edge
                        frame.children += 1;
                        visited[vidx] = true;
                        parent[vidx] = frame.idx as u16;
                        disc[vidx] = timer;
                        low[vidx] = timer;
                        timer += 1;

                        stack.push(Frame {
                            hex: v,
                            idx: vidx,
                            neighbor_i: 0,
                            children: 0,
                        });
                    } else if vidx != parent[frame.idx] as usize {
                        // Back edge
                        if disc[vidx] < low[frame.idx] {
                            low[frame.idx] = disc[vidx];
                        }
                    }
                }
            } else {
                // Done processing all neighbors
                let finished = stack.pop().unwrap();
                let u_idx = finished.idx;

                if let Some(parent_frame) = stack.last_mut() {
                    let p_idx = parent_frame.idx;
                    // Update parent's low
                    if low[u_idx] < low[p_idx] {
                        low[p_idx] = low[u_idx];
                    }
                    // Check if parent is AP: not root and low[child] >= disc[parent]
                    if parent[p_idx] != u16::MAX && low[u_idx] >= disc[p_idx] {
                        let p_hex = grid_to_hex(p_idx / GRID_SIZE, p_idx % GRID_SIZE);
                        if !aps.contains(&p_hex) {
                            aps.push(p_hex);
                        }
                    }
                } else {
                    // This was root
                    if finished.children > 1 {
                        let root_hex = grid_to_hex(u_idx / GRID_SIZE, u_idx % GRID_SIZE);
                        if !aps.contains(&root_hex) {
                            aps.push(root_hex);
                        }
                    }
                }
            }
        }

        aps
    }

    /// All pieces on the board of a given color.
    pub fn pieces_on_board(&self, color: PieceColor) -> Vec<Piece> {
        let mut result = Vec::new();
        let start = if color == PieceColor::White { 0 } else { PIECES_PER_PLAYER };
        let end = start + PIECES_PER_PLAYER;
        for idx in start..end {
            if self.piece_positions[idx].is_some() {
                // Reconstruct piece from linear index
                result.push(piece_from_linear_index(idx));
            }
        }
        result
    }

    /// All (position, top_piece) pairs.
    pub fn all_top_pieces(&self) -> Vec<(Hex, Piece)> {
        let mut result = Vec::new();
        for r in 0..GRID_SIZE {
            for c in 0..GRID_SIZE {
                if let Some(piece) = self.grid[r][c].top() {
                    result.push((grid_to_hex(r, c), piece));
                }
            }
        }
        result
    }

    /// Iterate over all occupied positions and their stacks.
    pub fn iter_occupied(&self) -> impl Iterator<Item = (Hex, &StackSlot)> {
        (0..GRID_SIZE).flat_map(move |r| {
            (0..GRID_SIZE).filter_map(move |c| {
                if !self.grid[r][c].is_empty() {
                    Some((grid_to_hex(r, c), &self.grid[r][c]))
                } else {
                    None
                }
            })
        })
    }
}

/// Reconstruct a Piece from its linear index (0..21).
fn piece_from_linear_index(idx: usize) -> Piece {
    use crate::piece::{ALL_PIECE_TYPES, PIECE_COUNTS};

    let (color, offset) = if idx < PIECES_PER_PLAYER {
        (PieceColor::White, idx)
    } else {
        (PieceColor::Black, idx - PIECES_PER_PLAYER)
    };

    let mut remaining = offset;
    for (i, &pt) in ALL_PIECE_TYPES.iter().enumerate() {
        let count = PIECE_COUNTS[i] as usize;
        if remaining < count {
            return Piece::new(color, pt, remaining as u8 + 1);
        }
        remaining -= count;
    }
    unreachable!()
}

impl Piece {
    /// Construct from raw u8.
    #[inline]
    pub fn from_raw(raw: u8) -> Self {
        Piece(raw)
    }
}

// Make Piece(u8) accessible from piece module
// We need the struct to be the same - it already is since we re-export

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::*;

    #[test]
    fn test_place_and_remove() {
        let mut board = Board::new();
        let p = Piece::new(PieceColor::White, PieceType::Queen, 1);
        board.place_piece(p, (0, 0));
        assert_eq!(board.occupied_count, 1);
        assert!(board.is_occupied((0, 0)));
        assert_eq!(board.top_piece((0, 0)), Some(p));
        assert_eq!(board.piece_position(p), Some((0, 0)));

        board.remove_piece(p).unwrap();
        assert_eq!(board.occupied_count, 0);
        assert!(!board.is_occupied((0, 0)));
    }

    #[test]
    fn test_stacking() {
        let mut board = Board::new();
        let q = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let b = Piece::new(PieceColor::Black, PieceType::Beetle, 1);

        board.place_piece(q, (0, 0));
        board.place_piece(b, (0, 0));

        assert_eq!(board.stack_height((0, 0)), 2);
        assert_eq!(board.top_piece((0, 0)), Some(b));
        assert_eq!(board.occupied_count, 1);

        board.remove_piece(b).unwrap();
        assert_eq!(board.top_piece((0, 0)), Some(q));
        assert_eq!(board.occupied_count, 1);
    }

    #[test]
    fn test_can_slide() {
        let mut board = Board::new();
        let q = Piece::new(PieceColor::White, PieceType::Queen, 1);
        board.place_piece(q, (0, 0));

        // No blocking pieces: can always slide
        assert!(board.can_slide((0, 0), (1, 0)));

        // Add blocking pieces to create a gate
        let s1 = Piece::new(PieceColor::White, PieceType::Spider, 1);
        let s2 = Piece::new(PieceColor::White, PieceType::Spider, 2);
        // Common neighbors of (0,0) and (1,0) are (1,-1) and (0,1)
        board.place_piece(s1, (1, -1));
        board.place_piece(s2, (0, 1));
        assert!(!board.can_slide((0, 0), (1, 0)));
    }

    #[test]
    fn test_connectivity() {
        let mut board = Board::new();
        let p1 = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let p2 = Piece::new(PieceColor::Black, PieceType::Queen, 1);
        let p3 = Piece::new(PieceColor::White, PieceType::Ant, 1);

        board.place_piece(p1, (0, 0));
        board.place_piece(p2, (1, 0));
        board.place_piece(p3, (2, 0));

        assert!(board.is_connected(None));
        // Removing middle piece disconnects
        assert!(!board.is_connected(Some(p2)));
        // Removing end piece keeps connected
        assert!(board.is_connected(Some(p3)));
    }

    #[test]
    fn test_articulation_points() {
        let mut board = Board::new();
        let p1 = Piece::new(PieceColor::White, PieceType::Queen, 1);
        let p2 = Piece::new(PieceColor::Black, PieceType::Queen, 1);
        let p3 = Piece::new(PieceColor::White, PieceType::Ant, 1);

        board.place_piece(p1, (0, 0));
        board.place_piece(p2, (1, 0));
        board.place_piece(p3, (2, 0));

        let aps = board.articulation_points();
        assert!(aps.contains(&(1, 0))); // middle piece is AP
        assert!(!aps.contains(&(0, 0)));
        assert!(!aps.contains(&(2, 0)));
    }

    #[test]
    fn test_all_top_pieces_after_place() {
        use crate::piece::{Piece, PieceColor, PieceType};
        let mut board = Board::new();
        let piece = Piece::new(PieceColor::White, PieceType::Queen, 1);
        board.place_piece(piece, (0, 0));

        let tops = board.all_top_pieces();
        assert_eq!(tops.len(), 1);
        let (pos, p) = &tops[0];
        assert_eq!(p.to_string(), "wQ1");
        println!("top piece: {} at {:?}", p, pos);
    }

    #[test]
    fn test_hex_to_grid_roundtrip() {
        for q in -11..=11 {
            for r in -11..=11 {
                let h: Hex = (q, r);
                let (row, col) = hex_to_grid(h).unwrap();
                let h2 = grid_to_hex(row, col);
                assert_eq!(h, h2);
            }
        }
    }
}
