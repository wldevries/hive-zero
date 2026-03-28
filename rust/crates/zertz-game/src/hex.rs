/// Zertz-specific hex utilities built on core-game's axial coordinate system.

pub use core_game::hex::*;

/// Radius of the standard Zertz board (rows 4-5-6-7-6-5-4).
pub const RADIUS: i8 = 3;

/// Number of cells on the standard board.
pub const BOARD_SIZE: usize = 37;

/// Grid size for NN encoding (RADIUS*2 + 1).
pub const GRID_SIZE: usize = 7;

/// Check if a hex coordinate is on the standard Zertz board.
#[inline]
pub fn is_valid(h: Hex) -> bool {
    h.0.abs() <= RADIUS && h.1.abs() <= RADIUS && (h.0 + h.1).abs() <= RADIUS
}

/// All 37 valid hex positions on the standard board, in a deterministic order.
/// Ordered by (r, q) ascending — row by row, left to right.
pub const fn all_hexes() -> [Hex; BOARD_SIZE] {
    let mut hexes = [(0i8, 0i8); BOARD_SIZE];
    let mut i = 0;
    let mut r = -RADIUS;
    while r <= RADIUS {
        let q_min = if -RADIUS > -RADIUS - r { -RADIUS } else { -RADIUS - r };
        let q_max = if RADIUS < RADIUS - r { RADIUS } else { RADIUS - r };
        let mut q = q_min;
        while q <= q_max {
            hexes[i] = (q, r);
            i += 1;
            q += 1;
        }
        r += 1;
    }
    hexes
}

/// Precomputed lookup table: index → hex coordinate. O(1) `index_to_hex`.
pub const ALL_HEXES: [Hex; BOARD_SIZE] = all_hexes();

/// Neighbors of `h` that are on the Zertz board.
pub fn valid_neighbors(h: Hex) -> impl Iterator<Item = Hex> {
    hex_neighbors(h).into_iter().filter(|&n| is_valid(n))
}

/// Map a hex coordinate to a grid position (row, col) in a 7x7 grid for NN encoding.
/// Rows are left-aligned, matching row lengths [4,5,6,7,6,5,4].
#[inline]
pub fn hex_to_grid(h: Hex) -> (usize, usize) {
    let (q, r) = h;
    let grid_row = (r + RADIUS) as usize;
    let q_min = (-RADIUS).max(-RADIUS - r);
    let grid_col = (q - q_min) as usize;
    (grid_row, grid_col)
}

/// Map a hex coordinate to a deterministic linear index (0..36) for policy encoding.
/// Uses the same (r, q) ordering as `all_hexes()`.
#[inline]
pub fn hex_to_index(h: Hex) -> usize {
    let (q, r) = h;
    // Count all hexes in rows before r, plus offset within row.
    let mut idx = 0usize;
    let mut row = -RADIUS;
    while row < r {
        let q_min = (-RADIUS).max(-RADIUS - row);
        let q_max = RADIUS.min(RADIUS - row);
        idx += (q_max - q_min + 1) as usize;
        row += 1;
    }
    let q_min = (-RADIUS).max(-RADIUS - r);
    idx + (q - q_min) as usize
}

/// Map a linear index (0..36) back to hex coordinates. O(1) table lookup.
#[inline]
pub fn index_to_hex(idx: usize) -> Hex {
    ALL_HEXES[idx]
}

/// Convert a boardspace coordinate (col A=0..G=6, row 0-based within column) to hex.
/// Boardspace columns go left-to-right, with col_lengths [4,5,6,7,6,5,4].
/// Column c in boardspace maps to q-axis direction.
pub fn boardspace_to_hex(col: u8, row: u8) -> Option<Hex> {
    // Boardspace column c has col_lengths[c] cells.
    // Column c maps to axial q, and within the column, row maps to r.
    // Column 0 (A) has 4 cells: these map to the leftmost column of the hex grid.
    //
    // Boardspace layout (columns are vertical, labeled A-G left to right):
    //   Col A (4 cells), Col B (5), Col C (6), Col D (7), Col E (6), Col F (5), Col G (4)
    //
    // In axial coordinates, this maps to:
    //   Col A: q = -3, r ∈ [0, 3]     → 4 cells
    //   Col B: q = -2, r ∈ [-1, 3]    → 5 cells
    //   Col C: q = -1, r ∈ [-2, 3]    → 6 cells
    //   Col D: q = 0,  r ∈ [-3, 3]    → 7 cells
    //   Col E: q = 1,  r ∈ [-3, 2]    → 6 cells
    //   Col F: q = 2,  r ∈ [-3, 1]    → 5 cells
    //   Col G: q = 3,  r ∈ [-3, 0]    → 4 cells
    let col_lengths: [u8; 7] = [4, 5, 6, 7, 6, 5, 4];
    if col >= 7 || row >= col_lengths[col as usize] {
        return None;
    }
    let q = col as i8 - RADIUS; // A=0 → q=-3, D=3 → q=0, G=6 → q=3
    let r_min = (-RADIUS).max(-RADIUS - q);
    let r = r_min + row as i8;
    Some((q, r))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_size() {
        assert_eq!(all_hexes().len(), BOARD_SIZE);
    }

    #[test]
    fn test_all_hexes_valid() {
        for h in all_hexes() {
            assert!(is_valid(h), "hex {:?} should be valid", h);
        }
    }

    #[test]
    fn test_origin_valid() {
        assert!(is_valid((0, 0)));
    }

    #[test]
    fn test_boundary_valid() {
        assert!(is_valid((3, 0)));
        assert!(is_valid((-3, 0)));
        assert!(is_valid((0, 3)));
        assert!(is_valid((0, -3)));
        assert!(is_valid((3, -3)));
        assert!(is_valid((-3, 3)));
    }

    #[test]
    fn test_outside_invalid() {
        assert!(!is_valid((4, 0)));
        assert!(!is_valid((2, 2))); // |q+r| = 4 > 3
        assert!(!is_valid((-2, -2)));
    }

    #[test]
    fn test_center_has_6_neighbors() {
        assert_eq!(valid_neighbors((0, 0)).count(), 6);
    }

    #[test]
    fn test_corner_has_3_neighbors() {
        assert_eq!(valid_neighbors((3, 0)).count(), 3);
    }

    #[test]
    fn test_index_roundtrip() {
        for (i, h) in all_hexes().iter().enumerate() {
            assert_eq!(hex_to_index(*h), i);
            assert_eq!(index_to_hex(i), *h);
        }
    }

    #[test]
    fn test_row_lengths() {
        // Each row should have the expected number of cells
        let expected = [4, 5, 6, 7, 6, 5, 4];
        for (ri, &len) in expected.iter().enumerate() {
            let r = ri as i8 - RADIUS;
            let count = all_hexes().iter().filter(|h| h.1 == r).count();
            assert_eq!(count, len, "row r={r} should have {len} cells");
        }
    }

    #[test]
    fn test_hex_to_grid_row_col_bounds() {
        for h in all_hexes() {
            let (row, col) = hex_to_grid(h);
            assert!(row < GRID_SIZE, "row {row} out of bounds for {:?}", h);
            assert!(col < GRID_SIZE, "col {col} out of bounds for {:?}", h);
        }
    }

    #[test]
    fn test_hex_to_grid_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for h in all_hexes() {
            let pos = hex_to_grid(h);
            assert!(seen.insert(pos), "duplicate grid position {:?} for {:?}", pos, h);
        }
    }

    #[test]
    fn test_boardspace_to_hex_counts() {
        // Each boardspace column should produce the right number of valid hexes
        let col_lengths = [4u8, 5, 6, 7, 6, 5, 4];
        for (c, &len) in col_lengths.iter().enumerate() {
            for r in 0..len {
                assert!(
                    boardspace_to_hex(c as u8, r).is_some(),
                    "boardspace ({c}, {r}) should be valid"
                );
            }
            assert!(boardspace_to_hex(c as u8, len).is_none());
        }
    }

    #[test]
    fn test_boardspace_covers_all_hexes() {
        let mut seen = std::collections::HashSet::new();
        let col_lengths = [4u8, 5, 6, 7, 6, 5, 4];
        for (c, &len) in col_lengths.iter().enumerate() {
            for r in 0..len {
                let h = boardspace_to_hex(c as u8, r).unwrap();
                assert!(is_valid(h), "boardspace ({c}, {r}) → {:?} should be valid", h);
                assert!(seen.insert(h), "duplicate hex {:?} from boardspace ({c}, {r})", h);
            }
        }
        assert_eq!(seen.len(), BOARD_SIZE);
    }
}
