/// YINSH board geometry: 85 valid cells on an irregular hex grid.
///
/// Columns A..K (0..10), rows 1..11 (0-indexed 0..10). The valid cells per column:
///   A(0): rows 1-4    B(1): rows 0-6    C(2): rows 0-7    D(3): rows 0-8
///   E(4): rows 0-9    F(5): rows 1-9    G(6): rows 1-10   H(7): rows 2-10
///   I(8): rows 3-10   J(9): rows 4-10   K(10): rows 6-9

pub const BOARD_SIZE: usize = 85;
pub const GRID_SIZE: usize = 11;

/// Valid row range per column (0-indexed, inclusive).
pub const COL_STARTS: [u8; 11] = [1, 0, 0, 0, 0, 1, 1, 2, 3, 4, 6];
pub const COL_ENDS:   [u8; 11] = [4, 6, 7, 8, 9, 9, 10, 10, 10, 10, 9];

/// Cumulative count of cells in columns before index `c`.
pub const COL_OFFSETS: [usize; 11] = [0, 4, 11, 19, 28, 38, 47, 57, 66, 74, 81];

/// 6 hex neighbor directions as (dcol, drow).
pub const DIRECTIONS: [(i8, i8); 6] = [
    (0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1),
];

/// 3 row directions for 5-in-a-row detection (the positive half of DIRECTIONS).
pub const ROW_DIRS: [(i8, i8); 3] = [(0, 1), (1, 0), (1, -1)];

/// All 85 valid cells, ordered by (col, row).
pub const ALL_CELLS: [(u8, u8); BOARD_SIZE] = {
    let mut cells = [(0u8, 0u8); BOARD_SIZE];
    let mut i = 0;
    let mut col = 0u8;
    while col < 11 {
        let mut row = COL_STARTS[col as usize];
        let end = COL_ENDS[col as usize];
        while row <= end {
            cells[i] = (col, row);
            i += 1;
            row += 1;
        }
        col += 1;
    }
    cells
};

/// Lookup table: [col][row] -> cell index (or 255 if invalid).
pub const INVALID_IDX: u8 = 255;
pub const CELL_INDEX_TABLE: [[u8; 11]; 11] = {
    let mut table = [[INVALID_IDX; 11]; 11];
    let mut i = 0;
    while i < BOARD_SIZE {
        let cell = ALL_CELLS[i];
        table[cell.0 as usize][cell.1 as usize] = i as u8;
        i += 1;
    }
    table
};

#[inline]
pub fn is_valid(col: u8, row: u8) -> bool {
    (col as usize) < 11 && (row as usize) < 11
        && CELL_INDEX_TABLE[col as usize][row as usize] != INVALID_IDX
}

#[inline]
pub fn is_valid_i8(col: i8, row: i8) -> bool {
    col >= 0 && row >= 0 && col < 11 && row < 11
        && CELL_INDEX_TABLE[col as usize][row as usize] != INVALID_IDX
}

#[inline]
pub fn cell_index(col: u8, row: u8) -> usize {
    CELL_INDEX_TABLE[col as usize][row as usize] as usize
}

#[inline]
pub fn cell_index_i8(col: i8, row: i8) -> usize {
    CELL_INDEX_TABLE[col as usize][row as usize] as usize
}

#[inline]
pub fn index_to_cell(idx: usize) -> (u8, u8) {
    ALL_CELLS[idx]
}

pub fn neighbors(col: u8, row: u8) -> impl Iterator<Item = (u8, u8)> {
    DIRECTIONS.iter().filter_map(move |&(dc, dr)| {
        let nc = col as i8 + dc;
        let nr = row as i8 + dr;
        if is_valid_i8(nc, nr) { Some((nc as u8, nr as u8)) } else { None }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_size() {
        assert_eq!(ALL_CELLS.len(), BOARD_SIZE);
        let per_col = [4, 7, 8, 9, 10, 9, 10, 9, 8, 7, 4];
        assert_eq!(per_col.iter().sum::<usize>(), BOARD_SIZE);
    }

    #[test]
    fn test_index_roundtrip() {
        for i in 0..BOARD_SIZE {
            let (c, r) = index_to_cell(i);
            assert_eq!(cell_index(c, r), i);
        }
    }

    #[test]
    fn test_corners() {
        assert!(is_valid(0, 1));  // A2
        assert!(is_valid(0, 4));  // A5
        assert!(!is_valid(0, 0)); // A1 invalid
        assert!(!is_valid(0, 5)); // A6 invalid
        assert!(is_valid(10, 6)); // K7
        assert!(is_valid(10, 9)); // K10
        assert!(!is_valid(10, 5)); // K6 invalid
        assert!(!is_valid(10, 10)); // K11 invalid
    }

    #[test]
    fn test_offsets() {
        // cell_index(K, 9) = 84 (last cell)
        assert_eq!(cell_index(10, 9), 84);
        // cell_index(A, 1) = 0 (first cell)
        assert_eq!(cell_index(0, 1), 0);
        // cell_index(B, 0) = 4
        assert_eq!(cell_index(1, 0), 4);
    }

    #[test]
    fn test_center_has_6_neighbors() {
        // E5 = (4, 4), a central cell
        assert_eq!(neighbors(4, 4).count(), 6);
    }

    #[test]
    fn test_corner_has_fewer_neighbors() {
        // A2 = (0, 1), corner
        let n: Vec<_> = neighbors(0, 1).collect();
        assert!(n.len() < 6);
        assert!(!n.is_empty());
    }
}
