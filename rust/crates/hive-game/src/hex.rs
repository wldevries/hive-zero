/// Axial hex coordinate system for Hive board.
/// Flat-top hexagon orientation, using (q, r) axial coordinates.

/// Six directions in axial coordinates (flat-top hexagons).
/// Order: E, NE, NW, W, SW, SE (clockwise from right)
pub const DIRECTIONS: [(i8, i8); 6] = [
    (1, 0),   // 0: E  (right)
    (1, -1),  // 1: NE (top-right)
    (0, -1),  // 2: NW (top-left)
    (-1, 0),  // 3: W  (left)
    (-1, 1),  // 4: SW (bottom-left)
    (0, 1),   // 5: SE (bottom-right)
];

/// Maps direction index to its opposite.
pub const OPPOSITE_DIR: [usize; 6] = [3, 4, 5, 0, 1, 2];

/// Hex coordinate as (q, r).
pub type Hex = (i8, i8);

pub const ORIGIN: Hex = (0, 0);

#[inline]
pub fn hex_add(a: Hex, b: Hex) -> Hex {
    (a.0 + b.0, a.1 + b.1)
}

#[inline]
pub fn hex_sub(a: Hex, b: Hex) -> Hex {
    (a.0 - b.0, a.1 - b.1)
}

#[inline]
pub fn hex_distance(a: Hex, b: Hex) -> i8 {
    let dq = (a.0 - b.0) as i16;
    let dr = (a.1 - b.1) as i16;
    ((dq.abs() + dr.abs() + (dq + dr).abs()) / 2) as i8
}

#[inline]
pub fn hex_neighbors(h: Hex) -> [Hex; 6] {
    [
        (h.0 + 1, h.1),
        (h.0 + 1, h.1 - 1),
        (h.0, h.1 - 1),
        (h.0 - 1, h.1),
        (h.0 - 1, h.1 + 1),
        (h.0, h.1 + 1),
    ]
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_distance() {
        assert_eq!(hex_distance((0, 0), (1, 0)), 1);
        assert_eq!(hex_distance((0, 0), (2, -1)), 2);
        assert_eq!(hex_distance((0, 0), (0, 0)), 0);
        assert_eq!(hex_distance((0, 0), (3, -3)), 3);
    }

    #[test]
    fn test_hex_neighbors() {
        let n = hex_neighbors((0, 0));
        assert_eq!(n.len(), 6);
        for (i, dir) in DIRECTIONS.iter().enumerate() {
            assert_eq!(n[i], *dir);
        }
    }

    #[test]
    fn test_opposite_dir() {
        for i in 0..6 {
            let opp = OPPOSITE_DIR[i];
            let d = DIRECTIONS[i];
            let o = DIRECTIONS[opp];
            assert_eq!(hex_add(d, o), (0, 0));
        }
    }
}
