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

/// Rotate a direction index (0-5) clockwise by `n` steps.
/// Direction order: E=0, NE=1, NW=2, W=3, SW=4, SE=5.
#[inline]
pub fn rotate_dir(dir: usize, n: u8) -> usize {
    (dir + n as usize) % 6
}

/// Mirror a direction index across the E-W axis.
/// E(0)↔E(0), NE(1)↔SE(5), NW(2)↔SW(4), W(3)↔W(3).
#[inline]
pub fn mirror_dir(dir: usize) -> usize {
    match dir {
        0 => 0, // E stays E
        1 => 5, // NE -> SE
        2 => 4, // NW -> SW
        3 => 3, // W stays W
        4 => 2, // SW -> NW
        5 => 1, // SE -> NE
        _ => dir,
    }
}

/// Apply D6 symmetry transform to a direction index.
/// sym 0-5: rotate by sym steps. sym 6-11: mirror then rotate by (sym-6) steps.
#[inline]
pub fn transform_dir(dir: usize, sym: u8) -> usize {
    if sym < 6 {
        rotate_dir(dir, sym)
    } else {
        rotate_dir(mirror_dir(dir), sym - 6)
    }
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
