/// Symmetry group trait for board games.
///
/// A symmetry maps board positions and moves such that game rules are invariant
/// under the mapping. Used for data augmentation in AlphaZero-style training.
///
/// The group must satisfy:
/// - `identity()` is the neutral element
/// - `compose(a, b)` applies b then a (i.e. a ∘ b)
/// - `inverse()` satisfies `compose(s, s.inverse()) == identity()`
/// - `all()` returns every element exactly once

use rand::seq::IndexedRandom;
use rand::Rng;

pub trait Symmetry: 'static + Default + Copy + Clone + Eq + std::fmt::Debug + Send + Sync {
    /// All elements of the symmetry group.
    fn all() -> &'static [Self];

    /// Order of the group (number of elements).
    fn order() -> usize {
        Self::all().len()
    }

    /// The identity element (no transformation). Same as `Default::default()`.
    fn identity() -> Self {
        Self::default()
    }

    /// The inverse of this element: `compose(self, self.inverse()) == identity()`.
    fn inverse(self) -> Self;

    /// Compose two symmetries: apply `other` first, then `self`.
    fn compose(self, other: Self) -> Self;

    /// Pick a uniformly random element.
    fn random<R: Rng>(rng: &mut R) -> Self {
        *Self::all().choose(rng).expect("symmetry group cannot be empty")
    }

    /// Whether this is the trivial group (only the identity).
    fn is_trivial() -> bool {
        Self::all().len() == 1
    }
}

/// The trivial symmetry group with only the identity element.
/// Use this for games with no useful symmetries.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct UnitSymmetry;

impl Symmetry for UnitSymmetry {
    fn all() -> &'static [Self] {
        &[Self]
    }

    fn inverse(self) -> Self {
        Self
    }

    fn compose(self, _other: Self) -> Self {
        Self
    }
}

/// Element of the D6 dihedral group — the symmetry group of a regular hexagon.
///
/// 12 elements: 6 rotations (0°, 60°, 120°, 180°, 240°, 300°) and 6 reflections.
/// Represented as an optional mirror followed by a rotation (0–5 steps of 60° clockwise).
///
/// Convention: `mirror` is applied first (reflection across the E–W axis in flat-top
/// axial coordinates), then `rotation` steps of 60° clockwise.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct D6Symmetry {
    /// Whether to mirror (reflect) before rotating.
    pub mirror: bool,
    /// Number of 60° clockwise rotation steps (0–5).
    pub rotation: u8,
}

impl D6Symmetry {
    pub const fn new(mirror: bool, rotation: u8) -> Self {
        D6Symmetry { mirror, rotation: rotation % 6 }
    }

    /// Construct from a flat index 0–11.
    /// 0–5: pure rotations, 6–11: mirror + rotation.
    pub const fn from_index(index: u8) -> Self {
        D6Symmetry {
            mirror: index >= 6,
            rotation: index % 6,
        }
    }

    /// Convert to a flat index 0–11.
    pub const fn to_index(self) -> u8 {
        if self.mirror { 6 + self.rotation } else { self.rotation }
    }

    /// Transform a hex direction index (0–5) by this symmetry.
    ///
    /// Direction order: E=0, NE=1, NW=2, W=3, SW=4, SE=5.
    /// Mirror reflects across E–W axis, then rotation adds steps mod 6.
    #[inline]
    pub fn transform_dir(self, dir: usize) -> usize {
        let d = if self.mirror {
            (6 - dir) % 6
        } else {
            dir
        };
        (d + self.rotation as usize) % 6
    }

    /// Transform an axial hex coordinate (q, r) by this symmetry.
    ///
    /// Mirror reflects across the q axis (negates r and adjusts),
    /// then rotation applies 60° clockwise steps in axial coords.
    pub fn transform_hex(self, q: i32, r: i32) -> (i32, i32) {
        // First apply mirror if needed (reflect across E-W / q axis).
        let (mut q2, mut r2) = if self.mirror {
            // Mirror: (q, r) → (q + r, -r)
            (q + r, -r)
        } else {
            (q, r)
        };

        // Then apply rotation: each 60° CW step in axial coords.
        // The six rotations cycle through:
        //   0: (q, r)
        //   1: (-r, q+r)
        //   2: (-q-r, q)
        //   3: (-q, -r)
        //   4: (r, -q-r)
        //   5: (q+r, -q)
        for _ in 0..self.rotation {
            let new_q = -r2;
            let new_r = q2 + r2;
            q2 = new_q;
            r2 = new_r;
        }

        (q2, r2)
    }
}

/// All 12 D6 elements, precomputed.
const D6_ALL: [D6Symmetry; 12] = [
    D6Symmetry::new(false, 0),
    D6Symmetry::new(false, 1),
    D6Symmetry::new(false, 2),
    D6Symmetry::new(false, 3),
    D6Symmetry::new(false, 4),
    D6Symmetry::new(false, 5),
    D6Symmetry::new(true, 0),
    D6Symmetry::new(true, 1),
    D6Symmetry::new(true, 2),
    D6Symmetry::new(true, 3),
    D6Symmetry::new(true, 4),
    D6Symmetry::new(true, 5),
];

impl Symmetry for D6Symmetry {
    fn all() -> &'static [Self] {
        &D6_ALL
    }

    fn inverse(self) -> Self {
        if self.mirror {
            // Mirror is self-inverse, but rotation order changes:
            // (mirror then rotate R) inverse = (rotate R then mirror) = mirror then rotate R
            // For mirror ∘ rot(r), inverse is rot(-r) ∘ mirror = mirror ∘ rot(r)
            // since mirror ∘ rot(r) ∘ mirror = rot(-r).
            // So inverse of (mirror, r) is (mirror, r).
            self
        } else {
            // Pure rotation: inverse is rotate by (6 - r) mod 6.
            D6Symmetry::new(false, (6 - self.rotation) % 6)
        }
    }

    fn compose(self, other: Self) -> Self {
        // self ∘ other: apply other first, then self.
        // Representation: each element is mirror^m ∘ rot(r).
        //
        // Key identity: rot(a) ∘ mirror = mirror ∘ rot(-a)
        // (because mirror conjugates rotation to its inverse).
        //
        // Case 1: self=(false, a), other=(false, b) → (false, a+b)
        // Case 2: self=(false, a), other=(true, b) → (true, a+b)
        // Case 3: self=(true, a), other=(false, b) → (true, a-b)
        // Case 4: self=(true, a), other=(true, b) → (false, a-b)
        let mirror = self.mirror ^ other.mirror;
        let rotation = if self.mirror {
            (self.rotation + 6 - other.rotation) % 6
        } else {
            (self.rotation + other.rotation) % 6
        };
        D6Symmetry::new(mirror, rotation)
    }
}

/// Apply a D6 symmetry to a `(num_channels, grid_size, grid_size)` f32 tensor in-place.
///
/// Convention: `sym.transform_hex` maps input coords → output coords, matching
/// `d6_grid_permutations` in the Python bindings. Out-of-bounds output cells are 0.
pub fn apply_d6_sym_spatial(tensor: &mut [f32], sym: D6Symmetry, num_channels: usize, grid_size: usize) {
    if sym == D6Symmetry::identity() {
        return;
    }
    let center = grid_size as i32 / 2;
    let gs = grid_size;
    let num_cells = gs * gs;
    debug_assert_eq!(tensor.len(), num_channels * num_cells);
    let mut out = vec![0f32; tensor.len()];
    for in_row in 0..gs {
        for in_col in 0..gs {
            let q = in_col as i32 - center;
            let r = in_row as i32 - center;
            let (q2, r2) = sym.transform_hex(q, r);
            let out_row = r2 + center;
            let out_col = q2 + center;
            if out_row >= 0 && out_row < gs as i32 && out_col >= 0 && out_col < gs as i32 {
                let in_cell = in_row * gs + in_col;
                let out_cell = out_row as usize * gs + out_col as usize;
                for c in 0..num_channels {
                    out[c * num_cells + out_cell] = tensor[c * num_cells + in_cell];
                }
            }
        }
    }
    tensor.copy_from_slice(&out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_symmetry() {
        assert_eq!(UnitSymmetry::all().len(), 1);
        assert!(UnitSymmetry::is_trivial());
    }

    #[test]
    fn test_d6_order() {
        assert_eq!(D6Symmetry::order(), 12);
        assert!(!D6Symmetry::is_trivial());
    }

    #[test]
    fn test_d6_identity() {
        let id = D6Symmetry::identity();
        assert_eq!(id, D6Symmetry::new(false, 0));
        assert_eq!(id.transform_hex(3, -1), (3, -1));
    }

    #[test]
    fn test_d6_index_roundtrip() {
        for i in 0..12u8 {
            let sym = D6Symmetry::from_index(i);
            assert_eq!(sym.to_index(), i);
        }
    }

    #[test]
    fn test_d6_inverse() {
        // s ∘ s⁻¹ = identity for all elements
        let id = D6Symmetry::identity();
        for &s in D6Symmetry::all() {
            assert_eq!(s.compose(s.inverse()), id, "s={:?}", s);
            assert_eq!(s.inverse().compose(s), id, "s={:?}", s);
        }
    }

    #[test]
    fn test_d6_closure() {
        // Composing any two elements yields another element in the group.
        let all: Vec<D6Symmetry> = D6Symmetry::all().to_vec();
        for &a in &all {
            for &b in &all {
                let c = a.compose(b);
                assert!(all.contains(&c), "a={:?} b={:?} c={:?}", a, b, c);
            }
        }
    }

    #[test]
    fn test_d6_associativity() {
        for &a in D6Symmetry::all() {
            for &b in D6Symmetry::all() {
                for &c in D6Symmetry::all() {
                    assert_eq!(
                        a.compose(b.compose(c)),
                        a.compose(b).compose(c),
                        "a={:?} b={:?} c={:?}", a, b, c
                    );
                }
            }
        }
    }

    #[test]
    fn test_d6_transform_hex_rotations() {
        // Full 360° rotation returns to start.
        let (q, r) = (2, -1);
        let rot6 = D6Symmetry::new(false, 0); // identity after 6 steps
        assert_eq!(rot6.transform_hex(q, r), (q, r));

        // Apply rot(1) six times manually.
        let rot1 = D6Symmetry::new(false, 1);
        let (mut cq, mut cr) = (q, r);
        for _ in 0..6 {
            let (nq, nr) = rot1.transform_hex(cq, cr);
            cq = nq;
            cr = nr;
        }
        assert_eq!((cq, cr), (q, r));
    }

    #[test]
    fn test_d6_mirror_involution() {
        // Mirror applied twice = identity.
        let m = D6Symmetry::new(true, 0);
        let (q, r) = (3, -2);
        let (q2, r2) = m.transform_hex(q, r);
        let (q3, r3) = m.transform_hex(q2, r2);
        assert_eq!((q3, r3), (q, r));
    }

    #[test]
    fn test_d6_transform_consistent_with_compose() {
        // transform(s1 ∘ s2, p) == transform(s1, transform(s2, p))
        let (q, r) = (2, -3);
        for &s1 in D6Symmetry::all() {
            for &s2 in D6Symmetry::all() {
                let composed = s1.compose(s2);
                let direct = composed.transform_hex(q, r);

                let (iq, ir) = s2.transform_hex(q, r);
                let chained = s1.transform_hex(iq, ir);

                assert_eq!(direct, chained, "s1={:?} s2={:?}", s1, s2);
            }
        }
    }

    #[test]
    fn test_d6_origin_invariant() {
        // Origin should be fixed under all symmetries.
        for &s in D6Symmetry::all() {
            assert_eq!(s.transform_hex(0, 0), (0, 0), "s={:?}", s);
        }
    }

    #[test]
    fn test_apply_d6_sym_spatial_roundtrip() {
        // Applying sym then sym.inverse() should recover the original tensor for
        // hexagonally-valid cells (cells within the hex radius stay in bounds under
        // all D6 symmetries, since hex symmetries preserve distance from origin).
        let grid_size = 7usize;  // radius 3 — all 37 valid hex cells within radius 3
        let radius = (grid_size / 2) as i32;
        let center = radius;
        let num_channels = 3usize;
        let n = num_channels * grid_size * grid_size;

        // Build a tensor with non-zero values only in hexagonally-valid cells.
        let mut original = vec![0f32; n];
        let mut val = 1.0f32;
        for r in 0..grid_size {
            for c in 0..grid_size {
                let q = c as i32 - center;
                let rr = r as i32 - center;
                // Valid hex cell: |q| ≤ radius, |r| ≤ radius, |q+r| ≤ radius
                if q.abs() <= radius && rr.abs() <= radius && (q + rr).abs() <= radius {
                    for ch in 0..num_channels {
                        original[ch * grid_size * grid_size + r * grid_size + c] = val;
                        val += 1.0;
                    }
                }
            }
        }

        for &sym in D6Symmetry::all() {
            let mut tensor = original.clone();
            super::apply_d6_sym_spatial(&mut tensor, sym, num_channels, grid_size);
            super::apply_d6_sym_spatial(&mut tensor, sym.inverse(), num_channels, grid_size);
            for (i, (&a, &b)) in original.iter().zip(tensor.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "sym={:?} i={} a={} b={}", sym, i, a, b);
            }
        }
    }
}
