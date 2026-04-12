/// Encode/decode Hive moves for the factorized (source, dest) neural network policy.
/// Must produce identical layout to Python move_encoder.py.
///
/// Policy layout — flat vector of size 11 * grid_size * grid_size:
///
///   [0 .. 5*G*G)       — placement head (piece_type × dest)
///       place_offset = type_idx * G*G + row*G + col
///
///   [5*G*G .. 6*G*G)   — movement source head (src hex)
///       src_offset = 5*G*G + row*G + col
///
///   [6*G*G .. 11*G*G)  — movement destination head (piece_type × dest)
///       dst_offset = 6*G*G + type_idx*G*G + row*G + col
///
/// MCTS prior computation:
///   Placement(type, dest):    prior = policy[place_offset]
///   Movement(src, dest):      prior = policy[src_offset] + policy[dst_offset]
///
/// Training targets (marginals stored in the flat policy vector):
///   Placement visits go to policy[place_offset].
///   Movement visits are split: policy[src_offset] += visits, policy[dst_offset] += visits.

use core_game::hex::Hex;
use crate::piece::Piece;
use crate::piece::PieceType;
use crate::game::{Game, Move};
use core_game::game::PolicyIndex;

/// Conceptual "channels" for policy_size = NUM_POLICY_CHANNELS * G * G:
///   5 placement channels + 1 src channel + 5 dst channels (piece_type × dest).
pub const NUM_POLICY_CHANNELS: usize = 11;
/// Number of placement head channels (one per piece type).
pub const NUM_PLACE_CHANNELS: usize = 5;

/// Compute total policy size for a given grid_size.
pub fn policy_size(grid_size: usize) -> usize {
    NUM_POLICY_CHANNELS * grid_size * grid_size
}

/// Offset of the placement head in the flat policy vector.
#[inline]
pub fn place_section_offset() -> usize { 0 }
/// Offset of the movement-source head.
#[inline]
pub fn src_section_offset(grid_size: usize) -> usize { NUM_PLACE_CHANNELS * grid_size * grid_size }
/// Offset of the movement-destination head.
#[inline]
pub fn dst_section_offset(grid_size: usize) -> usize { (NUM_PLACE_CHANNELS + 1) * grid_size * grid_size }

/// Map hex coordinates to encoding grid (shared with board encoding).
#[inline]
fn hex_to_grid(h: Hex, grid_size: usize) -> Option<(usize, usize)> {
    let center = (grid_size / 2) as i16;
    let col = h.0 as i16 + center;
    let row = h.1 as i16 + center;
    if col >= 0 && col < grid_size as i16 && row >= 0 && row < grid_size as i16 {
        Some((row as usize, col as usize))
    } else {
        None
    }
}

#[inline]
fn cell(row: usize, col: usize, grid_size: usize) -> usize {
    row * grid_size + col
}

#[inline]
fn base_piece_type_idx(piece_type: PieceType) -> Option<usize> {
    match piece_type {
        PieceType::Queen => Some(0),
        PieceType::Spider => Some(1),
        PieceType::Beetle => Some(2),
        PieceType::Grasshopper => Some(3),
        PieceType::Ant => Some(4),
        PieceType::Mosquito | PieceType::Ladybug | PieceType::Pillbug => None,
    }
}

/// Encode a placement move (piece_type, dest) as a Single PolicyIndex.
/// Returns None if dest is out of grid bounds.
pub fn encode_placement(piece: Piece, dest: Hex, grid_size: usize) -> Option<PolicyIndex> {
    let (row, col) = hex_to_grid(dest, grid_size)?;
    let type_idx = base_piece_type_idx(piece.piece_type())?;
    Some(PolicyIndex::Single(type_idx * grid_size * grid_size + cell(row, col, grid_size)))
}

/// Encode a movement move (src, piece, dest) as a Sum PolicyIndex.
/// The dst index is piece-type-conditioned: dst_offset + type_idx * G² + cell.
/// Returns None if either hex is out of grid bounds.
pub fn encode_movement(src: Hex, piece: Piece, dest: Hex, grid_size: usize) -> Option<PolicyIndex> {
    let (sr, sc) = hex_to_grid(src, grid_size)?;
    let (dr, dc) = hex_to_grid(dest, grid_size)?;
    let src_idx = src_section_offset(grid_size) + cell(sr, sc, grid_size);
    let type_idx = base_piece_type_idx(piece.piece_type())?;
    let dst_idx = dst_section_offset(grid_size) + type_idx * grid_size * grid_size + cell(dr, dc, grid_size);
    Some(PolicyIndex::Sum(src_idx, dst_idx))
}

/// Encode any Move struct as a PolicyIndex. Returns None for pass or out-of-grid.
pub fn encode_game_move(mv: &Move, grid_size: usize) -> Option<PolicyIndex> {
    let piece = mv.piece?;
    let dest = mv.to?;
    if mv.from.is_none() {
        // Placement
        encode_placement(piece, dest, grid_size)
    } else {
        // Movement — piece type conditions the destination channel
        encode_movement(mv.from.unwrap(), piece, dest, grid_size)
    }
}

/// Legacy flat-index encode for a placement (for ONNX / Python compat).
/// Returns the flat index in `[0 .. 5*G*G)`.
pub fn encode_placement_flat(piece: Piece, dest: Hex, grid_size: usize) -> Option<usize> {
    match encode_placement(piece, dest, grid_size) {
        Some(PolicyIndex::Single(idx)) => Some(idx),
        _ => None,
    }
}

/// Create a binary mask over the policy space for legal moves.
///
/// Returns `(mask, indexed_moves)` where:
/// - `mask` has 1.0 at every policy position touched by a legal move.
/// - `indexed_moves` has one entry per unique network action (placements with the
///   same piece-type and dest are deduplicated; movements are always unique since
///   each board source hex holds at most one piece).
pub fn get_legal_move_mask(game: &mut Game, grid_size: usize) -> (Vec<f32>, Vec<(PolicyIndex, Move)>) {
    let ps = policy_size(grid_size);
    let mut mask = vec![0.0f32; ps];
    let valid_moves = game.valid_moves();
    let mut indexed_moves = Vec::with_capacity(valid_moves.len());

    for mv in valid_moves {
        match encode_game_move(&mv, grid_size) {
            Some(PolicyIndex::Single(idx)) if idx < ps => {
                // Placement — deduplicate: skip if same (type, dest) already seen
                if mask[idx] == 0.0 {
                    mask[idx] = 1.0;
                    indexed_moves.push((PolicyIndex::Single(idx), mv));
                }
            }
            Some(PolicyIndex::Sum(a, b)) if a < ps && b < ps => {
                // Movement — src hex is unique per piece, no dedup needed
                mask[a] = 1.0;
                mask[b] = 1.0;
                indexed_moves.push((PolicyIndex::Sum(a, b), mv));
            }
            _ => {}
        }
    }

    (mask, indexed_moves)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::GRID_SIZE;
    use crate::piece::{PieceColor, PieceType};

    const GS: usize = GRID_SIZE;

    #[test]
    fn test_encode_placement_single() {
        let piece = Piece::new(PieceColor::White, PieceType::Queen, 1);
        // Queen type_idx=0, center=(11,11)
        match encode_placement(piece, (0, 0), GS) {
            Some(PolicyIndex::Single(idx)) => {
                assert_eq!(idx, 0 * GS * GS + 11 * GS + 11); // type 0 at center
            }
            other => panic!("expected Single, got {:?}", other),
        }
    }

    #[test]
    fn test_encode_movement_sum() {
        // Movement ant (1,0) -> (0,0): src=(11,12), dst=(11,11)
        // Ant type_idx = 4
        let ant = Piece::new(PieceColor::White, PieceType::Ant, 1);
        match encode_movement((1, 0), ant, (0, 0), GS) {
            Some(PolicyIndex::Sum(a, b)) => {
                let src_off = src_section_offset(GS);
                let dst_off = dst_section_offset(GS);
                assert_eq!(a, src_off + 11 * GS + 12); // (1,0) -> row11, col12
                // Ant type_idx=4, dest=(0,0) -> center (row11, col11)
                assert_eq!(b, dst_off + 4 * GS * GS + 11 * GS + 11);
            }
            other => panic!("expected Sum, got {:?}", other),
        }
    }

    #[test]
    fn test_same_type_placements_deduplicate() {
        // Ant1 and Ant2 placing at same dest encode identically → same SingleIndex
        let ant1 = Piece::new(PieceColor::White, PieceType::Ant, 1);
        let ant2 = Piece::new(PieceColor::White, PieceType::Ant, 2);
        assert_eq!(
            encode_placement(ant1, (0, 0), GS),
            encode_placement(ant2, (0, 0), GS),
        );
    }

    #[test]
    fn test_different_type_placements_differ() {
        let ant = Piece::new(PieceColor::White, PieceType::Ant, 1);
        let gh  = Piece::new(PieceColor::White, PieceType::Grasshopper, 1);
        assert_ne!(
            encode_placement(ant, (0, 0), GS),
            encode_placement(gh, (0, 0), GS),
        );
    }

    #[test]
    fn test_movement_different_src_differ() {
        // Same dest and piece type, different src → different Sum indices
        let piece = Piece::new(PieceColor::White, PieceType::Ant, 1);
        let e1 = encode_movement((1, 0), piece, (0, 0), GS);
        let e2 = encode_movement((2, 0), piece, (0, 0), GS);
        assert_ne!(e1, e2);
    }

    #[test]
    fn test_policy_size() {
        assert_eq!(policy_size(23), NUM_POLICY_CHANNELS * 23 * 23);
    }

    #[test]
    fn test_expansion_piece_is_not_encoded_in_base_policy_layout() {
        let mos = Piece::new(PieceColor::White, PieceType::Mosquito, 1);
        assert_eq!(encode_placement(mos, (0, 0), GS), None);
        assert_eq!(encode_movement((1, 0), mos, (0, 0), GS), None);
    }

    #[test]
    fn test_section_offsets() {
        let gs = 17usize;
        assert_eq!(place_section_offset(), 0);
        assert_eq!(src_section_offset(gs), 5 * gs * gs);
        assert_eq!(dst_section_offset(gs), 6 * gs * gs);
    }
}
