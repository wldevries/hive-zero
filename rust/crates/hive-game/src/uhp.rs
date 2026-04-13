/// UHP (Universal Hive Protocol) move parsing and formatting.

use crate::game::{Game, Move};
use core_game::hex::{DIRECTIONS, hex_neighbors};
use crate::piece::Piece;

/// Parse a UHP move string and resolve the piece and destination hex.
/// Returns `None` if parsing fails (unknown piece, missing reference, etc.).
pub fn parse_uhp_move(game: &Game, move_str: &str) -> Option<Move> {
    let parts: Vec<&str> = move_str.split_whitespace().collect();
    if parts.is_empty() {
        return None;
    }

    let piece = Piece::from_str(parts[0])?;

    let to_pos = if parts.len() == 1 {
        // First overall move: place at origin.
        (0, 0)
    } else {
        let ref_str = parts[1];
        let bytes = ref_str.as_bytes();
        let first = bytes[0] as char;
        let last = bytes[bytes.len() - 1] as char;

        let is_dir_char = |c: char| matches!(c, '-' | '/' | '\\');

        if is_dir_char(first) {
            // Prefix notation: e.g. "/wQ" = SW of wQ
            let dir = match first {
                '-'  => DIRECTIONS[3], // W
                '/'  => DIRECTIONS[4], // SW
                '\\' => DIRECTIONS[2], // NW
                _    => return None,
            };
            let ref_piece = Piece::from_str(&ref_str[1..])?;
            let ref_pos = game.board.piece_position(ref_piece)?;
            (ref_pos.0 + dir.0, ref_pos.1 + dir.1)
        } else if is_dir_char(last) {
            // Suffix notation: e.g. "wQ-" = East of wQ
            let dir = match last {
                '-'  => DIRECTIONS[0], // E
                '/'  => DIRECTIONS[1], // NE
                '\\' => DIRECTIONS[5], // SE
                _    => return None,
            };
            let ref_piece = Piece::from_str(&ref_str[..ref_str.len() - 1])?;
            let ref_pos = game.board.piece_position(ref_piece)?;
            (ref_pos.0 + dir.0, ref_pos.1 + dir.1)
        } else {
            // No direction character: beetle stacking onto the reference piece.
            let ref_piece = Piece::from_str(ref_str)?;
            game.board.piece_position(ref_piece)?
        }
    };

    // Determine placement vs movement by checking if the piece is on the board.
    let from = game.board.piece_position(piece);
    Some(match from {
        None => Move::placement(piece, to_pos),
        Some(f) => Move::movement(piece, f, to_pos),
    })
}

/// Format a move as UHP MoveString in the context of the given game state.
pub fn format_move_uhp(game: &Game, mv: &Move) -> String {
    if mv.is_pass() {
        return "pass".to_string();
    }
    let piece = mv.piece.unwrap();
    let piece_str = piece.to_uhp_string();
    let to_pos = mv.to.unwrap();

    // First overall move: just piece name
    if game.move_count == 0 && mv.from.is_none() {
        return piece_str;
    }

    // Stacking (beetle on top of another piece)
    if mv.from.is_some() {
        if let Some(top) = game.board.top_piece(to_pos) {
            return format!("{} {}", piece_str, top.to_uhp_string());
        }
    }

    // Find an adjacent occupied hex to use as reference.
    // Maps direction index (target relative to reference) to UHP notation:
    //   UHP suffix '-' = E, suffix '/' = NE, suffix '\' = SE
    //   UHP prefix '-' = W, prefix '/' = SW, prefix '\' = NW
    let dir_to_uhp: [(char, bool); 6] = [
        ('-', false),   // 0: E  -> suffix '-'
        ('/', false),   // 1: NE -> suffix '/'
        ('\\', true),   // 2: NW -> prefix '\'
        ('-', true),    // 3: W  -> prefix '-'
        ('/', true),    // 4: SW -> prefix '/'
        ('\\', false),  // 5: SE -> suffix '\'
    ];

    let neighbors = hex_neighbors(to_pos);
    for (i, &neighbor) in neighbors.iter().enumerate() {
        if let Some(top) = game.board.top_piece(neighbor) {
            if top.to_uhp_string() != piece_str || mv.from == Some(neighbor) {
                let opp = (i + 3) % 6;
                let (ch, is_prefix) = dir_to_uhp[opp];
                let ref_str = top.to_uhp_string();
                if is_prefix {
                    return format!("{} {}{}", piece_str, ch, ref_str);
                } else {
                    return format!("{} {}{}", piece_str, ref_str, ch);
                }
            }
        }
    }

    // Fallback: use any neighbor including the piece itself
    for (i, &neighbor) in neighbors.iter().enumerate() {
        if let Some(top) = game.board.top_piece(neighbor) {
            let opp = (i + 3) % 6;
            let (ch, is_prefix) = dir_to_uhp[opp];
            let ref_str = top.to_uhp_string();
            if is_prefix {
                return format!("{} {}{}", piece_str, ch, ref_str);
            } else {
                return format!("{} {}{}", piece_str, ref_str, ch);
            }
        }
    }

    piece_str
}

/// Transform the directional part of a UHP move string by a D6 symmetry.
///
/// UHP direction characters map to direction indices:
///   suffix '-' = E(0), '/' = NE(1), '\\' = NW(2)
///   prefix '-' = W(3), '/' = SW(4), '\\' = SE(5)
///
/// Moves without a direction (first move, beetle stacking, pass) are unchanged.
pub fn transform_uhp_move(move_str: &str, sym: core_game::symmetry::D6Symmetry) -> String {
    if sym == core_game::symmetry::D6Symmetry::default() {
        return move_str.to_string();
    }

    let parts: Vec<&str> = move_str.split_whitespace().collect();
    if parts.len() != 2 {
        // First move (just piece name), pass, or unparseable — return as-is
        return move_str.to_string();
    }

    let ref_str = parts[1];
    let bytes = ref_str.as_bytes();
    let first = bytes[0] as char;
    let last = bytes[bytes.len() - 1] as char;

    let is_dir_char = |c: char| matches!(c, '-' | '/' | '\\');

    // Must match format_move_uhp's dir_to_uhp table:
    //   0:E=suffix'-', 1:NE=suffix'/', 2:NW=prefix'\\',
    //   3:W=prefix'-', 4:SW=prefix'/', 5:SE=suffix'\\'
    let dir_char: [char; 6] = ['-', '/', '\\', '-', '/', '\\'];
    let dir_is_prefix: [bool; 6] = [false, false, true, true, true, false];

    let (orig_dir, ref_piece_str) = if is_dir_char(first) {
        let dir = match first {
            '-'  => 3, // W  (prefix '-')
            '/'  => 4, // SW (prefix '/')
            '\\' => 2, // NW (prefix '\\')
            _    => return move_str.to_string(),
        };
        (dir, &ref_str[1..])
    } else if is_dir_char(last) {
        let dir = match last {
            '-'  => 0, // E  (suffix '-')
            '/'  => 1, // NE (suffix '/')
            '\\' => 5, // SE (suffix '\\')
            _    => return move_str.to_string(),
        };
        (dir, &ref_str[..ref_str.len() - 1])
    } else {
        // Beetle stacking — no direction to transform
        return move_str.to_string();
    };

    let new_dir = sym.transform_dir(orig_dir);
    let ch = dir_char[new_dir];
    if dir_is_prefix[new_dir] {
        format!("{} {}{}", parts[0], ch, ref_piece_str)
    } else {
        format!("{} {}{}", parts[0], ref_piece_str, ch)
    }
}

/// Parse a UHP move string and play it, validating against legal moves.
/// Returns true if the move was valid and played.
pub fn parse_and_play_uhp(game: &mut Game, move_str: &str) -> bool {
    if move_str.eq_ignore_ascii_case("pass") {
        game.play_pass();
        return true;
    }

    let mv = match parse_uhp_move(game, move_str) {
        Some(m) => m,
        None => return false,
    };

    let valid = game.valid_moves();
    if valid.iter().any(|m| m.piece == mv.piece && m.to == mv.to) {
        game.play_move(&mv).unwrap();
        true
    } else {
        false
    }
}

/// Play a UHP move string without checking legality.
/// Use for trusted replay — skips expensive valid_moves() generation.
/// Returns Err with a message if the move string cannot be parsed.
pub fn play_uhp_unchecked(game: &mut Game, move_str: &str) -> Result<(), String> {
    if move_str.eq_ignore_ascii_case("pass") {
        game.play_pass();
        return Ok(());
    }

    let mv = parse_uhp_move(game, move_str)
        .ok_or_else(|| format!("cannot parse UHP move: {}", move_str))?;
    game.play_move(&mv)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_game::symmetry::D6Symmetry;

    fn sym(index: u8) -> D6Symmetry {
        D6Symmetry::from_index(index)
    }

    #[test]
    fn test_transform_identity() {
        assert_eq!(transform_uhp_move("wS1 wQ1-", sym(0)), "wS1 wQ1-");
        assert_eq!(transform_uhp_move("wQ1", sym(0)), "wQ1");
        assert_eq!(transform_uhp_move("wB1 wQ1", sym(0)), "wB1 wQ1"); // beetle stack
    }

    #[test]
    fn test_transform_rotate_60() {
        // E(0) -> NE(1): suffix '-' -> suffix '/'
        assert_eq!(transform_uhp_move("wS1 wQ1-", sym(1)), "wS1 wQ1/");
        // NE(1) -> NW(2): suffix '/' -> prefix '\\'
        assert_eq!(transform_uhp_move("wS1 wQ1/", sym(1)), "wS1 \\wQ1");
        // W(3) -> SW(4): prefix '-' -> prefix '/'
        assert_eq!(transform_uhp_move("wS1 -wQ1", sym(1)), "wS1 /wQ1");
        // SE(5) -> E(0): suffix '\\' -> suffix '-'
        assert_eq!(transform_uhp_move("wS1 wQ1\\", sym(1)), "wS1 wQ1-");
    }

    #[test]
    fn test_transform_rotate_180() {
        // E(0) -> W(3): suffix '-' -> prefix '-'
        assert_eq!(transform_uhp_move("wS1 wQ1-", sym(3)), "wS1 -wQ1");
        // NE(1) -> SW(4): suffix '/' -> prefix '/'
        assert_eq!(transform_uhp_move("wS1 wQ1/", sym(3)), "wS1 /wQ1");
    }

    #[test]
    fn test_transform_mirror() {
        // Mirror: E(0)->E(0), NE(1)->SE(5), NW(2)->SW(4), W(3)->W(3), SW(4)->NW(2), SE(5)->NE(1)
        assert_eq!(transform_uhp_move("wS1 wQ1-", sym(6)), "wS1 wQ1-");   // E stays E
        assert_eq!(transform_uhp_move("wS1 wQ1/", sym(6)), "wS1 wQ1\\");  // NE(1)->SE(5): suffix'/'->suffix'\\'
        assert_eq!(transform_uhp_move("wS1 \\wQ1", sym(6)), "wS1 /wQ1");  // NW(2)->SW(4): prefix'\\'->prefix'/'
        assert_eq!(transform_uhp_move("wS1 -wQ1", sym(6)), "wS1 -wQ1");   // W stays W
        assert_eq!(transform_uhp_move("wS1 /wQ1", sym(6)), "wS1 \\wQ1");  // SW(4)->NW(2): prefix'/'->prefix'\\'
        assert_eq!(transform_uhp_move("wS1 wQ1\\", sym(6)), "wS1 wQ1/");  // SE(5)->NE(1): suffix'\\'->suffix'/'
    }

    #[test]
    fn test_transform_no_direction() {
        // First move, pass, beetle stacking — all unchanged
        assert_eq!(transform_uhp_move("wQ1", sym(5)), "wQ1");
        assert_eq!(transform_uhp_move("pass", sym(5)), "pass");
        assert_eq!(transform_uhp_move("wB1 wQ1", sym(5)), "wB1 wQ1");
    }

    #[test]
    fn test_transform_roundtrip_12() {
        // All 12 transforms of the same move should be valid and distinct where expected
        let original = "wS1 wQ1-"; // E
        let results: Vec<String> = (0..12).map(|s| transform_uhp_move(original, sym(s))).collect();
        // sym 0 and sym 6 both map E->E
        assert_eq!(results[0], results[6]);
        // sym 3 maps E->W
        assert_eq!(results[3], "wS1 -wQ1");
    }
}
