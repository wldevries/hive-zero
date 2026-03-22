/// UHP (Universal Hive Protocol) move parsing and formatting.

use crate::game::{Game, Move};
use crate::hex::{DIRECTIONS, hex_neighbors};
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
        game.play_move(&mv);
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
    game.play_move(&mv);
    Ok(())
}
