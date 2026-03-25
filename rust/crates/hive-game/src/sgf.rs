/// Boardspace SGF file parser — extracts UHP move strings from Hive SGF files.
///
/// Port of hive/sgf.py to Rust.

use crate::game::{Game, Move};
use crate::hex::Hex;
use crate::piece::{Piece, PieceColor, PieceType};

/// Check if content contains expansion pieces (M=Mosquito, L=Ladybug, P=Pillbug).
fn has_expansion_pieces(content: &str) -> bool {
    // Match \b[wb][MLP]\d*\b — piece names with expansion types
    let bytes = content.as_bytes();
    for i in 0..bytes.len().saturating_sub(1) {
        let c0 = bytes[i];
        let c1 = bytes[i + 1];
        if (c0 == b'w' || c0 == b'b') && (c1 == b'M' || c1 == b'L' || c1 == b'P') {
            // Check word boundary before
            if i > 0 && bytes[i - 1].is_ascii_alphanumeric() {
                continue;
            }
            // Check what follows: optional digit(s) then word boundary
            let mut j = i + 2;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j >= bytes.len() || !bytes[j].is_ascii_alphanumeric() {
                return true;
            }
        }
    }
    false
}

/// Return "base" or "expansion" for the game type.
pub fn game_type(content: &str) -> &'static str {
    // Check SU field first
    if let Some(start) = content.find("SU[") {
        let rest = &content[start + 3..];
        if let Some(end) = rest.find(']') {
            let su = &rest[..end].to_lowercase();
            if su.contains("hive-") {
                return "expansion";
            }
        }
    }
    if has_expansion_pieces(content) {
        return "expansion";
    }
    "base"
}

/// Extract a field value like RE[...] or P0[id "..."] from SGF content.
pub fn extract_field(content: &str, prefix: &str) -> Option<String> {
    let start = content.find(prefix)?;
    let rest = &content[start + prefix.len()..];
    let end = rest.find(']')?;
    Some(rest[..end].to_string())
}

/// Extract player name from P0[id "name"] or P1[id "name"].
pub fn extract_player(content: &str, player_idx: u8) -> String {
    let prefix = format!("P{}[id \"", player_idx);
    if let Some(start) = content.find(&prefix) {
        let rest = &content[start + prefix.len()..];
        if let Some(end) = rest.find('"') {
            return rest[..end].to_string();
        }
    }
    if player_idx == 0 {
        "White".to_string()
    } else {
        "Black".to_string()
    }
}


// ---------------------------------------------------------------------------
// Direct SGF → Game replay (no UHP string intermediary)
// ---------------------------------------------------------------------------

/// Parse a piece name from SGF (with optional color prefix, case-insensitive type).
fn parse_piece_sgf(s: &str, default_color: PieceColor) -> Option<Piece> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let (color, type_start) = if bytes.len() >= 2
        && (bytes[0] == b'w' || bytes[0] == b'b')
        && (bytes[1] as char).is_ascii_alphabetic()
    {
        let c = if bytes[0] == b'w' { PieceColor::White } else { PieceColor::Black };
        (c, 1)
    } else {
        (default_color, 0)
    };
    if type_start >= bytes.len() {
        return None;
    }
    let type_char = (bytes[type_start] as char).to_ascii_uppercase();
    let piece_type = PieceType::from_char(type_char)?;
    let number = if type_start + 1 < bytes.len() && bytes[type_start + 1].is_ascii_digit() {
        bytes[type_start + 1] - b'0'
    } else {
        1
    };
    Some(Piece::new(color, piece_type, number))
}

/// Convert Boardspace column string to index (A=0, B=1, ...).
#[inline]
fn col_to_index(s: &str) -> i32 {
    (s.as_bytes()[0].to_ascii_uppercase() - b'A') as i32
}

/// Convert Boardspace grid coords to hex, given the origin of the first move.
#[inline]
fn boardspace_to_hex(col: i32, row: i32, origin_col: i32, origin_row: i32) -> Hex {
    let q = (col - origin_col) as i8;
    let r = -(row - origin_row) as i8;
    (q, r)
}

/// Case-insensitive prefix check without allocation.
fn starts_with_ci(s: &str, prefix: &[u8]) -> bool {
    let b = s.as_bytes();
    b.len() >= prefix.len()
        && b[..prefix.len()]
            .iter()
            .zip(prefix)
            .all(|(a, p)| a.to_ascii_lowercase() == *p)
}

/// Build a Move from a piece and Boardspace destination coords, using the live game state.
fn make_move(game: &Game, piece: Piece, col: i32, row: i32, origin: &mut Option<(i32, i32)>) -> Move {
    let hex = if let Some((oc, or)) = *origin {
        boardspace_to_hex(col, row, oc, or)
    } else {
        *origin = Some((col, row));
        (0i8, 0i8)
    };

    if let Some(from) = game.board.piece_position(piece) {
        Move::movement(piece, from, hex)
    } else {
        Move::placement(piece, hex)
    }
}

/// Replay SGF content directly into a Game, converting Boardspace coords to hex
/// without building UHP strings. Returns Ok(moves_played) on success.
pub fn replay_into_game(content: &str, game: &mut Game) -> Result<usize, String> {
    replay_into_game_inner(content, game, &mut |_, _| {})
}

/// Like `replay_into_game` but calls `on_move(game, &move)` before each move is played.
pub fn replay_into_game_verbose(
    content: &str,
    game: &mut Game,
    mut on_move: impl FnMut(&Game, &Move),
) -> Result<usize, String> {
    replay_into_game_inner(content, game, &mut on_move)
}

fn replay_into_game_inner(
    content: &str,
    game: &mut Game,
    on_move: &mut impl FnMut(&Game, &Move),
) -> Result<usize, String> {
    let mut origin: Option<(i32, i32)> = None;
    let mut move_count: usize = 0;
    // Pending move awaiting "done": (piece, dest_col, dest_row)
    let mut pending: Option<(Piece, i32, i32)> = None;

    let bytes = content.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] != b';' {
            i += 1;
            continue;
        }
        i += 1;

        // Skip whitespace
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }

        if i >= bytes.len() || bytes[i] != b'P' { continue; }
        i += 1;

        if i >= bytes.len() || (bytes[i] != b'0' && bytes[i] != b'1') { continue; }
        let player_color = if bytes[i] == b'0' { PieceColor::White } else { PieceColor::Black };
        i += 1;

        if i >= bytes.len() || bytes[i] != b'[' { continue; }
        i += 1;

        // Skip turn number
        while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
        while i < bytes.len() && bytes[i] == b' ' { i += 1; }

        // Read action until matching ']'
        let action_start = i;
        let mut bracket_depth = 1;
        while i < bytes.len() && bracket_depth > 0 {
            if bytes[i] == b'[' { bracket_depth += 1; }
            else if bytes[i] == b']' { bracket_depth -= 1; }
            if bracket_depth > 0 { i += 1; }
        }
        let action = content[action_start..i].trim();
        i += 1;

        // done / start → finalize pending move
        if starts_with_ci(action, b"done") || starts_with_ci(action, b"start") {
            if let Some((piece, dest_col, dest_row)) = pending.take() {
                if game.is_game_over() { break; }
                let mv = make_move(game, piece, dest_col, dest_row, &mut origin);
                on_move(game, &mv);
                game.play_move(&mv).map_err(|msg| {
                    format!("move {} rejected: {} ({:?})", move_count + 1, msg, piece)
                })?;
                move_count += 1;
            }
            continue;
        }

        // Pick from reserve — ignore
        if starts_with_ci(action, b"pick ") && !starts_with_ci(action, b"pickb") {
            continue;
        }

        // Pick from board — check for undo of pending move
        if starts_with_ci(action, b"pickb ") || starts_with_ci(action, b"pickb\t") {
            let mut parts = action.split_whitespace();
            parts.next(); // skip "Pickb"
            if let (Some(col_str), Some(row_str)) = (parts.next(), parts.next()) {
                let pick_col = col_to_index(col_str);
                let pick_row: i32 = row_str.parse().unwrap_or(0);
                if let Some(ref pm) = pending {
                    if pick_col == pm.1 && pick_row == pm.2 {
                        pending = None; // undo: player picked up piece they just dropped
                    }
                }
            }
            continue;
        }

        // Drop: "Dropb wS1 N 13 ."
        if starts_with_ci(action, b"dropb ") || starts_with_ci(action, b"dropb\t") {
            let mut parts = action.split_whitespace();
            parts.next(); // skip "Dropb"
            if let (Some(piece_str), Some(col_str), Some(row_str)) =
                (parts.next(), parts.next(), parts.next())
            {
                if let Some(piece) = parse_piece_sgf(piece_str, player_color) {
                    let dest_col = col_to_index(col_str);
                    let dest_row: i32 = row_str.parse().unwrap_or(0);
                    pending = Some((piece, dest_col, dest_row));
                }
            }
            continue;
        }

        // movedone (must check before "move")
        if starts_with_ci(action, b"movedone ") {
            let mut parts = action.split_whitespace();
            parts.next(); // skip "movedone"
            if let (Some(color_str), Some(piece_str), Some(col_str), Some(row_str)) =
                (parts.next(), parts.next(), parts.next(), parts.next())
            {
                let color = if color_str.as_bytes()[0].to_ascii_uppercase() == b'W' {
                    PieceColor::White
                } else {
                    PieceColor::Black
                };
                if let Some(piece) = parse_piece_sgf(piece_str, color) {
                    let dest_col = col_to_index(col_str);
                    let dest_row: i32 = row_str.parse().unwrap_or(0);
                    if game.is_game_over() { break; }
                    let mv = make_move(game, piece, dest_col, dest_row, &mut origin);
                    on_move(game, &mv);
                    game.play_move(&mv).map_err(|msg| {
                        format!("move {} rejected: {} ({:?})", move_count + 1, msg, piece)
                    })?;
                    move_count += 1;
                }
            }
            continue;
        }

        // Move: "Move B bS1 M 12 /wS1"
        if starts_with_ci(action, b"move ") {
            let mut parts = action.split_whitespace();
            parts.next(); // skip "Move"
            if let (Some(color_str), Some(piece_str), Some(col_str), Some(row_str)) =
                (parts.next(), parts.next(), parts.next(), parts.next())
            {
                let color = if color_str.as_bytes()[0].to_ascii_uppercase() == b'W' {
                    PieceColor::White
                } else {
                    PieceColor::Black
                };
                if let Some(piece) = parse_piece_sgf(piece_str, color) {
                    let dest_col = col_to_index(col_str);
                    let dest_row: i32 = row_str.parse().unwrap_or(0);
                    if game.is_game_over() { break; }
                    let mv = make_move(game, piece, dest_col, dest_row, &mut origin);
                    on_move(game, &mv);
                    game.play_move(&mv).map_err(|msg| {
                        format!("move {} rejected: {} ({:?})", move_count + 1, msg, piece)
                    })?;
                    move_count += 1;
                }
            }
            continue;
        }

        if action.eq_ignore_ascii_case("pass") {
            if game.is_game_over() { break; }
            let mv = Move::pass();
            on_move(game, &mv);
            game.play_pass();
            move_count += 1;
            continue;
        }
    }

    Ok(move_count)
}

/// Determine game result from SGF metadata (RE field, Resign, AcceptDraw).
/// Returns "p0_wins", "p1_wins", "draw", or "unknown".
pub fn result_from_metadata(content: &str, p0: &str, p1: &str) -> &'static str {
    // Newer format: explicit RE field containing winner's name
    if let Some(re_val) = extract_field(content, "RE[") {
        if re_val.contains(p0) {
            return "p0_wins";
        }
        if re_val.contains(p1) {
            return "p1_wins";
        }
        if re_val.to_lowercase().contains("draw") {
            return "draw";
        }
    }

    // Older format: resign action — the resigning player loses
    let bytes = content.as_bytes();
    for i in 0..bytes.len().saturating_sub(10) {
        if bytes[i] == b'P' && (bytes[i + 1] == b'0' || bytes[i + 1] == b'1') && bytes[i + 2] == b'[' {
            // Find the action text
            let rest = &content[i + 2..];
            if let Some(end) = rest.find(']') {
                let inner = &rest[1..end];
                let lower = inner.to_lowercase();
                if lower.contains(" resign") {
                    return if bytes[i + 1] == b'0' { "p1_wins" } else { "p0_wins" };
                }
                if lower.contains(" acceptdraw") {
                    return "draw";
                }
            }
        }
    }

    "unknown"
}

/// Check if the game was a timeout (one player ran out of time).
pub fn is_timeout(content: &str) -> bool {
    if let Some(re_val) = extract_field(content, "RE[") {
        let lower = re_val.to_lowercase();
        if lower.contains("time") || lower.contains("timeout") {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_type_base() {
        assert_eq!(game_type("SU[Hive]"), "base");
        assert_eq!(game_type("SU[hive]"), "base");
    }

    #[test]
    fn test_game_type_expansion() {
        assert_eq!(game_type("SU[Hive-ULP]"), "expansion");
        assert_eq!(game_type("some wM1 content"), "expansion");
    }

}
