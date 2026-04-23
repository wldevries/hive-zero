/// Boardspace SGF file parser for Hive games.
///
/// Boardspace uses an SGF-like format (not UHP) where each player action is
/// encoded as a `;P0[...]` or `;P1[...]` entry. This module handles the
/// Hive-specific action types (Drop, Move, Movdone, Pick, Pickb) and maps
/// boardspace grid coordinates to axial hex coordinates.

use crate::game::{Game, Move};
use core_game::hex::Hex;
use crate::piece::{Piece, PieceColor, PieceType};

pub use core_game::sgf::{
    extract_player, extract_prop, is_timeout, result_from_metadata, scan_player_actions,
};

// ---------------------------------------------------------------------------
// Hive-specific: expansion detection and game type
// ---------------------------------------------------------------------------

/// Check if content contains expansion pieces (M=Mosquito, L=Ladybug, P=Pillbug).
fn has_expansion_pieces(content: &str) -> bool {
    let bytes = content.as_bytes();
    for i in 0..bytes.len().saturating_sub(1) {
        let c0 = bytes[i];
        let c1 = bytes[i + 1];
        if (c0 == b'w' || c0 == b'b') && (c1 == b'M' || c1 == b'L' || c1 == b'P') {
            if i > 0 && bytes[i - 1].is_ascii_alphanumeric() {
                continue;
            }
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

/// Return `"base"` or `"expansion"` for the game type.
pub fn game_type(content: &str) -> &'static str {
    if let Some(su) = extract_prop(content, "SU") {
        if su.to_lowercase().contains("hive-") {
            return "expansion";
        }
    }
    if has_expansion_pieces(content) {
        return "expansion";
    }
    "base"
}

// ---------------------------------------------------------------------------
// Hive-specific: direct SGF → Game replay
// ---------------------------------------------------------------------------

/// Parse a piece name from an SGF action string (with optional color prefix).
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

/// Convert a boardspace column label to a 0-based index.
/// Single letters: 'A'=0 .. 'Z'=25.
/// Edge overflow uses the edge letter with a numeric prefix:
/// '1A'=-1, '2A'=-2, ... and '1Z'=26, '2Z'=27, ...
#[inline]
fn col_to_index(s: &str) -> i32 {
    let bytes = s.as_bytes();
    if bytes.len() >= 2 && bytes[..bytes.len() - 1].iter().all(u8::is_ascii_digit) {
        let n: i32 = s[..s.len() - 1].parse().unwrap_or(0);
        let letter = bytes[bytes.len() - 1].to_ascii_uppercase();
        match letter {
            b'A' => -n,
            b'Z' => 25 + n,
            _ => (letter - b'A') as i32,
        }
    } else {
        (bytes[0].to_ascii_uppercase() - b'A') as i32
    }
}

/// Convert boardspace grid coords to axial hex, relative to the first-move origin.
///
/// Boardspace column labels A–Z represent 0–25 and rows 1–26 each form a 26-value
/// cycle. When the hive grows past the edge of the display, coordinates wrap around
/// (e.g. one step east of Z=25 becomes A=0, one step above row=1 becomes row=26).
/// Disambiguate by picking the offset closest to the known origin.
#[inline]
fn boardspace_to_hex(col: i32, row: i32, origin_col: i32, origin_row: i32) -> Hex {
    // Only apply wrapping for single-letter columns (0–25); extended notation
    // (negative values from "1A" etc., or >25 from "1Z" etc.) is already unambiguous.
    let col = if col >= 0 && col <= 25 {
        let d0 = col - origin_col;
        let d1 = col + 26 - origin_col;
        let d2 = col - 26 - origin_col;
        if d1.abs() < d0.abs() && d1.abs() <= d2.abs() {
            col + 26
        } else if d2.abs() < d0.abs() {
            col - 26
        } else {
            col
        }
    } else {
        col
    };
    // Rows 1–26 also form a 26-value cycle; apply the same wrap disambiguation.
    let row = if row >= 1 && row <= 26 {
        let d0 = row - origin_row;
        let d1 = row + 26 - origin_row;
        let d2 = row - 26 - origin_row;
        if d1.abs() < d0.abs() && d1.abs() <= d2.abs() {
            row + 26
        } else if d2.abs() < d0.abs() {
            row - 26
        } else {
            row
        }
    } else {
        row
    };
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

/// Build a Move from a piece and boardspace destination coords.
///
/// `frame_offset` maps SGF coordinates (first-move-relative) into the current
/// game frame after any recentering shifts that were applied by previous moves.
fn make_move(
    game: &Game,
    piece: Piece,
    col: i32,
    row: i32,
    origin: &mut Option<(i32, i32)>,
    frame_offset: (i8, i8),
) -> Move {
    let sgf_hex = if let Some((oc, or)) = *origin {
        boardspace_to_hex(col, row, oc, or)
    } else {
        *origin = Some((col, row));
        (0i8, 0i8)
    };

    let hex = (
        sgf_hex.0.saturating_add(frame_offset.0),
        sgf_hex.1.saturating_add(frame_offset.1),
    );

    if let Some(from) = game.board.piece_position(piece) {
        Move::movement(piece, from, hex)
    } else {
        Move::placement(piece, hex)
    }
}

/// Replay SGF content directly into a Game. Returns `Ok(moves_played)` on success.
pub fn replay_into_game(content: &str, game: &mut Game) -> Result<usize, String> {
    replay_into_game_inner(content, game, &mut |_, _| {})
}

/// Like `replay_into_game` but calls `on_move(game, &move)` before each move is played.
pub fn replay_into_game_verbose(
    content: &str,
    game: &mut Game,
    mut on_move: impl FnMut(&mut Game, &Move),
) -> Result<usize, String> {
    replay_into_game_inner(content, game, &mut on_move)
}

fn replay_into_game_inner(
    content: &str,
    game: &mut Game,
    on_move: &mut impl FnMut(&mut Game, &Move),
) -> Result<usize, String> {
    let mut origin: Option<(i32, i32)> = None;
    let mut move_count: usize = 0;
    // Pending move awaiting "done": (piece, dest_col, dest_row)
    let mut pending: Option<(Piece, i32, i32)> = None;
    // Cumulative recentering shift, so boardspace→axial conversion stays in frame.
    let mut frame_offset: (i8, i8) = (0, 0);

    // Collect all actions first so we can use early-return error propagation.
    let mut actions: Vec<(u8, String)> = Vec::new();
    scan_player_actions(content, |p, a| actions.push((p, a.to_string())));

    for (player_idx, raw_action) in &actions {
        let player_color = if *player_idx == 0 { PieceColor::White } else { PieceColor::Black };
        // Strip leading sequence number (e.g., "57 done" → "done").
        let action = {
            let s = raw_action.as_str();
            let bytes = s.as_bytes();
            if !bytes.is_empty() && bytes[0].is_ascii_digit() {
                s.trim_start_matches(|c: char| c.is_ascii_digit()).trim_start()
            } else {
                s
            }
        };

        // done / start → finalize pending move
        if starts_with_ci(action, b"done") || starts_with_ci(action, b"start") {
            if let Some((piece, dest_col, dest_row)) = pending.take() {
                if game.is_game_over() { break; }
                let mv = make_move(game, piece, dest_col, dest_row, &mut origin, frame_offset);
                on_move(game, &mv);
                game.play_move(&mv).map_err(|msg| {
                    format!("move {} rejected: {} ({:?})", move_count + 1, msg, piece)
                })?;
                let (dq, dr) = game.last_recenter_shift();
                frame_offset = (frame_offset.0.saturating_add(dq), frame_offset.1.saturating_add(dr));
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

        // Movedone (must check before "move"): "Movedone W wS1 N 13 /wQ"
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
                    let mv = make_move(game, piece, dest_col, dest_row, &mut origin, frame_offset);
                    on_move(game, &mv);
                    game.play_move(&mv).map_err(|msg| {
                        format!("move {} rejected: {} ({:?})", move_count + 1, msg, piece)
                    })?;
                    let (dq, dr) = game.last_recenter_shift();
                    frame_offset = (frame_offset.0.saturating_add(dq), frame_offset.1.saturating_add(dr));
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
                    let mv = make_move(game, piece, dest_col, dest_row, &mut origin, frame_offset);
                    on_move(game, &mv);
                    game.play_move(&mv).map_err(|msg| {
                        format!("move {} rejected: {} ({:?})", move_count + 1, msg, piece)
                    })?;
                    let (dq, dr) = game.last_recenter_shift();
                    frame_offset = (frame_offset.0.saturating_add(dq), frame_offset.1.saturating_add(dr));
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

// ---------------------------------------------------------------------------
// Determine game result from SGF metadata
// ---------------------------------------------------------------------------

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

    #[test]
    fn test_col_to_index_edge_overflow() {
        assert_eq!(col_to_index("A"), 0);
        assert_eq!(col_to_index("Z"), 25);
        assert_eq!(col_to_index("1A"), -1);
        assert_eq!(col_to_index("2A"), -2);
        assert_eq!(col_to_index("1Z"), 26);
        assert_eq!(col_to_index("2Z"), 27);
    }

    #[test]
    fn test_boardspace_column_wraparound() {
        // Normal case: A=0 is close to origin D=3, no wrap needed
        let h = boardspace_to_hex(0, 9, 3, 9);
        assert_eq!(h, (-3i8, 0i8));

        // Wrap forward: A is one east of Z=25 when origin is X=23
        // (the game that triggered this bug: bA2 placed east of bG3 at Z)
        let h = boardspace_to_hex(0, 12, 23, 9);
        assert_eq!(h, (3i8, -3i8));

        // Wrap backward: Z=25 is one west of A=0 when origin is E=4
        // (25 steps right of E is > max diameter 15, so must wrap to -1 steps)
        let h = boardspace_to_hex(25, 9, 4, 9);
        assert_eq!(h, (-5i8, 0i8));
    }

    #[test]
    fn test_boardspace_row_wraparound() {
        // Normal case: row=6 relative to origin row=6, no wrap
        let h = boardspace_to_hex(5, 6, 5, 6);
        assert_eq!(h, (0i8, 0i8));

        // Wrap backward: row=26 is one above row=1 when origin is row=6
        // (HV-Cesar1984-SmartBot: wG2 placed at B 26 adjacent to wS2 at C 1)
        // Without wrap: r = -(26-6) = -20; with wrap: row=26-26=0, r = -(0-6) = 6
        let h = boardspace_to_hex(2, 26, 5, 6);
        assert_eq!(h, (-3i8, 6i8));

        // Wrap forward: row=1 is one below row=26 when origin is row=22
        // (25 steps below origin 22 wraps to one step up)
        let h = boardspace_to_hex(5, 1, 5, 22);
        assert_eq!(h, (0i8, -5i8));
    }
}
