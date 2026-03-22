/// Boardspace SGF file parser — extracts UHP move strings from Hive SGF files.
///
/// Port of hive/sgf.py to Rust.

use std::collections::HashMap;

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

/// Add color prefix and normalize piece type letter to uppercase.
fn ensure_color(piece: &str, color: char) -> String {
    let bytes = piece.as_bytes();
    if bytes.len() >= 2
        && (bytes[0] == b'w' || bytes[0] == b'b')
        && (bytes[1] as char).is_ascii_alphabetic()
    {
        // Has color prefix — normalize type letter to uppercase
        let mut result = String::with_capacity(piece.len());
        result.push(bytes[0] as char);
        result.push_str(&piece[1..].to_uppercase());
        return result;
    }
    // No color prefix — add color and uppercase
    let mut result = String::with_capacity(piece.len() + 1);
    result.push(color);
    result.push_str(&piece.to_uppercase());
    result
}

/// UHP ref for a beetle dropping from a stack to an adjacent empty hex.
fn drop_down_ref(old_coords: &str, new_coords: &str, piece: &str) -> String {
    let (cx, cy) = parse_coords(old_coords);
    let (dx, dy) = parse_coords(new_coords);

    if cy == dy {
        if cx > dx {
            format!("-{}", piece)
        } else {
            format!("{}-", piece)
        }
    } else if cx == dx {
        if cy > dy {
            format!("{}\\", piece)
        } else {
            format!("\\{}", piece)
        }
    } else if cy > dy {
        format!("/{}", piece)
    } else {
        format!("{}/", piece)
    }
}

fn parse_coords(s: &str) -> (char, i32) {
    let parts: Vec<&str> = s.split('-').collect();
    let col = parts[0].chars().next().unwrap();
    let row: i32 = parts[1].parse().unwrap();
    (col, row)
}

/// Resolve the reference string from Boardspace SGF to UHP notation.
fn resolve_ref(
    ref_raw: &str,
    new_coords: &str,
    piece: &str,
    move_count: usize,
    coord_stack: &HashMap<String, Vec<String>>,
    piece_coords: &HashMap<String, String>,
) -> String {
    let r = ref_raw.replace("\\\\", "\\");

    if r == "." {
        if move_count == 0 {
            return String::new(); // first placement
        }
        if let Some(stack) = coord_stack.get(new_coords) {
            if !stack.is_empty() {
                return stack.last().unwrap().clone(); // beetle stacking
            }
        }
        if let Some(old_coords) = piece_coords.get(piece) {
            return drop_down_ref(old_coords, new_coords, piece);
        }
        return String::new();
    }

    if r.len() > 1 && r.ends_with('.') {
        return r[..r.len() - 1].to_string(); // "bQ." notation — strip dot
    }

    r
}

fn update_coords(
    coord_stack: &mut HashMap<String, Vec<String>>,
    piece_coords: &mut HashMap<String, String>,
    piece: &str,
    old_coords: Option<&str>,
    new_coords: &str,
) {
    if let Some(old) = old_coords {
        if let Some(stack) = coord_stack.get_mut(old) {
            if let Some(pos) = stack.iter().position(|p| p == piece) {
                stack.remove(pos);
            }
        }
    }
    coord_stack
        .entry(new_coords.to_string())
        .or_default()
        .push(piece.to_string());
    piece_coords.insert(piece.to_string(), new_coords.to_string());
}

/// Parse UHP moves from Boardspace SGF content.
///
/// Returns a Vec of UHP move strings.
pub fn parse_moves(content: &str) -> Vec<String> {
    let mut coord_stack: HashMap<String, Vec<String>> = HashMap::new();
    let mut piece_coords: HashMap<String, String> = HashMap::new();
    let mut move_count: usize = 0;
    let mut pending_old_coords: Option<String> = None;
    // Buffered move: (piece, uhp_ref, old_coords, new_coords)
    let mut pending_move: Option<(String, String, Option<String>, String)> = None;
    let mut moves = Vec::new();

    // Match: ;  P0[<num> <action>] or ;  P1[<num> <action>]
    // We'll parse manually since we don't have regex in no-std
    let mut i = 0;
    let bytes = content.as_bytes();

    while i < bytes.len() {
        // Find next ; P0[ or ; P1[
        if bytes[i] != b';' {
            i += 1;
            continue;
        }
        i += 1;

        // Skip whitespace
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }

        if i >= bytes.len() || bytes[i] != b'P' {
            continue;
        }
        i += 1;

        if i >= bytes.len() || (bytes[i] != b'0' && bytes[i] != b'1') {
            continue;
        }
        let player_idx = bytes[i];
        let player_color = if player_idx == b'0' { 'w' } else { 'b' };
        i += 1;

        if i >= bytes.len() || bytes[i] != b'[' {
            continue;
        }
        i += 1;

        // Skip the turn number
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        // Skip whitespace after number
        while i < bytes.len() && bytes[i] == b' ' {
            i += 1;
        }

        // Read the action until ']'
        let action_start = i;
        // Find the closing ], handling possible TM[...] suffix
        let mut bracket_depth = 1;
        while i < bytes.len() && bracket_depth > 0 {
            if bytes[i] == b'[' {
                bracket_depth += 1;
            } else if bytes[i] == b']' {
                bracket_depth -= 1;
            }
            if bracket_depth > 0 {
                i += 1;
            }
        }
        let action = &content[action_start..i];
        let action = action.trim();
        i += 1; // skip ]

        // Process the action
        let low = action.to_lowercase();

        if low.starts_with("done") || low.starts_with("start") {
            if let Some((piece, uhp_ref, old_coords, new_coords)) = pending_move.take() {
                if uhp_ref.is_empty() {
                    moves.push(piece.clone());
                } else {
                    moves.push(format!("{} {}", piece, uhp_ref));
                }
                update_coords(
                    &mut coord_stack,
                    &mut piece_coords,
                    &piece,
                    old_coords.as_deref(),
                    &new_coords,
                );
                move_count += 1;
            }
            pending_old_coords = None;
            continue;
        }

        // Pick from reserve: "Pick W 4 wS1" or "pick b 1 A1"
        if low.starts_with("pick ") && !low.starts_with("pickb") {
            pending_old_coords = None;
            continue;
        }

        // Pick from board: "Pickb N 13 wS1"
        if low.starts_with("pickb ") || low.starts_with("pickb\t") {
            let parts: Vec<&str> = action.split_whitespace().collect();
            if parts.len() >= 4 {
                let col = parts[1].to_uppercase();
                let row = parts[2];
                let picked_coords = format!("{}-{}", col, row);

                if let Some(ref pm) = pending_move {
                    if picked_coords == pm.3 {
                        // Undo: picking up piece just dropped this turn
                        pending_old_coords = pm.2.clone();
                        pending_move = None;
                    } else {
                        pending_old_coords = Some(picked_coords);
                    }
                } else {
                    pending_old_coords = Some(picked_coords);
                }
            }
            continue;
        }

        // Drop: "Dropb wS1 N 13 ."
        if low.starts_with("dropb ") || low.starts_with("dropb\t") {
            let parts: Vec<&str> = action.split_whitespace().collect();
            if parts.len() >= 5 {
                let piece = ensure_color(parts[1], player_color);
                let col = parts[2].to_uppercase();
                let row = parts[3];
                let ref_raw = parts[4..].join(" ");
                let new_coords = format!("{}-{}", col, row);
                let old_coords = pending_old_coords.take();

                let uhp_ref = resolve_ref(
                    ref_raw.trim(),
                    &new_coords,
                    &piece,
                    move_count,
                    &coord_stack,
                    &piece_coords,
                );
                pending_move = Some((piece, uhp_ref, old_coords, new_coords));
            }
            continue;
        }

        // movedone: "movedone B bS1 M 12 /wS1" — must check before "move"
        if low.starts_with("movedone ") {
            let parts: Vec<&str> = action.split_whitespace().collect();
            if parts.len() >= 6 {
                let color = if parts[1].to_uppercase() == "W" {
                    'w'
                } else {
                    'b'
                };
                let piece = ensure_color(parts[2], color);
                let col = parts[3].to_uppercase();
                let row = parts[4];
                let ref_raw = parts[5..].join(" ");
                let new_coords = format!("{}-{}", col, row);
                let old_coords = piece_coords.get(&piece).cloned();

                let uhp_ref = resolve_ref(
                    ref_raw.trim(),
                    &new_coords,
                    &piece,
                    move_count,
                    &coord_stack,
                    &piece_coords,
                );
                if uhp_ref.is_empty() {
                    moves.push(piece.clone());
                } else {
                    moves.push(format!("{} {}", piece, uhp_ref));
                }
                update_coords(
                    &mut coord_stack,
                    &mut piece_coords,
                    &piece,
                    old_coords.as_deref(),
                    &new_coords,
                );
                move_count += 1;
            }
            continue;
        }

        // Move: "Move B bS1 M 12 /wS1" or "move W B1 N 13 ."
        if low.starts_with("move ") {
            let parts: Vec<&str> = action.split_whitespace().collect();
            if parts.len() >= 6 {
                let color = if parts[1].to_uppercase() == "W" {
                    'w'
                } else {
                    'b'
                };
                let piece = ensure_color(parts[2], color);
                let col = parts[3].to_uppercase();
                let row = parts[4];
                let ref_raw = parts[5..].join(" ");
                let new_coords = format!("{}-{}", col, row);
                let old_coords = piece_coords.get(&piece).cloned();

                let uhp_ref = resolve_ref(
                    ref_raw.trim(),
                    &new_coords,
                    &piece,
                    move_count,
                    &coord_stack,
                    &piece_coords,
                );
                if uhp_ref.is_empty() {
                    moves.push(piece.clone());
                } else {
                    moves.push(format!("{} {}", piece, uhp_ref));
                }
                update_coords(
                    &mut coord_stack,
                    &mut piece_coords,
                    &piece,
                    old_coords.as_deref(),
                    &new_coords,
                );
                move_count += 1;
            }
            continue;
        }

        if low == "pass" {
            moves.push("pass".to_string());
            move_count += 1;
            continue;
        }
    }

    moves
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

    #[test]
    fn test_ensure_color() {
        assert_eq!(ensure_color("wQ", 'w'), "wQ");
        assert_eq!(ensure_color("bA1", 'b'), "bA1");
        assert_eq!(ensure_color("A1", 'w'), "wA1");
        assert_eq!(ensure_color("q", 'b'), "bQ");
        assert_eq!(ensure_color("ba1", 'b'), "bA1");
    }
}
