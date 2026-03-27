//! Shared utilities for parsing boardspace SGF game records.
//!
//! Boardspace uses an SGF-like format for all its games (Hive, Zertz, etc.)
//! with player actions encoded as `;P0[...]` and `;P1[...]` entries.

/// Extract the value of an SGF property `KEY[value]`.
///
/// `key` should not include the opening bracket (e.g., pass `"RE"` not `"RE["`).
pub fn extract_prop<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let pattern = format!("{key}[");
    let start = text.find(&pattern)?;
    let value_start = start + pattern.len();
    let end = text[value_start..].find(']')?;
    Some(&text[value_start..value_start + end])
}

/// Extract the player name from a `P0[id "name"]` or `P1[id "name"]` property.
///
/// Returns `"White"` or `"Black"` as a fallback if the property is missing.
pub fn extract_player(text: &str, player_idx: u8) -> String {
    let prefix = format!("P{}[id \"", player_idx);
    if let Some(start) = text.find(&prefix) {
        let rest = &text[start + prefix.len()..];
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

/// Determine the game result from SGF metadata.
///
/// Checks (in order):
/// 1. The `RE[...]` field for player names or "draw"
/// 2. Any `resign` action in the action stream
/// 3. Any `acceptdraw` action in the action stream
///
/// Returns `"p0_wins"`, `"p1_wins"`, `"draw"`, or `"unknown"`.
pub fn result_from_metadata(content: &str, p0: &str, p1: &str) -> &'static str {
    if let Some(re_val) = extract_prop(content, "RE") {
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

    let mut found: Option<&'static str> = None;
    scan_player_actions(content, |player, action| {
        if found.is_some() {
            return;
        }
        let lower = action.to_lowercase();
        if lower.contains("resign") {
            found = Some(if player == 0 { "p1_wins" } else { "p0_wins" });
        } else if lower.contains("acceptdraw") {
            found = Some("draw");
        }
    });
    found.unwrap_or("unknown")
}

/// Check whether a game ended by timeout (based on the `RE[...]` field).
pub fn is_timeout(content: &str) -> bool {
    if let Some(re_val) = extract_prop(content, "RE") {
        let lower = re_val.to_lowercase();
        lower.contains("time") || lower.contains("timeout")
    } else {
        false
    }
}

/// Scan SGF text for all `P0[...]` and `P1[...]` action entries.
///
/// Calls `f(player_idx, content)` for each entry found after a `;`.
/// `player_idx` is 0 for `P0` and 1 for `P1`.
/// `content` is the trimmed text between the brackets.
/// Nested brackets within the content are handled correctly.
pub fn scan_player_actions<F: FnMut(u8, &str)>(text: &str, mut f: F) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b';' {
            i += 1;
            continue;
        }
        i += 1;

        // Skip spaces and tabs after ';' (not newlines, to avoid crossing line boundaries)
        while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
            i += 1;
        }

        // Expect P0[ or P1[
        if i + 2 >= bytes.len()
            || bytes[i] != b'P'
            || (bytes[i + 1] != b'0' && bytes[i + 1] != b'1')
            || bytes[i + 2] != b'['
        {
            continue;
        }

        let player = bytes[i + 1] - b'0';
        let content_start = i + 3;

        // Find closing ']', tracking nested bracket depth
        let mut j = content_start;
        let mut depth = 1usize;
        while j < bytes.len() && depth > 0 {
            match bytes[j] {
                b'[' => depth += 1,
                b']' => depth -= 1,
                _ => {}
            }
            if depth > 0 {
                j += 1;
            }
        }

        let content = text[content_start..j].trim();
        i = j + 1;

        if !content.is_empty() {
            f(player, content);
        }
    }
}
