//! Parser for boardspace.net Zertz game records (SGF-like format).
//!
//! Handles both old format (no Done/TM markers, 2006-era) and new format
//! (with Done, Pick, TM timestamps, 2020s-era). Supports standard and
//! tournament (`zertz+11`) game variants.

use std::fmt;

pub use core_game::sgf::{extract_player, extract_prop};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A coordinate on the boardspace hex grid (0-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coord {
    pub col: u8, // A=0, B=1, ...
    pub row: u8, // 0-based row within column
}

impl Coord {
    /// Parse from boardspace strings like ("F", "2") -> col=5, row=1.
    pub fn parse(col_str: &str, row_str: &str) -> Result<Self, ParseError> {
        let col_char = col_str
            .bytes()
            .next()
            .ok_or_else(|| ParseError::InvalidCoord(format!("{col_str} {row_str}")))?;
        let col = col_char
            .checked_sub(b'A')
            .ok_or_else(|| ParseError::InvalidCoord(format!("{col_str} {row_str}")))?;
        let row_num: u8 = row_str
            .parse()
            .map_err(|_| ParseError::InvalidCoord(format!("{col_str} {row_str}")))?;
        if row_num == 0 {
            return Err(ParseError::InvalidCoord(format!("{col_str} {row_str}")));
        }
        Ok(Coord {
            col,
            row: row_num - 1,
        })
    }
}

impl fmt::Display for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", (b'A' + self.col) as char, self.row + 1)
    }
}

/// Marble color as encoded in boardspace format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White = 0,
    Grey = 1,
    Black = 2,
}

impl Color {
    fn from_code(code: u8) -> Result<Self, ParseError> {
        match code {
            0 => Ok(Color::White),
            1 => Ok(Color::Grey),
            2 => Ok(Color::Black),
            _ => Err(ParseError::InvalidColor(code)),
        }
    }
}

/// Board variant detected from the SU header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Variant {
    Standard,
    Tournament { extra_rings: u8 },
}

/// A reconstructed game turn.
#[derive(Debug, Clone)]
pub enum Turn {
    /// Place marble + remove ring.
    Place {
        color: Color,
        at: Coord,
        remove: Coord,
    },
    /// Place marble, no ring to remove.
    PlaceOnly { color: Color, at: Coord },
    /// Capture: list of (from, to) jumps.
    Capture { jumps: Vec<(Coord, Coord)> },
}

/// A complete parsed game.
#[derive(Debug, Clone)]
pub struct GameRecord {
    pub variant: Variant,
    pub player0: String,
    pub player1: String,
    pub result: String,
    pub game_name: String,
    pub first_player: u8, // 0 or 1
    pub turns: Vec<(u8, Turn)>,
}

#[derive(Debug)]
pub enum ParseError {
    InvalidFormat(String),
    InvalidCoord(String),
    InvalidColor(u8),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::InvalidFormat(s) => write!(f, "invalid format: {s}"),
            ParseError::InvalidCoord(s) => write!(f, "invalid coord: {s}"),
            ParseError::InvalidColor(c) => write!(f, "invalid color code: {c}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Raw actions (internal)
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum RawAction {
    Start(u8),
    RtoB { color: Color, at: Coord },
    RemoveRing(Coord),
    BtoB { from: Coord, to: Coord },
    Pick(#[allow(dead_code)] Color),
    Pickb(#[allow(dead_code)] Coord),
    BtoR(Coord),
    RtoR,
    Done,
}

// ---------------------------------------------------------------------------
// Header parsing
// ---------------------------------------------------------------------------

fn parse_header(text: &str) -> (Variant, String, String, String, String) {
    let variant = match extract_prop(text, "SU") {
        Some(su) => {
            let su_lower = su.to_lowercase();
            if let Some(rest) = su_lower.strip_prefix("zertz+") {
                let extra: u8 = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(11);
                Variant::Tournament { extra_rings: extra }
            } else {
                Variant::Standard
            }
        }
        None => Variant::Standard,
    };

    let p0 = extract_player(text, 0);
    let p1 = extract_player(text, 1);
    let result = extract_prop(text, "RE").unwrap_or("").to_string();
    let game_name = extract_prop(text, "GN").unwrap_or("").to_string();

    (variant, p0, p1, result, game_name)
}

// ---------------------------------------------------------------------------
// Action parsing
// ---------------------------------------------------------------------------

fn parse_action_content(player: u8, content: &str) -> Result<(u8, RawAction), ParseError> {
    let parts: Vec<&str> = content.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(ParseError::InvalidFormat(format!(
            "too few parts: {content}"
        )));
    }

    // parts[0] is sequence number, parts[1] is command
    let cmd = parts[1];
    match cmd {
        "Start" => {
            let who = if parts.get(2) == Some(&"P0") { 0 } else { 1 };
            Ok((player, RawAction::Start(who)))
        }
        "RtoB" => {
            // RtoB 2 <color> <col> <row>
            if parts.len() < 6 {
                return Err(ParseError::InvalidFormat(format!("short RtoB: {content}")));
            }
            let color_code: u8 = parts[3]
                .parse()
                .map_err(|_| ParseError::InvalidFormat(format!("bad color: {}", parts[3])))?;
            let coord = Coord::parse(parts[4], parts[5])?;
            Ok((
                player,
                RawAction::RtoB {
                    color: Color::from_code(color_code)?,
                    at: coord,
                },
            ))
        }
        "R-" => {
            if parts.len() < 4 {
                return Err(ParseError::InvalidFormat(format!("short R-: {content}")));
            }
            let coord = Coord::parse(parts[2], parts[3])?;
            Ok((player, RawAction::RemoveRing(coord)))
        }
        "BtoB" => {
            if parts.len() < 6 {
                return Err(ParseError::InvalidFormat(format!("short BtoB: {content}")));
            }
            let from = Coord::parse(parts[2], parts[3])?;
            let to = Coord::parse(parts[4], parts[5])?;
            Ok((player, RawAction::BtoB { from, to }))
        }
        "Pick" => {
            if parts.len() < 4 {
                return Err(ParseError::InvalidFormat(format!("short Pick: {content}")));
            }
            let color_code: u8 = parts[3]
                .parse()
                .map_err(|_| ParseError::InvalidFormat(format!("bad color: {}", parts[3])))?;
            Ok((player, RawAction::Pick(Color::from_code(color_code)?)))
        }
        "Pickb" => {
            if parts.len() < 4 {
                return Err(ParseError::InvalidFormat(format!("short Pickb: {content}")));
            }
            let coord = Coord::parse(parts[2], parts[3])?;
            Ok((player, RawAction::Pickb(coord)))
        }
        "BtoR" => {
            // Board to Reserve: player puts a placed marble back. Undoes placement.
            if parts.len() < 4 {
                return Err(ParseError::InvalidFormat(format!("short BtoR: {content}")));
            }
            let coord = Coord::parse(parts[2], parts[3])?;
            Ok((player, RawAction::BtoR(coord)))
        }
        "RtoR" => Ok((player, RawAction::RtoR)),
        "Done" => Ok((player, RawAction::Done)),
        _ => Err(ParseError::InvalidFormat(format!(
            "unknown command: {cmd}"
        ))),
    }
}

/// Extract all raw actions from the SGF text.
fn extract_actions(text: &str) -> Vec<(u8, RawAction)> {
    let mut actions = Vec::new();

    // Each action line contains P0[...] or P1[...] after a semicolon.
    // We scan for P0[ or P1[ patterns and extract the content up to ].
    for line in text.lines() {
        let line = line.trim();
        if !line.starts_with(';') {
            continue;
        }

        // A line may contain multiple P0/P1 entries (rare but possible).
        let mut pos = 0;
        let bytes = line.as_bytes();
        while pos < bytes.len() {
            // Look for P0[ or P1[
            if pos + 2 < bytes.len()
                && bytes[pos] == b'P'
                && (bytes[pos + 1] == b'0' || bytes[pos + 1] == b'1')
                && bytes[pos + 2] == b'['
            {
                let player = bytes[pos + 1] - b'0';
                let content_start = pos + 3;
                // Find closing ]
                if let Some(rel_end) = line[content_start..].find(']') {
                    let content = &line[content_start..content_start + rel_end];
                    let content = content.trim();
                    if !content.is_empty() {
                        if let Ok(action) = parse_action_content(player, content) {
                            actions.push(action);
                        }
                    }
                    pos = content_start + rel_end + 1;
                } else {
                    break;
                }
            } else {
                pos += 1;
            }
        }
    }

    actions
}

// ---------------------------------------------------------------------------
// Turn reconstruction
// ---------------------------------------------------------------------------

/// Check if two hex coordinates are adjacent (1 step apart).
/// On our hex grid with columns of varying lengths, adjacent means
/// same column ±1 row, or neighboring column with row offset 0 or ±1.
#[allow(dead_code)]
fn is_hex_adjacent(a: Coord, b: Coord) -> bool {
    let dc = (a.col as i8 - b.col as i8).unsigned_abs();
    let dr = (a.row as i8 - b.row as i8).unsigned_abs();
    match dc {
        0 => dr == 1,
        1 => dr <= 1,
        _ => false,
    }
}

fn flush_turn(player: u8, actions: &[RawAction], turns: &mut Vec<(u8, Turn)>) {
    // Process actions sequentially to handle undo/redo in old-format games.
    //
    // Key patterns:
    //   RtoB X, BtoB X→Y           — relocation (single BtoB from placement)
    //   RtoB X, BtoB X→Y, BtoR Y   — full undo (placement cancelled)
    //   RtoB X, BtoB X→Y→Z→...     — capture chain after placement (multi-BtoB)
    //   BtoB A→B, BtoB B→C         — capture chain (no placement involved)

    // Step 1: Build a list of (RtoB, BtoB, BtoR, R-) events in order.
    // Each RtoB starts a new "placement attempt". BtoR cancels it.
    let mut final_place: Option<(Color, Coord)> = None;
    let mut current_place_at: Option<Coord> = None;
    let mut btobs_for_current_place: Vec<(Coord, Coord)> = Vec::new();
    let mut capture_jumps: Vec<(Coord, Coord)> = Vec::new();
    let mut remove_coord: Option<Coord> = None;

    for a in actions {
        match a {
            RawAction::RtoB { color, at } => {
                // New placement attempt. Any previous BtoBs from a cancelled
                // placement are discarded (they were part of the undo chain).
                final_place = Some((*color, *at));
                current_place_at = Some(*at);
                btobs_for_current_place.clear();
            }
            RawAction::BtoB { from, to } => {
                if from == to {
                    continue;
                }
                // Does this BtoB relate to the current placement attempt?
                if let Some(pat) = current_place_at {
                    if *from == pat || btobs_for_current_place.last().is_some_and(|(_, t)| t == from) {
                        btobs_for_current_place.push((*from, *to));
                        continue;
                    }
                }
                // Not related to placement — it's a capture.
                capture_jumps.push((*from, *to));
            }
            RawAction::BtoR(_coord) => {
                // Player put the marble back — cancel the current placement attempt
                // and discard any BtoBs related to it.
                final_place = None;
                current_place_at = None;
                btobs_for_current_place.clear();
            }
            RawAction::RemoveRing(coord) => {
                remove_coord = Some(*coord);
            }
            _ => {}
        }
    }

    // Step 2: Resolve BtoBs from the final placement.
    // If placement survived (no BtoR), interpret its BtoBs.
    if let Some((_color, at)) = final_place {
        if btobs_for_current_place.len() == 1 {
            // Single BtoB from placement = relocation (player reconsidered).
            // Don't add to captures.
            let (_from, to) = btobs_for_current_place[0];
            final_place = Some((_color, to));
        } else if btobs_for_current_place.len() > 1 {
            // Multi-BtoB chain from placement position in old format.
            // These are the player dragging the marble around the board before
            // dropping it at the final position. Use the last destination.
            let last_dest = btobs_for_current_place.last().unwrap().1;
            final_place = Some((_color, last_dest));
        }
        // If 0 BtoBs: placement stands as-is.
        let _ = at; // suppress unused warning
    }

    // Step 3: Determine if captures start from the placement position.
    // If so, the placement must be emitted first (the capture uses the
    // just-placed marble). Otherwise, captures come first (normal order).
    let captures_from_place = !capture_jumps.is_empty()
        && final_place.is_some_and(|(_, at)| capture_jumps[0].0 == at);

    if captures_from_place {
        // Emit placement first, then captures.
        if let Some((color, at)) = final_place {
            match remove_coord {
                Some(remove) => turns.push((player, Turn::Place { color, at, remove })),
                None => turns.push((player, Turn::PlaceOnly { color, at })),
            }
        }
        turns.push((player, Turn::Capture { jumps: capture_jumps }));
    } else {
        // Normal order: captures first, then placement.
        if !capture_jumps.is_empty() {
            turns.push((player, Turn::Capture { jumps: capture_jumps }));
        }
        if let Some((color, at)) = final_place {
            match remove_coord {
                Some(remove) => turns.push((player, Turn::Place { color, at, remove })),
                None => turns.push((player, Turn::PlaceOnly { color, at })),
            }
        }
    }
}

fn group_into_turns(actions: Vec<(u8, RawAction)>) -> (u8, Vec<(u8, Turn)>) {
    let mut turns = Vec::new();
    let mut first_player = 0u8;
    let mut current_player: Option<u8> = None;
    let mut current_actions: Vec<RawAction> = Vec::new();
    let mut has_done = false;

    for (player, action) in actions {
        match &action {
            RawAction::Start(p) => {
                first_player = *p;
                current_player = Some(*p);
                continue;
            }
            RawAction::Done => {
                has_done = true;
                if !current_actions.is_empty() {
                    flush_turn(
                        current_player.unwrap_or(player),
                        &current_actions,
                        &mut turns,
                    );
                    current_actions.clear();
                }
                continue;
            }
            _ => {}
        }

        // Old format: detect turn boundary on player change.
        if !has_done {
            if let Some(cp) = current_player {
                if player != cp && !current_actions.is_empty() {
                    flush_turn(cp, &current_actions, &mut turns);
                    current_actions.clear();
                }
            }
        }

        current_player = Some(player);
        current_actions.push(action);
    }

    // Flush remaining.
    if !current_actions.is_empty() {
        if let Some(cp) = current_player {
            flush_turn(cp, &current_actions, &mut turns);
        }
    }

    (first_player, turns)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a single game record from SGF text.
pub fn parse_game(text: &str) -> Result<GameRecord, ParseError> {
    let (variant, player0, player1, result, game_name) = parse_header(text);
    let actions = extract_actions(text);

    if actions.is_empty() {
        return Err(ParseError::InvalidFormat("no actions found".into()));
    }

    let (first_player, turns) = group_into_turns(actions);

    Ok(GameRecord {
        variant,
        player0,
        player1,
        result,
        game_name,
        first_player,
        turns,
    })
}

/// Iterate over all .sgf games inside a zip archive, streaming from a reader.
/// Calls `f` for each successfully parsed game.
#[cfg(feature = "replay")]
pub fn iter_games_in_zip<R, F>(reader: R, mut f: F) -> Result<(), String>
where
    R: std::io::Read + std::io::Seek,
    F: FnMut(&str, GameRecord),
{
    core_game::sgf::iter_sgf_texts_in_zip(reader, |name, text| {
        if let Ok(record) = parse_game(&text) {
            f(name, record);
        }
    })
}
