//! Parser for boardspace.net Yinsh game records (SGF-like format).
//!
//! Action grammar inside `Px[...]`:
//!   `Start White|Black`        — declares first player's color
//!   `place wr|br COL ROW`      — setup-phase ring placement
//!   `place w|b COL ROW`        — pick up a ring (start of normal-phase move)
//!   `drop board COL ROW`       — drop the picked-up ring
//!   `remove w|b COL ROW COL ROW`  — remove a 5-marker row (start..end inclusive)
//!   `remove wr|br COL ROW`     — remove a ring after a row was claimed
//!   `Done`                     — terminates one semantic action

use std::fmt;

pub use core_game::sgf::{extract_player, extract_prop};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coord {
    pub col: u8, // A=0
    pub row: u8, // 0-indexed (boardspace row N → row N-1)
}

impl Coord {
    pub fn parse(col_str: &str, row_str: &str) -> Result<Self, ParseError> {
        let col_byte = col_str
            .bytes()
            .next()
            .ok_or_else(|| ParseError::InvalidCoord(format!("{col_str} {row_str}")))?
            .to_ascii_uppercase();
        let col = col_byte
            .checked_sub(b'A')
            .ok_or_else(|| ParseError::InvalidCoord(format!("{col_str} {row_str}")))?;
        let row_num: u8 = row_str
            .parse()
            .map_err(|_| ParseError::InvalidCoord(format!("{col_str} {row_str}")))?;
        if row_num == 0 {
            return Err(ParseError::InvalidCoord(format!("{col_str} {row_str}")));
        }
        Ok(Coord { col, row: row_num - 1 })
    }
}

impl fmt::Display for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", (b'A' + self.col) as char, self.row + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

#[derive(Debug, Clone)]
pub enum Turn {
    PlaceRing { color: Color, at: Coord },
    MoveRing { color: Color, from: Coord, to: Coord },
    RemoveRow { color: Color, from: Coord, to: Coord },
    RemoveRing { color: Color, at: Coord },
}

#[derive(Debug, Clone)]
pub struct GameRecord {
    pub player0: String,
    pub player1: String,
    pub result: String,
    pub game_name: String,
    pub first_player_color: Color,
    pub turns: Vec<(u8, Turn)>,
}

#[derive(Debug)]
pub enum ParseError {
    InvalidFormat(String),
    InvalidCoord(String),
    UnsupportedSetup(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::InvalidFormat(s) => write!(f, "invalid format: {s}"),
            ParseError::InvalidCoord(s) => write!(f, "invalid coord: {s}"),
            ParseError::UnsupportedSetup(s) => write!(f, "unsupported setup: {s}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Raw actions
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum RawAction {
    Start(Color),
    PlaceRing { color: Color, at: Coord },
    PickRing { at: Coord },
    DropBoard { at: Coord },
    PickBoard,
    Move { from: Coord, to: Coord },
    RemoveRow { color: Color, from: Coord, to: Coord },
    RemoveRing { color: Color, at: Coord },
    Done,
}

/// Index of the first non-numeric, non-empty token in `parts`. Boardspace usually
/// prefixes actions with a sequence number (`1 place wr F 6`) but sometimes omits
/// it (`move I 4 E 4`), so we locate the keyword instead of trusting a fixed offset.
fn find_cmd(parts: &[&str]) -> Option<usize> {
    parts.iter().position(|s| !s.bytes().all(|b| b.is_ascii_digit()))
}

fn parse_action_content(content: &str) -> Result<Option<RawAction>, ParseError> {
    let parts: Vec<&str> = content.split_whitespace().collect();
    let i = match find_cmd(&parts) {
        Some(i) => i,
        None => return Ok(None),
    };
    let cmd = parts[i];
    let rest = &parts[i + 1..];

    match cmd.to_ascii_lowercase().as_str() {
        "start" => {
            let color_str = rest.first().copied().unwrap_or("");
            let color = match color_str.to_ascii_lowercase().as_str() {
                "white" => Color::White,
                "black" => Color::Black,
                _ => return Err(ParseError::InvalidFormat(format!("bad Start: {content}"))),
            };
            Ok(Some(RawAction::Start(color)))
        }
        "place" => {
            if rest.len() < 3 {
                return Err(ParseError::InvalidFormat(format!("short place: {content}")));
            }
            let kind = rest[0];
            let coord = Coord::parse(rest[1], rest[2])?;
            match kind {
                "wr" => Ok(Some(RawAction::PlaceRing { color: Color::White, at: coord })),
                "br" => Ok(Some(RawAction::PlaceRing { color: Color::Black, at: coord })),
                "w" | "b" => Ok(Some(RawAction::PickRing { at: coord })),
                _ => Err(ParseError::InvalidFormat(format!("unknown place kind: {kind}"))),
            }
        }
        "drop" => {
            // `drop board COL ROW`
            if rest.len() < 3 {
                return Err(ParseError::InvalidFormat(format!("short drop: {content}")));
            }
            let coord = Coord::parse(rest[1], rest[2])?;
            Ok(Some(RawAction::DropBoard { at: coord }))
        }
        "pick" => {
            // `pick board COL ROW` — re-pick a ring just dropped. We only need the
            // signal that the placement chain restarts; the coord matches the
            // last drop and is used for engine bookkeeping, not the move.
            Ok(Some(RawAction::PickBoard))
        }
        "move" => {
            // `move FROM_COL FROM_ROW TO_COL TO_ROW`
            if rest.len() < 4 {
                return Err(ParseError::InvalidFormat(format!("short move: {content}")));
            }
            let from = Coord::parse(rest[0], rest[1])?;
            let to = Coord::parse(rest[2], rest[3])?;
            Ok(Some(RawAction::Move { from, to }))
        }
        "remove" => {
            if rest.len() < 3 {
                return Err(ParseError::InvalidFormat(format!("short remove: {content}")));
            }
            let kind = rest[0];
            let color = match kind {
                "w" | "wr" => Color::White,
                "b" | "br" => Color::Black,
                _ => return Err(ParseError::InvalidFormat(format!("unknown remove kind: {kind}"))),
            };
            match kind {
                "wr" | "br" => {
                    let coord = Coord::parse(rest[1], rest[2])?;
                    Ok(Some(RawAction::RemoveRing { color, at: coord }))
                }
                _ => {
                    if rest.len() < 5 {
                        return Err(ParseError::InvalidFormat(format!(
                            "short remove row: {content}"
                        )));
                    }
                    let from = Coord::parse(rest[1], rest[2])?;
                    let to = Coord::parse(rest[3], rest[4])?;
                    Ok(Some(RawAction::RemoveRow { color, from, to }))
                }
            }
        }
        "done" => Ok(Some(RawAction::Done)),
        // Ignore time/ranking/etc.
        _ => Ok(None),
    }
}

fn extract_actions(text: &str) -> Vec<(u8, RawAction)> {
    let mut actions = Vec::new();
    core_game::sgf::scan_player_actions(text, |player, content| {
        if let Ok(Some(action)) = parse_action_content(content) {
            actions.push((player, action));
        }
    });
    actions
}

// ---------------------------------------------------------------------------
// Turn assembly
// ---------------------------------------------------------------------------

fn build_turn(buf: &[(u8, RawAction)]) -> Option<Turn> {
    // Setup phase: if there are any PlaceRing actions, the LAST one wins
    // (boardspace records intermediate drag positions before the final placement).
    let mut place_ring: Option<(Color, Coord)> = None;

    // Normal-phase ring move: `place w X` or `move X Y` starts a chain; subsequent
    // `drop board Y` / `pick board Y` events update the destination but the
    // original pick source remains. We track first pick + last drop.
    let mut pick_from: Option<Coord> = None;
    let mut last_drop: Option<Coord> = None;

    let mut remove_row: Option<(Color, Coord, Coord)> = None;
    let mut remove_ring: Option<(Color, Coord)> = None;

    for (_p, a) in buf {
        match a {
            RawAction::PlaceRing { color, at } => place_ring = Some((*color, *at)),
            RawAction::PickRing { at } => {
                if pick_from.is_none() {
                    pick_from = Some(*at);
                }
            }
            RawAction::DropBoard { at } => last_drop = Some(*at),
            RawAction::PickBoard => {
                // Placement chain restarts; original pick stays.
                last_drop = None;
            }
            RawAction::Move { from, to } => {
                if pick_from.is_none() {
                    pick_from = Some(*from);
                }
                last_drop = Some(*to);
            }
            RawAction::RemoveRow { color, from, to } => remove_row = Some((*color, *from, *to)),
            RawAction::RemoveRing { color, at } => remove_ring = Some((*color, *at)),
            _ => {}
        }
    }

    if let Some((c, at)) = place_ring {
        return Some(Turn::PlaceRing { color: c, at });
    }
    if let (Some(from), Some(to)) = (pick_from, last_drop) {
        // Color is set to a placeholder; replay derives the moving colour from the
        // engine's `next_player`. We only need from/to to be correct.
        return Some(Turn::MoveRing { color: Color::White, from, to });
    }
    if let Some((c, from, to)) = remove_row {
        return Some(Turn::RemoveRow { color: c, from, to });
    }
    if let Some((c, at)) = remove_ring {
        return Some(Turn::RemoveRing { color: c, at });
    }
    None
}

fn group_into_turns(actions: Vec<(u8, RawAction)>) -> (Color, Vec<(u8, Turn)>) {
    let mut turns = Vec::new();
    let mut first_color = Color::White;
    let mut buf: Vec<(u8, RawAction)> = Vec::new();

    for (player, action) in actions {
        match action {
            RawAction::Start(c) => {
                first_color = c;
            }
            RawAction::Done => {
                if !buf.is_empty() {
                    let p = buf[0].0;
                    if let Some(turn) = build_turn(&buf) {
                        turns.push((p, turn));
                    }
                    buf.clear();
                }
            }
            other => {
                buf.push((player, other));
            }
        }
    }

    // Flush any actions not terminated by Done.
    if !buf.is_empty() {
        let p = buf[0].0;
        if let Some(turn) = build_turn(&buf) {
            turns.push((p, turn));
        }
    }

    (first_color, turns)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn parse_game(text: &str) -> Result<GameRecord, ParseError> {
    if let Some(su) = extract_prop(text, "SU") {
        if !su.to_lowercase().contains("yinsh") {
            return Err(ParseError::UnsupportedSetup(su.to_string()));
        }
    }
    let player0 = extract_player(text, 0);
    let player1 = extract_player(text, 1);
    let result = extract_prop(text, "RE").unwrap_or("").to_string();
    let game_name = extract_prop(text, "GN").unwrap_or("").to_string();

    let actions = extract_actions(text);
    if actions.is_empty() {
        return Err(ParseError::InvalidFormat("no actions found".into()));
    }

    let (first_player_color, turns) = group_into_turns(actions);

    Ok(GameRecord {
        player0,
        player1,
        result,
        game_name,
        first_player_color,
        turns,
    })
}

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
