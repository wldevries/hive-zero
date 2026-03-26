//! Replay parsed boardspace games on the Zertz engine.
//!
//! Maps boardspace coordinates to engine cell indices, reconstructs
//! `ZertzMove` values, and plays them on a `ZertzBoard`.

use crate::parser::{Color, Coord, GameRecord, Turn, Variant};
use crate::zertz::{
    find_capture_path, find_intermediate, standard_neighbours, BoardLayout, Marble, ZertzBoard,
    ZertzMove, MAX_CAPTURE_JUMPS,
};

// ---------------------------------------------------------------------------
// Coordinate mapping
// ---------------------------------------------------------------------------

fn color_to_marble(c: Color) -> Marble {
    match c {
        Color::White => Marble::White,
        Color::Grey => Marble::Grey,
        Color::Black => Marble::Black,
    }
}

fn coord_to_index(layout: &BoardLayout, coord: Coord) -> Result<usize, ReplayError> {
    layout
        .coord_to_index(coord.col, coord.row)
        .ok_or(ReplayError::BadCoord(coord))
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum ReplayError {
    BadCoord(Coord),
    NoIntermediate { from: Coord, to: Coord },
    IllegalMove { turn: usize, mv: String },
    EngineError { turn: usize, msg: String },
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplayError::BadCoord(c) => write!(f, "bad coordinate: {c}"),
            ReplayError::NoIntermediate { from, to } => {
                write!(f, "no intermediate cell between {from} and {to}")
            }
            ReplayError::IllegalMove { turn, mv } => {
                write!(f, "illegal move at turn {turn}: {mv}")
            }
            ReplayError::EngineError { turn, msg } => {
                write!(f, "engine error at turn {turn}: {msg}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

/// Result of replaying a game.
#[derive(Debug)]
pub struct ReplayResult {
    pub turns_played: usize,
    pub total_turns: usize,
    pub final_board: ZertzBoard,
    pub error: Option<ReplayError>,
}

/// Replay a parsed game record on the engine with optional verbose output.
pub fn replay_game(record: &GameRecord) -> ReplayResult {
    replay_game_inner(record, false)
}

/// Replay with verbose per-turn output for debugging.
pub fn replay_game_verbose(record: &GameRecord) -> ReplayResult {
    replay_game_inner(record, true)
}

fn replay_game_inner(record: &GameRecord, verbose: bool) -> ReplayResult {
    let (layout, nbrs): (BoardLayout, &[[u8; 6]]) = match &record.variant {
        Variant::Standard => (BoardLayout::standard(), standard_neighbours().as_slice()),
        Variant::Tournament { .. } => {
            // Tournament boards use a different neighbour table.
            // For now, we build it on the fly. Could be cached.
            // We'd need to expose tournament_neighbours or build_neighbours.
            // For the MVP, skip tournament games.
            return ReplayResult {
                turns_played: 0,
                total_turns: record.turns.len(),
                final_board: ZertzBoard::default(),
                error: Some(ReplayError::EngineError {
                    turn: 0,
                    msg: "tournament boards not yet supported for replay".into(),
                }),
            };
        }
    };

    let mut board = ZertzBoard::default();
    let total = record.turns.len();

    for (i, (player, turn)) in record.turns.iter().enumerate() {
        if verbose {
            println!("--- Turn {i} (P{player}) : {turn:?}");
        }
        let mv = match turn {
            Turn::Place { color, at, remove } => {
                let place_idx = match coord_to_index(&layout, *at) {
                    Ok(v) => v,
                    Err(e) => {
                        return ReplayResult {
                            turns_played: i,
                            total_turns: total,
                            final_board: board,
                            error: Some(e),
                        }
                    }
                };
                let remove_idx = match coord_to_index(&layout, *remove) {
                    Ok(v) => v,
                    Err(e) => {
                        return ReplayResult {
                            turns_played: i,
                            total_turns: total,
                            final_board: board,
                            error: Some(e),
                        }
                    }
                };
                ZertzMove::Place {
                    color: color_to_marble(*color),
                    place_at: place_idx as u8,
                    remove: remove_idx as u8,
                }
            }
            Turn::PlaceOnly { color, at } => {
                let place_idx = match coord_to_index(&layout, *at) {
                    Ok(v) => v,
                    Err(e) => {
                        return ReplayResult {
                            turns_played: i,
                            total_turns: total,
                            final_board: board,
                            error: Some(e),
                        }
                    }
                };
                ZertzMove::PlaceOnly {
                    color: color_to_marble(*color),
                    place_at: place_idx as u8,
                }
            }
            Turn::Capture { jumps } => {
                if jumps.is_empty() {
                    continue;
                }
                // Build capture chain: convert each (from, to) to (from, over, to).
                let mut capture_jumps = [(0u8, 0u8, 0u8); MAX_CAPTURE_JUMPS];
                let mut len = 0u8;

                for (from_coord, to_coord) in jumps.iter() {
                    let from_idx = match coord_to_index(&layout, *from_coord) {
                        Ok(v) => v,
                        Err(e) => {
                            return ReplayResult {
                                turns_played: i,
                                total_turns: total,
                                final_board: board,
                                error: Some(e),
                            }
                        }
                    };
                    let to_idx = match coord_to_index(&layout, *to_coord) {
                        Ok(v) => v,
                        Err(e) => {
                            return ReplayResult {
                                turns_played: i,
                                total_turns: total,
                                final_board: board,
                                error: Some(e),
                            }
                        }
                    };
                    // Try direct 2-step hop first.
                    if let Some(over_idx) = find_intermediate(nbrs, from_idx, to_idx) {
                        if (len as usize) >= MAX_CAPTURE_JUMPS {
                            return ReplayResult {
                                turns_played: i,
                                total_turns: total,
                                final_board: board,
                                error: Some(ReplayError::EngineError {
                                    turn: i,
                                    msg: format!("capture chain too long (>{MAX_CAPTURE_JUMPS} hops)"),
                                }),
                            };
                        }
                        capture_jumps[len as usize] = (from_idx as u8, over_idx as u8, to_idx as u8);
                        len += 1;
                    } else {
                        // Old-format multi-hop: find path via board state.
                        match find_capture_path(nbrs, board.rings(), from_idx, to_idx) {
                            Some(path) => {
                                for (f, o, t) in path {
                                    if (len as usize) >= MAX_CAPTURE_JUMPS {
                                        return ReplayResult {
                                            turns_played: i,
                                            total_turns: total,
                                            final_board: board,
                                            error: Some(ReplayError::EngineError {
                                                turn: i,
                                                msg: format!("capture chain too long (>{MAX_CAPTURE_JUMPS} hops)"),
                                            }),
                                        };
                                    }
                                    capture_jumps[len as usize] = (f as u8, o as u8, t as u8);
                                    len += 1;
                                }
                            }
                            None => {
                                return ReplayResult {
                                    turns_played: i,
                                    total_turns: total,
                                    final_board: board,
                                    error: Some(ReplayError::NoIntermediate {
                                        from: *from_coord,
                                        to: *to_coord,
                                    }),
                                }
                            }
                        }
                    }
                }

                ZertzMove::Capture {
                    jumps: capture_jumps,
                    len,
                }
            }
        };

        if verbose {
            println!("  -> {mv}");
        }
        if let Err(msg) = board.play_unchecked(mv) {
            if verbose {
                println!("ERROR at turn {i}: {msg}");
                println!("{board}");
            }
            return ReplayResult {
                turns_played: i,
                total_turns: total,
                final_board: board,
                error: Some(ReplayError::EngineError { turn: i, msg }),
            };
        }
        if verbose {
            println!("{board}");
        }
    }

    ReplayResult {
        turns_played: total,
        total_turns: total,
        final_board: board,
        error: None,
    }
}
