//! Replay parsed boardspace games on the Zertz engine.
//!
//! Maps boardspace coordinates to hex coordinates, reconstructs
//! `ZertzMove` values, and plays them on a `ZertzBoard`.

use std::collections::HashMap;
use std::path::Path;

use crate::hex::boardspace_to_hex;
use crate::sgf::{self, Color, Coord, GameRecord, Turn, Variant};
use crate::zertz::{
    find_capture_path, find_intermediate, Marble, ZertzBoard, ZertzMove, MAX_CAPTURE_JUMPS,
};

// ---------------------------------------------------------------------------
// Coordinate mapping
// ---------------------------------------------------------------------------

fn color_to_marble(c: Color) -> Marble {
    match c {
        Color::White => Marble::White,
        Color::Grey  => Marble::Grey,
        Color::Black => Marble::Black,
    }
}

fn coord_to_hex(coord: Coord) -> Result<crate::hex::Hex, ReplayError> {
    boardspace_to_hex(coord.col, coord.row).ok_or(ReplayError::BadCoord(coord))
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
// Single-game replay (public API)
// ---------------------------------------------------------------------------

/// Result of replaying a game.
#[derive(Debug)]
pub struct ReplayResult {
    pub turns_played: usize,
    pub total_turns: usize,
    pub final_board: ZertzBoard,
    pub error: Option<ReplayError>,
}

/// Replay a parsed game record on the engine.
pub fn replay_game(record: &GameRecord) -> ReplayResult {
    replay_game_inner(record, false)
}

/// Replay with verbose per-turn output for debugging.
pub fn replay_game_verbose(record: &GameRecord) -> ReplayResult {
    replay_game_inner(record, true)
}

fn replay_game_inner(record: &GameRecord, verbose: bool) -> ReplayResult {
    match &record.variant {
        Variant::Standard => {}
        Variant::Tournament { .. } => {
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
                let place_hex = match coord_to_hex(*at) {
                    Ok(v) => v,
                    Err(e) => return ReplayResult { turns_played: i, total_turns: total, final_board: board, error: Some(e) },
                };
                let remove_hex = match coord_to_hex(*remove) {
                    Ok(v) => v,
                    Err(e) => return ReplayResult { turns_played: i, total_turns: total, final_board: board, error: Some(e) },
                };
                ZertzMove::Place {
                    color: color_to_marble(*color),
                    place_at: place_hex,
                    remove: remove_hex,
                }
            }
            Turn::PlaceOnly { color, at } => {
                let place_hex = match coord_to_hex(*at) {
                    Ok(v) => v,
                    Err(e) => return ReplayResult { turns_played: i, total_turns: total, final_board: board, error: Some(e) },
                };
                ZertzMove::PlaceOnly {
                    color: color_to_marble(*color),
                    place_at: place_hex,
                }
            }
            Turn::Capture { jumps } => {
                if jumps.is_empty() { continue; }
                let mut capture_jumps = [((0i8, 0i8), (0i8, 0i8), (0i8, 0i8)); MAX_CAPTURE_JUMPS];
                let mut len = 0u8;

                for (from_coord, to_coord) in jumps.iter() {
                    let from_hex = match coord_to_hex(*from_coord) {
                        Ok(v) => v,
                        Err(e) => return ReplayResult { turns_played: i, total_turns: total, final_board: board, error: Some(e) },
                    };
                    let to_hex = match coord_to_hex(*to_coord) {
                        Ok(v) => v,
                        Err(e) => return ReplayResult { turns_played: i, total_turns: total, final_board: board, error: Some(e) },
                    };

                    if let Some(over_hex) = find_intermediate(board.rings(), from_hex, to_hex) {
                        if (len as usize) >= MAX_CAPTURE_JUMPS {
                            return ReplayResult { turns_played: i, total_turns: total, final_board: board,
                                error: Some(ReplayError::EngineError { turn: i, msg: format!("capture chain too long (>{MAX_CAPTURE_JUMPS} hops)") }) };
                        }
                        capture_jumps[len as usize] = (from_hex, over_hex, to_hex);
                        len += 1;
                    } else {
                        match find_capture_path(board.rings(), from_hex, to_hex) {
                            Some(path) => {
                                for (f, o, t) in path {
                                    if (len as usize) >= MAX_CAPTURE_JUMPS {
                                        return ReplayResult { turns_played: i, total_turns: total, final_board: board,
                                            error: Some(ReplayError::EngineError { turn: i, msg: format!("capture chain too long (>{MAX_CAPTURE_JUMPS} hops)") }) };
                                    }
                                    capture_jumps[len as usize] = (f, o, t);
                                    len += 1;
                                }
                            }
                            None => {
                                return ReplayResult { turns_played: i, total_turns: total, final_board: board,
                                    error: Some(ReplayError::NoIntermediate { from: *from_coord, to: *to_coord }) };
                            }
                        }
                    }
                }

                ZertzMove::Capture { jumps: capture_jumps, len }
            }
        };

        if verbose { println!("  -> {mv}"); }
        if let Err(msg) = board.play_unchecked(mv) {
            if verbose {
                println!("ERROR at turn {i}: {msg}");
                println!("{board}");
            }
            return ReplayResult { turns_played: i, total_turns: total, final_board: board,
                error: Some(ReplayError::EngineError { turn: i, msg }) };
        }
        if verbose { println!("{board}"); }
    }

    ReplayResult { turns_played: total, total_turns: total, final_board: board, error: None }
}

// ---------------------------------------------------------------------------
// Batch replay runner
// ---------------------------------------------------------------------------

struct ReplayStats {
    total_games: usize,
    total_ok: usize,
    total_fail: usize,
    total_skipped: usize,
    errors: Vec<(String, String)>,
    error_categories: HashMap<String, usize>,
}

impl ReplayStats {
    fn new() -> Self {
        ReplayStats {
            total_games: 0,
            total_ok: 0,
            total_fail: 0,
            total_skipped: 0,
            errors: Vec::new(),
            error_categories: HashMap::new(),
        }
    }

    fn process_zip(&mut self, path: &Path) {
        let zip_name = path.display().to_string();
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => { eprintln!("  failed to open {zip_name}: {e}"); return; }
        };

        let before = self.total_games;
        let result = sgf::iter_games_in_zip(file, |sgf_name, record| {
            self.total_games += 1;
            let game_id = format!("{zip_name}/{sgf_name}");

            let result = replay_game(&record);
            if let Some(ref err) = result.error {
                let msg = format!("{err}");
                if msg.contains("tournament") {
                    self.total_skipped += 1;
                } else {
                    self.total_fail += 1;
                    let category = msg.replace(|c: char| c.is_ascii_digit(), "#");
                    *self.error_categories.entry(category).or_insert(0) += 1;
                    if self.errors.len() < 10 {
                        self.errors.push((game_id, format!(
                            "{} (played {}/{})", err, result.turns_played, result.total_turns
                        )));
                    }
                }
            } else {
                self.total_ok += 1;
            }
        });

        let count = self.total_games - before;
        let ok = self.total_ok;
        eprint!("\r  {zip_name}: {count} games (total ok: {ok})   ");

        if let Err(e) = result {
            eprintln!("\n  zip error {zip_name}: {e}");
        }
    }
}

pub fn run_replay(games_path: &str) {
    let path = Path::new(games_path);
    if !path.exists() {
        eprintln!("Path not found: {games_path}");
        return;
    }

    let mut stats = ReplayStats::new();

    println!("Replaying games from {games_path}...");
    if path.is_file() && path.extension().is_some_and(|e| e == "zip") {
        stats.process_zip(path);
    } else if path.is_dir() {
        core_game::sgf::visit_zip_dir(path, &mut |p| stats.process_zip(p));
    } else {
        eprintln!("Not a zip file or directory: {games_path}");
        return;
    }

    eprintln!();
    println!();
    println!("Replay results:");
    println!("  Total games:    {}", stats.total_games);
    println!("  Replayed OK:    {}", stats.total_ok);
    println!("  Failed:         {}", stats.total_fail);
    println!("  Skipped (tournament): {}", stats.total_skipped);
    let non_skipped = stats.total_ok + stats.total_fail;
    if non_skipped > 0 {
        let pct = stats.total_ok as f64 / non_skipped as f64 * 100.0;
        println!("  Success rate:   {pct:.1}% (of non-skipped)");
    }

    if !stats.error_categories.is_empty() {
        println!();
        println!("Error categories:");
        let mut cats: Vec<_> = stats.error_categories.iter().collect();
        cats.sort_by(|a, b| b.1.cmp(a.1));
        for (cat, count) in &cats {
            println!("  {count:5}  {cat}");
        }
    }

    if !stats.errors.is_empty() {
        println!();
        println!("First {} errors:", stats.errors.len());
        for (game_id, err) in &stats.errors {
            println!("  {game_id}");
            println!("    {err}");
        }
    }
}

// ---------------------------------------------------------------------------
// Debug: verbose replay of a single game
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// process: compute ELO rankings, write CSVs
// ---------------------------------------------------------------------------

const K: f64 = 32.0;
const DEFAULT_ELO: f64 = 1500.0;

fn expected_score(ra: f64, rb: f64) -> f64 {
    1.0 / (1.0 + 10.0_f64.powf((rb - ra) / 400.0))
}

#[derive(Default)]
struct PlayerStats {
    games: u32,
    wins: u32,
    losses: u32,
    draws: u32,
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

pub fn run_process(games_path: &str, skip_timeout: bool) {
    let path = Path::new(games_path);
    if !path.exists() {
        eprintln!("Path not found: {games_path}");
        return;
    }

    let mut elo: HashMap<String, f64> = HashMap::new();
    let mut stats: HashMap<String, PlayerStats> = HashMap::new();
    let mut outcomes: Vec<String> = Vec::new();

    let mut total: usize = 0;
    let mut determined: usize = 0;
    let mut unknown_result: usize = 0;
    let mut parse_errors: usize = 0;
    let mut skipped_tournament: usize = 0;
    let mut skipped_timeout: usize = 0;

    println!("Processing games from {games_path}...");

    let mut process_zip = |zip_path: &Path| {
        let zip_name = zip_path.display().to_string();
        let file = match std::fs::File::open(zip_path) {
            Ok(f) => f,
            Err(e) => { eprintln!("  failed to open {zip_name}: {e}"); return; }
        };

        let _ = core_game::sgf::iter_sgf_texts_in_zip(file, |sgf_name, text| {
            total += 1;
            if total % 10000 == 0 {
                eprint!("\r  {} games processed ({} determined, {} unknown)...   ",
                        total, determined, unknown_result);
            }

            let record = match sgf::parse_game(&text) {
                Ok(r) => r,
                Err(_) => { parse_errors += 1; return; }
            };

            if matches!(record.variant, Variant::Tournament { .. }) {
                skipped_tournament += 1;
                return;
            }

            if skip_timeout && core_game::sgf::is_timeout(&text) {
                skipped_timeout += 1;
                return;
            }

            let p0 = &record.player0;
            let p1 = &record.player1;
            let move_count = record.turns.len();

            let result = core_game::sgf::result_from_metadata(&text, p0, p1);
            if result == "unknown" { unknown_result += 1; } else { determined += 1; }

            outcomes.push(format!("{},{},{},{},{},{}",
                csv_escape(&zip_name), csv_escape(sgf_name),
                csv_escape(p0), csv_escape(p1),
                move_count, result));

            for p in [p0, p1] {
                stats.entry(p.clone()).or_default().games += 1;
            }

            match result {
                "p0_wins" => {
                    stats.entry(p0.clone()).or_default().wins += 1;
                    stats.entry(p1.clone()).or_default().losses += 1;
                }
                "p1_wins" => {
                    stats.entry(p1.clone()).or_default().wins += 1;
                    stats.entry(p0.clone()).or_default().losses += 1;
                }
                "draw" => {
                    stats.entry(p0.clone()).or_default().draws += 1;
                    stats.entry(p1.clone()).or_default().draws += 1;
                }
                _ => return,
            }

            let ra = *elo.get(p0).unwrap_or(&DEFAULT_ELO);
            let rb = *elo.get(p1).unwrap_or(&DEFAULT_ELO);
            let ea = expected_score(ra, rb);
            let (sa, sb) = match result {
                "p0_wins" => (1.0, 0.0),
                "p1_wins" => (0.0, 1.0),
                _         => (0.5, 0.5),
            };
            elo.insert(p0.clone(), ra + K * (sa - ea));
            elo.insert(p1.clone(), rb + K * (sb - (1.0 - ea)));
        });
    };

    if path.is_file() && path.extension().is_some_and(|e| e == "zip") {
        process_zip(path);
    } else if path.is_dir() {
        core_game::sgf::visit_zip_dir(path, &mut process_zip);
    } else {
        eprintln!("Not a zip file or directory: {games_path}");
        return;
    }

    eprintln!();
    println!();
    println!("Done: {} games total, {} outcomes determined, {} unknown result, \
              {} parse errors, {} skipped tournament, {} skipped timeout",
        total, determined, unknown_result, parse_errors, skipped_tournament, skipped_timeout);

    let outcomes_path = path.join("game_outcomes.csv");
    if let Ok(mut f) = std::fs::File::create(&outcomes_path) {
        use std::io::Write;
        writeln!(f, "zip_file,sgf_name,p0,p1,move_count,result").ok();
        for line in &outcomes {
            writeln!(f, "{line}").ok();
        }
        println!("Wrote {}", outcomes_path.display());
    }

    let elo_path = path.join("player_elo.csv");
    if let Ok(mut f) = std::fs::File::create(&elo_path) {
        use std::io::Write;
        writeln!(f, "player,elo,games,wins,losses,draws").ok();
        let mut ranked: Vec<_> = stats.iter().collect();
        ranked.sort_by(|a, b| {
            let ea = elo.get(a.0).unwrap_or(&DEFAULT_ELO);
            let eb = elo.get(b.0).unwrap_or(&DEFAULT_ELO);
            eb.partial_cmp(ea).unwrap()
        });
        for (player, s) in ranked {
            let e = elo.get(player).unwrap_or(&DEFAULT_ELO);
            writeln!(f, "{},{:.1},{},{},{},{}", csv_escape(player), e, s.games, s.wins, s.losses, s.draws).ok();
        }
        println!("Wrote {}", elo_path.display());
    }
}

// ---------------------------------------------------------------------------
// Debug: verbose replay of a single game
// ---------------------------------------------------------------------------

pub fn run_debug(zip_path: &str, sgf_name: &str) {
    let file = std::fs::File::open(zip_path).expect("failed to open zip");
    let mut archive = zip::ZipArchive::new(file).expect("failed to read zip");

    let mut found = None;
    for i in 0..archive.len() {
        let entry = archive.by_index(i).unwrap();
        if entry.name().contains(sgf_name) {
            found = Some(i);
            println!("Found: {}", entry.name());
            break;
        }
    }

    let idx = found.unwrap_or_else(|| {
        eprintln!("SGF not found in zip: {sgf_name}");
        for i in 0..archive.len() {
            let entry = archive.by_index(i).unwrap();
            if entry.name().ends_with(".sgf") {
                eprintln!("  {}", entry.name());
            }
        }
        std::process::exit(1);
    });

    let mut entry = archive.by_index(idx).unwrap();
    let mut contents = String::new();
    std::io::Read::read_to_string(&mut entry, &mut contents).expect("failed to read SGF");

    println!("--- Raw SGF ---");
    println!("{contents}");
    println!();

    let record = sgf::parse_game(&contents).expect("failed to parse game");
    println!("Variant: {:?}", record.variant);
    println!("Players: {} vs {}", record.player0, record.player1);
    println!("Result: {}", record.result);
    println!("Turns: {}", record.turns.len());
    println!();

    let result = replay_game_verbose(&record);
    println!();
    if let Some(ref err) = result.error {
        println!("REPLAY ERROR at turn {}: {err}", result.turns_played);
    } else {
        println!("Replay OK ({} turns)", result.turns_played);
    }
}
