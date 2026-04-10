//! Replay parsed boardspace games on the Zertz engine.
//!
//! Maps boardspace coordinates to hex coordinates, reconstructs
//! `ZertzMove` values, and plays them on a `ZertzBoard`.

use std::collections::HashMap;
use std::path::Path;

use core_game::game::{Game, Outcome};
use core_game::hex::Hex;
use zertz_game::hex::boardspace_to_hex;
use zertz_game::zertz::{classify_win, WinType};
use zertz_game::sgf::{self, Color, Coord, GameRecord, Turn, Variant};
use zertz_game::zertz::{
    find_capture_path, find_intermediate, Marble, Ring, ZertzBoard, ZertzMove, MAX_CAPTURE_JUMPS,
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

fn coord_to_hex(coord: Coord) -> Result<Hex, ReplayError> {
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
// Stats: aggregate game statistics from boardspace games
// ---------------------------------------------------------------------------

pub fn run_stats(games_path: &str) {
    let path = Path::new(games_path);
    if !path.exists() {
        eprintln!("Path not found: {games_path}");
        return;
    }

    let mut total_games: u64 = 0;
    let mut replayed_ok: u64 = 0;
    let mut skipped: u64 = 0;
    let mut errors: u64 = 0;

    // Result tracking (from metadata)
    let mut p0_wins: u64 = 0;
    let mut p1_wins: u64 = 0;
    let mut draws: u64 = 0;
    let mut unknown_result: u64 = 0;

    // First player advantage
    let mut first_player_wins: u64 = 0;
    let mut second_player_wins: u64 = 0;

    // Win condition breakdown (from engine replay)
    let mut win_4w: u64 = 0;
    let mut win_5g: u64 = 0;
    let mut win_6b: u64 = 0;
    let mut win_3each: u64 = 0;

    // Game length
    let mut lengths: Vec<u32> = Vec::new();

    // Rings remaining at end
    let mut rings_remaining: Vec<u32> = Vec::new();

    // First move: color placed, ball position, and ring removed position
    let mut first_color: HashMap<String, u64> = HashMap::new();
    let mut first_ball_position: HashMap<String, u64> = HashMap::new();
    let mut first_ring_removed: HashMap<String, u64> = HashMap::new();

    let mut process_zip = |zip_path: &Path| {
        let zip_name = zip_path.display().to_string();
        let file = match std::fs::File::open(zip_path) {
            Ok(f) => f,
            Err(e) => { eprintln!("  failed to open {zip_name}: {e}"); return; }
        };

        let _ = core_game::sgf::iter_sgf_texts_in_zip(file, |_sgf_name, text| {
            total_games += 1;
            if total_games % 5000 == 0 {
                eprint!("\r  {} games processed...   ", total_games);
            }

            let record = match sgf::parse_game(&text) {
                Ok(r) => r,
                Err(_) => { errors += 1; return; }
            };

            if matches!(record.variant, Variant::Tournament { .. }) {
                skipped += 1;
                return;
            }

            let p0 = &record.player0;
            let p1 = &record.player1;
            let result_str = core_game::sgf::result_from_metadata(&text, p0, p1);

            match result_str {
                "p0_wins" => p0_wins += 1,
                "p1_wins" => p1_wins += 1,
                "draw" => draws += 1,
                _ => unknown_result += 1,
            }

            // First move analysis
            if let Some((_player, first_turn)) = record.turns.first() {
                match first_turn {
                    Turn::Place { color, at, remove } => {
                        let color_name = match color {
                            Color::White => "White",
                            Color::Grey => "Grey",
                            Color::Black => "Black",
                        };
                        *first_color.entry(color_name.to_string()).or_default() += 1;
                        *first_ball_position.entry(format!("{at}")).or_default() += 1;
                        *first_ring_removed.entry(format!("{remove}")).or_default() += 1;
                    }
                    Turn::PlaceOnly { color, at } => {
                        let color_name = match color {
                            Color::White => "White",
                            Color::Grey => "Grey",
                            Color::Black => "Black",
                        };
                        *first_color.entry(color_name.to_string()).or_default() += 1;
                        *first_ball_position.entry(format!("{at}")).or_default() += 1;
                    }
                    Turn::Capture { .. } => {
                        *first_color.entry("Capture".to_string()).or_default() += 1;
                    }
                }
            }

            // Replay the game for engine-derived stats
            let replay_result = replay_game(&record);
            if replay_result.error.is_some() {
                errors += 1;
                return;
            }
            replayed_ok += 1;

            let board = &replay_result.final_board;
            lengths.push(replay_result.turns_played as u32);

            // Count rings remaining (Empty or Occupied, not Removed)
            let remaining = board.rings().iter()
                .filter(|r| matches!(r, Ring::Empty | Ring::Occupied(_)))
                .count() as u32;
            rings_remaining.push(remaining);

            // Win condition and first/second player advantage from engine
            match board.outcome() {
                Outcome::WonBy(p) => {
                    match classify_win(board, p) {
                        WinType::FourWhite => win_4w += 1,
                        WinType::FiveGrey => win_5g += 1,
                        WinType::SixBlack => win_6b += 1,
                        WinType::ThreeEach => win_3each += 1,
                        WinType::Draw => {}
                    }
                    // Player1 = P0, Player2 = P1 in the engine
                    let winner_idx: u8 = match p {
                        core_game::game::Player::Player1 => 0,
                        core_game::game::Player::Player2 => 1,
                    };
                    if winner_idx == record.first_player {
                        first_player_wins += 1;
                    } else {
                        second_player_wins += 1;
                    }
                }
                _ => {}
            }
        });
    };

    println!("Computing stats from {games_path}...");
    if path.is_file() && path.extension().is_some_and(|e| e == "zip") {
        process_zip(path);
    } else if path.is_dir() {
        core_game::sgf::visit_zip_dir(path, &mut process_zip);
    } else {
        eprintln!("Not a zip file or directory: {games_path}");
        return;
    }
    eprintln!();

    // --- Print report ---
    println!();
    println!("=== Boardspace Zertz Game Statistics ===");
    println!();
    println!("Games:   {} total, {} replayed OK, {} skipped (tournament), {} errors",
        total_games, replayed_ok, skipped, errors);
    println!();

    // Results
    let determined = p0_wins + p1_wins + draws;
    println!("--- Results (from metadata) ---");
    println!("  P0 wins:   {:5}  ({:.1}%)", p0_wins, pct(p0_wins, determined));
    println!("  P1 wins:   {:5}  ({:.1}%)", p1_wins, pct(p1_wins, determined));
    println!("  Draws:     {:5}  ({:.1}%)", draws, pct(draws, determined));
    println!("  Unknown:   {:5}", unknown_result);
    println!();

    // First player advantage
    let decided = first_player_wins + second_player_wins;
    println!("--- First/Second Player Advantage (engine-verified) ---");
    println!("  First player wins:  {:5}  ({:.1}%)", first_player_wins, pct(first_player_wins, decided));
    println!("  Second player wins: {:5}  ({:.1}%)", second_player_wins, pct(second_player_wins, decided));
    println!();

    // Win conditions
    let total_wins = win_4w + win_5g + win_6b + win_3each;
    println!("--- Win Conditions (engine-verified) ---");
    println!("  4 White:   {:5}  ({:.1}%)", win_4w, pct(win_4w, total_wins));
    println!("  5 Grey:    {:5}  ({:.1}%)", win_5g, pct(win_5g, total_wins));
    println!("  6 Black:   {:5}  ({:.1}%)", win_6b, pct(win_6b, total_wins));
    println!("  3 Each:    {:5}  ({:.1}%)", win_3each, pct(win_3each, total_wins));
    println!();

    // Game length
    if !lengths.is_empty() {
        lengths.sort();
        let n = lengths.len();
        let sum: u64 = lengths.iter().map(|&x| x as u64).sum();
        let avg = sum as f64 / n as f64;
        let median = if n % 2 == 0 {
            (lengths[n / 2 - 1] + lengths[n / 2]) as f64 / 2.0
        } else {
            lengths[n / 2] as f64
        };
        println!("--- Game Length (turns) ---");
        println!("  Min:    {:5}", lengths[0]);
        println!("  Max:    {:5}", lengths[n - 1]);
        println!("  Mean:   {:8.1}", avg);
        println!("  Median: {:8.1}", median);
        println!("  P10:    {:5}", lengths[n / 10]);
        println!("  P90:    {:5}", lengths[n * 9 / 10]);
        println!();
    }

    // Rings remaining
    if !rings_remaining.is_empty() {
        rings_remaining.sort();
        let n = rings_remaining.len();
        let sum: u64 = rings_remaining.iter().map(|&x| x as u64).sum();
        let avg = sum as f64 / n as f64;
        let median = if n % 2 == 0 {
            (rings_remaining[n / 2 - 1] + rings_remaining[n / 2]) as f64 / 2.0
        } else {
            rings_remaining[n / 2] as f64
        };
        // Distribution
        let mut ring_counts: HashMap<u32, u64> = HashMap::new();
        for &r in &rings_remaining {
            *ring_counts.entry(r).or_default() += 1;
        }
        println!("--- Rings Remaining at Game End (of 37) ---");
        println!("  Min:    {:5}", rings_remaining[0]);
        println!("  Max:    {:5}", rings_remaining[n - 1]);
        println!("  Mean:   {:8.1}", avg);
        println!("  Median: {:8.1}", median);
        let mut sorted_rings: Vec<_> = ring_counts.iter().collect();
        sorted_rings.sort_by_key(|(k, _)| *k);
        println!("  Distribution:");
        for (count, freq) in sorted_rings {
            println!("    {:2} rings: {:5}  ({:.1}%)", count, freq, pct(*freq, n as u64));
        }
        println!();
    }

    // First move
    println!("--- First Move: Color Placed ---");
    let total_first = first_color.values().sum::<u64>();
    let mut fc: Vec<_> = first_color.iter().collect();
    fc.sort_by(|a, b| b.1.cmp(a.1));
    for (color, count) in &fc {
        println!("  {:8} {:5}  ({:.1}%)", color, count, pct(**count, total_first));
    }
    println!();

    println!("--- First Move: Ball Position (top 15) ---");
    let total_ball = first_ball_position.values().sum::<u64>();
    let mut fb: Vec<_> = first_ball_position.iter().collect();
    fb.sort_by(|a, b| b.1.cmp(a.1));
    for (pos, count) in fb.iter().take(15) {
        println!("  {:8} {:5}  ({:.1}%)", pos, count, pct(**count, total_ball));
    }
    println!();

    println!("--- First Move: Ring Removed (top 15) ---");
    let total_ring = first_ring_removed.values().sum::<u64>();
    let mut fr: Vec<_> = first_ring_removed.iter().collect();
    fr.sort_by(|a, b| b.1.cmp(a.1));
    for (pos, count) in fr.iter().take(15) {
        println!("  {:8} {:5}  ({:.1}%)", pos, count, pct(**count, total_ring));
    }
    println!();

    // Write stats to file
    let stats_path = path.join("game_stats.txt");
    if let Ok(mut f) = std::fs::File::create(&stats_path) {
        use std::io::Write;
        writeln!(f, "Boardspace Zertz Game Statistics").ok();
        writeln!(f, "================================").ok();
        writeln!(f).ok();
        writeln!(f, "Games: {} total, {} replayed OK, {} skipped, {} errors",
            total_games, replayed_ok, skipped, errors).ok();
        writeln!(f).ok();
        writeln!(f, "Results (metadata):").ok();
        writeln!(f, "  P0 wins: {} ({:.1}%)", p0_wins, pct(p0_wins, determined)).ok();
        writeln!(f, "  P1 wins: {} ({:.1}%)", p1_wins, pct(p1_wins, determined)).ok();
        writeln!(f, "  Draws: {} ({:.1}%)", draws, pct(draws, determined)).ok();
        writeln!(f, "  Unknown: {}", unknown_result).ok();
        writeln!(f).ok();
        writeln!(f, "First/Second Player Advantage (engine-verified):").ok();
        writeln!(f, "  First player wins: {} ({:.1}%)", first_player_wins, pct(first_player_wins, decided)).ok();
        writeln!(f, "  Second player wins: {} ({:.1}%)", second_player_wins, pct(second_player_wins, decided)).ok();
        writeln!(f).ok();
        writeln!(f, "Win Conditions (engine-verified):").ok();
        writeln!(f, "  4 White: {} ({:.1}%)", win_4w, pct(win_4w, total_wins)).ok();
        writeln!(f, "  5 Grey: {} ({:.1}%)", win_5g, pct(win_5g, total_wins)).ok();
        writeln!(f, "  6 Black: {} ({:.1}%)", win_6b, pct(win_6b, total_wins)).ok();
        writeln!(f, "  3 Each: {} ({:.1}%)", win_3each, pct(win_3each, total_wins)).ok();
        writeln!(f).ok();
        if !lengths.is_empty() {
            let n = lengths.len();
            let sum: u64 = lengths.iter().map(|&x| x as u64).sum();
            writeln!(f, "Game Length (turns):").ok();
            writeln!(f, "  Min: {}, Max: {}, Mean: {:.1}, Median: {:.1}",
                lengths[0], lengths[n-1],
                sum as f64 / n as f64,
                if n % 2 == 0 { (lengths[n/2-1]+lengths[n/2]) as f64 / 2.0 } else { lengths[n/2] as f64 }
            ).ok();
            writeln!(f).ok();
        }
        if !rings_remaining.is_empty() {
            let n = rings_remaining.len();
            let sum: u64 = rings_remaining.iter().map(|&x| x as u64).sum();
            writeln!(f, "Rings Remaining (of 37):").ok();
            writeln!(f, "  Min: {}, Max: {}, Mean: {:.1}, Median: {:.1}",
                rings_remaining[0], rings_remaining[n-1],
                sum as f64 / n as f64,
                if n % 2 == 0 { (rings_remaining[n/2-1]+rings_remaining[n/2]) as f64 / 2.0 } else { rings_remaining[n/2] as f64 }
            ).ok();
            writeln!(f).ok();
        }
        writeln!(f, "First Move Color:").ok();
        for (color, count) in &fc {
            writeln!(f, "  {}: {} ({:.1}%)", color, count, pct(**count, total_first)).ok();
        }
        writeln!(f).ok();
        writeln!(f, "First Move Ball Position (top 15):").ok();
        for (pos, count) in fb.iter().take(15) {
            writeln!(f, "  {}: {} ({:.1}%)", pos, count, pct(**count, total_ball)).ok();
        }
        writeln!(f).ok();
        writeln!(f, "First Move Ring Removed (top 15):").ok();
        for (pos, count) in fr.iter().take(15) {
            writeln!(f, "  {}: {} ({:.1}%)", pos, count, pct(**count, total_ring)).ok();
        }
        println!("Wrote {}", stats_path.display());
    }
}

fn pct(n: u64, total: u64) -> f64 {
    if total == 0 { 0.0 } else { n as f64 / total as f64 * 100.0 }
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

// ---------------------------------------------------------------------------
// Playback: interactive step-through of a randomly chosen boardspace game
// ---------------------------------------------------------------------------

/// Collect all parseable standard game records from a zip file.
fn collect_games_from_zip(path: &Path, out: &mut Vec<(String, GameRecord)>) {
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => { eprintln!("  failed to open {}: {e}", path.display()); return; }
    };
    let _ = sgf::iter_games_in_zip(file, |name, record| {
        if matches!(record.variant, Variant::Standard) {
            out.push((name.to_string(), record));
        }
    });
}

pub fn run_playback(games_path: &str, auto_ms: Option<u64>) {
    let path = Path::new(games_path);
    if !path.exists() {
        eprintln!("Path not found: {games_path}");
        return;
    }

    // Collect all standard games.
    let mut all_games: Vec<(String, GameRecord)> = Vec::new();
    if path.is_file() && path.extension().is_some_and(|e| e == "zip") {
        collect_games_from_zip(path, &mut all_games);
    } else if path.is_dir() {
        core_game::sgf::visit_zip_dir(path, &mut |p| collect_games_from_zip(p, &mut all_games));
    } else {
        eprintln!("Not a zip file or directory: {games_path}");
        return;
    }

    if all_games.is_empty() {
        eprintln!("No standard games found in {games_path}");
        return;
    }

    // Pick one at random.
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    let (sgf_name, record) = all_games.choose(&mut rng).unwrap();
    println!("Game: {sgf_name}");
    println!("  {} vs {}  result: {}", record.player0, record.player1, record.result);
    println!("  {} turns", record.turns.len());
    println!();

    // Step through the game, displaying the board after each turn.
    let mut board = ZertzBoard::default();
    println!("Initial position:");
    println!("{board}");

    for (i, (player, turn)) in record.turns.iter().enumerate() {
        // Describe the move.
        let p_name = if *player == 0 { &record.player0 } else { &record.player1 };
        let turn_desc = match turn {
            Turn::Place { color, at, remove } =>
                format!("Place {:?} at {} remove {}", color, at, remove),
            Turn::PlaceOnly { color, at } =>
                format!("Place {:?} at {}", color, at),
            Turn::Capture { jumps } => {
                let hops: Vec<String> = jumps.iter()
                    .map(|(from, to)| format!("{from}→{to}"))
                    .collect();
                format!("Capture {}", hops.join(" "))
            }
        };
        println!("Turn {} | P{player} ({p_name}): {turn_desc}", i + 1);

        // Build and apply the move (reuse replay_game_inner logic via coord helpers).
        let mv = match turn {
            Turn::Place { color, at, remove } => {
                let place_hex = match coord_to_hex(*at) { Ok(h) => h, Err(e) => { eprintln!("  bad coord: {e}"); break; } };
                let remove_hex = match coord_to_hex(*remove) { Ok(h) => h, Err(e) => { eprintln!("  bad coord: {e}"); break; } };
                ZertzMove::Place { color: color_to_marble(*color), place_at: place_hex, remove: remove_hex }
            }
            Turn::PlaceOnly { color, at } => {
                let place_hex = match coord_to_hex(*at) { Ok(h) => h, Err(e) => { eprintln!("  bad coord: {e}"); break; } };
                ZertzMove::PlaceOnly { color: color_to_marble(*color), place_at: place_hex }
            }
            Turn::Capture { jumps } => {
                if jumps.is_empty() { continue; }
                let mut capture_jumps = [((0i8,0i8),(0i8,0i8),(0i8,0i8)); MAX_CAPTURE_JUMPS];
                let mut len = 0u8;
                let mut ok = true;
                for (from_c, to_c) in jumps {
                    let from_h = match coord_to_hex(*from_c) { Ok(h) => h, Err(e) => { eprintln!("  bad coord: {e}"); ok = false; break; } };
                    let to_h   = match coord_to_hex(*to_c)   { Ok(h) => h, Err(e) => { eprintln!("  bad coord: {e}"); ok = false; break; } };
                    if let Some(over_h) = find_intermediate(board.rings(), from_h, to_h) {
                        if (len as usize) >= MAX_CAPTURE_JUMPS { eprintln!("  capture chain too long"); ok = false; break; }
                        capture_jumps[len as usize] = (from_h, over_h, to_h);
                        len += 1;
                    } else {
                        match find_capture_path(board.rings(), from_h, to_h) {
                            Some(path) => {
                                for (f, o, t) in path {
                                    if (len as usize) >= MAX_CAPTURE_JUMPS { eprintln!("  capture chain too long"); ok = false; break; }
                                    capture_jumps[len as usize] = (f, o, t);
                                    len += 1;
                                }
                            }
                            None => { eprintln!("  no path between {from_c} and {to_c}"); ok = false; break; }
                        }
                    }
                }
                if !ok { break; }
                ZertzMove::Capture { jumps: capture_jumps, len }
            }
        };

        if let Err(e) = board.play_unchecked(mv) {
            eprintln!("  engine error: {e}");
            break;
        }

        // Wait or sleep between turns.
        match auto_ms {
            None => {
                // Press Enter to advance.
                use std::io::{BufRead, Write};
                print!("  [Enter for next move, q+Enter to quit] ");
                let _ = std::io::stdout().flush();
                let stdin = std::io::stdin();
                let line = stdin.lock().lines().next().and_then(|l| l.ok()).unwrap_or_default();
                if line.trim() == "q" { break; }
            }
            Some(ms) => {
                std::thread::sleep(std::time::Duration::from_millis(ms));
            }
        }

        println!("{board}");
    }

    println!("Game over.");
    use core_game::game::Outcome;
    match board.outcome() {
        Outcome::WonBy(p) => println!("Winner: P{}", if p == core_game::game::Player::Player1 { 1 } else { 2 }),
        Outcome::Draw    => println!("Draw"),
        Outcome::Ongoing => println!("(game ended early)"),
    }
}
