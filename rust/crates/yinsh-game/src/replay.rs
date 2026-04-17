//! Replay parsed boardspace games on the Yinsh engine.

use std::collections::HashMap;
use std::path::Path;

use yinsh_game::board::{YinshBoard, YinshMove};
use yinsh_game::hex::{ROW_DIRS, cell_index, is_valid};
use yinsh_game::sgf::{self, Color, Coord, GameRecord, Turn};

// ---------------------------------------------------------------------------
// Errors / result
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum ReplayError {
    BadCoord(Coord),
    UnsupportedStartColor,
    BadRowDirection { from: Coord, to: Coord },
    EngineError { turn: usize, msg: String },
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplayError::BadCoord(c) => write!(f, "bad coordinate: {c}"),
            ReplayError::UnsupportedStartColor => {
                write!(f, "game starts with Black (not yet supported)")
            }
            ReplayError::BadRowDirection { from, to } => {
                write!(f, "bad row direction from {from} to {to}")
            }
            ReplayError::EngineError { turn, msg } => {
                write!(f, "engine error at turn {turn}: {msg}")
            }
        }
    }
}

pub struct ReplayResult {
    pub turns_played: usize,
    pub total_turns: usize,
    pub final_board: YinshBoard,
    pub error: Option<ReplayError>,
}

// ---------------------------------------------------------------------------
// Single-game replay
// ---------------------------------------------------------------------------

fn coord_to_idx(c: Coord) -> Result<usize, ReplayError> {
    if !is_valid(c.col, c.row) {
        return Err(ReplayError::BadCoord(c));
    }
    Ok(cell_index(c.col, c.row))
}

fn turn_to_move(turn: &Turn) -> Result<YinshMove, ReplayError> {
    Ok(match turn {
        Turn::PlaceRing { at, .. } => YinshMove::PlaceRing(coord_to_idx(*at)?),
        Turn::MoveRing { from, to, .. } => YinshMove::MoveRing {
            from: coord_to_idx(*from)?,
            to: coord_to_idx(*to)?,
        },
        Turn::RemoveRow { from, to, .. } => {
            // SGF gives the two endpoints in either order. Engine stores rows as
            // (start, positive_row_dir_index), so swap endpoints when the SGF
            // orientation runs backwards along the axis.
            let dc = (to.col as i8 - from.col as i8).signum();
            let dr = (to.row as i8 - from.row as i8).signum();
            if let Some(dir) = ROW_DIRS.iter().position(|&d| d == (dc, dr)) {
                YinshMove::RemoveRow { start: coord_to_idx(*from)?, dir }
            } else if let Some(dir) = ROW_DIRS.iter().position(|&d| d == (-dc, -dr)) {
                YinshMove::RemoveRow { start: coord_to_idx(*to)?, dir }
            } else {
                return Err(ReplayError::BadRowDirection { from: *from, to: *to });
            }
        }
        Turn::RemoveRing { at, .. } => YinshMove::RemoveRing(coord_to_idx(*at)?),
    })
}

pub fn replay_game(record: &GameRecord) -> ReplayResult {
    replay_game_inner(record, false)
}

pub fn replay_game_verbose(record: &GameRecord) -> ReplayResult {
    replay_game_inner(record, true)
}

fn replay_game_inner(record: &GameRecord, verbose: bool) -> ReplayResult {
    if !matches!(record.first_player_color, Color::White) {
        return ReplayResult {
            turns_played: 0,
            total_turns: record.turns.len(),
            final_board: YinshBoard::default(),
            error: Some(ReplayError::UnsupportedStartColor),
        };
    }

    let mut board = YinshBoard::default();
    let total = record.turns.len();
    for (i, (player, turn)) in record.turns.iter().enumerate() {
        if verbose {
            println!("--- Turn {i} (P{player}): {turn:?}");
        }
        let mv = match turn_to_move(turn) {
            Ok(m) => m,
            Err(e) => {
                return ReplayResult {
                    turns_played: i,
                    total_turns: total,
                    final_board: board,
                    error: Some(e),
                };
            }
        };
        if let Err(msg) = board.apply_move(mv) {
            return ReplayResult {
                turns_played: i,
                total_turns: total,
                final_board: board,
                error: Some(ReplayError::EngineError { turn: i, msg }),
            };
        }
    }

    ReplayResult {
        turns_played: total,
        total_turns: total,
        final_board: board,
        error: None,
    }
}

// ---------------------------------------------------------------------------
// Batch replay
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
            Err(e) => {
                eprintln!("  failed to open {zip_name}: {e}");
                return;
            }
        };

        let before = self.total_games;
        let result = sgf::iter_games_in_zip(file, |sgf_name, record| {
            self.total_games += 1;
            let game_id = format!("{zip_name}/{sgf_name}");
            let result = replay_game(&record);
            if let Some(ref err) = result.error {
                let msg = format!("{err}");
                if matches!(err, ReplayError::UnsupportedStartColor) {
                    self.total_skipped += 1;
                } else {
                    self.total_fail += 1;
                    let category = msg.replace(|c: char| c.is_ascii_digit(), "#");
                    *self.error_categories.entry(category).or_insert(0) += 1;
                    if self.errors.len() < 10 {
                        self.errors.push((
                            game_id,
                            format!(
                                "{} (played {}/{})",
                                err, result.turns_played, result.total_turns
                            ),
                        ));
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
    println!("  Skipped:        {}", stats.total_skipped);
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
    println!("Players: {} vs {}", record.player0, record.player1);
    println!("Result:  {}", record.result);
    println!("Turns:   {}", record.turns.len());
    println!();

    let result = replay_game_verbose(&record);
    println!();
    if let Some(ref err) = result.error {
        println!(
            "REPLAY ERROR at turn {} of {}: {err}",
            result.turns_played, result.total_turns
        );
    } else {
        println!("Replay OK ({} turns)", result.turns_played);
    }
}
