//! Replay parsed boardspace games on the Yinsh engine.

use std::collections::HashMap;
use std::path::Path;

use core_game::game::{Outcome, Player};
use yinsh_game::board::{Phase, YinshBoard, YinshMove, INITIAL_MARKERS};
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
// Stats
// ---------------------------------------------------------------------------

struct GameMoveData {
    turns_total: usize,
    final_board: YinshBoard,
    error: Option<ReplayError>,
    normal_valid: Vec<u32>,
    remove_row_valid: Vec<u32>,
    remove_ring_valid: Vec<u32>,
}

fn replay_collecting_stats(record: &GameRecord) -> GameMoveData {
    if !matches!(record.first_player_color, Color::White) {
        return GameMoveData {
            turns_total: record.turns.len(),
            final_board: YinshBoard::default(),
            error: Some(ReplayError::UnsupportedStartColor),
            normal_valid: vec![],
            remove_row_valid: vec![],
            remove_ring_valid: vec![],
        };
    }

    let mut board = YinshBoard::default();
    let mut normal_valid = Vec::new();
    let mut remove_row_valid = Vec::new();
    let mut remove_ring_valid = Vec::new();
    let total = record.turns.len();

    for (i, (_, turn)) in record.turns.iter().enumerate() {
        let phase = board.phase;
        let count = board.legal_moves().len() as u32;
        match phase {
            Phase::Setup => {}
            Phase::Normal => normal_valid.push(count),
            Phase::RemoveRow => remove_row_valid.push(count),
            Phase::RemoveRing => remove_ring_valid.push(count),
        }

        let mv = match turn_to_move(turn) {
            Ok(m) => m,
            Err(e) => return GameMoveData {
                turns_total: total,
                final_board: board,
                error: Some(e),
                normal_valid,
                remove_row_valid,
                remove_ring_valid,
            },
        };
        if let Err(msg) = board.apply_move(mv) {
            return GameMoveData {
                turns_total: total,
                final_board: board,
                error: Some(ReplayError::EngineError { turn: i, msg }),
                normal_valid,
                remove_row_valid,
                remove_ring_valid,
            };
        }
    }

    GameMoveData {
        turns_total: total,
        final_board: board,
        error: None,
        normal_valid,
        remove_row_valid,
        remove_ring_valid,
    }
}

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

    let mut p0_wins: u64 = 0;
    let mut p1_wins: u64 = 0;
    let mut draws: u64 = 0;
    let mut unknown_result: u64 = 0;

    let mut first_player_wins: u64 = 0;
    let mut second_player_wins: u64 = 0;

    let mut score_3_0: u64 = 0;
    let mut score_3_1: u64 = 0;
    let mut score_3_2: u64 = 0;
    let mut score_draw: u64 = 0;

    let mut game_lengths: Vec<u32> = Vec::new();
    let mut normal_turn_counts: Vec<u32> = Vec::new();
    let mut normal_valid_moves: Vec<u32> = Vec::new();
    let mut remove_row_valid_moves: Vec<u32> = Vec::new();
    let mut remove_ring_valid_moves: Vec<u32> = Vec::new();
    let mut markers_used: Vec<u32> = Vec::new();
    let mut rings_removed: Vec<u32> = Vec::new();
    let mut first_ring_pos: HashMap<String, u64> = HashMap::new();

    let mut process_zip = |zip_path: &Path| {
        let zip_name = zip_path.display().to_string();
        let file = match std::fs::File::open(zip_path) {
            Ok(f) => f,
            Err(e) => { eprintln!("  failed to open {zip_name}: {e}"); return; }
        };

        let _ = core_game::sgf::iter_sgf_texts_in_zip(file, |_sgf_name, text| {
            total_games += 1;
            if total_games % 2000 == 0 {
                eprint!("\r  {} games processed...   ", total_games);
            }

            let record = match sgf::parse_game(&text) {
                Ok(r) => r,
                Err(_) => { errors += 1; return; }
            };

            let result_str = core_game::sgf::result_from_metadata(&text, &record.player0, &record.player1);
            match result_str {
                "p0_wins" => p0_wins += 1,
                "p1_wins" => p1_wins += 1,
                "draw"    => draws += 1,
                _         => unknown_result += 1,
            }

            if let Some((_, Turn::PlaceRing { at, .. })) = record.turns.first() {
                let label = format!("{}{}", (b'A' + at.col) as char, at.row + 1);
                *first_ring_pos.entry(label).or_default() += 1;
            }

            let gs = replay_collecting_stats(&record);
            match gs.error {
                Some(ReplayError::UnsupportedStartColor) => { skipped += 1; return; }
                Some(_) => { errors += 1; return; }
                None => {}
            }
            replayed_ok += 1;

            let board = &gs.final_board;
            game_lengths.push(gs.turns_total as u32);
            normal_turn_counts.push(gs.normal_valid.len() as u32);
            normal_valid_moves.extend_from_slice(&gs.normal_valid);
            remove_row_valid_moves.extend_from_slice(&gs.remove_row_valid);
            remove_ring_valid_moves.extend_from_slice(&gs.remove_ring_valid);
            markers_used.push((INITIAL_MARKERS - board.markers_in_pool) as u32);
            rings_removed.push((board.white_score + board.black_score) as u32);

            match board.outcome {
                Outcome::WonBy(winner) => {
                    let min_s = board.white_score.min(board.black_score);
                    match min_s {
                        0 => score_3_0 += 1,
                        1 => score_3_1 += 1,
                        2 => score_3_2 += 1,
                        _ => {}
                    }
                    if winner == Player::Player1 {
                        first_player_wins += 1;
                    } else {
                        second_player_wins += 1;
                    }
                }
                Outcome::Draw    => score_draw += 1,
                Outcome::Ongoing => {}
            }
        });
        eprint!("\r  {zip_name}: {} total   ", total_games);
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

    println!();
    println!("=== Boardspace Yinsh Game Statistics ===");
    println!();
    println!("Games:   {} total, {} replayed OK, {} skipped (Black-first), {} errors",
        total_games, replayed_ok, skipped, errors);
    println!();

    let determined = p0_wins + p1_wins + draws;
    println!("--- Results (from metadata) ---");
    println!("  P0 wins:   {:6}  ({:.1}%)", p0_wins, pct(p0_wins, determined));
    println!("  P1 wins:   {:6}  ({:.1}%)", p1_wins, pct(p1_wins, determined));
    println!("  Draws:     {:6}  ({:.1}%)", draws, pct(draws, determined));
    println!("  Unknown:   {:6}", unknown_result);
    println!();

    let decided = first_player_wins + second_player_wins;
    println!("--- First/Second Player Advantage (engine-verified) ---");
    println!("  First player wins:  {:6}  ({:.1}%)", first_player_wins, pct(first_player_wins, decided));
    println!("  Second player wins: {:6}  ({:.1}%)", second_player_wins, pct(second_player_wins, decided));
    println!();

    let total_decisive = score_3_0 + score_3_1 + score_3_2;
    println!("--- Score Breakdown (engine-verified) ---");
    println!("  3-0:  {:6}  ({:.1}% of decisive)", score_3_0, pct(score_3_0, total_decisive));
    println!("  3-1:  {:6}  ({:.1}% of decisive)", score_3_1, pct(score_3_1, total_decisive));
    println!("  3-2:  {:6}  ({:.1}% of decisive)", score_3_2, pct(score_3_2, total_decisive));
    println!("  Draw: {:6}", score_draw);
    println!();

    print_dist("Game Length (all turns per game)", &mut game_lengths);
    print_dist("Normal-Phase Turns per Game (ring moves only)", &mut normal_turn_counts);
    print_dist("Valid Moves per Turn — Normal Phase", &mut normal_valid_moves);

    if !remove_row_valid_moves.is_empty() {
        let n = remove_row_valid_moves.len();
        let mut counts: HashMap<u32, u64> = HashMap::new();
        for &v in &remove_row_valid_moves { *counts.entry(v).or_default() += 1; }
        println!("--- Valid Moves per Turn — RemoveRow Phase (n={n}) ---");
        let mut kv: Vec<_> = counts.iter().collect();
        kv.sort_by_key(|(k, _)| *k);
        for (choices, freq) in &kv {
            println!("  {:2} choice(s): {:6}  ({:.1}%)", choices, freq, pct(**freq, n as u64));
        }
        println!();
    }

    if !remove_ring_valid_moves.is_empty() {
        let n = remove_ring_valid_moves.len();
        let mut counts: HashMap<u32, u64> = HashMap::new();
        for &v in &remove_ring_valid_moves { *counts.entry(v).or_default() += 1; }
        println!("--- Valid Moves per Turn — RemoveRing Phase (n={n}) ---");
        let mut kv: Vec<_> = counts.iter().collect();
        kv.sort_by_key(|(k, _)| *k);
        for (choices, freq) in &kv {
            println!("  {:2} ring(s) to choose from: {:6}  ({:.1}%)", choices, freq, pct(**freq, n as u64));
        }
        println!();
    }

    print_dist("Markers Used per Game (of 51)", &mut markers_used);

    if !rings_removed.is_empty() {
        let n = rings_removed.len();
        let mut counts: HashMap<u32, u64> = HashMap::new();
        for &v in &rings_removed { *counts.entry(v).or_default() += 1; }
        println!("--- Rings Removed per Game (white_score + black_score, max 6) ---");
        let mut kv: Vec<_> = counts.iter().collect();
        kv.sort_by_key(|(k, _)| *k);
        for (removed, freq) in &kv {
            println!("  {:2} ring(s): {:6}  ({:.1}%)", removed, freq, pct(**freq, n as u64));
        }
        println!();
    }

    println!("--- First Ring Placement (top 15) ---");
    let total_first = first_ring_pos.values().sum::<u64>();
    let mut fv: Vec<_> = first_ring_pos.iter().collect();
    fv.sort_by(|a, b| b.1.cmp(a.1));
    for (pos, count) in fv.iter().take(15) {
        println!("  {:4}  {:6}  ({:.1}%)", pos, count, pct(**count, total_first));
    }
    println!();

    let out_path = if path.is_dir() {
        path.join("game_stats.txt")
    } else {
        path.parent().unwrap_or(path).join("game_stats.txt")
    };
    if let Ok(mut f) = std::fs::File::create(&out_path) {
        use std::io::Write;
        let w = &mut f;
        writeln!(w, "Boardspace Yinsh Game Statistics").ok();
        writeln!(w, "================================").ok();
        writeln!(w).ok();
        writeln!(w, "Games: {} total, {} replayed OK, {} skipped, {} errors",
            total_games, replayed_ok, skipped, errors).ok();
        writeln!(w).ok();
        writeln!(w, "Results (metadata):").ok();
        writeln!(w, "  P0 wins: {} ({:.1}%)", p0_wins, pct(p0_wins, determined)).ok();
        writeln!(w, "  P1 wins: {} ({:.1}%)", p1_wins, pct(p1_wins, determined)).ok();
        writeln!(w, "  Draws:   {} ({:.1}%)", draws, pct(draws, determined)).ok();
        writeln!(w, "  Unknown: {}", unknown_result).ok();
        writeln!(w).ok();
        writeln!(w, "First/Second Player Advantage (engine-verified):").ok();
        writeln!(w, "  First player wins:  {} ({:.1}%)", first_player_wins, pct(first_player_wins, decided)).ok();
        writeln!(w, "  Second player wins: {} ({:.1}%)", second_player_wins, pct(second_player_wins, decided)).ok();
        writeln!(w).ok();
        writeln!(w, "Score Breakdown (engine-verified):").ok();
        writeln!(w, "  3-0:  {} ({:.1}%)", score_3_0, pct(score_3_0, total_decisive)).ok();
        writeln!(w, "  3-1:  {} ({:.1}%)", score_3_1, pct(score_3_1, total_decisive)).ok();
        writeln!(w, "  3-2:  {} ({:.1}%)", score_3_2, pct(score_3_2, total_decisive)).ok();
        writeln!(w, "  Draw: {}", score_draw).ok();
        writeln!(w).ok();
        write_dist(w, "Game Length (all turns)", &game_lengths);
        write_dist(w, "Normal-Phase Turns per Game", &normal_turn_counts);
        write_dist(w, "Valid Moves per Turn (Normal Phase)", &normal_valid_moves);
        write_dist(w, "Markers Used per Game", &markers_used);
        writeln!(w, "First Ring Placement (top 15):").ok();
        for (pos, count) in fv.iter().take(15) {
            writeln!(w, "  {}: {} ({:.1}%)", pos, count, pct(**count, total_first)).ok();
        }
        println!("Wrote {}", out_path.display());
    }
}

fn print_dist(label: &str, data: &mut Vec<u32>) {
    if data.is_empty() { return; }
    data.sort_unstable();
    let n = data.len();
    let sum: u64 = data.iter().map(|&x| x as u64).sum();
    let avg = sum as f64 / n as f64;
    let median = if n % 2 == 0 {
        (data[n / 2 - 1] + data[n / 2]) as f64 / 2.0
    } else {
        data[n / 2] as f64
    };
    println!("--- {label} (n={n}) ---");
    println!("  Min:    {:8}", data[0]);
    println!("  Max:    {:8}", data[n - 1]);
    println!("  Mean:   {:11.1}", avg);
    println!("  Median: {:11.1}", median);
    println!("  P10:    {:8}", data[(n / 10).max(1) - 1]);
    println!("  P25:    {:8}", data[n / 4]);
    println!("  P75:    {:8}", data[(n * 3 / 4).min(n - 1)]);
    println!("  P90:    {:8}", data[(n * 9 / 10).min(n - 1)]);
    println!();
}

fn write_dist(f: &mut std::fs::File, label: &str, data: &[u32]) {
    use std::io::Write;
    if data.is_empty() { return; }
    let mut sorted = data.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let sum: u64 = sorted.iter().map(|&x| x as u64).sum();
    writeln!(f, "{label}:").ok();
    writeln!(f, "  Min: {}, Max: {}, Mean: {:.1}, Median: {:.1}",
        sorted[0], sorted[n - 1], sum as f64 / n as f64,
        if n % 2 == 0 { (sorted[n/2-1]+sorted[n/2]) as f64 / 2.0 } else { sorted[n/2] as f64 }
    ).ok();
    writeln!(f, "  P10: {}, P25: {}, P75: {}, P90: {}",
        sorted[(n / 10).max(1) - 1], sorted[n / 4],
        sorted[(n * 3 / 4).min(n - 1)], sorted[(n * 9 / 10).min(n - 1)]
    ).ok();
    writeln!(f).ok();
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
