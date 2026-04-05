/// Replay Boardspace Hive games through the Rust engine.

use std::collections::HashMap;
use std::path::Path;

use hive_game::game::Game;
use hive_game::piece::{PieceColor, player_pieces};
use hive_game::sgf;

// ---------------------------------------------------------------------------
// Single-game replay (public API used by pretrain / process)
// ---------------------------------------------------------------------------

pub struct ReplayResult {
    pub turns_played: usize,
    pub total_turns: usize,
    pub final_state: String,
    pub error: Option<String>,
    /// The game at the point replay stopped (for stats collection).
    pub game: Option<Game>,
}

/// Replay a game given its SGF content. Returns the result.
pub fn replay_game(content: &str) -> ReplayResult {
    let gtype = sgf::game_type(content);
    if gtype == "expansion" {
        return ReplayResult {
            turns_played: 0,
            total_turns: 0,
            final_state: String::new(),
            error: Some("expansion game (skipped)".to_string()),
            game: None,
        };
    }

    let mut game = Game::new();
    match sgf::replay_into_game(content, &mut game) {
        Ok(moves_played) => {
            let state = game.state.as_str().to_string();
            ReplayResult {
                turns_played: moves_played,
                total_turns: moves_played,
                final_state: state,
                error: None,
                game: Some(game),
            }
        }
        Err(msg) => {
            let mc = game.move_count as usize;
            ReplayResult {
                turns_played: mc,
                total_turns: 0,
                final_state: game.state.as_str().to_string(),
                error: Some(msg),
                game: Some(game),
            }
        }
    }
}

/// Replay a game with verbose output (prints each move).
pub fn replay_game_verbose(content: &str) -> ReplayResult {
    let gtype = sgf::game_type(content);
    let p0 = sgf::extract_player(content, 0);
    let p1 = sgf::extract_player(content, 1);
    let result_field = sgf::extract_prop(content, "RE").unwrap_or_default();

    println!("Game type: {}", gtype);
    println!("White (P0): {}  |  Black (P1): {}", p0, p1);
    println!("Result: {}", result_field);

    if gtype == "expansion" {
        println!("Skipping expansion game.");
        return ReplayResult {
            turns_played: 0,
            total_turns: 0,
            final_state: String::new(),
            error: Some("expansion game (skipped)".to_string()),
            game: None,
        };
    }

    println!();

    let mut game = Game::new();

    match sgf::replay_into_game_verbose(content, &mut game, |game_before, mv| {
        let i = game_before.move_count as usize;
        let color = if i % 2 == 0 { "White" } else { "Black" };
        let uhp = hive_game::uhp::format_move_uhp(game_before, mv);
        println!("Move {} ({}): {}", i + 1, color, uhp);
    }) {
        Ok(moves_played) => {
            println!("  State: {}", game.state.as_str());
            let state = game.state.as_str().to_string();
            ReplayResult {
                turns_played: moves_played,
                total_turns: moves_played,
                final_state: state,
                error: None,
                game: Some(game),
            }
        }
        Err(msg) => {
            println!("  ERROR: {}", msg);
            let valid = game.valid_moves();
            let valid_uhp: Vec<String> = valid.iter().map(|m| hive_game::uhp::format_move_uhp(&game, m)).collect();
            println!("  Valid moves ({}):", valid_uhp.len());
            for vm in valid_uhp.iter().take(20) {
                println!("    {}", vm);
            }
            if valid_uhp.len() > 20 {
                println!("    ... and {} more", valid_uhp.len() - 20);
            }
            let mc = game.move_count as usize;
            ReplayResult {
                turns_played: mc,
                total_turns: 0,
                final_state: game.state.as_str().to_string(),
                error: Some(msg),
                game: Some(game),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Board dimension statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct BoardDims {
    diameter: i8,
    q_span: i8,
    r_span: i8,
    s_span: i8,
    move_count: u16,
    white_pieces: u8,
    black_pieces: u8,
}

fn board_dims(game: &Game) -> BoardDims {
    let mut positions = Vec::new();
    let mut white_pieces: u8 = 0;
    let mut black_pieces: u8 = 0;
    for color in [PieceColor::White, PieceColor::Black] {
        for piece in player_pieces(color) {
            if let Some(pos) = game.board.piece_position(piece) {
                positions.push(pos);
                match color {
                    PieceColor::White => white_pieces += 1,
                    PieceColor::Black => black_pieces += 1,
                }
            }
        }
    }

    if positions.is_empty() {
        return BoardDims { diameter: 0, q_span: 0, r_span: 0, s_span: 0,
                           move_count: game.move_count, white_pieces: 0, black_pieces: 0 };
    }

    let mut min_q = i8::MAX; let mut max_q = i8::MIN;
    let mut min_r = i8::MAX; let mut max_r = i8::MIN;
    let mut min_s = i8::MAX; let mut max_s = i8::MIN;

    for (q, r) in &positions {
        let s = -q - r;
        min_q = min_q.min(*q); max_q = max_q.max(*q);
        min_r = min_r.min(*r); max_r = max_r.max(*r);
        min_s = min_s.min(s);  max_s = max_s.max(s);
    }

    let mut diameter: i8 = 0;
    for i in 0..positions.len() {
        for j in i+1..positions.len() {
            let (q1, r1) = positions[i];
            let (q2, r2) = positions[j];
            let s1 = -q1 - r1;
            let s2 = -q2 - r2;
            let d = (q1-q2).abs().max((r1-r2).abs()).max((s1-s2).abs());
            diameter = diameter.max(d);
        }
    }

    BoardDims {
        diameter,
        q_span: max_q - min_q,
        r_span: max_r - min_r,
        s_span: max_s - min_s,
        move_count: game.move_count,
        white_pieces,
        black_pieces,
    }
}

fn percentile(sorted: &[i8], p: f64) -> i8 {
    let idx = ((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_dimension_stats(dims: &[BoardDims]) {
    let n = dims.len();
    println!();
    println!("Board dimension statistics ({n} games):");
    println!();

    let mut diameters: Vec<i8> = dims.iter().map(|d| d.diameter).collect();
    let mut q_spans: Vec<i8>   = dims.iter().map(|d| d.q_span).collect();
    let mut r_spans: Vec<i8>   = dims.iter().map(|d| d.r_span).collect();
    let mut s_spans: Vec<i8>   = dims.iter().map(|d| d.s_span).collect();
    let mut white_pieces: Vec<i8> = dims.iter().map(|d| d.white_pieces as i8).collect();
    let mut black_pieces: Vec<i8> = dims.iter().map(|d| d.black_pieces as i8).collect();
    let mut total_pieces: Vec<i8> = dims.iter().map(|d| (d.white_pieces + d.black_pieces) as i8).collect();
    let mut move_counts_u16: Vec<u16> = dims.iter().map(|d| d.move_count).collect();

    diameters.sort(); q_spans.sort(); r_spans.sort(); s_spans.sort();
    white_pieces.sort(); black_pieces.sort(); total_pieces.sort();
    move_counts_u16.sort();

    let avg    = |v: &[i8]| -> f64 { v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64 };
    let avg_u16 = |v: &[u16]| -> f64 { v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64 };

    println!("  {:>18}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}", "", "min", "p25", "med", "avg", "p75", "max");

    macro_rules! row {
        ($label:expr, $v:expr) => {
            println!("  {:>18}  {:>5}  {:>5}  {:>5}  {:>5.1}  {:>5}  {:>5}",
                $label, $v[0], percentile($v, 25.0), percentile($v, 50.0),
                avg($v), percentile($v, 75.0), $v[n-1]);
        }
    }
    row!("diameter",     &diameters);
    row!("q span",       &q_spans);
    row!("r span",       &r_spans);
    row!("s span",       &s_spans);
    row!("white pieces", &white_pieces);
    row!("black pieces", &black_pieces);
    row!("total pieces", &total_pieces);

    let mc_p = |p: f64| -> u16 {
        let idx = ((move_counts_u16.len() as f64 - 1.0) * p / 100.0).round() as usize;
        move_counts_u16[idx.min(move_counts_u16.len() - 1)]
    };
    println!("  {:>18}  {:>5}  {:>5}  {:>5}  {:>5.1}  {:>5}  {:>5}",
        "move count", move_counts_u16[0], mc_p(25.0), mc_p(50.0),
        avg_u16(&move_counts_u16), mc_p(75.0), move_counts_u16[n-1]);
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
    dims: Vec<BoardDims>,
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
            dims: Vec::new(),
        }
    }

    fn process_zip(&mut self, path: &Path) {
        let zip_name = path.display().to_string();
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => { eprintln!("  failed to open {zip_name}: {e}"); return; }
        };

        let before = self.total_games;
        let _ = core_game::sgf::iter_sgf_texts_in_zip(file, |sgf_name, text| {
            self.total_games += 1;
            let game_id = format!("{zip_name}/{sgf_name}");

            let result = replay_game(&text);
            if let Some(ref err) = result.error {
                if err.contains("expansion") || err.contains("tournament") {
                    self.total_skipped += 1;
                } else {
                    self.total_fail += 1;
                    let category = err.chars()
                        .map(|c| if c.is_ascii_digit() { '#' } else { c })
                        .collect::<String>();
                    *self.error_categories.entry(category).or_insert(0) += 1;
                    if self.errors.len() < 10 {
                        self.errors.push((game_id, format!(
                            "{} (played {}/{})", err, result.turns_played, result.total_turns
                        )));
                    }
                }
            } else {
                self.total_ok += 1;
                if let Some(ref game) = result.game {
                    self.dims.push(board_dims(game));
                }
            }
        });

        let count = self.total_games - before;
        let ok = self.total_ok;
        eprint!("\r  {zip_name}: {count} games (total ok: {ok})   ");
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
    println!("  Skipped (expansion): {}", stats.total_skipped);
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

    if !stats.dims.is_empty() {
        print_dimension_stats(&stats.dims);
    }
}

// ---------------------------------------------------------------------------
// Debug: verbose replay of a single game
// ---------------------------------------------------------------------------

fn read_iso8859(path: &Path) -> String {
    let buf = std::fs::read(path).expect("failed to read file");
    buf.iter().map(|&b| b as char).collect()
}

fn debug_content(contents: &str) {
    println!("--- Raw SGF ---");
    println!("{contents}");
    println!();

    let result = replay_game_verbose(contents);
    println!();
    if let Some(ref err) = result.error {
        println!("REPLAY ERROR at turn {}: {err}", result.turns_played);
    } else {
        println!("Replay OK ({} turns)", result.turns_played);
    }
}

pub fn run_debug(args: &[String]) {
    let path = Path::new(&args[0]);

    if path.extension().is_some_and(|e| e == "sgf") {
        debug_content(&read_iso8859(path));
        return;
    }

    let sgf_name = args.get(1).expect("need sgf name when debugging from zip");
    let file = std::fs::File::open(path).expect("failed to open zip");
    let mut archive = zip::ZipArchive::new(file).expect("failed to read zip");

    let mut found = None;
    for i in 0..archive.len() {
        let entry = archive.by_index(i).unwrap();
        if entry.name().contains(sgf_name.as_str()) {
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
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut entry, &mut buf).expect("failed to read SGF");
    let contents: String = buf.iter().map(|&b| b as char).collect();
    debug_content(&contents);
}
