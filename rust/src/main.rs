mod sgf;
mod replay;

use hive_engine::game::Game;
use hive_engine::piece::{PieceColor, player_pieces, PIECES_PER_PLAYER};

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

/// Board dimensions in cube coordinates for a single game.
#[derive(Debug, Clone, Copy)]
struct BoardDims {
    /// Max distance between any two pieces (diameter).
    diameter: i8,
    /// Bounding box: range of q, r, s (cube) coordinates.
    q_span: i8,
    r_span: i8,
    s_span: i8,
    move_count: u16,
    /// Number of white pieces on the board (placed from reserve).
    white_pieces: u8,
    /// Number of black pieces on the board.
    black_pieces: u8,
}

/// Compute board dimensions from a game state.
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
        return BoardDims {
            diameter: 0,
            q_span: 0,
            r_span: 0,
            s_span: 0,
            move_count: game.move_count,
            white_pieces: 0,
            black_pieces: 0,
        };
    }

    let mut min_q = i8::MAX;
    let mut max_q = i8::MIN;
    let mut min_r = i8::MAX;
    let mut max_r = i8::MIN;
    let mut min_s = i8::MAX;
    let mut max_s = i8::MIN;

    for &(q, r) in &positions {
        let s = -q - r; // cube coordinate constraint: q + r + s = 0
        min_q = min_q.min(q);
        max_q = max_q.max(q);
        min_r = min_r.min(r);
        max_r = max_r.max(r);
        min_s = min_s.min(s);
        max_s = max_s.max(s);
    }

    // Diameter: max hex distance between any two occupied positions
    let mut diameter: i8 = 0;
    for i in 0..positions.len() {
        for j in i + 1..positions.len() {
            let d = hive_engine::hex::hex_distance(positions[i], positions[j]);
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

struct ReplayStats {
    total_games: usize,
    total_ok: usize,
    total_fail: usize,
    total_skipped: usize,
    errors: Vec<(String, String)>,
    error_categories: HashMap<String, usize>,
    /// Board dimensions of all successfully replayed games.
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
            Err(e) => {
                eprintln!("  failed to open {zip_name}: {e}");
                return;
            }
        };

        let mut archive = match zip::ZipArchive::new(file) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("  failed to read zip {zip_name}: {e}");
                return;
            }
        };

        let before = self.total_games;
        for i in 0..archive.len() {
            let mut entry = match archive.by_index(i) {
                Ok(e) => e,
                Err(_) => continue,
            };
            if !entry.name().ends_with(".sgf") {
                continue;
            }
            let sgf_name = entry.name().to_string();
            let mut contents = String::new();
            if entry.read_to_string(&mut contents).is_err() {
                // Try reading as iso-8859-1
                let mut buf = Vec::new();
                // Re-read since read_to_string consumed it
                drop(entry);
                let mut entry = archive.by_index(i).unwrap();
                buf.clear();
                let _ = std::io::Read::read_to_end(&mut entry, &mut buf);
                contents = buf.iter().map(|&b| b as char).collect();
            }

            self.total_games += 1;
            let game_id = format!("{}/{}", zip_name, sgf_name);

            let result = replay::replay_game(&contents);
            if let Some(ref err) = result.error {
                if err.contains("expansion") || err.contains("tournament") {
                    self.total_skipped += 1;
                } else {
                    self.total_fail += 1;
                    let category = err
                        .chars()
                        .map(|c| if c.is_ascii_digit() { '#' } else { c })
                        .collect::<String>();
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
                if let Some(ref game) = result.game {
                    self.dims.push(board_dims(game));
                }
            }
        }

        let count = self.total_games - before;
        let ok = self.total_ok;
        eprint!("\r  {zip_name}: {count} games (total ok: {ok})   ");
    }

    fn visit_dir(&mut self, dir: &Path) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        let mut paths: Vec<_> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            if path.is_dir() {
                self.visit_dir(&path);
            } else if path.extension().is_some_and(|e| e == "zip") {
                self.process_zip(&path);
            }
        }
    }
}

fn run_replay(games_path: &str) {
    let path = Path::new(games_path);
    if !path.exists() {
        eprintln!("Path not found: {games_path}");
        return;
    }

    let mut stats = ReplayStats::new();

    if path.is_file() && path.extension().is_some_and(|e| e == "zip") {
        println!("Replaying games from {games_path}...");
        stats.process_zip(path);
    } else if path.is_dir() {
        println!("Replaying games from {games_path}...");
        stats.visit_dir(path);
    } else {
        eprintln!("Not a zip file or directory: {games_path}");
        return;
    }

    eprintln!(); // newline after progress
    println!();
    println!("Replay results:");
    println!("  Total games:    {}", stats.total_games);
    println!("  Replayed OK:    {}", stats.total_ok);
    println!("  Failed:         {}", stats.total_fail);
    println!("  Skipped (expansion): {}", stats.total_skipped);
    if stats.total_games > 0 {
        let non_skipped = stats.total_ok + stats.total_fail;
        if non_skipped > 0 {
            let pct = stats.total_ok as f64 / non_skipped as f64 * 100.0;
            println!("  Success rate:   {pct:.1}% (of non-skipped)");
        }
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

    // Board dimension statistics
    if !stats.dims.is_empty() {
        print_dimension_stats(&stats.dims);
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

    // Collect each metric
    let mut diameters: Vec<i8> = dims.iter().map(|d| d.diameter).collect();
    let mut q_spans: Vec<i8> = dims.iter().map(|d| d.q_span).collect();
    let mut r_spans: Vec<i8> = dims.iter().map(|d| d.r_span).collect();
    let mut s_spans: Vec<i8> = dims.iter().map(|d| d.s_span).collect();
    let mut white_pieces: Vec<i8> = dims.iter().map(|d| d.white_pieces as i8).collect();
    let mut black_pieces: Vec<i8> = dims.iter().map(|d| d.black_pieces as i8).collect();
    let mut total_pieces: Vec<i8> = dims.iter().map(|d| (d.white_pieces + d.black_pieces) as i8).collect();
    let mut move_counts: Vec<i8> = dims.iter().map(|d| d.move_count.min(127) as i8).collect();

    // Use u16 for move counts since they can exceed 127
    let mut move_counts_u16: Vec<u16> = dims.iter().map(|d| d.move_count).collect();
    move_counts_u16.sort();

    diameters.sort();
    q_spans.sort();
    r_spans.sort();
    s_spans.sort();
    white_pieces.sort();
    black_pieces.sort();
    total_pieces.sort();

    let avg = |v: &[i8]| -> f64 { v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64 };
    let avg_u16 = |v: &[u16]| -> f64 { v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64 };

    println!("  {:>18}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}",
             "", "min", "p25", "med", "avg", "p75", "max");
    println!("  {:>18}  {:>5}  {:>5}  {:>5}  {:>5.1}  {:>5}  {:>5}",
             "diameter", diameters[0], percentile(&diameters, 25.0),
             percentile(&diameters, 50.0), avg(&diameters),
             percentile(&diameters, 75.0), diameters[n - 1]);
    println!("  {:>18}  {:>5}  {:>5}  {:>5}  {:>5.1}  {:>5}  {:>5}",
             "q span", q_spans[0], percentile(&q_spans, 25.0),
             percentile(&q_spans, 50.0), avg(&q_spans),
             percentile(&q_spans, 75.0), q_spans[n - 1]);
    println!("  {:>18}  {:>5}  {:>5}  {:>5}  {:>5.1}  {:>5}  {:>5}",
             "r span", r_spans[0], percentile(&r_spans, 25.0),
             percentile(&r_spans, 50.0), avg(&r_spans),
             percentile(&r_spans, 75.0), r_spans[n - 1]);
    println!("  {:>18}  {:>5}  {:>5}  {:>5}  {:>5.1}  {:>5}  {:>5}",
             "s span", s_spans[0], percentile(&s_spans, 25.0),
             percentile(&s_spans, 50.0), avg(&s_spans),
             percentile(&s_spans, 75.0), s_spans[n - 1]);
    println!("  {:>18}  {:>5}  {:>5}  {:>5}  {:>5.1}  {:>5}  {:>5}",
             "white pieces", white_pieces[0], percentile(&white_pieces, 25.0),
             percentile(&white_pieces, 50.0), avg(&white_pieces),
             percentile(&white_pieces, 75.0), white_pieces[n - 1]);
    println!("  {:>18}  {:>5}  {:>5}  {:>5}  {:>5.1}  {:>5}  {:>5}",
             "black pieces", black_pieces[0], percentile(&black_pieces, 25.0),
             percentile(&black_pieces, 50.0), avg(&black_pieces),
             percentile(&black_pieces, 75.0), black_pieces[n - 1]);
    println!("  {:>18}  {:>5}  {:>5}  {:>5}  {:>5.1}  {:>5}  {:>5}",
             "total pieces", total_pieces[0], percentile(&total_pieces, 25.0),
             percentile(&total_pieces, 50.0), avg(&total_pieces),
             percentile(&total_pieces, 75.0), total_pieces[n - 1]);

    let mc_p = |p: f64| -> u16 {
        let idx = ((move_counts_u16.len() as f64 - 1.0) * p / 100.0).round() as usize;
        move_counts_u16[idx.min(move_counts_u16.len() - 1)]
    };
    println!("  {:>18}  {:>5}  {:>5}  {:>5}  {:>5.1}  {:>5}  {:>5}",
             "move count", move_counts_u16[0], mc_p(25.0),
             mc_p(50.0), avg_u16(&move_counts_u16),
             mc_p(75.0), move_counts_u16[n - 1]);
}

fn run_debug(zip_path: &str, sgf_name: &str) {
    let file = std::fs::File::open(zip_path).expect("failed to open zip");
    let mut archive = zip::ZipArchive::new(file).expect("failed to read zip");

    // Find the SGF file (partial match on name)
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
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut entry, &mut buf).expect("failed to read SGF");
    // Decode as iso-8859-1
    let contents: String = buf.iter().map(|&b| b as char).collect();

    println!("--- Raw SGF ---");
    println!("{contents}");
    println!();

    let result = replay::replay_game_verbose(&contents);
    println!();
    if let Some(ref err) = result.error {
        println!("REPLAY ERROR at turn {}: {err}", result.turns_played);
    } else {
        println!("Replay OK ({} turns)", result.turns_played);
    }
}

fn run_random(n: u32) {
    use rand::Rng;

    println!("Playing {n} random games...");
    let mut total_moves = 0u64;
    let mut results = HashMap::new();

    for _ in 0..n {
        let mut game = Game::new();
        let mut rng = rand::thread_rng();

        while !game.is_game_over() {
            let moves = game.valid_moves();
            if moves.is_empty() {
                game.play_pass();
            } else {
                let idx = rng.gen_range(0..moves.len());
                game.play_move(&moves[idx]);
            }
            total_moves += 1;
        }

        *results.entry(game.state.as_str().to_string()).or_insert(0u32) += 1;
    }

    println!("Total moves: {total_moves}");
    println!("Avg moves/game: {:.1}", total_moves as f64 / n as f64);
    println!("Results:");
    let mut sorted: Vec<_> = results.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (state, count) in sorted {
        println!("  {state}: {count}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match mode {
        "random" => {
            let n: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            run_random(n);
        }
        "replay" => {
            let path = args
                .get(2)
                .map(|s| s.as_str())
                .unwrap_or("games/boardspace");
            run_replay(path);
        }
        "debug" => {
            let zip_path = args.get(2).expect("need zip path");
            let sgf_name = args.get(3).expect("need sgf name");
            run_debug(zip_path, sgf_name);
        }
        _ => {
            eprintln!("Usage: hive-zero <random [N]|replay [path]|debug <zip> <sgf>>");
            eprintln!("  random [N]     - play N random games (default 100)");
            eprintln!("  replay [path]  - replay boardspace games from zip dir/file");
            eprintln!("  debug <z> <s>  - verbose replay of a single game from zip");
            std::process::exit(1);
        }
    }
}
