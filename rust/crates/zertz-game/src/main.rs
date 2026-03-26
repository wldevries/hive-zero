mod board_encoding;
pub mod hex;
mod mcts;
mod move_encoding;
mod parser;
mod random_play;
mod replay;
mod zertz;

use std::fs::File;
use std::io::Read;
use std::path::Path;

struct ReplayStats {
    total_games: usize,
    total_ok: usize,
    total_fail: usize,
    total_skipped: usize,
    errors: Vec<(String, String)>,
    error_categories: std::collections::HashMap<String, usize>,
}

impl ReplayStats {
    fn new() -> Self {
        ReplayStats {
            total_games: 0,
            total_ok: 0,
            total_fail: 0,
            total_skipped: 0,
            errors: Vec::new(),
            error_categories: std::collections::HashMap::new(),
        }
    }

    fn process_zip(&mut self, path: &Path) {
        let zip_name = path.display().to_string();
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("  failed to open {zip_name}: {e}");
                return;
            }
        };

        let before = self.total_games;
        let result = parser::iter_games_in_zip(file, |sgf_name, record| {
            self.total_games += 1;
            let game_id = format!("{zip_name}/{sgf_name}");

            let result = replay::replay_game(&record);
            if let Some(ref err) = result.error {
                let msg = format!("{err}");
                if msg.contains("tournament") {
                    self.total_skipped += 1;
                } else {
                    self.total_fail += 1;
                    // Categorize: strip turn numbers and positions for grouping.
                    let category = msg
                        .replace(|c: char| c.is_ascii_digit(), "#")
                        .to_string();
                    *self.error_categories.entry(category).or_insert(0) += 1;
                    if self.errors.len() < 10 {
                        self.errors.push((
                            game_id.clone(),
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
    println!("  Skipped (tournament): {}", stats.total_skipped);
    if stats.total_games > 0 {
        let pct = stats.total_ok as f64 / (stats.total_ok + stats.total_fail) as f64 * 100.0;
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

fn run_debug(zip_path: &str, sgf_name: &str) {
    let file = File::open(zip_path).expect("failed to open zip");
    let mut archive = zip::ZipArchive::new(file).expect("failed to read zip");

    // Find the SGF file (partial match on name).
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
        // List available files.
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
    entry.read_to_string(&mut contents).expect("failed to read SGF");

    println!("--- Raw SGF ---");
    println!("{contents}");
    println!();

    let record = parser::parse_game(&contents).expect("failed to parse game");
    println!("Variant: {:?}", record.variant);
    println!("Players: {} vs {}", record.player0, record.player1);
    println!("Result: {}", record.result);
    println!("Turns: {}", record.turns.len());
    println!();

    let result = replay::replay_game_verbose(&record);
    println!();
    if let Some(ref err) = result.error {
        println!("REPLAY ERROR at turn {}: {err}", result.turns_played);
    } else {
        println!("Replay OK ({} turns)", result.turns_played);
    }
}

fn run_mcts_demo(simulations: u32) {
    let board = zertz::ZertzBoard::default();
    let uniform_policy = vec![1.0 / move_encoding::POLICY_SIZE as f32; move_encoding::POLICY_SIZE];
    let mut search = mcts::search::MctsSearch::new(100_000);
    search.init(&board, &uniform_policy);

    let batch_size = 8;
    let rounds = simulations / batch_size as u32;

    println!("Running MCTS demo: {simulations} simulations (uniform policy)");
    println!("Board:\n{board}");
    println!("Legal moves: {}", board.legal_moves().len());

    for _ in 0..rounds {
        let leaves = search.select_leaves(batch_size);
        if leaves.is_empty() {
            break;
        }
        let policies: Vec<Vec<f32>> = leaves.iter().map(|_| uniform_policy.clone()).collect();
        let values: Vec<f32> = vec![0.0; leaves.len()];
        search.expand_and_backprop(&leaves, &policies, &values);
    }

    println!("Root visits: {}", search.arena.get(search.root).visit_count);
    println!("Root value:  {:.4}", search.root_value());
    println!();

    let dist = search.get_visit_distribution();
    println!("Top moves by visit share:");
    let mut sorted = dist.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (mv, prob) in sorted.iter().take(10) {
        println!("  {mv:?}  {prob:.4}");
    }

    if let Some(best) = search.best_move() {
        println!("\nBest move: {best:?}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("random");

    match mode {
        "random" => {
            let n: u32 = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(100);
            random_play::run_random_games(n);
        }
        "replay" => {
            let path = args
                .get(2)
                .map(|s| s.as_str())
                .unwrap_or("../games/boardspace");
            run_replay(path);
        }
        "debug" => {
            let zip_path = args.get(2).expect("need zip path");
            let sgf_name = args.get(3).expect("need sgf name");
            run_debug(zip_path, sgf_name);
        }
        "mcts" => {
            let sims: u32 = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(800);
            run_mcts_demo(sims);
        }
        _ => {
            eprintln!("Usage: zertz-zero <random [N]|replay [path]|debug <zip> <sgf>|mcts [sims]>");
            eprintln!("  random [N]     - play N random games (default 100)");
            eprintln!("  replay [path]  - replay boardspace games from zip dir/file");
            eprintln!("  debug <z> <s>  - verbose replay of a single game from zip");
            eprintln!("  mcts [sims]    - run MCTS demo with uniform policy (default 800)");
            std::process::exit(1);
        }
    }
}
