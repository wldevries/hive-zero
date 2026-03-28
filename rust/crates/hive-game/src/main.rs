mod replay;

use hive_game::sgf;
use hive_game::game::Game;

use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// process: replay all games, compute ELO, write CSVs
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

fn run_process(games_path: &str, skip_timeout: bool) {
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
    let mut errors: usize = 0;
    let mut skipped_expansion: usize = 0;
    let mut skipped_timeout: usize = 0;

    let process_content = |content: &str,
                           zip_name: &str,
                           sgf_name: &str,
                           total: &mut usize,
                           determined: &mut usize,
                           errors: &mut usize,
                           skipped_expansion: &mut usize,
                           skipped_timeout: &mut usize,
                           elo: &mut HashMap<String, f64>,
                           stats: &mut HashMap<String, PlayerStats>,
                           outcomes: &mut Vec<String>,
                           skip_timeout: bool| {
        *total += 1;
        if *total % 10000 == 0 {
            eprint!("\r  {} games processed ({} determined, {} errors)...   ",
                    total, determined, errors);
        }

        let gtype = sgf::game_type(content);
        if gtype == "expansion" {
            *skipped_expansion += 1;
            return;
        }

        if skip_timeout && sgf::is_timeout(content) {
            *skipped_timeout += 1;
            return;
        }

        let p0 = sgf::extract_player(content, 0);
        let p1 = sgf::extract_player(content, 1);

        let mut game = Game::new();
        let replay_result = sgf::replay_into_game(content, &mut game);
        let move_count = game.move_count;

        let result = match &replay_result {
            Ok(_) => {
                match game.state.as_str() {
                    "WhiteWins" => { *determined += 1; "p0_wins" }
                    "BlackWins" => { *determined += 1; "p1_wins" }
                    "Draw"      => { *determined += 1; "draw" }
                    _ => {
                        let r = sgf::result_from_metadata(content, &p0, &p1);
                        if r != "unknown" { *determined += 1; }
                        r
                    }
                }
            }
            Err(_) => {
                let r = sgf::result_from_metadata(content, &p0, &p1);
                if r == "unknown" { *errors += 1; } else { *determined += 1; }
                r
            }
        };

        outcomes.push(format!("{},{},{},{},{},{},{}",
            csv_escape(zip_name), csv_escape(sgf_name),
            csv_escape(&p0), csv_escape(&p1),
            gtype, move_count, result));

        for p in [&p0, &p1] {
            stats.entry(p.clone()).or_default().games += 1;
        }

        if result == "p0_wins" || result == "p1_wins" || result == "draw" {
            match result {
                "p0_wins" => {
                    stats.entry(p0.clone()).or_default().wins += 1;
                    stats.entry(p1.clone()).or_default().losses += 1;
                }
                "p1_wins" => {
                    stats.entry(p1.clone()).or_default().wins += 1;
                    stats.entry(p0.clone()).or_default().losses += 1;
                }
                _ => {
                    stats.entry(p0.clone()).or_default().draws += 1;
                    stats.entry(p1.clone()).or_default().draws += 1;
                }
            }
            let ra = *elo.get(&p0).unwrap_or(&DEFAULT_ELO);
            let rb = *elo.get(&p1).unwrap_or(&DEFAULT_ELO);
            let ea = expected_score(ra, rb);
            let (sa, sb) = match result {
                "p0_wins" => (1.0, 0.0),
                "p1_wins" => (0.0, 1.0),
                _         => (0.5, 0.5),
            };
            elo.insert(p0, ra + K * (sa - ea));
            elo.insert(p1, rb + K * (sb - (1.0 - ea)));
        }
    };

    println!("Processing games from {games_path}...");

    let mut zip_paths: Vec<_> = Vec::new();
    collect_zips(path, &mut zip_paths);
    zip_paths.sort();

    for zip_path in &zip_paths {
        let zip_name = zip_path.display().to_string();
        let file = match std::fs::File::open(zip_path) {
            Ok(f) => f,
            Err(e) => { eprintln!("  failed to open {zip_name}: {e}"); continue; }
        };
        let _ = core_game::sgf::iter_sgf_texts_in_zip(file, |sgf_name, text| {
            process_content(&text, &zip_name, sgf_name,
                &mut total, &mut determined, &mut errors,
                &mut skipped_expansion, &mut skipped_timeout,
                &mut elo, &mut stats, &mut outcomes, skip_timeout);
        });
    }

    let mut sgf_paths: Vec<_> = Vec::new();
    collect_sgfs(path, &mut sgf_paths);
    sgf_paths.sort();

    for sgf_path in &sgf_paths {
        let buf = match std::fs::read(sgf_path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let content: String = buf.iter().map(|&b| b as char).collect();
        let parent = sgf_path.parent()
            .map(|p| p.file_name().unwrap_or_default().to_string_lossy().to_string())
            .unwrap_or_default();
        let name = sgf_path.file_name().unwrap_or_default().to_string_lossy().to_string();
        process_content(&content, &parent, &name,
            &mut total, &mut determined, &mut errors,
            &mut skipped_expansion, &mut skipped_timeout,
            &mut elo, &mut stats, &mut outcomes, skip_timeout);
    }

    eprintln!();
    println!();
    println!("Done: {} games total, {} outcomes determined, {} parse errors, \
              {} skipped expansion, {} skipped timeout, {} unknown",
        total, determined, errors, skipped_expansion, skipped_timeout,
        total - determined - errors - skipped_expansion - skipped_timeout);

    let outcomes_path = Path::new(games_path).join("game_outcomes.csv");
    if let Ok(mut f) = std::fs::File::create(&outcomes_path) {
        use std::io::Write;
        writeln!(f, "zip_file,sgf_name,p0,p1,game_type,move_count,result").ok();
        for line in &outcomes {
            writeln!(f, "{}", line).ok();
        }
        println!("Wrote {}", outcomes_path.display());
    }

    let elo_path = Path::new(games_path).join("player_elo.csv");
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

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn collect_zips(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    if dir.is_file() && dir.extension().is_some_and(|e| e == "zip") {
        out.push(dir.to_path_buf());
        return;
    }
    if !dir.is_dir() { return; }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() { collect_zips(&p, out); }
            else if p.extension().is_some_and(|e| e == "zip") { out.push(p); }
        }
    }
}

fn collect_sgfs(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    if !dir.is_dir() { return; }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() { collect_sgfs(&p, out); }
            else if p.extension().is_some_and(|e| e == "sgf") { out.push(p); }
        }
    }
}

// ---------------------------------------------------------------------------
// MCTS benchmark
// ---------------------------------------------------------------------------

fn run_mcts(simulations: u32, batch_size: usize) {
    use core_game::mcts::search::MctsSearch;
    use core_game::game::NNGame;
    use hive_game::move_encoding::policy_size;
    use hive_game::uhp::format_move_uhp;

    let mut game = Game::new();
    let ps = game.policy_size();
    let uniform_policy = vec![1.0 / ps as f32; ps];
    let mut search = MctsSearch::<Game>::new(100_000);
    search.init(&game, &uniform_policy);

    let rounds = simulations / batch_size as u32;

    let valid_moves = game.valid_moves();
    println!("Running MCTS benchmark: {simulations} simulations, batch_size={batch_size}");
    println!("Legal moves at root: {}", valid_moves.len());
    println!();

    let start = std::time::Instant::now();

    for _ in 0..rounds {
        let mut leaves = search.select_leaves(batch_size);
        if leaves.is_empty() { break; }
        let policies: Vec<Vec<f32>> = leaves.iter().map(|_| uniform_policy.clone()).collect();
        let values: Vec<f32> = vec![0.0; leaves.len()];
        search.expand_and_backprop(&mut leaves, &policies, &values);
    }

    let elapsed = start.elapsed();
    let root = search.arena.get(search.root);
    let sims_per_sec = root.visit_count as f64 / elapsed.as_secs_f64();

    println!("Root visits: {}", root.visit_count);
    println!("Root value:  {:.4}", root.value());
    println!("Time:        {:.3}s", elapsed.as_secs_f64());
    println!("Throughput:  {:.0} sims/s", sims_per_sec);
    println!();

    let dist = search.get_visit_distribution();
    println!("Top moves by visit share:");
    let mut sorted = dist.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (mv, prob) in sorted.iter().take(10) {
        let uhp = format_move_uhp(&game, mv);
        println!("  {uhp:>20}  {prob:.4}");
    }

    if let Some(best) = search.best_move() {
        let uhp = format_move_uhp(&game, &best);
        println!("\nBest move: {uhp}");
    }
}

// ---------------------------------------------------------------------------
// Random play
// ---------------------------------------------------------------------------

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
                game.play_move(&moves[idx]).unwrap();
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

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match mode {
        "random" => {
            let n: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            run_random(n);
        }
        "replay" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("games/hive/boardspace");
            replay::run_replay(path);
        }
        "debug" => {
            if args.len() < 3 {
                eprintln!("Usage: hive-zero debug <sgf-file> OR hive-zero debug <zip> <sgf-name>");
                std::process::exit(1);
            }
            replay::run_debug(&args[2..]);
        }
        "process" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("games/hive/boardspace");
            let skip_timeout = args.iter().any(|a| a == "--skip-timeout-games");
            run_process(path, skip_timeout);
        }
        "mcts" => {
            let sims: u32  = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(800);
            let batch: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
            run_mcts(sims, batch);
        }
        _ => {
            eprintln!("Usage: hive-zero <command> [args]");
            eprintln!();
            eprintln!("Commands:");
            eprintln!("  random [N]               play N random games (default 100)");
            eprintln!("  replay [path]            replay boardspace games");
            eprintln!("  debug <sgf|zip> [name]   verbose replay of one game");
            eprintln!("  process [path]           replay all games, compute ELO, write CSVs");
            eprintln!("  mcts [sims] [batch]      MCTS benchmark");
            std::process::exit(1);
        }
    }
}
