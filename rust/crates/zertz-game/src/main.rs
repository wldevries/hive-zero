use zertz_game::{mcts, move_encoding, random_play, replay, zertz};

// ---------------------------------------------------------------------------
// MCTS demo
// ---------------------------------------------------------------------------

fn run_mcts_demo(simulations: u32) {
    use mcts::search::{PolicyHeads, POLICY_HEADS_TOTAL, PLACE_HEAD_SIZE, CAP_HEAD_SIZE};

    let board = zertz::ZertzBoard::default();
    let uniform_buf = vec![0.0f32; POLICY_HEADS_TOTAL];
    let heads = PolicyHeads {
        place: &uniform_buf[..PLACE_HEAD_SIZE],
        cap_source: &uniform_buf[PLACE_HEAD_SIZE..PLACE_HEAD_SIZE + CAP_HEAD_SIZE],
        cap_dest: &uniform_buf[PLACE_HEAD_SIZE + CAP_HEAD_SIZE..],
    };
    let mut search = mcts::search::MctsSearch::new(100_000);
    search.init(&board, &heads);

    let batch_size = 8;
    let rounds = simulations / batch_size as u32;

    println!("Running MCTS demo: {simulations} simulations, batch_size={batch_size}");

    let start = std::time::Instant::now();
    for _ in 0..rounds {
        let mut leaves = search.select_leaves(batch_size);
        if leaves.is_empty() { break; }
        let heads_list: Vec<PolicyHeads> = leaves.iter().map(|_| PolicyHeads {
            place: &uniform_buf[..PLACE_HEAD_SIZE],
            cap_source: &uniform_buf[PLACE_HEAD_SIZE..PLACE_HEAD_SIZE + CAP_HEAD_SIZE],
            cap_dest: &uniform_buf[PLACE_HEAD_SIZE + CAP_HEAD_SIZE..],
        }).collect();
        let values: Vec<f32> = vec![0.0; leaves.len()];
        search.expand_and_backprop(&mut leaves, &heads_list, &values);
    }
    let elapsed = start.elapsed();

    let root = search.arena.get(search.root);
    println!("Root visits: {}", root.visit_count);
    println!("Root value:  {:.4}", root.value());
    println!("Time:        {:.3}s", elapsed.as_secs_f64());
    println!("Throughput:  {:.0} sims/s", root.visit_count as f64 / elapsed.as_secs_f64());
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("random");

    match mode {
        "random" => {
            let n: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            random_play::run_random_games(n);
        }
        "replay" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("../games/zertz");
            replay::run_replay(path);
        }
        "debug" => {
            let zip_path = args.get(2).expect("need zip path");
            let sgf_name = args.get(3).expect("need sgf name");
            replay::run_debug(zip_path, sgf_name);
        }
        "process" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("../games/zertz");
            let skip_timeout = args.iter().any(|a| a == "--skip-timeout-games");
            replay::run_process(path, skip_timeout);
        }
        "mcts" => {
            let sims: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(800);
            run_mcts_demo(sims);
        }
        "stats" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("../games/zertz");
            replay::run_stats(path);
        }
        "playback" => {
            // playback [path] [--auto <ms>]
            // Randomly picks a game from the boardspace library and plays through it.
            // Without --auto: press Enter to advance each turn (interactive).
            // With --auto <ms>: advance automatically every <ms> milliseconds.
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("../games/zertz");
            let auto_ms = args.windows(2).find(|w| w[0] == "--auto")
                .and_then(|w| w[1].parse::<u64>().ok());
            replay::run_playback(path, auto_ms);
        }
        _ => {
            eprintln!("Usage: zertz-zero <random [N]|replay [path]|process [path]|stats [path]|debug <zip> <sgf>|mcts [sims]|playback [path] [--auto <ms>]>");
            eprintln!("  random [N]              - play N random games (default 100)");
            eprintln!("  replay [path]           - replay boardspace games from zip dir/file");
            eprintln!("  process [path]          - compute ELO rankings, write CSVs");
            eprintln!("  stats [path]            - aggregate game statistics, write game_stats.txt");
            eprintln!("  debug <zip> <sgf>       - verbose replay of a single game from zip");
            eprintln!("  mcts [sims]             - run MCTS demo with uniform policy (default 800)");
            eprintln!("  playback [path]         - interactively step through a random boardspace game");
            eprintln!("    --auto <ms>           - auto-advance every <ms> milliseconds");
            std::process::exit(1);
        }
    }
}
