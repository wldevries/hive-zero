mod board_encoding;
pub mod hex;
mod mcts;
mod move_encoding;
mod sgf;
mod random_play;
mod replay;
mod zertz;

// ---------------------------------------------------------------------------
// MCTS demo
// ---------------------------------------------------------------------------

fn run_mcts_demo(simulations: u32) {
    let board = zertz::ZertzBoard::default();
    let uniform_policy = vec![1.0 / move_encoding::POLICY_SIZE as f32; move_encoding::POLICY_SIZE];
    let mut search = mcts::search::MctsSearch::new(100_000);
    search.init(&board, &uniform_policy);

    let batch_size = 8;
    let rounds = simulations / batch_size as u32;

    println!("Running MCTS demo: {simulations} simulations, batch_size={batch_size}");

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
        _ => {
            eprintln!("Usage: zertz-zero <random [N]|replay [path]|process [path]|debug <zip> <sgf>|mcts [sims]>");
            eprintln!("  random [N]              - play N random games (default 100)");
            eprintln!("  replay [path]           - replay boardspace games from zip dir/file");
            eprintln!("  process [path]          - compute ELO rankings, write CSVs");
            eprintln!("  debug <zip> <sgf>       - verbose replay of a single game from zip");
            eprintln!("  mcts [sims]             - run MCTS demo with uniform policy (default 800)");
            std::process::exit(1);
        }
    }
}
