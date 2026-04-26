mod random_play;
mod replay;

use clap::{Parser, Subcommand};
// ---------------------------------------------------------------------------
// MCTS demo
// ---------------------------------------------------------------------------

fn run_mcts_demo(simulations: u32) {
    use core_game::mcts::search::MctsSearch;
    use zertz_game::move_encoding::NN_POLICY_SIZE;
    use zertz_game::zertz::ZertzBoard;

    let board = ZertzBoard::default();
    let uniform_policy = vec![0.0f32; NN_POLICY_SIZE];
    let mut search = MctsSearch::<ZertzBoard>::new(100_000);
    search.init(&board, &uniform_policy);

    let batch_size = 8usize;
    let rounds = (simulations as usize) / batch_size;

    println!("Running MCTS demo: {simulations} simulations, batch_size={batch_size}");

    let start = std::time::Instant::now();
    for _ in 0..rounds {
        let leaves = search.select_leaves(batch_size);
        if leaves.is_empty() { break; }
        let nl = leaves.len();
        let policies: Vec<Vec<f32>> = (0..nl).map(|_| uniform_policy.clone()).collect();
        let values: Vec<f32> = vec![0.0; nl];
        search.expand_and_backprop(&policies, &values, &[]);
    }
    let elapsed = start.elapsed();

    let dist = search.get_visit_distribution();
    let total_visits: f32 = dist.iter().map(|(_, v)| v).sum::<f32>();
    let root_visits = (total_visits * simulations as f32) as u32;
    println!("Root visits: {root_visits}");
    println!("Time:        {:.3}s", elapsed.as_secs_f64());
    println!("Throughput:  {:.0} sims/s", root_visits as f64 / elapsed.as_secs_f64());
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "zertz-tools", about = "Zertz game tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Play N random games
    Random {
        /// Number of games to play
        #[arg(default_value_t = 100)]
        n: u32,
    },
    /// Replay boardspace games from a zip dir/file
    Replay {
        /// Path to zip dir or file
        #[arg(default_value = "../games/zertz")]
        path: String,
    },
    /// Verbose replay of a single game from a zip
    Debug {
        /// Path to the zip file
        zip_path: String,
        /// Name of the SGF file inside the zip
        sgf_name: String,
    },
    /// Compute ELO rankings, write CSVs
    Process {
        /// Path to zip dir or file
        #[arg(default_value = "../games/zertz")]
        path: String,
        /// Discard training data from games that hit the move cap
        #[arg(long)]
        skip_timeout_games: bool,
    },
    /// Aggregate game statistics, write game_stats.txt
    Stats {
        /// Path to zip dir or file
        #[arg(default_value = "../games/zertz")]
        path: String,
    },
    /// Run MCTS demo with uniform policy
    Mcts {
        /// Number of simulations
        #[arg(default_value_t = 800)]
        simulations: u32,
    },
    /// Interactively step through a random boardspace game
    Playback {
        /// Path to zip dir or file
        #[arg(default_value = "../games/zertz")]
        path: String,
        /// Auto-advance every N milliseconds
        #[arg(long)]
        auto: Option<u64>,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Random { n } => random_play::run_random_games(n),
        Command::Replay { path } => replay::run_replay(&path),
        Command::Debug { zip_path, sgf_name } => replay::run_debug(&zip_path, &sgf_name),
        Command::Process { path, skip_timeout_games } => replay::run_process(&path, skip_timeout_games),
        Command::Stats { path } => replay::run_stats(&path),
        Command::Mcts { simulations } => run_mcts_demo(simulations),
        Command::Playback { path, auto } => replay::run_playback(&path, auto),
    }
}
