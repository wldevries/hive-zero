mod process;
mod random_play;
mod replay;
mod tournament;
mod tune;
mod weights;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "yinsh-tools", about = "YINSH game tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Replay boardspace games from a zip dir/file
    Replay {
        /// Path to zip dir or file
        #[arg(default_value = "games/yinsh/boardspace")]
        path: String,
    },
    /// Verbose replay of a single game from a zip
    Debug {
        zip_path: String,
        sgf_name: String,
    },
    /// Play a random game and print each board state
    Random {
        /// Maximum number of moves to play
        #[arg(default_value = "300")]
        moves: usize,
    },
    /// Compute statistics from boardspace games
    Stats {
        /// Path to zip dir or file
        #[arg(default_value = "games/yinsh/boardspace")]
        path: String,
    },
    /// Replay all games, compute Elo, write game_outcomes.csv + player_elo.csv
    Process {
        /// Path to zip dir or file
        #[arg(default_value = "games/yinsh/boardspace")]
        path: String,
        /// Skip games whose RE field indicates a timeout
        #[arg(long)]
        skip_timeout_games: bool,
    },
    /// Texel-tune the alphabeta eval weights from boardspace game outcomes
    Tune {
        /// Path to zip dir or file
        #[arg(default_value = "games/yinsh/boardspace")]
        path: String,
        /// Override the cache CSV path (default: <path>/tune_positions.csv)
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Re-extract positions even if the cache exists
        #[arg(long)]
        regen_cache: bool,
        /// Probability of sampling each Normal-phase position [0..1]
        #[arg(long, default_value_t = 0.5)]
        sample_rate: f32,
        /// Sigmoid scale: predicted_win = sigmoid(eval / k)
        #[arg(long, default_value_t = 200.0)]
        k: f32,
        /// Adam learning rate
        #[arg(long, default_value_t = 1.0)]
        lr: f32,
        /// Number of full-batch gradient steps
        #[arg(long, default_value_t = 2000)]
        epochs: usize,
        /// Fraction of games held out for validation
        #[arg(long, default_value_t = 0.1)]
        val_frac: f32,
        /// Write fitted weights to this file (loadable via Tournament --weights-*)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Skip games where either player's Elo is below this (0 = no filter)
        #[arg(long, default_value_t = 0.0)]
        min_player_elo: f32,
        /// Path to player_elo.csv (default: <path>/player_elo.csv)
        #[arg(long)]
        player_elo_csv: Option<PathBuf>,
        /// Allow negative fitted weights (default: clamp to ≥0)
        #[arg(long)]
        allow_negative_weights: bool,
    },
    /// Head-to-head alphabeta tournament between two weight sets
    Tournament {
        /// Engine A weights file (omit for built-in DEFAULT_WEIGHTS)
        #[arg(long)]
        weights_a: Option<PathBuf>,
        /// Engine B weights file (omit for built-in DEFAULT_WEIGHTS)
        #[arg(long)]
        weights_b: Option<PathBuf>,
        /// Alphabeta search depth (same for both engines)
        #[arg(long, default_value_t = 3)]
        depth: u32,
        /// Total games to play (rounded up to an even number for pairing)
        #[arg(long, default_value_t = 40)]
        games: u32,
        /// Random first N moves before alphabeta takes over (10 = full Setup)
        #[arg(long, default_value_t = 12)]
        random_opening_moves: u32,
        /// Hard cap on moves per game; treated as a draw if hit
        #[arg(long, default_value_t = 500)]
        max_moves: u32,
        /// Base RNG seed (paired games share it; pair k uses seed+k)
        #[arg(long, default_value_t = 0xCAFE)]
        seed: u64,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Replay { path } => replay::run_replay(&path),
        Command::Debug { zip_path, sgf_name } => replay::run_debug(&zip_path, &sgf_name),
        Command::Random { moves } => random_play::run_random_game(moves),
        Command::Stats { path } => replay::run_stats(&path),
        Command::Process { path, skip_timeout_games } => {
            process::run_process(&path, skip_timeout_games)
        }
        Command::Tune {
            path, cache, regen_cache, sample_rate, k, lr, epochs, val_frac, output,
            min_player_elo, player_elo_csv, allow_negative_weights,
        } => tune::run_tune(tune::TuneOptions {
            games_path: path,
            cache_path: cache,
            regen_cache,
            sample_rate,
            k,
            lr,
            epochs,
            val_frac,
            output,
            min_player_elo,
            player_elo_csv,
            clamp_nonneg: !allow_negative_weights,
        }),
        Command::Tournament {
            weights_a, weights_b, depth, games, random_opening_moves, max_moves, seed,
        } => tournament::run_tournament(tournament::TournamentOptions {
            weights_a,
            weights_b,
            depth,
            games,
            random_opening_moves,
            max_moves,
            seed,
        }),
    }
}
