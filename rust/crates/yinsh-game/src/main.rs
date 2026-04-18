mod random_play;
mod replay;

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
        #[arg(default_value = "../../../games/yinsh/boardspace")]
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
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Replay { path } => replay::run_replay(&path),
        Command::Debug { zip_path, sgf_name } => replay::run_debug(&zip_path, &sgf_name),
        Command::Random { moves } => random_play::run_random_game(moves),
    }
}
