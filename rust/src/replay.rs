/// Replay Boardspace Hive games through the Rust engine.

use hive_engine::game::Game;
use crate::sgf;

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

    let moves = sgf::parse_moves(content);
    let total_turns = moves.len();
    let mut game = Game::new();

    for (i, move_str) in moves.iter().enumerate() {
        if game.is_game_over() {
            break;
        }
        if let Err(msg) = hive_engine::uhp::play_uhp_unchecked(&mut game, move_str) {
            return ReplayResult {
                turns_played: i,
                total_turns,
                final_state: game.state.as_str().to_string(),
                error: Some(format!("move {} rejected: {} ({:?})", i + 1, msg, move_str)),
                game: Some(game),
            };
        }
    }

    let state = game.state.as_str().to_string();
    let mc = game.move_count as usize;
    ReplayResult {
        turns_played: mc,
        total_turns,
        final_state: state,
        error: None,
        game: Some(game),
    }
}

/// Replay a game with verbose output (prints each move).
pub fn replay_game_verbose(content: &str) -> ReplayResult {
    let gtype = sgf::game_type(content);
    let p0 = sgf::extract_player(content, 0);
    let p1 = sgf::extract_player(content, 1);
    let result_field = sgf::extract_field(content, "RE[").unwrap_or_default();

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

    let moves = sgf::parse_moves(content);
    let total_turns = moves.len();
    println!("Parsed {} moves", total_turns);
    println!();

    let mut game = Game::new();

    for (i, move_str) in moves.iter().enumerate() {
        let color = if i % 2 == 0 { "White" } else { "Black" };
        println!("Move {} ({}): {}", i + 1, color, move_str);

        if game.is_game_over() {
            println!("  Game already over: {}", game.state.as_str());
            break;
        }

        if let Err(msg) = hive_engine::uhp::play_uhp_unchecked(&mut game, move_str) {
            println!("  ERROR: {}", msg);
            let valid = game.valid_moves();
            let valid_uhp: Vec<String> = valid.iter().map(|m| hive_engine::uhp::format_move_uhp(&game, m)).collect();
            println!("  Valid moves ({}):", valid_uhp.len());
            for vm in valid_uhp.iter().take(20) {
                println!("    {}", vm);
            }
            if valid_uhp.len() > 20 {
                println!("    ... and {} more", valid_uhp.len() - 20);
            }
            return ReplayResult {
                turns_played: i,
                total_turns,
                final_state: game.state.as_str().to_string(),
                error: Some(format!("move {} rejected: {} ({:?})", i + 1, msg, move_str)),
                game: Some(game),
            };
        }
        println!("  State: {}", game.state.as_str());
    }

    let state = game.state.as_str().to_string();
    let mc = game.move_count as usize;
    ReplayResult {
        turns_played: mc,
        total_turns,
        final_state: state,
        error: None,
        game: Some(game),
    }
}
