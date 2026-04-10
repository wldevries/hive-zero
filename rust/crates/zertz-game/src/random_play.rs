//! Random game simulation for testing game rules.

use core_game::game::{Game, Outcome, Player};
use rand::seq::SliceRandom;
use zertz_game::zertz::{classify_win, WinType, ZertzBoard};

pub struct GameResult {
    pub moves: u32,
    pub outcome: Outcome,
    pub win_type: WinType,
    pub board_full: bool,
    pub final_board: ZertzBoard,
}

pub fn run_game(rng: &mut impl rand::Rng) -> GameResult {
    let mut board = ZertzBoard::default();
    let mut moves = 0;

    loop {
        let outcome = board.outcome();
        if outcome != Outcome::Ongoing {
            let win_type = match outcome {
                Outcome::WonBy(p) => classify_win(&board, p),
                Outcome::Draw => WinType::Draw,
                Outcome::Ongoing => unreachable!(),
            };
            return GameResult { moves, outcome, win_type, board_full: board.board_full, final_board: board };
        }

        let legal = board.legal_moves();
        if legal.is_empty() {
            return GameResult { moves, outcome: Outcome::Draw, win_type: WinType::Draw, board_full: board.board_full, final_board: board };
        }

        let mv = legal.choose(rng).unwrap();
        board.play_move(mv).unwrap();
        moves += 1;
    }
}

pub fn run_random_games(n: u32) {
    let mut rng = rand::thread_rng();

    let mut wins_a = 0u32;
    let mut wins_b = 0u32;
    let mut draws = 0u32;
    let mut total_moves = 0u64;
    let mut win_types = [0u32; 4]; // FourWhite, FiveGrey, SixBlack, ThreeEach
    let mut board_full_count = 0u32;
    let mut total_jump = 0u64;
    let mut total_isolation = 0u64;

    for game in 0..n {
        let result = run_game(&mut rng);
        total_moves += result.moves as u64;
        let b = &result.final_board;
        let game_jump: u64 = b.jump_captures.iter().flatten().map(|&x| x as u64).sum();
        let game_iso: u64 = b.isolation_captures.iter().flatten().map(|&x| x as u64).sum();
        total_jump += game_jump;
        total_isolation += game_iso;
        println!("--- Game {} ({} moves) jump={} iso={} ---", game + 1, result.moves, game_jump, game_iso);
        println!("{}", b);
        match result.outcome {
            Outcome::WonBy(Player::Player1) => wins_a += 1,
            Outcome::WonBy(Player::Player2) => wins_b += 1,
            Outcome::Draw => draws += 1,
            Outcome::Ongoing => {}
        }
        if result.board_full { board_full_count += 1; }
        match result.win_type {
            WinType::FourWhite => win_types[0] += 1,
            WinType::FiveGrey  => win_types[1] += 1,
            WinType::SixBlack  => win_types[2] += 1,
            WinType::ThreeEach => win_types[3] += 1,
            WinType::Draw      => {}
        }
    }

    let avg_moves = total_moves as f64 / n as f64;
    let pct = |count: u32| count as f64 / n as f64 * 100.0;
    let total_marbles = total_jump + total_isolation;

    println!("Results over {n} games");
    println!("  Player A wins: {:5}  ({:.1}%)", wins_a, pct(wins_a));
    println!("  Player B wins: {:5}  ({:.1}%)", wins_b, pct(wins_b));
    println!("  Draws:         {:5}  ({:.1}%)", draws, pct(draws));
    println!("  Avg moves:     {avg_moves:.1}");
    println!();
    println!("Marble acquisition (avg per game):");
    println!("  Jump captures: {:.1}  ({:.1}%)", total_jump as f64 / n as f64, total_jump as f64 / total_marbles as f64 * 100.0);
    println!("  Isolation:     {:.1}  ({:.1}%)", total_isolation as f64 / n as f64, total_isolation as f64 / total_marbles as f64 * 100.0);
    println!();
    println!("Win types (excluding draws):");
    let decisive = (wins_a + wins_b) as f64;
    let wpct = |count: u32| count as f64 / decisive * 100.0;
    println!("  4 white:      {:5}  ({:.1}%)", win_types[0], wpct(win_types[0]));
    println!("  5 grey:       {:5}  ({:.1}%)", win_types[1], wpct(win_types[1]));
    println!("  6 black:      {:5}  ({:.1}%)", win_types[2], wpct(win_types[2]));
    println!("  3 of each:    {:5}  ({:.1}%)", win_types[3], wpct(win_types[3]));
    println!();
    println!("  via board full (F2): {:5}  ({:.1}%)", board_full_count, wpct(board_full_count));
}
