//! Head-to-head alphabeta tournament: validate fitted weights against the
//! defaults (or any two weight files against each other).
//!
//! Plays paired games — each opening seed produces two games, with engine A
//! taking each side once. This controls for first-player advantage and
//! opening luck. Results are reported as W/L/D from A's perspective plus an
//! Elo difference.

use std::path::PathBuf;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use core_game::game::{Outcome, Player};
use yinsh_game::alphabeta::{
    DEFAULT_WEIGHTS, N_FEATURES, alphabeta_best_move_with_weights,
};
use yinsh_game::board::YinshBoard;

use crate::weights;

pub struct TournamentOptions {
    pub weights_a: Option<PathBuf>,
    pub weights_b: Option<PathBuf>,
    pub depth: u32,
    pub games: u32,
    pub random_opening_moves: u32,
    pub max_moves: u32,
    pub seed: u64,
}

#[derive(Default, Clone, Copy)]
struct Score {
    wins: u32,
    losses: u32,
    draws: u32,
    timeouts: u32,
}

impl Score {
    fn played(&self) -> u32 { self.wins + self.losses + self.draws }
    fn points(&self) -> f64 { self.wins as f64 + 0.5 * self.draws as f64 }
}

struct GameResult {
    outcome: Outcome,
    moves: u32,
    timed_out: bool,
}

pub fn run_tournament(opts: TournamentOptions) {
    let weights_a = match &opts.weights_a {
        Some(p) => weights::load_weights(p).expect("failed to load weights A"),
        None => DEFAULT_WEIGHTS,
    };
    let weights_b = match &opts.weights_b {
        Some(p) => weights::load_weights(p).expect("failed to load weights B"),
        None => DEFAULT_WEIGHTS,
    };

    let label_a = opts.weights_a.as_deref().map(|p| p.display().to_string())
        .unwrap_or_else(|| "DEFAULT".to_string());
    let label_b = opts.weights_b.as_deref().map(|p| p.display().to_string())
        .unwrap_or_else(|| "DEFAULT".to_string());

    println!("Engine A: {label_a}");
    println!("  weights: {:?}", weights_a);
    println!("Engine B: {label_b}");
    println!("  weights: {:?}", weights_b);
    println!(
        "depth={}, games={} (paired), random_opening={}, max_moves={}, seed={}",
        opts.depth, opts.games, opts.random_opening_moves, opts.max_moves, opts.seed
    );
    println!();

    let pairs = opts.games.div_ceil(2);
    let mut score = Score::default();
    let mut a_white = Score::default();
    let mut a_black = Score::default();
    let mut total_moves: u64 = 0;

    for pair in 0..pairs {
        let opening_seed = opts.seed.wrapping_add(pair as u64);

        // Game 1: A as White, B as Black
        let g1 = play_game(
            &weights_a, &weights_b,
            opts.depth, opts.random_opening_moves, opts.max_moves, opening_seed,
        );
        record(&mut score, &mut a_white, &g1, /*a_is_white=*/true);
        total_moves += g1.moves as u64;

        // Game 2: B as White, A as Black — same opening seed
        let g2 = play_game(
            &weights_b, &weights_a,
            opts.depth, opts.random_opening_moves, opts.max_moves, opening_seed,
        );
        record(&mut score, &mut a_black, &g2, /*a_is_white=*/false);
        total_moves += g2.moves as u64;

        eprint!(
            "\r  pair {}/{}: A {}W/{}L/{}D ({} timeouts)   ",
            pair + 1, pairs,
            score.wins, score.losses, score.draws, score.timeouts,
        );
    }
    eprintln!();

    let n = score.played();
    let pts = score.points();
    let win_rate = if n > 0 { pts / n as f64 } else { 0.5 };
    let elo_diff = elo_from_winrate(win_rate);

    println!();
    println!("=== Tournament results (A perspective) ===");
    println!("Games played:  {n}");
    println!("Total moves:   {total_moves}  (avg {:.1}/game)",
        total_moves as f64 / n.max(1) as f64);
    println!("Wins:          {}", score.wins);
    println!("Losses:        {}", score.losses);
    println!("Draws:         {}", score.draws);
    println!("Timeouts:      {}", score.timeouts);
    println!("Points:        {:.1} / {n}", pts);
    println!("Win rate:      {:.1}%", win_rate * 100.0);
    println!("Elo diff (A − B): {:+.1}", elo_diff);
    println!();
    println!("By color:");
    print_color_split("  A as White:", &a_white);
    print_color_split("  A as Black:", &a_black);
    println!();
    if let Some(ci) = wilson_ci_95(pts, n as f64) {
        println!("Score 95% CI:  [{:.3}, {:.3}]", ci.0, ci.1);
        let elo_lo = elo_from_winrate(ci.0);
        let elo_hi = elo_from_winrate(ci.1);
        println!("Elo 95% CI:    [{:+.0}, {:+.0}]", elo_lo, elo_hi);
    }
}

fn print_color_split(label: &str, s: &Score) {
    let n = s.played();
    let pts = s.points();
    let win_rate = if n > 0 { pts / n as f64 } else { 0.0 };
    println!(
        "{label} {} W / {} L / {} D ({} timeouts) — {:.1}%",
        s.wins, s.losses, s.draws, s.timeouts, win_rate * 100.0,
    );
}

fn record(total: &mut Score, by_color: &mut Score, g: &GameResult, a_is_white: bool) {
    let entry = |s: &mut Score, kind: &str| match kind {
        "win"  => { s.wins += 1; }
        "loss" => { s.losses += 1; }
        "draw" => { s.draws += 1; }
        _ => {}
    };
    if g.timed_out {
        // Timeouts count as draws for scoring purposes but are tracked
        // separately so we can spot pathological eval combinations.
        total.timeouts += 1;
        by_color.timeouts += 1;
        entry(total, "draw");
        entry(by_color, "draw");
        return;
    }
    let kind = match g.outcome {
        Outcome::WonBy(Player::Player1) => if a_is_white { "win"  } else { "loss" },
        Outcome::WonBy(Player::Player2) => if a_is_white { "loss" } else { "win"  },
        Outcome::Draw                   => "draw",
        Outcome::Ongoing                => "draw", // shouldn't reach here
    };
    entry(total, kind);
    entry(by_color, kind);
}

fn play_game(
    white_weights: &[f32; N_FEATURES],
    black_weights: &[f32; N_FEATURES],
    depth: u32,
    random_opening: u32,
    max_moves: u32,
    seed: u64,
) -> GameResult {
    let mut board = YinshBoard::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut moves = 0u32;

    while !matches!(board.outcome, Outcome::WonBy(_) | Outcome::Draw) {
        if moves >= max_moves {
            return GameResult { outcome: board.outcome, moves, timed_out: true };
        }
        let legal = board.legal_moves();
        if legal.is_empty() {
            // Engines could be in a state with no legal moves (shouldn't
            // happen in yinsh outside Pass, but guard against it).
            break;
        }
        let mv = if moves < random_opening {
            legal[rng.random_range(0..legal.len())]
        } else {
            let weights = match board.next_player {
                Player::Player1 => white_weights,
                Player::Player2 => black_weights,
            };
            alphabeta_best_move_with_weights(&board, depth, weights)
        };
        board.apply_move(mv).expect("legal move should apply");
        moves += 1;
    }
    GameResult { outcome: board.outcome, moves, timed_out: false }
}

fn elo_from_winrate(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    -400.0 * (1.0 / p - 1.0).log10()
}

/// Wilson 95% CI for the proportion `pts/n`. Returns `None` for n=0.
fn wilson_ci_95(pts: f64, n: f64) -> Option<(f64, f64)> {
    if n <= 0.0 { return None; }
    let z = 1.96;
    let p = pts / n;
    let denom = 1.0 + z * z / n;
    let center = (p + z * z / (2.0 * n)) / denom;
    let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denom;
    Some(((center - half).max(0.0), (center + half).min(1.0)))
}
