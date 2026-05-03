//! Replay every boardspace YINSH game, compute per-player Elo, and write
//! `game_outcomes.csv` + `player_elo.csv`. Mirrors the hive-tools `process`
//! command. Engine-verified outcomes take precedence over SGF metadata; we fall
//! back to `result_from_metadata` only when the replay errors out.

use std::collections::HashMap;
use std::path::Path;

use core_game::game::{Outcome, Player};
use yinsh_game::sgf::{self, Color, GameRecord};

use crate::replay;

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

fn engine_result(record: &GameRecord) -> Option<&'static str> {
    // Black-first games aren't supported by the engine replay yet; fall back
    // to metadata for those.
    if !matches!(record.first_player_color, Color::White) {
        return None;
    }
    let result = replay::replay_game(record);
    if result.error.is_some() {
        return None;
    }
    match result.final_board.outcome {
        Outcome::WonBy(Player::Player1) => Some("p0_wins"),
        Outcome::WonBy(Player::Player2) => Some("p1_wins"),
        Outcome::Draw => Some("draw"),
        Outcome::Ongoing => None,
    }
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

pub fn run_process(games_path: &str, skip_timeout: bool) {
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
    let mut parse_errors: usize = 0;
    let mut skipped_unsupported: usize = 0;
    let mut skipped_timeout: usize = 0;

    println!("Processing games from {games_path}...");

    let mut process_zip = |zip_path: &Path| {
        let zip_name = zip_path.display().to_string();
        let file = match std::fs::File::open(zip_path) {
            Ok(f) => f,
            Err(e) => { eprintln!("  failed to open {zip_name}: {e}"); return; }
        };

        let _ = core_game::sgf::iter_sgf_texts_in_zip(file, |sgf_name, text| {
            total += 1;
            if total % 2000 == 0 {
                eprint!("\r  {} games processed ({} determined)...   ", total, determined);
            }

            if skip_timeout && core_game::sgf::is_timeout(&text) {
                skipped_timeout += 1;
                return;
            }

            let record = match sgf::parse_game(&text) {
                Ok(r) => r,
                Err(_) => { parse_errors += 1; return; }
            };

            if !matches!(record.first_player_color, Color::White) {
                skipped_unsupported += 1;
                return;
            }

            let p0 = record.player0.clone();
            let p1 = record.player1.clone();
            let move_count = record.turns.len();

            // Engine-verified result, falling back to metadata on replay failure.
            let result = engine_result(&record).unwrap_or_else(|| {
                core_game::sgf::result_from_metadata(&text, &p0, &p1)
            });

            if result == "p0_wins" || result == "p1_wins" || result == "draw" {
                determined += 1;
            }

            outcomes.push(format!("{},{},{},{},{},{}",
                csv_escape(&zip_name), csv_escape(sgf_name),
                csv_escape(&p0), csv_escape(&p1),
                move_count, result));

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
        });
    };

    if path.is_file() && path.extension().is_some_and(|e| e == "zip") {
        process_zip(path);
    } else if path.is_dir() {
        core_game::sgf::visit_zip_dir(path, &mut process_zip);
    } else {
        eprintln!("Not a zip file or directory: {games_path}");
        return;
    }

    eprintln!();
    println!();
    let unknown = total
        .saturating_sub(determined)
        .saturating_sub(parse_errors)
        .saturating_sub(skipped_unsupported)
        .saturating_sub(skipped_timeout);
    println!(
        "Done: {} games total, {} outcomes determined, {} parse errors, \
         {} skipped (Black-first), {} skipped (timeout), {} unknown",
        total, determined, parse_errors, skipped_unsupported, skipped_timeout, unknown
    );

    let out_dir = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent().unwrap_or(path).to_path_buf()
    };

    let outcomes_path = out_dir.join("game_outcomes.csv");
    if let Ok(mut f) = std::fs::File::create(&outcomes_path) {
        use std::io::Write;
        writeln!(f, "zip_file,sgf_name,p0,p1,move_count,result").ok();
        for line in &outcomes {
            writeln!(f, "{}", line).ok();
        }
        println!("Wrote {}", outcomes_path.display());
    }

    let elo_path = out_dir.join("player_elo.csv");
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
            writeln!(f, "{},{:.1},{},{},{},{}",
                csv_escape(player), e, s.games, s.wins, s.losses, s.draws).ok();
        }
        println!("Wrote {}", elo_path.display());
    }

}
