mod replay;

use clap::{Parser, Subcommand};
use core_game::hex::{Hex, hex_neighbors};
use hive_game::sgf;
use hive_game::game::Game;
use hive_game::piece::{PieceColor, PieceType, player_pieces};

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
        let leaf_ids = search.select_leaves(batch_size);
        if leaf_ids.is_empty() { break; }
        let policies: Vec<Vec<f32>> = leaf_ids.iter().map(|_| uniform_policy.clone()).collect();
        let values: Vec<f32> = vec![0.0; leaf_ids.len()];
        search.expand_and_backprop(&policies, &values);
    }

    let elapsed = start.elapsed();
    let sims_per_sec = search.root_visit_count() as f64 / elapsed.as_secs_f64();

    println!("Root visits: {}", search.root_visit_count());
    println!("Root value:  {:.4}", search.root_value());
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
    use hive_game::uhp::format_move_uhp;

    let verbose = n == 1;
    if !verbose {
        println!("Playing {n} random games...");
    }

    let mut total_moves = 0u64;
    let mut results = HashMap::new();

    for game_idx in 0..n {
        let mut game = Game::new();
        let mut rng = rand::thread_rng();
        let mut move_num = 0u32;

        let mut last_to:   Option<Hex> = None;
        let mut last_from: Option<Hex> = None;

        while !game.is_game_over() {
            let moves = game.valid_moves();
            if verbose {
                println!("\n--- Move {} | {} to play ---", move_num + 1, game.turn_color.as_char());
                if move_num > 0 {
                    println!("{}", game.board.render(last_to, last_from));
                }
            }
            if moves.is_empty() {
                if verbose { println!("  (pass)"); }
                game.play_pass();
                last_to   = None;
                last_from = None;
            } else {
                let idx = rng.gen_range(0..moves.len());
                let mv = moves[idx];
                if verbose {
                    let uhp = format_move_uhp(&game, &mv);
                    println!("  -> {uhp}");
                }
                last_to   = mv.to;
                last_from = mv.from;
                game.play_move(&mv).unwrap();
            }
            move_num += 1;
            total_moves += 1;
        }

        if verbose {
            println!("\n--- Final position ({move_num} moves) ---");
            println!("{}", game.board.render(last_to, last_from));
            println!("\nResult: {}", game.state.as_str());
        } else {
            println!("Game {}: {} ({move_num} moves)", game_idx + 1, game.state.as_str());
        }

        *results.entry(game.state.as_str().to_string()).or_insert(0u32) += 1;
    }

    if !verbose {
        println!();
        println!("Total moves: {total_moves}");
        println!("Avg moves/game: {:.1}", total_moves as f64 / n as f64);
        println!("Results:");
        let mut sorted: Vec<_> = results.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (state, count) in sorted {
            println!("  {state}: {count}");
        }
    }
}

// ---------------------------------------------------------------------------
// Stats: aggregate game statistics from boardspace games
// ---------------------------------------------------------------------------

fn pct(n: u64, total: u64) -> f64 {
    if total == 0 { 0.0 } else { n as f64 / total as f64 * 100.0 }
}

fn run_stats(games_path: &str) {
    let path = Path::new(games_path);
    if !path.exists() {
        eprintln!("Path not found: {games_path}");
        return;
    }

    let mut total_games: u64 = 0;
    let mut replayed_ok: u64 = 0;
    let mut skipped_expansion: u64 = 0;
    let mut errors: u64 = 0;

    // Results
    let mut white_wins: u64 = 0;
    let mut black_wins: u64 = 0;
    let mut draws: u64 = 0;
    let mut in_progress: u64 = 0; // games that didn't finish

    // Game length
    let mut lengths: Vec<u32> = Vec::new();

    // First/second player advantage (White always moves first in Hive)
    // white_wins already covers first-player wins

    // First bug played
    let mut first_bug: HashMap<String, u64> = HashMap::new();
    let mut second_bug: HashMap<String, u64> = HashMap::new();

    // Pieces on board at end
    let mut total_pieces_end: Vec<u32> = Vec::new();
    let mut white_pieces_end: Vec<u32> = Vec::new();
    let mut black_pieces_end: Vec<u32> = Vec::new();

    // Per-piece-type count on board at end (summed over both colors)
    let mut piece_type_counts: HashMap<String, Vec<u32>> = HashMap::new();

    // Board dimensions at end
    let mut diameters: Vec<u32> = Vec::new();
    let mut q_spans: Vec<u32> = Vec::new();
    let mut r_spans: Vec<u32> = Vec::new();
    let mut s_spans: Vec<u32> = Vec::new();

    // Queen neighbor count at game end (for the loser)
    let mut queen_neighbors_at_end: Vec<u32> = Vec::new();

    // Beetle-on-queen at game end
    let mut beetle_on_queen_wins: u64 = 0;

    // Killer piece: piece type of the last move in decisive games
    let mut killer_piece: HashMap<String, u64> = HashMap::new();
    // Surrounding pieces: what piece types sit around the losing queen, split by side
    let mut surround_attacker: HashMap<String, u64> = HashMap::new();
    let mut surround_defender: HashMap<String, u64> = HashMap::new();

    // Piece movement frequency: how many times each piece type is moved (not placed)
    let mut move_freq: HashMap<String, u64> = HashMap::new();
    let mut total_movements: u64 = 0;

    // Turn queen was played
    let mut white_queen_turn: Vec<u32> = Vec::new();
    let mut black_queen_turn: Vec<u32> = Vec::new();

    println!("Computing stats from {games_path}...");

    let mut zip_paths: Vec<_> = Vec::new();
    collect_zips(path, &mut zip_paths);
    zip_paths.sort();

    for zip_path in &zip_paths {
        let zip_name = zip_path.display().to_string();
        let file = match std::fs::File::open(zip_path) {
            Ok(f) => f,
            Err(e) => { eprintln!("  failed to open {zip_name}: {e}"); continue; }
        };
        let _ = core_game::sgf::iter_sgf_texts_in_zip(file, |_sgf_name, text| {
            total_games += 1;
            if total_games % 10000 == 0 {
                eprint!("\r  {} games processed...   ", total_games);
            }

            let gtype = sgf::game_type(&text);
            if gtype == "expansion" {
                skipped_expansion += 1;
                return;
            }

            let mut game = Game::new();
            if sgf::replay_into_game(&text, &mut game).is_err() {
                errors += 1;
                return;
            }
            replayed_ok += 1;

            // Result
            match game.state.as_str() {
                "WhiteWins" => white_wins += 1,
                "BlackWins" => black_wins += 1,
                "Draw" => draws += 1,
                _ => { in_progress += 1; }
            }

            // Game length
            lengths.push(game.move_count as u32);

            // First and second bug played
            let history = game.move_history();
            if let Some(mv) = history.first() {
                if let Some(piece) = mv.piece {
                    let name = format!("{}", piece.piece_type().as_char());
                    *first_bug.entry(name).or_default() += 1;
                }
            }
            if history.len() >= 2 {
                if let Some(piece) = history[1].piece {
                    let name = format!("{}", piece.piece_type().as_char());
                    *second_bug.entry(name).or_default() += 1;
                }
            }

            // Piece movement frequency (moves, not placements)
            for mv in history {
                if let Some(piece) = mv.piece {
                    if mv.from.is_some() {
                        // This is a movement, not a placement
                        let key = format!("{}", piece.piece_type().as_char());
                        *move_freq.entry(key).or_default() += 1;
                        total_movements += 1;
                    }
                }
            }

            // Pieces on board at end
            let w_pieces: u32 = player_pieces(PieceColor::White).iter()
                .filter(|p| game.board.piece_position(**p).is_some())
                .count() as u32;
            let b_pieces: u32 = player_pieces(PieceColor::Black).iter()
                .filter(|p| game.board.piece_position(**p).is_some())
                .count() as u32;
            white_pieces_end.push(w_pieces);
            black_pieces_end.push(b_pieces);
            total_pieces_end.push(w_pieces + b_pieces);

            // Per piece type on board
            for pt in &[PieceType::Queen, PieceType::Spider, PieceType::Beetle,
                        PieceType::Grasshopper, PieceType::Ant] {
                let mut count = 0u32;
                for color in &[PieceColor::White, PieceColor::Black] {
                    for p in player_pieces(*color) {
                        if p.piece_type() == *pt && game.board.piece_position(p).is_some() {
                            count += 1;
                        }
                    }
                }
                let key = format!("{}", pt.as_char());
                piece_type_counts.entry(key).or_insert_with(Vec::new).push(count);
            }

            // Board dimensions
            {
                let mut positions = Vec::new();
                for color in [PieceColor::White, PieceColor::Black] {
                    for piece in player_pieces(color) {
                        if let Some(pos) = game.board.piece_position(piece) {
                            positions.push(pos);
                        }
                    }
                }
                if positions.len() >= 2 {
                    let mut max_d: i8 = 0;
                    for i in 0..positions.len() {
                        for j in i+1..positions.len() {
                            let (q1, r1) = positions[i];
                            let (q2, r2) = positions[j];
                            let s1 = -q1 - r1;
                            let s2 = -q2 - r2;
                            let d = (q1-q2).abs().max((r1-r2).abs()).max((s1-s2).abs());
                            max_d = max_d.max(d);
                        }
                    }
                    diameters.push(max_d as u32);
                    let min_q = positions.iter().map(|(q,_)| *q).min().unwrap();
                    let max_q = positions.iter().map(|(q,_)| *q).max().unwrap();
                    let min_r = positions.iter().map(|(_,r)| *r).min().unwrap();
                    let max_r = positions.iter().map(|(_,r)| *r).max().unwrap();
                    let ss: Vec<i8> = positions.iter().map(|(q,r)| -q - r).collect();
                    let min_s = ss.iter().copied().min().unwrap();
                    let max_s = ss.iter().copied().max().unwrap();
                    q_spans.push((max_q - min_q + 1) as u32);
                    r_spans.push((max_r - min_r + 1) as u32);
                    s_spans.push((max_s - min_s + 1) as u32);
                }
            }

            // Queen turn placement
            for (i, mv) in history.iter().enumerate() {
                if let Some(piece) = mv.piece {
                    if piece.piece_type() == PieceType::Queen && mv.from.is_none() {
                        let turn = (i as u32) + 1;
                        match piece.color() {
                            PieceColor::White => white_queen_turn.push(turn),
                            PieceColor::Black => black_queen_turn.push(turn),
                        }
                    }
                }
            }

            // Queen neighbor analysis for decisive games
            let wq = hive_game::piece::Piece::new(PieceColor::White, PieceType::Queen, 1);
            let bq = hive_game::piece::Piece::new(PieceColor::Black, PieceType::Queen, 1);

            // Analyze decisive games: killer piece, surrounding pieces, beetle on queen
            let (losing_queen_pos, winner_color) = match game.state.as_str() {
                "WhiteWins" => (game.board.piece_position(bq), Some(PieceColor::White)),
                "BlackWins" => (game.board.piece_position(wq), Some(PieceColor::Black)),
                _ => (None, None),
            };
            if let Some(pos) = losing_queen_pos {
                let winner_color = winner_color.unwrap();
                let n = hex_neighbors(pos).iter()
                    .filter(|&&h| game.board.is_occupied(h))
                    .count() as u32;
                queen_neighbors_at_end.push(n);

                // Beetle on queen
                let stack = game.board.stack_at(pos);
                if stack.height() > 1 {
                    beetle_on_queen_wins += 1;
                }

                // Surrounding piece types, split by attacker/defender
                for &nh in &hex_neighbors(pos) {
                    if let Some(top) = game.board.top_piece(nh) {
                        let key = format!("{}", top.piece_type().as_char());
                        if top.color() == winner_color {
                            *surround_attacker.entry(key).or_default() += 1;
                        } else {
                            *surround_defender.entry(key).or_default() += 1;
                        }
                    }
                }

                // Killer piece: the last move's piece type
                if let Some(last_mv) = history.last() {
                    if let Some(piece) = last_mv.piece {
                        let key = format!("{}", piece.piece_type().as_char());
                        *killer_piece.entry(key).or_default() += 1;
                    }
                }
            }
        });
    }

    // Also process loose .sgf files
    let mut sgf_paths: Vec<_> = Vec::new();
    collect_sgfs(path, &mut sgf_paths);
    sgf_paths.sort();

    for sgf_path in &sgf_paths {
        let buf = match std::fs::read(sgf_path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let text: String = buf.iter().map(|&b| b as char).collect();
        total_games += 1;

        let gtype = sgf::game_type(&text);
        if gtype == "expansion" {
            skipped_expansion += 1;
            continue;
        }

        let mut game = Game::new();
        if sgf::replay_into_game(&text, &mut game).is_err() {
            errors += 1;
            continue;
        }
        replayed_ok += 1;

        match game.state.as_str() {
            "WhiteWins" => white_wins += 1,
            "BlackWins" => black_wins += 1,
            "Draw" => draws += 1,
            _ => { in_progress += 1; }
        }
        lengths.push(game.move_count as u32);

        let history = game.move_history();
        if let Some(mv) = history.first() {
            if let Some(piece) = mv.piece {
                *first_bug.entry(format!("{}", piece.piece_type().as_char())).or_default() += 1;
            }
        }
        if history.len() >= 2 {
            if let Some(piece) = history[1].piece {
                *second_bug.entry(format!("{}", piece.piece_type().as_char())).or_default() += 1;
            }
        }

        let w_pieces: u32 = player_pieces(PieceColor::White).iter()
            .filter(|p| game.board.piece_position(**p).is_some()).count() as u32;
        let b_pieces: u32 = player_pieces(PieceColor::Black).iter()
            .filter(|p| game.board.piece_position(**p).is_some()).count() as u32;
        white_pieces_end.push(w_pieces);
        black_pieces_end.push(b_pieces);
        total_pieces_end.push(w_pieces + b_pieces);

        for pt in &[PieceType::Queen, PieceType::Spider, PieceType::Beetle,
                    PieceType::Grasshopper, PieceType::Ant] {
            let mut count = 0u32;
            for color in &[PieceColor::White, PieceColor::Black] {
                for p in player_pieces(*color) {
                    if p.piece_type() == *pt && game.board.piece_position(p).is_some() {
                        count += 1;
                    }
                }
            }
            piece_type_counts.entry(format!("{}", pt.as_char())).or_insert_with(Vec::new).push(count);
        }

        for (i, mv) in history.iter().enumerate() {
            if let Some(piece) = mv.piece {
                if piece.piece_type() == PieceType::Queen && mv.from.is_none() {
                    let turn = (i as u32) + 1;
                    match piece.color() {
                        PieceColor::White => white_queen_turn.push(turn),
                        PieceColor::Black => black_queen_turn.push(turn),
                    }
                }
            }
        }
    }

    eprintln!();

    // --- Print report ---
    println!();
    println!("=== Boardspace Hive Game Statistics ===");
    println!();
    println!("Games:   {} total, {} replayed OK, {} skipped (expansion), {} errors",
        total_games, replayed_ok, skipped_expansion, errors);
    println!();

    let decided = white_wins + black_wins + draws;
    println!("--- Results ---");
    println!("  White wins: {:5}  ({:.1}%)", white_wins, pct(white_wins, decided));
    println!("  Black wins: {:5}  ({:.1}%)", black_wins, pct(black_wins, decided));
    println!("  Draws:      {:5}  ({:.1}%)", draws, pct(draws, decided));
    println!("  In progress:{:5}", in_progress);
    println!();

    // White = first player in Hive
    let decisive = white_wins + black_wins;
    println!("--- First Player (White) Advantage ---");
    println!("  White (1st) wins: {:5}  ({:.1}%)", white_wins, pct(white_wins, decisive));
    println!("  Black (2nd) wins: {:5}  ({:.1}%)", black_wins, pct(black_wins, decisive));
    println!();

    // Game length
    if !lengths.is_empty() {
        lengths.sort();
        let n = lengths.len();
        let sum: u64 = lengths.iter().map(|&x| x as u64).sum();
        let avg = sum as f64 / n as f64;
        let median = if n % 2 == 0 {
            (lengths[n/2-1] + lengths[n/2]) as f64 / 2.0
        } else { lengths[n/2] as f64 };
        println!("--- Game Length (moves) ---");
        println!("  Min:    {:5}", lengths[0]);
        println!("  Max:    {:5}", lengths[n-1]);
        println!("  Mean:   {:8.1}", avg);
        println!("  Median: {:8.1}", median);
        println!("  P10:    {:5}", lengths[n/10]);
        println!("  P90:    {:5}", lengths[n*9/10]);
        println!();
    }

    // First and second bug
    println!("--- First Bug Played (White's opening) ---");
    let total_first = first_bug.values().sum::<u64>();
    let mut fb: Vec<_> = first_bug.iter().collect();
    fb.sort_by(|a, b| b.1.cmp(a.1));
    for (bug, count) in &fb {
        let name = match bug.as_str() {
            "Q" => "Queen", "S" => "Spider", "B" => "Beetle",
            "G" => "Grasshopper", "A" => "Ant", x => x,
        };
        println!("  {:12} {:5}  ({:.1}%)", name, count, pct(**count, total_first));
    }
    println!();

    println!("--- Second Bug Played (Black's opening) ---");
    let total_second = second_bug.values().sum::<u64>();
    let mut sb: Vec<_> = second_bug.iter().collect();
    sb.sort_by(|a, b| b.1.cmp(a.1));
    for (bug, count) in &sb {
        let name = match bug.as_str() {
            "Q" => "Queen", "S" => "Spider", "B" => "Beetle",
            "G" => "Grasshopper", "A" => "Ant", x => x,
        };
        println!("  {:12} {:5}  ({:.1}%)", name, count, pct(**count, total_second));
    }
    println!();

    // Pieces on board
    if !total_pieces_end.is_empty() {
        total_pieces_end.sort();
        white_pieces_end.sort();
        black_pieces_end.sort();
        let n = total_pieces_end.len();
        let avg_t = total_pieces_end.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let avg_w = white_pieces_end.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let avg_b = black_pieces_end.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let med = |v: &[u32]| if v.len() % 2 == 0 {
            (v[v.len()/2-1] + v[v.len()/2]) as f64 / 2.0
        } else { v[v.len()/2] as f64 };
        println!("--- Pieces on Board at Game End (max 11 per side, 22 total) ---");
        println!("  Total:  min={}, max={}, mean={:.1}, median={:.1}",
            total_pieces_end[0], total_pieces_end[n-1], avg_t, med(&total_pieces_end));
        println!("  White:  min={}, max={}, mean={:.1}, median={:.1}",
            white_pieces_end[0], white_pieces_end[n-1], avg_w, med(&white_pieces_end));
        println!("  Black:  min={}, max={}, mean={:.1}, median={:.1}",
            black_pieces_end[0], black_pieces_end[n-1], avg_b, med(&black_pieces_end));
        println!();

        // Per piece type average on board
        println!("  By type (avg on board / max possible):");
        for (label, key, max) in &[
            ("Queen", "Q", 2), ("Spider", "S", 4), ("Beetle", "B", 4),
            ("Grasshopper", "G", 6), ("Ant", "A", 6),
        ] {
            if let Some(counts) = piece_type_counts.get(*key) {
                let avg = counts.iter().map(|&x| x as f64).sum::<f64>() / counts.len() as f64;
                println!("    {:12} {:.2} / {}", label, avg, max);
            }
        }
        println!();
    }

    // Board dimensions
    if !diameters.is_empty() {
        diameters.sort();
        q_spans.sort();
        r_spans.sort();
        s_spans.sort();
        let n = diameters.len();
        let pavg = |v: &[u32]| v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
        let pp = |v: &[u32], p: f64| -> u32 {
            let idx = ((v.len() as f64 - 1.0) * p / 100.0).round() as usize;
            v[idx.min(v.len() - 1)]
        };
        println!("--- Board Size at Game End ---");
        println!("  {:>12}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}",
            "", "min", "p10", "p25", "med", "p75", "p90", "max");
        for (label, v) in [("diameter", &diameters), ("q span", &q_spans),
                           ("r span", &r_spans), ("s span", &s_spans)] {
            println!("  {:>12}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}",
                label, v[0], pp(v, 10.0), pp(v, 25.0),
                pp(v, 50.0), pp(v, 75.0), pp(v, 90.0), v[n-1]);
        }
        println!("  Avg diameter: {:.1}", pavg(&diameters));
        println!();
    }

    // Queen placement turn
    if !white_queen_turn.is_empty() {
        white_queen_turn.sort();
        black_queen_turn.sort();
        let n_w = white_queen_turn.len();
        let n_b = black_queen_turn.len();
        let avg_w = white_queen_turn.iter().map(|&x| x as f64).sum::<f64>() / n_w as f64;
        let avg_b = black_queen_turn.iter().map(|&x| x as f64).sum::<f64>() / n_b as f64;
        let med = |v: &[u32]| if v.len() % 2 == 0 {
            (v[v.len()/2-1] + v[v.len()/2]) as f64 / 2.0
        } else { v[v.len()/2] as f64 };
        println!("--- Queen Placement Move Number ---");
        println!("  White: mean={:.1}, median={:.1}, min={}, max={}",
            avg_w, med(&white_queen_turn), white_queen_turn[0], white_queen_turn[n_w-1]);
        println!("  Black: mean={:.1}, median={:.1}, min={}, max={}",
            avg_b, med(&black_queen_turn), black_queen_turn[0], black_queen_turn[n_b-1]);
        println!();
    }

    // Piece movement frequency
    println!("--- Piece Movement Frequency (moves after placement) ---");
    let mut mf: Vec<_> = move_freq.iter().collect();
    mf.sort_by(|a, b| b.1.cmp(a.1));
    for (bug, count) in &mf {
        let name = match bug.as_str() {
            "Q" => "Queen", "S" => "Spider", "B" => "Beetle",
            "G" => "Grasshopper", "A" => "Ant", x => x,
        };
        println!("  {:12} {:7}  ({:.1}%)", name, count, pct(**count, total_movements));
    }
    println!("  Total movements: {}", total_movements);
    println!();

    // Killer piece
    println!("--- Killer Piece (last move in decisive games) ---");
    let total_kills = killer_piece.values().sum::<u64>();
    let mut kp: Vec<_> = killer_piece.iter().collect();
    kp.sort_by(|a, b| b.1.cmp(a.1));
    for (bug, count) in &kp {
        let name = match bug.as_str() {
            "Q" => "Queen", "S" => "Spider", "B" => "Beetle",
            "G" => "Grasshopper", "A" => "Ant", x => x,
        };
        println!("  {:12} {:5}  ({:.1}%)", name, count, pct(**count, total_kills));
    }
    println!();

    // Surrounding pieces
    fn bug_name(b: &str) -> &'static str {
        match b {
            "Q" => "Queen", "S" => "Spider", "B" => "Beetle",
            "G" => "Grasshopper", "A" => "Ant", _ => "?",
        }
    }
    let total_atk = surround_attacker.values().sum::<u64>();
    let total_def = surround_defender.values().sum::<u64>();
    let total_surround = total_atk + total_def;
    println!("--- Pieces Surrounding Losing Queen ({} total, {:.1} avg per game) ---",
        total_surround, total_surround as f64 / decisive.max(1) as f64);
    println!("  Attacker's pieces ({}, {:.1}%):", total_atk, pct(total_atk, total_surround));
    let mut sa: Vec<_> = surround_attacker.iter().collect();
    sa.sort_by(|a, b| b.1.cmp(a.1));
    for (bug, count) in &sa {
        println!("    {:12} {:7}  ({:.1}% of attacker)", bug_name(bug), count, pct(**count, total_atk));
    }
    println!("  Defender's own pieces ({}, {:.1}%):", total_def, pct(total_def, total_surround));
    let mut sd: Vec<_> = surround_defender.iter().collect();
    sd.sort_by(|a, b| b.1.cmp(a.1));
    for (bug, count) in &sd {
        println!("    {:12} {:7}  ({:.1}% of defender)", bug_name(bug), count, pct(**count, total_def));
    }
    println!();

    // Beetle on queen
    println!("--- Winning Position Details ---");
    println!("  Beetle on losing queen: {:5}  ({:.1}% of decisive games)",
        beetle_on_queen_wins, pct(beetle_on_queen_wins, decisive));
    if !queen_neighbors_at_end.is_empty() {
        // Should all be 6, but let's verify
        let mut qn_counts: HashMap<u32, u64> = HashMap::new();
        for &n in &queen_neighbors_at_end {
            *qn_counts.entry(n).or_default() += 1;
        }
        println!("  Losing queen neighbor count distribution:");
        let mut sorted_qn: Vec<_> = qn_counts.iter().collect();
        sorted_qn.sort_by_key(|(k, _)| *k);
        for (n, count) in sorted_qn {
            println!("    {} neighbors: {:5}  ({:.1}%)", n, count,
                pct(*count, queen_neighbors_at_end.len() as u64));
        }
    }
    println!();

    // Write stats to file
    let stats_path = path.join("game_stats.txt");
    if let Ok(mut f) = std::fs::File::create(&stats_path) {
        use std::io::Write;
        writeln!(f, "Boardspace Hive Game Statistics").ok();
        writeln!(f, "===============================").ok();
        writeln!(f).ok();
        writeln!(f, "Games: {} total, {} replayed OK, {} skipped (expansion), {} errors",
            total_games, replayed_ok, skipped_expansion, errors).ok();
        writeln!(f).ok();
        writeln!(f, "Results:").ok();
        writeln!(f, "  White wins: {} ({:.1}%)", white_wins, pct(white_wins, decided)).ok();
        writeln!(f, "  Black wins: {} ({:.1}%)", black_wins, pct(black_wins, decided)).ok();
        writeln!(f, "  Draws: {} ({:.1}%)", draws, pct(draws, decided)).ok();
        writeln!(f, "  In progress: {}", in_progress).ok();
        writeln!(f).ok();
        writeln!(f, "First Player (White) Advantage:").ok();
        writeln!(f, "  White (1st) wins: {} ({:.1}%)", white_wins, pct(white_wins, decisive)).ok();
        writeln!(f, "  Black (2nd) wins: {} ({:.1}%)", black_wins, pct(black_wins, decisive)).ok();
        writeln!(f).ok();
        if !lengths.is_empty() {
            let n = lengths.len();
            let sum: u64 = lengths.iter().map(|&x| x as u64).sum();
            writeln!(f, "Game Length (moves): min={}, max={}, mean={:.1}, median={:.1}",
                lengths[0], lengths[n-1], sum as f64/n as f64,
                if n%2==0 { (lengths[n/2-1]+lengths[n/2]) as f64/2.0 } else { lengths[n/2] as f64 }
            ).ok();
            writeln!(f).ok();
        }
        writeln!(f, "First Bug (White): {:?}", fb.iter().map(|(b,c)| format!("{}={}", b, c)).collect::<Vec<_>>()).ok();
        writeln!(f, "Second Bug (Black): {:?}", sb.iter().map(|(b,c)| format!("{}={}", b, c)).collect::<Vec<_>>()).ok();
        writeln!(f).ok();
        if !total_pieces_end.is_empty() {
            let n = total_pieces_end.len();
            writeln!(f, "Pieces on Board at End: min={}, max={}, mean={:.1}",
                total_pieces_end[0], total_pieces_end[n-1],
                total_pieces_end.iter().map(|&x| x as f64).sum::<f64>() / n as f64).ok();
        }
        writeln!(f).ok();
        writeln!(f, "Beetle on losing queen: {} ({:.1}% of decisive)",
            beetle_on_queen_wins, pct(beetle_on_queen_wins, decisive)).ok();
        println!("Wrote {}", stats_path.display());
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "hive-tools", about = "Hive game tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Play N random games (default 1 shows board after each move)
    Random {
        /// Number of games to play (1 = turn-by-turn display)
        #[arg(default_value_t = 1)]
        n: u32,
    },
    /// Replay boardspace games from a path
    Replay {
        /// Path to zip dir or file
        #[arg(default_value = "games/hive/boardspace")]
        path: String,
    },
    /// Verbose replay of one game (SGF file, or zip + SGF name)
    Debug {
        /// SGF file or zip archive
        path: String,
        /// Name of the SGF inside the zip (required when path is a zip)
        sgf_name: Option<String>,
    },
    /// Replay all games, compute ELO, write CSVs
    Process {
        /// Path to zip dir or file
        #[arg(default_value = "games/hive/boardspace")]
        path: String,
        /// Discard training data from games that hit the move cap
        #[arg(long)]
        skip_timeout_games: bool,
    },
    /// Aggregate game statistics, write game_stats.txt
    Stats {
        /// Path to zip dir or file
        #[arg(default_value = "games/hive/boardspace")]
        path: String,
    },
    /// Run MCTS benchmark with uniform policy
    Mcts {
        /// Number of simulations
        #[arg(default_value_t = 800)]
        simulations: u32,
        /// Batch size
        #[arg(default_value_t = 8)]
        batch_size: usize,
    },
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Random { n } => run_random(n),
        Command::Replay { path } => replay::run_replay(&path),
        Command::Debug { path, sgf_name } => {
            let mut args = vec![path];
            if let Some(name) = sgf_name { args.push(name); }
            replay::run_debug(&args);
        }
        Command::Process { path, skip_timeout_games } => run_process(&path, skip_timeout_games),
        Command::Stats { path } => run_stats(&path),
        Command::Mcts { simulations, batch_size } => run_mcts(simulations, batch_size),
    }
}
