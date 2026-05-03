//! Texel tuning: fit linear eval weights from boardspace YINSH outcomes.
//!
//! Walks boardspace zips, replays each game, samples Normal-phase positions,
//! labels each with the eventual game result from the side-to-move's
//! perspective (1.0 win / 0.5 draw / 0.0 loss), and fits the 6 alphabeta
//! weights via gradient descent on a logistic MSE loss.
//!
//! Position extraction is the slow part, so the (features, label) matrix is
//! cached to a CSV between runs. Fitting is a few thousand passes over the
//! cached matrix and takes seconds.

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use core_game::game::{Outcome, Player};
use rand::RngExt;
use yinsh_game::alphabeta::{DEFAULT_WEIGHTS, N_FEATURES, extract_features};
use yinsh_game::board::Phase;
use yinsh_game::sgf::{self, Color};

use crate::replay;

const FEATURE_NAMES: [&str; N_FEATURES] = [
    "SCORE", "ROW5", "ROW4", "ROW3", "MOBILITY", "MARKER",
];

#[derive(Debug, Clone)]
struct LabeledPosition {
    game_id: u32,
    features: [f32; N_FEATURES],
    /// 1.0 = the player to move eventually wins, 0.5 = draw, 0.0 = loses.
    label: f32,
}

pub struct TuneOptions {
    pub games_path: String,
    pub cache_path: Option<PathBuf>,
    pub regen_cache: bool,
    pub sample_rate: f32,
    pub k: f32,
    pub lr: f32,
    pub epochs: usize,
    pub val_frac: f32,
}

pub fn run_tune(opts: TuneOptions) {
    let path = Path::new(&opts.games_path);
    if !path.exists() {
        eprintln!("Path not found: {}", opts.games_path);
        return;
    }

    let cache_path = opts.cache_path.clone().unwrap_or_else(|| {
        let dir = if path.is_dir() {
            path.to_path_buf()
        } else {
            path.parent().unwrap_or(path).to_path_buf()
        };
        dir.join("tune_positions.csv")
    });

    let positions = if cache_path.exists() && !opts.regen_cache {
        println!("Loading positions from cache: {}", cache_path.display());
        load_positions(&cache_path).expect("failed to load cache")
    } else {
        println!("Extracting positions from {} (sample_rate={})...",
            opts.games_path, opts.sample_rate);
        let extracted = extract_all(path, opts.sample_rate);
        println!("Caching {} positions to {}", extracted.len(), cache_path.display());
        save_positions(&cache_path, &extracted).expect("failed to write cache");
        extracted
    };

    if positions.is_empty() {
        eprintln!("No positions to fit on. Bailing.");
        return;
    }

    println!("Total positions: {}", positions.len());

    // Train/val split by game_id (not by position) to prevent leakage between
    // positions of the same game. Hashed split avoids chronological skew —
    // game_ids are assigned in zip-order, so a simple cutoff would put only
    // the most recent years in val.
    let val_thresh = ((opts.val_frac as f64) * (u64::MAX as f64)) as u64;
    let is_val = |gid: u32| -> bool {
        // SplitMix64 — quick, well-distributed scrambler.
        let mut z = (gid as u64).wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (z ^ (z >> 31)) < val_thresh
    };
    let mut train: Vec<&LabeledPosition> = Vec::new();
    let mut val: Vec<&LabeledPosition> = Vec::new();
    let mut train_games = std::collections::HashSet::new();
    let mut val_games = std::collections::HashSet::new();
    for p in &positions {
        if is_val(p.game_id) {
            val.push(p);
            val_games.insert(p.game_id);
        } else {
            train.push(p);
            train_games.insert(p.game_id);
        }
    }
    println!(
        "Train: {} positions ({} games), Val: {} positions ({} games), K={}",
        train.len(), train_games.len(), val.len(), val_games.len(), opts.k
    );

    let mut weights = DEFAULT_WEIGHTS;
    println!(
        "Initial: train_loss={:.6}, val_loss={:.6}",
        mse_loss(&weights, &train, opts.k),
        mse_loss(&weights, &val, opts.k),
    );

    // Adam optimizer — handles the very different feature scales without
    // per-feature normalization.
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps = 1e-8f32;
    let mut m = [0.0f32; N_FEATURES];
    let mut v = [0.0f32; N_FEATURES];

    for epoch in 1..=opts.epochs {
        let g = mse_gradient(&weights, &train, opts.k);
        let bc1 = 1.0 - beta1.powi(epoch as i32);
        let bc2 = 1.0 - beta2.powi(epoch as i32);
        for i in 0..N_FEATURES {
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            weights[i] -= opts.lr * m_hat / (v_hat.sqrt() + eps);
        }
        let log_now = epoch == 1
            || epoch == opts.epochs
            || (epoch <= 100 && epoch % 10 == 0)
            || (epoch <= 1000 && epoch % 100 == 0)
            || epoch % 500 == 0;
        if log_now {
            println!(
                "Epoch {epoch:>5}: train_loss={:.6}, val_loss={:.6}",
                mse_loss(&weights, &train, opts.k),
                mse_loss(&weights, &val, opts.k),
            );
        }
    }

    println!();
    println!("--- Fitted weights (initial vs fitted) ---");
    for i in 0..N_FEATURES {
        println!(
            "  {:9}  initial={:>12.4}  fitted={:>12.4}  delta={:>+10.4}",
            FEATURE_NAMES[i],
            DEFAULT_WEIGHTS[i],
            weights[i],
            weights[i] - DEFAULT_WEIGHTS[i],
        );
    }

    println!();
    println!("--- Suggested Rust constants ---");
    println!("pub const DEFAULT_WEIGHTS: [f32; N_FEATURES] = [");
    for i in 0..N_FEATURES {
        println!("    {:>12.4}, // {}", weights[i], FEATURE_NAMES[i]);
    }
    println!("];");
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

fn extract_all(path: &Path, sample_rate: f32) -> Vec<LabeledPosition> {
    let mut positions: Vec<LabeledPosition> = Vec::new();
    let mut next_game_id: u32 = 0;
    let mut total: u64 = 0;
    let mut used: u64 = 0;
    let mut rng = rand::rng();

    let mut process_zip = |zip_path: &Path| {
        let zip_name = zip_path.display().to_string();
        let file = match std::fs::File::open(zip_path) {
            Ok(f) => f,
            Err(e) => { eprintln!("  failed to open {zip_name}: {e}"); return; }
        };

        let _ = core_game::sgf::iter_sgf_texts_in_zip(file, |_sgf_name, text| {
            total += 1;
            if total % 2000 == 0 {
                eprint!(
                    "\r  {} games scanned, {} usable, {} positions...   ",
                    total, used, positions.len()
                );
            }

            let record = match sgf::parse_game(&text) {
                Ok(r) => r,
                Err(_) => return,
            };
            if !matches!(record.first_player_color, Color::White) { return; }

            // First pass: replay the whole game to confirm it terminates and
            // determine the engine-verified outcome.
            let result = replay::replay_game(&record);
            if result.error.is_some() { return; }
            let (label_p1, label_p2) = match result.final_board.outcome {
                Outcome::WonBy(Player::Player1) => (1.0f32, 0.0f32),
                Outcome::WonBy(Player::Player2) => (0.0f32, 1.0f32),
                Outcome::Draw                   => (0.5f32, 0.5f32),
                Outcome::Ongoing                => return,
            };

            let game_id = next_game_id;
            next_game_id += 1;
            used += 1;

            // Second pass: sample Normal-phase positions during replay.
            let _ = replay::replay_game_observed(&record, |board, _i| {
                if board.phase != Phase::Normal { return; }
                if rng.random::<f32>() > sample_rate { return; }
                let features = extract_features(board);
                let label = match board.next_player {
                    Player::Player1 => label_p1,
                    Player::Player2 => label_p2,
                };
                positions.push(LabeledPosition { game_id, features, label });
            });
        });
    };

    if path.is_file() && path.extension().is_some_and(|e| e == "zip") {
        process_zip(path);
    } else if path.is_dir() {
        core_game::sgf::visit_zip_dir(path, &mut process_zip);
    }
    eprintln!();
    positions
}

// ---------------------------------------------------------------------------
// Loss + gradient
// ---------------------------------------------------------------------------

#[inline]
fn dot(weights: &[f32; N_FEATURES], features: &[f32; N_FEATURES]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..N_FEATURES {
        s += weights[i] * features[i];
    }
    s
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn mse_loss(weights: &[f32; N_FEATURES], positions: &[&LabeledPosition], k: f32) -> f32 {
    if positions.is_empty() { return 0.0; }
    let mut sum = 0.0f32;
    for p in positions {
        let s = sigmoid(dot(weights, &p.features) / k);
        let d = p.label - s;
        sum += d * d;
    }
    sum / positions.len() as f32
}

/// Gradient of MSE loss w.r.t. weights. For one position with features f,
/// label y, scale K, score s = w·f, sigmoid σ = σ(s/K):
///
///   d/dw_j [(y - σ)²] = -2 (y - σ) * σ(1-σ) / K * f_j
fn mse_gradient(
    weights: &[f32; N_FEATURES],
    positions: &[&LabeledPosition],
    k: f32,
) -> [f32; N_FEATURES] {
    let mut grad = [0.0f32; N_FEATURES];
    if positions.is_empty() { return grad; }
    for p in positions {
        let sig = sigmoid(dot(weights, &p.features) / k);
        let common = -2.0 * (p.label - sig) * sig * (1.0 - sig) / k;
        for i in 0..N_FEATURES {
            grad[i] += common * p.features[i];
        }
    }
    let inv_n = 1.0 / positions.len() as f32;
    for i in 0..N_FEATURES { grad[i] *= inv_n; }
    grad
}

// ---------------------------------------------------------------------------
// Cache I/O
// ---------------------------------------------------------------------------

fn save_positions(path: &Path, positions: &[LabeledPosition]) -> std::io::Result<()> {
    let mut f = BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        f,
        "game_id,score_diff,row5_diff,row4_diff,row3_diff,mobility_diff,marker_diff,label"
    )?;
    for p in positions {
        writeln!(
            f, "{},{},{},{},{},{},{},{}",
            p.game_id,
            p.features[0], p.features[1], p.features[2],
            p.features[3], p.features[4], p.features[5],
            p.label,
        )?;
    }
    Ok(())
}

fn load_positions(path: &Path) -> std::io::Result<Vec<LabeledPosition>> {
    let f = BufReader::new(std::fs::File::open(path)?);
    let mut positions = Vec::new();
    for (i, line) in f.lines().enumerate() {
        if i == 0 { continue; } // header
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 8 { continue; }
        let game_id = parts[0].parse().unwrap_or(0);
        let mut features = [0.0f32; N_FEATURES];
        let mut ok = true;
        for j in 0..N_FEATURES {
            match parts[1 + j].parse::<f32>() {
                Ok(x) => features[j] = x,
                Err(_) => { ok = false; break; }
            }
        }
        if !ok { continue; }
        let label = match parts[7].parse::<f32>() {
            Ok(x) => x,
            Err(_) => continue,
        };
        positions.push(LabeledPosition { game_id, features, label });
    }
    Ok(positions)
}
