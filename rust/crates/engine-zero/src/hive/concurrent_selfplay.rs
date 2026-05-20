//! Concurrent self-play with cross-game batching for the token Hive
//! transformer.
//!
//! The Phase F single-game `play_one_game` runs each Hive game to
//! completion before starting the next, which means every inference call
//! batches at most `play_batch` leaves from one game — wasted GPU. This
//! module runs K games in lockstep: every round, all active games each
//! contribute up to `play_batch` MCTS leaves to one big inference call,
//! the results are distributed back, and the loop continues until every
//! game finishes. Effective batch size is up to `K * play_batch`.
//!
//! Lockstep structure ("phases"):
//!
//!   1. INIT — each game tokenizes its current root, all roots go into one
//!      inference call, each search is initialised with its policy and
//!      Dirichlet noise applied. Sims-done counter reset.
//!   2. SIM — every round: each game selects up to `play_batch` leaves,
//!      all leaves across all games go into ONE inference call, results
//!      distributed back per game. Loop until every active game has done
//!      `simulations` sims for the current move.
//!   3. ADVANCE — each game samples a chosen move (temperature-aware),
//!      records the per-ply training sample (token batch + visit
//!      distribution + root_q), plays the move, decides whether to re-INIT
//!      or mark complete.
//!
//! Games complete at different ply counts; once a game finishes, its slot
//! is dropped from subsequent rounds rather than backfilled (keeps the
//! code simple, costs only the tail latency of the longest game).

use std::collections::HashMap;
use std::sync::mpsc::{channel, sync_channel};
use std::thread;

use core_game::game::PolicyIndex;
use core_game::mcts::arena::NodeId;
use core_game::mcts::search::{
    CpuctStrategy, ForcedExploration, MctsSearch, RootNoise, SearchParams,
};

use hive_game::game::{Game, Move};
use hive_game::tokenize::{self, TokenBatch};

use super::inference::HiveTokenOrtEngine;

/// Per-ply training sample captured during a game.
pub struct PlySample {
    pub tokens: TokenBatch,
    /// Visit distribution as (mover_slot, dest_slot, prob) triples.
    pub move_src: Vec<u8>,
    pub move_dst: Vec<u8>,
    pub move_probs: Vec<f32>,
    pub root_q: f32,
    pub mover_is_white: bool,
}

/// Per-game outcome plus training samples.
pub struct ConcurrentGameResult {
    pub outcome: String,
    pub move_count: u32,
    pub final_uhp: String,
    pub final_board_render: String,
    pub samples: Vec<PlySample>,
}

/// Session-level MCTS stats, aggregated across every ply of every game.
/// Mirrors the legacy CNN-side `SelfPlayResult.{top1,search_depth,valid_moves}_{mean,std}`.
#[derive(Default)]
pub struct ConcurrentSessionStats {
    pub top1_visit_fraction_mean: f32,
    pub top1_visit_fraction_std: f32,
    pub search_depth_mean: f32,
    pub search_depth_std: f32,
    pub valid_moves_mean: f32,
    pub valid_moves_std: f32,
}

/// Running accumulators built up across plies; consumed once to compute
/// the per-session mean/std at the end of the run.
#[derive(Default)]
struct StatsAccumulators {
    top1_sum: f64,
    top1_sum_sq: f64,
    top1_count: u64,
    depth_sum: f64,
    depth_sum_sq: f64,
    depth_count: u64,
    moves_sum: f64,
    moves_sum_sq: f64,
    moves_count: u64,
}

fn mean_std(sum: f64, sum_sq: f64, count: u64) -> (f32, f32) {
    if count == 0 { return (0.0, 0.0); }
    let n = count as f64;
    let mean = sum / n;
    let var = (sum_sq / n) - mean * mean;
    (mean as f32, var.max(0.0).sqrt() as f32)
}

impl StatsAccumulators {
    fn finalize(&self) -> ConcurrentSessionStats {
        let (top1_visit_fraction_mean, top1_visit_fraction_std) =
            mean_std(self.top1_sum, self.top1_sum_sq, self.top1_count);
        let (search_depth_mean, search_depth_std) =
            mean_std(self.depth_sum, self.depth_sum_sq, self.depth_count);
        let (valid_moves_mean, valid_moves_std) =
            mean_std(self.moves_sum, self.moves_sum_sq, self.moves_count);
        ConcurrentSessionStats {
            top1_visit_fraction_mean,
            top1_visit_fraction_std,
            search_depth_mean,
            search_depth_std,
            valid_moves_mean,
            valid_moves_std,
        }
    }
}

/// Self-play configuration shared across all concurrent games.
#[derive(Clone)]
pub struct ConcurrentSelfPlayConfig {
    pub num_games: usize,
    pub simulations: usize,
    pub play_batch: usize,
    pub c_puct: f32,
    pub draw_contempt: f32,
    pub dir_alpha: f32,
    pub dir_epsilon: f32,
    pub max_moves: u32,
    pub temperature_moves: u32,
    pub temperature_start: f32,
    pub temperature_end: f32,
    pub grid_size: usize,
    pub tournament_mode: bool,
    pub rng_seed: u64,
}

/// Per-game in-progress state during the concurrent run.
struct GameSlot {
    game: Game,
    search: MctsSearch<Game>,
    /// Number of sims credited for the *current* move (incremented at
    /// select-leaves time, not at distribute time — keeps the sim budget
    /// honest when some selections hit terminal nodes that get backpropped
    /// internally and don't need NN evaluation).
    sims_for_current_move: usize,
    /// Whether the search has been initialised at the current root.
    initialised: bool,
    /// True while an INIT batch entry for this slot is submitted but its
    /// result hasn't been distributed yet. Prevents building a duplicate
    /// INIT in the next pipelined round before the worker responds.
    init_in_flight: bool,
    /// Count of SIM leaves submitted to the worker but not yet distributed.
    /// ADVANCE is blocked until this drains to zero so the visit
    /// distribution at advance time reflects every sim we counted toward
    /// the cap.
    sims_in_flight: usize,
    samples: Vec<PlySample>,
    /// White's value once the game finishes — None until then.
    rng_state: u64,
    /// Whether this game has finished (game over or hit max_moves).
    complete: bool,
}

/// Progress callback fired once per SIM round.
/// `(round_idx, active_games, batch_size, sims_per_active_game)`.
pub type ProgressFn<'a> = Box<dyn FnMut(usize, usize, usize, &[(u32, usize)]) + 'a>;

/// Drive `cfg.num_games` games concurrently with cross-game batching.
///
/// Inference is pipelined on a dedicated worker thread: each main-loop
/// iteration submits batch N+1 to the worker (non-blocking) and then waits
/// on + distributes batch N, so the GPU forward for the previous batch
/// overlaps with leaf selection + encoding for the next one. Mirrors the
/// Yinsh/Zertz/Hive-CNN self-play pipelining pattern.
///
/// `play_batch` is halved relative to the caller's value so per-game
/// virtual-loss accumulation stays constant: at any time two batches are
/// alive (one on the GPU + one being prepared), so VL builds up over
/// `2 × play_batch` leaves per game vs. `play_batch` in the synchronous
/// version. Halving keeps the same effective VL ceiling and avoids over-
/// pessimistic exploration on narrow trees.
///
/// `progress` is invoked once per inference round if provided.
pub fn play_games_concurrent_ort(
    cfg: &ConcurrentSelfPlayConfig,
    engine: &mut HiveTokenOrtEngine,
    mut progress: Option<ProgressFn>,
) -> Result<(Vec<ConcurrentGameResult>, ConcurrentSessionStats), String> {
    // Halve play_batch so per-game in-flight VL leaves stay constant —
    // matches Hive-CNN / Yinsh / Zertz pipelined behavior.
    let play_batch = (cfg.play_batch / 2).max(1);

    let mut stats = StatsAccumulators::default();
    let mut slots: Vec<GameSlot> = (0..cfg.num_games)
        .map(|i| GameSlot {
            game: if cfg.tournament_mode {
                Game::new_tournament_with_grid_size(cfg.grid_size)
            } else {
                Game::new_with_grid_size(cfg.grid_size)
            },
            search: MctsSearch::<Game>::new(cfg.simulations.saturating_add(64)),
            sims_for_current_move: 0,
            initialised: false,
            init_in_flight: false,
            sims_in_flight: 0,
            samples: Vec::new(),
            rng_state: cfg.rng_seed.wrapping_add(i as u64 * 0x9E37_79B9_7F4A_7C15),
            complete: false,
        })
        .collect();

    let search_params = build_params(cfg);

    // Worker channels: req carries owned batches, resp carries owned
    // (policies, values, draws). Bounded request capacity = 1 (one batch
    // queued + one being processed by the worker), matching Yinsh/Zertz.
    type InferOk = (Vec<Vec<f32>>, Vec<f32>, Vec<f32>);
    let (req_tx, req_rx) = sync_channel::<Vec<TokenBatch>>(1);
    let (resp_tx, resp_rx) = channel::<Result<InferOk, String>>();

    let result = thread::scope(|scope| -> Result<Vec<ConcurrentGameResult>, String> {
        scope.spawn(move || {
            while let Ok(batches) = req_rx.recv() {
                let out = engine
                    .infer_token_batches(&batches)
                    .map_err(|e| e.to_string());
                if resp_tx.send(out).is_err() {
                    break;
                }
            }
        });

        // Pending = (routing, per-game SIM stashes, batch_size_for_progress).
        // routing is the same flat list of `(game_idx, RoutingKind)` the
        // synchronous version used; stashes are keyed by game_idx since
        // each SIM entry adds its slot's stash (caller-owned) here so the
        // next round's select_leaves can refill the search's internal stash
        // without clobbering this batch's leaves.
        let mut pending: Option<(Vec<(usize, RoutingKind)>, HashMap<usize, Vec<(NodeId, Game)>>, usize)> = None;
        let mut round_idx: usize = 0;

        loop {
            // ---- Phase A: distribute the PREVIOUS batch's results --------
            // Blocks on the worker. By the time we get here, we've already
            // submitted the new batch (further down in the previous iter),
            // so the GPU stays busy across this boundary.
            if let Some((prev_routing, mut prev_stashes, _)) = pending.take() {
                let (policies, values, draws) = resp_rx
                    .recv()
                    .map_err(|_| "worker thread died before responding".to_string())??;
                let mut cursor: usize = 0;
                for (gi, kind) in &prev_routing {
                    let slot = &mut slots[*gi];
                    match kind {
                        RoutingKind::Init => {
                            slot.search.init(&slot.game, &policies[cursor]);
                            if let RootNoise::Dirichlet { alpha, epsilon } = search_params.root_noise {
                                slot.search.params = search_params.clone();
                                slot.search.apply_root_dirichlet(alpha, epsilon);
                            } else {
                                slot.search.params = search_params.clone();
                            }
                            slot.initialised = true;
                            slot.init_in_flight = false;
                            slot.sims_for_current_move = 0;
                            slot.sims_in_flight = 0;
                            cursor += 1;
                        }
                        RoutingKind::Sim { count } => {
                            let pol_slice = &policies[cursor..cursor + count];
                            let val_slice = &values[cursor..cursor + count];
                            let draw_slice = &draws[cursor..cursor + count];
                            let stash = prev_stashes
                                .remove(gi)
                                .ok_or_else(|| format!("missing stash for game {gi}"))?;
                            slot.search.expand_and_backprop_with_stash(
                                stash, pol_slice, val_slice, draw_slice,
                            );
                            slot.sims_in_flight = slot.sims_in_flight.saturating_sub(*count);
                            cursor += count;
                        }
                    }
                }
            }

            // ---- Phase B: ADVANCE slots whose sims are fully resolved ----
            // Only fires when the slot has no SIMs in flight — otherwise the
            // visit distribution at advance_one time would be missing the
            // backed-up values for the not-yet-distributed leaves we already
            // counted toward `sims_for_current_move`.
            for slot in slots.iter_mut() {
                if slot.complete || !slot.initialised { continue; }
                if slot.sims_in_flight > 0 { continue; }
                if slot.sims_for_current_move < cfg.simulations { continue; }
                advance_one(slot, cfg, &mut stats)?;
            }

            // ---- Phase C: build INIT + SIM batches for the next round ----
            let mut batches: Vec<TokenBatch> = Vec::new();
            let mut routing: Vec<(usize, RoutingKind)> = Vec::new();
            let mut new_stashes: HashMap<usize, Vec<(NodeId, Game)>> = HashMap::new();

            // INIT: any uninitialised slot without an INIT already in flight.
            for (gi, slot) in slots.iter_mut().enumerate() {
                if slot.complete || slot.initialised || slot.init_in_flight { continue; }
                let mut root_game = slot.game.clone();
                let (tb, _) = tokenize::tokenize_and_priors(&mut root_game);
                batches.push(tb);
                routing.push((gi, RoutingKind::Init));
            }

            // SIM: initialised slots with sim budget remaining. We use the
            // halved `play_batch` so two pipelined batches still hit the
            // same effective per-game VL ceiling as the synchronous path.
            for (gi, slot) in slots.iter_mut().enumerate() {
                if slot.complete || !slot.initialised { continue; }
                if slot.sims_for_current_move >= cfg.simulations { continue; }
                let want = play_batch.min(cfg.simulations - slot.sims_for_current_move);
                let leaves = slot.search.select_leaves(want);
                // sims_for_current_move credits `want` regardless of
                // leaves.len() — terminal selections get backpropped
                // internally and don't need NN evaluation, but still count
                // toward the per-move sim budget.
                slot.sims_for_current_move += want;
                if leaves.is_empty() { continue; }
                for leaf in &leaves {
                    let mut g = slot
                        .search
                        .stashed_game(*leaf)
                        .ok_or_else(|| "stashed leaf missing".to_string())?
                        .clone();
                    let (tb, _) = tokenize::tokenize_and_priors(&mut g);
                    batches.push(tb);
                }
                // Take the stash NOW (after we've finished reading via
                // stashed_game) so the next round's select_leaves writes
                // into a fresh internal stash and can't clobber this
                // batch's leaves. Virtual loss remains applied to the
                // leaves until expand_and_backprop_with_stash runs, so
                // the next select still steers away from them.
                let leaf_count = leaves.len();
                let stash = slot.search.take_stash();
                new_stashes.insert(gi, stash);
                slot.sims_in_flight += leaf_count;
                routing.push((gi, RoutingKind::Sim { count: leaf_count }));
            }

            // ---- Phase D: submit the new batch (non-blocking) ------------
            let new_pending = if !batches.is_empty() {
                // Mark init_in_flight for INIT entries so the next iter
                // doesn't build a duplicate INIT before the result arrives.
                for (gi, kind) in &routing {
                    if matches!(kind, RoutingKind::Init) {
                        slots[*gi].init_in_flight = true;
                    }
                }
                let bs = batches.len();
                req_tx
                    .send(batches)
                    .map_err(|_| "worker thread closed request channel".to_string())?;
                Some((routing, new_stashes, bs))
            } else {
                None
            };

            // ---- Phase E: progress + termination check -------------------
            round_idx += 1;
            if let Some(cb) = progress.as_mut() {
                let snapshot: Vec<(u32, usize)> = slots
                    .iter()
                    .filter(|s| !s.complete)
                    .map(|s| (s.game.move_count as u32, s.sims_for_current_move))
                    .collect();
                let active = snapshot.len();
                let bs = new_pending.as_ref().map_or(0, |(_, _, b)| *b);
                cb(round_idx, active, bs, &snapshot);
            }

            pending = new_pending;

            // Nothing in flight + nothing to do → we're done. (If we still
            // had unfinished work we'd have built a batch this round and
            // pending would be Some.)
            if pending.is_none() && slots.iter().all(|s| s.complete) {
                break;
            }
        }

        // Drop sender so the worker's recv() returns Err and the thread exits.
        drop(req_tx);

        let mut out = Vec::with_capacity(slots.len());
        for slot in slots.drain(..) {
            out.push(ConcurrentGameResult {
                outcome: slot.game.state.as_str().to_string(),
                move_count: slot.game.move_count as u32,
                final_uhp: slot.game.game_string(),
                final_board_render: slot.game.board.render(None, None),
                samples: slot.samples,
            });
        }
        Ok(out)
    })?;

    Ok((result, stats.finalize()))
}

enum RoutingKind {
    /// First inference for this game's current root.
    Init,
    /// `count` leaves selected this round for this game.
    Sim { count: usize },
}

fn build_params(cfg: &ConcurrentSelfPlayConfig) -> SearchParams {
    let root_noise = if cfg.dir_alpha > 0.0 && cfg.dir_epsilon > 0.0 {
        RootNoise::Dirichlet { alpha: cfg.dir_alpha, epsilon: cfg.dir_epsilon }
    } else {
        RootNoise::None
    };
    let mut p = SearchParams::new(
        CpuctStrategy::Constant { c_puct: cfg.c_puct },
        ForcedExploration::None,
        root_noise,
    );
    p.draw_contempt = cfg.draw_contempt;
    p
}

/// Advance one game by one ply: pick chosen move from visit distribution,
/// save the per-ply training sample, play the move, mark complete or
/// schedule re-init.
fn advance_one(
    slot: &mut GameSlot,
    cfg: &ConcurrentSelfPlayConfig,
    stats: &mut StatsAccumulators,
) -> Result<(), String> {
    if !slot.initialised || slot.complete {
        return Ok(());
    }
    let mover_color = slot.game.turn_color;
    let mover_is_white = matches!(mover_color, hive_game::piece::PieceColor::White);

    // ---- Record per-ply MCTS stats (top-1, search depth, root child count).
    // Done before we re-init for the next ply so the search still has the
    // current move's tree state. Mirrors the CNN-side accumulation in
    // hive-game/src/search.rs.
    let top1 = slot.search.root_top1_visit_fraction() as f64;
    stats.top1_sum += top1;
    stats.top1_sum_sq += top1 * top1;
    stats.top1_count += 1;
    let (ds, dss, dc) = slot.search.take_depth_stats();
    stats.depth_sum += ds;
    stats.depth_sum_sq += dss;
    stats.depth_count += dc;
    let moves = slot.search.root_child_count() as f64;
    stats.moves_sum += moves;
    stats.moves_sum_sq += moves * moves;
    stats.moves_count += 1;

    // Re-tokenize the root to recover indexed_moves for the (mover, dest)
    // slot lookup, and to capture the token batch for the training sample.
    let (tokens, indexed) = {
        let mut g = slot.game.clone();
        tokenize::tokenize_and_priors(&mut g)
    };

    let visits = slot.search.get_visit_distribution();
    let mut move_src: Vec<u8> = Vec::with_capacity(visits.len());
    let mut move_dst: Vec<u8> = Vec::with_capacity(visits.len());
    let mut move_probs: Vec<f32> = Vec::with_capacity(visits.len());
    for (mv, prob) in &visits {
        for (enc, imv) in &indexed {
            if imv == mv {
                if let PolicyIndex::DotProduct { src_cell, dst_cell, .. } = *enc {
                    move_src.push(src_cell as u8);
                    move_dst.push(dst_cell as u8);
                    move_probs.push(*prob);
                }
                break;
            }
        }
    }

    let root_q = slot.search.root_value_raw();

    slot.samples.push(PlySample {
        tokens,
        move_src: move_src.clone(),
        move_dst: move_dst.clone(),
        move_probs: move_probs.clone(),
        root_q,
        mover_is_white,
    });

    // Pick chosen move via temperature-aware sampling on the visit dist.
    let chosen_mv = sample_chosen_move(slot, &visits, cfg)?;
    slot.game.play_move(&chosen_mv).map_err(|e| e)?;

    // Decide what's next for this game.
    if slot.game.is_game_over() || slot.game.move_count as u32 >= cfg.max_moves {
        slot.complete = true;
    } else {
        slot.initialised = false; // forces a fresh INIT next round
        slot.sims_for_current_move = 0;
    }
    Ok(())
}

fn sample_chosen_move(
    slot: &mut GameSlot,
    visits: &[(Move, f32)],
    cfg: &ConcurrentSelfPlayConfig,
) -> Result<Move, String> {
    if visits.is_empty() {
        return Ok(Move::pass());
    }
    let ply = slot.game.move_count as u32;
    let t = if cfg.temperature_moves == 0 || ply >= cfg.temperature_moves {
        cfg.temperature_end
    } else {
        let frac = ply as f32 / cfg.temperature_moves.max(1) as f32;
        cfg.temperature_start + frac * (cfg.temperature_end - cfg.temperature_start)
    };

    if t <= 0.0 {
        // Argmax — but uniformly randomise across visit-count ties so
        // an untrained net's roughly-uniform visit distribution doesn't
        // collapse onto the first move in valid_moves() order. Without
        // this, every game opens with the same piece/cell and self-play
        // converges to DrawByRepetition. Once the net is trained the
        // top visit count usually wins outright and the tie-break is a
        // no-op.
        let best_v = visits.iter().map(|(_, p)| *p)
            .fold(f32::NEG_INFINITY, f32::max);
        // Treat anything within 1e-6 of best_v as tied.
        let tied: Vec<usize> = visits.iter().enumerate()
            .filter(|(_, (_, p))| (best_v - *p).abs() < 1e-6)
            .map(|(i, _)| i)
            .collect();
        let pick = if tied.len() == 1 {
            tied[0]
        } else {
            let r = next_unit_f32(&mut slot.rng_state);
            tied[((r * tied.len() as f32) as usize).min(tied.len() - 1)]
        };
        return Ok(visits[pick].0);
    }

    // visits are already normalized; reshape by 1/T then sample.
    let mut weighted: Vec<f32> = visits.iter().map(|(_, p)| p.powf(1.0 / t)).collect();
    let total: f32 = weighted.iter().sum();
    if total <= 0.0 {
        return Ok(visits[0].0);
    }
    for w in &mut weighted { *w /= total; }

    // Deterministic splitmix64-style RNG seeded per slot for reproducibility.
    let r = next_unit_f32(&mut slot.rng_state);
    let mut acc = 0.0f32;
    for (i, w) in weighted.iter().enumerate() {
        acc += *w;
        if r < acc { return Ok(visits[i].0); }
    }
    Ok(visits[visits.len() - 1].0)
}

#[inline]
fn next_unit_f32(state: &mut u64) -> f32 {
    // splitmix64
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z = z ^ (z >> 31);
    (z >> 40) as f32 / (1u64 << 24) as f32
}
