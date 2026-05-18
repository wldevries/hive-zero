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

use core_game::game::PolicyIndex;
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
    /// Number of sims completed for the *current* move.
    sims_for_current_move: usize,
    /// Whether the search has been initialised at the current root.
    initialised: bool,
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
/// `progress` is invoked once per inference round if provided.
pub fn play_games_concurrent_ort(
    cfg: &ConcurrentSelfPlayConfig,
    engine: &mut HiveTokenOrtEngine,
    mut progress: Option<ProgressFn>,
) -> Result<Vec<ConcurrentGameResult>, String> {
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
            samples: Vec::new(),
            rng_state: cfg.rng_seed.wrapping_add(i as u64 * 0x9E37_79B9_7F4A_7C15),
            complete: false,
        })
        .collect();

    let search_params = build_params(cfg);

    let mut round_idx: usize = 0;
    loop {
        // ---- Compute active games and per-slot intent ------------------
        // Each non-complete game is either:
        //   (a) needs INIT (initialised == false) — its root needs an
        //       inference call to populate the policy_prior;
        //   (b) needs SIMS (sims_for_current_move < simulations) — it
        //       contributes `play_batch` leaves to this round;
        //   (c) ready to ADVANCE (sims done) — it picks a move and
        //       transitions to INIT (or complete).
        //
        // We do ADVANCE before the inference batch so the post-advance
        // INIT can ride along in this round's batch.
        for slot in slots.iter_mut() {
            if slot.complete || !slot.initialised { continue; }
            if slot.sims_for_current_move < cfg.simulations { continue; }
            advance_one(slot, cfg)?;
        }

        // Gather inference batches.
        // First the INIT batches (one token batch per uninitialised game),
        // then the SIM batches (`play_batch` per active game).
        let mut batches: Vec<TokenBatch> = Vec::new();
        // `routing` maps batch index → (game_idx, RoutingKind).
        let mut routing: Vec<(usize, RoutingKind)> = Vec::new();

        for (gi, slot) in slots.iter_mut().enumerate() {
            if slot.complete { continue; }
            if !slot.initialised {
                let mut root_game = slot.game.clone();
                let (tb, _) = tokenize::tokenize_and_priors(&mut root_game);
                batches.push(tb);
                routing.push((gi, RoutingKind::Init));
            }
        }

        // SIM batches require INIT to be done first. For uninitialised
        // games we defer SIMs to the next round (so the policy_prior is
        // available before any leaves get expanded). This costs one round
        // per ply per game but keeps the loop logic flat — fine because
        // INIT and SIM rounds overlap across games anyway.
        for (gi, slot) in slots.iter_mut().enumerate() {
            if slot.complete || !slot.initialised { continue; }
            if slot.sims_for_current_move >= cfg.simulations { continue; }
            let want = cfg.play_batch.min(cfg.simulations - slot.sims_for_current_move);
            let leaves = slot.search.select_leaves(want);
            // Note: sims_for_current_move credits `want` regardless of
            // `leaves.len()`. select_leaves can return fewer than `want`
            // leaves when some selections hit terminal nodes inside the
            // tree — those terminals are still real simulations (their
            // value gets backpropped internally), they just don't need NN
            // evaluation. Crediting `want` here keeps the per-move sim
            // budget honest and prevents "stuck" slots when terminal-only
            // batches occur.
            slot.sims_for_current_move += want;
            if leaves.is_empty() { continue; }
            for leaf in &leaves {
                let mut g = slot.search.stashed_game(*leaf)
                    .ok_or_else(|| "stashed leaf missing".to_string())?
                    .clone();
                let (tb, _) = tokenize::tokenize_and_priors(&mut g);
                batches.push(tb);
            }
            routing.push((gi, RoutingKind::Sim { count: leaves.len() }));
        }

        if batches.is_empty() {
            // No INIT or SIM batches to dispatch this round. But this DOESN'T
            // necessarily mean we're done: a slot may have just crossed the
            // sims_done >= simulations threshold during this round's SIM
            // phase (e.g. via empty leaves from select_leaves on the final
            // partial batch — every selection in that batch hit a terminal
            // node, so leaves.len() = 0 but sims_for_current_move still
            // incremented by `want`). Those slots are READY TO ADVANCE in
            // the NEXT iter — we must let the loop continue rather than
            // terminate here.
            let any_active = slots.iter().any(|s| !s.complete);
            if !any_active {
                break;
            }
            // Otherwise fall through to round_idx++ and continue. The next
            // iter's ADVANCE phase will pick up any slot with sims_done >=
            // simulations, play its move, and re-INIT for the next ply —
            // which WILL contribute a batch.
            round_idx += 1;
            if let Some(cb) = progress.as_mut() {
                let snapshot: Vec<(u32, usize)> = slots.iter()
                    .filter(|s| !s.complete)
                    .map(|s| (s.game.move_count as u32, s.sims_for_current_move))
                    .collect();
                let active = snapshot.len();
                cb(round_idx, active, 0, &snapshot);
            }
            continue;
        }

        // ---- One big inference call across all games -------------------
        let (policies, values, draws) = engine
            .infer_token_batches(&batches)
            .map_err(|e| e.to_string())?;

        // ---- Distribute results back to each game ---------------------
        let mut cursor: usize = 0;
        for (gi, kind) in &routing {
            let slot = &mut slots[*gi];
            match kind {
                RoutingKind::Init => {
                    // policies[cursor] is the root policy for this game.
                    slot.search.init(&slot.game, &policies[cursor]);
                    if let RootNoise::Dirichlet { alpha, epsilon } = search_params.root_noise {
                        slot.search.params = search_params.clone();
                        slot.search.apply_root_dirichlet(alpha, epsilon);
                    } else {
                        slot.search.params = search_params.clone();
                    }
                    slot.initialised = true;
                    slot.sims_for_current_move = 0;
                    cursor += 1;
                }
                RoutingKind::Sim { count } => {
                    let pol_slice = &policies[cursor..cursor + count];
                    let val_slice = &values[cursor..cursor + count];
                    let draw_slice = &draws[cursor..cursor + count];
                    slot.search.expand_and_backprop(pol_slice, val_slice, draw_slice);
                    cursor += count;
                }
            }
        }

        round_idx += 1;
        if let Some(cb) = progress.as_mut() {
            let snapshot: Vec<(u32, usize)> = slots.iter()
                .filter(|s| !s.complete)
                .map(|s| (s.game.move_count as u32, s.sims_for_current_move))
                .collect();
            let active = snapshot.len();
            cb(round_idx, active, batches.len(), &snapshot);
        }
    }

    // ---- Assemble final results -----------------------------------------
    let mut out = Vec::with_capacity(slots.len());
    for slot in slots {
        out.push(ConcurrentGameResult {
            outcome: slot.game.state.as_str().to_string(),
            move_count: slot.game.move_count as u32,
            final_uhp: slot.game.game_string(),
            final_board_render: slot.game.board.render(None, None),
            samples: slot.samples,
        });
    }
    Ok(out)
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
fn advance_one(slot: &mut GameSlot, cfg: &ConcurrentSelfPlayConfig) -> Result<(), String> {
    if !slot.initialised || slot.complete {
        return Ok(());
    }
    let mover_color = slot.game.turn_color;
    let mover_is_white = matches!(mover_color, hive_game::piece::PieceColor::White);

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
        // Argmax.
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, (_, p)) in visits.iter().enumerate() {
            if *p > best_v { best_v = *p; best_i = i; }
        }
        return Ok(visits[best_i].0);
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
