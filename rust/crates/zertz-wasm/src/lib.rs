//! WebAssembly bindings for the Zertz game engine.
//!
//! # Building
//! ```
//! wasm-pack build rust/crates/zertz-wasm --target web --out-dir web/pkg
//! ```
//!
//! # JS inference callback contract (for best_move)
//! `eval_fn(boards: Float32Array, reserves: Float32Array, n: number)`
//! must return `[place: Float32Array, cap_source: Float32Array, cap_dest: Float32Array, value: Float32Array]`
//! with lengths `n * PLACE_HEAD_SIZE`, `n * CAP_HEAD_SIZE`, `n * CAP_HEAD_SIZE`, `n`.

use js_sys::{Array, Float32Array, Function};
use wasm_bindgen::prelude::*;

use core_game::game::{Game, Outcome, Player};
use zertz_game::board_encoding::{encode_board, GRID_SIZE, NUM_CHANNELS, RESERVE_SIZE};
use zertz_game::hex::{all_hexes, BOARD_SIZE};
use zertz_game::mcts::search::{CAP_HEAD_SIZE, MctsSearch, PLACE_HEAD_SIZE, POLICY_HEADS_TOTAL, PolicyHeads};
use zertz_game::notation::{move_to_str, str_to_move};
use zertz_game::search::best_move_core;
use zertz_game::zertz::{Ring, ZertzBoard, ZertzMove};

const BOARD_FLAT: usize = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;
const MAX_ROLLOUT_MOVES: usize = 200;

// ---------------------------------------------------------------------------
// Constants exposed to JS
// ---------------------------------------------------------------------------

#[wasm_bindgen] pub fn board_flat_size() -> u32 { BOARD_FLAT as u32 }
#[wasm_bindgen] pub fn reserve_size()     -> u32 { RESERVE_SIZE as u32 }
#[wasm_bindgen] pub fn place_head_size()  -> u32 { PLACE_HEAD_SIZE as u32 }
#[wasm_bindgen] pub fn cap_head_size()    -> u32 { CAP_HEAD_SIZE as u32 }
#[wasm_bindgen] pub fn board_cell_count() -> u32 { BOARD_SIZE as u32 }

/// Returns flat [(q0, r0), (q1, r1), ...] for the 37 board cells (i8 pairs).
#[wasm_bindgen]
pub fn hex_coords() -> Vec<i8> {
    all_hexes().iter().flat_map(|&(q, r)| [q, r]).collect()
}

// ---------------------------------------------------------------------------
// NN eval callback (for best_move)
// ---------------------------------------------------------------------------

struct JsEvalFn(Function);
/// SAFETY: WASM is single-threaded.
unsafe impl Send for JsEvalFn {}
unsafe impl Sync for JsEvalFn {}

fn call_js_eval(
    eval_fn: &Function,
    boards: &[f32],
    reserves: &[f32],
    n: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let boards_js = Float32Array::from(boards);
    let reserves_js = Float32Array::from(reserves);
    let n_js = JsValue::from(n as u32);
    let result = eval_fn
        .call3(&JsValue::NULL, &boards_js, &reserves_js, &n_js)
        .map_err(|e| format!("eval_fn threw: {:?}", e))?;
    let arr = Array::from(&result);
    Ok((
        Float32Array::from(arr.get(0)).to_vec(),
        Float32Array::from(arr.get(1)).to_vec(),
        Float32Array::from(arr.get(2)).to_vec(),
        Float32Array::from(arr.get(3)).to_vec(),
    ))
}

// ---------------------------------------------------------------------------
// Random rollout
// ---------------------------------------------------------------------------

fn random_rollout_value(board: &ZertzBoard, rollouts: usize, rng: &mut impl rand::Rng) -> f32 {
    use rand::seq::IndexedRandom;
    let perspective = board.next_player();
    let mut total = 0.0f32;
    for _ in 0..rollouts {
        let mut b = board.clone();
        let mut moves_played = 0;
        let value = loop {
            match b.outcome() {
                Outcome::WonBy(w) => break if w == perspective { 1.0f32 } else { -1.0 },
                Outcome::Draw => break 0.0,
                Outcome::Ongoing => {
                    if moves_played >= MAX_ROLLOUT_MOVES { break 0.0; }
                    let moves = b.legal_moves();
                    if moves.is_empty() { break 0.0; }
                    let mv = *moves.choose(rng).unwrap();
                    let _ = b.play(mv);
                    moves_played += 1;
                }
            }
        };
        total += value;
    }
    total / rollouts.max(1) as f32
}

// ---------------------------------------------------------------------------
// ZertzGame
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct ZertzGame {
    board: ZertzBoard,
}

#[wasm_bindgen]
impl ZertzGame {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ZertzGame {
        ZertzGame { board: ZertzBoard::default() }
    }

    /// All legal moves as a JS Array of strings.
    pub fn valid_moves(&self) -> Array {
        self.board.legal_moves().into_iter()
            .map(|mv| JsValue::from_str(&move_to_str(mv)))
            .collect()
    }

    /// Apply a move string. Throws on error.
    pub fn play(&mut self, move_str: &str) -> Result<(), JsValue> {
        let mv = str_to_move(move_str).map_err(|e| JsValue::from_str(&e))?;
        self.board.play(mv).map_err(|e| JsValue::from_str(&e))
    }

    /// ASCII board string for debugging.
    pub fn board_str(&self) -> String { format!("{}", self.board) }

    /// "ongoing" | "p1" | "p2" | "draw"
    pub fn outcome(&self) -> String {
        match self.board.outcome() {
            Outcome::Ongoing => "ongoing".into(),
            Outcome::WonBy(Player::Player1) => "p1".into(),
            Outcome::WonBy(Player::Player2) => "p2".into(),
            Outcome::Draw => "draw".into(),
        }
    }

    /// 0 = Player1, 1 = Player2.
    pub fn next_player(&self) -> u8 {
        match self.board.next_player() { Player::Player1 => 0, Player::Player2 => 1 }
    }

    /// Cell state for each of the 37 board positions (same order as hex_coords()).
    ///   0 = ring removed
    ///   1 = ring empty
    ///   2 = white marble
    ///   3 = grey marble
    ///   4 = black marble
    pub fn cell_states(&self) -> Vec<u8> {
        self.board.rings().iter().map(|r| match r {
            Ring::Removed        => 0,
            Ring::Empty          => 1,
            Ring::Occupied(m)    => 2 + m.index() as u8,
        }).collect()
    }

    /// Shared marble supply remaining: [white, grey, black].
    pub fn supply_counts(&self) -> Vec<u8> { self.board.supply().to_vec() }

    /// Captured marbles: [p1_white, p1_grey, p1_black, p2_white, p2_grey, p2_black].
    pub fn capture_counts(&self) -> Vec<u8> {
        let c = self.board.captures();
        vec![c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2]]
    }

    /// True when we're in the middle of a capture chain (same player must continue capturing).
    pub fn is_mid_capture(&self) -> bool { self.board.is_mid_capture() }

    /// Encode the current position: [board: Float32Array, reserve: Float32Array].
    pub fn encode(&self) -> Array {
        let mut board_buf = vec![0f32; BOARD_FLAT];
        let mut reserve_buf = vec![0f32; RESERVE_SIZE];
        encode_board(&self.board, &mut board_buf, &mut reserve_buf);
        Array::of2(
            &Float32Array::from(board_buf.as_slice()),
            &Float32Array::from(reserve_buf.as_slice()),
        )
    }

    /// NN-guided MCTS. eval_fn receives (boards, reserves, n) and returns
    /// [place, cap_source, cap_dest, value] as Float32Arrays.
    /// Run this in a Web Worker to avoid blocking the main thread.
    pub fn best_move(
        &self,
        eval_fn: &Function,
        simulations: usize,
        c_puct: f32,
    ) -> Result<String, JsValue> {
        let wrapped = JsEvalFn(eval_fn.clone());
        let core_eval: zertz_game::search::EvalFn =
            Box::new(move |boards: &[f32], reserves: &[f32], n: usize| {
                call_js_eval(&wrapped.0, boards, reserves, n)
            });
        let mv = best_move_core(&self.board, simulations, c_puct, core_eval)
            .map_err(|e| JsValue::from_str(&e))?;
        Ok(move_to_str(mv))
    }

    /// Random-rollout MCTS — no neural network needed.
    /// Uses uniform policy priors and random game completions for leaf values.
    /// `rollouts_per_leaf`: random games played per leaf (1 is fast, 3–5 is stronger).
    /// Run this in a Web Worker to avoid blocking the main thread.
    pub fn best_move_random(
        &self,
        simulations: usize,
        rollouts_per_leaf: usize,
        c_puct: f32,
    ) -> Result<String, JsValue> {
        if self.board.outcome() != Outcome::Ongoing {
            return Err(JsValue::from_str("Game is already over"));
        }

        let uniform = vec![0.0f32; POLICY_HEADS_TOTAL];
        let root_heads = PolicyHeads {
            place:      &uniform[..PLACE_HEAD_SIZE],
            cap_source: &uniform[PLACE_HEAD_SIZE..PLACE_HEAD_SIZE + CAP_HEAD_SIZE],
            cap_dest:   &uniform[PLACE_HEAD_SIZE + CAP_HEAD_SIZE..],
        };

        let mut search = MctsSearch::new(simulations + 128);
        search.c_puct = c_puct;
        search.init(&self.board, &root_heads);

        let mut rng = rand::rng();
        let batch = 8usize;
        let mut done = 0;

        while done < simulations {
            let n = batch.min(simulations - done);
            let leaves = search.select_leaves(n);
            if leaves.is_empty() { break; }
            let nl = leaves.len();

            let heads_list: Vec<PolicyHeads> = leaves.iter().map(|_| PolicyHeads {
                place:      &uniform[..PLACE_HEAD_SIZE],
                cap_source: &uniform[PLACE_HEAD_SIZE..PLACE_HEAD_SIZE + CAP_HEAD_SIZE],
                cap_dest:   &uniform[PLACE_HEAD_SIZE + CAP_HEAD_SIZE..],
            }).collect();

            let values: Vec<f32> = leaves.iter().map(|&leaf| {
                let board = search.get_leaf_board(leaf);
                random_rollout_value(board, rollouts_per_leaf, &mut rng)
            }).collect();

            search.expand_and_backprop(&leaves, &heads_list, &values);
            done += nl;
        }

        let dist = search.get_pruned_visit_distribution();
        let best = dist.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(mv, _)| *mv)
            .unwrap_or(ZertzMove::Pass);

        Ok(move_to_str(best))
    }
}
