/// PyO3 Python bindings for Zertz self-play. v2

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;

use zertz_game::board_encoding::{encode_board, GRID_SIZE, NUM_CHANNELS};
use zertz_game::hex::{is_valid, Hex};
use zertz_game::mcts::arena::NodeId;
use zertz_game::mcts::search::MctsSearch;
use zertz_game::move_encoding::{encode_move, POLICY_SIZE};
use zertz_game::zertz::{Marble, ZertzBoard, ZertzMove, MAX_CAPTURE_JUMPS};
use core_game::game::{Game, Outcome, Player};

const BOARD_FLAT: usize = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

#[pyclass(name = "ZertzSelfPlayResult")]
pub struct PyZertzSelfPlayResult {
    board_data: Vec<f32>,
    policy_data: Vec<f32>,
    value_targets: Vec<f32>,
    weights: Vec<f32>,
    value_only_flags: Vec<bool>,
    num_samples: usize,
    wins_p1: u32,
    wins_p2: u32,
    draws: u32,
    total_moves: u32,
    game_lengths: Vec<u32>,
    decisive_lengths: Vec<u32>,
    full_search_turns: u32,
    total_turns: u32,
    /// Up to 3 sample boards: (label, board_string) pairs for display.
    sample_board_data: Vec<(String, String)>,
}

#[pymethods]
impl PyZertzSelfPlayResult {
    /// Returns (boards[N,C,H,W], policies[N,POLICY_SIZE], values[N], weights[N], value_only[N])
    fn training_data<'py>(&self, py: Python<'py>) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Vec<bool>,
    ) {
        let n = self.num_samples;
        let boards = numpy::ndarray::Array2::from_shape_vec(
            (n, BOARD_FLAT), self.board_data.clone(),
        ).unwrap();
        let policies = numpy::ndarray::Array2::from_shape_vec(
            (n, POLICY_SIZE), self.policy_data.clone(),
        ).unwrap();
        let values = numpy::ndarray::Array1::from(self.value_targets.clone());
        let weights = numpy::ndarray::Array1::from(self.weights.clone());
        (
            PyArray2::from_owned_array_bound(py, boards),
            PyArray2::from_owned_array_bound(py, policies),
            PyArray1::from_owned_array_bound(py, values),
            PyArray1::from_owned_array_bound(py, weights),
            self.value_only_flags.clone(),
        )
    }

    #[getter] fn num_samples(&self) -> usize { self.num_samples }
    #[getter] fn wins_p1(&self) -> u32 { self.wins_p1 }
    #[getter] fn wins_p2(&self) -> u32 { self.wins_p2 }
    #[getter] fn draws(&self) -> u32 { self.draws }
    #[getter] fn total_moves(&self) -> u32 { self.total_moves }
    #[getter] fn game_lengths(&self) -> Vec<u32> { self.game_lengths.clone() }
    #[getter] fn decisive_lengths(&self) -> Vec<u32> { self.decisive_lengths.clone() }
    #[getter] fn full_search_turns(&self) -> u32 { self.full_search_turns }
    #[getter] fn total_turns(&self) -> u32 { self.total_turns }
    /// Returns list of (label, board_string) for up to 3 decisive games.
    fn sample_boards(&self) -> Vec<(String, String)> { self.sample_board_data.clone() }
}

// ---------------------------------------------------------------------------
// Per-turn record
// ---------------------------------------------------------------------------

struct TurnRecord {
    board_offset: usize,
    player: Player,
    is_value_only: bool,
    policy_vector: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

#[pyclass(name = "ZertzSelfPlaySession")]
pub struct PyZertzSelfPlaySession {
    num_games: usize,
    simulations: usize,
    max_moves: u32,
    temperature: f32,
    temp_threshold: u32,
    c_puct: f32,
    play_batch_size: usize,
    playout_cap_p: f32,
    fast_cap: usize,
}

#[pymethods]
impl PyZertzSelfPlaySession {
    #[new]
    #[pyo3(signature = (
        num_games,
        simulations = 100,
        max_moves = 200,
        temperature = 1.0,
        temp_threshold = 15,
        c_puct = 1.5,
        play_batch_size = 2,
        playout_cap_p = 0.0,
        fast_cap = 20,
    ))]
    fn new(
        num_games: usize,
        simulations: usize,
        max_moves: u32,
        temperature: f32,
        temp_threshold: u32,
        c_puct: f32,
        play_batch_size: usize,
        playout_cap_p: f32,
        fast_cap: usize,
    ) -> Self {
        PyZertzSelfPlaySession {
            num_games, simulations, max_moves, temperature, temp_threshold,
            c_puct, play_batch_size, playout_cap_p, fast_cap,
        }
    }

    /// Play all games to completion.
    /// eval_fn(boards[N,C,H,W]) -> (policy[N,P], value[N])
    /// progress_fn(finished, total, active, total_moves) called after each turn.
    #[pyo3(signature = (eval_fn, progress_fn=None))]
    fn play_games(
        &self,
        py: Python<'_>,
        eval_fn: &Bound<'_, PyAny>,
        progress_fn: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyZertzSelfPlayResult> {
        let num_games = self.num_games;
        let use_playout_cap = self.playout_cap_p > 0.0;

        let mut boards: Vec<ZertzBoard> = (0..num_games).map(|_| ZertzBoard::default()).collect();
        // Size arena to just cover the simulations needed. With lazy child creation
        // at most ~simulations nodes are ever live per game per turn.
        let arena_capacity = self.simulations + 64;
        let mut searches: Vec<MctsSearch> = (0..num_games).map(|_| {
            let mut s = MctsSearch::new(arena_capacity);
            s.c_puct = self.c_puct;
            s
        }).collect();
        let mut move_counts: Vec<u32> = vec![0; num_games];
        let mut active: Vec<bool> = vec![true; num_games];
        let mut finished_count: u32 = 0;

        let mut histories: Vec<Vec<TurnRecord>> = (0..num_games).map(|_| Vec::new()).collect();
        let mut board_buf: Vec<f32> = Vec::new();

        let mut rng = rand::thread_rng();

        // Stats
        let mut wins_p1 = 0u32;
        let mut wins_p2 = 0u32;
        let mut draws = 0u32;
        let mut total_moves = 0u32;
        let mut game_lengths: Vec<u32> = Vec::new();
        let mut decisive_lengths: Vec<u32> = Vec::new();
        let mut full_search_turns: u32 = 0;
        let mut total_turns: u32 = 0;
        let mut sample_board_data: Vec<(String, String)> = Vec::new();

        // --- Main game loop ---
        while active.iter().any(|&a| a) {
            let mcts_games: Vec<usize> = (0..num_games).filter(|&gi| active[gi]).collect();
            if mcts_games.is_empty() { break; }

            let n = mcts_games.len();
            total_turns += n as u32;

            // Decide fast vs full search per game
            let is_full: Vec<bool> = if use_playout_cap {
                (0..n).map(|_| rng.gen::<f32>() < self.playout_cap_p).collect()
            } else {
                vec![true; n]
            };
            let sim_caps: Vec<usize> = is_full.iter()
                .map(|&f| if f { self.simulations } else { self.fast_cap })
                .collect();
            full_search_turns += is_full.iter().filter(|&&f| f).count() as u32;

            // Encode current positions
            let mut turn_board_offsets: Vec<usize> = Vec::with_capacity(n);
            let mut flat_boards = vec![0f32; n * BOARD_FLAT];
            for (i, &gi) in mcts_games.iter().enumerate() {
                let off = board_buf.len();
                board_buf.resize(off + BOARD_FLAT, 0.0);
                encode_board(&boards[gi], &mut board_buf[off..off + BOARD_FLAT]);
                encode_board(&boards[gi], &mut flat_boards[i * BOARD_FLAT..(i + 1) * BOARD_FLAT]);
                turn_board_offsets.push(off);
            }

            // Initial NN eval for MCTS root policy
            let board_arr = numpy::ndarray::Array2::from_shape_vec((n, BOARD_FLAT), flat_boards).unwrap();
            let board_np = PyArray2::from_owned_array_bound(py, board_arr);
            let board_4d = board_np.reshape([n, NUM_CHANNELS, GRID_SIZE, GRID_SIZE]).unwrap();

            let result = eval_fn.call1((board_4d,))?;
            let tuple = result.downcast::<PyTuple>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("eval_fn must return (policy, value) tuple")
            })?;
            let policy_arr: PyReadonlyArray2<f32> = tuple.get_item(0)?.extract()?;
            let value_arr: PyReadonlyArray1<f32> = tuple.get_item(1)?.extract()?;
            let init_policies = policy_arr.as_slice()?.to_vec();

            // Init MCTS trees
            for (i, &gi) in mcts_games.iter().enumerate() {
                let policy = &init_policies[i * POLICY_SIZE..(i + 1) * POLICY_SIZE];
                searches[gi].init(&boards[gi], policy);
                if is_full[i] {
                    searches[gi].apply_root_dirichlet(0.3, 0.25);
                }
            }

            // --- Simulation rounds ---
            // play_batch_size = number of selection rounds per NN call (same semantics as Hive).
            // Each round selects 1 leaf per active game, so the actual NN batch is
            // play_batch_size × active_games.
            let mut game_sims: Vec<usize> = vec![0; n];
            loop {
                let mut leaf_ids: Vec<NodeId> = Vec::new();
                let mut leaf_game_idx: Vec<usize> = Vec::new();

                // Run play_batch_size rounds; each round = 1 leaf per active game.
                for _round in 0..self.play_batch_size {
                    let mut any_collected = false;
                    for (i, &gi) in mcts_games.iter().enumerate() {
                        if game_sims[i] >= sim_caps[i] { continue; }
                        let leaves = searches[gi].select_leaves(1);
                        let count = leaves.len();
                        if count > 0 { any_collected = true; }
                        for leaf in leaves {
                            leaf_ids.push(leaf);
                            leaf_game_idx.push(i);
                        }
                        game_sims[i] += count;
                    }
                    if !any_collected { break; }
                }

                if leaf_ids.is_empty() { break; }

                let nl = leaf_ids.len();
                let mut leaf_boards_flat = vec![0f32; nl * BOARD_FLAT];
                for (k, (&leaf, &i)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
                    let gi = mcts_games[i];
                    let encoded = searches[gi].encode_leaf(leaf);
                    leaf_boards_flat[k * BOARD_FLAT..(k + 1) * BOARD_FLAT].copy_from_slice(&encoded);
                }

                let leaf_arr = numpy::ndarray::Array2::from_shape_vec((nl, BOARD_FLAT), leaf_boards_flat).unwrap();
                let leaf_np = PyArray2::from_owned_array_bound(py, leaf_arr);
                let leaf_4d = leaf_np.reshape([nl, NUM_CHANNELS, GRID_SIZE, GRID_SIZE]).unwrap();

                let leaf_result = eval_fn.call1((leaf_4d,))?;
                let leaf_tuple = leaf_result.downcast::<PyTuple>().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err("eval_fn must return (policy, value) tuple")
                })?;
                let lp_arr: PyReadonlyArray2<f32> = leaf_tuple.get_item(0)?.extract()?;
                let lv_arr: PyReadonlyArray1<f32> = leaf_tuple.get_item(1)?.extract()?;
                let leaf_policies = lp_arr.as_slice()?.to_vec();
                let leaf_values = lv_arr.as_slice()?.to_vec();

                let mut per_game_leaves: Vec<Vec<NodeId>> = vec![Vec::new(); n];
                let mut per_game_policies: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n];
                let mut per_game_values: Vec<Vec<f32>> = vec![Vec::new(); n];
                for (k, (&leaf, &i)) in leaf_ids.iter().zip(leaf_game_idx.iter()).enumerate() {
                    per_game_leaves[i].push(leaf);
                    per_game_policies[i].push(leaf_policies[k * POLICY_SIZE..(k + 1) * POLICY_SIZE].to_vec());
                    per_game_values[i].push(leaf_values[k]);
                }
                for (i, &gi) in mcts_games.iter().enumerate() {
                    if per_game_leaves[i].is_empty() { continue; }
                    searches[gi].expand_and_backprop(
                        &per_game_leaves[i],
                        &per_game_policies[i],
                        &per_game_values[i],
                    );
                }

                if game_sims.iter().zip(sim_caps.iter()).all(|(s, c)| *s >= *c) { break; }
            }

            // --- Select and apply moves ---
            for (i, &gi) in mcts_games.iter().enumerate() {
                let dist = searches[gi].get_pruned_visit_distribution();
                let mut policy_vec = vec![0.0f32; POLICY_SIZE];
                for (mv, prob) in &dist {
                    policy_vec[encode_move(mv)] = *prob;
                }

                histories[gi].push(TurnRecord {
                    board_offset: turn_board_offsets[i],
                    player: boards[gi].next_player(),
                    is_value_only: !is_full[i],
                    policy_vector: policy_vec,
                });

                let mv = if dist.is_empty() {
                    ZertzMove::Pass
                } else if move_counts[gi] < self.temp_threshold && self.temperature > 0.01 {
                    let weights: Vec<f32> = dist.iter()
                        .map(|(_, p)| p.powf(1.0 / self.temperature))
                        .collect();
                    let wi = WeightedIndex::new(&weights).unwrap();
                    dist[wi.sample(&mut rng)].0
                } else {
                    dist.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0
                };

                boards[gi].play(mv).expect("MCTS selected illegal move");
                move_counts[gi] += 1;
                total_moves += 1;

                if boards[gi].outcome() != Outcome::Ongoing || move_counts[gi] >= self.max_moves {
                    active[gi] = false;
                    finished_count += 1;
                    let len = move_counts[gi];
                    game_lengths.push(len);
                    match boards[gi].outcome() {
                        Outcome::WonBy(Player::Player1) => {
                            wins_p1 += 1;
                            decisive_lengths.push(len);
                            let label = format!("P1 wins ({} moves)", len);
                            sample_board_data.push((label, format!("{}", boards[gi])));
                        }
                        Outcome::WonBy(Player::Player2) => {
                            wins_p2 += 1;
                            decisive_lengths.push(len);
                            let label = format!("P2 wins ({} moves)", len);
                            sample_board_data.push((label, format!("{}", boards[gi])));
                        }
                        _ => { draws += 1; }
                    }
                }
            }

            // Progress callback
            if let Some(pfn) = progress_fn {
                let active_count = active.iter().filter(|&&a| a).count() as u32;
                pfn.call1((finished_count, num_games as u32, active_count, total_moves)).ok();
            }
        }

        // --- Build training data ---
        let total_samples: usize = histories.iter().map(|h| h.len()).sum();
        let mut board_data = Vec::with_capacity(total_samples * BOARD_FLAT);
        let mut policy_data = Vec::with_capacity(total_samples * POLICY_SIZE);
        let mut value_targets = Vec::with_capacity(total_samples);
        let mut weights = Vec::with_capacity(total_samples);
        let mut value_only_flags = Vec::with_capacity(total_samples);

        for (gi, history) in histories.iter().enumerate() {
            let outcome = boards[gi].outcome();
            for record in history {
                board_data.extend_from_slice(
                    &board_buf[record.board_offset..record.board_offset + BOARD_FLAT],
                );
                policy_data.extend_from_slice(&record.policy_vector);
                let value = match outcome {
                    Outcome::WonBy(winner) => {
                        if winner == record.player { 1.0f32 } else { -1.0f32 }
                    }
                    _ => 0.0f32,
                };
                value_targets.push(value);
                weights.push(1.0f32);
                value_only_flags.push(record.is_value_only);
            }
        }

        Ok(PyZertzSelfPlayResult {
            board_data,
            policy_data,
            value_targets,
            weights,
            value_only_flags,
            num_samples: total_samples,
            wins_p1,
            wins_p2,
            draws,
            total_moves,
            game_lengths,
            decisive_lengths,
            full_search_turns,
            total_turns,
            sample_board_data,
        })
    }
}

// ---------------------------------------------------------------------------
// Move string utilities  (A-G columns, 1-7 rows, matching board display)
// ---------------------------------------------------------------------------

fn hex_to_coord(h: Hex) -> String {
    let (q, r) = h;
    let col = (b'A' + (q + 3) as u8) as char;
    let row = 4 - r;  // r=-3→7, r=0→4, r=3→1
    format!("{}{}", col, row)
}

fn coord_to_hex(s: &str) -> Result<Hex, String> {
    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return Err(format!("Expected cell like D4, got '{}'", s));
    }
    let col_b = bytes[0].to_ascii_uppercase();
    let row_b = bytes[1];
    if !(b'A'..=b'G').contains(&col_b) {
        return Err(format!("Column must be A-G, got '{}'", col_b as char));
    }
    if !(b'1'..=b'7').contains(&row_b) {
        return Err(format!("Row must be 1-7, got '{}'", row_b as char));
    }
    let q = col_b as i8 - b'A' as i8 - 3;
    let r = 4 - (row_b - b'0') as i8;
    let h = (q, r);
    if !is_valid(h) {
        return Err(format!("'{}' is not on the Zertz board", s));
    }
    Ok(h)
}

pub fn move_to_str(mv: ZertzMove) -> String {
    match mv {
        ZertzMove::Place { color, place_at, remove } =>
            format!("{} {} {}", color, hex_to_coord(place_at), hex_to_coord(remove)),
        ZertzMove::PlaceOnly { color, place_at } =>
            format!("{} {}", color, hex_to_coord(place_at)),
        ZertzMove::Capture { jumps, len } => {
            let mut s = format!("CAP {}", hex_to_coord(jumps[0].0));
            for i in 0..len as usize {
                s.push(' ');
                s.push_str(&hex_to_coord(jumps[i].2));
            }
            s
        }
        ZertzMove::Pass => "pass".to_string(),
    }
}

fn str_to_move(s: &str) -> Result<ZertzMove, String> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.is_empty() {
        return Err("Empty move string".to_string());
    }
    let first = parts[0].to_ascii_uppercase();
    match first.as_str() {
        "CAP" => {
            if parts.len() < 3 {
                return Err("CAP needs at least from + one landing: CAP D4 D6".to_string());
            }
            let positions: Result<Vec<Hex>, _> = parts[1..].iter().map(|s| coord_to_hex(s)).collect();
            let positions = positions?;
            let n_hops = positions.len() - 1;
            if n_hops > MAX_CAPTURE_JUMPS {
                return Err(format!("Too many hops (max {})", MAX_CAPTURE_JUMPS));
            }
            let mut jumps = [((0i8, 0i8), (0i8, 0i8), (0i8, 0i8)); MAX_CAPTURE_JUMPS];
            for i in 0..n_hops {
                let (fq, fr) = positions[i];
                let (tq, tr) = positions[i + 1];
                let dq = tq - fq;
                let dr = tr - fr;
                if dq % 2 != 0 || dr % 2 != 0 {
                    return Err(format!(
                        "Hop from {} to {} is not a valid 2-step jump",
                        hex_to_coord(positions[i]), hex_to_coord(positions[i + 1])
                    ));
                }
                jumps[i] = (positions[i], (fq + dq / 2, fr + dr / 2), positions[i + 1]);
            }
            Ok(ZertzMove::Capture { jumps, len: n_hops as u8 })
        }
        "PASS" => Ok(ZertzMove::Pass),
        "W" | "G" | "B" => {
            let color = match first.as_str() {
                "W" => Marble::White,
                "G" => Marble::Grey,
                _   => Marble::Black,
            };
            match parts.len() {
                2 => Ok(ZertzMove::PlaceOnly { color, place_at: coord_to_hex(parts[1])? }),
                3 => Ok(ZertzMove::Place { color, place_at: coord_to_hex(parts[1])?, remove: coord_to_hex(parts[2])? }),
                _ => Err(format!("Expected 'W/G/B place [remove]', got '{}'", s)),
            }
        }
        _ => Err(format!("Unknown move format '{}': expected CAP/W/G/B/pass", s)),
    }
}

// ---------------------------------------------------------------------------
// ZertzGame — interactive game state exposed to Python
// ---------------------------------------------------------------------------

#[pyclass(name = "ZertzGame")]
struct PyZertzGame {
    board: ZertzBoard,
}

#[pymethods]
impl PyZertzGame {
    #[new]
    fn new() -> Self {
        PyZertzGame { board: ZertzBoard::default() }
    }

    fn valid_moves(&self) -> Vec<String> {
        self.board.legal_moves().into_iter().map(move_to_str).collect()
    }

    fn play(&mut self, move_str: &str) -> PyResult<()> {
        let mv = str_to_move(move_str)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        self.board.play(mv)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn board_str(&self) -> String {
        format!("{}", self.board)
    }

    fn outcome(&self) -> &str {
        match self.board.outcome() {
            Outcome::Ongoing              => "ongoing",
            Outcome::WonBy(Player::Player1) => "p1",
            Outcome::WonBy(Player::Player2) => "p2",
            Outcome::Draw                 => "draw",
        }
    }

    fn next_player(&self) -> u8 {
        match self.board.next_player() {
            Player::Player1 => 0,
            Player::Player2 => 1,
        }
    }

    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let mut buf = vec![0f32; BOARD_FLAT];
        encode_board(&self.board, &mut buf);
        PyArray1::from_vec_bound(py, buf)
    }

    /// Run MCTS for `simulations` sims and return the best move string.
    /// eval_fn(boards[N, C, H, W]) -> (policy[N, POLICY_SIZE], value[N])
    #[pyo3(signature = (eval_fn, simulations=200, c_puct=1.5))]
    fn best_move(
        &self,
        py: Python<'_>,
        eval_fn: &Bound<'_, PyAny>,
        simulations: usize,
        c_puct: f32,
    ) -> PyResult<String> {
        if self.board.outcome() != Outcome::Ongoing {
            return Err(pyo3::exceptions::PyValueError::new_err("Game is already over"));
        }

        let mut search = MctsSearch::new(simulations + 64);
        search.c_puct = c_puct;

        // Initial NN eval on root position
        let mut buf = vec![0f32; BOARD_FLAT];
        encode_board(&self.board, &mut buf);
        let root_arr = numpy::ndarray::Array2::from_shape_vec((1, BOARD_FLAT), buf).unwrap();
        let root_np = PyArray2::from_owned_array_bound(py, root_arr);
        let root_4d = root_np.reshape([1, NUM_CHANNELS, GRID_SIZE, GRID_SIZE]).unwrap();

        let result = eval_fn.call1((root_4d,))?;
        let tuple = result.downcast::<PyTuple>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("eval_fn must return (policy, value) tuple")
        })?;
        let policy_arr: PyReadonlyArray2<f32> = tuple.get_item(0)?.extract()?;
        let root_policy = policy_arr.as_slice()?.to_vec();

        search.init(&self.board, &root_policy);
        search.apply_root_dirichlet(0.3, 0.25);

        // Simulation rounds (batch_size=8)
        let batch = 8usize;
        let mut done = 0usize;
        while done < simulations {
            let leaves = search.select_leaves(batch.min(simulations - done));
            if leaves.is_empty() { break; }
            let nl = leaves.len();

            let mut flat = vec![0f32; nl * BOARD_FLAT];
            for (k, &leaf) in leaves.iter().enumerate() {
                let enc = search.encode_leaf(leaf);
                flat[k * BOARD_FLAT..(k + 1) * BOARD_FLAT].copy_from_slice(&enc);
            }

            let leaf_arr = numpy::ndarray::Array2::from_shape_vec((nl, BOARD_FLAT), flat).unwrap();
            let leaf_np = PyArray2::from_owned_array_bound(py, leaf_arr);
            let leaf_4d = leaf_np.reshape([nl, NUM_CHANNELS, GRID_SIZE, GRID_SIZE]).unwrap();

            let res = eval_fn.call1((leaf_4d,))?;
            let tup = res.downcast::<PyTuple>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("eval_fn must return (policy, value) tuple")
            })?;
            let lp: PyReadonlyArray2<f32> = tup.get_item(0)?.extract()?;
            let lv: PyReadonlyArray1<f32> = tup.get_item(1)?.extract()?;

            let leaf_policies: Vec<Vec<f32>> = (0..nl)
                .map(|k| lp.as_slice().unwrap()[k * POLICY_SIZE..(k + 1) * POLICY_SIZE].to_vec())
                .collect();
            let leaf_values: Vec<f32> = lv.as_slice()?.to_vec();

            search.expand_and_backprop(&leaves, &leaf_policies, &leaf_values);
            done += nl;
        }

        let dist = search.get_pruned_visit_distribution();
        let best = dist.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(mv, _)| move_to_str(*mv))
            .unwrap_or_else(|| "pass".to_string());
        Ok(best)
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyZertzSelfPlaySession>()?;
    m.add_class::<PyZertzSelfPlayResult>()?;
    m.add_class::<PyZertzGame>()?;
    m.add("ZERTZ_POLICY_SIZE", POLICY_SIZE)?;
    m.add("ZERTZ_NUM_CHANNELS", NUM_CHANNELS)?;
    m.add("ZERTZ_GRID_SIZE", GRID_SIZE)?;
    Ok(())
}
