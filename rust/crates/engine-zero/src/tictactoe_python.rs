/// PyO3 Python bindings for Tic-Tac-Toe self-play.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use rand::RngExt;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;

use tictactoe_game::game::{
    TicTacToe, GRID_SIZE, CHANNELS_PER_STEP, POLICY_SIZE,
};
use core_game::game::{Game, NNGame, Outcome, Player};
use core_game::mcts::search::{CpuctStrategy, ForcedExploration, MctsSearch, RootNoise, SearchParams};

/// Board flat size for a given history length.
fn board_flat(history_length: usize) -> usize {
    CHANNELS_PER_STEP * history_length * GRID_SIZE * GRID_SIZE
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

#[pyclass(name = "TTTSelfPlayResult")]
pub struct PyTTTSelfPlayResult {
    board_data: Vec<f32>,
    policy_data: Vec<f32>,
    value_targets: Vec<f32>,
    value_only_flags: Vec<bool>,
    num_samples: usize,
    board_flat: usize,
    #[allow(dead_code)]
    board_channels: usize,
    wins_p1: u32,
    wins_p2: u32,
    draws: u32,
    game_lengths: Vec<u32>,
    full_search_turns: u32,
    total_turns: u32,
    /// Final board states: Vec of (board[9], outcome) where outcome is 0=draw, 1=X wins, 2=O wins.
    final_boards: Vec<(Vec<u8>, u8)>,
}

#[pymethods]
impl PyTTTSelfPlayResult {
    /// Returns (boards, policies, values, value_only)
    fn training_data<'py>(&self, py: Python<'py>) -> (
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Vec<bool>,
    ) {
        let n = self.num_samples;
        let bf = self.board_flat;
        let boards = numpy::ndarray::Array2::from_shape_vec(
            (n, bf), self.board_data.clone(),
        ).unwrap();
        let policies = numpy::ndarray::Array2::from_shape_vec(
            (n, POLICY_SIZE), self.policy_data.clone(),
        ).unwrap();
        let values = numpy::ndarray::Array1::from(self.value_targets.clone());
        (
            PyArray2::from_owned_array(py, boards),
            PyArray2::from_owned_array(py, policies),
            PyArray1::from_owned_array(py, values),
            self.value_only_flags.clone(),
        )
    }

    #[getter] fn num_samples(&self) -> usize { self.num_samples }
    #[getter] fn wins_p1(&self) -> u32 { self.wins_p1 }
    #[getter] fn wins_p2(&self) -> u32 { self.wins_p2 }
    #[getter] fn draws(&self) -> u32 { self.draws }
    #[getter] fn game_lengths(&self) -> Vec<u32> { self.game_lengths.clone() }
    #[getter] fn full_search_turns(&self) -> u32 { self.full_search_turns }
    #[getter] fn total_turns(&self) -> u32 { self.total_turns }

    /// Returns list of (board[9], outcome) for all finished games.
    fn final_boards(&self) -> Vec<(Vec<u8>, u8)> { self.final_boards.clone() }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

#[pyclass(name = "TTTSelfPlaySession")]
pub struct PyTTTSelfPlaySession {
    num_games: usize,
    simulations: usize,
    max_moves: u32,
    temperature: f32,
    temp_threshold: u32,
    #[allow(dead_code)]
    c_puct: f32,
    dir_alpha: f32,
    dir_epsilon: f32,
    playout_cap_p: f32,
    fast_cap: usize,
    history_length: usize,
}

#[pymethods]
impl PyTTTSelfPlaySession {
    #[new]
    #[pyo3(signature = (
        num_games,
        simulations = 100,
        max_moves = 9,
        temperature = 1.0,
        temp_threshold = 5,
        c_puct = 1.5,
        dir_alpha = 0.3,
        dir_epsilon = 0.25,
        playout_cap_p = 0.0,
        fast_cap = 20,
        history_length = 1,
    ))]
    fn new(
        num_games: usize,
        simulations: usize,
        max_moves: u32,
        temperature: f32,
        temp_threshold: u32,
        c_puct: f32,
        dir_alpha: f32,
        dir_epsilon: f32,
        playout_cap_p: f32,
        fast_cap: usize,
        history_length: usize,
    ) -> Self {
        PyTTTSelfPlaySession {
            num_games, simulations, max_moves, temperature, temp_threshold,
            c_puct, dir_alpha, dir_epsilon, playout_cap_p, fast_cap,
            history_length,
        }
    }

    #[pyo3(signature = (eval_fn, progress_fn=None))]
    fn play_games(
        &self,
        py: Python<'_>,
        eval_fn: &Bound<'_, PyAny>,
        progress_fn: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyTTTSelfPlayResult> {
        let num_games = self.num_games;
        let use_playout_cap = self.playout_cap_p > 0.0;
        let history_length = self.history_length;
        let bf = board_flat(history_length);
        let num_ch = CHANNELS_PER_STEP * history_length;
        let search_params = SearchParams::new(
            CpuctStrategy::Constant { c_puct: self.c_puct },
            ForcedExploration::None,
            RootNoise::Dirichlet { alpha: self.dir_alpha, epsilon: self.dir_epsilon },
        );

        let mut games: Vec<TicTacToe> = (0..num_games)
            .map(|_| TicTacToe::with_history(history_length))
            .collect();
        let mut searches: Vec<MctsSearch<TicTacToe>> = (0..num_games).map(|_| {
            let mut s = MctsSearch::<TicTacToe>::new(4096);
            s.params = search_params.clone();
            s
        }).collect();
        let mut move_counts: Vec<u32> = vec![0; num_games];
        let mut active: Vec<bool> = vec![true; num_games];
        let mut finished_count: u32 = 0;

        // Per-game training history
        struct TurnRecord {
            board_offset: usize,
            player: Player,
            is_value_only: bool,
            policy_vector: Vec<f32>,
        }
        let mut histories: Vec<Vec<TurnRecord>> = (0..num_games).map(|_| Vec::new()).collect();
        let mut board_buf: Vec<f32> = Vec::new();

        let mut rng = rand::rng();

        // Stats
        let mut wins_p1 = 0u32;
        let mut wins_p2 = 0u32;
        let mut draws = 0u32;
        let mut game_lengths: Vec<u32> = Vec::new();
        let mut full_search_turns: u32 = 0;
        let mut total_turns: u32 = 0;
        let mut final_boards: Vec<(Vec<u8>, u8)> = Vec::new();

        // --- Main game loop ---
        while active.iter().any(|&a| a) {
            let mcts_games: Vec<usize> = (0..num_games).filter(|&gi| active[gi]).collect();
            if mcts_games.is_empty() { break; }

            let n = mcts_games.len();
            total_turns += n as u32;

            // Decide fast vs full search per game
            let is_full: Vec<bool> = if use_playout_cap {
                (0..n).map(|_| rng.random::<f32>() < self.playout_cap_p).collect()
            } else {
                vec![true; n]
            };
            let sim_caps: Vec<usize> = is_full.iter()
                .map(|&f| if f { self.simulations } else { self.fast_cap })
                .collect();
            full_search_turns += is_full.iter().filter(|&&f| f).count() as u32;

            // Encode current positions
            let mut turn_board_offsets: Vec<usize> = Vec::with_capacity(n);
            let mut flat_boards = vec![0f32; n * bf];
            for (i, &gi) in mcts_games.iter().enumerate() {
                let boff = board_buf.len();
                board_buf.resize(boff + bf, 0.0);
                games[gi].encode_board(&mut board_buf[boff..boff + bf], &mut []);
                flat_boards[i * bf..(i + 1) * bf]
                    .copy_from_slice(&board_buf[boff..boff + bf]);
                turn_board_offsets.push(boff);
            }

            // Initial NN eval for root policy
            let (init_policies, _init_values) = infer_batch(py, eval_fn, &flat_boards, n, num_ch)?;

            // Init MCTS trees
            for (i, &gi) in mcts_games.iter().enumerate() {
                let policy = &init_policies[i * POLICY_SIZE..(i + 1) * POLICY_SIZE];
                searches[gi].init(&games[gi], policy);
                if is_full[i] {
                    searches[gi].apply_root_dirichlet(self.dir_alpha, self.dir_epsilon);
                }
            }

            // --- Simulation loop ---
            let mut game_sims: Vec<usize> = vec![0; n];
            loop {
                let mut leaf_data: Vec<(usize, Vec<f32>)> = Vec::new(); // (index in mcts_games, board encoding)
                let mut leaf_node_map: Vec<(usize, core_game::mcts::arena::NodeId)> = Vec::new();

                for (i, &gi) in mcts_games.iter().enumerate() {
                    if game_sims[i] >= sim_caps[i] { continue; }
                    let leaves = searches[gi].select_leaves(1);
                    let count = leaves.len();
                    for &leaf in &leaves {
                        let (board_enc, _reserve) = searches[gi].encode_leaf(leaf);
                        leaf_data.push((i, board_enc));
                        leaf_node_map.push((i, leaf));
                    }
                    game_sims[i] += count.max(1);
                }

                if leaf_data.is_empty() { break; }

                let nl = leaf_data.len();
                let mut leaf_boards_flat = vec![0f32; nl * bf];
                for (k, (_, board_enc)) in leaf_data.iter().enumerate() {
                    leaf_boards_flat[k * bf..(k + 1) * bf]
                        .copy_from_slice(board_enc);
                }

                let (leaf_policies_flat, leaf_values) = infer_batch(py, eval_fn, &leaf_boards_flat, nl, num_ch)?;

                // Group leaves by game and expand
                let mut per_game_policies: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n];
                let mut per_game_values: Vec<Vec<f32>> = vec![Vec::new(); n];
                for (k, &(i, _leaf)) in leaf_node_map.iter().enumerate() {
                    let policy = leaf_policies_flat[k * POLICY_SIZE..(k + 1) * POLICY_SIZE].to_vec();
                    per_game_policies[i].push(policy);
                    per_game_values[i].push(leaf_values[k]);
                }
                for (i, &gi) in mcts_games.iter().enumerate() {
                    if per_game_policies[i].is_empty() { continue; }
                    searches[gi].expand_and_backprop(
                        &per_game_policies[i],
                        &per_game_values[i],
                    );
                }

                if game_sims.iter().zip(sim_caps.iter()).all(|(s, c)| *s >= *c) { break; }
            }

            // --- Select and apply moves ---
            for (i, &gi) in mcts_games.iter().enumerate() {
                let search = &searches[gi];
                let dist = match search.params.forced_exploration {
                    ForcedExploration::Soft { .. } => search.get_pruned_visit_distribution(),
                    _ => search.get_visit_distribution(),
                };

                let mut policy_vec = vec![0.0f32; POLICY_SIZE];
                for &(mv, prob) in &dist {
                    if !TicTacToe::is_pass(&mv) {
                        policy_vec[mv.cell()] = prob;
                    }
                }

                histories[gi].push(TurnRecord {
                    board_offset: turn_board_offsets[i],
                    player: games[gi].next_player(),
                    is_value_only: !is_full[i],
                    policy_vector: policy_vec,
                });

                // Select move
                let mv = if dist.is_empty() {
                    TicTacToe::pass_move()
                } else if move_counts[gi] < self.temp_threshold && self.temperature > 0.01 {
                    // Temperature sampling
                    let mut probs: Vec<f32> = dist.iter().map(|(_, p)| p.powf(1.0 / self.temperature)).collect();
                    let total: f32 = probs.iter().sum();
                    if total > 0.0 {
                        for p in probs.iter_mut() { *p /= total; }
                    }
                    let wi = WeightedIndex::new(&probs).unwrap();
                    dist[wi.sample(&mut rng)].0
                } else {
                    // Argmax
                    dist.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0
                };

                games[gi].play_move(&mv).unwrap();
                move_counts[gi] += 1;

                if games[gi].is_game_over() || move_counts[gi] >= self.max_moves {
                    active[gi] = false;
                    finished_count += 1;
                }
            }

            // Progress callback
            if let Some(progress_fn) = progress_fn {
                let active_count = active.iter().filter(|&&a| a).count();
                let total_move_count: u32 = move_counts.iter().sum();
                progress_fn.call1((
                    finished_count,
                    num_games as u32,
                    active_count as u32,
                    total_move_count,
                ))?;
            }
        }

        // --- Build training data ---
        let mut result_board_data: Vec<f32> = Vec::new();
        let mut result_policy_data: Vec<f32> = Vec::new();
        let mut result_value_targets: Vec<f32> = Vec::new();
        let mut result_value_only: Vec<bool> = Vec::new();
        let mut num_samples = 0usize;

        for gi in 0..num_games {
            let outcome = games[gi].outcome();
            let game_len = move_counts[gi];
            game_lengths.push(game_len);

            match outcome {
                Outcome::WonBy(Player::Player1) => wins_p1 += 1,
                Outcome::WonBy(Player::Player2) => wins_p2 += 1,
                _ => draws += 1,
            }

            // Store final board state
            {
                use tictactoe_game::game::Cell;
                let board_cells: Vec<u8> = games[gi].board.iter().map(|c| match c {
                    Cell::Empty => 0,
                    Cell::X => 1,
                    Cell::O => 2,
                }).collect();
                let outcome_code = match outcome {
                    Outcome::Draw | Outcome::Ongoing => 0,
                    Outcome::WonBy(Player::Player1) => 1,
                    Outcome::WonBy(Player::Player2) => 2,
                };
                final_boards.push((board_cells, outcome_code));
            }

            let value_for_p1 = match outcome {
                Outcome::WonBy(Player::Player1) => 1.0f32,
                Outcome::WonBy(Player::Player2) => -1.0f32,
                _ => 0.0f32,
            };

            for record in &histories[gi] {
                let value = if record.player == Player::Player1 {
                    value_for_p1
                } else {
                    -value_for_p1
                };

                result_board_data.extend_from_slice(
                    &board_buf[record.board_offset..record.board_offset + bf]
                );
                result_policy_data.extend_from_slice(&record.policy_vector);
                result_value_targets.push(value);
                result_value_only.push(record.is_value_only);
                num_samples += 1;
            }
        }

        Ok(PyTTTSelfPlayResult {
            board_data: result_board_data,
            policy_data: result_policy_data,
            value_targets: result_value_targets,
            value_only_flags: result_value_only,
            num_samples,
            board_flat: bf,
            board_channels: num_ch,
            wins_p1,
            wins_p2,
            draws,
            game_lengths,
            full_search_turns,
            total_turns,
            final_boards,
        })
    }
}

/// Call Python eval_fn to get (policy, value) for a batch of boards.
fn infer_batch(
    py: Python<'_>,
    eval_fn: &Bound<'_, PyAny>,
    boards_flat: &[f32],
    batch_size: usize,
    num_channels: usize,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let bf = num_channels * GRID_SIZE * GRID_SIZE;
    let board_arr = numpy::ndarray::Array2::from_shape_vec(
        (batch_size, bf), boards_flat.to_vec(),
    ).unwrap();
    let board_np = PyArray2::from_owned_array(py, board_arr);
    let board_4d = board_np.reshape([batch_size, num_channels, GRID_SIZE, GRID_SIZE])?;

    let result = eval_fn.call1((board_4d,))?;
    let tuple = result.cast::<PyTuple>().map_err(|_|
        pyo3::exceptions::PyRuntimeError::new_err("eval_fn must return (policy, value) tuple")
    )?;
    let policy: PyReadonlyArray2<f32> = tuple.get_item(0)?.extract()?;
    let value: PyReadonlyArray1<f32> = tuple.get_item(1)?.extract()?;
    Ok((
        policy.as_slice().unwrap().to_vec(),
        value.as_slice().unwrap().to_vec(),
    ))
}

// ---------------------------------------------------------------------------
// Interactive play helper
// ---------------------------------------------------------------------------

#[pyclass(name = "TTTGame")]
pub struct PyTTTGame {
    game: TicTacToe,
}

#[pymethods]
impl PyTTTGame {
    #[new]
    #[pyo3(signature = (history_length = 1))]
    fn new(history_length: usize) -> Self {
        PyTTTGame { game: TicTacToe::with_history(history_length) }
    }

    /// Play a move at the given cell index (0-8).
    fn play_move(&mut self, cell: u8) -> PyResult<()> {
        use tictactoe_game::game::TTTMove;
        self.game.play_move(&TTTMove(cell))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Get the board as a list of 9 ints (0=empty, 1=X, 2=O).
    fn board(&self) -> Vec<u8> {
        use tictactoe_game::game::Cell;
        self.game.board.iter().map(|c| match c {
            Cell::Empty => 0,
            Cell::X => 1,
            Cell::O => 2,
        }).collect()
    }

    /// Current player: 1 (X) or 2 (O).
    fn current_player(&self) -> u8 {
        match self.game.next_player() {
            Player::Player1 => 1,
            Player::Player2 => 2,
        }
    }

    /// Game outcome: None if ongoing, 0 for draw, 1 for X wins, 2 for O wins.
    fn outcome(&self) -> Option<u8> {
        match self.game.outcome() {
            Outcome::Ongoing => None,
            Outcome::Draw => Some(0),
            Outcome::WonBy(Player::Player1) => Some(1),
            Outcome::WonBy(Player::Player2) => Some(2),
        }
    }

    /// Run MCTS and return (best_move_cell, value_estimate).
    fn best_move(
        &mut self,
        py: Python<'_>,
        eval_fn: &Bound<'_, PyAny>,
        simulations: usize,
        c_puct: f32,
    ) -> PyResult<(u8, f32)> {
        if self.game.is_game_over() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Game is already over"));
        }

        let num_ch = self.game.num_channels();
        let bf = board_flat(self.game.history_length);
        let search_params = SearchParams::new(CpuctStrategy::Constant { c_puct }, ForcedExploration::None, RootNoise::None);

        let mut search = MctsSearch::<TicTacToe>::new(4096);
        search.params = search_params;

        // Initial eval
        let mut board_enc = vec![0.0f32; bf];
        self.game.encode_board(&mut board_enc, &mut []);
        let (init_policy, _init_value) = infer_batch(py, eval_fn, &board_enc, 1, num_ch)?;
        search.init(&self.game, &init_policy);

        // Run simulations
        let mut sims_done = 0usize;
        while sims_done < simulations {
            let leaves = search.select_leaves(1);
            let count = leaves.len();
            if count == 0 {
                sims_done += 1;
                continue;
            }

            let mut leaf_boards = vec![0f32; count * bf];
            for (k, &leaf) in leaves.iter().enumerate() {
                let (enc, _) = search.encode_leaf(leaf);
                leaf_boards[k * bf..(k + 1) * bf].copy_from_slice(&enc);
            }

            let (policies_flat, values) = infer_batch(py, eval_fn, &leaf_boards, count, num_ch)?;
            let policies: Vec<Vec<f32>> = (0..count)
                .map(|i| policies_flat[i * POLICY_SIZE..(i + 1) * POLICY_SIZE].to_vec())
                .collect();
            search.expand_and_backprop(&policies, &values);
            sims_done += count;
        }

        let root_value = search.root_value();
        let best = search.best_move()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No legal moves"))?;
        Ok((best.0, root_value))
    }

    fn render(&self) -> String {
        self.game.render()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTTTSelfPlaySession>()?;
    m.add_class::<PyTTTSelfPlayResult>()?;
    m.add_class::<PyTTTGame>()?;
    Ok(())
}
