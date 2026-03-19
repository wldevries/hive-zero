/// PyO3 Python bindings for hive_engine.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};

use crate::board::GRID_SIZE;
use crate::board_encoding::{NUM_CHANNELS, RESERVE_SIZE};
use crate::game::{self, Game};
use crate::mcts::search::MctsSearch;
use crate::move_encoding::{self, POLICY_SIZE};
use crate::piece::{Piece, PieceColor, PieceType};

/// Create a 1D numpy array from a slice.
fn make_array1<'py>(py: Python<'py>, data: &[f32]) -> Bound<'py, PyArray1<f32>> {
    let arr = numpy::ndarray::Array1::from(data.to_vec());
    PyArray1::from_owned_array_bound(py, arr)
}

/// Create a 2D numpy array from a flat slice with shape.
fn make_array2<'py>(py: Python<'py>, data: &[f32], rows: usize, cols: usize) -> Bound<'py, PyArray2<f32>> {
    let arr = numpy::ndarray::Array2::from_shape_vec((rows, cols), data.to_vec()).unwrap();
    PyArray2::from_owned_array_bound(py, arr)
}

/// Create a 3D numpy array from a flat slice with shape.
fn make_array3<'py>(py: Python<'py>, data: &[f32], d0: usize, d1: usize, d2: usize) -> Bound<'py, PyArray3<f32>> {
    let arr = numpy::ndarray::Array3::from_shape_vec((d0, d1, d2), data.to_vec()).unwrap();
    PyArray3::from_owned_array_bound(py, arr)
}

/// Rust Game exposed to Python.
#[pyclass(name = "RustGame")]
pub struct PyGame {
    pub game: game::Game,
}

#[pymethods]
impl PyGame {
    #[new]
    fn new() -> Self {
        PyGame { game: game::Game::new() }
    }

    /// Deep copy.
    fn copy(&self) -> PyGame {
        PyGame { game: self.game.clone() }
    }

    /// Game state as string.
    #[getter]
    fn state(&self) -> &str {
        self.game.state.as_str()
    }

    #[getter]
    fn is_game_over(&self) -> bool {
        self.game.is_game_over()
    }

    #[getter]
    fn turn_color(&self) -> &str {
        match self.game.turn_color {
            PieceColor::White => "w",
            PieceColor::Black => "b",
        }
    }

    #[getter]
    fn turn_number(&self) -> u16 {
        self.game.turn_number
    }

    #[getter]
    fn move_count(&self) -> u16 {
        self.game.move_count
    }

    /// Get all valid moves as list of (piece_str, from_pos_or_None, to_pos).
    fn valid_moves(&self) -> Vec<(String, Option<(i8, i8)>, (i8, i8))> {
        self.game.valid_moves().iter().map(|mv| {
            let piece_str = mv.piece.unwrap().to_uhp_string();
            let from = mv.from;
            let to = mv.to.unwrap();
            (piece_str, from, to)
        }).collect()
    }

    /// Play a move.
    #[pyo3(signature = (piece_str, from_pos, to_pos))]
    fn play_move(&mut self, piece_str: &str, from_pos: Option<(i8, i8)>, to_pos: (i8, i8)) {
        let piece = Piece::from_str(piece_str).expect("invalid piece string");
        let mv = match from_pos {
            None => game::Move::placement(piece, to_pos),
            Some(f) => game::Move::movement(piece, f, to_pos),
        };
        self.game.play_move(&mv);
    }

    /// Play a pass.
    fn play_pass(&mut self) {
        self.game.play_pass();
    }

    /// Undo last move.
    fn undo(&mut self) {
        self.game.undo();
    }

    /// Encode board state as numpy arrays.
    /// Returns (board_tensor[23,23,23], reserve_vector[10]).
    fn encode_board<'py>(&self, py: Python<'py>) -> (Bound<'py, PyArray3<f32>>, Bound<'py, PyArray1<f32>>) {
        let mut board_data = vec![0.0f32; NUM_CHANNELS * GRID_SIZE * GRID_SIZE];
        let mut reserve_data = vec![0.0f32; RESERVE_SIZE];
        crate::board_encoding::encode_board(&self.game, &mut board_data, &mut reserve_data);

        let board_tensor = make_array3(py, &board_data, NUM_CHANNELS, GRID_SIZE, GRID_SIZE);
        let reserve_vec = make_array1(py, &reserve_data);
        (board_tensor, reserve_vec)
    }

    /// Get legal move mask and indexed moves.
    fn get_legal_move_mask<'py>(
        &self, py: Python<'py>
    ) -> (Bound<'py, PyArray1<f32>>, Vec<(usize, String, Option<(i8, i8)>, (i8, i8))>) {
        let (mask, indexed_moves) = move_encoding::get_legal_move_mask(&self.game);
        let mask_array = make_array1(py, &mask);
        let moves_list: Vec<_> = indexed_moves.iter().map(|(idx, mv)| {
            let piece_str = mv.piece.unwrap().to_uhp_string();
            (*idx, piece_str, mv.from, mv.to.unwrap())
        }).collect();
        (mask_array, moves_list)
    }

    /// Encode a specific move as policy index. Returns -1 if out of grid.
    #[pyo3(signature = (piece_str, from_pos, to_pos))]
    fn encode_move(&self, piece_str: &str, from_pos: Option<(i8, i8)>, to_pos: (i8, i8)) -> i64 {
        let piece = Piece::from_str(piece_str).expect("invalid piece string");
        match move_encoding::encode_move_checked(piece, from_pos, to_pos) {
            Some(idx) => idx as i64,
            None => -1,
        }
    }

    /// Check if piece is in reserve.
    fn reserve_has(&self, piece_str: &str) -> bool {
        let piece = Piece::from_str(piece_str).expect("invalid piece string");
        self.game.reserve_has(piece)
    }

    /// Get reserve count for a piece type.
    fn reserve_count(&self, color: &str, piece_type: &str) -> u8 {
        let c = match color {
            "w" => PieceColor::White,
            "b" => PieceColor::Black,
            _ => panic!("invalid color"),
        };
        let pt = PieceType::from_char(piece_type.chars().next().unwrap()).expect("invalid piece type");
        self.game.reserve_count(c, pt)
    }

    /// Get turn string like "White[1]".
    fn turn_string(&self) -> String {
        self.game.turn_string()
    }

    /// All (hex, top_piece_str) pairs for occupied positions.
    /// Returns list of ((q, r), piece_str) e.g. [((0, 0), "wQ"), ((-1, 0), "bA1")].
    fn all_top_pieces(&self) -> Vec<((i8, i8), String)> {
        self.game.board.all_top_pieces().iter().map(|(hex, piece)| {
            (*hex, piece.to_uhp_string())
        }).collect()
    }

    /// Full stack at a position (bottom to top) as list of piece strings.
    #[pyo3(signature = (q, r))]
    fn stack_at(&self, q: i8, r: i8) -> Vec<String> {
        let slot = self.game.board.stack_at((q, r));
        slot.iter().map(|p| p.to_uhp_string()).collect()
    }

    /// Heuristic value for unfinished games.
    /// Returns (white_score, black_score) based on queen pressure.
    fn heuristic_value(&self) -> (f32, f32) {
        self.game.heuristic_value()
    }

    /// UHP GameString: "Base;State;Turn;move1;move2;..."
    #[getter]
    fn game_string(&self) -> String {
        self.game.game_string()
    }

    /// Format a move as UHP MoveString in the current game context.
    #[pyo3(signature = (piece_str, from_pos, to_pos))]
    fn format_move_uhp(&self, piece_str: &str, from_pos: Option<(i8, i8)>, to_pos: (i8, i8)) -> String {
        let piece = Piece::from_str(piece_str).expect("invalid piece string");
        let mv = match from_pos {
            None => game::Move::placement(piece, to_pos),
            Some(f) => game::Move::movement(piece, f, to_pos),
        };
        self.game.format_move_uhp(&mv)
    }
}

/// Rust MCTS search exposed to Python.
#[pyclass(name = "RustMCTS")]
pub struct PyMCTS {
    search: MctsSearch,
    c_puct: f32,
    batch_size: usize,
}

#[pymethods]
impl PyMCTS {
    #[new]
    #[pyo3(signature = (c_puct=1.5, batch_size=8, capacity=100000))]
    fn new(c_puct: f32, batch_size: usize, capacity: usize) -> Self {
        PyMCTS {
            search: MctsSearch::new(capacity),
            c_puct,
            batch_size,
        }
    }

    /// Run full MCTS search with callback for NN evaluation.
    fn search(
        &mut self,
        py: Python<'_>,
        game: &PyGame,
        eval_fn: &Bound<'_, PyAny>,
        max_simulations: usize,
    ) -> Option<(String, Option<(i8, i8)>, (i8, i8))> {
        self.search.c_puct = self.c_puct;

        let initial_policy = self.eval_single(py, &game.game, eval_fn);
        self.search.init(&game.game, &initial_policy);

        let root_children = self.search.arena.get(self.search.root).child_count;
        if root_children == 0 {
            return None;
        }

        let mut sims_done = 0;
        while sims_done < max_simulations {
            let batch = std::cmp::min(self.batch_size, max_simulations - sims_done);
            let leaves = self.search.select_leaves(batch);

            if !leaves.is_empty() {
                let (policies, values) = self.eval_batch(py, &leaves, eval_fn);
                self.search.expand_and_backprop(&leaves, &policies, &values);
            }

            sims_done += batch;
        }

        self.search.best_move().map(|mv| {
            let piece_str = mv.piece.unwrap().to_uhp_string();
            (piece_str, mv.from, mv.to.unwrap())
        })
    }

    /// Run MCTS and return policy distribution.
    fn get_policy(
        &mut self,
        py: Python<'_>,
        game: &PyGame,
        eval_fn: &Bound<'_, PyAny>,
        max_simulations: usize,
        temperature: f32,
    ) -> (Vec<(String, Option<(i8, i8)>, (i8, i8))>, Vec<f32>) {
        self.search.c_puct = self.c_puct;

        let initial_policy = self.eval_single(py, &game.game, eval_fn);
        self.search.init(&game.game, &initial_policy);

        let root_children = self.search.arena.get(self.search.root).child_count;
        if root_children == 0 {
            return (Vec::new(), Vec::new());
        }

        let mut sims_done = 0;
        while sims_done < max_simulations {
            let batch = std::cmp::min(self.batch_size, max_simulations - sims_done);
            let leaves = self.search.select_leaves(batch);

            if !leaves.is_empty() {
                let (policies, values) = self.eval_batch(py, &leaves, eval_fn);
                self.search.expand_and_backprop(&leaves, &policies, &values);
            }

            sims_done += batch;
        }

        let dist = self.search.get_visit_distribution();

        let moves: Vec<_> = dist.iter().map(|(mv, _)| {
            match mv.piece {
                Some(p) => (p.to_uhp_string(), mv.from, mv.to.unwrap()),
                None => ("pass".to_string(), None, (0, 0)),
            }
        }).collect();

        let visit_counts: Vec<f32> = dist.iter().map(|(_, v)| *v).collect();

        let probs = if temperature == 0.0 {
            let mut p = vec![0.0f32; visit_counts.len()];
            if let Some(max_idx) = visit_counts.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
            {
                p[max_idx] = 1.0;
            }
            p
        } else {
            let powered: Vec<f32> = visit_counts.iter().map(|&v| v.powf(1.0 / temperature)).collect();
            let total: f32 = powered.iter().sum();
            if total > 0.0 {
                powered.iter().map(|v| v / total).collect()
            } else {
                let n = powered.len() as f32;
                vec![1.0 / n; powered.len()]
            }
        };

        (moves, probs)
    }
    // --- Low-level API for cross-game batched evaluation ---

    /// Initialize MCTS tree for a game position with a pre-computed initial policy.
    fn init_search(&mut self, game: &PyGame, initial_policy: PyReadonlyArray1<f32>) {
        self.search.c_puct = self.c_puct;
        let policy = initial_policy.as_slice().unwrap();
        self.search.init(&game.game, policy);
    }

    /// Number of children at root (0 = no encodable valid moves).
    fn root_child_count(&self) -> u16 {
        self.search.arena.get(self.search.root).child_count
    }

    /// Select leaf nodes needing NN eval. Terminal/duplicate nodes are handled
    /// internally. Returns list of leaf node IDs.
    #[pyo3(signature = (batch_size))]
    fn select_leaves_batch(&mut self, batch_size: usize) -> Vec<u32> {
        self.search.select_leaves(batch_size)
    }

    /// Encode leaf game states as numpy arrays for NN evaluation.
    /// Returns (boards[N, C*H*W], reserves[N, R]).
    fn encode_leaves<'py>(
        &self, py: Python<'py>, leaf_ids: Vec<u32>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
        let n = leaf_ids.len();
        let board_size = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;
        let mut all_boards = vec![0.0f32; n * board_size];
        let mut all_reserves = vec![0.0f32; n * RESERVE_SIZE];

        for (i, &leaf) in leaf_ids.iter().enumerate() {
            let game_state = &self.search.arena.get(leaf).game;
            crate::board_encoding::encode_board(
                game_state,
                &mut all_boards[i * board_size..(i + 1) * board_size],
                &mut all_reserves[i * RESERVE_SIZE..(i + 1) * RESERVE_SIZE],
            );
        }

        (make_array2(py, &all_boards, n, board_size),
         make_array2(py, &all_reserves, n, RESERVE_SIZE))
    }

    /// Expand leaves with NN output and backpropagate values.
    fn expand_and_backprop_batch(
        &mut self,
        leaf_ids: Vec<u32>,
        policies: PyReadonlyArray2<f32>,
        values: PyReadonlyArray1<f32>,
    ) {
        let policy_data = policies.as_slice().unwrap();
        let value_data = values.as_slice().unwrap();
        let n = leaf_ids.len();

        let policies_vec: Vec<Vec<f32>> = (0..n).map(|i| {
            policy_data[i * POLICY_SIZE..(i + 1) * POLICY_SIZE].to_vec()
        }).collect();
        let values_vec: Vec<f32> = value_data[..n].to_vec();

        self.search.expand_and_backprop(&leaf_ids, &policies_vec, &values_vec);
    }

    /// Get visit count distribution after search completes.
    /// Returns (moves, visit_proportions).
    fn visit_distribution(&self) -> (Vec<(String, Option<(i8, i8)>, (i8, i8))>, Vec<f32>) {
        let dist = self.search.get_visit_distribution();
        let moves: Vec<_> = dist.iter().map(|(mv, _)| {
            match mv.piece {
                Some(p) => (p.to_uhp_string(), mv.from, mv.to.unwrap()),
                None => ("pass".to_string(), None, (0, 0)),
            }
        }).collect();
        let probs: Vec<f32> = dist.iter().map(|(_, v)| *v).collect();
        (moves, probs)
    }
}

impl PyMCTS {
    fn eval_single(&self, py: Python<'_>, game: &game::Game, eval_fn: &Bound<'_, PyAny>) -> Vec<f32> {
        let mut board_data = vec![0.0f32; NUM_CHANNELS * GRID_SIZE * GRID_SIZE];
        let mut reserve_data = vec![0.0f32; RESERVE_SIZE];
        crate::board_encoding::encode_board(game, &mut board_data, &mut reserve_data);

        let board_arr = make_array2(py, &board_data, 1, NUM_CHANNELS * GRID_SIZE * GRID_SIZE);
        let board_4d = board_arr.reshape([1, NUM_CHANNELS, GRID_SIZE, GRID_SIZE]).unwrap();
        let reserve_arr = make_array2(py, &reserve_data, 1, RESERVE_SIZE);

        let result = eval_fn.call1((board_4d, reserve_arr)).expect("eval_fn call failed");
        let tuple = result.downcast::<PyTuple>().expect("eval_fn must return tuple");

        let policy_arr: PyReadonlyArray2<f32> = tuple.get_item(0).unwrap().extract().unwrap();
        policy_arr.as_slice().unwrap().to_vec()
    }

    fn eval_batch(
        &self,
        py: Python<'_>,
        leaves: &[u32],
        eval_fn: &Bound<'_, PyAny>,
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        let n = leaves.len();
        let board_size = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;
        let mut all_boards = vec![0.0f32; n * board_size];
        let mut all_reserves = vec![0.0f32; n * RESERVE_SIZE];

        for (i, &leaf) in leaves.iter().enumerate() {
            let game = &self.search.arena.get(leaf).game;
            crate::board_encoding::encode_board(
                game,
                &mut all_boards[i * board_size..(i + 1) * board_size],
                &mut all_reserves[i * RESERVE_SIZE..(i + 1) * RESERVE_SIZE],
            );
        }

        let board_arr = make_array2(py, &all_boards, n, board_size);
        let board_4d = board_arr.reshape([n, NUM_CHANNELS, GRID_SIZE, GRID_SIZE]).unwrap();
        let reserve_arr = make_array2(py, &all_reserves, n, RESERVE_SIZE);

        let result = eval_fn.call1((board_4d, reserve_arr)).expect("eval_fn call failed");
        let tuple = result.downcast::<PyTuple>().expect("eval_fn must return tuple");

        let policy_arr: PyReadonlyArray2<f32> = tuple.get_item(0).unwrap().extract().unwrap();
        let value_arr: PyReadonlyArray1<f32> = tuple.get_item(1).unwrap().extract().unwrap();

        let policies: Vec<Vec<f32>> = (0..n).map(|i| {
            policy_arr.as_slice().unwrap()[i * POLICY_SIZE..(i + 1) * POLICY_SIZE].to_vec()
        }).collect();
        let values: Vec<f32> = value_arr.as_slice().unwrap().to_vec();

        (policies, values)
    }
}

/// Batch MCTS: owns multiple search trees and parallelizes CPU work with rayon.
#[pyclass(name = "RustBatchMCTS")]
pub struct PyBatchMCTS {
    searches: Vec<MctsSearch>,
    c_puct: f32,
    leaf_batch_size: usize,
    use_forced_playouts: bool,
    last_init_values: Vec<f32>,
}

#[pymethods]
impl PyBatchMCTS {
    #[new]
    #[pyo3(signature = (num_games, c_puct=1.5, leaf_batch_size=16, capacity=100000, use_forced_playouts=false))]
    fn new(num_games: usize, c_puct: f32, leaf_batch_size: usize, capacity: usize, use_forced_playouts: bool) -> Self {
        let searches: Vec<MctsSearch> = (0..num_games)
            .map(|_| {
                let mut s = MctsSearch::new(capacity);
                s.c_puct = c_puct;
                s.use_forced_playouts = use_forced_playouts;
                s
            })
            .collect();
        PyBatchMCTS { searches, c_puct, leaf_batch_size, use_forced_playouts, last_init_values: Vec::new() }
    }

    /// Initialize MCTS trees for all games with pre-computed policies.
    /// policies shape: [num_games, POLICY_SIZE]
    fn init_searches(&mut self, games: Vec<PyRef<PyGame>>, policies: PyReadonlyArray2<f32>) {
        use rayon::prelude::*;

        let policy_data = policies.as_slice().unwrap();
        // Collect game clones first (can't send PyRef across threads)
        let game_clones: Vec<Game> = games.iter().map(|g| g.game.clone()).collect();

        let use_forced = self.use_forced_playouts;
        self.searches.par_iter_mut().enumerate().for_each(|(i, search)| {
            search.c_puct = self.c_puct;
            search.use_forced_playouts = use_forced;
            let policy = &policy_data[i * POLICY_SIZE..(i + 1) * POLICY_SIZE];
            search.init(&game_clones[i], policy);
        });
    }

    /// Get root child counts for all trees.
    fn root_child_counts(&self) -> Vec<u16> {
        self.searches.iter().map(|s| s.arena.get(s.root).child_count).collect()
    }

    /// Select leaves and encode them for all active games in parallel.
    /// active_indices: which game indices to process.
    /// Returns (all_boards [N, board_size], all_reserves [N, reserve_size],
    ///          game_leaf_counts: list of how many leaves per game).
    fn select_and_encode<'py>(
        &mut self,
        py: Python<'py>,
        active_indices: Vec<usize>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>, Vec<usize>) {
        use rayon::prelude::*;

        let board_size = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;
        let leaf_batch_size = self.leaf_batch_size;
        let searches = &mut self.searches;

        // Collect mutable refs to only the active searches
        // We gather (index_in_active, &mut MctsSearch) pairs
        let mut active_searches: Vec<(usize, &mut MctsSearch)> = Vec::with_capacity(active_indices.len());
        for (pos, &gi) in active_indices.iter().enumerate() {
            // Split borrow: take a raw pointer, rebuild mutable ref.
            // This is safe because active_indices contains unique indices.
            let ptr = &mut searches[gi] as *mut MctsSearch;
            active_searches.push((pos, unsafe { &mut *ptr }));
        }

        // Parallel: select leaves + encode for each active game
        let results: Vec<(Vec<u32>, Vec<f32>, Vec<f32>)> = active_searches
            .par_iter_mut()
            .map(|(_pos, search)| {
                let leaves = search.select_leaves(leaf_batch_size);

                let mut boards = vec![0.0f32; leaves.len() * board_size];
                let mut reserves = vec![0.0f32; leaves.len() * RESERVE_SIZE];

                for (j, &leaf) in leaves.iter().enumerate() {
                    let game_state = &search.arena.get(leaf).game;
                    crate::board_encoding::encode_board(
                        game_state,
                        &mut boards[j * board_size..(j + 1) * board_size],
                        &mut reserves[j * RESERVE_SIZE..(j + 1) * RESERVE_SIZE],
                    );
                }

                // Stash leaf IDs for backprop
                search.stashed_leaves = leaves.clone();

                (leaves, boards, reserves)
            })
            .collect();

        // Flatten boards/reserves into single arrays
        let mut total_leaves = 0usize;
        let mut game_leaf_counts = Vec::with_capacity(active_indices.len());
        for (leaves, _, _) in &results {
            game_leaf_counts.push(leaves.len());
            total_leaves += leaves.len();
        }

        let mut all_boards = vec![0.0f32; total_leaves * board_size];
        let mut all_reserves = vec![0.0f32; total_leaves * RESERVE_SIZE];

        let mut offset = 0;
        for (_, boards, reserves) in &results {
            let n = boards.len() / board_size;
            all_boards[offset * board_size..(offset + n) * board_size]
                .copy_from_slice(boards);
            all_reserves[offset * RESERVE_SIZE..(offset + n) * RESERVE_SIZE]
                .copy_from_slice(reserves);
            offset += n;
        }

        (
            make_array2(py, &all_boards, total_leaves, board_size),
            make_array2(py, &all_reserves, total_leaves, RESERVE_SIZE),
            game_leaf_counts,
        )
    }

    /// Expand leaves and backpropagate NN results, in parallel across games.
    /// policies shape: [total_leaves, POLICY_SIZE], values shape: [total_leaves]
    /// active_indices + game_leaf_counts must match what select_and_encode returned.
    fn expand_and_backprop_all(
        &mut self,
        active_indices: Vec<usize>,
        game_leaf_counts: Vec<usize>,
        policies: PyReadonlyArray2<f32>,
        values: PyReadonlyArray1<f32>,
    ) {
        use rayon::prelude::*;

        let policy_data = policies.as_slice().unwrap();
        let value_data = values.as_slice().unwrap();

        // Build per-game offsets
        let mut offsets = Vec::with_capacity(active_indices.len());
        let mut offset = 0usize;
        for &count in &game_leaf_counts {
            offsets.push(offset);
            offset += count;
        }

        let searches = &mut self.searches;

        // Collect mutable refs to active searches
        let mut active_searches: Vec<(usize, &mut MctsSearch)> = Vec::with_capacity(active_indices.len());
        for (pos, &gi) in active_indices.iter().enumerate() {
            let ptr = &mut searches[gi] as *mut MctsSearch;
            active_searches.push((pos, unsafe { &mut *ptr }));
        }

        // Parallel expand + backprop
        active_searches.par_iter_mut().for_each(|(pos, search)| {
            let off = offsets[*pos];
            let n = game_leaf_counts[*pos];
            let leaf_ids = std::mem::take(&mut search.stashed_leaves);

            let policies_vec: Vec<Vec<f32>> = (0..n).map(|j| {
                policy_data[(off + j) * POLICY_SIZE..(off + j + 1) * POLICY_SIZE].to_vec()
            }).collect();
            let values_vec: Vec<f32> = value_data[off..off + n].to_vec();

            search.expand_and_backprop(&leaf_ids, &policies_vec, &values_vec);
        });
    }

    /// Apply Dirichlet noise to root priors for the given game indices.
    /// Call this after init_searches, before run_simulations, during self-play only.
    #[pyo3(signature = (active_indices, alpha=0.3, epsilon=0.25))]
    fn apply_root_dirichlet(&mut self, active_indices: Vec<usize>, alpha: f32, epsilon: f32) {
        for gi in active_indices {
            self.searches[gi].apply_root_dirichlet(alpha, epsilon);
        }
    }

    /// Run simulation loop with per-game simulation caps.
    /// Each game in active_indices has its own cap from per_game_caps.
    /// Games that reach their cap stop producing leaves; others continue.
    /// eval_fn(boards_2d, reserves_2d) -> (policies_2d, values_1d)
    fn run_simulations(
        &mut self,
        py: Python<'_>,
        active_indices: Vec<usize>,
        per_game_caps: Vec<usize>,
        eval_fn: &Bound<'_, PyAny>,
    ) {
        use rayon::prelude::*;

        let board_size = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;
        let leaf_batch_size = self.leaf_batch_size;
        let mut sims_done: Vec<usize> = vec![0; active_indices.len()];
        let mut searching: Vec<usize> = (0..active_indices.len()).collect();

        while !searching.is_empty() {
            // --- CPU (rayon): select leaves + encode ---
            let active_gis: Vec<usize> = searching.iter().map(|&si| active_indices[si]).collect();
            let searches = &mut self.searches;

            let mut active_searches: Vec<&mut MctsSearch> = Vec::with_capacity(active_gis.len());
            for &gi in &active_gis {
                let ptr = &mut searches[gi] as *mut MctsSearch;
                active_searches.push(unsafe { &mut *ptr });
            }

            // Cap per-game batch to remaining simulations.
            let per_game_batch: Vec<usize> = searching.iter()
                .map(|&si| (per_game_caps[si] - sims_done[si]).min(leaf_batch_size).max(1))
                .collect();

            let results: Vec<(Vec<u32>, Vec<f32>, Vec<f32>)> = active_searches
                .par_iter_mut()
                .zip(per_game_batch.par_iter())
                .map(|(search, &batch)| {
                    let leaves = search.select_leaves(batch);
                    let mut boards = vec![0.0f32; leaves.len() * board_size];
                    let mut reserves = vec![0.0f32; leaves.len() * RESERVE_SIZE];
                    for (j, &leaf) in leaves.iter().enumerate() {
                        let game_state = &search.arena.get(leaf).game;
                        crate::board_encoding::encode_board(
                            game_state,
                            &mut boards[j * board_size..(j + 1) * board_size],
                            &mut reserves[j * RESERVE_SIZE..(j + 1) * RESERVE_SIZE],
                        );
                    }
                    search.stashed_leaves = leaves.clone();
                    (leaves, boards, reserves)
                })
                .collect();

            // Track sims by actual unique leaves found, not batch size
            let mut total_leaves = 0usize;
            let mut leaf_counts = Vec::with_capacity(searching.len());
            for (si_idx, &si) in searching.iter().enumerate() {
                let n = results[si_idx].0.len();
                sims_done[si] += n.max(1); // at least 1 to avoid infinite loop
                leaf_counts.push(n);
                total_leaves += n;
            }

            if total_leaves == 0 {
                searching.retain(|&si| sims_done[si] < per_game_caps[si]);
                continue;
            }

            // Flatten into contiguous arrays
            let mut all_boards = vec![0.0f32; total_leaves * board_size];
            let mut all_reserves = vec![0.0f32; total_leaves * RESERVE_SIZE];
            let mut offset = 0;
            for (_, boards, reserves) in &results {
                let n = boards.len() / board_size;
                if n > 0 {
                    all_boards[offset * board_size..(offset + n) * board_size]
                        .copy_from_slice(boards);
                    all_reserves[offset * RESERVE_SIZE..(offset + n) * RESERVE_SIZE]
                        .copy_from_slice(reserves);
                    offset += n;
                }
            }

            // --- GPU: single Python callback ---
            let board_arr = make_array2(py, &all_boards, total_leaves, board_size);
            let board_4d = board_arr.reshape([total_leaves, NUM_CHANNELS, GRID_SIZE, GRID_SIZE]).unwrap();
            let reserve_arr = make_array2(py, &all_reserves, total_leaves, RESERVE_SIZE);

            let result = eval_fn.call1((board_4d, reserve_arr)).expect("eval_fn call failed");
            let tuple = result.downcast::<PyTuple>().expect("eval_fn must return tuple");

            let policy_arr: PyReadonlyArray2<f32> = tuple.get_item(0).unwrap().extract().unwrap();
            let value_arr: PyReadonlyArray1<f32> = tuple.get_item(1).unwrap().extract().unwrap();
            let policy_data = policy_arr.as_slice().unwrap();
            let value_data = value_arr.as_slice().unwrap();

            // --- CPU (rayon): expand + backprop ---
            let mut offsets = Vec::with_capacity(active_gis.len());
            let mut off = 0usize;
            for &count in &leaf_counts {
                offsets.push(off);
                off += count;
            }

            let mut active_searches2: Vec<&mut MctsSearch> = Vec::with_capacity(active_gis.len());
            for &gi in &active_gis {
                let ptr = &mut searches[gi] as *mut MctsSearch;
                active_searches2.push(unsafe { &mut *ptr });
            }

            active_searches2.par_iter_mut().enumerate().for_each(|(i, search)| {
                let off = offsets[i];
                let n = leaf_counts[i];
                let leaf_ids = std::mem::take(&mut search.stashed_leaves);
                let policies_vec: Vec<Vec<f32>> = (0..n).map(|j| {
                    policy_data[(off + j) * POLICY_SIZE..(off + j + 1) * POLICY_SIZE].to_vec()
                }).collect();
                let values_vec: Vec<f32> = value_data[off..off + n].to_vec();
                search.expand_and_backprop(&leaf_ids, &policies_vec, &values_vec);
            });

            searching.retain(|&si| sims_done[si] < per_game_caps[si]);
        }
    }

    /// Combined init + dirichlet + simulate in a single call.
    /// Reduces Python↔Rust round-trips from 5 to 2 per turn.
    /// dirichlet_indices: batch indices (into active_indices) that get Dirichlet noise.
    #[pyo3(signature = (active_indices, games, per_game_caps, dirichlet_indices, eval_fn))]
    fn run_turn(
        &mut self,
        py: Python<'_>,
        active_indices: Vec<usize>,
        games: Vec<PyRef<PyGame>>,
        per_game_caps: Vec<usize>,
        dirichlet_indices: Vec<usize>,
        eval_fn: &Bound<'_, PyAny>,
    ) {
        use rayon::prelude::*;

        let board_size = NUM_CHANNELS * GRID_SIZE * GRID_SIZE;

        // --- Encode all positions (rayon-parallel) ---
        let game_clones: Vec<Game> = games.iter().map(|g| g.game.clone()).collect();
        let num = active_indices.len();

        let encoded: Vec<(Vec<f32>, Vec<f32>)> = game_clones.par_iter().map(|game| {
            let mut board = vec![0.0f32; board_size];
            let mut reserve = vec![0.0f32; RESERVE_SIZE];
            crate::board_encoding::encode_board(game, &mut board, &mut reserve);
            (board, reserve)
        }).collect();

        // Flatten into contiguous arrays for GPU
        let mut all_boards = vec![0.0f32; num * board_size];
        let mut all_reserves = vec![0.0f32; num * RESERVE_SIZE];
        for (i, (board, reserve)) in encoded.iter().enumerate() {
            all_boards[i * board_size..(i + 1) * board_size].copy_from_slice(board);
            all_reserves[i * RESERVE_SIZE..(i + 1) * RESERVE_SIZE].copy_from_slice(reserve);
        }

        // --- GPU: initial policy eval ---
        let board_arr = make_array2(py, &all_boards, num, board_size);
        let board_4d = board_arr.reshape([num, NUM_CHANNELS, GRID_SIZE, GRID_SIZE]).unwrap();
        let reserve_arr = make_array2(py, &all_reserves, num, RESERVE_SIZE);

        let result = eval_fn.call1((board_4d, reserve_arr)).expect("eval_fn call failed");
        let tuple = result.downcast::<PyTuple>().expect("eval_fn must return tuple");
        let policy_arr: PyReadonlyArray2<f32> = tuple.get_item(0).unwrap().extract().unwrap();
        let init_values: PyReadonlyArray1<f32> = tuple.get_item(1).unwrap().extract().unwrap();

        let policy_data = policy_arr.as_slice().unwrap();
        // Store init values for Python to retrieve later
        self.last_init_values = init_values.as_slice().unwrap().to_vec();

        // --- Init MCTS trees (rayon-parallel) ---
        let use_forced = self.use_forced_playouts;
        let c_puct = self.c_puct;
        let searches = &mut self.searches;

        // Only reinit the active searches
        let mut active_searches: Vec<(usize, &mut MctsSearch)> = Vec::with_capacity(num);
        for (pos, &gi) in active_indices.iter().enumerate() {
            let ptr = &mut searches[gi] as *mut MctsSearch;
            active_searches.push((pos, unsafe { &mut *ptr }));
        }

        active_searches.par_iter_mut().for_each(|(pos, search)| {
            search.c_puct = c_puct;
            search.use_forced_playouts = use_forced;
            let policy = &policy_data[*pos * POLICY_SIZE..(*pos + 1) * POLICY_SIZE];
            search.init(&game_clones[*pos], policy);
        });

        // --- Apply Dirichlet noise to full-search games ---
        for &bi in &dirichlet_indices {
            let gi = active_indices[bi];
            self.searches[gi].apply_root_dirichlet(0.3, 0.25);
        }

        // --- Run simulations with per-game caps ---
        self.run_simulations(py, active_indices, per_game_caps, eval_fn);
    }

    /// Get the initial values from the last run_turn call.
    /// Returns values in the same order as the active_indices passed to run_turn.
    fn last_init_values(&self) -> Vec<f32> {
        self.last_init_values.clone()
    }

    /// Get visit distributions for specified games.
    /// Uses pruned distributions when forced playouts are enabled.
    fn visit_distributions(&self, game_indices: Vec<usize>)
        -> Vec<(Vec<(String, Option<(i8, i8)>, (i8, i8))>, Vec<f32>)>
    {
        game_indices.iter().map(|&gi| {
            let dist = if self.use_forced_playouts {
                self.searches[gi].get_pruned_visit_distribution()
            } else {
                self.searches[gi].get_visit_distribution()
            };
            let moves: Vec<_> = dist.iter().map(|(mv, _)| {
                match mv.piece {
                    Some(p) => (p.to_uhp_string(), mv.from, mv.to.unwrap()),
                    None => ("pass".to_string(), None, (0, 0)),
                }
            }).collect();
            let probs: Vec<f32> = dist.iter().map(|(_, v)| *v).collect();
            (moves, probs)
        }).collect()
    }
}

/// Register Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_class::<PyMCTS>()?;
    m.add_class::<PyBatchMCTS>()?;
    Ok(())
}
