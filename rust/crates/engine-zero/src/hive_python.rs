/// PyO3 Python bindings for engine_zero.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};

use hive_game::board_encoding::{NUM_CHANNELS, RESERVE_SIZE};
use hive_game::game::{self, Game};
use core_game::mcts::search::MctsSearch;
use hive_game::move_encoding;
use hive_game::piece::{Piece, PieceColor, PieceType};

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
#[pyclass(name = "HiveGame")]
pub struct PyGame {
    pub game: game::Game,
}

#[pymethods]
impl PyGame {
    #[new]
    #[pyo3(signature = (tournament_mode=false, grid_size=23))]
    fn new(tournament_mode: bool, grid_size: usize) -> PyResult<Self> {
        if grid_size % 2 == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("grid_size must be odd, got {grid_size}")));
        }
        if grid_size > hive_game::board::GRID_SIZE {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("grid_size {grid_size} exceeds max board size {}", hive_game::board::GRID_SIZE)));
        }
        let game = if tournament_mode {
            game::Game::new_tournament_with_grid_size(grid_size)
        } else {
            game::Game::new_with_grid_size(grid_size)
        };
        Ok(PyGame { game })
    }

    /// Deep copy.
    fn copy(&self) -> PyGame {
        PyGame { game: self.game.clone() }
    }

    #[getter]
    fn tournament_mode(&self) -> bool {
        self.game.tournament_mode
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
    fn valid_moves(&mut self) -> Vec<(String, Option<(i8, i8)>, (i8, i8))> {
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
        self.game.play_move(&mv).unwrap();
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
    /// Returns (board_tensor[C,H,W], reserve_vector[10]).
    fn encode_board<'py>(&self, py: Python<'py>) -> (Bound<'py, PyArray3<f32>>, Bound<'py, PyArray1<f32>>) {
        let gs = self.game.nn_grid_size;
        let mut board_data = vec![0.0f32; NUM_CHANNELS * gs * gs];
        let mut reserve_data = vec![0.0f32; RESERVE_SIZE];
        hive_game::board_encoding::encode_board(&self.game, &mut board_data, &mut reserve_data, gs);

        let board_tensor = make_array3(py, &board_data, NUM_CHANNELS, gs, gs);
        let reserve_vec = make_array1(py, &reserve_data);
        (board_tensor, reserve_vec)
    }

    /// Get the NN encoding grid size.
    #[getter]
    fn grid_size(&self) -> usize {
        self.game.nn_grid_size
    }

    /// Get legal move mask and indexed moves.
    /// Returns (mask, moves) where each move has:
    ///   (primary_idx, secondary_idx_or_none, piece_str, from_pos, to_pos)
    /// Placements: secondary = None (Single index).
    /// Movements:  secondary = Some(dst_idx) (Sum of src + dst indices).
    fn get_legal_move_mask<'py>(
        &mut self, py: Python<'py>
    ) -> (Bound<'py, PyArray1<f32>>, Vec<(usize, Option<usize>, String, Option<(i8, i8)>, (i8, i8))>) {
        use core_game::game::PolicyIndex;
        let gs = self.game.nn_grid_size;
        let (mask, indexed_moves) = move_encoding::get_legal_move_mask(&mut self.game, gs);
        let mask_array = make_array1(py, &mask);
        let moves_list: Vec<_> = indexed_moves.iter().map(|(enc, mv)| {
            let piece_str = mv.piece.unwrap().to_uhp_string();
            let (primary, secondary) = match *enc {
                PolicyIndex::Single(idx) => (idx, None),
                PolicyIndex::Sum(a, b) => (a, Some(b)),
            };
            (primary, secondary, piece_str, mv.from, mv.to.unwrap())
        }).collect();
        (mask_array, moves_list)
    }

    /// Encode a placement move as flat policy index. Returns -1 if out of grid.
    /// For movement encoding use encode_movement instead.
    #[pyo3(signature = (piece_str, to_pos))]
    fn encode_placement(&self, piece_str: &str, to_pos: (i8, i8)) -> i64 {
        let piece = Piece::from_str(piece_str).expect("invalid piece string");
        match move_encoding::encode_placement_flat(piece, to_pos, self.game.nn_grid_size) {
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

    /// Render the board as ANSI-coloured flat-top hex ASCII art.
    /// Automatically highlights the last move: destination in orange, source in gray.
    fn render_board(&self) -> String {
        let last = self.game.move_history().last();
        let highlight = last.and_then(|mv| mv.to);
        let source    = last.and_then(|mv| mv.from);
        self.game.board.render(highlight, source)
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
        hive_game::uhp::format_move_uhp(&self.game, &mv)
    }

    /// Play a move given a UHP move string (e.g. "wQ", "wS1 wA1-", "pass").
    /// Parses the reference piece and direction to resolve the target hex, then
    /// finds and plays the matching valid move.
    /// Returns True if the move was found and played, False if not valid.
    fn play_move_uhp(&mut self, move_str: &str) -> bool {
        hive_game::uhp::parse_and_play_uhp(&mut self.game, move_str)
    }

    /// Run MCTS for `simulations` sims and return the best move as a UHP string.
    /// eval_fn(board_4d[N, C, H, W], reserve[N, R]) -> (policy[N, P], value[N])
    #[pyo3(signature = (eval_fn, simulations=800, c_puct=1.5))]
    fn best_move(
        &mut self,
        py: Python<'_>,
        eval_fn: &Bound<'_, PyAny>,
        simulations: usize,
        c_puct: f32,
    ) -> PyResult<String> {
        if self.game.is_game_over() {
            return Err(pyo3::exceptions::PyValueError::new_err("Game is already over"));
        }
        if self.game.valid_moves().is_empty() {
            return Ok("pass".to_string());
        }

        let gs = self.game.nn_grid_size;
        let board_size = NUM_CHANNELS * gs * gs;
        let ps = move_encoding::policy_size(gs);
        let batch = 8usize;

        let mut search = MctsSearch::<Game>::new(simulations + 64);
        search.c_puct = c_puct;

        // Initial NN eval on root
        let mut board_buf = vec![0.0f32; board_size];
        let mut reserve_buf = vec![0.0f32; RESERVE_SIZE];
        hive_game::board_encoding::encode_board(&self.game, &mut board_buf, &mut reserve_buf, gs);

        let root_board = make_array2(py, &board_buf, 1, board_size);
        let root_board_4d = root_board.reshape([1usize, NUM_CHANNELS, gs, gs]).unwrap();
        let root_reserve = make_array2(py, &reserve_buf, 1, RESERVE_SIZE);

        let result = eval_fn.call1((root_board_4d, root_reserve))?;
        let tuple = result.downcast::<PyTuple>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("eval_fn must return (policy, value)")
        })?;
        let root_policy: PyReadonlyArray2<f32> = tuple.get_item(0)?.extract()?;
        let root_policy_vec = root_policy.as_slice()?.to_vec();

        search.init(&self.game, &root_policy_vec);
        search.apply_root_dirichlet(0.3, 0.25);

        let mut done = 0usize;
        while done < simulations {
            let leaf_ids = search.select_leaves(batch.min(simulations - done));
            if leaf_ids.is_empty() { break; }
            let nl = leaf_ids.len();

            let mut boards = vec![0.0f32; nl * board_size];
            let mut reserves = vec![0.0f32; nl * RESERVE_SIZE];
            for (k, &lid) in leaf_ids.iter().enumerate() {
                let (board, reserve) = search.encode_leaf(lid);
                boards[k * board_size..(k + 1) * board_size].copy_from_slice(&board);
                reserves[k * RESERVE_SIZE..(k + 1) * RESERVE_SIZE].copy_from_slice(&reserve);
            }

            let leaf_board = make_array2(py, &boards, nl, board_size);
            let leaf_board_4d = leaf_board.reshape([nl, NUM_CHANNELS, gs, gs]).unwrap();
            let leaf_reserve = make_array2(py, &reserves, nl, RESERVE_SIZE);

            let res = eval_fn.call1((leaf_board_4d, leaf_reserve))?;
            let tup = res.downcast::<PyTuple>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("eval_fn must return (policy, value)")
            })?;
            let lp: PyReadonlyArray2<f32> = tup.get_item(0)?.extract()?;
            let lv: PyReadonlyArray1<f32> = tup.get_item(1)?.extract()?;
            let lp_data = lp.as_slice()?.to_vec();
            let lv_data: Vec<f32> = lv.as_slice()?.to_vec();

            let policies_vec: Vec<Vec<f32>> = (0..nl)
                .map(|k| lp_data[k * ps..(k + 1) * ps].to_vec())
                .collect();

            search.expand_and_backprop(&policies_vec, &lv_data);
            done += nl;
        }

        let dist = search.get_pruned_visit_distribution();
        let best = dist.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(mv, _)| {
                if mv.piece.is_none() {
                    "pass".to_string()
                } else {
                    hive_game::uhp::format_move_uhp(&self.game, mv)
                }
            })
            .unwrap_or_else(|| "pass".to_string());
        Ok(best)
    }
}

/// Batch MCTS: owns multiple search trees and parallelizes CPU work with rayon.
#[pyclass(name = "RustBatchMCTS")]
pub struct PyBatchMCTS {
    searches: Vec<MctsSearch<Game>>,
    c_puct: f32,
    leaf_batch_size: usize,
    use_forced_playouts: bool,
    grid_size: usize,
}

#[pymethods]
impl PyBatchMCTS {
    #[new]
    #[pyo3(signature = (num_games, c_puct=1.5, leaf_batch_size=16, capacity=100000, use_forced_playouts=false, grid_size=23))]
    fn new(num_games: usize, c_puct: f32, leaf_batch_size: usize, capacity: usize, use_forced_playouts: bool, grid_size: usize) -> Self {
        let searches: Vec<MctsSearch<Game>> = (0..num_games)
            .map(|_| {
                let mut s = MctsSearch::<Game>::new(capacity);
                s.c_puct = c_puct;
                s.use_forced_playouts = use_forced_playouts;
                s
            })
            .collect();
        PyBatchMCTS { searches, c_puct, leaf_batch_size, use_forced_playouts, grid_size }
    }

    /// Initialize MCTS trees for all games with pre-computed policies.
    /// policies shape: [num_games, policy_size]
    fn init_searches(&mut self, games: Vec<PyRef<PyGame>>, policies: PyReadonlyArray2<f32>) {
        use rayon::prelude::*;

        let policy_data = policies.as_slice().unwrap();
        let ps = move_encoding::policy_size(self.grid_size);
        // Collect game clones first (can't send PyRef across threads)
        let game_clones: Vec<Game> = games.iter().map(|g| g.game.clone()).collect();

        let use_forced = self.use_forced_playouts;
        self.searches.par_iter_mut().enumerate().for_each(|(i, search)| {
            search.c_puct = self.c_puct;
            search.use_forced_playouts = use_forced;
            let policy = &policy_data[i * ps..(i + 1) * ps];
            search.init(&game_clones[i], policy);
        });
    }

    /// Get root child counts for all trees.
    fn root_child_counts(&self) -> Vec<u16> {
        self.searches.iter().map(|s| s.root_child_count()).collect()
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

        let gs = self.grid_size;
        let board_size = NUM_CHANNELS * gs * gs;
        let ps = move_encoding::policy_size(gs);
        let leaf_batch_size = self.leaf_batch_size;
        let mut sims_done: Vec<usize> = vec![0; active_indices.len()];
        let mut searching: Vec<usize> = (0..active_indices.len()).collect();

        while !searching.is_empty() {
            // --- CPU (rayon): select leaves + encode ---
            let active_gis: Vec<usize> = searching.iter().map(|&si| active_indices[si]).collect();
            let searches = &mut self.searches;

            let mut active_searches: Vec<&mut MctsSearch<Game>> = Vec::with_capacity(active_gis.len());
            for &gi in &active_gis {
                let ptr = &mut searches[gi] as *mut MctsSearch<Game>;
                active_searches.push(unsafe { &mut *ptr });
            }

            // Cap per-game batch to remaining simulations.
            let per_game_batch: Vec<usize> = searching.iter()
                .map(|&si| (per_game_caps[si] - sims_done[si]).min(leaf_batch_size).max(1))
                .collect();

            let results: Vec<(usize, Vec<f32>, Vec<f32>)> = active_searches
                .par_iter_mut()
                .zip(per_game_batch.par_iter())
                .map(|(search, &batch)| {
                    let leaf_ids = search.select_leaves(batch);
                    let n = leaf_ids.len();
                    let mut boards = vec![0.0f32; n * board_size];
                    let mut reserves = vec![0.0f32; n * RESERVE_SIZE];
                    for (j, &lid) in leaf_ids.iter().enumerate() {
                        let (board, reserve) = search.encode_leaf(lid);
                        boards[j * board_size..(j + 1) * board_size].copy_from_slice(&board);
                        reserves[j * RESERVE_SIZE..(j + 1) * RESERVE_SIZE].copy_from_slice(&reserve);
                    }
                    (n, boards, reserves)
                })
                .collect();

            // Track sims by actual unique leaves found, not batch size
            let mut total_leaves = 0usize;
            let mut leaf_counts = Vec::with_capacity(searching.len());
            for (si_idx, &si) in searching.iter().enumerate() {
                let n = results[si_idx].0;
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
            for (n, boards, reserves) in &results {
                if *n > 0 {
                    all_boards[offset * board_size..(offset + n) * board_size]
                        .copy_from_slice(boards);
                    all_reserves[offset * RESERVE_SIZE..(offset + n) * RESERVE_SIZE]
                        .copy_from_slice(reserves);
                    offset += n;
                }
            }

            // --- GPU: single Python callback ---
            let board_arr = make_array2(py, &all_boards, total_leaves, board_size);
            let board_4d = board_arr.reshape([total_leaves, NUM_CHANNELS, gs, gs]).unwrap();
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

            let mut active_searches2: Vec<&mut MctsSearch<Game>> = Vec::with_capacity(active_gis.len());
            for &gi in &active_gis {
                let ptr = &mut searches[gi] as *mut MctsSearch<Game>;
                active_searches2.push(unsafe { &mut *ptr });
            }

            active_searches2.par_iter_mut().enumerate().for_each(|(i, search)| {
                let off = offsets[i];
                let n = leaf_counts[i];
                let policies_vec: Vec<Vec<f32>> = (0..n).map(|j| {
                    policy_data[(off + j) * ps..(off + j + 1) * ps].to_vec()
                }).collect();
                let values_vec: Vec<f32> = value_data[off..off + n].to_vec();
                search.expand_and_backprop(&policies_vec, &values_vec);
            });

            searching.retain(|&si| sims_done[si] < per_game_caps[si]);
        }
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

/// Compute the 12 D6 symmetry gather-permutation tables for a grid_size x grid_size board.
///
/// Returns a list of 12 numpy arrays, each of shape (grid_size*grid_size,).
/// perm[output_cell] = input_cell, or grid_size*grid_size (sentinel for out-of-bounds → zero).
#[pyfunction]
fn d6_grid_permutations<'py>(py: Python<'py>, grid_size: usize) -> Vec<Bound<'py, PyArray1<i64>>> {
    use core_game::symmetry::{D6Symmetry, Symmetry};

    let center = grid_size as i32 / 2;
    let num_cells = grid_size * grid_size;

    D6Symmetry::all().iter().map(|sym| {
        let mut perm = vec![num_cells as i64; num_cells];
        for row in 0..grid_size {
            for col in 0..grid_size {
                let q = col as i32 - center;
                let r = row as i32 - center;
                let (q2, r2) = sym.transform_hex(q, r);
                let nr = r2 + center;
                let nc = q2 + center;
                if nr >= 0 && nr < grid_size as i32 && nc >= 0 && nc < grid_size as i32 {
                    perm[nr as usize * grid_size + nc as usize] = (row * grid_size + col) as i64;
                }
            }
        }
        let arr = numpy::ndarray::Array1::from(perm);
        PyArray1::from_owned_array_bound(py, arr)
    }).collect()
}

/// Parse SGF content and return list of UHP move strings.
#[pyfunction]
fn parse_sgf_moves(content: &str) -> PyResult<Vec<String>> {
    let mut game = Game::new();
    let mut moves = Vec::new();
    hive_game::sgf::replay_into_game_verbose(content, &mut game, |g, mv| {
        if mv.piece.is_none() {
            moves.push("pass".to_string());
        } else {
            moves.push(hive_game::uhp::format_move_uhp(g, mv));
        }
    }).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    Ok(moves)
}

/// Register Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_class::<PyBatchMCTS>()?;
    m.add_function(wrap_pyfunction!(parse_sgf_moves, m)?)?;
    m.add_function(wrap_pyfunction!(d6_grid_permutations, m)?)?;
    Ok(())
}
