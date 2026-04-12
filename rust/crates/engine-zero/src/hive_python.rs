/// PyO3 Python bindings for hive.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};

use core_game::mcts::search::{CpuctStrategy, ForcedExploration, RootNoise, SearchParams};

use hive_game::board_encoding::{NUM_CHANNELS, RESERVE_SIZE};
use hive_game::game::{self, Game};
use hive_game::move_encoding;
use hive_game::piece::{Piece, PieceColor, PieceType};
use hive_game::search::best_move_core;

fn call_python_eval(
    eval_fn: &Bound<'_, PyAny>,
    boards: &[f32],
    reserves: &[f32],
    batch_size: usize,
    grid_size: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let board_size = NUM_CHANNELS * grid_size * grid_size;
    let board_arr = numpy::ndarray::Array2::from_shape_vec(
        (batch_size, board_size),
        boards.to_vec(),
    ).map_err(|e| e.to_string())?;
    let board_np = PyArray2::from_owned_array(eval_fn.py(), board_arr);
    let board_4d = board_np
        .reshape([batch_size, NUM_CHANNELS, grid_size, grid_size])
        .map_err(|e| e.to_string())?;

    let reserve_arr = numpy::ndarray::Array2::from_shape_vec(
        (batch_size, RESERVE_SIZE),
        reserves.to_vec(),
    ).map_err(|e| e.to_string())?;
    let reserve_np = PyArray2::from_owned_array(eval_fn.py(), reserve_arr);

    let result = eval_fn.call1((board_4d, reserve_np)).map_err(|e| e.to_string())?;
    let tuple = result
        .cast::<PyTuple>()
        .map_err(|_| "eval_fn must return (policy, value)".to_string())?;
    let policy = tuple
        .get_item(0)
        .map_err(|e| e.to_string())?
        .cast::<PyArray2<f32>>()
        .map_err(|e| e.to_string())?
        .readonly();
    let value = tuple
        .get_item(1)
        .map_err(|e| e.to_string())?
        .cast::<PyArray1<f32>>()
        .map_err(|e| e.to_string())?
        .readonly();
    Ok((
        policy.as_slice().map_err(|e| e.to_string())?.to_vec(),
        value.as_slice().map_err(|e| e.to_string())?.to_vec(),
    ))
}

/// Create a 1D numpy array from a slice.
fn make_array1<'py>(py: Python<'py>, data: &[f32]) -> Bound<'py, PyArray1<f32>> {
    let arr = numpy::ndarray::Array1::from(data.to_vec());
    PyArray1::from_owned_array(py, arr)
}

/// Create a 3D numpy array from a flat slice with shape.
fn make_array3<'py>(py: Python<'py>, data: &[f32], d0: usize, d1: usize, d2: usize) -> Bound<'py, PyArray3<f32>> {
    let arr = numpy::ndarray::Array3::from_shape_vec((d0, d1, d2), data.to_vec()).unwrap();
    PyArray3::from_owned_array(py, arr)
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

    /// Board dimension stats over occupied hexes (axial coords q, r, s=-q-r).
    /// Returns (max_abs_q, max_abs_r, max_abs_s, span_q, span_r, span_s).
    /// Returns all zeros if the board is empty.
    fn board_dims(&self) -> (i32, i32, i32, i32, i32, i32) {
        let mut min_q = i32::MAX; let mut max_q = i32::MIN;
        let mut min_r = i32::MAX; let mut max_r = i32::MIN;
        let mut min_s = i32::MAX; let mut max_s = i32::MIN;
        for ((q, r), _) in self.game.board.iter_occupied() {
            let q = q as i32; let r = r as i32; let s = -q - r;
            min_q = min_q.min(q); max_q = max_q.max(q);
            min_r = min_r.min(r); max_r = max_r.max(r);
            min_s = min_s.min(s); max_s = max_s.max(s);
        }
        if min_q == i32::MAX {
            return (0, 0, 0, 0, 0, 0);
        }
        let max_abs_q = min_q.abs().max(max_q.abs());
        let max_abs_r = min_r.abs().max(max_r.abs());
        let max_abs_s = min_s.abs().max(max_s.abs());
        (max_abs_q, max_abs_r, max_abs_s, max_q - min_q, max_r - min_r, max_s - min_s)
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
        _py: Python<'_>,
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
        let core_eval: hive_game::search::EvalFn<'_> = Box::new(move |boards, reserves, batch_size| {
            call_python_eval(eval_fn, boards, reserves, batch_size, gs)
        });
        let search_params = SearchParams::new(
            CpuctStrategy::Constant { c_puct },
            ForcedExploration::None,
            RootNoise::None,
        );
        let best = best_move_core(&self.game, simulations, &search_params, core_eval)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(if best.piece.is_none() {
            "pass".to_string()
        } else {
            hive_game::uhp::format_move_uhp(&self.game, &best)
        })
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
        PyArray1::from_owned_array(py, arr)
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
    m.add_function(wrap_pyfunction!(parse_sgf_moves, m)?)?;
    m.add_function(wrap_pyfunction!(d6_grid_permutations, m)?)?;
    Ok(())
}
