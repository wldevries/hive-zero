/// PyO3 Python bindings for Hive (token-based transformer architecture).
///
/// FFI contract for the inference callback (`eval_fn`):
/// ```text
///     eval_fn(
///         categoricals : np.ndarray[B, L, 5]  uint8
///         positions    : np.ndarray[B, L, 2]  int8
///         flags        : np.ndarray[B, L, F]  float32
///         mask         : np.ndarray[B, L]     bool
///     ) -> (
///         policy : np.ndarray[B, 2 * L * D]   float32  [Q || K] D-major
///         wdl    : np.ndarray[B, 3]           float32
///         aux    : np.ndarray[B, 6]           float32
///     )
/// ```
/// `L = hive_game::tokenize::SEQ_LEN`, `D = hive_game::tokenize::BILINEAR_DIM`.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};

use core_game::mcts::search::{
    CpuctStrategy, ForcedExploration, MctsSearch, RootNoise, SearchParams,
};

use hive_game::game::{self, Game, Move};
use hive_game::piece::{Piece, PieceColor, PieceType};
use hive_game::tokenize::{
    self, TokenBatch, BILINEAR_DIM, F_FLAGS, SEQ_LEN,
};

// ---- Python callback bridge: TokenBatch ↔ numpy --------------------------

/// Pack a slice of token batches into the four-tensor numpy form the
/// Python model expects.
fn token_batches_to_numpy<'py>(
    py: Python<'py>,
    batches: &[TokenBatch],
) -> PyResult<(
    Bound<'py, PyArray3<u8>>,
    Bound<'py, PyArray3<i8>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray2<bool>>,
)> {
    let b = batches.len();
    let l = SEQ_LEN;
    let f = F_FLAGS;

    // (B, L, 5) categoricals
    let mut cat = vec![0u8; b * l * 5];
    for (bi, t) in batches.iter().enumerate() {
        let base = bi * l * 5;
        for li in 0..l {
            cat[base + li * 5 + 0] = t.kind[li];
            cat[base + li * 5 + 1] = t.piece_type[li];
            cat[base + li * 5 + 2] = t.color[li];
            cat[base + li * 5 + 3] = t.stack_depth[li];
            cat[base + li * 5 + 4] = t.count[li];
        }
    }
    let cat_arr = numpy::ndarray::Array3::from_shape_vec((b, l, 5), cat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // (B, L, 2) positions
    let mut pos = vec![0i8; b * l * 2];
    for (bi, t) in batches.iter().enumerate() {
        let base = bi * l * 2;
        for li in 0..l {
            pos[base + li * 2 + 0] = t.q[li];
            pos[base + li * 2 + 1] = t.r[li];
        }
    }
    let pos_arr = numpy::ndarray::Array3::from_shape_vec((b, l, 2), pos)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // (B, L, F) flags
    let mut flg = vec![0f32; b * l * f];
    for (bi, t) in batches.iter().enumerate() {
        let base = bi * l * f;
        flg[base..base + l * f].copy_from_slice(&t.flags);
    }
    let flg_arr = numpy::ndarray::Array3::from_shape_vec((b, l, f), flg)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // (B, L) mask
    let mut msk = vec![false; b * l];
    for (bi, t) in batches.iter().enumerate() {
        let base = bi * l;
        msk[base..base + l].copy_from_slice(&t.mask);
    }
    let msk_arr = numpy::ndarray::Array2::from_shape_vec((b, l), msk)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        PyArray3::from_owned_array(py, cat_arr),
        PyArray3::from_owned_array(py, pos_arr),
        PyArray3::from_owned_array(py, flg_arr),
        PyArray2::from_owned_array(py, msk_arr),
    ))
}

/// Invoke the Python `eval_fn` on a batch of token batches. Returns per-leaf
/// (policy_flat, value, draw) — value = W − L; draw = D — to match the
/// `(Vec<Vec<f32>>, Vec<f32>, Vec<f32>)` shape MCTS' expand_and_backprop wants.
fn call_python_eval_tokens(
    eval_fn: &Bound<'_, PyAny>,
    batches: &[TokenBatch],
) -> PyResult<(Vec<Vec<f32>>, Vec<f32>, Vec<f32>)> {
    let py = eval_fn.py();
    let (cat, pos, flg, msk) = token_batches_to_numpy(py, batches)?;
    let result = eval_fn.call1((cat, pos, flg, msk))?;
    let tuple = result
        .cast::<PyTuple>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err(
            "eval_fn must return (policy, wdl, aux)",
        ))?;
    let policy = tuple.get_item(0)?.cast::<PyArray2<f32>>()
        .map_err(|e| pyo3::exceptions::PyTypeError::new_err(e.to_string()))?
        .readonly();
    let wdl = tuple.get_item(1)?.cast::<PyArray2<f32>>()
        .map_err(|e| pyo3::exceptions::PyTypeError::new_err(e.to_string()))?
        .readonly();
    let _aux = tuple.get_item(2).ok(); // tolerated but unused at inference

    let policy_slice = policy.as_slice()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let wdl_slice = wdl.as_slice()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let n = batches.len();
    let pol_per = 2 * SEQ_LEN * BILINEAR_DIM;
    let mut policies = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    let mut draws = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * pol_per;
        policies.push(policy_slice[start..start + pol_per].to_vec());
        // WDL → (W − L, D)
        values.push(wdl_slice[i * 3] - wdl_slice[i * 3 + 2]);
        draws.push(wdl_slice[i * 3 + 1]);
    }
    Ok((policies, values, draws))
}

// ---- token-aware best_move loop ------------------------------------------

/// Run MCTS for `simulations` sims using the token-based eval callback and
/// return the best move. Single-game, no pipelining — the caller drives one
/// batch at a time.
fn best_move_token(
    game: &Game,
    simulations: usize,
    play_batch: usize,
    params: &SearchParams,
    eval_fn: &Bound<'_, PyAny>,
) -> PyResult<Move> {
    let mut search = MctsSearch::<Game>::new(simulations.saturating_add(64));

    // Root expansion: tokenize once, evaluate, init the search.
    let mut root_game = game.clone();
    let (root_tokens, _) = tokenize::tokenize_and_priors(&mut root_game);
    let (root_policies, _, _) = call_python_eval_tokens(eval_fn, &[root_tokens])?;
    search.init(game, &root_policies[0]);

    // Optional root noise.
    if let RootNoise::Dirichlet { alpha, epsilon } = params.root_noise {
        search.params = params.clone();
        search.apply_root_dirichlet(alpha, epsilon);
    } else {
        search.params = params.clone();
    }

    let mut done = 0;
    while done < simulations {
        let want = play_batch.min(simulations - done);
        let leaves = search.select_leaves(want);
        if leaves.is_empty() {
            done += want;
            continue;
        }
        // Tokenize each non-terminal leaf via stashed_game.
        let mut batches: Vec<TokenBatch> = Vec::with_capacity(leaves.len());
        for &leaf in &leaves {
            let mut g = search
                .stashed_game(leaf)
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                    "stashed leaf missing during select_leaves"))?
                .clone();
            let (tb, _) = tokenize::tokenize_and_priors(&mut g);
            batches.push(tb);
        }
        let (policies, values, draws) = call_python_eval_tokens(eval_fn, &batches)?;
        search.expand_and_backprop(&policies, &values, &draws);
        done += leaves.len();
    }

    Ok(search.best_move().unwrap_or(Move::pass()))
}

// ---- numpy helpers --------------------------------------------------------

fn make_array1<'py, T: numpy::Element>(py: Python<'py>, data: Vec<T>) -> Bound<'py, PyArray1<T>> {
    let arr = numpy::ndarray::Array1::from(data);
    PyArray1::from_owned_array(py, arr)
}

// ---- PyGame: Rust Game exposed to Python ---------------------------------

#[pyclass(name = "HiveGame")]
pub struct PyGame {
    pub game: Game,
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
                format!("grid_size {grid_size} exceeds max board size {}",
                        hive_game::board::GRID_SIZE)));
        }
        let game = if tournament_mode {
            game::Game::new_tournament_with_grid_size(grid_size)
        } else {
            game::Game::new_with_grid_size(grid_size)
        };
        Ok(PyGame { game })
    }

    fn copy(&self) -> PyGame {
        PyGame { game: self.game.clone() }
    }

    #[getter]
    fn tournament_mode(&self) -> bool { self.game.tournament_mode }

    #[getter]
    fn state(&self) -> &str { self.game.state.as_str() }

    #[getter]
    fn is_game_over(&self) -> bool { self.game.is_game_over() }

    #[getter]
    fn turn_color(&self) -> &str {
        match self.game.turn_color {
            PieceColor::White => "w",
            PieceColor::Black => "b",
        }
    }

    #[getter]
    fn turn_number(&self) -> u16 { self.game.turn_number }

    #[getter]
    fn move_count(&self) -> u16 { self.game.move_count }

    #[getter]
    fn grid_size(&self) -> usize { self.game.nn_grid_size }

    fn valid_moves(&mut self) -> Vec<(String, Option<(i8, i8)>, (i8, i8))> {
        self.game.valid_moves().iter().map(|mv| {
            let piece_str = mv.piece.unwrap().to_uhp_string();
            (piece_str, mv.from, mv.to.unwrap())
        }).collect()
    }

    #[pyo3(signature = (piece_str, from_pos, to_pos))]
    fn play_move(&mut self, piece_str: &str, from_pos: Option<(i8, i8)>, to_pos: (i8, i8)) {
        let piece = Piece::from_str(piece_str).expect("invalid piece string");
        let mv = match from_pos {
            None => game::Move::placement(piece, to_pos),
            Some(f) => game::Move::movement(piece, f, to_pos),
        };
        self.game.play_move(&mv).unwrap();
    }

    fn play_pass(&mut self) { self.game.play_pass(); }
    fn undo(&mut self) { self.game.undo(); }

    /// Encode the current position as a token batch.
    /// Returns `(categoricals[L, 5] uint8, positions[L, 2] int8,
    ///          flags[L, F] float32, mask[L] bool)`.
    fn encode_tokens<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray2<i8>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<bool>>,
    )> {
        let (tb, _) = tokenize::tokenize_and_priors(&mut self.game);
        // Re-pack from per-field Vecs into (L, K) arrays.
        let mut cat = vec![0u8; SEQ_LEN * 5];
        for li in 0..SEQ_LEN {
            cat[li * 5 + 0] = tb.kind[li];
            cat[li * 5 + 1] = tb.piece_type[li];
            cat[li * 5 + 2] = tb.color[li];
            cat[li * 5 + 3] = tb.stack_depth[li];
            cat[li * 5 + 4] = tb.count[li];
        }
        let cat_arr = numpy::ndarray::Array2::from_shape_vec((SEQ_LEN, 5), cat)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let mut pos = vec![0i8; SEQ_LEN * 2];
        for li in 0..SEQ_LEN {
            pos[li * 2 + 0] = tb.q[li];
            pos[li * 2 + 1] = tb.r[li];
        }
        let pos_arr = numpy::ndarray::Array2::from_shape_vec((SEQ_LEN, 2), pos)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let flg_arr = numpy::ndarray::Array2::from_shape_vec((SEQ_LEN, F_FLAGS), tb.flags)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok((
            PyArray2::from_owned_array(py, cat_arr),
            PyArray2::from_owned_array(py, pos_arr),
            PyArray2::from_owned_array(py, flg_arr),
            make_array1(py, tb.mask),
        ))
    }

    /// Tokenize the current position and also return per-legal-move slot pairs.
    ///
    /// Returns `(categoricals, positions, flags, mask, moves)` where each
    /// entry of `moves` is `(mover_slot, dest_slot, piece_uhp, from_or_None,
    /// to)`. This is the entry point training-data writers will use to
    /// translate MCTS visit distributions into (mover, dest) token-pair
    /// targets.
    fn encode_tokens_with_moves<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray2<i8>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<bool>>,
        Vec<(usize, usize, String, Option<(i8, i8)>, (i8, i8))>,
    )> {
        use core_game::game::PolicyIndex;
        let (tb, indexed) = tokenize::tokenize_and_priors(&mut self.game);

        let mut cat = vec![0u8; SEQ_LEN * 5];
        for li in 0..SEQ_LEN {
            cat[li * 5 + 0] = tb.kind[li];
            cat[li * 5 + 1] = tb.piece_type[li];
            cat[li * 5 + 2] = tb.color[li];
            cat[li * 5 + 3] = tb.stack_depth[li];
            cat[li * 5 + 4] = tb.count[li];
        }
        let cat_arr = numpy::ndarray::Array2::from_shape_vec((SEQ_LEN, 5), cat)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let mut pos = vec![0i8; SEQ_LEN * 2];
        for li in 0..SEQ_LEN {
            pos[li * 2 + 0] = tb.q[li];
            pos[li * 2 + 1] = tb.r[li];
        }
        let pos_arr = numpy::ndarray::Array2::from_shape_vec((SEQ_LEN, 2), pos)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let flg_arr = numpy::ndarray::Array2::from_shape_vec((SEQ_LEN, F_FLAGS), tb.flags)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let moves: Vec<_> = indexed.iter().map(|(enc, mv)| {
            let (mover_slot, dest_slot) = match *enc {
                PolicyIndex::DotProduct { src_cell, dst_cell, .. } => (src_cell, dst_cell),
                _ => unreachable!("tokenize must emit DotProduct"),
            };
            (mover_slot, dest_slot, mv.piece.unwrap().to_uhp_string(), mv.from, mv.to.unwrap())
        }).collect();
        Ok((
            PyArray2::from_owned_array(py, cat_arr),
            PyArray2::from_owned_array(py, pos_arr),
            PyArray2::from_owned_array(py, flg_arr),
            make_array1(py, tb.mask),
            moves,
        ))
    }

    fn reserve_has(&self, piece_str: &str) -> bool {
        let piece = Piece::from_str(piece_str).expect("invalid piece string");
        self.game.reserve_has(piece)
    }

    fn reserve_count(&self, color: &str, piece_type: &str) -> u8 {
        let c = match color {
            "w" => PieceColor::White,
            "b" => PieceColor::Black,
            _ => panic!("invalid color"),
        };
        let pt = PieceType::from_char(piece_type.chars().next().unwrap())
            .expect("invalid piece type");
        self.game.reserve_count(c, pt)
    }

    fn turn_string(&self) -> String { self.game.turn_string() }

    fn all_top_pieces(&self) -> Vec<((i8, i8), String)> {
        self.game.board.all_top_pieces().iter().map(|(hex, piece)| {
            (*hex, piece.to_uhp_string())
        }).collect()
    }

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

    #[pyo3(signature = (q, r))]
    fn stack_at(&self, q: i8, r: i8) -> Vec<String> {
        let slot = self.game.board.stack_at((q, r));
        slot.iter().map(|p| p.to_uhp_string()).collect()
    }

    fn render_board(&self) -> String {
        let (source, highlight) = self.game.last_move_display_coords();
        self.game.board.render(highlight, source)
    }

    fn evaluate(&self) -> (f32, f32) { self.game.evaluate() }

    #[getter]
    fn game_string(&self) -> String { self.game.game_string() }

    #[pyo3(signature = (piece_str, from_pos, to_pos))]
    fn format_move_uhp(&self, piece_str: &str, from_pos: Option<(i8, i8)>, to_pos: (i8, i8)) -> String {
        let piece = Piece::from_str(piece_str).expect("invalid piece string");
        let mv = match from_pos {
            None => game::Move::placement(piece, to_pos),
            Some(f) => game::Move::movement(piece, f, to_pos),
        };
        hive_game::uhp::format_move_uhp(&self.game, &mv)
    }

    fn play_move_uhp(&mut self, move_str: &str) -> bool {
        hive_game::uhp::parse_and_play_uhp(&mut self.game, move_str)
    }

    fn last_recenter_shift(&self) -> (i8, i8) { self.game.last_recenter_shift() }

    /// Run MCTS for `simulations` sims and return the best move as a UHP string.
    ///
    /// `eval_fn(categoricals[B, L, 5] uint8, positions[B, L, 2] int8,
    ///          flags[B, L, F] float32, mask[B, L] bool) ->
    ///          (policy[B, 2*L*D] float32, wdl[B, 3] float32, aux[B, 6] float32)`
    ///
    /// `play_batch` controls how many leaves are selected and evaluated per
    /// inference call (typical: 8–32 for batched GPU efficiency).
    #[pyo3(signature = (
        eval_fn,
        simulations=800,
        c_puct=1.5,
        draw_contempt=0.0,
        dir_alpha=0.0,
        dir_epsilon=0.0,
        play_batch=8,
    ))]
    fn best_move(
        &mut self,
        _py: Python<'_>,
        eval_fn: &Bound<'_, PyAny>,
        simulations: usize,
        c_puct: f32,
        draw_contempt: f32,
        dir_alpha: f32,
        dir_epsilon: f32,
        play_batch: usize,
    ) -> PyResult<String> {
        if self.game.is_game_over() {
            return Err(pyo3::exceptions::PyValueError::new_err("Game is already over"));
        }
        if self.game.valid_moves().is_empty() {
            return Ok("pass".to_string());
        }

        let root_noise = if dir_alpha > 0.0 && dir_epsilon > 0.0 {
            RootNoise::Dirichlet { alpha: dir_alpha, epsilon: dir_epsilon }
        } else {
            RootNoise::None
        };
        let mut params = SearchParams::new(
            CpuctStrategy::Constant { c_puct },
            ForcedExploration::None,
            root_noise,
        );
        params.draw_contempt = draw_contempt;

        let best = best_move_token(&self.game, simulations, play_batch.max(1), &params, eval_fn)?;
        Ok(if best.piece.is_none() {
            "pass".to_string()
        } else {
            hive_game::uhp::format_move_uhp(&self.game, &best)
        })
    }

    /// Run alphabeta search to the given depth using the built-in heuristic
    /// evaluation, returning the best move as a UHP string. No NN involved.
    #[pyo3(signature = (depth=3))]
    fn best_move_alphabeta(&mut self, depth: u32) -> PyResult<String> {
        if self.game.is_game_over() {
            return Err(pyo3::exceptions::PyValueError::new_err("Game is already over"));
        }
        if self.game.valid_moves().is_empty() {
            return Ok("pass".to_string());
        }
        let best = hive_game::alphabeta::alphabeta_best_move(&self.game, depth);
        Ok(if best.piece.is_none() {
            "pass".to_string()
        } else {
            hive_game::uhp::format_move_uhp(&self.game, &best)
        })
    }

    /// Run MCTS using a pre-loaded ORT/QNN session — DEFERRED for the token
    /// architecture. Phase G (ORT) will reimplement this.
    #[pyo3(signature = (_ort_session, _simulations=800, _c_puct=1.5, _draw_contempt=0.0,
                       _dir_alpha=0.0, _dir_epsilon=0.0, _play_batch=8))]
    fn best_move_ort(
        &mut self,
        _ort_session: Bound<'_, PyAny>,
        _simulations: usize,
        _c_puct: f32,
        _draw_contempt: f32,
        _dir_alpha: f32,
        _dir_epsilon: f32,
        _play_batch: usize,
    ) -> PyResult<String> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "best_move_ort is not yet ported to the token architecture (Phase G)",
        ))
    }
}

// ---- module-level helpers exposed to Python ------------------------------

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

/// 12 D6 axial-coordinate transform matrices, each (2, 2) int8 mapping
/// `[q, r]^T → R · [q, r]^T`. Used by the training pipeline to permute
/// per-token (q, r) positions during D6 symmetry augmentation.
#[pyfunction]
fn d6_axial_transforms<'py>(py: Python<'py>) -> Vec<Bound<'py, PyArray2<i8>>> {
    use core_game::symmetry::{D6Symmetry, Symmetry};
    D6Symmetry::all().iter().map(|sym| {
        // Each row is the image of (1, 0) and (0, 1) under sym.
        let (a, b) = sym.transform_hex(1, 0);
        let (c, d) = sym.transform_hex(0, 1);
        let mat: Vec<i8> = vec![a as i8, b as i8, c as i8, d as i8];
        let arr = numpy::ndarray::Array2::from_shape_vec((2, 2), mat).unwrap();
        PyArray2::from_owned_array(py, arr)
    }).collect()
}

/// Hex direction permutation table for D6: shape (12, 6). Row `s` gives the
/// permutation `pi` such that direction `d` maps to direction `pi[d]` under
/// symmetry index `s`. Used by training to remap direction-keyed relative-bias
/// buckets in HexRelativeBias.
///
/// Hex direction ordering matches `core_game::hex::HEX_DIRS`:
///   0=E (1,0), 1=NE (1,-1), 2=NW (0,-1), 3=W (-1,0), 4=SW (-1,1), 5=SE (0,1)
#[pyfunction]
fn d6_direction_permutations<'py>(py: Python<'py>) -> Bound<'py, PyArray2<i8>> {
    use core_game::hex::DIRECTIONS;
    use core_game::symmetry::{D6Symmetry, Symmetry};
    let mut table = vec![0i8; 12 * 6];
    for (si, sym) in D6Symmetry::all().iter().enumerate() {
        for (di, &(dq, dr)) in DIRECTIONS.iter().enumerate() {
            let (mq, mr) = sym.transform_hex(dq as i32, dr as i32);
            // Find which hex-dir index this matches.
            let mapped = DIRECTIONS.iter().position(|&(q, r)| q as i32 == mq && r as i32 == mr)
                .expect("hex dir image must be one of the 6 dirs");
            table[si * 6 + di] = mapped as i8;
        }
    }
    let arr = numpy::ndarray::Array2::from_shape_vec((12, 6), table).unwrap();
    PyArray2::from_owned_array(py, arr)
}

#[pyfunction]
fn parse_sgf_moves(content: &str) -> PyResult<Vec<String>> {
    if hive_game::sgf::game_type(content) != "base" {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expansion-piece game (M/L/P) — base-game parser only",
        ));
    }
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

/// Token vocabulary sizes exposed to Python so the training and self-play
/// code doesn't have to duplicate the constants.
#[pyfunction]
fn token_vocab() -> (usize, usize, usize) {
    (SEQ_LEN, F_FLAGS, BILINEAR_DIM)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_function(wrap_pyfunction!(parse_sgf_moves, m)?)?;
    m.add_function(wrap_pyfunction!(d6_grid_permutations, m)?)?;
    m.add_function(wrap_pyfunction!(d6_axial_transforms, m)?)?;
    m.add_function(wrap_pyfunction!(d6_direction_permutations, m)?)?;
    m.add_function(wrap_pyfunction!(token_vocab, m)?)?;
    Ok(())
}
