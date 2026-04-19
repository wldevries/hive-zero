use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use hive_game::board_encoding::{NUM_CHANNELS, RESERVE_SIZE, f32_to_bf16};
use hive_game::game::Game;
use hive_game::move_encoding;
use hive_game::search::{self, play_battle_core, play_selfplay_core};

use crate::inference::HiveInference;

struct SelfPlayConfig {
    num_games: usize,
    simulations: usize,
    max_moves: u32,
    temperature: f32,
    temp_threshold: u32,
    playout_cap_p: f32,
    fast_cap: usize,
    forced_playouts: bool,
    c_puct: f32,
    dir_alpha: f32,
    dir_epsilon: f32,
    leaf_batch_size: usize,
    resign_threshold: Option<f32>,
    resign_moves: u32,
    resign_min_moves: u32,
    calibration_frac: f32,
    random_opening_moves_min: u32,
    random_opening_moves_max: u32,
    skip_timeout_games: bool,
    use_heuristic: bool,
    grid_size: usize,
    fixed_batch_size: Option<usize>,
}

fn call_python_eval_bf16(
    py: Python<'_>,
    eval_fn: &Bound<'_, PyAny>,
    boards: &[f32],
    reserves: &[f32],
    batch_size: usize,
    grid_size: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let board_flat = NUM_CHANNELS * grid_size * grid_size;
    let boards_bf16: Vec<u16> = boards.iter().map(|&value| f32_to_bf16(value)).collect();
    let reserves_bf16: Vec<u16> = reserves.iter().map(|&value| f32_to_bf16(value)).collect();

    let board_arr = numpy::ndarray::Array2::from_shape_vec((batch_size, board_flat), boards_bf16)
        .map_err(|e| e.to_string())?;
    let board_np = PyArray2::from_owned_array(py, board_arr);
    let board_4d = board_np
        .reshape([batch_size, NUM_CHANNELS, grid_size, grid_size])
        .map_err(|e| e.to_string())?;

    let reserve_arr = numpy::ndarray::Array2::from_shape_vec((batch_size, RESERVE_SIZE), reserves_bf16)
        .map_err(|e| e.to_string())?;
    let reserve_np = PyArray2::from_owned_array(py, reserve_arr);

    let result = eval_fn.call1((board_4d, reserve_np)).map_err(|e| e.to_string())?;
    let tuple = result
        .cast::<PyTuple>()
        .map_err(|_| "eval_fn must return (policy, value) tuple".to_string())?;
    let policy_arr = tuple
        .get_item(0)
        .map_err(|e| e.to_string())?
        .cast::<PyArray2<f32>>()
        .map_err(|e| e.to_string())?
        .readonly();
    let value_arr = tuple
        .get_item(1)
        .map_err(|e| e.to_string())?
        .cast::<PyArray1<f32>>()
        .map_err(|e| e.to_string())?
        .readonly();
    Ok((
        policy_arr.as_slice().map_err(|e| e.to_string())?.to_vec(),
        value_arr.as_slice().map_err(|e| e.to_string())?.to_vec(),
    ))
}

fn infer_padded<F>(
    mut infer_once: F,
    boards: &[f32],
    reserves: &[f32],
    actual: usize,
    target: usize,
    grid_size: usize,
) -> Result<(Vec<f32>, Vec<f32>), String>
where
    F: FnMut(&[f32], &[f32], usize) -> Result<(Vec<f32>, Vec<f32>), String>,
{
    let policy_size = move_encoding::policy_size(grid_size);
    let board_size = NUM_CHANNELS * grid_size * grid_size;

    let mut out_policy = Vec::with_capacity(actual * policy_size);
    let mut out_value = Vec::with_capacity(actual);

    let mut offset = 0usize;
    while offset < actual {
        let chunk = (actual - offset).min(target);
        let board_slice = &boards[offset * board_size..(offset + chunk) * board_size];
        let reserve_slice = &reserves[offset * RESERVE_SIZE..(offset + chunk) * RESERVE_SIZE];

        let (mut policy, mut value) = if chunk == target {
            infer_once(board_slice, reserve_slice, target)?
        } else {
            let mut padded_boards = board_slice.to_vec();
            padded_boards.resize(target * board_size, 0.0);
            let mut padded_reserves = reserve_slice.to_vec();
            padded_reserves.resize(target * RESERVE_SIZE, 0.0);
            let (mut policy, mut value) = infer_once(&padded_boards, &padded_reserves, target)?;
            policy.truncate(chunk * policy_size);
            value.truncate(chunk);
            (policy, value)
        };

        out_policy.append(&mut policy);
        out_value.append(&mut value);
        offset += chunk;
    }

    Ok((out_policy, out_value))
}

fn make_selfplay_progress<'py>(
    py: Python<'py>,
    progress_fn: Option<&'py Bound<'py, PyAny>>,
) -> Option<search::SelfPlayProgressFn<'py>> {
    progress_fn.map(|progress_fn| {
        Box::new(move |finished, total, active, total_moves, resigned, draws, max_turn| {
            progress_fn
                .call1((finished, total, active, total_moves, resigned, draws, max_turn))
                .ok();
            py.check_signals().ok();
        }) as search::SelfPlayProgressFn<'py>
    })
}

fn make_battle_progress<'py>(
    py: Python<'py>,
    progress_fn: Option<&'py Bound<'py, PyAny>>,
) -> Option<search::BattleProgressFn<'py>> {
    progress_fn.map(|progress_fn| {
        Box::new(move |finished, total, active, total_moves| {
            progress_fn.call1((finished, total, active, total_moves)).ok();
            py.check_signals().ok();
        }) as search::BattleProgressFn<'py>
    })
}

fn into_py_selfplay_result(result: search::SelfPlayResult) -> PySelfPlayResult {
    PySelfPlayResult {
        grid_size: result.grid_size,
        board_data: result.board_data,
        reserve_data: result.reserve_data,
        place_idx_data: result.place_idx_data,
        place_prob_data: result.place_prob_data,
        place_offsets: result.place_offsets,
        movement_src_data: result.movement_src_data,
        movement_dst_data: result.movement_dst_data,
        movement_prob_data: result.movement_prob_data,
        movement_offsets: result.movement_offsets,
        value_targets: result.value_targets,
        value_only_flags: result.value_only_flags,
        policy_only_flags: result.policy_only_flags,
        my_queen_danger: result.my_queen_danger,
        opp_queen_danger: result.opp_queen_danger,
        my_queen_escape: result.my_queen_escape,
        opp_queen_escape: result.opp_queen_escape,
        my_mobility: result.my_mobility,
        opp_mobility: result.opp_mobility,
        num_samples: result.num_samples,
        wins_w: result.wins_w,
        wins_b: result.wins_b,
        draws: result.draws,
        draws_timeout: result.draws_timeout,
        draws_repetition: result.draws_repetition,
        resignations: result.resignations,
        total_moves: result.total_moves,
        full_search_turns: result.full_search_turns,
        total_turns: result.total_turns,
        calibration_total: result.calibration_total,
        calibration_would_resign: result.calibration_would_resign,
        calibration_false_positives: result.calibration_false_positives,
        use_playout_cap: result.use_playout_cap,
        final_games: result.final_games,
    }
}

fn into_py_battle_result(result: search::BattleResult) -> PyHiveBattleResult {
    PyHiveBattleResult {
        wins_model1: result.wins_model1,
        wins_model2: result.wins_model2,
        draws: result.draws,
        game_lengths: result.game_lengths,
    }
}

#[pyclass(name = "SelfPlayResult")]
pub struct PySelfPlayResult {
    grid_size: usize,
    board_data: Vec<f32>,
    reserve_data: Vec<f32>,
    place_idx_data: Vec<u16>,
    place_prob_data: Vec<f32>,
    place_offsets: Vec<u32>,
    movement_src_data: Vec<u16>,
    movement_dst_data: Vec<u16>,
    movement_prob_data: Vec<f32>,
    movement_offsets: Vec<u32>,
    value_targets: Vec<f32>,
    value_only_flags: Vec<bool>,
    policy_only_flags: Vec<bool>,
    my_queen_danger: Vec<f32>,
    opp_queen_danger: Vec<f32>,
    my_queen_escape: Vec<f32>,
    opp_queen_escape: Vec<f32>,
    my_mobility: Vec<f32>,
    opp_mobility: Vec<f32>,
    num_samples: usize,
    wins_w: u32,
    wins_b: u32,
    draws: u32,
    draws_timeout: u32,
    draws_repetition: u32,
    resignations: u32,
    total_moves: u32,
    full_search_turns: u32,
    total_turns: u32,
    calibration_total: u32,
    calibration_would_resign: u32,
    calibration_false_positives: u32,
    use_playout_cap: bool,
    final_games: Vec<Game>,
}

#[pymethods]
impl PySelfPlayResult {
    fn training_data<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray2<f32>>,   // boards (n, board_size)
        Bound<'py, PyArray2<f32>>,   // reserves (n, RESERVE_SIZE)
        // placement targets (sparse CSR → padded)
        (
            Bound<'py, PyArray2<u16>>,   // place_idx (n, MAX_PLACEMENTS)
            Bound<'py, PyArray2<f32>>,   // place_probs (n, MAX_PLACEMENTS)
            Bound<'py, PyArray1<i32>>,   // num_placements (n,)
        ),
        Bound<'py, PyArray1<f32>>,   // values (n,)
        Vec<bool>,                   // value_only_flags
        Vec<bool>,                   // policy_only_flags
        Bound<'py, PyArray2<f32>>,   // aux (n, 6)
        // movement targets
        (
            Bound<'py, PyArray2<u16>>,   // movement_srcs (n, MAX_MOVE_PAIRS)
            Bound<'py, PyArray2<u16>>,   // movement_dsts (n, MAX_MOVE_PAIRS)
            Bound<'py, PyArray2<f32>>,   // movement_probs (n, MAX_MOVE_PAIRS)
            Bound<'py, PyArray1<i32>>,   // num_movements (n,)
        ),
    ) {
        const MAX_PLACEMENTS: usize = 128;
        const MAX_MOVE_PAIRS: usize = 256;
        let n = self.num_samples;
        let board_size = NUM_CHANNELS * self.grid_size * self.grid_size;

        let boards = numpy::ndarray::Array2::from_shape_vec((n, board_size), self.board_data.clone()).unwrap();
        let reserves = numpy::ndarray::Array2::from_shape_vec((n, RESERVE_SIZE), self.reserve_data.clone()).unwrap();
        let values = numpy::ndarray::Array1::from(self.value_targets.clone());

        let mut aux_data = Vec::with_capacity(n * 6);
        for index in 0..n {
            aux_data.push(self.my_queen_danger[index]);
            aux_data.push(self.opp_queen_danger[index]);
            aux_data.push(self.my_queen_escape[index]);
            aux_data.push(self.opp_queen_escape[index]);
            aux_data.push(self.my_mobility[index]);
            aux_data.push(self.opp_mobility[index]);
        }
        let aux = numpy::ndarray::Array2::from_shape_vec((n, 6), aux_data).unwrap();

        // Build padded placement arrays from CSR data
        let mut place_idx = vec![0u16; n * MAX_PLACEMENTS];
        let mut place_probs = vec![0.0f32; n * MAX_PLACEMENTS];
        let mut num_placements = vec![0i32; n];
        for i in 0..n {
            let start = self.place_offsets[i] as usize;
            let end = self.place_offsets[i + 1] as usize;
            let count = (end - start).min(MAX_PLACEMENTS);
            num_placements[i] = count as i32;
            let row = i * MAX_PLACEMENTS;
            place_idx[row..row + count].copy_from_slice(&self.place_idx_data[start..start + count]);
            place_probs[row..row + count].copy_from_slice(&self.place_prob_data[start..start + count]);
        }

        // Build padded movement arrays from CSR data
        let mut mv_srcs = vec![0u16; n * MAX_MOVE_PAIRS];
        let mut mv_dsts = vec![0u16; n * MAX_MOVE_PAIRS];
        let mut mv_probs = vec![0.0f32; n * MAX_MOVE_PAIRS];
        let mut num_movements = vec![0i32; n];

        for i in 0..n {
            let start = self.movement_offsets[i] as usize;
            let end = self.movement_offsets[i + 1] as usize;
            let count = (end - start).min(MAX_MOVE_PAIRS);
            num_movements[i] = count as i32;
            let row = i * MAX_MOVE_PAIRS;
            mv_srcs[row..row + count].copy_from_slice(&self.movement_src_data[start..start + count]);
            mv_dsts[row..row + count].copy_from_slice(&self.movement_dst_data[start..start + count]);
            mv_probs[row..row + count].copy_from_slice(&self.movement_prob_data[start..start + count]);
        }

        let place_idx_arr = numpy::ndarray::Array2::from_shape_vec((n, MAX_PLACEMENTS), place_idx).unwrap();
        let place_probs_arr = numpy::ndarray::Array2::from_shape_vec((n, MAX_PLACEMENTS), place_probs).unwrap();
        let num_place_arr = numpy::ndarray::Array1::from(num_placements);
        let mv_srcs_arr = numpy::ndarray::Array2::from_shape_vec((n, MAX_MOVE_PAIRS), mv_srcs).unwrap();
        let mv_dsts_arr = numpy::ndarray::Array2::from_shape_vec((n, MAX_MOVE_PAIRS), mv_dsts).unwrap();
        let mv_probs_arr = numpy::ndarray::Array2::from_shape_vec((n, MAX_MOVE_PAIRS), mv_probs).unwrap();
        let num_mv_arr = numpy::ndarray::Array1::from(num_movements);

        (
            PyArray2::from_owned_array(py, boards),
            PyArray2::from_owned_array(py, reserves),
            (
                PyArray2::from_owned_array(py, place_idx_arr),
                PyArray2::from_owned_array(py, place_probs_arr),
                PyArray1::from_owned_array(py, num_place_arr),
            ),
            PyArray1::from_owned_array(py, values),
            self.value_only_flags.clone(),
            self.policy_only_flags.clone(),
            PyArray2::from_owned_array(py, aux),
            (
                PyArray2::from_owned_array(py, mv_srcs_arr),
                PyArray2::from_owned_array(py, mv_dsts_arr),
                PyArray2::from_owned_array(py, mv_probs_arr),
                PyArray1::from_owned_array(py, num_mv_arr),
            ),
        )
    }

    #[getter]
    fn num_samples(&self) -> usize { self.num_samples }
    #[getter]
    fn wins_w(&self) -> u32 { self.wins_w }
    #[getter]
    fn wins_b(&self) -> u32 { self.wins_b }
    #[getter]
    fn draws(&self) -> u32 { self.draws }
    #[getter]
    fn draws_timeout(&self) -> u32 { self.draws_timeout }
    #[getter]
    fn draws_repetition(&self) -> u32 { self.draws_repetition }
    #[getter]
    fn resignations(&self) -> u32 { self.resignations }
    #[getter]
    fn total_moves(&self) -> u32 { self.total_moves }
    #[getter]
    fn full_search_turns(&self) -> u32 { self.full_search_turns }
    #[getter]
    fn total_turns(&self) -> u32 { self.total_turns }
    #[getter]
    fn use_playout_cap(&self) -> bool { self.use_playout_cap }
    #[getter]
    fn calibration_total(&self) -> u32 { self.calibration_total }
    #[getter]
    fn calibration_would_resign(&self) -> u32 { self.calibration_would_resign }
    #[getter]
    fn calibration_false_positives(&self) -> u32 { self.calibration_false_positives }

    fn final_games(&self) -> Vec<crate::hive_python::PyGame> {
        self.final_games
            .iter()
            .map(|game| crate::hive_python::PyGame { game: game.clone() })
            .collect()
    }
}

#[pyclass(name = "RustSelfPlaySession")]
pub struct PySelfPlaySession {
    config: SelfPlayConfig,
}

#[pymethods]
impl PySelfPlaySession {
    #[new]
    #[pyo3(signature = (
        num_games,
        simulations = 100,
        max_moves = 200,
        temperature = 1.0,
        temp_threshold = 30,
        playout_cap_p = 0.0,
        fast_cap = 20,
        forced_playouts = false,
        c_puct = 1.5,
        dir_alpha = 0.3,
        dir_epsilon = 0.25,
        leaf_batch_size = 1,
        resign_threshold = None,
        resign_moves = 5,
        resign_min_moves = 20,
        calibration_frac = 0.1,
        random_opening_moves_min = 0,
        random_opening_moves_max = 0,
        skip_timeout_games = false,
        use_heuristic = false,
        grid_size = 23,
        fixed_batch_size = None,
    ))]
    fn new(
        num_games: usize,
        simulations: usize,
        max_moves: u32,
        temperature: f32,
        temp_threshold: u32,
        playout_cap_p: f32,
        fast_cap: usize,
        forced_playouts: bool,
        c_puct: f32,
        dir_alpha: f32,
        dir_epsilon: f32,
        leaf_batch_size: usize,
        resign_threshold: Option<f32>,
        resign_moves: u32,
        resign_min_moves: u32,
        calibration_frac: f32,
        random_opening_moves_min: u32,
        random_opening_moves_max: u32,
        skip_timeout_games: bool,
        use_heuristic: bool,
        grid_size: usize,
        fixed_batch_size: Option<usize>,
    ) -> Self {
        PySelfPlaySession {
            config: SelfPlayConfig {
                num_games,
                simulations,
                max_moves,
                temperature,
                temp_threshold,
                playout_cap_p,
                fast_cap,
                forced_playouts,
                c_puct,
                dir_alpha,
                dir_epsilon,
                leaf_batch_size,
                resign_threshold,
                resign_moves,
                resign_min_moves,
                calibration_frac,
                random_opening_moves_min,
                random_opening_moves_max,
                skip_timeout_games,
                use_heuristic,
                grid_size,
                fixed_batch_size,
            },
        }
    }

    #[pyo3(signature = (eval_fn=None, progress_fn=None, opening_sequences=None, onnx_path=None))]
    fn play_games(
        &self,
        py: Python<'_>,
        eval_fn: Option<&Bound<'_, PyAny>>,
        progress_fn: Option<&Bound<'_, PyAny>>,
        opening_sequences: Option<Vec<Vec<String>>>,
        onnx_path: Option<String>,
    ) -> PyResult<PySelfPlayResult> {
        let cfg = &self.config;
        let opening_sequences = opening_sequences.unwrap_or_default();
        let progress_core = make_selfplay_progress(py, progress_fn);

        let result = if let Some(path) = onnx_path {
            let mut engine = crate::inference::HiveOrtEngine::load(&path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let core_eval: search::EvalFn<'_> = Box::new(move |boards, reserves, actual| {
                let target = cfg.fixed_batch_size.unwrap_or(actual);
                infer_padded(
                    |chunk_boards, chunk_reserves, batch_size| {
                        let result = engine
                            .infer_batch(
                                chunk_boards,
                                chunk_reserves,
                                batch_size,
                                NUM_CHANNELS,
                                cfg.grid_size,
                                RESERVE_SIZE,
                            )
                            .map_err(|e| e.to_string())?;
                        Ok((result.policy, result.value))
                    },
                    boards,
                    reserves,
                    actual,
                    target,
                    cfg.grid_size,
                )
            });
            play_selfplay_core(
                cfg.num_games,
                cfg.simulations,
                cfg.max_moves,
                cfg.temperature,
                cfg.temp_threshold,
                cfg.playout_cap_p,
                cfg.fast_cap,
                cfg.forced_playouts,
                cfg.c_puct,
                cfg.dir_alpha,
                cfg.dir_epsilon,
                cfg.leaf_batch_size,
                cfg.resign_threshold,
                cfg.resign_moves,
                cfg.resign_min_moves,
                cfg.calibration_frac,
                cfg.random_opening_moves_min,
                cfg.random_opening_moves_max,
                cfg.skip_timeout_games,
                cfg.use_heuristic,
                cfg.grid_size,
                core_eval,
                progress_core,
                opening_sequences,
            )
        } else {
            let eval_fn = eval_fn.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("eval_fn is required when onnx_path is not provided")
            })?;
            let core_eval: search::EvalFn<'_> = Box::new(move |boards, reserves, actual| {
                let target = cfg.fixed_batch_size.unwrap_or(actual);
                infer_padded(
                    |chunk_boards, chunk_reserves, batch_size| {
                        call_python_eval_bf16(
                            py,
                            eval_fn,
                            chunk_boards,
                            chunk_reserves,
                            batch_size,
                            cfg.grid_size,
                        )
                    },
                    boards,
                    reserves,
                    actual,
                    target,
                    cfg.grid_size,
                )
            });
            play_selfplay_core(
                cfg.num_games,
                cfg.simulations,
                cfg.max_moves,
                cfg.temperature,
                cfg.temp_threshold,
                cfg.playout_cap_p,
                cfg.fast_cap,
                cfg.forced_playouts,
                cfg.c_puct,
                cfg.dir_alpha,
                cfg.dir_epsilon,
                cfg.leaf_batch_size,
                cfg.resign_threshold,
                cfg.resign_moves,
                cfg.resign_min_moves,
                cfg.calibration_frac,
                cfg.random_opening_moves_min,
                cfg.random_opening_moves_max,
                cfg.skip_timeout_games,
                cfg.use_heuristic,
                cfg.grid_size,
                core_eval,
                progress_core,
                opening_sequences,
            )
        }
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        Ok(into_py_selfplay_result(result))
    }

    #[pyo3(signature = (eval_fn1, eval_fn2, progress_fn=None))]
    fn play_battle(
        &self,
        py: Python<'_>,
        eval_fn1: &Bound<'_, PyAny>,
        eval_fn2: &Bound<'_, PyAny>,
        progress_fn: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyHiveBattleResult> {
        let cfg = &self.config;
        let progress_core = make_battle_progress(py, progress_fn);
        let core_eval1: search::EvalFn<'_> = Box::new(move |boards, reserves, batch_size| {
            call_python_eval_bf16(py, eval_fn1, boards, reserves, batch_size, cfg.grid_size)
        });
        let core_eval2: search::EvalFn<'_> = Box::new(move |boards, reserves, batch_size| {
            call_python_eval_bf16(py, eval_fn2, boards, reserves, batch_size, cfg.grid_size)
        });

        let result = play_battle_core(
            cfg.num_games,
            cfg.simulations,
            cfg.max_moves,
            cfg.c_puct,
            cfg.leaf_batch_size,
            cfg.grid_size,
            core_eval1,
            core_eval2,
            progress_core,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(into_py_battle_result(result))
    }
}

#[pyclass(name = "HiveBattleResult")]
pub struct PyHiveBattleResult {
    wins_model1: u32,
    wins_model2: u32,
    draws: u32,
    game_lengths: Vec<u32>,
}

#[pymethods]
impl PyHiveBattleResult {
    #[getter]
    fn wins_model1(&self) -> u32 { self.wins_model1 }
    #[getter]
    fn wins_model2(&self) -> u32 { self.wins_model2 }
    #[getter]
    fn draws(&self) -> u32 { self.draws }
    #[getter]
    fn game_lengths(&self) -> Vec<u32> { self.game_lengths.clone() }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySelfPlaySession>()?;
    m.add_class::<PySelfPlayResult>()?;
    m.add_class::<PyHiveBattleResult>()?;
    Ok(())
}
