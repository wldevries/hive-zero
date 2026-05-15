//! Yinsh ORT inference engine. Single flat policy output `policy[B, POLICY_SIZE]`
//! and `wdl[B, 3]`. Shared QNN/onnxruntime setup lives in [`crate::ort_common`].

use std::time::{Duration, Instant};

use ort::session::Session;
use ort::value::Tensor;

use crate::ort_common::{
    find_onnxruntime_dylib, find_qnn_htp_dll, prepend_qnn_dir_to_path, register_qnn_plugin,
};

/// ONNX Runtime inference engine for Yinsh.
pub struct YinshOrtEngine {
    session: Session,
    num_channels: usize,
    grid_size: usize,
    reserve_size: usize,
    policy_size: usize,

    // Per-call wall-clock accumulators, drained from outside via `phase_times`.
    // `t_input`: Tensor::from_array allocations + the boards/reserves to_vec copies.
    // `t_run`: session.run (host↔device transfer + GPU compute + ORT marshaling).
    // `t_extract`: try_extract_tensor + the policy/wdl output copies.
    t_input: Duration,
    t_run: Duration,
    t_extract: Duration,
}

impl YinshOrtEngine {
    pub fn load(onnx_path: &str) -> Result<Self, ort::Error> {
        prepend_qnn_dir_to_path();
        ort::init_from(find_onnxruntime_dylib())?.commit();
        register_qnn_plugin();

        let session = Session::builder()?
            .with_execution_providers([
                ort::ep::CUDA::default().build(),
                ort::ep::QNN::default()
                    .with_backend_path(find_qnn_htp_dll())
                    .with_htp_fp16_precision(true)
                    .with_htp_graph_finalization_optimization_mode(3)
                    .build(),
            ])?
            .commit_from_file(onnx_path)?;

        // Pull dimensions from the yinsh_game crate so they cannot drift from
        // the Rust encoding.
        let num_channels = yinsh_game::board_encoding::NUM_CHANNELS;
        let grid_size = yinsh_game::hex::GRID_SIZE;
        let reserve_size = yinsh_game::board_encoding::RESERVE_SIZE;
        let policy_size = yinsh_game::move_encoding::POLICY_SIZE;

        Ok(Self {
            session,
            num_channels,
            grid_size,
            reserve_size,
            policy_size,
            t_input: Duration::ZERO,
            t_run: Duration::ZERO,
            t_extract: Duration::ZERO,
        })
    }

    /// Drain the accumulated per-phase wall-clock times for diagnostics.
    /// Caller is expected to read this once after the self-play session
    /// completes; we don't reset on read since the engine is one-shot.
    pub fn phase_times(&self) -> (Duration, Duration, Duration) {
        (self.t_input, self.t_run, self.t_extract)
    }

    /// Returns `(policy_flat[B*POLICY_SIZE], values[B], draws[B])`.
    /// `values[i] = W-L`, `draws[i] = D` extracted from the WDL softmax output.
    pub fn infer_batch(
        &mut self,
        boards: &[f32],
        reserves: &[f32],
        batch_size: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let t0 = Instant::now();
        let board_tensor = Tensor::from_array((
            [batch_size, self.num_channels, self.grid_size, self.grid_size],
            boards.to_vec(),
        ))
        .map_err(|e| e.to_string())?;
        let reserve_tensor = Tensor::from_array((
            [batch_size, self.reserve_size],
            reserves.to_vec(),
        ))
        .map_err(|e| e.to_string())?;
        self.t_input += t0.elapsed();

        let t1 = Instant::now();
        let outputs = self
            .session
            .run(ort::inputs![
                "board" => board_tensor,
                "reserve" => reserve_tensor,
            ])
            .map_err(|e| e.to_string())?;
        self.t_run += t1.elapsed();

        let t2 = Instant::now();
        let (_, policy_data) = outputs["policy"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let (_, wdl_data) = outputs["wdl"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;

        debug_assert_eq!(policy_data.len(), batch_size * self.policy_size);
        debug_assert_eq!(wdl_data.len(), batch_size * 3);

        let mut values = Vec::with_capacity(batch_size);
        let mut draws = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            values.push(wdl_data[i * 3] - wdl_data[i * 3 + 2]); // W - L
            draws.push(wdl_data[i * 3 + 1]);                      // D
        }

        let result = Ok((policy_data.to_vec(), values, draws));
        self.t_extract += t2.elapsed();
        result
    }
}
