//! Yinsh ORT inference engine. Single flat policy output `policy[B, POLICY_SIZE]`
//! and `wdl[B, 3]`. Shared QNN/onnxruntime setup lives in [`crate::ort_common`].

use std::sync::{Arc, Mutex};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ort::session::Session;
use ort::value::TensorRef;

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
    // `t_input`: TensorRef::from_array_view setup (borrowed slices, no host copy).
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
    ///
    /// Uses `TensorRef::from_array_view` so the input slices are borrowed
    /// rather than copied into an owned `Vec<f32>` — the only host→device
    /// copy is the one ORT performs inside `session.run` itself.
    pub fn infer_batch(
        &mut self,
        boards: &[f32],
        reserves: &[f32],
        batch_size: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let t0 = Instant::now();
        let board_tensor = TensorRef::from_array_view((
            [batch_size, self.num_channels, self.grid_size, self.grid_size],
            boards,
        ))
        .map_err(|e| e.to_string())?;
        let reserve_tensor = TensorRef::from_array_view((
            [batch_size, self.reserve_size],
            reserves,
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

// ---------------------------------------------------------------------------
// Yinsh ORT worker thread (for pipelined self-play)
// ---------------------------------------------------------------------------

/// One request to the worker. `respond` is a single-use channel; the worker
/// posts the result and the caller blocks on `recv()` when ready to consume.
struct YinshWorkerRequest {
    boards: Vec<f32>,
    reserves: Vec<f32>,
    batch_size: usize,
    respond: SyncSender<Result<(Vec<f32>, Vec<f32>, Vec<f32>), String>>,
}

/// Per-phase wall-clock totals drained from the worker after it shuts down.
#[derive(Clone, Copy, Default)]
pub struct YinshWorkerTiming {
    pub t_input: Duration,
    pub t_run: Duration,
    pub t_extract: Duration,
}

/// Dedicated worker thread that owns a `YinshOrtEngine` and processes
/// inference requests off an mpsc channel. Lets `play_selfplay_core` submit
/// a batch and continue selecting leaves for the next batch while the GPU
/// runs the current one (one batch in flight at a time).
///
/// Lifecycle: spawn → submit/recv pairs → `shutdown_and_drain_timing()` to
/// close the channel, join the thread, and read its `phase_times`. Dropping
/// without explicit shutdown also closes the channel (sender is dropped) and
/// detaches the thread, but the per-phase timing will be lost.
pub struct YinshInferenceWorker {
    request_tx: SyncSender<YinshWorkerRequest>,
    handle: Option<JoinHandle<()>>,
    timing: Arc<Mutex<YinshWorkerTiming>>,
}

impl YinshInferenceWorker {
    /// Spawn the worker. The ONNX session is constructed on the worker
    /// thread so it doesn't migrate after init (CUDA EP can be picky about
    /// thread-affinity for the bound stream).
    pub fn spawn(onnx_path: String) -> Result<Self, String> {
        // Bounded queue depth = 2: at most one batch waiting in the channel
        // plus one being processed by the worker. send() will block past
        // that, which acts as backpressure if the caller submits faster than
        // the worker can drain.
        let (request_tx, request_rx) = sync_channel::<YinshWorkerRequest>(2);
        let timing = Arc::new(Mutex::new(YinshWorkerTiming::default()));
        let timing_for_worker = Arc::clone(&timing);

        // Load on the worker thread so session ownership stays there. Surface
        // load failures via an init channel before returning the handle.
        let (init_tx, init_rx) = sync_channel::<Result<(), String>>(1);
        let onnx_path_for_thread = onnx_path.clone();
        let handle = thread::Builder::new()
            .name("yinsh-ort-worker".into())
            .spawn(move || {
                let mut engine = match YinshOrtEngine::load(&onnx_path_for_thread) {
                    Ok(e) => {
                        let _ = init_tx.send(Ok(()));
                        e
                    }
                    Err(e) => {
                        let _ = init_tx.send(Err(e.to_string()));
                        return;
                    }
                };

                while let Ok(req) = request_rx.recv() {
                    let result = engine.infer_batch(&req.boards, &req.reserves, req.batch_size);
                    let _ = req.respond.send(result);
                }

                // Channel closed → drain accumulated phase timers so the
                // outer caller can read them after `join`.
                let (t_input, t_run, t_extract) = engine.phase_times();
                if let Ok(mut t) = timing_for_worker.lock() {
                    *t = YinshWorkerTiming { t_input, t_run, t_extract };
                }
            })
            .map_err(|e| format!("spawn yinsh-ort-worker: {e}"))?;

        match init_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                request_tx,
                handle: Some(handle),
                timing,
            }),
            Ok(Err(e)) => {
                let _ = handle.join();
                Err(e)
            }
            Err(_) => Err("worker thread died before load completed".into()),
        }
    }

    /// Post one inference request. Returns a receiver the caller can block
    /// on later via `recv()`. Sends are bounded by the channel capacity, so
    /// this can block if the worker is more than one batch behind.
    pub fn submit(
        &self,
        boards: Vec<f32>,
        reserves: Vec<f32>,
        batch_size: usize,
    ) -> Result<Receiver<Result<(Vec<f32>, Vec<f32>, Vec<f32>), String>>, String> {
        let (resp_tx, resp_rx) = sync_channel(1);
        self.request_tx
            .send(YinshWorkerRequest { boards, reserves, batch_size, respond: resp_tx })
            .map_err(|_| "yinsh-ort-worker request channel closed".to_string())?;
        Ok(resp_rx)
    }

    /// Close the request channel, join the worker, and return the drained
    /// per-phase timing. Consumes self so accidental further use is a
    /// compile error.
    pub fn shutdown_and_drain_timing(mut self) -> YinshWorkerTiming {
        let (dummy_tx, _) = sync_channel::<YinshWorkerRequest>(0);
        drop(std::mem::replace(&mut self.request_tx, dummy_tx));
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.timing.lock().map(|t| *t).unwrap_or_default()
    }
}

impl Drop for YinshInferenceWorker {
    fn drop(&mut self) {
        // Best-effort cleanup if the caller forgot shutdown_and_drain_timing.
        if let Some(handle) = self.handle.take() {
            let (dummy_tx, _) = sync_channel::<YinshWorkerRequest>(0);
            drop(std::mem::replace(&mut self.request_tx, dummy_tx));
            let _ = handle.join();
        }
    }
}
