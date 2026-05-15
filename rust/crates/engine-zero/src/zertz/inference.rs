//! Zertz ORT inference: trait, single-threaded engine, and dedicated worker
//! thread for pipelined self-play. Outputs are factorized as
//! `place[B, 4, 7, 7]` and `cap_dir[B, 6, 7, 7]`; the engine concatenates
//! per sample to a flat policy of length 490 to match the trained head
//! layout. Shared QNN/onnxruntime setup lives in [`crate::ort_common`].

use std::sync::{Arc, Mutex};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ort::session::Session;
use ort::value::TensorRef;

use crate::ort_common::{
    find_onnxruntime_dylib, find_qnn_htp_dll, prepend_qnn_dir_to_path, register_qnn_plugin,
};

/// Game-agnostic inference interface for Zertz. Implementations may use ORT,
/// tract (WASM), or any other backend. No `Send` bound — Python-backed impls
/// hold GIL tokens and cannot be `Send`.
pub trait ZertzInference {
    fn infer_batch(
        &mut self,
        boards: &[f32],
        reserves: &[f32],
        batch_size: usize,
        num_channels: usize,
        grid_size: usize,
        reserve_size: usize,
    ) -> Result<ZertzInferenceResult, Box<dyn std::error::Error + Send + Sync>>;
}

/// Result of a batch inference call for Zertz.
pub struct ZertzInferenceResult {
    /// Flat policy logits: [B * NN_POLICY_SIZE] = [B * 490]
    /// Layout per sample: place[4*49] || cap_dir[6*49]
    pub policy: Vec<f32>,
    /// Value per sample: [B]
    pub value: Vec<f32>,
}

/// ONNX Runtime inference engine for Zertz.
pub struct ZertzOrtEngine {
    session: Session,

    // Per-call wall-clock accumulators, drained from outside via `phase_times`.
    // `t_input`: TensorRef::from_array_view setup (borrowed slices, no host copy).
    // `t_run`: session.run (host↔device transfer + GPU compute + ORT marshaling).
    // `t_extract`: try_extract_tensor + the place/cap_dir/value output copies.
    t_input: Duration,
    t_run: Duration,
    t_extract: Duration,
}

impl ZertzInference for ZertzOrtEngine {
    fn infer_batch(
        &mut self,
        boards: &[f32],
        reserves: &[f32],
        batch_size: usize,
        num_channels: usize,
        grid_size: usize,
        reserve_size: usize,
    ) -> Result<ZertzInferenceResult, Box<dyn std::error::Error + Send + Sync>> {
        self.infer(boards, reserves, batch_size, num_channels, grid_size, reserve_size)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }
}

impl ZertzOrtEngine {
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
        Ok(Self {
            session,
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

    /// Run inference on a batch of boards and reserves.
    ///
    /// - `boards`: f32 data, shape [B, NUM_CHANNELS, GRID_SIZE, GRID_SIZE] flattened
    /// - `reserves`: f32 data, shape [B, RESERVE_SIZE] flattened
    /// - `batch_size`: B
    ///
    /// Uses `TensorRef::from_array_view` so the input slices are borrowed
    /// rather than copied into an owned `Vec<f32>` — the only host→device
    /// copy is the one ORT performs inside `session.run` itself.
    pub fn infer(
        &mut self,
        boards: &[f32],
        reserves: &[f32],
        batch_size: usize,
        num_channels: usize,
        grid_size: usize,
        reserve_size: usize,
    ) -> Result<ZertzInferenceResult, ort::Error> {
        let t0 = Instant::now();
        let board_tensor = TensorRef::from_array_view((
            [batch_size, num_channels, grid_size, grid_size],
            boards,
        ))?;
        let reserve_tensor = TensorRef::from_array_view((
            [batch_size, reserve_size],
            reserves,
        ))?;
        self.t_input += t0.elapsed();

        let t1 = Instant::now();
        let outputs = self.session.run(ort::inputs![
            "board" => board_tensor,
            "reserve" => reserve_tensor,
        ])?;
        self.t_run += t1.elapsed();

        let t2 = Instant::now();
        let (_, place_data) = outputs["place"].try_extract_tensor::<f32>()?;
        let (_, cap_dir_data) = outputs["cap_dir"].try_extract_tensor::<f32>()?;
        let (_, value_data) = outputs["value"].try_extract_tensor::<f32>()?;

        // Concatenate per-sample: [place(196), cap_dir(294)] = flat 490
        let place_per = place_data.len() / batch_size;
        let cap_per = cap_dir_data.len() / batch_size;
        let mut policy = Vec::with_capacity(batch_size * (place_per + cap_per));
        for i in 0..batch_size {
            policy.extend_from_slice(&place_data[i * place_per..(i + 1) * place_per]);
            policy.extend_from_slice(&cap_dir_data[i * cap_per..(i + 1) * cap_per]);
        }

        let result = ZertzInferenceResult {
            policy,
            value: value_data.to_vec(),
        };
        self.t_extract += t2.elapsed();
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Zertz ORT worker thread (for pipelined self-play)
// ---------------------------------------------------------------------------

/// One request to the worker. `respond` is a single-use channel; the worker
/// posts the result and the caller blocks on `recv()` when ready to consume.
struct ZertzWorkerRequest {
    boards: Vec<f32>,
    reserves: Vec<f32>,
    batch_size: usize,
    respond: SyncSender<Result<ZertzInferenceResult, String>>,
}

/// Per-phase wall-clock totals drained from the worker after it shuts down.
#[derive(Clone, Copy, Default)]
pub struct ZertzWorkerTiming {
    pub t_input: Duration,
    pub t_run: Duration,
    pub t_extract: Duration,
}

/// Dedicated worker thread that owns a `ZertzOrtEngine` and processes
/// inference requests off an mpsc channel. Lets `play_selfplay_core` submit
/// a batch and continue selecting leaves for the next batch while the GPU
/// runs the current one (one batch in flight at a time).
///
/// Lifecycle: spawn → submit/recv pairs → `shutdown_and_drain_timing()` to
/// close the channel, join the thread, and read its `phase_times`. Dropping
/// without explicit shutdown also closes the channel (sender is dropped) and
/// detaches the thread, but the per-phase timing will be lost.
pub struct ZertzInferenceWorker {
    request_tx: SyncSender<ZertzWorkerRequest>,
    handle: Option<JoinHandle<()>>,
    timing: Arc<Mutex<ZertzWorkerTiming>>,
    num_channels: usize,
    grid_size: usize,
    reserve_size: usize,
}

impl ZertzInferenceWorker {
    /// Spawn the worker. The ONNX session is constructed on the worker
    /// thread so it doesn't migrate after init (CUDA EP can be picky about
    /// thread-affinity for the bound stream).
    pub fn spawn(
        onnx_path: String,
        num_channels: usize,
        grid_size: usize,
        reserve_size: usize,
    ) -> Result<Self, String> {
        // Bounded queue depth = 2: at most one batch waiting in the channel
        // plus one being processed by the worker. send() will block past
        // that, which acts as backpressure if the caller submits faster than
        // the worker can drain.
        let (request_tx, request_rx) = sync_channel::<ZertzWorkerRequest>(2);
        let timing = Arc::new(Mutex::new(ZertzWorkerTiming::default()));
        let timing_for_worker = Arc::clone(&timing);

        // Load on the worker thread so session ownership stays there. Surface
        // load failures via an init channel before returning the handle.
        let (init_tx, init_rx) = sync_channel::<Result<(), String>>(1);
        let onnx_path_for_thread = onnx_path.clone();
        let handle = thread::Builder::new()
            .name("zertz-ort-worker".into())
            .spawn(move || {
                let mut engine = match ZertzOrtEngine::load(&onnx_path_for_thread) {
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
                    let result = engine
                        .infer(
                            &req.boards,
                            &req.reserves,
                            req.batch_size,
                            num_channels,
                            grid_size,
                            reserve_size,
                        )
                        .map_err(|e| e.to_string());
                    let _ = req.respond.send(result);
                }

                // Channel closed → drain accumulated phase timers so the
                // outer caller can read them after `join`.
                let (t_input, t_run, t_extract) = engine.phase_times();
                if let Ok(mut t) = timing_for_worker.lock() {
                    *t = ZertzWorkerTiming { t_input, t_run, t_extract };
                }
            })
            .map_err(|e| format!("spawn zertz-ort-worker: {e}"))?;

        match init_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                request_tx,
                handle: Some(handle),
                timing,
                num_channels,
                grid_size,
                reserve_size,
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
    ) -> Result<Receiver<Result<ZertzInferenceResult, String>>, String> {
        let (resp_tx, resp_rx) = sync_channel(1);
        self.request_tx
            .send(ZertzWorkerRequest { boards, reserves, batch_size, respond: resp_tx })
            .map_err(|_| "zertz-ort-worker request channel closed".to_string())?;
        Ok(resp_rx)
    }

    /// Dimensions baked in at spawn time, exposed for callers that build
    /// their own input buffers (kept in sync with the trained model).
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.num_channels, self.grid_size, self.reserve_size)
    }

    /// Close the request channel, join the worker, and return the drained
    /// per-phase timing. Consumes self so accidental further use is a
    /// compile error.
    pub fn shutdown_and_drain_timing(mut self) -> ZertzWorkerTiming {
        let (dummy_tx, _) = sync_channel::<ZertzWorkerRequest>(0);
        drop(std::mem::replace(&mut self.request_tx, dummy_tx));
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.timing.lock().map(|t| *t).unwrap_or_default()
    }
}

impl Drop for ZertzInferenceWorker {
    fn drop(&mut self) {
        // Best-effort cleanup if the caller forgot shutdown_and_drain_timing.
        if let Some(handle) = self.handle.take() {
            let (dummy_tx, _) = sync_channel::<ZertzWorkerRequest>(0);
            drop(std::mem::replace(&mut self.request_tx, dummy_tx));
            let _ = handle.join();
        }
    }
}
