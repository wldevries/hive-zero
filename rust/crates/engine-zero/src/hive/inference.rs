//! Hive ORT inference: trait, single-threaded engine, and dedicated worker
//! thread for pipelined self-play. Splits out from the former monolithic
//! `inference.rs`; shared QNN/onnxruntime setup lives in [`crate::ort_common`].

use std::sync::{Arc, Mutex};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ort::session::Session;
use ort::value::TensorRef;

use hive_game::tokenize::{TokenBatch, BILINEAR_DIM, F_FLAGS, SEQ_LEN};

use crate::ort_common::{
    find_onnxruntime_dylib, find_qnn_htp_dll, prepend_qnn_dir_to_path, register_qnn_plugin,
};

/// Game-agnostic inference interface for Hive. Implementations may use ORT,
/// tract (WASM), or any other backend. No `Send` bound — Python-backed impls
/// hold GIL tokens and cannot be `Send`.
pub trait HiveInference {
    fn infer_batch(
        &mut self,
        boards: &[f32],
        reserves: &[f32],
        batch_size: usize,
        num_channels: usize,
        grid_size: usize,
        reserve_size: usize,
    ) -> Result<HiveInferenceResult, Box<dyn std::error::Error + Send + Sync>>;
}

/// Result of a batch inference call for Hive.
pub struct HiveInferenceResult {
    /// Flattened policy logits: [B * policy_size]
    pub policy: Vec<f32>,
    /// WDL probabilities: [B * 3], layout per sample: [P(win), P(draw), P(loss)]
    pub wdl: Vec<f32>,
}

/// ONNX Runtime inference engine for Hive.
pub struct HiveOrtEngine {
    session: Session,

    // Per-call wall-clock accumulators, drained from outside via `phase_times`.
    // `t_input`: Tensor::from_array allocations + the boards/reserves to_vec copies.
    // `t_run`: session.run (host↔device transfer + GPU compute + ORT marshaling).
    // `t_extract`: try_extract_tensor + the policy/wdl output copies.
    t_input: Duration,
    t_run: Duration,
    t_extract: Duration,
}

impl HiveInference for HiveOrtEngine {
    fn infer_batch(
        &mut self,
        boards: &[f32],
        reserves: &[f32],
        batch_size: usize,
        num_channels: usize,
        grid_size: usize,
        reserve_size: usize,
    ) -> Result<HiveInferenceResult, Box<dyn std::error::Error + Send + Sync>> {
        self.infer(boards, reserves, batch_size, num_channels, grid_size, reserve_size)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }
}

impl HiveOrtEngine {
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
    /// completes.
    pub fn phase_times(&self) -> (Duration, Duration, Duration) {
        (self.t_input, self.t_run, self.t_extract)
    }

    /// Run inference on a batch of boards and reserves.
    ///
    /// - `boards`: f32 data, shape [B, NUM_CHANNELS, grid_size, grid_size] flattened
    /// - `reserves`: f32 data, shape [B, RESERVE_SIZE] flattened
    /// - `batch_size`: B
    /// - `num_channels`, `grid_size`, `reserve_size`: tensor dimensions
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
    ) -> Result<HiveInferenceResult, ort::Error> {
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
        let (_, policy_data) = outputs["policy"].try_extract_tensor::<f32>()?;
        let (_, wdl_data) = outputs["wdl"].try_extract_tensor::<f32>()?;

        let result = HiveInferenceResult {
            policy: policy_data.to_vec(),
            wdl: wdl_data.to_vec(),
        };
        self.t_extract += t2.elapsed();
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Hive ORT worker thread (for pipelined self-play)
// ---------------------------------------------------------------------------

/// One request to the worker. `respond` is a single-use channel; the worker
/// posts the result and the caller blocks on `recv()` when ready to consume.
struct HiveWorkerRequest {
    boards: Vec<f32>,
    reserves: Vec<f32>,
    batch_size: usize,
    respond: SyncSender<Result<HiveInferenceResult, String>>,
}

/// Per-phase wall-clock totals drained from the worker after it shuts down.
#[derive(Clone, Copy, Default)]
pub struct HiveWorkerTiming {
    pub t_input: Duration,
    pub t_run: Duration,
    pub t_extract: Duration,
}

/// Dedicated worker thread that owns a `HiveOrtEngine` and processes
/// inference requests off an mpsc channel. Lets `play_selfplay_core` submit
/// a batch and continue selecting leaves for the next batch while the GPU
/// runs the current one (one batch in flight).
///
/// Lifecycle: spawn → submit/recv pairs → `shutdown_and_drain_timing()` to
/// close the channel, join the thread, and read its `phase_times`. Dropping
/// without explicit shutdown also closes the channel (sender is dropped) and
/// detaches the thread, but the per-phase timing will be lost.
pub struct HiveInferenceWorker {
    request_tx: SyncSender<HiveWorkerRequest>,
    handle: Option<JoinHandle<()>>,
    timing: Arc<Mutex<HiveWorkerTiming>>,
    num_channels: usize,
    grid_size: usize,
    reserve_size: usize,
}

impl HiveInferenceWorker {
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
        let (request_tx, request_rx) = sync_channel::<HiveWorkerRequest>(2);
        let timing = Arc::new(Mutex::new(HiveWorkerTiming::default()));
        let timing_for_worker = Arc::clone(&timing);

        // Spawn first, load engine inside the thread so session ownership
        // stays on the worker. If load fails we surface the error via the
        // first request's respond channel — we can't propagate it eagerly
        // without leaking the spawn.
        //
        // To keep the spawn API fallible-on-load, we do load + initial send
        // synchronously on a barrier: spawn the thread, have it try to load,
        // then signal success/failure via an init channel.
        let (init_tx, init_rx) = sync_channel::<Result<(), String>>(1);
        let onnx_path_for_thread = onnx_path.clone();
        let handle = thread::Builder::new()
            .name("hive-ort-worker".into())
            .spawn(move || {
                let mut engine = match HiveOrtEngine::load(&onnx_path_for_thread) {
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
                    *t = HiveWorkerTiming { t_input, t_run, t_extract };
                }
            })
            .map_err(|e| format!("spawn hive-ort-worker: {e}"))?;

        // Wait for load to either succeed or fail.
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
    ) -> Result<Receiver<Result<HiveInferenceResult, String>>, String> {
        let (resp_tx, resp_rx) = sync_channel(1);
        self.request_tx
            .send(HiveWorkerRequest { boards, reserves, batch_size, respond: resp_tx })
            .map_err(|_| "hive-ort-worker request channel closed".to_string())?;
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
    pub fn shutdown_and_drain_timing(mut self) -> HiveWorkerTiming {
        // Replace the sender with a fresh closed channel so the worker's
        // recv() sees disconnection; the original sender is dropped here.
        let (dummy_tx, _) = sync_channel::<HiveWorkerRequest>(0);
        drop(std::mem::replace(&mut self.request_tx, dummy_tx));
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.timing.lock().map(|t| *t).unwrap_or_default()
    }
}

impl Drop for HiveInferenceWorker {
    fn drop(&mut self) {
        // Best-effort cleanup if the caller forgot to call
        // `shutdown_and_drain_timing`: replace the sender so the worker's
        // recv() sees the channel close, then join. Timing is lost.
        if let Some(handle) = self.handle.take() {
            let (dummy_tx, _) = sync_channel::<HiveWorkerRequest>(0);
            drop(std::mem::replace(&mut self.request_tx, dummy_tx));
            let _ = handle.join();
        }
    }
}

// ===========================================================================
// Token-based ORT engine for the HiveTransformer architecture
// ===========================================================================

/// ONNX Runtime inference engine for the token-based Hive transformer.
/// Accepts a batch of `TokenBatch`es, returns the per-sample
/// (policy_flat, value, draw) tuple in the same shape MCTS' expand_and_backprop
/// consumes. Independent of the CNN-side `HiveOrtEngine` so both can coexist
/// during the transition.
pub struct HiveTokenOrtEngine {
    session: Session,
    t_input: Duration,
    t_run: Duration,
    t_extract: Duration,
}

impl HiveTokenOrtEngine {
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

    pub fn phase_times(&self) -> (Duration, Duration, Duration) {
        (self.t_input, self.t_run, self.t_extract)
    }

    /// Run inference on a batch of token batches.
    ///
    /// Returns `(policies, values, draws)` where:
    ///   - `policies[i]`: flat policy tensor for leaf `i` (length `2*L*D`),
    ///     `[Q || K]` D-major, exactly what `PolicyIndex::DotProduct` reads.
    ///   - `values[i]`: W − L (zero-sum) — already split from WDL.
    ///   - `draws[i]`: D probability (symmetric).
    pub fn infer_token_batches(
        &mut self,
        batches: &[TokenBatch],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>, Vec<f32>), ort::Error> {
        let b = batches.len();
        let l = SEQ_LEN;
        let f = F_FLAGS;

        // ---- pack inputs into contiguous buffers in the right dtype --------
        let t0 = Instant::now();
        let mut cat = vec![0i64; b * l * 5];
        let mut pos = vec![0i64; b * l * 2];
        let mut flg = vec![0f32; b * l * f];
        let mut msk = vec![false; b * l];
        for (bi, tok) in batches.iter().enumerate() {
            let cat_base = bi * l * 5;
            let pos_base = bi * l * 2;
            let flg_base = bi * l * f;
            let msk_base = bi * l;
            for li in 0..l {
                cat[cat_base + li * 5 + 0] = tok.kind[li] as i64;
                cat[cat_base + li * 5 + 1] = tok.piece_type[li] as i64;
                cat[cat_base + li * 5 + 2] = tok.color[li] as i64;
                cat[cat_base + li * 5 + 3] = tok.z[li] as i64;
                cat[cat_base + li * 5 + 4] = tok.count[li] as i64;
                pos[pos_base + li * 2 + 0] = tok.q[li] as i64;
                pos[pos_base + li * 2 + 1] = tok.r[li] as i64;
            }
            flg[flg_base..flg_base + l * f].copy_from_slice(&tok.flags);
            msk[msk_base..msk_base + l].copy_from_slice(&tok.mask);
        }

        let cat_t = TensorRef::from_array_view(([b, l, 5usize], cat.as_slice()))?;
        let pos_t = TensorRef::from_array_view(([b, l, 2usize], pos.as_slice()))?;
        let flg_t = TensorRef::from_array_view(([b, l, f], flg.as_slice()))?;
        let msk_t = TensorRef::from_array_view(([b, l], msk.as_slice()))?;
        self.t_input += t0.elapsed();

        // ---- run ORT session ----------------------------------------------
        let t1 = Instant::now();
        let outputs = self.session.run(ort::inputs![
            "categoricals" => cat_t,
            "positions"    => pos_t,
            "flags"        => flg_t,
            "mask"         => msk_t,
        ])?;
        self.t_run += t1.elapsed();

        // ---- extract policy + wdl -----------------------------------------
        let t2 = Instant::now();
        let (_, policy_data) = outputs["policy"].try_extract_tensor::<f32>()?;
        let (_, wdl_data)    = outputs["wdl"].try_extract_tensor::<f32>()?;

        let pol_per = 2 * SEQ_LEN * BILINEAR_DIM;
        let mut policies = Vec::with_capacity(b);
        let mut values   = Vec::with_capacity(b);
        let mut draws    = Vec::with_capacity(b);
        for i in 0..b {
            let start = i * pol_per;
            policies.push(policy_data[start..start + pol_per].to_vec());
            // WDL layout is [P(win), P(draw), P(loss)] per sample.
            values.push(wdl_data[i * 3] - wdl_data[i * 3 + 2]);
            draws.push(wdl_data[i * 3 + 1]);
        }
        self.t_extract += t2.elapsed();
        Ok((policies, values, draws))
    }
}
