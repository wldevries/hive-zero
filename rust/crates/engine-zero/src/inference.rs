//! Rust-native ONNX inference via the `ort` crate, replacing the Python eval callback.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ort::session::Session;
use ort::value::{Tensor, TensorRef};

// ---------------------------------------------------------------------------
// Inference traits (game-specific)
// ---------------------------------------------------------------------------

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

/// Result of a batch inference call for Hive.
pub struct HiveInferenceResult {
    /// Flattened policy logits: [B * policy_size]
    pub policy: Vec<f32>,
    /// WDL probabilities: [B * 3], layout per sample: [P(win), P(draw), P(loss)]
    pub wdl: Vec<f32>,
}

/// Result of a batch inference call for Zertz.
pub struct ZertzInferenceResult {
    /// Flat policy logits: [B * NN_POLICY_SIZE] = [B * 490]
    /// Layout per sample: place[4*49] || cap_dir[6*49]
    pub policy: Vec<f32>,
    /// Value per sample: [B]
    pub value: Vec<f32>,
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

/// Ensure the directory containing QnnHtp.dll is on PATH so ORT can find
/// onnxruntime_providers_qnn.dll. Needed for the 2.x unbundled layout where
/// QNN DLLs live in a separate onnxruntime_qnn/ directory.
fn prepend_qnn_dir_to_path() {
    let htp = find_qnn_htp_dll();
    if let Some(dir) = std::path::Path::new(&htp).parent() {
        let dir_str = dir.to_string_lossy();
        let current = std::env::var("PATH").unwrap_or_default();
        if !current.contains(dir_str.as_ref()) {
            // SAFETY: single-threaded startup, before any threads are spawned
            unsafe { std::env::set_var("PATH", format!("{};{}", dir_str, current)); }
        }
    }
}

fn find_onnxruntime_dylib() -> PathBuf {
    PathBuf::from(r".venv\Lib\site-packages\onnxruntime\capi\onnxruntime.dll")
}

fn find_qnn_htp_dll() -> String {
    // onnxruntime-qnn <=1.x bundles DLLs into onnxruntime/capi/ (alongside onnxruntime.dll).
    // onnxruntime-qnn 2.x+ puts them in a separate onnxruntime_qnn/ directory.
    let bundled = std::path::Path::new(r".venv\Lib\site-packages\onnxruntime\capi\QnnHtp.dll");
    let unbundled = std::path::Path::new(r".venv\Lib\site-packages\onnxruntime_qnn\QnnHtp.dll");
    let p = if bundled.exists() { bundled } else { unbundled };
    p.canonicalize()
        .unwrap_or_else(|_| p.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// onnxruntime-qnn 2.x ships the QNN EP as a standalone plugin library that must be registered
/// with the ORT environment before sessions can use it.
fn find_qnn_provider_dll() -> PathBuf {
    let p = std::path::Path::new(r".venv\Lib\site-packages\onnxruntime_qnn\onnxruntime_providers_qnn.dll");
    p.canonicalize().unwrap_or_else(|_| p.to_path_buf())
}

/// Register the QNN plugin EP library with the current ORT environment (onnxruntime-qnn ≥ 2.0).
/// Silently skips if the plugin DLL is absent or registration fails (e.g. no QNN hardware).
fn register_qnn_plugin() {
    let provider_dll = find_qnn_provider_dll();
    if !provider_dll.exists() {
        return;
    }
    if let Ok(env) = ort::environment::Environment::current() {
        let _ = env.register_ep_library("QNN", provider_dll);
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

/// ONNX Runtime inference engine for Yinsh.
/// Single flat policy output `policy[B, POLICY_SIZE]` and `wdl[B, 3]`.
pub struct YinshOrtEngine {
    session: Session,
    num_channels: usize,
    grid_size: usize,
    reserve_size: usize,
    policy_size: usize,

    // Per-call wall-clock accumulators, drained from outside via `take_phase_times`.
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

/// ONNX Runtime inference engine for Zertz.
pub struct ZertzOrtEngine {
    session: Session,
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
        self.infer(boards.to_vec(), reserves.to_vec(), batch_size, num_channels, grid_size, reserve_size)
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
        Ok(Self { session })
    }

    /// Run inference on a batch of boards and reserves.
    ///
    /// - `boards`: f32 data, shape [B, NUM_CHANNELS, GRID_SIZE, GRID_SIZE] flattened
    /// - `reserves`: f32 data, shape [B, RESERVE_SIZE] flattened
    /// - `batch_size`: B
    pub fn infer(
        &mut self,
        boards: Vec<f32>,
        reserves: Vec<f32>,
        batch_size: usize,
        num_channels: usize,
        grid_size: usize,
        reserve_size: usize,
    ) -> Result<ZertzInferenceResult, ort::Error> {
        let board_tensor = Tensor::from_array((
            [batch_size, num_channels, grid_size, grid_size],
            boards,
        ))?;
        let reserve_tensor = Tensor::from_array((
            [batch_size, reserve_size],
            reserves,
        ))?;

        let outputs = self.session.run(ort::inputs![
            "board" => board_tensor,
            "reserve" => reserve_tensor,
        ])?;

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

        Ok(ZertzInferenceResult {
            policy,
            value: value_data.to_vec(),
        })
    }
}
