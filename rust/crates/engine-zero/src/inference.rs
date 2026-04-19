//! Rust-native ONNX inference via the `ort` crate, replacing the Python eval callback.

use std::path::PathBuf;

use ort::session::Session;
use ort::value::Tensor;

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
    /// Value per sample: [B]
    pub value: Vec<f32>,
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
        self.infer(boards.to_vec(), reserves.to_vec(), batch_size, num_channels, grid_size, reserve_size)
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
        Ok(Self { session })
    }

    /// Run inference on a batch of boards and reserves.
    ///
    /// - `boards`: f32 data, shape [B, NUM_CHANNELS, grid_size, grid_size] flattened
    /// - `reserves`: f32 data, shape [B, RESERVE_SIZE] flattened
    /// - `batch_size`: B
    /// - `num_channels`, `grid_size`, `reserve_size`: tensor dimensions
    pub fn infer(
        &mut self,
        boards: Vec<f32>,
        reserves: Vec<f32>,
        batch_size: usize,
        num_channels: usize,
        grid_size: usize,
        reserve_size: usize,
    ) -> Result<HiveInferenceResult, ort::Error> {
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

        let (_, policy_data) = outputs["policy"].try_extract_tensor::<f32>()?;
        let (_, value_data) = outputs["value"].try_extract_tensor::<f32>()?;

        Ok(HiveInferenceResult {
            policy: policy_data.to_vec(),
            value: value_data.to_vec(),
        })
    }
}

/// ONNX Runtime inference engine for Yinsh.
/// Single flat policy output `policy[B, POLICY_SIZE]` and `value[B, 1]`.
pub struct YinshOrtEngine {
    session: Session,
    num_channels: usize,
    grid_size: usize,
    reserve_size: usize,
    policy_size: usize,
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
        })
    }

    /// Returns `(policy_flat[B*POLICY_SIZE], value_flat[B])`.
    pub fn infer_batch(
        &mut self,
        boards: &[f32],
        reserves: &[f32],
        batch_size: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
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

        let outputs = self
            .session
            .run(ort::inputs![
                "board" => board_tensor,
                "reserve" => reserve_tensor,
            ])
            .map_err(|e| e.to_string())?;

        let (_, policy_data) = outputs["policy"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let (_, value_data) = outputs["value"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;

        debug_assert_eq!(policy_data.len(), batch_size * self.policy_size);
        debug_assert_eq!(value_data.len(), batch_size);

        Ok((policy_data.to_vec(), value_data.to_vec()))
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
