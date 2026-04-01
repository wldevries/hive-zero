//! Rust-native ONNX inference via the `ort` crate, replacing the Python eval callback.

use std::path::{Path, PathBuf};

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
    /// Place logits: [B * PLACE_HEAD_SIZE]
    pub place: Vec<f32>,
    /// Capture source logits: [B * CAP_HEAD_SIZE]
    pub cap_source: Vec<f32>,
    /// Capture dest logits: [B * CAP_HEAD_SIZE]
    pub cap_dest: Vec<f32>,
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

fn find_onnxruntime_dylib() -> PathBuf {
    // Find our custom ONNX Runtime dylib path somehow (i.e. resolving it from the root of our program's install folder)
    // The path should point to the `libonnxruntime` binary, which looks like:
    // - on Windows: C:\Program Files\...\onnxruntime.dll
    // - on Linux: /etc/.../libonnxruntime.so
    // - on macOS: /.../libonnxruntime.dylib
    PathBuf::from(r".venv\Lib\site-packages\onnxruntime\capi\onnxruntime.dll")
}
 

impl HiveOrtEngine {
    pub fn load(onnx_path: &str) -> Result<Self, ort::Error> {
        let dylib_path = find_onnxruntime_dylib();
 
        // Initialize ort with the path to the dylib. This **must** be called before any other usage of `ort`!
        // `init_from` returns a `Result<EnvironmentBuilder>` which you can use to further configure the environment
        // before `.commit()`ing; see the Environment docs for more information on what you can configure.
        // `init_from` will return an `Err` if it fails to load the dylib.
        ort::init_from(dylib_path)?.commit();

        let session = Session::builder()?
            .with_execution_providers([
                ort::ep::CUDA::default().build().error_on_failure()
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
        let dylib_path = find_onnxruntime_dylib();
 
        // Initialize ort with the path to the dylib. This **must** be called before any other usage of `ort`!
        // `init_from` returns a `Result<EnvironmentBuilder>` which you can use to further configure the environment
        // before `.commit()`ing; see the Environment docs for more information on what you can configure.
        // `init_from` will return an `Err` if it fails to load the dylib.
        ort::init_from(dylib_path)?.commit();

        let session = Session::builder()?
            .with_execution_providers([ort::ep::CUDA::default().build()])?
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
        let (_, cap_source_data) = outputs["cap_source"].try_extract_tensor::<f32>()?;
        let (_, cap_dest_data) = outputs["cap_dest"].try_extract_tensor::<f32>()?;
        let (_, value_data) = outputs["value"].try_extract_tensor::<f32>()?;

        Ok(ZertzInferenceResult {
            place: place_data.to_vec(),
            cap_source: cap_source_data.to_vec(),
            cap_dest: cap_dest_data.to_vec(),
            value: value_data.to_vec(),
        })
    }
}
