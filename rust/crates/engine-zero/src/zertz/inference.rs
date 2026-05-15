//! Zertz ORT inference: trait + engine. Outputs are factorized as
//! `place[B, 4, 7, 7]` and `cap_dir[B, 6, 7, 7]`; the engine concatenates
//! per sample to a flat policy of length 490 to match the trained head
//! layout. Shared QNN/onnxruntime setup lives in [`crate::ort_common`].

use ort::session::Session;
use ort::value::Tensor;

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
