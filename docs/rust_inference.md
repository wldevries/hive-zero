# Plan: Rust-Native GPU Inference for Self-Play Training

## Context
Self-play training is bottlenecked by the Python↔Rust bridge for NN inference. Every MCTS simulation batch crosses from Rust → Python (PyO3) → PyTorch GPU → Python → Rust. This incurs:
- **PyO3 call overhead**: GIL acquisition, numpy array construction/extraction per batch
- **Small batch inefficiency**: with 20-32 parallel games and `play_batch_size=2`, batches are only 40-64 samples — GPU underutilized
- **Memory copies**: Rust → numpy → torch tensor → GPU, then GPU → torch → numpy → Rust

Goal: eliminate Python from the self-play loop entirely by using the `ort` crate (Rust bindings for ONNX Runtime with CUDA support) for GPU inference directly from Rust.

## Architecture

### Current (Python bridge)
```
Rust MCTS loop
  └─ select leaves, encode boards
  └─ PyO3 call → Python eval_fn
       └─ numpy → torch tensor → GPU
       └─ model.forward()
       └─ GPU → torch → numpy
  └─ PyO3 return → Rust
  └─ expand + backprop
```

### Proposed (Rust-native)
```
Rust MCTS loop
  └─ select leaves, encode boards
  └─ ort::Session::run() → CUDA GPU
  └─ expand + backprop

Python only for:
  └─ model training (gradient updates)
  └─ ONNX export after each training iteration
```

## Training Loop Changes

The self-play + training cycle becomes:

```
1. Python: train model on replay buffer → save checkpoint
2. Python: export model to ONNX (torch.onnx.export)
3. Rust binary: load ONNX model via ort (CUDA), run N self-play games
   └─ write training data to disk (numpy .npz or flat binary)
4. Python: load training data from disk, add to replay buffer
5. Python: train model → goto 2
```

Alternatively, keep it as a single Python process that:
1. Exports ONNX to a temp file
2. Calls Rust via PyO3 with the ONNX path (no eval_fn callback)
3. Rust loads the ONNX model via ort, runs self-play, returns training data
4. Python trains on the returned data

The second approach is simpler — keeps the existing orchestration in Python but removes the per-batch callback.

## Phase 1: ONNX Model Export

**File:** `scripts/export_onnx.py` (shared by both Hive and Zertz)

### Hive export
- Input: `board_tensor [B, C, H, W]` (bf16 or f32) + `reserve [B, 10]`
- Output: `policy_logits [B, policy_size]`, `value [B, 1]`, `aux [B, 6]`
- The model does reserve broadcast+concat internally — bake into ONNX graph
- Dynamic batch axis

### Zertz export
- Input: `board_tensor [B, 6, 7, 7]` + `reserve [B, 22]`
- Output: `place_logits [B, 196]`, `cap_source [B, 49]`, `cap_dest [B, 49]`, `value [B, 1]`
- Same broadcast+concat baked in

### Export after each training iteration
Add to the training loop: after `save_checkpoint()`, also `torch.onnx.export()` to a known path. The Rust side reloads the ONNX model at the start of each self-play generation.

## Phase 2: Rust ort Integration

### New dependency
Add `ort` to `engine-zero` (the crate that already has the self-play code):

```toml
[dependencies]
ort = { version = "2", features = ["cuda"] }  # or "load-dynamic"
```

The `ort` crate can either link ONNX Runtime statically or load it dynamically. Using `load-dynamic` avoids compile-time linking — just needs the ONNX Runtime shared library at runtime (can be installed via pip: `pip install onnxruntime-gpu`).

### OrtInferenceEngine trait

Create a Rust-side inference abstraction that replaces the Python callback:

```rust
// rust/crates/engine-zero/src/inference.rs

pub trait InferenceEngine: Send {
    /// Run batch inference. Input shapes depend on the game.
    /// Returns raw output tensors as flat f32 vecs.
    fn infer(&self, boards: &[f32], reserves: &[f32], batch_size: usize)
        -> InferenceResult;
}

pub struct InferenceResult {
    pub policy_data: Vec<f32>,  // flattened [B, policy_size]
    pub value_data: Vec<f32>,   // [B]
    // Game-specific extra outputs:
    pub aux_data: Option<Vec<f32>>,        // Hive: [B, 6]
    pub cap_source: Option<Vec<f32>>,      // Zertz: [B, 49]
    pub cap_dest: Option<Vec<f32>>,        // Zertz: [B, 49]
}
```

### OrtEngine implementation

```rust
pub struct OrtEngine {
    session: ort::Session,
}

impl OrtEngine {
    pub fn load(onnx_path: &str) -> Result<Self, ort::Error> {
        let session = ort::Session::builder()?
            .with_execution_providers([
                ort::CUDAExecutionProvider::default().build(),
            ])?
            .commit_from_file(onnx_path)?;
        Ok(Self { session })
    }
}

impl InferenceEngine for OrtEngine {
    fn infer(&self, boards: &[f32], reserves: &[f32], batch_size: usize)
        -> InferenceResult
    {
        // Construct input tensors (zero-copy view into boards/reserves slices)
        // Run session
        // Extract output tensors
    }
}
```

### Key files to create/modify
- **New:** `rust/crates/engine-zero/src/inference.rs` — trait + OrtEngine
- **Modify:** `rust/crates/engine-zero/src/selfplay.rs` — replace `eval_fn` callback with `InferenceEngine`
- **Modify:** `rust/crates/engine-zero/src/zertz_python.rs` — same for Zertz
- **Modify:** `rust/crates/engine-zero/Cargo.toml` — add `ort` dependency

## Phase 3: Refactor Self-Play to Use InferenceEngine

### Hive: `run_simulations_internal`

Current signature (line 860 of selfplay.rs):
```rust
fn run_simulations_internal(
    py: Python<'_>,
    searches: &mut Vec<MctsSearch<Game>>,
    ...
    eval_fn: &Bound<'_, PyAny>,
    grid_size: usize,
)
```

New version (no Python dependency):
```rust
fn run_simulations_internal(
    searches: &mut Vec<MctsSearch<Game>>,
    ...
    engine: &dyn InferenceEngine,
    grid_size: usize,
)
```

Changes inside the function:
- Remove `PyArray2` construction and `eval_fn.call1()`
- Replace with: build flat `Vec<f32>` of boards/reserves → `engine.infer()` → extract policy/value slices
- Remove all `numpy` and `pyo3` imports from the hot path
- The bf16 encoding can stay (ort supports bf16 inputs) or switch to f32 (simpler, minor perf diff for small batches)

### Zertz: `play_games` simulation loop

Current (line 327 of zertz_python.rs): builds numpy arrays, calls `eval_fn.call1()`

New version:
- Build flat f32 slices from leaf encodings
- Call `engine.infer()`
- Extract place/cap_source/cap_dest/value from `InferenceResult`

### Batching improvement opportunity

With the Python bridge gone, we can batch more aggressively:
- Current: `rounds_per_flush=1` means 1 GPU call per selection round (batch = num_active_games)
- New: can accumulate more rounds before flushing since there's no PyO3 overhead per call
- But also: the per-call cost drops so much that smaller more frequent batches may be fine
- Tunable: `rounds_per_flush` remains configurable

## Phase 4: PyO3 Integration (Keep Python Orchestration)

### Approach A: New PyO3 method on existing session classes

Add `play_games_ort(onnx_path, progress_fn)` alongside existing `play_games(eval_fn, progress_fn)`:

```rust
#[pymethods]
impl PySelfPlaySession {
    /// Self-play with Rust-native ONNX inference (no Python eval callback).
    fn play_games_ort(
        &mut self,
        py: Python<'_>,
        onnx_path: &str,
        progress_fn: &Bound<'_, PyAny>,
        opening_sequences: Option<Vec<Vec<String>>>,
    ) -> PyResult<PySelfPlayResult> {
        let engine = OrtEngine::load(onnx_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        // Release GIL during self-play since no Python callbacks needed
        py.allow_threads(|| {
            self.run_games_internal(&engine, ...)
        })
    }
}
```

**Key benefit**: `py.allow_threads()` releases the GIL for the entire self-play run. Python's training thread (or other work) could theoretically run in parallel.

### Approach B: Standalone Rust binary

A separate `zertz-selfplay` / `hive-selfplay` binary that:
- Takes CLI args: ONNX path, num games, simulations, etc.
- Writes training data to disk (e.g., `.npz` via the `npyz` crate, or raw binary)
- Python loads the data files

This decouples self-play from Python entirely but adds IPC complexity.

**Recommendation**: Start with Approach A (simpler, keeps existing orchestration).

### Python-side changes

**Hive** (`hive/selfplay/rust_selfplay.py`):
```python
def play_games(self, num_games, opening_sequences=None):
    # Export current model to ONNX
    onnx_path = self._export_onnx()
    # Call Rust with ONNX path instead of eval callback
    result = session.play_games_ort(onnx_path, progress_fn, opening_sequences)
    return result
```

**Zertz** (`zertz/selfplay/selfplay.py`):
```python
# In the generation loop:
onnx_path = self._export_onnx()
result = session.play_games_ort(onnx_path, progress_fn)
```

### ONNX export helper
Add to both `hive/nn/model.py` and `zertz/nn/model.py`:
```python
def export_onnx(model, path, grid_size=7):
    model.eval()
    dummy_board = torch.zeros(1, NUM_CHANNELS, grid_size, grid_size)
    dummy_reserve = torch.zeros(1, RESERVE_SIZE)
    torch.onnx.export(
        model, (dummy_board, dummy_reserve), path,
        input_names=["board", "reserve"],
        output_names=["place", "cap_source", "cap_dest", "value"],
        dynamic_axes={"board": {0: "batch"}, "reserve": {0: "batch"},
                      "place": {0: "batch"}, "cap_source": {0: "batch"},
                      "cap_dest": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
    )
```

## Phase 5: Preserve Backward Compatibility

Keep the existing `play_games(eval_fn, ...)` methods working. The `play_games_ort()` method is additive. This lets us:
- A/B test Rust inference vs Python inference
- Fall back to Python if ort/CUDA issues arise
- Gradually migrate

## Performance Expectations

### Eliminated overhead per batch
- PyO3 GIL acquisition: ~1-5μs
- numpy array construction (malloc + copy): ~10-50μs for typical batch
- Python→torch tensor conversion: ~5-20μs
- torch→numpy result extraction: ~5-20μs
- **Total per batch: ~20-100μs** (× hundreds of batches per generation = 10-50ms)

### The bigger win: GIL release
With `py.allow_threads()`, the entire self-play run happens without holding the GIL. This means:
- No GIL contention with Python's garbage collector
- Rayon parallelism works without GIL restrictions
- Could overlap self-play with Python-side data processing

### Batching flexibility
Without the bridge cost, we can tune batch sizes purely based on GPU utilization:
- Larger batches (accumulate more rounds) for better GPU throughput
- Or smaller batches for lower latency per simulation — may not matter for throughput

## Key Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `rust/crates/engine-zero/Cargo.toml` | Modify | Add `ort` dependency |
| `rust/crates/engine-zero/src/inference.rs` | New | `InferenceEngine` trait + `OrtEngine` |
| `rust/crates/engine-zero/src/selfplay.rs` | Modify | Add `run_simulations_ort()`, `play_games_ort()` |
| `rust/crates/engine-zero/src/zertz_python.rs` | Modify | Add `play_games_ort()` for Zertz |
| `hive/selfplay/rust_selfplay.py` | Modify | Call `play_games_ort()` with ONNX path |
| `zertz/selfplay/selfplay.py` | Modify | Call `play_games_ort()` with ONNX path |
| `hive/nn/model.py` | Modify | Add `export_onnx()` |
| `zertz/nn/model.py` | Modify | Add `export_onnx()` |
| `scripts/export_onnx.py` | New | CLI tool for one-off ONNX export |

## Verification
1. **ONNX export**: export model, verify outputs match PyTorch with `onnxruntime` Python package
2. **ort loading**: load ONNX in Rust, run single inference, compare outputs
3. **Self-play parity**: run 1 generation with both `play_games` and `play_games_ort`, compare training data distributions (won't be identical due to MCTS randomness, but should be similar)
4. **Performance benchmark**: time a full generation with both methods, measure speedup
5. **Training stability**: run several generations end-to-end, verify loss curves are comparable

## Risks
- **ort + CUDA setup**: ONNX Runtime shared library must match CUDA version. Using `load-dynamic` feature + `pip install onnxruntime-gpu` is the easiest path.
- **Numerical differences**: ONNX Runtime's CUDA kernels may produce slightly different results than PyTorch. This is fine for self-play (MCTS is stochastic anyway) but worth verifying.
- **bf16 support**: if the model uses autocast bf16, the ONNX export may need explicit dtype handling. Alternative: export as fp32, let ONNX Runtime handle optimization.
- **ONNX export edge cases**: BatchNorm, residual connections, and the reserve broadcast should all be standard ONNX ops. Test early.
