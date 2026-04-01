# Plan: Static Zertz Web App (Play vs AI)

## Context
Build a fully client-side static website where a human plays Zertz against a trained AI. Everything runs in the browser — no server needed.

## Architecture: All-in-Rust/wasm + Plain JS UI

```
JS (UI only)  ──click──▶  Rust/wasm
                           ├─ Game logic (existing zertz-game crate)
                           ├─ Board encoding (existing)
                           ├─ MCTS search (existing, reused)
                           └─ tract-onnx inference (loads .onnx model)
```

- **Rust/wasm** handles game logic, MCTS, and NN inference (via `tract-onnx`, a pure-Rust ONNX runtime that compiles to wasm32)
- **Plain JavaScript** only does SVG rendering and click handling
- No JS MCTS reimplementation needed — reuse existing Rust MCTS directly

## File Structure
```
web/
  index.html
  css/style.css
  js/
    app.js          -- game loop, UI events, orchestration
    board.js        -- SVG hex board rendering
  wasm/             -- wasm-pack output
  model/
    zertz.onnx      -- exported model
rust/crates/zertz-wasm/   -- new wasm-bindgen crate (game + MCTS + tract)
scripts/export_zertz_onnx.py
```

## Phase 1: ONNX Model Export

**File:** `scripts/export_zertz_onnx.py`

- Load checkpoint via `zertz.nn.model.load_checkpoint()` or create random model if no checkpoint
- `torch.onnx.export()` with opset 17
- Inputs: `board_tensor [1,6,7,7]` + `reserve_vector [1,22]`
- Outputs: `place_logits [1,196]`, `cap_source [1,49]`, `cap_dest [1,49]`, `value [1,1]`
- Reserve broadcast+concat baked into the ONNX graph (standard ops)
- Verify round-trip with `onnxruntime` in Python

**Existing code to reuse:**
- `zertz/nn/model.py` — `ZertzNet`, `load_checkpoint()`, `create_model()`

## Phase 2: Rust wasm Crate

**New crate:** `rust/crates/zertz-wasm/`

### Dependencies
```toml
[dependencies]
zertz-game = { path = "../zertz-game", default-features = false }
wasm-bindgen = "0.2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tract-onnx = "0.22"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["console"] }
getrandom = { version = "0.2", features = ["js"] }
```

### Feature-gating for wasm compatibility
The `zertz-game` and `core-game` crates depend on `zip` (for SGF/replay parsing) and `rand` (for random play). These don't compile to wasm or aren't needed. Add feature gates:

**`core-game/Cargo.toml`:**
- Add `default = ["sgf"]` feature
- Gate `zip` dep and `sgf` module behind `sgf` feature

**`zertz-game/Cargo.toml`:**
- Add `default = ["replay", "random-play"]` feature
- Gate `zip` dep and `replay`/`random_play`/`sgf` modules behind features
- Gate `rand`/`rand_distr` behind `random-play` feature

**`zertz-game/src/lib.rs`:** conditional `mod` declarations with `#[cfg(feature = "...")]`

### Wasm API

```rust
#[wasm_bindgen]
pub struct ZertzApp {
    model: SimplePlan<...>,  // tract optimized model
}

#[wasm_bindgen]
pub struct ZertzGame {
    board: ZertzBoard,
}

#[wasm_bindgen]
impl ZertzApp {
    // Load ONNX model from bytes (fetched by JS)
    pub fn new(model_bytes: &[u8]) -> Result<ZertzApp, JsValue>

    // Run MCTS and return best move as JSON
    // simulations: number of MCTS simulations (100-800)
    pub fn best_move(&self, game: &ZertzGame, simulations: u32) -> Result<String, JsValue>
}

#[wasm_bindgen]
impl ZertzGame {
    pub fn new() -> Self
    pub fn is_game_over(&self) -> bool
    pub fn outcome(&self) -> String           // "ongoing"/"p1"/"p2"
    pub fn next_player(&self) -> u8
    pub fn is_mid_capture(&self) -> bool

    // Board state for rendering
    pub fn rings_json(&self) -> String        // [{index, q, r, state: "empty"/"removed"/"white"/"grey"/"black"}, ...]
    pub fn supply(&self) -> Vec<u8>           // [W, G, B]
    pub fn captures_json(&self) -> String     // {p1: [W,G,B], p2: [W,G,B]}

    // Legal moves for UI highlighting
    pub fn legal_moves_json(&self) -> String  // [{type, color?, place_at?, remove?, from?, to?}, ...]

    // Apply a move (JSON string matching legal_moves_json format)
    pub fn play_move(&mut self, move_json: &str) -> Result<(), JsValue>
}
```

### MCTS integration with tract

The existing MCTS in `zertz-game/src/mcts/search.rs` uses `PolicyHeads` for expansion. We create a wrapper that:
1. Encodes the board via `board_encoding::encode_board()`
2. Runs tract inference to get place/cap_source/cap_dest/value tensors
3. Feeds `PolicyHeads` into the existing `expand_with_policy()` function
4. Runs the existing select → expand → backprop loop

The key adaptation: the current MCTS `select_and_expand` is batched (for parallel self-play). For the web, we write a simpler single-game loop that calls tract synchronously per simulation. This is a new function in the wasm crate, not modifying the existing MCTS code.

**Existing code to reuse:**
- `zertz-game/src/mcts/search.rs` — `MctsSearch`, `PolicyHeads`, `expand_with_policy()`, `MctsNode`, `Edge`, `NodeArena`
- `zertz-game/src/mcts/node.rs` — node/edge structs
- `zertz-game/src/mcts/arena.rs` — arena allocator
- `zertz-game/src/board_encoding.rs` — `encode_board()`
- `zertz-game/src/zertz.rs` — `ZertzBoard`, `ZertzMove`, `clone_light()`

### Build
```bash
cd rust/crates/zertz-wasm
wasm-pack build --target web --out-dir ../../../web/wasm
```

### Key files to modify
- `rust/Cargo.toml` — add `zertz-wasm` to workspace
- `rust/crates/core-game/Cargo.toml` — feature-gate `zip`/`sgf`
- `rust/crates/core-game/src/lib.rs` — conditional `mod sgf`
- `rust/crates/zertz-game/Cargo.toml` — feature-gate `zip`/`rand`/`replay`/`random_play`
- `rust/crates/zertz-game/src/lib.rs` — conditional modules
- **New:** `rust/crates/zertz-wasm/Cargo.toml`
- **New:** `rust/crates/zertz-wasm/src/lib.rs`

## Phase 3: Frontend (Plain JavaScript + SVG)

### Board rendering (`web/js/board.js`)
- SVG hex grid: 37 flat-top hexagons in 4-5-6-7-6-5-4 layout
- Hex → pixel: `x = size * 3/2 * q`, `y = size * sqrt(3) * (r + q/2)`
- Ring states: empty (beige outline), removed (hidden/faded), marble (colored circle inside hex)
- Click handlers on hexagons for move input

### Game UI (`web/js/app.js`)
- Player choice dialog at start (who goes first)
- Supply display: clickable marble icons for color selection during placement
- Capture display: each player's captured marbles with win progress
- Turn indicator
- Move input flow:
  - **Placement**: select color → click empty ring to place → click edge ring to remove
  - **Capture** (mandatory): click marble → click hop destination → auto-continue if chain
  - Highlight legal targets at each step
- AI thinking: "Thinking..." overlay with disabled board
- New game button, difficulty selector (simulations: 100/200/400)

### Loading sequence
1. Load HTML/CSS/JS
2. Fetch .wasm module + .onnx model in parallel (progress bar for model)
3. `ZertzApp.new(model_bytes)` — initializes tract model
4. Player choice dialog → start game

### Async AI moves
`ZertzApp.best_move()` is synchronous in wasm (runs full MCTS). To avoid blocking UI:
- Call it via a Web Worker, OR
- Use `setTimeout` chunking if Web Worker adds too much complexity (tract model can't be transferred to worker easily)
- Start with blocking call + "Thinking..." overlay; optimize later if needed

## Phase 4: Integration & Polish
- Error handling for wasm/ONNX loading failures
- Mobile-friendly responsive SVG
- Highlight last AI move
- Game over screen with result

## Verification
1. **ONNX export**: `python scripts/export_zertz_onnx.py` produces valid .onnx file
2. **Wasm build**: `wasm-pack build` succeeds without errors
3. **Browser test**: `python -m http.server` in `web/`, open in browser
4. **Game logic**: play legal moves, verify state updates in UI
5. **AI**: AI returns legal moves, game plays to completion
6. **Edge cases**: mid-capture chains, isolation captures, board-full ending

## Resolved Decisions
- **Checkpoint**: No trained model yet. Export script supports random model for testing.
- **Player order**: User picks who goes first at game start.
- **Visual style**: Clean, minimal. Iterate later.
- **NN runtime**: `tract-onnx` in Rust/wasm (pure Rust, compiles to wasm32).
- **MCTS**: Reuse existing Rust MCTS, add a simple single-game wrapper for synchronous tract inference.


ort-tract?
https://ort.pyke.io/backends/tract