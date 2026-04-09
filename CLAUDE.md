# Hive AI Engine

## Project Overview
A Hive AI engine in Python + Rust implementing the Universal Hive Protocol (UHP) over stdin/stdout.
Uses AlphaZero-style MCTS + neural network for move selection.
All game logic, MCTS, and encoding run in Rust (`engine_zero` via PyO3) for performance.

## Architecture
```
hive/
  core/        - Game logic: hex coords, pieces, board, movement rules, game state, renderer
  encoding/    - Board-to-tensor and move encoding for neural network I/O
  nn/          - PyTorch AlphaZero-style model (policy + value heads), training loop
  uhp/         - UHP stdin/stdout protocol engine
  selfplay/    - Self-play training loop (Rust-only, no Python MCTS)
    selfplay.py       - SelfPlayTrainer orchestrator, playout cap randomization
    rust_selfplay.py  - RustParallelSelfPlay (rayon-parallel batched MCTS, playout cap)
rust/crates/
  core-game/src/       - Game abstractions and shared Rust logic
    game.rs            - Game/NNGame/Outcome traits
    hex.rs             - Axial hex coordinates, directions
    mcts/              - MCTS (search, arena allocator, nodes)
    symmetry.rs        - D6 symmetry transforms
    sgf.rs             - SGF parsing utilities
  hive-game/src/       - Hive-specific game logic
    board.rs           - Board state, 23x23 grid, hex-to-grid conversion
    board_encoding.rs  - Board tensor encoding (mirrors hive/encoding/)
    game.rs            - Game state, move application, undo, heuristic evaluation
    move_encoding.rs   - Move-to-policy-index encoding (mirrors hive/encoding/)
    piece.rs           - Piece types and colors
    rules.rs           - Movement rules per piece type
    uhp.rs             - UHP move formatting/parsing
    sgf.rs             - SGF replay
  tictactoe-game/src/  - TicTacToe game logic (AlphaZero pipeline validation)
    game.rs            - TicTacToe board, encoding, move types
  zertz-game/src/      - Zertz game logic
    zertz.rs           - Board, rules, move types
    board_encoding.rs  - Board tensor encoding
    move_encoding.rs   - Factorized policy encoding
  engine-zero/src/     - PyO3 extension module `engine_zero`
    hive_python.rs     - PyO3 bindings: HiveGame (includes best_move MCTS loop)
    hive_selfplay.rs   - Hive self-play session (parallel MCTS, Python inference callback)
    tictactoe_python.rs - PyO3 bindings: TTTSelfPlaySession, TTTGame
    zertz_python.rs    - PyO3 bindings: ZertzGame
    inference.rs       - Shared inference helpers
    lib.rs             - Module registration
```

## Key Design Decisions
- **Hex coordinates**: Axial (q, r) system. Flat-top hexagons.
- **Grid size**: Parametric — default 23x23, configurable via `--grid-size` CLI flag. Stored in .pt checkpoint. The physical board stays 23x23 for game logic; the NN encoding grid can be smaller (must be odd). Smaller grids = faster training (computation scales as grid_size²). Boardspace stats show max diameter=15, so 17x17 covers all observed games.
- **Board encoding**: grid_size x grid_size grid with 24 channels, fully current-player-relative. Channels 0-4: current player's base pieces (Q,S,B,G,A); 5-9: opponent's base pieces; 10-13: current player's stacker at depths 1-4; 14-17: opponent's stacker at depths 1-4; 18: hive edge (binary: empty cells adjacent to at least one occupied cell); 19-20: queen distance (hex distance normalized by grid_size); 21-22: queen adjacency (binary); 23: pinned pieces (articulation points). Reserve vector current-player-relative (0-4: mine, 5-9: opponent's).
- **Move encoding**: Per-piece policy space (11 channels x grid_size x grid_size). Channel = piece index within current player (0=Queen, 1-2=Spider, 3-4=Beetle, 5-7=Grasshopper, 8-10=Ant). Destination cell stores the logit. Same channel scheme covers both placement and movement — no direction encoding. Canonical placement ordering enforced in valid_moves(): only lowest-numbered reserve piece per type is offered.
- **Base game only**: Queen, Beetle, Grasshopper, Spider, Ant. No expansions.
- **Beetle stacking**: Up to depth 7.
- **One Hive Rule**: All pieces must remain connected after any move.
- **Gate blocking**: Pieces cannot slide through gates (two adjacent occupied hexes).
- **Rust-only engine**: All game logic, MCTS, and encoding in Rust. Python handles NN inference and training only. When editing move encoding or board encoding, update both Python and Rust versions.

## Training Pipeline
- **SGD + momentum 0.9**, constant LR (default 0.02, set via --lr). Previously tried cosine annealing with warm restarts — removed in favour of manual LR adjustment.
- **1 epoch** per iteration (avoids overfitting on stale replay buffer data)
- **Playout cap randomization**: per-turn random fast/full search (KataGo-style), fast turns train value only
- **Symmetry augmentation**: D6 hex symmetries (12 transforms), active and in use via `--augment-symmetry`.
- **Replay buffer**: in-memory only, not persisted to disk. Lost on process exit. Pretrain and selfplay run as separate processes so the buffer is always empty at the start of selfplay.
- **Fast-cap turns**: no Dirichlet noise, play strongest move, added to buffer with value-only training (policy loss masked)
- **Heuristic value** for unfinished games: queen neighbor pressure only (no draw penalty)
- **Auxiliary heads**: Six sigmoid outputs from a dedicated pathway off the trunk (conv1x1→FC64→FC6), predicting per-position metrics for both current and opponent player. Trained with MSE, always active (not masked). Provides gradient signal on every position even in drawn games.
  - Queen danger (neighbors/6, 0–1)
  - Queen escape (legal slide destinations / 6, 0–1)
  - Piece mobility (fraction of pieces with ≥1 legal move, 0–1)
- **Opening diversity**: Two mechanisms to avoid early-game convergence:
  - `--random-opening-moves MIN-MAX`: play N random moves (uniform in [min, max]) before MCTS takes over
  - `--opening-book PATH`: use boardspace game openings, with `--boardspace-frac` controlling the mix vs random openings
- **Resignation**: `--resign-threshold` (default -0.97) with `--resign-min-moves` safety. Calibration games (10%) play to completion to measure false positive rate.
- **Skip timeout games**: `--skip-timeout-games` discards all training data from games that hit the move cap
- **RustSelfPlaySession**: full simulation loop in Rust with a single Python GPU callback (or ORT) per inference batch. Inference is the bottleneck; MCTS ops (init, expand_and_backprop) use rayon across games but are negligible relative to GPU time.

### Known issue: self-play draw convergence (Hive only)
From-scratch Hive self-play converges to draws within a few iterations. The network can't learn to win
(surrounding the queen requires coordinated attacks), so games hit the move cap, value targets are ~0,
and the value head learns to predict 0 everywhere. Tried mitigations: draw penalty, heuristic values
for unfinished games, opening randomization (helps somewhat), pretraining on boardspace games (delays
but doesn't prevent). See `docs/IDEAS.md` for analysis.

### Zertz architecture
- **Board encoding**: 6 channels on 7x7 grid (RADIUS=3 hexagonal board, 37 valid cells). Channels 0-3: marble colors + empty rings. Channel 4: capture turn flag (1.0 everywhere if capture/mid-capture). Channel 5: mid-capture source position. Reserve vector of 22 floats (current-player-relative: supply, captures, combo win progress, single-color win progress cap/threshold, rings remaining/37), broadcast spatially and concatenated with board tensor before the trunk.
- **Policy heads**: Three factorized conv1x1 heads off the trunk (no FC layer):
  - `place` [4, 7, 7]: ch 0-2 = place White/Grey/Black ball, ch 3 = remove ring
  - `cap_source` [1, 7, 7]: which marble starts a capture hop
  - `cap_dest` [1, 7, 7]: where the marble lands
- **Move prior computation** (Rust MCTS): scores are sums of head logits per move type, then softmax over legal moves.
  - `Place(color, pos, remove)`: `place[color, pos] + place[3, remove]`
  - `PlaceOnly(color, pos)`: `place[color, pos]`
  - `Capture(from, to)`: `cap_source[from] + cap_dest[to]`
  - Mid-capture continuation: `cap_dest[to]` only
- **Sequential captures**: Multi-hop capture chains are sequential MCTS decisions. Each hop is a separate game state/tree node. `ZertzBoard.mid_capture` tracks in-progress chains; same-player consecutive turns handled in MCTS backprop.
- **Training data**: Flat POLICY_SIZE=5587 visit distributions stored in Rust, marginalized to per-head targets in Python training loop.
- **Policy loss**: Independent cross-entropy per head (place color/position, place remove, capture source, capture dest). Mid-capture turns only train cap_dest.

### Zertz: no draw convergence problem
Zertz games end naturally before ~40 turns because rings are removed from the board each turn,
making the game finite by construction. When win rates converge to ~50/50, value loss rises because
balanced outcomes are harder to predict — fix is higher simulation count.

## Package Manager
Use `uv` for all dependency management. Do NOT use pip directly.
```bash
uv run python main.py             # Start UHP engine (default)
uv run python main.py train       # Run self-play training
uv run python -m pytest tests/    # Run tests
```

## Building the Rust Extension
The Rust code is built as a PyO3 extension module via maturin. The package name is `hive-zero`
but the module is imported as `engine_zero` (set via `[lib] name` in Cargo.toml).

`uv run` automatically rebuilds when Rust source files change (via `cache-keys` in pyproject.toml).
Just run `uv run python main.py ...` after editing — no manual rebuild needed.

## Testing
```bash
uv run python -m pytest tests/
```

## PyTorch Installation
PyTorch CUDA is installed directly via:
```bash
uv pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

## Dependencies
- Python 3.12+
- PyTorch (CUDA 12.8)
- numpy
- Rust toolchain + maturin (for building the native extension)
- rayon (Rust, for parallel MCTS)

## UHP Commands
`info`, `newgame`, `play`, `validmoves`, `bestmove`, `undo`, `options`, `pass`

## Git
Only commit when explicitly asked by the user.
