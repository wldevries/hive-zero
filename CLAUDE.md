# Hive AI Engine

## Project Overview
A Hive AI engine in Python + Rust implementing the Universal Hive Protocol (UHP) over stdin/stdout.
Uses AlphaZero-style MCTS + neural network for move selection.
All game logic, MCTS, and encoding run in Rust (`hive_engine` via PyO3) for performance.

## Architecture
```
hive/
  core/        - Game logic: hex coords, pieces, board, movement rules, game state, renderer
  encoding/    - Board-to-tensor and move encoding for neural network I/O + symmetry augmentation
  nn/          - PyTorch AlphaZero-style model (policy + value heads), training loop
  uhp/         - UHP stdin/stdout protocol engine
  selfplay/    - Self-play training loop (Rust-only, no Python MCTS)
    selfplay.py       - SelfPlayTrainer orchestrator, playout cap randomization
    rust_selfplay.py  - RustParallelSelfPlay (rayon-parallel batched MCTS, playout cap)
rust/
  src/         - Rust game engine (PyO3 extension module `hive_engine`)
    board.rs          - Board state, 23x23 grid, hex-to-grid conversion
    board_encoding.rs - Board tensor encoding (mirrors hive/encoding/)
    game.rs           - Game state, move application, undo, heuristic evaluation
    hex.rs            - Axial hex coordinates, directions
    move_encoding.rs  - Move-to-policy-index encoding (mirrors hive/encoding/)
    mcts/             - Rust MCTS (search, arena allocator, nodes)
    piece.rs          - Piece types and colors
    python.rs         - PyO3 bindings: RustGame, RustMCTS, RustBatchMCTS
    rules.rs          - Movement rules per piece type
```

## Key Design Decisions
- **Hex coordinates**: Axial (q, r) system. Flat-top hexagons.
- **Board encoding**: 23x23 grid with 23 channels (piece type x color, stack depths, reserves).
- **Move encoding**: (source, destination) hex pairs in flat policy space (12 channels x 23 x 23 = 6348). Placement moves use fixed off-board source slots per piece type. Movement uses direction channels (6 directions + 1 stacking).
- **Base game only**: Queen, Beetle, Grasshopper, Spider, Ant. No expansions.
- **Beetle stacking**: Up to depth 7.
- **One Hive Rule**: All pieces must remain connected after any move.
- **Gate blocking**: Pieces cannot slide through gates (two adjacent occupied hexes).
- **Rust-only engine**: All game logic, MCTS, and encoding in Rust. Python handles NN inference and training only. When editing move encoding or board encoding, update both Python and Rust versions.

## Training Pipeline
- **SGD + momentum 0.9** with cosine annealing + warm restarts (T_0=30, lr_max=0.05, lr_min=1e-5)
- **1 epoch** per iteration (avoids overfitting on stale replay buffer data)
- **Playout cap randomization**: per-turn random fast/full search (KataGo-style), fast turns train value only
- **Symmetry augmentation** at buffer insertion time (12 hex symmetries), not during training
- **Replay buffer**: 50k positions max, deque-based O(1) eviction
- **Fast-cap turns**: no Dirichlet noise, play strongest move, value-only training (zero policy target)
- **Heuristic value** for unfinished games: queen neighbor pressure + beetle-on-queen bonus
- **Rayon parallelism**: MCTS tree ops (select, encode, expand, backprop) parallelized across games
- **RustBatchMCTS.run_simulations**: full simulation loop in Rust with single Python GPU callback per round

## Package Manager
Use `uv` for all dependency management. Do NOT use pip directly.
```bash
uv run python main.py             # Start UHP engine (default)
uv run python main.py train       # Run self-play training
uv run python -m pytest tests/    # Run tests
```

## Building the Rust Extension
The Rust code is built as a PyO3 extension module via maturin. The package name is `hive-zero`
but the module is imported as `hive_engine` (set via `[lib] name` in Cargo.toml).

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
