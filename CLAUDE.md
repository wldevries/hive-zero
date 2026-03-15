# Hive AI Engine

## Project Overview
A Hive AI engine in Python + Rust implementing the Universal Hive Protocol (UHP) over stdin/stdout.
Uses AlphaZero-style MCTS + neural network for move selection.
The Rust extension (`hive_engine`) reimplements game logic, MCTS, and encoding for ~45x speedup.

## Architecture
```
hive/
  core/        - Game logic: hex coords, pieces, board, movement rules, game state, renderer
  encoding/    - Board-to-tensor and move encoding for neural network I/O
  nn/          - PyTorch AlphaZero-style model (policy + value heads)
  mcts/        - Monte Carlo Tree Search with legal move masking
  uhp/         - UHP stdin/stdout protocol engine
  selfplay/    - Self-play training loop
    selfplay.py       - SelfPlayTrainer orchestrator, fast/full MCTS scheduling
    rust_selfplay.py  - RustFastSelfPlay: Rust-accelerated game + MCTS self-play
rust/
  src/         - Rust game engine (PyO3 extension module `hive_engine`)
    board.rs          - Board state, 23x23 grid, hex-to-grid conversion
    board_encoding.rs - Board tensor encoding (mirrors hive/encoding/)
    game.rs           - Game state, move application, undo
    hex.rs            - Axial hex coordinates, directions
    move_encoding.rs  - Move-to-policy-index encoding (mirrors hive/encoding/)
    mcts/             - Rust MCTS (search, arena allocator, nodes)
    piece.rs          - Piece types and colors
    python.rs         - PyO3 bindings exposing RustGame, RustMCTS to Python
    rules.rs          - Movement rules per piece type
```

## Key Design Decisions
- **Hex coordinates**: Axial (q, r) system. Flat-top hexagons.
- **Board encoding**: 23x23 grid with ~30 channels (piece type x color, stack depths, reserves).
- **Move encoding**: (source, destination) hex pairs in flat policy space (12 channels x 23 x 23 = 6348). Placement moves use fixed off-board source slots per piece type. Movement uses direction channels (6 directions + 1 stacking).
- **Base game only**: Queen, Beetle, Grasshopper, Spider, Ant. No expansions (Mosquito/Ladybug/Pillbug) initially.
- **Beetle stacking**: Up to depth 7.
- **One Hive Rule**: All pieces must remain connected after any move.
- **Gate blocking**: Pieces cannot slide through gates (two adjacent occupied hexes).
- **Dual implementation**: Python game logic in `hive/core/` and parallel Rust implementation in `rust/src/`. Both must produce identical encoding indices. When editing move encoding or board encoding, update both Python and Rust versions.

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
If the Rust extension is not available, self-play falls back to pure Python automatically.

## Training CLI
All commands go through `main.py`:
```bash
uv run python main.py train --iterations 50 --games 20 --simulations 100
uv run python main.py train --mcts-after -1 --simulations 100  # Always use full MCTS
```

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

## UHP Commands
`info`, `newgame`, `play`, `validmoves`, `bestmove`, `undo`, `options`, `pass`
