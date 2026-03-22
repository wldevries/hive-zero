# Hive AI Engine

## Project Overview
A Hive AI engine in Python + Rust implementing the Universal Hive Protocol (UHP) over stdin/stdout.
Uses AlphaZero-style MCTS + neural network for move selection.
All game logic, MCTS, and encoding run in Rust (`hive_engine` via PyO3) for performance.

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
- **Board encoding**: 23x23 grid with 39 channels, fully current-player-relative and piece-identity-aware. Channels 0-10: current player's pieces at base (Q,S1,S2,B1,B2,G1,G2,G3,A1,A2,A3); 11-21: opponent's pieces at base; 22-37: stacked beetles by identity and depth (4 beetles × 4 depths, channel=22+player_offset(0/8)+(beetle_num-1)*4+(depth-1)); 38: stack height. Reserve vector current-player-relative (0-4: mine, 5-9: opponent's). Policy head also uses piece-identity channels 0-10 (same indexing).
- **Move encoding**: Per-piece policy space (11 channels x 23 x 23 = 5819). Channel = piece index within current player (0=Queen, 1-2=Spider, 3-4=Beetle, 5-7=Grasshopper, 8-10=Ant). Destination cell stores the logit. Same channel scheme covers both placement and movement — no direction encoding. Canonical placement ordering enforced in valid_moves(): only lowest-numbered reserve piece per type is offered.
- **Base game only**: Queen, Beetle, Grasshopper, Spider, Ant. No expansions.
- **Beetle stacking**: Up to depth 7.
- **One Hive Rule**: All pieces must remain connected after any move.
- **Gate blocking**: Pieces cannot slide through gates (two adjacent occupied hexes).
- **Rust-only engine**: All game logic, MCTS, and encoding in Rust. Python handles NN inference and training only. When editing move encoding or board encoding, update both Python and Rust versions.

## Training Pipeline
- **SGD + momentum 0.9**, constant LR (default 0.02, set via --lr). Previously tried cosine annealing with warm restarts — removed in favour of manual LR adjustment.
- **1 epoch** per iteration (avoids overfitting on stale replay buffer data)
- **Playout cap randomization**: per-turn random fast/full search (KataGo-style), fast turns train value only
- **Symmetry augmentation**: previously implemented (12 hex symmetries) but removed — not in use.
- **Replay buffer**: in-memory only, not persisted to disk. Lost on process exit. Pretrain and selfplay run as separate processes so the buffer is always empty at the start of selfplay.
- **Fast-cap turns**: no Dirichlet noise, play strongest move, added to buffer with value-only training (policy loss masked)
- **Heuristic value** for unfinished games: queen neighbor pressure + beetle-on-queen bonus (no draw penalty)
- **Auxiliary queen danger heads**: Two extra outputs from the value head's shared hidden layer predicting current-player and opponent queen danger (neighbors/6 + beetle-on-top bonus, sigmoid, 0–1). Trained with MSE weighted at 0.15 each, always active (not masked). Provides gradient signal on every position even in drawn games.
- **Opening diversity**: Two mechanisms to avoid early-game convergence:
  - `--random-opening-moves MIN-MAX`: play N random moves (uniform in [min, max]) before MCTS takes over
  - `--opening-book PATH`: use boardspace game openings, with `--boardspace-frac` controlling the mix vs random openings
- **Resignation**: `--resign-threshold` (default -0.95) with `--resign-min-moves` safety. Calibration games (10%) play to completion to measure false positive rate.
- **Skip timeout games**: `--skip-timeout-games` discards all training data from games that hit the move cap
- **Rayon parallelism**: MCTS tree ops (select, encode, expand, backprop) parallelized across games
- **RustBatchMCTS.run_simulations**: full simulation loop in Rust with single Python GPU callback per round

### Known issue: self-play draw convergence
From-scratch self-play converges to draws within a few iterations. The network can't learn to win
(surrounding the queen requires coordinated attacks), so games hit the move cap, value targets are ~0,
and the value head learns to predict 0 everywhere. Tried mitigations: draw penalty, heuristic values
for unfinished games, opening randomization (helps somewhat), pretraining on boardspace games (delays
but doesn't prevent). See `docs/IDEAS.md` for analysis.

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
