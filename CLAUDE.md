# Hive AI Engine

## Project Overview
A Hive AI engine in Python implementing the Universal Hive Protocol (UHP) over stdin/stdout.
Uses AlphaZero-style MCTS + neural network for move selection.

## Architecture
```
hive/
  core/        - Game logic: hex coords, pieces, board, movement rules, game state
  encoding/    - Board-to-tensor and move encoding for neural network I/O
  nn/          - PyTorch AlphaZero-style model (policy + value heads)
  mcts/        - Monte Carlo Tree Search with legal move masking
  uhp/         - UHP stdin/stdout protocol engine
  selfplay/    - Self-play training data generation
```

## Key Design Decisions
- **Hex coordinates**: Axial (q, r) system. Flat-top hexagons.
- **Board encoding**: 23x23 grid with ~30 channels (piece type x color, stack depths, reserves).
- **Move encoding**: (source, destination) hex pairs in flat policy space. Placement moves use fixed off-board source slots per piece type.
- **Base game only**: Queen, Beetle, Grasshopper, Spider, Ant. No expansions (Mosquito/Ladybug/Pillbug) initially.
- **Beetle stacking**: Up to depth 7.
- **One Hive Rule**: All pieces must remain connected after any move.
- **Gate blocking**: Pieces cannot slide through gates (two adjacent occupied hexes).

## UHP Commands
`info`, `newgame`, `play`, `validmoves`, `bestmove`, `undo`, `options`, `pass`

## Package Manager
Use `uv` for all dependency management. Do NOT use pip directly.
```bash
uv run python main.py          # Start UHP engine
uv run python -m hive.selfplay # Run self-play training
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
