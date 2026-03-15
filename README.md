# Hive AI Engine

A Python + Rust AI engine for the [Hive](https://boardgamegeek.com/boardgame/2655/hive) board game, implementing the [Universal Hive Protocol (UHP)](https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol) for interoperability with UHP-compatible viewers.

## Features

- **Complete Hive rules**: Queen, Beetle, Grasshopper, Spider, Ant with full movement validation including One Hive rule, gate blocking, and beetle stacking
- **UHP compliant**: stdin/stdout protocol compatible with Mzinga and other UHP viewers
- **AlphaZero-style AI**: Convolutional neural network with policy and value heads, trained via self-play with SGD + stepped LR schedule
- **Rust game engine**: PyO3-based Rust extension (`hive_engine`) for game simulation, MCTS, and board encoding
- **Rayon-parallel MCTS**: Cross-game batched tree search with parallel CPU ops and batched GPU inference
- **Self-play training**: Automated pipeline with fast/MCTS cycling, warmup phase, and replay buffer

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- PyTorch (CUDA 12.8)
- Rust toolchain (for building the native extension)

## Setup

```bash
# Install Python dependencies
uv sync

# Install PyTorch with CUDA support
uv pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Build the Rust extension (auto-rebuilds on source changes via uv)
uv run maturin develop --release -m rust/Cargo.toml
```

## Usage

### Run as UHP Engine

```bash
uv run python main.py
```

The engine communicates over stdin/stdout using UHP. Connect it to any UHP-compatible viewer (e.g., [MzingaViewer](https://github.com/jonthysell/Mzinga)).

### Train via Self-Play

```bash
# Basic training run
uv run python main.py train

# With options
uv run python main.py train --iterations 50 --games 40 --simulations 100 --device cuda
```

Key training flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations` | 100 | Training iterations |
| `--games` | 20 | Self-play games per iteration |
| `--simulations` | 100 | MCTS simulations per move |
| `--epochs` | 1 | Training epochs per iteration |
| `--batch-size` | 512 | Training batch size |
| `--model` | `model.pt` | Model file path |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--blocks` | 6 | Residual blocks in network |
| `--channels` | 64 | Channels in network |
| `--max-moves` | 200 | Max moves per game |
| `--mcts-after` | 0 | Use full MCTS after this iteration (0=cycling, -1=always MCTS) |
| `--fast-iters` | 10 | Fast iterations per cycle |
| `--full-iters` | 2 | Full MCTS iterations per cycle |
| `--warmup-positions` | 10000 | Fill buffer before training (0=skip) |
| `--time-limit` | None | Stop after N minutes |

### Run Tests

```bash
uv run python -m pytest tests/
```

## UHP Commands

| Command | Description |
|---------|-------------|
| `info` | Engine identification and capabilities |
| `newgame` | Start a new game (base game) |
| `play <move>` | Play a move |
| `pass` | Pass turn (when no valid moves) |
| `validmoves` | List all legal moves |
| `bestmove time <hh:mm:ss>` | Find best move within time limit |
| `bestmove depth <n>` | Find best move to given search depth |
| `undo [n]` | Undo last n moves |
| `options` | List/get/set engine options |

## Architecture

```
hive/
  core/        Game logic (hex coordinates, pieces, board, rules, game state, renderer)
  encoding/    Neural network I/O (board tensors, move encoding, symmetry augmentation)
  nn/          PyTorch model (AlphaZero-style conv net, policy + value heads)
  uhp/         UHP protocol engine (stdin/stdout)
  selfplay/    Self-play training loop (Rust-accelerated)
rust/
  src/         Rust game engine exposed via PyO3 as `hive_engine`
               (board, game, rules, MCTS, move/board encoding, rayon parallelism)
```

## Move Notation

Follows UHP MoveString format:
- First move: `wS1` (piece name only)
- Placement: `bA1 wS1/` (place black Ant 1 to top-right of white Spider 1)
- Movement: `wQ wA1-` (move white Queen to the right of white Ant 1)
- Stacking: `wB1 wS1` (beetle climbs on top of white Spider 1)
- Pass: `pass`

## License

MIT
