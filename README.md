# Hive AI Engine

A Python + Rust AI engine for the [Hive](https://boardgamegeek.com/boardgame/2655/hive) board game, implementing the [Universal Hive Protocol (UHP)](https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol) for interoperability with UHP-compatible viewers.

## Features

- **Complete Hive rules**: Queen, Beetle, Grasshopper, Spider, Ant with full movement validation including One Hive rule, gate blocking, and beetle stacking
- **UHP compliant**: stdin/stdout protocol compatible with Mzinga and other UHP viewers
- **AlphaZero-style AI**: Convolutional neural network with policy and value heads, trained via self-play with SGD + cosine annealing with warm restarts
- **Rust game engine**: PyO3-based Rust extension (`engine_zero`) for game simulation, MCTS, and board encoding
- **Rayon-parallel MCTS**: Cross-game batched tree search with parallel CPU ops and batched GPU inference
- **Dirichlet noise**: Applied to MCTS root during self-play for exploration
- **Self-play training**: Automated pipeline with MCTS self-play and replay buffer
- **Playout cap randomization**: KataGo-style per-turn fast/full search; fast turns train value only, full turns train both policy and value
- **Resignation**: Configurable threshold-based resignation during self-play (disabled during warmup); ~10% calibration games track false-positive rate
- **Checkpoint evaluation**: Opt-in (`--checkpoint-eval`) self-play matches between the current model and best known model to prevent regressions

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
uv run python main.py train --iterations 200 --games 40 --simulations 25 --device cuda
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
| `--playout-cap-p` | 0.0 | Probability of full search per turn (KataGo-style, 0=disabled) |
| `--fast-cap` | 20 | Simulations for fast-search turns when playout cap is enabled |
| `--checkpoint-every` | 10 | Save checkpoint every N iterations |
| `--checkpoint-eval` | off | Enable self-play eval at each checkpoint |
| `--time-limit` | None | Stop after N minutes |

### Checkpoint Evaluation

Every `--checkpoint-every` iterations, the current model is pitted against `best_model.pt` in a self-play match. If the challenger scores ≥ 50%, it becomes the new `best_model.pt` and `model.pt` is updated to match — ensuring `model.pt` always reflects the best known weights.

On first run (no `best_model.pt`), the two most recent checkpoints play each other to establish a baseline.

Eval games use opening temperature sampling (first 8 moves) for game diversity, then switch to argmax.

### Evaluate Against Mzinga

```bash
uv run python main.py eval --games 10 --simulations 200 --mzinga-path path/to/MzingaEngine.exe
```

### Run Tests

```bash
uv run python -m pytest tests/
```

## Model Files

| File | Description |
|------|-------------|
| `model.pt` | Current training model (always mirrors `best_model.pt` after each eval) |
| `best_model.pt` | Best model found by checkpoint evaluation |
| `checkpoints/model_iter{N}.pt` | Periodic snapshots every `--checkpoint-every` iterations |
| `training_log.tsv` | Per-iteration training metrics and eval results |

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
  encoding/    Neural network I/O (board tensors, move encoding)
  nn/          PyTorch model (AlphaZero-style conv net, policy + value heads)
  uhp/         UHP protocol engine (stdin/stdout)
  selfplay/    Self-play training loop (Rust-accelerated)
  eval/        Engine-vs-engine match runner and model evaluation
rust/
  src/         Rust game engine exposed via PyO3 as `engine_zero`
               (board, game, rules, MCTS, move/board encoding, rayon parallelism)
```

## Move Notation

Follows UHP MoveString format:
- First move: `wS1` (piece name only)
- Placement: `bA1 wS1/` (place black Ant 1 to top-right of white Spider 1)
- Movement: `wQ wA1-` (move white Queen to the right of white Ant 1)
- Stacking: `wB1 wS1` (beetle climbs on top of white Spider 1)
- Pass: `pass`

## Zertz

An AlphaZero-style engine for [Zertz](https://boardgamegeek.com/boardgame/596/zertz) is included under `zertz/`. It uses the same Rust game backend (`engine_zero`) and PyTorch training loop.

### Train Zertz

```bash
uv run zertz train \
  --model zertz_b4_c128.pt \
  --blocks 4 --channels 128 \
  --games 100 --simulations 400 \
  --device cuda \
  --playout-cap-p 0.25 \
  --play-batch-size 2 \
  --augment-symmetry \
  --temp-threshold 30 \
  --comment "my run"
```

Key training flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `zertz.pt` | Model file path (also names the log CSV) |
| `--blocks` | 6 | Residual blocks |
| `--channels` | 64 | Network channels |
| `--games` | 20 | Self-play games per iteration |
| `--simulations` | 100 | MCTS simulations per move |
| `--playout-cap-p` | 0.0 | Fraction of full-search turns (KataGo-style) |
| `--fast-cap` | 20 | Simulations for fast-search turns |
| `--play-batch-size` | 2 | MCTS rounds per GPU inference call |
| `--temp-threshold` | 30 | Move number after which temperature drops to 0 |
| `--augment-symmetry` | off | Apply D6 hex symmetry augmentation (12× data) |
| `--lr` | 0.02 | Learning rate |
| `--max-moves` | 40 | Max moves per game |
| `--replay-window` | 8 | Replay buffer size (in iterations) |
| `--checkpoint-every` | 10 | Save checkpoint every N iterations |
| `--time-limit` | None | Stop after N minutes |
| `--comment` | `""` | Comment logged with each iteration |

Training logs are written to `{model_stem}_log.csv` (e.g. `zertz_b4_c128_log.csv`).

### Play Against the AI

```bash
uv run zertz play --model zertz_b4_c128.pt --simulations 400
```

Optional flags:
- `--color p1` or `--color p2` — force a side (default: random)
- Omit `--model` to play against a random AI

**Move format** (placement turn): `<color> <place> [remove]`
- Colors: `W` (white), `G` (grey), `B` (black)
- Coordinates: `A1`–`G7` (board is a hex grid, not all cells are valid)
- Example: `W D4 D3` — place a white marble on D4 and remove the ring at D3
- Omit `[remove]` when placing on an isolated ring (no removal required)

Capture turns show a numbered list — pick the number.

### View the Training Log

```bash
uv run python scripts/plot_log.py zertz_b4_c128_log.csv
```

Plots win percentages, loss curves, game length, and win conditions (white/grey/black/combo) across iterations. The window geometry is remembered between runs.

## Yinsh

An AlphaZero-style engine for [YINSH](https://boardgamegeek.com/boardgame/7854/yinsh) is included under `yinsh/`. It uses the same Rust game backend (`engine_zero`) and PyTorch training loop.

### Train Yinsh

```bash
uv run yinsh train \
  --name yinsh_b8_c96 \
  --blocks 8 --channels 96 \
  --games 16 --simulations 200 \
  --device cuda \
  --play-batch-size 8
```

Key training flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | `yinsh` | Model name; paths derived as `models/{name}/` |
| `--blocks` | 8 | Residual blocks |
| `--channels` | 96 | Network channels |
| `--games` | 16 | Self-play games per generation |
| `--simulations` | 200 | MCTS simulations per move |
| `--max-moves` | 400 | Max moves per game |
| `--replay-window` | 8 | Replay buffer size (in generations) |
| `--playout-cap-p` | 0.0 | Fraction of full-search turns (KataGo-style) |
| `--fast-cap` | 30 | Simulations for fast-search turns |
| `--play-batch-size` | 8 | MCTS rounds per GPU inference call |
| `--temp-threshold` | 20 | Move number after which temperature drops to 0 |
| `--augment-symmetry` | off | Apply D6 hex symmetry augmentation |
| `--lr` | 0.02 | Learning rate |
| `--checkpoint-every` | 10 | Save checkpoint every N generations |
| `--time-limit` | None | Stop after N minutes |

### Battle Two Models

```bash
uv run yinsh battle models/yinsh/checkpoints/yinsh_gen00100.pt models/yinsh/yinsh.pt --games 20
```

## License

MIT
