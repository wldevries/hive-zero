# Hive AI Engine

A Python AI engine for the [Hive](https://boardgamegeek.com/boardgame/2655/hive) board game, implementing the [Universal Hive Protocol (UHP)](https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol) for interoperability with UHP-compatible viewers.

## Features

- **Complete Hive rules**: Queen, Beetle, Grasshopper, Spider, Ant with full movement validation including One Hive rule, gate blocking, and beetle stacking
- **UHP compliant**: stdin/stdout protocol compatible with Mzinga and other UHP viewers
- **AlphaZero-style AI**: Convolutional neural network with policy and value heads, trained via self-play
- **MCTS**: Monte Carlo Tree Search with neural network evaluation and legal move masking
- **Self-play training**: Automated training pipeline generating games and updating the network

## Quick Start

### Requirements

- Python 3.10+
- PyTorch >= 2.0
- NumPy

### Install

```bash
pip install -r requirements.txt
```

### Run as UHP Engine

```bash
python main.py
```

The engine communicates over stdin/stdout using UHP. Connect it to any UHP-compatible viewer (e.g., [MzingaViewer](https://github.com/jonthysell/Mzinga)).

### Train via Self-Play

```bash
python -m hive.selfplay
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
  core/        Game logic (hex coordinates, pieces, board, rules, game state)
  encoding/    Neural network I/O (board tensors, move encoding)
  nn/          PyTorch model (AlphaZero-style conv net)
  mcts/        Monte Carlo Tree Search
  uhp/         UHP protocol engine
  selfplay/    Self-play training loop
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
