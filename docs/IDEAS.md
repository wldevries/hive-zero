# Ideas & Research Notes

## Problem: Self-play games converging to draws

As training progresses, the proportion of decisive games (wins/losses) drops and games
increasingly hit the max-move limit as draws. Hypothesis:

The replay buffer fills up with positions from games that ran long (post-turn-20+), which
biases the network toward learning mid/endgame move quality for crowded boards. But most
real Hive games — including boardspace games — are decided well before the move cap. The
network learns "how to play a full board" rather than "how to close out a win", because
closing-out positions are underrepresented. Early-game and finishing-move patterns erode.
The value head also has less signal near the cap since the heuristic is weak for positions
that are genuinely drawn-by-repetition vs. drawn-by-exhaustion.

### Action item: Analyse boardspace game length distribution

**Done.** Implemented in Rust as part of `hive-zero replay`. Statistics from 145,622
successfully replayed base-game games:

| Metric       | min | p25 | median | avg  | p75 | max |
|------------- |----:|----:|-------:|-----:|----:|----:|
| move count   |   0 |  31 |     39 | 42.5 |  50 | 381 |
| diameter     |   0 |   6 |      7 |  7.1 |   8 |  15 |
| total pieces |   0 |  17 |     20 | 18.9 |  22 |  22 |

Median game is 39 plies, confirming that most real games are decided well before any
reasonable move cap. The 23x23 grid (diameter 15 max) is sufficient for virtually all games.

### Action item: Validate engine rules against boardspace corpus

Replay every boardspace SGF through the Rust engine and check for rule violations or
desync. This gives confidence that the engine's movement rules (One Hive, gate blocking,
beetle stacking, slide rules per piece type) match what boardspace enforces. Any systematic
rule error found here is also a training bug since self-play data quality depends on correct
move generation.

**Done:**

- UHP parsing extracted to `rust/src/uhp.rs` as free functions (no Python dependency).
- Replay uses `play_uhp_unchecked` — skips `valid_moves()` so boardspace moves go straight
  to the engine. Invalid moves surface as errors instead of being silently filtered.
- Board mutations (`place_piece`, `move_piece`, `remove_piece`) return `Result` instead of
  panicking, so OOB and stack mismatches are reported per-game.
- Rust CLI (`hive-zero replay`) replays the full corpus with error reporting.
- Direct SGF → Game replay (`sgf::replay_into_game`) converts Boardspace grid coords to hex
  coords directly, bypassing UHP string generation/parsing entirely. 12x faster, 99.7% success.
- Board dimension statistics collected during replay (diameter, q/r/s span, piece counts,
  move count with min/p25/median/avg/p75/max).

**Results:** 238,517 total games; 92,512 skipped (expansions); 145,622 replayed OK; 383
failed (all OOB — see board recentering below). 99.7% success rate on non-skipped games.

**Remaining:**

- Fix remaining 383 OOB failures (games that exceed 23x23 grid — see board recentering below).
- Migrate `process_games.py` to use `RustGame` from `hive_engine` instead of Python `Game`.
- Once no code outside `hive/core/` references Python game logic, delete `hive/core/rules.py`,
  `hive/core/game.py`, `hive/core/board.py`, `hive/core/pieces.py`, `hive/core/hex.py`.
  Keep only what has no Rust equivalent (e.g. `render.py`).

## Additional auxiliary training heads

The queen danger (QD) auxiliary head provides gradient signal on every position, even in drawn
games, but the draw convergence problem persists. Two more auxiliary signals could help the
network develop stronger positional understanding:

### Queen escape squares

Number of empty hexes the queen can legally slide to, divided by 6. Measures how trapped the
queen is — complementary to QD. A queen with 4 occupied neighbors but 1 escape route is very
different from 4 neighbors and 0 escapes.

- **Target**: legal queen slide destinations / 6 (0 = trapped, ~0.83 = free). 0 if queen not placed.
- **Per-position, per-player**: computed at time of play, like QD. Two outputs: my_queen_escape, opp_queen_escape.
- **Training**: sigmoid output, MSE loss, same as QD.

### Piece mobility ratio

Fraction of a player's pieces on the board that have at least one legal move (not pinned by
the one-hive rule or fully surrounded). Measures positional squeeze — a player losing mobility
is getting outplayed.

- **Target**: pieces with ≥1 legal move / total pieces on board (0 = all pinned, 1 = all mobile). 0 if no pieces placed.
- **Per-position, per-player**: two outputs: my_mobility, opp_mobility.
- **Training**: sigmoid output, MSE loss, same as QD.

### Implementation

- Extend the existing auxiliary head in the model (currently 2 QD outputs) to 6 outputs:
  [my_qd, opp_qd, my_queen_escape, opp_queen_escape, my_mobility, opp_mobility].
- Existing trunk weights preserved via `strict=False` in `load_checkpoint`; new head layers
  randomly initialized. Training can continue from current checkpoint.
- Compute targets in `rust/src/selfplay.rs` alongside existing QD computation.
- Queen escape: check slide legality for each of queen's 6 neighbor hexes.
- Piece mobility: for each piece, check one-hive rule + has at least one destination.

## Board recentering and dynamic bounds

### Problem

The board is a fixed 23x23 grid centered at hex (0,0), allowing coordinates from -11 to +11
in both axes. Some boardspace games (especially long ones) drift far enough in one direction
that pieces exceed this range, causing an out-of-bounds error. The first piece is always
placed at (0,0), so games that grow asymmetrically waste half the grid in the opposite
direction.

### Solution: recenter the hive

Add a `Game::recenter()` operation that shifts all piece positions so the bounding box center
of the hive sits at (0,0). This maximizes usable space in every direction. Key properties:

- **Transparent to UHP**: moves reference pieces by name, not coordinates, so recentering
  between moves doesn't affect parsing.
- **Transparent to the model**: board encoding is already position-relative (channels encode
  piece identity at grid cells), so the same board state centered differently produces
  equivalent tensors up to translation — recentering just keeps the hive within bounds.
- **When to recenter**: before any move whose destination would be out of bounds. Can also
  recenter proactively when the bounding box drifts past a threshold (e.g. any piece within
  2 cells of the grid edge).

### Implementation sketch

1. `Board::bounding_box() -> Option<(i8, i8, i8, i8)>` — min/max q/r of all placed pieces.
2. `Board::shift(dq, dr)` — clear and rebuild the grid with all positions offset by (dq, dr).
   Update `piece_positions` accordingly.
3. `Game::recenter()` — compute bounding box center, call `shift(-center_q, -center_r)`,
   update `from`/`to` in `move_history` by the same offset.
4. In `play_uhp_unchecked`: if parsed destination is OOB, call `recenter()`, re-parse
   (reference piece positions changed), retry. If still OOB, return error.

### Future: variable board size

The grid is currently a compile-time `[[StackSlot; 23]; 23]`. To support larger boards:

- Change to `Vec<StackSlot>` with runtime `grid_size` field.
- Board encoding and policy head dimensions are tied to `GRID_SIZE` — a larger replay board
  would need separate encoding, or replay could use a bigger grid while training stays at 23.
- For training: keep the model's input size fixed (23x23). Constrain `valid_moves()` to only
  return moves within the trained grid after recentering. This prevents the model from ever
  seeing positions it can't encode, while recentering ensures it can use the full grid.
- For replay-only (no model): use a larger grid (e.g. 31x31) to avoid any OOB issues.
