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

Add analysis over the downloaded boardspace SGF corpus to answer:

- What is the median / 90th-percentile game length (in plies)?
- At what turn does the winning queen typically get surrounded?
- What fraction of games end before turn 20 / 30 / 40?
- How does game length correlate with opening (number of pieces placed in first N turns)?

**Preferred implementation:** Rust, inside the existing `scripts/` pipeline or as a new
`hive_engine` exposed function — keeps it fast for large corpora and consistent with the
rest of the engine. Python fallback acceptable if the Rust lift is too large.

Output should be a summary printed to stdout (and optionally a CSV for further analysis)
covering per-game: ply count, winner, decisive/draw, queen-surrounded turn.

### Action item: Validate engine rules against boardspace corpus

Replay every boardspace SGF through the Rust engine and check for rule violations or
desync. Currently `process_games.py` already replays games via the Python `Game` wrapper
but silently swallows errors. This action item makes errors visible and categorised:

- Count and log games where replay fails (move rejected by engine rules)
- For failed games, record the failing move and ply number
- Distinguish between parse errors (bad SGF) and actual rule disagreements
- Report a breakdown: total games, clean replays, parse failures, rule errors

This gives confidence that the engine's movement rules (One Hive, gate blocking, beetle
stacking, slide rules per piece type) match what boardspace enforces. Any systematic rule
error found here is also a training bug since self-play data quality depends on correct
move generation.

**Implementation:**

1. Move UHP move string parsing out of the Python `Game`. Two options — TBD:
   - Push into Rust (`hive_engine`) and expose via PyO3
   - Keep in Python but consolidate into `hive/uhp/` with no dependency on `hive/core/`
2. Move all game replay logic in `process_games.py` (and anywhere else that uses the
   Python `Game`) to use `RustGame` from `hive_engine` instead. The Rust engine is what
   self-play and MCTS actually run, so validation against it is the meaningful test.
3. Once no code outside of `hive/core/` references the Python game logic, delete
   `hive/core/rules.py`, `hive/core/game.py`, `hive/core/board.py`, `hive/core/pieces.py`,
   and `hive/core/hex.py`. Keep only what has no Rust equivalent (e.g. `render.py`).
4. With the above done, extend `process_games.py` to surface errors verbosely. A `--strict`
   flag that aborts on first rule error would help with debugging.

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
