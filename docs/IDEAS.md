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

**Done.** Auxiliary head extended from 2 to 6 outputs:
[my_qd, opp_qd, my_queen_escape, opp_queen_escape, my_mobility, opp_mobility].

- Trunk weights preserved via shape-filtered `load_state_dict`; `qd_fc2` (64→2 → 64→6)
  reinitialized, all other weights kept. QD loss recovered within 1 iteration.
- QE and mobility targets computed in `rust/src/selfplay.rs` per-position.
- Queen escape: checks slide legality, gate blocking, one-hive (articulation point), adjacency.
- Piece mobility: iterates pieces, checks one-hive + `get_moves()` for each.
- All 6 aux targets packed into a single [N, 6] numpy array from Rust → Python.
- CSV log: `qe_loss` and `mob_loss` columns appended at end (backward compatible).

### Observations

All three aux losses converged quickly (within 2-3 iterations) because the trunk already
encodes the relevant spatial features. However, the auxiliary signals haven't noticeably
improved the decisive game rate — the network recognizes positional patterns but still
struggles with multi-move attacking sequences. The bottleneck appears to be planning depth
(policy + MCTS search), not position evaluation.

## Supervised intermezzo during self-play

The network has good pattern recognition (aux losses converge quickly) but the policy head
can't learn attacking sequences from self-play alone — 500 simulations is too shallow to
discover 5-6 move queen-surrounding sequences in a 30+ branching factor game.

### Idea

Periodically inject a few epochs of supervised training on boardspace games between self-play
iterations. The network already understands the game; supervised data teaches the policy head
"this is what winning move sequences look like" on top of existing trunk features.

### Key constraints

A full pretrain run (8 chunks × 3 epochs × 100k positions) would overwrite self-play learning.
The intermezzo must be gentle:

- **Small batch**: 1-2 chunks (10-20k positions) instead of 800k
- **1 epoch**: same as self-play, no repeated passes
- **Low LR**: same or lower than self-play LR (≤ 0.0025)
- **Mix with replay buffer**: interleave supervised positions with recent self-play data
  to prevent catastrophic forgetting
- **Decisive games only**: filter boardspace data for games that ended in wins, not draws
- **Frequency**: every N self-play iterations (e.g. every 10)

### Expected benefit

The policy head sees real examples of how human players close out wins — move sequences that
create queen pressure, beetle climbs, grasshopper jumps to fill gaps. These patterns are
exactly what self-play fails to discover on its own.

## Curriculum for opening book

Currently boardspace openings are replayed verbatim but not trained on (not recorded to
history). The network learns good midgame play from these positions but never learns the
opening moves that created them.

### Idea: taper opening book reliance

Gradually reduce boardspace fraction and random opening moves over training:
- Early training: heavy book usage (0.7 frac, 0-8 random moves) — diverse positions
- Mid training: reduce to 0.3 frac, 0-4 random moves — force network to find own openings
- Late training: minimal or no book — fully self-play

This teaches the value head to distinguish "this opening leads to decisive games" from
"this opening leads to draws," which it can only learn from self-generated openings.

## Expansion pieces (transfer learning)

The base-game trunk has learned spatial reasoning, connectivity, and piece relationships that
transfer directly to expansion pieces (Mosquito, Ladybug, Pillbug).

### Approach

1. Expand input conv channels (39 → +expansion piece channels)
2. Expand policy head (11 → 14 piece channels)
3. Expand reserve vector (10 → 16)
4. Keep res blocks unchanged — trunk features are game-general
5. Train on expansion games (boardspace has 92k skipped expansion games)

Expected to converge much faster than training from scratch since the trunk already
understands piece interaction, hive connectivity, and queen pressure.

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
