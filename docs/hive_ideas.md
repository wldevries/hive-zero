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
- Migrate `process_games.py` to use `RustGame` from `engine_zero` instead of Python `Game`.
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

## Supervised mix during self-play

The network has good pattern recognition (aux losses converge quickly) but the policy head
can't learn attacking sequences from self-play alone — even 800 simulations is too shallow to
discover 5-6 move queen-surrounding sequences in a 30+ branching factor game. With playout cap
randomization (75% fast-cap at 20 sims), the policy head only learns from 25% of positions,
and even those can't find multi-move attacks.

### Evidence from training

Auxiliary heads (queen danger, queen escape, piece mobility) were added at iteration 54. The
aux losses converged within 2-3 iterations, confirming the trunk already encodes positional
features. But the decisive game rate still collapsed: ~34% at iter 54-65, ~15% at iter 76-85,
then <5% from iter 86 onward. The bottleneck is planning depth (policy + MCTS search), not
position evaluation.

### Why decisive games cluster early in training runs

Across multiple runs, wins concentrate in iterations ~15-50, then taper off. The likely
mechanism:

1. **Defense is a single-move skill**: "don't leave your queen exposed" is a local pattern
   the network learns quickly. Attack requires 5-6 coordinated moves — much harder.
2. **Early network is semi-random**: chaotic play accidentally exposes queens, creating
   decisive outcomes. The network hasn't yet learned to defend, so queens get surrounded
   by chance even without intentional attack.
3. **Defense converges first**: once both sides learn to keep their queen safe, games
   stalemate. The network reaches a local optimum of "don't lose" rather than "win."
4. **Crowded boards lock down**: with more pieces on the board, mobility drops and positions
   become rigid. The network may not know how to handle full-board positions where attack
   requires dismantling defensive structures.

The gradient signal for attack vanishes once defense is learned — no decisive games means
no policy gradient toward winning moves. This is a classic RL exploration collapse.

### Approach: interspersed supervised data (preferred over intermezzo)

Rather than periodically injecting separate supervised training epochs ("intermezzo"), mix
supervised positions directly into the replay buffer each iteration. This is better because:

- **Smooth gradients**: every batch sees both self-play and human positions, no oscillation
  between "imitate humans" and "self-play" modes
- **No catastrophic forgetting**: both signal sources train together in shuffled batches
- **Continuous signal**: policy head gets attacking patterns every iteration, not every Nth
- **Simpler**: just add samples to the existing replay buffer

Implementation: `--supervised-mix FRAC` (e.g. 0.2 = 20% of replay buffer is boardspace data).
Each iteration, after adding self-play data, top up the buffer with randomly sampled decisive
boardspace positions. Filter for wins only (no draws) — the whole point is teaching attacks.

### Auxiliary targets for supervised samples

Supervised positions from `game_to_samples()` don't have aux targets. Options:
1. **Mask aux loss** for supervised samples (cleanest — avoids corrupting aux heads)
2. Compute aux targets by running Rust engine per position (accurate but slow)
3. Train on zeros (aux loss is small, probably doesn't matter)

Prefer option 1 to keep aux heads clean.

### Other options considered

#### A. Separate intermezzo epochs (rejected)

Run a dedicated supervised epoch every N iterations. Problems: abrupt gradient shifts,
catastrophic forgetting risk in both directions, bursty signal (most iters get nothing).

#### B. Increase MCTS simulations (insufficient)

Tried 800 sims (up from 500). Still not enough — branching factor ~30 means ~4 moves deep,
short of the 5-6 coordinated moves needed to surround a queen. Diminishing returns without
better policy guidance.

#### C. Policy prior bias toward queen-adjacent moves (risky)

Add a handcrafted bonus to the NN policy prior for moves that land adjacent to the opponent
queen. This would focus MCTS search on attacking moves without needing to discover them from
scratch. **Risk: creates kamikaze behavior.** Moving a lone ant next to the queen is aggressive
but pointless without supporting pieces nearby. The network would learn to rush pieces at the
queen without building the positional foundation needed for a real surround. Could make play
worse, not better.

A softer version: only boost queen-adjacent moves when the aux head already predicts elevated
queen danger (support pieces are in place). But this adds complexity and couples the prior to
aux head quality.

#### D. Reward shaping / intermediate rewards (risky)

Instead of only getting +1/-1 at game end, provide small intermediate rewards for increasing
queen danger or reducing opponent queen escape. Would need careful tuning to avoid the network
optimizing for the shaping reward instead of winning. Could interact badly with the value head.
Same kamikaze risk as option C — the network might learn to create queen pressure at the expense
of overall position quality.

#### E. Asymmetric search / progressive widening

Use more simulations in positions where queen danger is high (attack is close to completion).
Or use progressive widening to focus early MCTS visits on a smaller set of promising moves.
Less explored, needs research.

#### F. Temperature / exploration schedule

Keep higher policy temperature (more random move selection) for longer during self-play to
maintain exploration. Currently the network becomes deterministic quickly, locking in defensive
play. Higher temperature in the first 30-40 moves could create more diverse positions where
attacks are possible. Downside: noisier training data, slower convergence of good patterns.

#### G. Opponent pool / historical snapshots

Play against older checkpoints (that don't defend as well) alongside self-play. Decisive games
against weaker opponents provide attack gradient signal. League training (AlphaStar-style) is
the full version but even a simple "play 20% of games against a checkpoint from 50 iterations
ago" could help. The weaker opponent leaves queen-attack opportunities that the current network
can learn from.

### Expected benefit

The policy head sees real examples of how human players close out wins — move sequences that
create queen pressure, beetle climbs, grasshopper jumps to fill gaps. These patterns are
exactly what self-play fails to discover on its own. With interspersed training, the network
gets continuous pressure to maintain these patterns alongside self-play learning.

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

## Symmetry augmentation (12x data)

Standard CNNs (including ResNets) are translation equivariant but NOT rotation/reflection
equivariant. The network must learn each rotated version of a spatial pattern independently.
On a hex board with D6 symmetry (6 rotations × 2 reflections = 12 symmetries), this wastes
significant channel capacity — up to 12x redundancy for rotation-dependent features.

### Approach: augment each position with all 12 symmetries

For every position generated during selfplay, add all 12 rotated/reflected versions to the
replay buffer with the same value target and appropriately transformed policy/board tensors.
One real position → 12 training samples.

- **Board tensor**: rotate/reflect the 23×23 hex grid. Axial coordinates transform under D6
  as rotations of 60° multiples and reflections across hex axes.
- **Policy target**: the policy is 11 channels × 23 × 23 (per-cell logits). Apply the same
  hex rotation to the destination cell coordinates. Piece channel stays the same (piece
  identity doesn't change under rotation).
- **Value target**: unchanged (game outcome is rotation-invariant).
- **Auxiliary targets**: unchanged (queen danger, escape, mobility are all scalar).

This is the standard AlphaZero approach for games with board symmetries. It's the cheapest
way to get rotation awareness: 12x training signal from the same selfplay data, no model
changes, no extra inference cost.

### Long-term: group equivariant convolutions

The principled solution is to constrain conv filters to respect hex D6 symmetry (e.g. using
escnn library). Each filter automatically works in all 12 orientations, effectively giving
96 channels the representational power of 1152 orientation-aware channels. More complex to
implement but eliminates the redundancy at the architecture level rather than patching it
with data augmentation.
