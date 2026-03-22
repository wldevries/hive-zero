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
