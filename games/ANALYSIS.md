# Boardspace Game Archive Analysis

## Source Data

238,426 SGF files across 21 yearly archives (2006–2026), including both base
and expansion games (Mosquito, Ladybug, Pillbug).

**Data cutoff: 2026-03-16** (downloaded from boardspace.net)

## Game Outcomes (`game_outcomes.csv`)

| Metric | Count |
|---|---|
| Total games | 238,426 |
| Base games | 145,967 |
| Expansion games | 92,459 |
| Outcomes determined | 215,512 (90.4%) |
| — P0 (White) wins | 105,124 |
| — P1 (Black) wins | 103,191 |
| — Draws | 7,197 |
| Unknown (abandoned / no metadata) | 22,914 |

Outcomes were determined by replaying each game through the Python game engine
to detect queen-surrounding terminal states, with a fallback to SGF metadata
(`RE[]` field or `Resign` action) for games that ended without queen surrounding.

The 22,914 unknowns break down into two categories from processing:

- **16,927 parse errors** — move replay threw an exception (bad move string,
  unrecognised piece, or rule violation), with no metadata fallback available.
  Likely a mix of expansion piece types the engine doesn't support, edge cases
  in the SGF format not covered by the parser, and genuinely corrupt records.
  Worth investigating to recover usable games or fix parser gaps.
- **5,987 incomplete** — replay succeeded but game never reached a terminal
  state and no resign/RE metadata was present. Most likely abandoned or
  disconnected games.

Both categories split roughly evenly between base and expansion games (base:
12,267 unknown, expansion: 10,647 unknown), suggesting parse errors are not
exclusively caused by expansion pieces.

White's slight win advantage (105k vs 103k) is consistent with a first-mover
edge in Hive.

## Player ELO (`player_elo.csv`)

ELO computed chronologically (K=32, starting at 1500). Only games with
determined outcomes affect ratings.

| Metric | Value |
|---|---|
| Total players | 11,760 |
| Mean ELO | 1500 |
| Median ELO | 1491 |
| Players with 20+ games | 1,846 |
| Players with 100+ games | 338 |
| Players ELO ≥ 1600 | 645 |
| Players ELO ≥ 1700 | 260 |

### Top 10 players

| Player | ELO | Games | Win% |
|---|---|---|---|
| seer | 2087 | 330 | 94.5% |
| image13 | 2072 | 2350 | 72.9% |
| titoburito | 2057 | 369 | 94.8% |
| Docster | 2047 | 762 | 81.8% |
| ampexian | 2039 | 1105 | 89.7% |
| chasheen | 2000 | 101 | 84.8% |
| BlackMagic | 1986 | 499 | 76.3% |
| diogocrist | 1974 | 529 | 84.0% |
| Seer | 1972 | 755 | 77.2% |
| cmo | 1969 | 199 | 92.3% |

## Recommended Training Subset

Filter: base games, both players ELO ≥ 1600 with 20+ games, outcome determined.

| Metric | Count |
|---|---|
| Games (all outcomes) | 38,675 |
| Games (determined outcomes only) | 36,487 |

At ~30 moves/game × 12 symmetry augmentations ≈ **13M training positions**.

### DuckDB query to extract this subset

```sql
WITH qualified AS (
    SELECT player FROM 'games/player_elo.csv'
    WHERE elo >= 1600 AND games >= 20
)
SELECT o.*
FROM 'games/game_outcomes.csv' o
WHERE o.game_type = 'base'
  AND o.result IN ('p0_wins', 'p1_wins', 'draw')
  AND o.p0 IN (SELECT player FROM qualified)
  AND o.p1 IN (SELECT player FROM qualified);
```

### Threshold sensitivity

| Min ELO | Min games | Base games (determined) |
|---|---|---|
| 1500 | 1 | 97,968 |
| 1600 | 20 | 36,487 |
| 1700 | 20 | 1,301 |

Lowering the threshold adds volume but includes weaker play; raising it
reduces noise but risks overfitting to a narrow pool of opponents.
