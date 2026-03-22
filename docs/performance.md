# Performance Analysis & Optimizations

## Board clone elimination in move generation (done)

**Problem**: Every piece movement function (`queen_moves`, `beetle_moves`, `spider_moves`,
`ant_moves`, `grasshopper_moves`) cloned the entire `Board` to temporarily remove the piece
before computing destinations. `Board` contains `Box<[[StackSlot; 23]; 23]>` — 4,232 bytes
heap-allocated per clone. In `valid_moves()`, this happened for every moveable piece (up to 11),
meaning up to 11 heap allocations per `valid_moves()` call.

**Fix**: Changed `get_moves()` to take `&mut Board`, remove the piece, compute moves, then
restore it. Same semantics, zero allocation. `valid_moves()` signature changed from `&self`
to `&mut self` accordingly.

**Impact**: Modest improvement for replay (~2%), but more significant for MCTS where
`valid_moves()` is called per simulation.

## Remaining opportunities

### MCTS: full Game clone per tree node (high impact)

Each `MctsNode` stores `pub game: Game`, and `expand_with_policy` clones the game for every
child. `Game` contains `Board` (4KB heap), `Vec<Move>` history, and `Vec<(u16, u16)>` reserve
history. A tree with 10K nodes = ~40MB just for board clones, causing massive cache thrashing.

**Fix**: Don't store game states in nodes. Reconstruct by replaying moves from root during
selection. Only the leaf needs the full game state for encoding. This is what most competitive
MCTS implementations do.

### `parse_and_play_uhp` valid-moves bypass for replay (done)

Replay used to call `valid_moves()` for every move just to validate it. Now
`play_uhp_unchecked()` skips validation entirely — parses the UHP string and applies the
move directly. Board mutations return `Result` so invalid moves are caught without panicking.

### Direct SGF → Game replay, eliminating UHP string round-trip (done)

**Problem**: The SGF parser (`sgf::parse_moves`) built a `Vec<String>` of UHP move strings,
then `play_uhp_unchecked` re-parsed those strings back into `Move` structs. This involved:
- `HashMap<String, Vec<String>>` for Boardspace coord→piece tracking
- `HashMap<String, String>` for piece→coord tracking
- Hundreds of `format!()`, `to_lowercase()`, `to_uppercase()` allocations per game
- UHP reference resolution (direction notation like "/wQ") that then got re-parsed

**Fix**: `sgf::replay_into_game()` converts Boardspace grid coordinates directly to hex
coordinates using a linear mapping (`q = col - origin_col`, `r = -(row - origin_row)`),
derived from the direction conventions in the old `drop_down_ref` function. Piece names are
parsed directly to `Piece` structs. No UHP strings, no HashMaps, no string allocations in
the hot loop. The `Game` state itself tracks piece positions, so no separate coord bookkeeping
is needed.

**Impact**: 12x faster replay (100s → 8s for 238K games). Also fixed ~2,800 games (97.8% →
99.7% success) that previously failed due to bugs in UHP reference string generation — the
direct coord path has no reference resolution to get wrong.

### Linear `.contains()` checks instead of bitset (medium impact)

- `articulation_points.contains(&pos)` — linear scan, called per piece
- `candidates.contains(&n)` in `get_placements` — linear dedup
- `visited.contains(&neighbor)` in `walk_any` (ant) — worst case, many hexes

**Fix**: Use a grid-indexed bitset (23×23 = 529 bits = 9 `u64`s) for O(1) lookups.

### `Vec` where fixed arrays suffice (low-medium impact)

Functions like `empty_neighbors` (max 6), `pieces_on_board` (max 11), `pieces_in_reserve`
(max 11) return heap-allocated `Vec`s. These could use `ArrayVec` or `SmallVec` to stay
on the stack.

### `can_slide` does O(36) work when O(1) is possible (low impact)

Computes `hex_neighbors` twice (12 entries) then does a nested loop (36 comparisons) to find
2 common neighbors. Since hex directions are fixed, common neighbors of adjacent hexes can be
computed directly from the direction index.

## Benchmarks

Full corpus replay (238K games, 146K base):
- Rust binary (UHP string round-trip): ~100s
- After board clone elimination: ~98s
- After direct SGF→Game replay: ~8s
