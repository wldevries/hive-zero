//! Generic negamax alphabeta search.
//!
//! Caller supplies an evaluation function that scores a position from the
//! perspective of the player about to move (current-player-relative). The
//! driver itself is game-agnostic — it relies only on the `Game` and
//! `Undoable` traits.
//!
//! Mzinga's `GameAI` adds iterative deepening, a transposition table, and
//! killer-move heuristics on top of the same alphabeta core. This module
//! exposes a `SearchContext` so those features can be plugged in later
//! without changing the recursion signature.
//!
//! Value convention:
//! - Terminal win for the player to move → `WIN_SCORE` (≈ +1e6).
//! - Terminal loss → `-WIN_SCORE`.
//! - Draw → 0.0.
//! - Heuristic eval is expected to live in (−1, 1) for ongoing positions,
//!   so it never collides with terminal magnitudes.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

use crate::game::{Game, Outcome, Undoable};

/// Scalar used for terminal win/loss positions. Larger than any reasonable
/// heuristic eval so that win-finding always dominates positional scoring.
pub const WIN_SCORE: f32 = 1_000_000.0;

/// How often (in nodes) `negamax` polls the deadline. Must be a power of two
/// so the check compiles to a cheap bit-and. 1024 keeps the overhead well
/// under 1% of node cost while still aborting within ~milliseconds of the
/// deadline.
const DEADLINE_POLL_MASK: u64 = 1023;

#[derive(Clone, Copy, Debug)]
pub struct AlphaBetaParams {
    pub max_depth: u32,
    /// Wall-clock budget for the entire search. When `Some`, iterative
    /// deepening polls the deadline inside `negamax` and aborts the
    /// in-flight iteration the moment it expires; the caller falls back
    /// to the deepest *fully-completed* iteration's best move.
    pub time_budget: Option<Duration>,
}

impl AlphaBetaParams {
    pub fn new(max_depth: u32) -> Self {
        Self { max_depth, time_budget: None }
    }

    /// Builder-style setter for the wall-clock budget.
    pub fn with_time_budget(mut self, budget: Option<Duration>) -> Self {
        self.time_budget = budget;
        self
    }
}

/// Trait for the move type to feed the driver's move-ordering machinery.
///
/// Generic over the game type `G` so that game-specific implementations can
/// look at the current board state when computing `priority`. The default
/// `priority` returns 0 (no preference), so games with no static tactical
/// signal can fall back entirely to the killer + history tiebreakers.
///
/// Sort precedence inside the driver (smaller key tried earlier):
///   1. `priority(game)` — game-specific static signal (e.g. Zertz's
///      "this placement exposes opp to a forced capture" classifier).
///   2. Killer slots at this ply — within-priority-bucket ordering for
///      moves that produced a β-cutoff at this exact ply across siblings.
///   3. Negated history score — accumulated β-cutoff bonus across the
///      whole tree, indexed by `history_index`.
///
/// `HISTORY_SIZE` must be strictly greater than the largest index any
/// `history_index` call can return; the driver indexes into a flat
/// `Vec<i32>` of that size. `history_index` returns `None` for moves that
/// should not influence history ordering (typically pass).
pub trait MoveOrdering<G: ?Sized>: Sized {
    /// Stable index for this move into the per-search history table, or
    /// `None` if the move should not influence ordering (e.g. pass).
    fn history_index(&self) -> Option<usize>;

    /// Total slots required for the history table.
    const HISTORY_SIZE: usize;

    /// Game-specific static priority, evaluated once per move per node sort.
    /// Smaller values are tried earlier. Default: 0 (delegate fully to the
    /// killer + history tiebreakers). Implementations should be cheap —
    /// O(few cell reads) per call — since they run inside the inner sort.
    fn priority(&self, _game: &G) -> i32 {
        0
    }
}

/// Trait for games that support transposition table caching: provide a
/// (Zobrist-style) key for the current position. The key must be deterministic
/// and must encode side-to-move. Two states that are tactically equivalent
/// (same board, same side, same legal moves, same recursive value) should
/// hash equally.
///
/// Phase 1 caveat: the key does NOT need to encode repetition history.
/// Implementations should override `tt_cacheable` to return false when the
/// current position's score is path-dependent (e.g. `DrawByRepetition` in
/// Hive). Path-dependence introduced *deeper* in the search subtree is not
/// addressed here; phase 2+ will tighten this if profiling shows it matters.
pub trait TranspositionKey {
    /// Returns the TT key for the current position.
    fn tt_key(&self) -> u64;

    /// Whether the score computed at this position can be safely cached.
    /// Default: yes. Override for path-dependent states (e.g. repetition).
    fn tt_cacheable(&self) -> bool {
        true
    }
}

/// What kind of bound the stored score is. Determines when a TT hit can be
/// returned directly: `Exact` always usable, `LowerBound` only when the
/// stored score already meets/exceeds the querying beta (a fail-high), and
/// `UpperBound` only when the stored score is at/below the querying alpha
/// (a fail-low).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundFlag {
    /// Stored score is the true minimax value at this depth.
    Exact,
    /// Stored score is a lower bound on the true value (true ≥ stored). The
    /// search at this node was cut off by a beta cutoff before exhausting
    /// every move, so the true score may be even higher.
    LowerBound,
    /// Stored score is an upper bound on the true value (true ≤ stored). No
    /// move improved the alpha that was passed in — the explored value is a
    /// ceiling but not necessarily the true maximum reachable here.
    UpperBound,
}

/// One transposition table entry. Carries a bound flag (phase 2) and the
/// move that produced the stored score (phase 3, used for move ordering).
#[derive(Clone, Copy, Debug)]
pub struct TtEntry<M: Copy> {
    /// Remaining search depth at the time this entry was computed. A probe
    /// can only return the cached score when this depth is at least the
    /// querying depth — otherwise the cached value is too shallow to trust.
    pub depth: u32,
    /// Score from the perspective of the side to move at this position.
    /// Interpretation depends on `flag`.
    pub score: f32,
    /// Whether `score` is exact, a lower bound, or an upper bound.
    pub flag: BoundFlag,
    /// The move that produced `score` (or that triggered the beta cutoff).
    /// Lifted to the front of the move list on subsequent visits to this
    /// position, even when the cached score itself can't be returned.
    /// `None` only for nodes where the search emitted no real move (e.g. a
    /// forced pass — currently not stored, but the field is here for
    /// completeness and forward compatibility).
    pub best_move: Option<M>,
}

/// Hash-map backed transposition table. Always-replace, generic over the
/// game's move type so the entry can carry a best_move for ordering.
pub struct TranspositionTable<M: Copy> {
    map: HashMap<u64, TtEntry<M>>,
}

impl<M: Copy> Default for TranspositionTable<M> {
    fn default() -> Self {
        Self { map: HashMap::new() }
    }
}

impl<M: Copy> TranspositionTable<M> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn probe(&self, key: u64) -> Option<&TtEntry<M>> {
        self.map.get(&key)
    }

    pub fn store(&mut self, key: u64, entry: TtEntry<M>) {
        self.map.insert(key, entry);
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Iterator over all stored entries. Intended for diagnostics / tests.
    pub fn entries(&self) -> impl Iterator<Item = (&u64, &TtEntry<M>)> {
        self.map.iter()
    }
}

/// Mutable per-search state. Carries params, node counts, the transposition
/// table, and the killer/history move-ordering tables.
///
/// Generic over both the move type `M` and the game type `G`. `G` only
/// appears in the `MoveOrdering<G>` bound that gates `HISTORY_SIZE`, so a
/// `PhantomData<fn() -> *const G>` field anchors it without affecting
/// auto-traits or layout. Type inference resolves `G` at every
/// `SearchContext::new` call site since the context is invariably handed
/// off to `negamax` / `alphabeta_*` next, which constrain both `M` and `G`.
pub struct SearchContext<M: Copy, G: ?Sized> {
    pub params: AlphaBetaParams,
    pub nodes_visited: u64,
    pub tt: TranspositionTable<M>,
    /// Times the cached score was returned directly from a TT probe
    /// (early termination of search at this node). With iterative
    /// deepening, shallow entries don't satisfy deeper queries, so most
    /// of the TT benefit shifts to `tt_order_hits` instead.
    pub tt_hits: u64,
    /// Times the TT's cached `best_move` was lifted to the front of the
    /// move list (move-ordering hit). The cached score wasn't usable, but
    /// the move was — and trying it first usually triggers a beta cutoff
    /// far faster than the default move order would.
    pub tt_order_hits: u64,
    /// When false, all TT probe and store sites are skipped — useful for
    /// equality tests that compare with-TT and no-TT results.
    pub tt_enabled: bool,
    /// Killer-move slots indexed by ply (depth from the root). Each ply
    /// keeps the two most recent moves that caused a β-cutoff there, which
    /// are tried first at sibling nodes at the same ply.
    pub killers: Vec<[Option<M>; 2]>,
    /// History heuristic: `history[mv.history_index()] += depth²` on every
    /// β-cutoff. Used as a tie-breaker after the priority and killers.
    pub history: Vec<i32>,
    /// Times a killer slot match was found and lifted in move ordering.
    pub killer_hits: u64,
    /// When false, killer + history ordering is bypassed (TT-only mode).
    /// Useful for benchmarks that isolate the new heuristics' contribution.
    pub ordering_enabled: bool,
    /// Wall-clock deadline for the search, derived from `params.time_budget`.
    /// `None` means no deadline.
    pub deadline: Option<Instant>,
    /// Set by `negamax` once the deadline has been crossed. Once true, every
    /// recursive call short-circuits with `0.0` (the value is meaningless;
    /// the iterative-deepening driver discards the partial iteration).
    pub aborted: bool,
    _phantom: PhantomData<fn() -> *const G>,
}

impl<M: Copy + MoveOrdering<G>, G: ?Sized> SearchContext<M, G> {
    pub fn new(params: AlphaBetaParams) -> Self {
        Self {
            params,
            nodes_visited: 0,
            tt: TranspositionTable::new(),
            tt_hits: 0,
            tt_order_hits: 0,
            tt_enabled: true,
            killers: Vec::new(),
            history: vec![0i32; <M as MoveOrdering<G>>::HISTORY_SIZE],
            killer_hits: 0,
            ordering_enabled: true,
            deadline: None,
            aborted: false,
            _phantom: PhantomData,
        }
    }

    /// Whether the search has been signalled to abort (deadline elapsed).
    #[inline]
    pub fn is_aborted(&self) -> bool {
        self.aborted
    }
}

/// Score a position for the current player from a terminal `WonBy` outcome.
/// `Draw` and `Ongoing` outcomes are deliberately handled elsewhere — `Draw`
/// delegates to `eval` so a game-specific evaluator can distinguish "real"
/// draws (e.g. mutual queen surround) from undesirable ones (repetition,
/// stalling) by returning a penalty.
fn won_score<G: Game>(game: &G, winner: crate::game::Player) -> f32 {
    if winner == game.next_player() {
        WIN_SCORE
    } else {
        -WIN_SCORE
    }
}

/// Negamax alphabeta on a mutable game state. Mutates `game` during recursion
/// (via `play_move`/`undo`) but always restores it before returning.
///
/// `ply` is the depth from the root (root child = 1). Used to index killer
/// slots so cutoffs at the same ply share their best move across the tree.
fn negamax<G, F>(
    game: &mut G,
    depth: u32,
    ply: u32,
    mut alpha: f32,
    beta: f32,
    eval: &mut F,
    ctx: &mut SearchContext<G::Move, G>,
) -> f32
where
    G: Game + Undoable + TranspositionKey,
    G::Move: PartialEq + MoveOrdering<G>,
    F: FnMut(&G) -> f32,
{
    ctx.nodes_visited += 1;

    // Deadline poll: cheap bitmask + an `Instant::now()` every 1024 nodes.
    // Once aborted, every recursive call returns immediately so the in-flight
    // iteration unwinds without further work; the iterative-deepening driver
    // discards the partial result and falls back to the previous depth.
    if ctx.aborted {
        return 0.0;
    }
    if (ctx.nodes_visited & DEADLINE_POLL_MASK) == 0 {
        if let Some(deadline) = ctx.deadline {
            if Instant::now() >= deadline {
                ctx.aborted = true;
                return 0.0;
            }
        }
    }

    match game.outcome() {
        Outcome::WonBy(winner) => return won_score(game, winner),
        // Draw: let the eval function decide. A neutral eval returns 0 (the
        // standard alphabeta convention); a Hive-style eval can return a
        // penalty for repetition draws so the bot avoids stalling.
        Outcome::Draw => return eval(game),
        Outcome::Ongoing => {}
    }
    if depth == 0 {
        return eval(game);
    }

    // TT probe: a cached score from a search at depth >= our remaining depth
    // is at least as informed as what we'd compute now. Bound flags constrain
    // when the cached score is *usable* given the current (alpha, beta):
    // Exact is always usable; LowerBound only when it already meets beta;
    // UpperBound only when it's already at or below alpha.
    //
    // Even on a non-returnable hit we keep the cached `best_move` so it can
    // be tried first by the moves loop — the same move that worked at
    // shallower depth is overwhelmingly likely to be best again, which makes
    // alphaβ pruning much more effective.
    let tt_key = if ctx.tt_enabled { game.tt_key() } else { 0 };
    let mut tt_move: Option<G::Move> = None;
    if ctx.tt_enabled {
        if let Some(entry) = ctx.tt.probe(tt_key) {
            if entry.depth >= depth {
                let usable = match entry.flag {
                    BoundFlag::Exact => true,
                    BoundFlag::LowerBound => entry.score >= beta,
                    BoundFlag::UpperBound => entry.score <= alpha,
                };
                if usable {
                    ctx.tt_hits += 1;
                    return entry.score;
                }
            }
            tt_move = entry.best_move;
        }
    }

    let mut moves = game.valid_moves();
    if moves.is_empty() {
        // Forced pass — recurse with one ply consumed and flipped sign.
        let pass = G::pass_move();
        game.play_move(&pass).expect("pass must be playable");
        let score = -negamax(game, depth.saturating_sub(1), ply + 1, -beta, -alpha, eval, ctx);
        game.undo();
        return score;
    }

    // Move ordering: lift the TT-suggested move (if any and still legal) to
    // the front of the list so it's tried first.
    let mut sort_start = 0usize;
    if let Some(mv) = tt_move {
        if let Some(idx) = moves.iter().position(|m| *m == mv) {
            if idx != 0 {
                moves.swap(0, idx);
            }
            ctx.tt_order_hits += 1;
            sort_start = 1;
        }
    }

    // Three-tier ordering for everything after the TT move (smaller key first):
    //   1. `priority(game)` — game-specific static signal. For games that
    //      don't override this it returns 0, so all moves stay in the same
    //      priority bucket and the killer/history order takes over.
    //   2. Killer tier (0 for slot 0, 1 for slot 1, 2 for everyone else) —
    //      within-priority-bucket preference for moves that produced a
    //      β-cutoff at this exact ply across siblings.
    //   3. Negated history score — accumulated β-cutoff bonus across the
    //      whole tree. Tiebreaker after both above.
    // Tuple sort handles lexicographic comparison naturally.
    if ctx.ordering_enabled && sort_start < moves.len() {
        let killers = ctx.killers.get(ply as usize).copied().unwrap_or([None, None]);
        moves[sort_start..].sort_by_key(|mv| {
            let prio = mv.priority(game);
            let killer_tier = if Some(*mv) == killers[0] {
                0i32
            } else if Some(*mv) == killers[1] {
                1
            } else {
                2
            };
            let hist = match mv.history_index() {
                Some(idx) => -(ctx.history[idx] as i64),
                None => 0,
            };
            (prio, killer_tier, hist)
        });
        if Some(moves[sort_start]) == killers[0] || Some(moves[sort_start]) == killers[1] {
            ctx.killer_hits += 1;
        }
    }

    // Snapshot the alpha that was passed in, before the moves loop mutates
    // it. A `best` that fails to exceed this original alpha is a fail-low
    // (only an upper bound on the true score).
    let alpha_orig = alpha;

    let mut best = f32::NEG_INFINITY;
    let mut best_move: Option<G::Move> = None;
    let mut cutoff = false;
    let mut cutoff_move: Option<G::Move> = None;
    let player_before = game.next_player();
    for mv in moves.iter() {
        game.play_move(mv).expect("valid_moves returned an unplayable move");
        // Same-player edges (e.g. Yinsh ClaimRow chains, Zertz mid-capture)
        // keep the score in the same player's frame: don't flip the sign or
        // negate the (alpha, beta) window. Player-changing edges are the
        // standard negamax case.
        let score = if game.next_player() == player_before {
            negamax(game, depth - 1, ply + 1, alpha, beta, eval, ctx)
        } else {
            -negamax(game, depth - 1, ply + 1, -beta, -alpha, eval, ctx)
        };
        game.undo();

        if score > best {
            best = score;
            best_move = Some(*mv);
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            cutoff = true;
            cutoff_move = Some(*mv);
            break; // beta cutoff
        }
    }

    // Killer + history bookkeeping on β-cutoff. The cutoff move is what
    // refuted the parent's choice; remembering it tightens future siblings'
    // ordering. Skip pass and any move whose `history_index` is None.
    if cutoff {
        if let Some(mv) = cutoff_move {
            // Killer: shift down [0] -> [1] unless the new move is already [0].
            while ctx.killers.len() <= ply as usize {
                ctx.killers.push([None, None]);
            }
            let slots = &mut ctx.killers[ply as usize];
            if slots[0] != Some(mv) {
                slots[1] = slots[0];
                slots[0] = Some(mv);
            }
            // History: depth² bonus.
            if let Some(idx) = mv.history_index() {
                let bonus = (depth as i32) * (depth as i32);
                ctx.history[idx] = ctx.history[idx].saturating_add(bonus);
            }
        }
    }

    // TT store: classify the score against the (alpha_orig, beta) window.
    // A beta cutoff means `best >= beta`, so the true value is at least
    // `best` — a lower bound. If the loop completed and `best` improved on
    // alpha_orig, the score is exact. Otherwise no move beat alpha_orig and
    // the score is only an upper bound. Cacheability gate still applies for
    // path-dependent positions (e.g. Hive `DrawByRepetition`).
    if ctx.tt_enabled && game.tt_cacheable() {
        let flag = if cutoff {
            BoundFlag::LowerBound
        } else if best > alpha_orig {
            BoundFlag::Exact
        } else {
            BoundFlag::UpperBound
        };
        ctx.tt.store(tt_key, TtEntry { depth, score: best, flag, best_move });
    }

    best
}

/// Find the best move for the player to move via iterative deepening up to
/// `params.max_depth`. Returns `(best_move, score)` from the current player's
/// perspective.
///
/// Iterative deepening searches depths 1, 2, …, max_depth in sequence, with
/// the same `SearchContext` (and TT) carried across iterations. Each shallow
/// iteration's TT entries seed the next iteration's move ordering, which
/// makes alphaβ pruning much more effective at the deepest depth — the total
/// cost of all iterations is typically less than a direct max_depth search
/// from an empty TT, and significantly less when TT reuse is good.
///
/// Mutates `game` during the search but restores it before returning, so the
/// caller can pass `&mut canonical` if they own it. Most callers will pass a
/// clone to keep canonical state untouched.
pub fn alphabeta_best_move<G, F>(
    game: &mut G,
    eval: &mut F,
    ctx: &mut SearchContext<G::Move, G>,
) -> (Option<G::Move>, f32)
where
    G: Game + Undoable + TranspositionKey,
    G::Move: PartialEq + MoveOrdering<G>,
    F: FnMut(&G) -> f32,
{
    match game.outcome() {
        Outcome::WonBy(winner) => return (None, won_score(game, winner)),
        Outcome::Draw => return (None, eval(game)),
        Outcome::Ongoing => {}
    }

    let mut moves = game.valid_moves();
    if moves.is_empty() {
        return (Some(G::pass_move()), 0.0);
    }

    let max_depth = ctx.params.max_depth;
    if max_depth == 0 {
        return (Some(moves[0]), eval(game));
    }

    // Set the deadline for this whole search. Done here (not in `new`) so a
    // single `SearchContext` could in principle be reused across calls, but
    // in practice each call constructs its own.
    if let Some(budget) = ctx.params.time_budget {
        ctx.deadline = Some(Instant::now() + budget);
    }
    ctx.aborted = false;

    let mut best_move = moves[0];
    let mut best_score = f32::NEG_INFINITY;

    // Iterative deepening: 1, 2, ..., max_depth. After each iteration, the
    // root-level `best_move` is lifted to position 0 of the move list so the
    // next deeper iteration tries it first — usually triggering immediate
    // alpha tightening that prunes the rest of the root moves much faster.
    // The remaining root moves are sorted by descending history so the
    // strongest-looking alternatives narrow alpha next, before the long tail.
    //
    // When a `time_budget` is set, `negamax` flips `ctx.aborted` once the
    // deadline elapses. We discard the partial iteration's `iter_best_move`
    // and return the deepest fully-completed iteration's result. Depth 1 is
    // cheap enough to always finish on any realistic budget; deeper
    // iterations are best-effort.
    for depth in 1..=max_depth {
        if let Some(idx) = moves.iter().position(|m| *m == best_move) {
            if idx != 0 {
                moves.swap(0, idx);
            }
        }
        if ctx.ordering_enabled && moves.len() > 1 {
            moves[1..].sort_by_key(|mv| {
                let prio = mv.priority(game);
                let hist = match mv.history_index() {
                    Some(idx) => -(ctx.history[idx] as i64),
                    None => 0,
                };
                (prio, hist)
            });
        }

        let mut alpha = f32::NEG_INFINITY;
        let beta = f32::INFINITY;
        let mut iter_best_move = moves[0];
        let mut iter_best_score = f32::NEG_INFINITY;
        let player_before = game.next_player();

        for mv in moves.iter() {
            game.play_move(mv).expect("valid_moves returned an unplayable move");
            let score = if game.next_player() == player_before {
                negamax(game, depth - 1, 1, alpha, beta, eval, ctx)
            } else {
                -negamax(game, depth - 1, 1, -beta, -alpha, eval, ctx)
            };
            game.undo();

            if ctx.aborted {
                break;
            }

            if score > iter_best_score {
                iter_best_score = score;
                iter_best_move = *mv;
            }
            if score > alpha {
                alpha = score;
            }
        }

        if ctx.aborted {
            // Discard partial iteration: keep the best move from the deepest
            // fully-completed depth (or, on the first iteration, the
            // pre-search default of moves[0]).
            break;
        }

        best_move = iter_best_move;
        best_score = iter_best_score;
    }

    (Some(best_move), best_score)
}

/// Like `alphabeta_best_move`, but returns the exact minimax score for *every*
/// root move at `params.max_depth` — the data needed to derive a full policy
/// distribution rather than just the argmax.
///
/// Iterative deepening 1..max_depth-1 runs the standard pruned search to seed
/// the TT and history heuristic. The final iteration searches each root move
/// with a full `(−∞, +∞)` window so the returned scores are true minimax
/// values rather than fail-low upper bounds (which they would be for any move
/// that doesn't beat the running alpha under a normal root loop). Each
/// subtree still benefits from TT and move ordering, so the cost increase
/// over `alphabeta_best_move` is moderate, not multiplicative-in-N.
///
/// Scores are from the perspective of the player to move at the root
/// (positive = good for that player). Returns an empty Vec if the game is
/// already terminal, and `vec![(pass, 0.0)]` if the only option is a pass.
pub fn alphabeta_root_scores<G, F>(
    game: &mut G,
    eval: &mut F,
    ctx: &mut SearchContext<G::Move, G>,
) -> Vec<(G::Move, f32)>
where
    G: Game + Undoable + TranspositionKey,
    G::Move: PartialEq + MoveOrdering<G>,
    F: FnMut(&G) -> f32,
{
    match game.outcome() {
        Outcome::WonBy(_) | Outcome::Draw => return Vec::new(),
        Outcome::Ongoing => {}
    }

    let mut moves = game.valid_moves();
    if moves.is_empty() {
        return vec![(G::pass_move(), 0.0)];
    }

    let max_depth = ctx.params.max_depth;
    if max_depth == 0 {
        let score = eval(game);
        return moves.iter().map(|&mv| (mv, score)).collect();
    }

    let mut best_move = moves[0];

    // ID iterations 1..max_depth-1: standard pruned search to seed TT/ordering.
    // Skipped when max_depth == 1.
    for depth in 1..max_depth {
        if let Some(idx) = moves.iter().position(|m| *m == best_move) {
            if idx != 0 {
                moves.swap(0, idx);
            }
        }
        if ctx.ordering_enabled && moves.len() > 1 {
            moves[1..].sort_by_key(|mv| {
                let prio = mv.priority(game);
                let hist = match mv.history_index() {
                    Some(idx) => -(ctx.history[idx] as i64),
                    None => 0,
                };
                (prio, hist)
            });
        }

        let mut alpha = f32::NEG_INFINITY;
        let beta = f32::INFINITY;
        let mut iter_best = moves[0];
        let mut iter_best_score = f32::NEG_INFINITY;
        let player_before = game.next_player();
        for mv in moves.iter() {
            game.play_move(mv).expect("valid_moves returned an unplayable move");
            let score = if game.next_player() == player_before {
                negamax(game, depth - 1, 1, alpha, beta, eval, ctx)
            } else {
                -negamax(game, depth - 1, 1, -beta, -alpha, eval, ctx)
            };
            game.undo();
            if score > iter_best_score {
                iter_best_score = score;
                iter_best = *mv;
            }
            if score > alpha {
                alpha = score;
            }
        }
        best_move = iter_best;
    }

    // Final iteration: full window per root move so every score is exact.
    if let Some(idx) = moves.iter().position(|m| *m == best_move) {
        if idx != 0 {
            moves.swap(0, idx);
        }
    }
    if ctx.ordering_enabled && moves.len() > 1 {
        moves[1..].sort_by_key(|mv| {
            let prio = mv.priority(game);
            let hist = match mv.history_index() {
                Some(idx) => -(ctx.history[idx] as i64),
                None => 0,
            };
            (prio, hist)
        });
    }

    let mut scores: Vec<(G::Move, f32)> = Vec::with_capacity(moves.len());
    let player_before = game.next_player();
    for mv in moves.iter() {
        game.play_move(mv).expect("valid_moves returned an unplayable move");
        let score = if game.next_player() == player_before {
            negamax(
                game,
                max_depth - 1,
                1,
                f32::NEG_INFINITY,
                f32::INFINITY,
                eval,
                ctx,
            )
        } else {
            -negamax(
                game,
                max_depth - 1,
                1,
                f32::NEG_INFINITY,
                f32::INFINITY,
                eval,
                ctx,
            )
        };
        game.undo();
        scores.push((*mv, score));
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Player;
    use crate::symmetry::UnitSymmetry;

    /// A tiny deterministic game for unit-testing the alphabeta driver:
    /// players take turns picking a number 1..=3 to subtract from a counter.
    /// Whoever takes the counter to exactly 0 loses (analog of misère Nim).
    #[derive(Clone, Debug)]
    struct CountdownGame {
        counter: u8,
        history: Vec<u8>,
        player: Player,
    }

    impl CountdownGame {
        fn new(start: u8) -> Self {
            Self { counter: start, history: Vec::new(), player: Player::Player1 }
        }
    }

    impl Game for CountdownGame {
        type Move = u8;
        type Symmetry = UnitSymmetry;

        fn next_player(&self) -> Player { self.player }
        fn outcome(&self) -> Outcome {
            if self.counter == 0 {
                // Whoever just moved took it to 0; that player loses.
                Outcome::WonBy(self.player)
            } else {
                Outcome::Ongoing
            }
        }
        fn valid_moves(&mut self) -> Vec<u8> {
            if self.counter == 0 { return vec![]; }
            (1..=3).filter(|&n| n <= self.counter).collect()
        }
        fn play_move(&mut self, mv: &u8) -> Result<(), String> {
            self.history.push(self.counter);
            self.counter -= mv;
            self.player = self.player.opposite();
            Ok(())
        }
        fn pass_move() -> u8 { 0 }
        fn is_pass(mv: &u8) -> bool { *mv == 0 }
    }

    impl Undoable for CountdownGame {
        fn undo(&mut self) {
            self.counter = self.history.pop().expect("undo with empty history");
            self.player = self.player.opposite();
        }
    }

    impl TranspositionKey for CountdownGame {
        fn tt_key(&self) -> u64 {
            let player_bit = matches!(self.player, Player::Player2) as u64;
            (self.counter as u64) | (player_bit << 32)
        }
    }

    impl MoveOrdering<CountdownGame> for u8 {
        const HISTORY_SIZE: usize = 4; // moves are 1..=3, with 0 == pass
        fn history_index(&self) -> Option<usize> {
            if *self == 0 { None } else { Some(*self as usize) }
        }
        // priority defaults to 0 — countdown has no static signal worth using.
    }

    #[test]
    fn negamax_finds_winning_move_at_counter_4() {
        // counter=4: the player to move wants to leave counter=1 for the opponent
        // (forcing them to take the last 1 and lose), so the winning move is 3.
        let mut game = CountdownGame::new(4);
        let mut ctx = SearchContext::new(AlphaBetaParams::new(6));
        let mut eval = |_: &CountdownGame| 0.0;
        let (mv, score) = alphabeta_best_move(&mut game, &mut eval, &mut ctx);
        assert_eq!(mv, Some(3));
        assert!(score > 0.0, "expected winning score, got {score}");
    }

    #[test]
    fn undo_restores_state() {
        let mut game = CountdownGame::new(5);
        let mut ctx = SearchContext::new(AlphaBetaParams::new(4));
        let mut eval = |_: &CountdownGame| 0.0;
        let snapshot_counter = game.counter;
        let snapshot_player = game.player;
        let _ = alphabeta_best_move(&mut game, &mut eval, &mut ctx);
        assert_eq!(game.counter, snapshot_counter);
        assert_eq!(game.player, snapshot_player);
        assert!(game.history.is_empty());
    }

    #[test]
    fn root_scores_one_per_legal_move() {
        let mut game = CountdownGame::new(4);
        let mut ctx = SearchContext::new(AlphaBetaParams::new(4));
        let mut eval = |_: &CountdownGame| 0.0;
        let scores = alphabeta_root_scores(&mut game, &mut eval, &mut ctx);
        // counter=4 → legal moves are 1, 2, 3.
        assert_eq!(scores.len(), 3);
        for &mv in &[1u8, 2, 3] {
            assert!(scores.iter().any(|(m, _)| *m == mv), "missing score for move {mv}");
        }
    }

    #[test]
    fn root_scores_best_matches_alphabeta_best_move() {
        // Score-level invariant: the score `alphabeta_best_move` reports for
        // its chosen best move must equal that move's entry in
        // `alphabeta_root_scores`, and must equal the maximum score in the
        // full root-scores list. (Move-level equality doesn't hold under
        // ties — `>` vs `Iterator::max_by` break ties differently.)
        for start in [3u8, 4, 5, 6, 7] {
            for depth in [1u32, 2, 3, 4] {
                let mut g1 = CountdownGame::new(start);
                let mut ctx1 = SearchContext::new(AlphaBetaParams::new(depth));
                let mut eval = |_: &CountdownGame| 0.0;
                let (best_mv, best_score) = alphabeta_best_move(&mut g1, &mut eval, &mut ctx1);

                let mut g2 = CountdownGame::new(start);
                let mut ctx2 = SearchContext::new(AlphaBetaParams::new(depth));
                let scores = alphabeta_root_scores(&mut g2, &mut eval, &mut ctx2);

                let max_root_score = scores
                    .iter()
                    .map(|(_, s)| *s)
                    .fold(f32::NEG_INFINITY, f32::max);
                assert!(
                    (best_score - max_root_score).abs() < 1e-3,
                    "best_move score {best_score} != max root_scores score {max_root_score} \
                     at start={start} depth={depth}",
                );

                let chosen = scores
                    .iter()
                    .find(|(m, _)| *m == best_mv.unwrap())
                    .expect("alphabeta best move must appear in root_scores");
                assert!(
                    (chosen.1 - best_score).abs() < 1e-3,
                    "score for chosen move {:?} diverged: root_scores={}, best_move={best_score} \
                     at start={start} depth={depth}",
                    chosen.0, chosen.1,
                );
            }
        }
    }

    #[test]
    fn root_scores_undo_restores_state() {
        let mut game = CountdownGame::new(6);
        let snapshot_counter = game.counter;
        let snapshot_player = game.player;
        let mut ctx = SearchContext::new(AlphaBetaParams::new(4));
        let mut eval = |_: &CountdownGame| 0.0;
        let _ = alphabeta_root_scores(&mut game, &mut eval, &mut ctx);
        assert_eq!(game.counter, snapshot_counter);
        assert_eq!(game.player, snapshot_player);
        assert!(game.history.is_empty());
    }
}
