/// MCTS search implementation with arena allocation and batch NN evaluation.
/// Game states are reconstructed by replaying moves from the root, not stored per node.

use super::arena::{NodeArena, NodeId};
use super::node::{MctsNode, Proven};
use crate::game::{GameEngine, Outcome, Player, PolicyIndex};

const DEFAULT_C_PUCT: f32 = 1.5;

/// Diagnostic snapshot for a single root child. Returned by
/// `MctsSearch::root_child_stats`. Lets callers dump per-move visit counts,
/// Q values, raw priors, and terminal status in one pass without poking at
/// the arena directly.
pub struct RootChildStat<M: Copy> {
    pub move_from_parent: M,
    pub visit_count: u32,
    /// Q from the root player's perspective (positive = good move for root).
    pub value: f32,
    /// Raw NN policy_prior; Dirichlet noise NOT mixed in.
    pub policy_prior: f32,
    /// Outcome of the state reached by this move (Ongoing if not terminal).
    pub outcome: Outcome,
}

#[derive(Clone)]
pub enum CpuctStrategy {
    Constant { c_puct: f32 },
    Dynamic { c_init: f32, c_base: f32 },
}

#[derive(Clone)]
pub enum RootNoise {
    None,
    Dirichlet { alpha: f32, epsilon: f32 },
}

#[derive(Clone, PartialEq)]
pub enum ForcedExploration {
    None,
    /// selection_k: how aggressively to force playouts during tree search (typical: 0.5)
    /// pruning_k: how aggressively to remove forced playouts from training targets (typical: 2.0)
    Soft { selection_k: f32, pruning_k: f32 }
}

#[derive(Clone)]
pub struct SearchParams {
    pub cpuct_strategy: CpuctStrategy,
    pub forced_exploration: ForcedExploration,
    pub root_noise: RootNoise,
    /// Cap children per node expansion to this many (top by policy score).
    /// Defaults to usize::MAX (no cap). Set to the simulation count to avoid
    /// allocating children that can never be visited.
    pub max_children: usize,
    /// Draw contempt: scalar value = W - L - contempt * D.
    /// Terminal draws return -contempt. Default 0.0 (draws are neutral at 0).
    pub draw_contempt: f32,
    /// Asymmetric contempt: when `Some(player)`, `draw_contempt` is applied
    /// only at nodes where that player chose the move (i.e. the parent's
    /// `turn_player == player`); the opponent's nodes use contempt 0. When
    /// `None`, `draw_contempt` is applied symmetrically at every node — the
    /// historical behavior. See `effective_contempt`.
    pub contempt_side: Option<Player>,
}

impl SearchParams {
    pub fn new(cpuct_strategy: CpuctStrategy, forced_exploration: ForcedExploration, root_noise: RootNoise) -> Self {
        Self { cpuct_strategy, forced_exploration, root_noise, max_children: usize::MAX, draw_contempt: 0.0, contempt_side: None }
    }

    pub fn inference(cpuct_strategy: CpuctStrategy) -> Self {
        Self {
            cpuct_strategy,
            forced_exploration: ForcedExploration::None,
            root_noise: RootNoise::None,
            max_children: usize::MAX,
            draw_contempt: 0.0,
            contempt_side: None,
        }
    }
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            cpuct_strategy: CpuctStrategy::Constant { c_puct: DEFAULT_C_PUCT },
            forced_exploration: ForcedExploration::None,
            root_noise: RootNoise::None,
            max_children: usize::MAX,
            draw_contempt: 0.0,
            contempt_side: None,
        }
    }
}

/// Effective contempt scalar to apply when scoring children chosen by
/// `parent_player`. Asymmetric mode returns 0 for the side that is not the
/// designated contempt side, so that side's MCTS subtree models the
/// contempted player's responses without any draw-aversion bias — and vice
/// versa, the contempted side's tree correctly models the opponent as
/// playing straight Q without contempt.
#[inline]
fn effective_contempt(params: &SearchParams, parent_player: Player) -> f32 {
    match params.contempt_side {
        None => params.draw_contempt,
        Some(side) if side == parent_player => params.draw_contempt,
        Some(_) => 0.0,
    }
}

/// Extract the Dirichlet epsilon from search params (0.0 if no noise).
#[inline]
fn dir_epsilon(params: &SearchParams) -> f32 {
    match params.root_noise {
        RootNoise::Dirichlet { epsilon, .. } => epsilon,
        RootNoise::None => 0.0,
    }
}

/// UCB score for child selection.
/// `node.value(contempt)` is from the parent's player's perspective, so it is
/// added directly without sign adjustment. See docs/mcts_value_convention.md.
/// `parent_player` is the player whose turn it is at the parent node; the
/// effective contempt depends on whether that player is the asymmetric
/// contempt side (see `effective_contempt`).
fn ucb_score<M: Copy>(node: &MctsNode<M>, parent_player: Player, parent_visits: u32, params: &SearchParams) -> f32 {
    let c_puct = calculate_cpuct(params, parent_visits);
    let eps = dir_epsilon(params);
    let contempt = effective_contempt(params, parent_player);
    calculate_ucb_score(node, c_puct, parent_visits, eps, contempt)
}

#[inline]
fn calculate_ucb_score<M: Copy>(node: &MctsNode<M>, c_puct: f32, parent_visits: u32, eps: f32, contempt: f32) -> f32 {
    calculate_ucb_score_parts(node.value(contempt), node.prior(eps), node.visit_count, c_puct, parent_visits)
}

#[inline]
pub fn calculate_ucb_score_parts(
    value: f32,
    prior: f32,
    visit_count: u32,
    c_puct: f32,
    parent_visits: u32,
) -> f32 {
    let parent = parent_visits as f32;
    let exploration = calculate_ucb_exploration(prior, visit_count, c_puct, parent);
    value + exploration
}

#[inline]
pub fn calculate_ucb_exploration(prior: f32, visit_count: u32, c_puct: f32, parent_visits: f32) -> f32 {
    c_puct * prior * parent_visits.sqrt() / (1.0 + visit_count as f32)
}

#[inline]
fn calculate_cpuct(params: &SearchParams, parent_visits: u32) -> f32 {
    let parent = parent_visits as f32;
    match params.cpuct_strategy {
        CpuctStrategy::Constant { c_puct } => c_puct,
        CpuctStrategy::Dynamic { c_init, c_base } => c_init + ((parent + c_base + 1.0) / c_base).ln(),
    }
}

/// Select the best child by UCB score.
fn best_child<M: Copy>(arena: &NodeArena<M>, node_id: NodeId, params: &SearchParams, eps: f32) -> NodeId {
    let node = arena.get(node_id);
    let parent_visits = node.visit_count;
    let c_puct = calculate_cpuct(params, parent_visits);
    let contempt = effective_contempt(params, node.turn_player);
    let mut best_id = node.first_child.expect("no children");
    let mut best_score = calculate_ucb_score(arena.get(best_id), c_puct, parent_visits, eps, contempt);

    let mut current = arena.get(best_id).next_sibling;
    while let Some(child_id) = current {
        let child = arena.get(child_id);
        let score = calculate_ucb_score(child, c_puct, parent_visits, eps, contempt);
        if score > best_score {
            best_score = score;
            best_id = child_id;
        }
        current = child.next_sibling;
    }

    best_id
}

/// Select a leaf node by traversing the tree.
/// If forced_playouts is true, at the root level, children with fewer visits
/// than their forced minimum get infinite urgency (KataGo forced playouts).
/// Returns (leaf_node_id, depth) where depth is the number of edges from root.
fn select_leaf<M: Copy>(arena: &NodeArena<M>, root: NodeId, params: &SearchParams) -> (NodeId, u32) {
    let eps = dir_epsilon(params);
    let mut node_id = root;
    let mut depth = 0u32;
    let mut is_root = true;
    loop {
        let node = arena.get(node_id);
        if !node.is_expanded || node.first_child.is_none() {
            return (node_id, depth);
        }
        if is_root {
            node_id = match params.forced_exploration {
                ForcedExploration::None => best_child(arena, node_id, params, eps),
                ForcedExploration::Soft { selection_k, .. } => best_child_with_forced(arena, node_id, params, eps, selection_k),
            };
        } else {
            node_id = best_child(arena, node_id, params, eps);
        }
        depth += 1;
        is_root = false;
    }
}

/// Select best child at root, forcing minimum playouts on children that have
/// been visited at least once. A child with visit_count < n_forced gets
/// infinite urgency so it is always selected first.
/// n_forced(c) = k * sqrt(P_noised(c) * N_total), k=2
fn best_child_with_forced<M: Copy>(arena: &NodeArena<M>, node_id: NodeId, params: &SearchParams, eps: f32, k: f32) -> NodeId {
    let node = arena.get(node_id);
    let parent_visits = node.visit_count;
    let parent_player = node.turn_player;
    let n_total = parent_visits as f32;

    let mut best_id = node.first_child.expect("no children");
    let mut best_score = child_score_with_forced(arena.get(best_id), parent_player, parent_visits, n_total, params, eps, k);

    let mut current = arena.get(best_id).next_sibling;
    while let Some(child_id) = current {
        let child = arena.get(child_id);
        let score = child_score_with_forced(child, parent_player, parent_visits, n_total, params, eps, k);
        if score > best_score {
            best_score = score;
            best_id = child_id;
        }
        current = child.next_sibling;
    }

    best_id
}

/// Score for a root child with forced playout logic.
/// If the child has been visited but is below its forced minimum, return infinity.
fn child_score_with_forced<M: Copy>(child: &MctsNode<M>, parent_player: Player, parent_visits: u32, n_total: f32, params: &SearchParams, eps: f32, k: f32) -> f32 {
    let mut score = ucb_score(child, parent_player, parent_visits, params);
    if child.visit_count > 0 {
        let n_forced = (k * (child.prior(eps) * n_total).sqrt()) as u32;
        if child.visit_count < n_forced {
            score = f32::INFINITY;
        }
    }
    score
}

/// Backpropagate a (W−L, D) pair up the tree.
/// `value` is the zero-sum W−L component from the perspective of the player to
/// move at `node_id`; sign-flips on every player boundary, exactly like a
/// regular zero-sum scalar. `draw` is the symmetric draw-probability component
/// and is added unflipped at every ancestor — both players see a draw with the
/// same magnitude, so contempt only enters at `value(contempt)` evaluation time.
/// Splitting the two components is what makes draw contempt mathematically
/// correct under the zero-sum sign-flip convention. See
/// docs/mcts_value_convention.md.
fn backpropagate<M: Copy>(arena: &mut NodeArena<M>, node_id: NodeId, value: f32, draw: f32) {
    // Invariant: `value` is the W−L component from the current node's player's perspective.
    // Transform to parent's perspective before storing, carry that value up.
    let mut node_id = node_id;
    let mut value = value;
    loop {
        let (parent, store_value) = {
            let node = arena.get(node_id);
            let sv = match node.parent {
                None => -value, // root convention: root_value() negates to recover root player's return
                Some(pid) => {
                    if arena.get(pid).turn_player == node.turn_player { value } else { -value }
                }
            };
            (node.parent, sv)
        };
        let node = arena.get_mut(node_id);
        node.visit_count += 1;
        node.value_sum += store_value;
        node.draw_sum += draw;
        match parent {
            None => break,
            Some(pid) => { value = store_value; node_id = pid; }
        }
    }
}

/// Apply virtual loss: increment visit_count and subtract 1.0 from value_sum at every
/// node up the tree. Subtracting 1.0 is pessimistic from every node's parent's perspective
/// regardless of player, deterring subsequent batch selections from taking the same path.
/// The placeholder is corrected by correct_virtual_loss.
fn apply_virtual_loss<M: Copy>(arena: &mut NodeArena<M>, mut node_id: NodeId) {
    loop {
        let node = arena.get_mut(node_id);
        node.visit_count += 1;
        node.value_sum -= 1.0;
        match node.parent {
            Some(parent) => node_id = parent,
            None => break,
        }
    }
}

/// Correct virtual loss by replacing the -1.0 placeholder with the real backed-up
/// (W−L, D) pair. Does NOT increment visit_count (already done by apply_virtual_loss).
/// `real_value` is the W−L component from the perspective of the player to move at
/// `node_id`; `real_draw` is the symmetric draw component (added unflipped at every
/// ancestor, mirroring `backpropagate`). Uses the same player-aware sign logic as
/// backpropagate for the W−L component.
fn correct_virtual_loss<M: Copy>(arena: &mut NodeArena<M>, node_id: NodeId, real_value: f32, real_draw: f32) {
    let mut node_id = node_id;
    let mut value = real_value;
    loop {
        let (parent, store_value) = {
            let node = arena.get(node_id);
            let sv = match node.parent {
                None => -value,
                Some(pid) => {
                    if arena.get(pid).turn_player == node.turn_player { value } else { -value }
                }
            };
            (node.parent, sv)
        };
        let node = arena.get_mut(node_id);
        // +1.0 undoes the apply_virtual_loss placeholder; store_value adds the real backed-up W−L.
        node.value_sum += 1.0 + store_value;
        // apply_virtual_loss did not touch draw_sum, so just add the real draw.
        node.draw_sum += real_draw;
        match parent {
            None => break,
            Some(pid) => { value = store_value; node_id = pid; }
        }
    }
}

/// Set the proven flag on a freshly-detected terminal leaf, then walk up the
/// tree setting `proven` on every ancestor whose outcome can now be deduced.
///
/// Propagation stops as soon as an ancestor's outcome is still uncertain (some
/// child unproven and no proven win-for-parent yet) — this is monotonic, so a
/// later terminal hit can resume propagation from where this run stopped.
///
/// `same_player` (parent vs. child) is checked per edge so this code works for
/// games with consecutive same-player turns (Yinsh ClaimRow, Zertz mid-capture).
fn propagate_proven<M: Copy>(arena: &mut NodeArena<M>, leaf: NodeId, leaf_proven: Proven) {
    arena.get_mut(leaf).proven = Some(leaf_proven);

    let mut current = leaf;
    loop {
        let parent_id = match arena.get(current).parent {
            None => return,
            Some(p) => p,
        };
        let parent_player = arena.get(parent_id).turn_player;

        // Aggregate child proven flags translated into parent's perspective.
        let mut any_win = false;
        let mut any_draw = false;
        let mut all_proven = true;
        let mut child_id = arena.get(parent_id).first_child;
        while let Some(cid) = child_id {
            let child = arena.get(cid);
            let same_player = parent_player == child.turn_player;
            match child.proven {
                Some(p) => {
                    let for_parent = if same_player { p } else { p.flip() };
                    match for_parent {
                        Proven::Win => { any_win = true; }
                        Proven::Draw => { any_draw = true; }
                        Proven::Loss => {} // bad for parent; doesn't help unless all are
                    }
                }
                None => { all_proven = false; }
            }
            if any_win { break; } // a single Win is enough for the parent
            child_id = child.next_sibling;
        }

        let new_proven = if any_win {
            Some(Proven::Win)
        } else if all_proven {
            if any_draw { Some(Proven::Draw) } else { Some(Proven::Loss) }
        } else {
            None
        };

        match new_proven {
            None => return,
            Some(p) => {
                if arena.get(parent_id).proven == Some(p) {
                    return; // unchanged; nothing to do upstream
                }
                arena.get_mut(parent_id).proven = Some(p);
                current = parent_id;
            }
        }
    }
}

/// Terminal game (W−L, D) pair from a perspective, mirroring the per-leaf
/// outputs of a WDL value head: a sure win gives (+1, 0), a sure loss gives
/// (−1, 0), a sure draw gives (0, 1). The two components are propagated
/// separately by `backpropagate` because the W−L component is zero-sum
/// (sign-flips on player change) while the D component is symmetric.
/// Contempt is applied at `node.value(contempt)` evaluation time, not here.
pub fn terminal_value(outcome: Outcome, perspective: Player) -> (f32, f32) {
    match outcome {
        Outcome::Ongoing => (0.0, 0.0),
        Outcome::Draw => (0.0, 1.0),
        Outcome::WonBy(winner) => {
            if winner == perspective { (1.0, 0.0) } else { (-1.0, 0.0) }
        }
    }
}

/// Expand a node with a policy vector, adding children to the arena.
/// `game` is the reconstructed game state at this node (not stored in the node).
/// `max_children` caps how many children are created (top by softmaxed policy score).
fn expand_with_policy<G: GameEngine>(
    arena: &mut NodeArena<G::Move>,
    node_id: NodeId,
    game: &mut G,
    policy: &[f32],
    max_children: usize,
) {
    arena.get_mut(node_id).is_expanded = true;
    // Cheap default: assume each child alternates the turn. This is wrong for
    // same-player chains (Yinsh ClaimRow), but child.turn_player is never
    // read until that child itself becomes a leaf — at which point
    // `select_leaves` reconstructs the game and fixes up the value to match
    // `game.next_player()`. See the fix-up call there for the invariant.
    let child_turn = game.next_player().opposite();

    let (_mask, indexed_moves) = game.get_legal_move_mask();
    if indexed_moves.is_empty() {
        // Pass — pass always alternates the turn.
        let child_id = arena.alloc(Some(node_id), G::pass_move(), 1.0, child_turn);
        arena.get_mut(node_id).first_child = Some(child_id);
        arena.get_mut(node_id).child_count = 1;
        return;
    }

    // Compute raw scores per legal move, then softmax over legal moves only.
    let mut scores: Vec<f32> = indexed_moves.iter()
        .map(|&(enc, _)| match enc {
            PolicyIndex::Single(idx) => policy[idx],
            PolicyIndex::Sum(a, b) => policy[a] + policy[b],
            PolicyIndex::DotProduct { q_offset, k_offset, src_cell, dst_cell, embed_dim, g2 } => {
                let mut dot = 0.0f32;
                for d in 0..embed_dim {
                    dot += policy[q_offset + d * g2 + src_cell] * policy[k_offset + d * g2 + dst_cell];
                }
                dot / (embed_dim as f32).sqrt()
            }
        })
        .collect();

    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0f32;
    for s in &mut scores {
        *s = (*s - max_score).exp();
        total += *s;
    }
    if total > 0.0 {
        for s in &mut scores {
            *s /= total;
        }
    }

    // Collect (score, move_index) pairs, optionally keeping only top-K by score.
    let n = indexed_moves.len();
    let keep = n.min(max_children);
    let mut order: Vec<usize> = (0..n).collect();
    if keep < n {
        // Partial sort: move the top-`keep` indices to the front (order within them is arbitrary).
        order.select_nth_unstable_by(keep - 1, |&a, &b| {
            scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(keep);
    }

    // Re-softmax over the kept subset so priors sum to 1.
    if keep < n {
        let kept_total: f32 = order.iter().map(|&i| scores[i]).sum();
        if kept_total > 0.0 {
            for &i in &order {
                scores[i] /= kept_total;
            }
        }
    }

    let mut first_child_id: Option<NodeId> = None;
    let mut prev_child_id: Option<NodeId> = None;

    for &i in &order {
        let (_, mv) = indexed_moves[i];
        let child_id = arena.alloc(Some(node_id), mv, scores[i], child_turn);

        if first_child_id.is_none() {
            first_child_id = Some(child_id);
        }
        if let Some(prev) = prev_child_id {
            arena.get_mut(prev).next_sibling = Some(child_id);
        }
        prev_child_id = Some(child_id);
    }

    arena.get_mut(node_id).first_child = first_child_id;
    arena.get_mut(node_id).child_count = keep as u16;
}

/// Single-game MCTS search engine, generic over any GameEngine.
/// Game states are reconstructed by replaying moves from root_game.
pub struct MctsSearch<G: GameEngine> {
    arena: NodeArena<G::Move>,
    root: NodeId,
    root_game: Option<G>,
    /// Accumulated leaves from select_leaves calls, consumed by expand_and_backprop.
    stashed_leaves: Vec<(NodeId, G)>,
    /// Whether to use forced playouts at the root (KataGo-style).
    pub params: SearchParams,
    /// Running depth stats across simulations since last take_depth_stats call.
    depth_sum: f64,
    depth_sum_sq: f64,
    depth_count: u64,
}

impl<G: GameEngine> MctsSearch<G> {
    pub fn new(capacity: usize) -> Self {
        MctsSearch {
            arena: NodeArena::new(capacity, G::pass_move()),
            root: 0, // will be set in init
            root_game: None,
            stashed_leaves: Vec::new(),
            params: SearchParams::default(),
            depth_sum: 0.0,
            depth_sum_sq: 0.0,
            depth_count: 0,
        }
    }

    /// Initialize search for a game position.
    pub fn init(&mut self, game: &G, policy: &[f32]) {
        self.arena.reset();
        self.stashed_leaves.clear();
        self.root_game = Some(game.clone());
        let root = self.arena.alloc(None, G::pass_move(), 0.0, game.next_player());
        self.root = root;
        let mut game_copy = game.clone();
        expand_with_policy::<G>(&mut self.arena, root, &mut game_copy, policy, self.params.max_children);
        self.depth_sum = 0.0;
        self.depth_sum_sq = 0.0;
        self.depth_count = 0;
    }

    /// Reconstruct the game state at a given node by replaying moves from root.
    pub fn reconstruct_game(&self, node_id: NodeId) -> G {
        // Collect moves from node back to root
        let mut moves = Vec::new();
        let mut current = node_id;
        while let Some(parent) = self.arena.get(current).parent {
            moves.push(self.arena.get(current).move_from_parent);
            current = parent;
        }
        moves.reverse();

        // Replay from root game
        let mut game = self.root_game.as_ref().expect("MctsSearch not initialized").clone();
        for mv in &moves {
            game.play_move(mv).unwrap();
        }
        game
    }

    /// Select leaves for NN evaluation.
    /// Reconstructed game states are stashed internally.
    /// Terminal nodes are handled immediately (no stash entry).
    /// Returns the NodeIds of non-terminal leaves in stash order.
    pub fn select_leaves(&mut self, batch_size: usize) -> Vec<NodeId> {
        let mut leaf_ids = Vec::new();

        for _ in 0..batch_size {
            let (leaf, depth) = select_leaf(&self.arena, self.root, &self.params);
            let d = depth as f64;
            self.depth_sum += d;
            self.depth_sum_sq += d * d;
            self.depth_count += 1;

            // Reconstruct game state at the leaf
            let game = self.reconstruct_game(leaf);

            // Lazy fix-up: `expand_with_policy` stored a placeholder
            // `parent.opposite()` for this node's `turn_player`, which is
            // wrong for same-player chains (Yinsh ClaimRow, etc.). Correct it
            // now using the actual reconstructed game state, before any code
            // reads `node.turn_player` on this leaf — terminal backprop,
            // virtual loss, or later `expand_and_backprop` /
            // `correct_virtual_loss`. Backprop walking up only reads
            // ancestors whose `turn_player` was already fixed up at *their*
            // first encounter, so the invariant "every visited node has the
            // right turn_player" holds inductively.
            self.arena.get_mut(leaf).turn_player = game.next_player();

            if game.is_game_over() {
                let (value, draw) = terminal_value(game.outcome(), game.next_player());
                // Mark this leaf as proven (from the leaf player's POV) and
                // propagate the proof upward as far as it can be deduced.
                let leaf_player = game.next_player();
                let leaf_proven = match game.outcome() {
                    Outcome::Draw => Some(Proven::Draw),
                    Outcome::WonBy(p) if p == leaf_player => Some(Proven::Win),
                    Outcome::WonBy(_) => Some(Proven::Loss),
                    Outcome::Ongoing => None, // unreachable given is_game_over
                };
                if let Some(p) = leaf_proven {
                    propagate_proven(&mut self.arena, leaf, p);
                }
                backpropagate(&mut self.arena, leaf, value, draw);
            } else {
                // Apply virtual loss so subsequent selections in this batch diverge.
                apply_virtual_loss(&mut self.arena, leaf);
                self.stashed_leaves.push((leaf, game));
                leaf_ids.push(leaf);
            }
        }

        leaf_ids
    }

    /// Encode a stashed leaf's board state for NN evaluation.
    /// The leaf must have been returned by a prior select_leaves call.
    pub fn encode_leaf(&self, leaf: NodeId) -> (Vec<f32>, Vec<f32>) {
        // Fast path: most recent stash entry (common when called right after select_leaves)
        if let Some((id, game)) = self.stashed_leaves.last() {
            if *id == leaf {
                return Self::encode_game(game);
            }
        }
        self.stashed_leaves.iter()
            .find(|(id, _)| *id == leaf)
            .map(|(_, g)| Self::encode_game(g))
            .expect("encode_leaf: leaf not found in stash — call select_leaves first")
    }

    /// Get the player whose turn it is at a stashed leaf.
    /// The leaf must have been returned by a prior select_leaves call.
    pub fn get_leaf_player(&self, leaf: NodeId) -> Player {
        if let Some((id, game)) = self.stashed_leaves.last() {
            if *id == leaf {
                return game.next_player();
            }
        }
        self.stashed_leaves.iter()
            .find(|(id, _)| *id == leaf)
            .map(|(_, g)| g.next_player())
            .expect("get_leaf_player: leaf not found in stash")
    }

    /// Expand all stashed leaves with NN outputs and backpropagate values.
    /// Consumes the stash. `policies`, `values`, and `draws` must be aligned with
    /// stash order (i.e. the order leaves were returned across all prior
    /// `select_leaves` calls). `values[i]` is the W−L scalar for leaf i;
    /// `draws[i]` is its D probability. Pass `&[]` for `draws` if the eval head
    /// does not predict draw probabilities (treated as all-zeros — no contempt
    /// contribution from these leaves).
    pub fn expand_and_backprop(
        &mut self,
        policies: &[Vec<f32>],
        values: &[f32],
        draws: &[f32],
    ) {
        let stashed = std::mem::take(&mut self.stashed_leaves);
        for (i, (leaf, mut game)) in stashed.into_iter().enumerate() {
            expand_with_policy::<G>(&mut self.arena, leaf, &mut game, &policies[i], self.params.max_children);
            let value = values[i];
            let draw = if draws.is_empty() { 0.0 } else { draws[i] };
            correct_virtual_loss(&mut self.arena, leaf, value, draw);
        }
    }

    /// Apply Dirichlet noise to root children.
    /// Stores raw noise samples in `dirichlet_noise`; `policy_prior` is untouched.
    /// Also records (alpha, epsilon) in `self.params.root_noise` so that UCB's
    /// `prior(dir_epsilon)` computation is always consistent with the stored noise.
    /// alpha: concentration parameter (e.g. 0.3 for Hive)
    /// epsilon: noise weight (e.g. 0.25)
    pub fn apply_root_dirichlet(&mut self, alpha: f32, epsilon: f32) {
        use rand_distr::Distribution;
        use rand_distr::multi::Dirichlet;

        let child_count = self.arena.get(self.root).child_count as usize;
        if child_count == 0 {
            return;
        }

        let alphas: Vec<f32> = vec![alpha; child_count];
        let dirichlet = match Dirichlet::new(&alphas) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut rng = rand::rng();
        let noise: Vec<f32> = dirichlet.sample(&mut rng);

        // Record epsilon so UCB reads the same value during this search.
        self.params.root_noise = RootNoise::Dirichlet { alpha, epsilon };

        let mut child_id = self.arena.get(self.root).first_child;
        let mut i = 0;
        while let Some(cid) = child_id {
            let child = self.arena.get_mut(cid);
            child.dirichlet_noise = noise[i];
            child_id = child.next_sibling;
            i += 1;
        }
    }

    /// Advance the root to the child reached by `mv`, preserving the subtree.
    ///
    /// Visit statistics (`visit_count`, `value_sum`) are kept intact so the next
    /// search starts warm — `simulations` acts as an incremental budget on top of
    /// whatever the subtree already accumulated.  Only `dirichlet_noise` on the
    /// new root itself is cleared (cosmetic: the root's prior is never read by UCB).
    /// Call `apply_root_dirichlet` afterwards to apply fresh noise to the new
    /// root's children before searching.
    ///
    /// Orphaned siblings and their subtrees are freed into the arena free list so
    /// their node slots are reused by subsequent searches. This keeps arena memory
    /// proportional to the current live tree rather than the full game history.
    ///
    /// Returns `true` on success, `false` if `mv` was not among the root's children
    /// (caller should fall back to a fresh `init`).
    pub fn reroot(&mut self, mv: G::Move) -> bool
    where
        G::Move: PartialEq,
    {
        self.stashed_leaves.clear();

        let old_root_id = self.root;
        let mut new_root_id = None;

        // Walk root children: free every branch except the chosen one.
        let mut current = self.arena.get(old_root_id).first_child;
        while let Some(cid) = current {
            // Capture next before potentially freeing cid's subtree.
            let next = self.arena.get(cid).next_sibling;
            if self.arena.get(cid).move_from_parent == mv {
                new_root_id = Some(cid);
            } else {
                self.arena.free_subtree(cid);
            }
            current = next;
        }

        if let Some(new_root) = new_root_id {
            let node = self.arena.get_mut(new_root);
            node.parent = None;
            node.dirichlet_noise = 0.0; // root's prior is never read by UCB
            self.root = new_root;
            self.root_game.as_mut().unwrap().play_move(&mv).ok();
            // Free the old root node (no longer reachable).
            self.arena.free_node(old_root_id);
            true
        } else {
            false
        }
    }

    /// Get the best move by visit count.
    pub fn best_move(&self) -> Option<G::Move> {
        let root = self.arena.get(self.root);
        // Children of root are evaluated from root's turn-player's perspective,
        // so the effective contempt depends on whether root.turn_player is the
        // designated contempt side.
        let contempt = effective_contempt(&self.params, root.turn_player);

        // Tiered selection over root children. A child's `proven` flag is from
        // the *child's* player's perspective; translate to root's POV per edge.
        // Tier 0 (lowest int = best): proven Win for root — forced mate, always pick.
        // Tier 1: unproven OR proven Draw for root — normal MCTS visit-count selection.
        // Tier 2: proven Loss for root — suicide moves; pick only if nothing better exists.
        let mut best_tier = u8::MAX;
        let mut best_visits = 0u32;
        let mut best_value = f32::NEG_INFINITY;
        let mut best_move = None;

        let mut child_id = root.first_child;
        while let Some(cid) = child_id {
            let child = self.arena.get(cid);
            let same_player = root.turn_player == child.turn_player;
            let tier = match child.proven {
                Some(p) => match if same_player { p } else { p.flip() } {
                    Proven::Win => 0u8,
                    Proven::Draw => 1u8,
                    Proven::Loss => 2u8,
                },
                None => 1u8,
            };
            let v = child.value(contempt);
            let visits = child.visit_count;
            let is_better = tier < best_tier
                || (tier == best_tier && visits > best_visits)
                || (tier == best_tier && visits == best_visits && best_move.is_some() && v > best_value);
            if best_move.is_none() || is_better {
                best_tier = tier;
                best_visits = visits;
                best_value = v;
                best_move = Some(child.move_from_parent);
            }
            child_id = child.next_sibling;
        }

        best_move
    }

    /// Diagnostic snapshot of one root child.
    /// `value` is from the root player's perspective (positive = good for root),
    /// matching how UCB reads it at the root level. `policy_prior` is the raw
    /// NN softmax — no Dirichlet noise mixed in. `outcome` is the outcome of the
    /// state reached by playing this move from the root (Ongoing if not terminal).
    pub fn root_child_stats(&self) -> Vec<RootChildStat<G::Move>> {
        let root = self.arena.get(self.root);
        let mut out = Vec::new();
        let mut child_id = root.first_child;
        while let Some(cid) = child_id {
            let child = self.arena.get(cid);
            let game = self.reconstruct_game(cid);
            out.push(RootChildStat {
                move_from_parent: child.move_from_parent,
                visit_count: child.visit_count,
                value: child.value(0.0),
                policy_prior: child.policy_prior,
                outcome: game.outcome(),
            });
            child_id = child.next_sibling;
        }
        out
    }

    /// Get visit count distribution for training policy.
    ///
    /// Applies the same tier preference as `best_move`: if any root child is a
    /// proven Win for root, the target distribution is restricted to those
    /// (visit-weighted); if all wins are absent, proven-Loss-for-root children
    /// are excluded so the policy head doesn't learn to put mass on suicide
    /// moves. Only when *every* child is a proven loss do we fall back to
    /// distributing over the loss tier.
    pub fn get_visit_distribution(&self) -> Vec<(G::Move, f32)> {
        let root = self.arena.get(self.root);
        let mut tier_win: Vec<(G::Move, f32)> = Vec::new();
        let mut tier_neutral: Vec<(G::Move, f32)> = Vec::new();
        let mut tier_loss: Vec<(G::Move, f32)> = Vec::new();

        let mut child_id = root.first_child;
        while let Some(cid) = child_id {
            let child = self.arena.get(cid);
            let same_player = root.turn_player == child.turn_player;
            let entry = (child.move_from_parent, child.visit_count as f32);
            match child.proven {
                Some(p) => match if same_player { p } else { p.flip() } {
                    Proven::Win => tier_win.push(entry),
                    Proven::Draw => tier_neutral.push(entry),
                    Proven::Loss => tier_loss.push(entry),
                },
                None => tier_neutral.push(entry),
            }
            child_id = child.next_sibling;
        }

        let mut result = if !tier_win.is_empty() {
            tier_win
        } else if !tier_neutral.is_empty() {
            tier_neutral
        } else {
            tier_loss
        };

        let total: f32 = result.iter().map(|(_, v)| *v).sum();
        if total > 0.0 {
            for item in &mut result {
                item.1 /= total;
            }
        }
        result
    }

    /// Get visit distribution with policy target pruning (KataGo).
    /// Only applies if forced exploration is enabled. Subtracts forced playouts from
    /// children that wouldn't have been chosen by normal PUCT, and prunes children
    /// reduced to <=1 visit. If forced exploration is disabled, returns unpruned distribution.
    ///
    /// Also applies the same proven-outcome tier filter as `get_visit_distribution`:
    /// proven-Win-for-root children replace the whole distribution if any exist;
    /// proven-Loss-for-root children are dropped unless they're the only option.
    pub fn get_pruned_visit_distribution(&self) -> Vec<(G::Move, f32)> {
        // Only prune if forced exploration is enabled
        let pruning_k = match &self.params.forced_exploration {
            ForcedExploration::None => return self.get_visit_distribution(),
            ForcedExploration::Soft { pruning_k, .. } => *pruning_k,
        };

        let root = self.arena.get(self.root);
        let parent_visits = root.visit_count;
        let n_total = parent_visits as f32;

        // Collect children: (move, raw_visits, prior, value, tier)
        let eps = dir_epsilon(&self.params);
        // Root's children are evaluated from root.turn_player's perspective,
        // so use that side's effective contempt.
        let contempt = effective_contempt(&self.params, root.turn_player);
        struct ChildInfo<M: Copy> {
            mv: M,
            visits: u32,
            prior: f32,
            value: f32,
            tier: u8, // 0 = proven win for root, 1 = neutral, 2 = proven loss for root
        }
        let mut children: Vec<ChildInfo<G::Move>> = Vec::new();
        let mut child_id = root.first_child;
        while let Some(cid) = child_id {
            let child = self.arena.get(cid);
            let same_player = root.turn_player == child.turn_player;
            let tier = match child.proven {
                Some(p) => match if same_player { p } else { p.flip() } {
                    Proven::Win => 0u8,
                    Proven::Draw => 1u8,
                    Proven::Loss => 2u8,
                },
                None => 1u8,
            };
            children.push(ChildInfo {
                mv: child.move_from_parent,
                visits: child.visit_count,
                prior: child.prior(eps),
                value: child.value(contempt),
                tier,
            });
            child_id = child.next_sibling;
        }

        if children.is_empty() {
            return Vec::new();
        }

        // Pick the best available tier and drop everything else before pruning.
        let best_tier = children.iter().map(|c| c.tier).min().unwrap();
        children.retain(|c| c.tier == best_tier);
        if children.is_empty() {
            return Vec::new();
        }

        // Find the best child (most visits)
        let best_idx = children.iter().enumerate()
            .max_by_key(|(_, c)| c.visits)
            .map(|(i, _)| i)
            .unwrap();
        let best_visits = children[best_idx].visits;
        let best_value = children[best_idx].value;

        // Compute PUCT score of best child (with its current visits)
        let c_puct = calculate_cpuct(&self.params, parent_visits);
        let best_puct = calculate_ucb_score_parts(
            best_value,
            children[best_idx].prior,
            best_visits,
            c_puct,
            parent_visits,
        );

        // For each other child, subtract forced playouts as long as
        // it doesn't cause their PUCT to exceed the best child's PUCT
        let mut adjusted_visits: Vec<u32> = children.iter().map(|c| c.visits).collect();

        for (i, child) in children.iter().enumerate() {
            if i == best_idx || child.visits == 0 {
                continue;
            }
            let n_forced = (pruning_k * (child.prior * n_total).sqrt()) as u32;
            if n_forced == 0 {
                continue;
            }
            // Try subtracting up to n_forced visits
            let max_subtract = n_forced.min(child.visits.saturating_sub(1));
            for subtract in (1..=max_subtract).rev() {
                let new_visits = child.visits - subtract;
                // Check: would PUCT(child) with new_visits still be < best_puct?
                let child_puct = calculate_ucb_score_parts(
                    child.value,
                    child.prior,
                    new_visits,
                    c_puct,
                    parent_visits,
                );
                if child_puct < best_puct {
                    adjusted_visits[i] = new_visits;
                    break;
                }
            }
        }

        // Prune children with <=1 adjusted visit (except the best)
        let mut result = Vec::new();
        let mut total_visits = 0u32;
        for (i, child) in children.iter().enumerate() {
            let v = adjusted_visits[i];
            if i != best_idx && v <= 1 {
                continue;
            }
            result.push((child.mv, v as f32));
            total_visits += v;
        }

        // Normalize
        if total_visits > 0 {
            for item in &mut result {
                item.1 /= total_visits as f32;
            }
        }

        result
    }

    /// Number of legal moves at the root (0 means only a pass is available).
    pub fn root_child_count(&self) -> u16 {
        self.arena.get(self.root).child_count
    }

    /// Total visit count at the root.
    pub fn root_visit_count(&self) -> u32 {
        self.arena.get(self.root).visit_count
    }

    /// Fraction of root child visits held by the single most-visited child.
    /// Returns 0.0 if no children have been visited yet.
    pub fn root_top1_visit_fraction(&self) -> f32 {
        let root = self.arena.get(self.root);
        let mut max_visits = 0u32;
        let mut total_visits = 0u32;
        let mut child_id = root.first_child;
        while let Some(cid) = child_id {
            let child = self.arena.get(cid);
            if child.visit_count > max_visits {
                max_visits = child.visit_count;
            }
            total_visits += child.visit_count;
            child_id = child.next_sibling;
        }
        if total_visits == 0 { return 0.0; }
        max_visits as f32 / total_visits as f32
    }

    /// Return accumulated depth stats (sum, sum_sq, count) and reset the accumulators.
    /// depth is the number of edges from root to the simulation leaf.
    pub fn take_depth_stats(&mut self) -> (f64, f64, u64) {
        let result = (self.depth_sum, self.depth_sum_sq, self.depth_count);
        self.depth_sum = 0.0;
        self.depth_sum_sq = 0.0;
        self.depth_count = 0;
        result
    }

    /// Mean value estimate at the root, from the root player's own perspective.
    /// `value_sum` accumulates in the "opposite" (phantom-parent) frame so the
    /// W−L component is negated to recover root's own perspective; `draw_sum`
    /// is symmetric (same magnitude for both players) and so the contempt
    /// term is subtracted, not added. See docs/mcts_value_convention.md.
    ///
    /// With asymmetric contempt the contempt scalar applied here is the one
    /// for the root player — i.e. the root player's own draw aversion shapes
    /// the value they expect to realise.
    pub fn root_value(&self) -> f32 {
        let root = self.arena.get(self.root);
        let n = root.visit_count;
        if n == 0 {
            return 0.0;
        }
        let n = n as f32;
        let raw_wl = -root.value_sum / n;
        let avg_draw = root.draw_sum / n;
        let contempt = effective_contempt(&self.params, root.turn_player);
        raw_wl - contempt * avg_draw
    }

    /// Mean W−L estimate at the root in root's own perspective, with NO
    /// contempt subtracted. This is the search-improved value target used for
    /// q-target mixing in training: contempt is a play-time aversion that
    /// should not bleed into the value head's regression target.
    pub fn root_value_raw(&self) -> f32 {
        let root = self.arena.get(self.root);
        let n = root.visit_count;
        if n == 0 {
            return 0.0;
        }
        -root.value_sum / n as f32
    }

    /// Encode a game state for NN evaluation.
    pub fn encode_game(game: &G) -> (Vec<f32>, Vec<f32>) {
        let mut board = vec![0.0f32; game.board_tensor_size()];
        let mut reserve = vec![0.0f32; G::RESERVE_SIZE];
        game.encode_board(&mut board, &mut reserve);
        (board, reserve)
    }
}

#[cfg(test)]
mod tests {
    //! Regression tests for the same-player-chain backprop bug
    //! (see docs/mcts_value_convention.md).
    //!
    //! Yinsh's ClaimRow phase and pre-refactor Zertz mid-captures keep the
    //! current player on consecutive nodes. Backprop's player-boundary sign
    //! flip only works if `expand_with_policy` records each child's *actual*
    //! post-move `next_player`, not a hardcoded `parent.opposite()`.

    use super::*;
    use crate::game::{Game, NNGame, Outcome, Player, PolicyIndex, Undoable};
    use crate::symmetry::UnitSymmetry;

    /// Tiny synthetic game with a single legal move that *keeps* the player
    /// (a same-player transition) plus a sibling move that flips it normally.
    /// At depth 0 only the same-player branch exists; both branches lead to a
    /// terminal where Player1 wins. Used to verify expand_with_policy and
    /// backprop sign together.
    #[derive(Clone)]
    struct ChainGame {
        depth: u8,             // remaining moves before terminal
        player: Player,
        last_mover: Player,
        history: Vec<(u8, Player, Player)>,
    }

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    enum ChainMove {
        Keep, // same-player chain (Yinsh ClaimRow analogue)
        Flip, // normal alternation
    }

    impl ChainGame {
        fn new(depth: u8) -> Self {
            Self { depth, player: Player::Player1, last_mover: Player::Player1, history: Vec::new() }
        }
    }

    impl Game for ChainGame {
        type Move = ChainMove;
        type Symmetry = UnitSymmetry;

        fn next_player(&self) -> Player { self.player }

        fn outcome(&self) -> Outcome {
            if self.depth == 0 { Outcome::WonBy(self.last_mover) } else { Outcome::Ongoing }
        }

        fn valid_moves(&mut self) -> Vec<ChainMove> {
            if self.depth == 0 { return vec![]; }
            vec![ChainMove::Keep, ChainMove::Flip]
        }

        fn play_move(&mut self, mv: &ChainMove) -> Result<(), String> {
            if self.depth == 0 { return Err("terminal".into()); }
            self.history.push((self.depth, self.player, self.last_mover));
            self.last_mover = self.player;
            self.depth -= 1;
            if matches!(mv, ChainMove::Flip) {
                self.player = self.player.opposite();
            }
            Ok(())
        }

        fn pass_move() -> ChainMove { ChainMove::Flip }
        fn is_pass(_mv: &ChainMove) -> bool { false }
    }

    impl Undoable for ChainGame {
        fn undo(&mut self) {
            let (d, p, lm) = self.history.pop().expect("undo with empty history");
            self.depth = d;
            self.player = p;
            self.last_mover = lm;
        }
    }

    impl NNGame for ChainGame {
        const BOARD_CHANNELS: usize = 1;
        const RESERVE_SIZE: usize = 0;
        const NUM_POLICY_CHANNELS: usize = 1;
        fn grid_size(&self) -> usize { 2 }
        fn encode_board(&self, _board: &mut [f32], _reserve: &mut [f32]) {}
        fn get_legal_move_mask(&mut self) -> (Vec<f32>, Vec<(PolicyIndex, ChainMove)>) {
            let mvs = self.valid_moves();
            let mask = vec![0.0f32; self.policy_size()];
            // Distinct policy indices for each child so priors don't collide.
            let indexed: Vec<(PolicyIndex, ChainMove)> = mvs.into_iter().enumerate()
                .map(|(i, mv)| (PolicyIndex::Single(i), mv))
                .collect();
            (mask, indexed)
        }
    }

    /// Every visited node's stored `turn_player` must match
    /// `game.next_player()` at that node's reconstructed state. With lazy
    /// fix-up, this is only guaranteed for nodes that have been encountered
    /// as a leaf at least once (visit_count >= 1) — `expand_with_policy`
    /// stores a placeholder that's wrong on same-player chains, and
    /// `select_leaves` corrects it on first encounter.
    #[test]
    fn turn_player_correct_after_visit() {
        let game = ChainGame::new(2);
        let mut search = MctsSearch::<ChainGame>::new(64);
        search.params = SearchParams::new(
            CpuctStrategy::Constant { c_puct: 1.5 },
            ForcedExploration::None,
            RootNoise::None,
        );

        let policy = vec![0.0f32; game.policy_size()];
        search.init(&game, &policy);

        // Run enough sims that every reachable node (root + 2 + 4 = 7) is visited.
        for _ in 0..64 {
            let leaves = search.select_leaves(1);
            if leaves.is_empty() { continue; }
            let mut policies = Vec::new();
            let mut values = Vec::new();
            for _ in &leaves {
                policies.push(vec![0.0f32; game.policy_size()]);
                values.push(0.0f32);
            }
            search.expand_and_backprop(&policies, &values, &[]);
        }

        let mut stack = vec![search.root];
        let mut visited_keep_child = false;
        while let Some(node_id) = stack.pop() {
            let node = search.arena.get(node_id);
            let mut child_id = node.first_child;
            while let Some(cid) = child_id {
                stack.push(cid);
                child_id = search.arena.get(cid).next_sibling;
            }
            if node.visit_count == 0 { continue; }
            let actual = search.reconstruct_game(node_id).next_player();
            assert_eq!(node.turn_player, actual,
                "node {} has stored turn_player {:?} but reconstructed game says {:?}",
                node_id, node.turn_player, actual);
            // Also confirm the Keep branch (same-player chain) was exercised:
            // it's the case the bug specifically hit.
            if node.parent.is_some() && node.move_from_parent == ChainMove::Keep {
                visited_keep_child = true;
            }
        }
        assert!(visited_keep_child,
            "Keep (same-player) child should have been visited; otherwise the test does not exercise the fix");
    }

    /// End-to-end: Player1 can win in one move via the Keep branch (depth=1).
    /// The terminal at the leaf returns +1 for Player1; backprop must propagate
    /// that as a positive value to the root, since Keep does not cross a
    /// player boundary. Pre-fix code stored Keep's child as Player2 and so
    /// flipped the sign — root would have seen the win as a loss.
    #[test]
    fn same_player_chain_backprop_preserves_sign() {
        let game = ChainGame::new(1);
        let mut search = MctsSearch::<ChainGame>::new(64);
        search.params = SearchParams::new(
            CpuctStrategy::Constant { c_puct: 1.5 },
            ForcedExploration::None,
            RootNoise::None,
        );

        let policy = vec![0.0f32; game.policy_size()];
        search.init(&game, &policy);

        // 16 sims is plenty — the tree only has 2 children (both terminal at depth 0).
        for _ in 0..16 {
            let leaves = search.select_leaves(1);
            if leaves.is_empty() { continue; }
            let mut policies = Vec::new();
            let mut values = Vec::new();
            for _ in &leaves {
                policies.push(vec![0.0f32; game.policy_size()]);
                values.push(0.0f32);
            }
            search.expand_and_backprop(&policies, &values, &[]);
        }

        // Root is Player1 to move; both children lead to terminal WonBy(Player1).
        // root_value() must be positive (≈ +1) — Player1 wins from root.
        let rv = search.root_value();
        assert!(rv > 0.5,
            "root_value should be ~+1 for a winning position, got {}", rv);
    }
}
