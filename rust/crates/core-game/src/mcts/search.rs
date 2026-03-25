/// MCTS search implementation with arena allocation and batch NN evaluation.
/// Game states are reconstructed by replaying moves from the root, not stored per node.

use super::arena::{NodeArena, NodeId};
use super::node::MctsNode;
use crate::game::{GameEngine, Outcome, Player};

const DEFAULT_C_PUCT: f32 = 1.5;

/// UCB score for child selection.
fn ucb_score<M: Copy>(node: &MctsNode<M>, parent_visits: u32, c_puct: f32) -> f32 {
    let exploration = c_puct * node.prior * (parent_visits as f32).sqrt()
        / (1.0 + node.visit_count as f32);
    node.value() + exploration
}

/// Select the best child by UCB score.
fn best_child<M: Copy>(arena: &NodeArena<M>, node_id: NodeId, c_puct: f32) -> NodeId {
    let node = arena.get(node_id);
    let parent_visits = node.visit_count;
    let mut best_id = node.first_child.expect("no children");
    let mut best_score = ucb_score(arena.get(best_id), parent_visits, c_puct);

    let mut current = arena.get(best_id).next_sibling;
    while let Some(child_id) = current {
        let child = arena.get(child_id);
        let score = ucb_score(child, parent_visits, c_puct);
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
fn select_leaf<M: Copy>(arena: &NodeArena<M>, root: NodeId, c_puct: f32, forced_playouts: bool) -> NodeId {
    let mut node_id = root;
    let mut is_root = true;
    loop {
        let node = arena.get(node_id);
        if !node.is_expanded || node.first_child.is_none() {
            return node_id;
        }
        if is_root && forced_playouts {
            node_id = best_child_with_forced(arena, node_id, c_puct);
        } else {
            node_id = best_child(arena, node_id, c_puct);
        }
        is_root = false;
    }
}

/// Select best child at root, forcing minimum playouts on children that have
/// been visited at least once. A child with visit_count < n_forced gets
/// infinite urgency so it is always selected first.
/// n_forced(c) = k * sqrt(P_noised(c) * N_total), k=2
fn best_child_with_forced<M: Copy>(arena: &NodeArena<M>, node_id: NodeId, c_puct: f32) -> NodeId {
    let node = arena.get(node_id);
    let parent_visits = node.visit_count;
    let n_total = parent_visits as f32;

    let mut best_id = node.first_child.expect("no children");
    let mut best_score = child_score_with_forced(arena.get(best_id), parent_visits, c_puct, n_total);

    let mut current = arena.get(best_id).next_sibling;
    while let Some(child_id) = current {
        let child = arena.get(child_id);
        let score = child_score_with_forced(child, parent_visits, c_puct, n_total);
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
fn child_score_with_forced<M: Copy>(child: &MctsNode<M>, parent_visits: u32, c_puct: f32, n_total: f32) -> f32 {
    if child.visit_count > 0 {
        let k = 2.0f32;
        let n_forced = (k * (child.prior * n_total).sqrt()) as u32;
        if child.visit_count < n_forced {
            return f32::INFINITY;
        }
    }
    ucb_score(child, parent_visits, c_puct)
}

/// Backpropagate a value up the tree.
fn backpropagate<M: Copy>(arena: &mut NodeArena<M>, mut node_id: NodeId, mut value: f32) {
    loop {
        let node = arena.get_mut(node_id);
        node.visit_count += 1;
        node.value_sum += value;
        value = -value;
        match node.parent {
            Some(parent) => node_id = parent,
            None => break,
        }
    }
}

/// Apply virtual loss: increment visit_count and subtract 1 from value_sum up the tree.
/// This deters subsequent selections from taking the same path within a batch.
fn apply_virtual_loss<M: Copy>(arena: &mut NodeArena<M>, mut node_id: NodeId) {
    let mut value = -1.0f32;
    loop {
        let node = arena.get_mut(node_id);
        node.visit_count += 1;
        node.value_sum += value;
        value = -value;
        match node.parent {
            Some(parent) => node_id = parent,
            None => break,
        }
    }
}

/// Correct virtual loss by replacing the -1 placeholder with the real value.
/// Does NOT increment visit_count (already done by apply_virtual_loss).
fn correct_virtual_loss<M: Copy>(arena: &mut NodeArena<M>, mut node_id: NodeId, mut real_value: f32) {
    let mut virtual_value = -1.0f32;
    loop {
        let node = arena.get_mut(node_id);
        // Subtract the virtual placeholder and add the real value.
        node.value_sum += real_value - virtual_value;
        real_value = -real_value;
        virtual_value = -virtual_value;
        match node.parent {
            Some(parent) => node_id = parent,
            None => break,
        }
    }
}

/// Terminal game value from a perspective.
fn terminal_value(outcome: Outcome, perspective: Player) -> f32 {
    match outcome {
        Outcome::Draw | Outcome::Ongoing => 0.0,
        Outcome::WonBy(winner) => {
            if winner == perspective { 1.0 } else { -1.0 }
        }
    }
}

/// Expand a node with a policy vector, adding children to the arena.
/// `game` is the reconstructed game state at this node (not stored in the node).
fn expand_with_policy<G: GameEngine>(
    arena: &mut NodeArena<G::Move>,
    node_id: NodeId,
    game: &mut G,
    policy: &[f32],
) {
    arena.get_mut(node_id).is_expanded = true;
    let child_turn = game.current_player().opposite();

    let (_mask, indexed_moves) = game.get_legal_move_mask();
    if indexed_moves.is_empty() {
        // Pass — child turn is opposite of current
        let child_id = arena.alloc(Some(node_id), G::pass_move(), 1.0, child_turn);
        arena.get_mut(node_id).first_child = Some(child_id);
        arena.get_mut(node_id).child_count = 1;
        return;
    }

    let mut total_prior = 0.0f32;
    let mut first_child_id: Option<NodeId> = None;
    let mut prev_child_id: Option<NodeId> = None;
    let mut child_count = 0u16;

    for &(idx, mv) in &indexed_moves {
        let prior = policy[idx];
        total_prior += prior;

        let child_id = arena.alloc(Some(node_id), mv, prior, child_turn);

        if first_child_id.is_none() {
            first_child_id = Some(child_id);
        }
        if let Some(prev) = prev_child_id {
            arena.get_mut(prev).next_sibling = Some(child_id);
        }
        prev_child_id = Some(child_id);
        child_count += 1;
    }

    // Normalize priors
    if total_prior > 0.0 {
        let mut child_id = first_child_id;
        while let Some(cid) = child_id {
            let child = arena.get_mut(cid);
            child.prior /= total_prior;
            child_id = child.next_sibling;
        }
    }

    arena.get_mut(node_id).first_child = first_child_id;
    arena.get_mut(node_id).child_count = child_count;
}

/// Single-game MCTS search engine, generic over any GameEngine.
/// Game states are reconstructed by replaying moves from root_game.
pub struct MctsSearch<G: GameEngine> {
    pub arena: NodeArena<G::Move>,
    pub c_puct: f32,
    pub root: NodeId,
    root_game: Option<G>,
    /// Temporary storage for leaf data between select and expand calls.
    pub stashed_leaves: Vec<(NodeId, G)>,
    /// Whether to use forced playouts at the root (KataGo-style).
    pub use_forced_playouts: bool,
}

impl<G: GameEngine> MctsSearch<G> {
    pub fn new(capacity: usize) -> Self {
        MctsSearch {
            arena: NodeArena::new(capacity, G::pass_move()),
            c_puct: DEFAULT_C_PUCT,
            root: 0, // will be set in init
            root_game: None,
            stashed_leaves: Vec::new(),
            use_forced_playouts: false,
        }
    }

    /// Initialize search for a game position.
    pub fn init(&mut self, game: &G, policy: &[f32]) {
        self.arena.reset();
        self.root_game = Some(game.clone());
        let root = self.arena.alloc(None, G::pass_move(), 0.0, game.current_player());
        self.root = root;
        let mut game_copy = game.clone();
        expand_with_policy::<G>(&mut self.arena, root, &mut game_copy, policy);
    }

    /// Reconstruct the game state at a given node by replaying moves from root.
    fn reconstruct_game(&self, node_id: NodeId) -> G {
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

    /// Run one batch of simulations. Returns list of (leaf_id, game) pairs
    /// that need NN evaluation. Terminal nodes are handled immediately.
    pub fn select_leaves(&mut self, batch_size: usize) -> Vec<(NodeId, G)> {
        let root_turn = self.arena.get(self.root).turn_player;
        let mut leaves = Vec::new();

        for _ in 0..batch_size {
            let leaf = select_leaf(&self.arena, self.root, self.c_puct, self.use_forced_playouts);

            // Reconstruct game state at the leaf
            let game = self.reconstruct_game(leaf);

            if game.is_game_over() {
                let value = terminal_value(game.outcome(), root_turn);
                backpropagate(&mut self.arena, leaf, value);
            } else {
                // Apply virtual loss so subsequent selections in this batch diverge.
                apply_virtual_loss(&mut self.arena, leaf);
                leaves.push((leaf, game));
            }
        }

        leaves
    }

    /// Expand leaf nodes with policies and backpropagate values.
    pub fn expand_and_backprop(
        &mut self,
        leaves: &mut Vec<(NodeId, G)>,
        policies: &[Vec<f32>],
        values: &[f32],
    ) {
        let root_turn = self.arena.get(self.root).turn_player;
        for (i, (leaf, game)) in leaves.iter_mut().enumerate() {
            expand_with_policy::<G>(&mut self.arena, *leaf, game, &policies[i]);
            let mut value = values[i];
            if self.arena.get(*leaf).turn_player != root_turn {
                value = -value;
            }
            // Replace virtual loss placeholder with real value (visit_count already correct).
            correct_virtual_loss(&mut self.arena, *leaf, value);
        }
    }

    /// Apply Dirichlet noise to root children priors.
    /// alpha: concentration parameter (e.g. 0.3 for Hive)
    /// epsilon: noise weight (e.g. 0.25)
    pub fn apply_root_dirichlet(&mut self, alpha: f32, epsilon: f32) {
        use rand::SeedableRng;
        use rand_distr::{Dirichlet, Distribution};

        let root = self.arena.get(self.root);
        let child_count = root.child_count as usize;
        if child_count == 0 {
            return;
        }

        let alphas = vec![alpha; child_count];
        let dirichlet = match Dirichlet::new(&alphas) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut rng = rand::rngs::StdRng::from_entropy();
        let noise: Vec<f32> = dirichlet.sample(&mut rng);

        let mut child_id = self.arena.get(self.root).first_child;
        let mut i = 0;
        while let Some(cid) = child_id {
            let child = self.arena.get_mut(cid);
            child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[i];
            child_id = child.next_sibling;
            i += 1;
        }
    }

    /// Get the best move by visit count.
    pub fn best_move(&self) -> Option<G::Move> {
        let root = self.arena.get(self.root);
        let mut best_visits = 0u32;
        let mut best_move = None;

        let mut child_id = root.first_child;
        while let Some(cid) = child_id {
            let child = self.arena.get(cid);
            if child.visit_count > best_visits {
                best_visits = child.visit_count;
                best_move = Some(child.move_from_parent);
            }
            child_id = child.next_sibling;
        }

        best_move
    }

    /// Get visit count distribution for training policy.
    pub fn get_visit_distribution(&self) -> Vec<(G::Move, f32)> {
        let root = self.arena.get(self.root);
        let mut result = Vec::new();
        let mut total_visits = 0u32;

        let mut child_id = root.first_child;
        while let Some(cid) = child_id {
            let child = self.arena.get(cid);
            result.push((child.move_from_parent, child.visit_count as f32));
            total_visits += child.visit_count;
            child_id = child.next_sibling;
        }

        if total_visits > 0 {
            for item in &mut result {
                item.1 /= total_visits as f32;
            }
        }

        result
    }

    /// Get visit distribution with policy target pruning (KataGo).
    /// Subtracts forced playouts from children that wouldn't have been chosen
    /// by normal PUCT, and prunes children reduced to <=1 visit.
    pub fn get_pruned_visit_distribution(&self) -> Vec<(G::Move, f32)> {
        let root = self.arena.get(self.root);
        let parent_visits = root.visit_count;
        let n_total = parent_visits as f32;

        // Collect children: (move, raw_visits, prior, value)
        struct ChildInfo<M: Copy> {
            mv: M,
            visits: u32,
            prior: f32,
            value: f32,
        }
        let mut children: Vec<ChildInfo<G::Move>> = Vec::new();
        let mut child_id = root.first_child;
        while let Some(cid) = child_id {
            let child = self.arena.get(cid);
            children.push(ChildInfo {
                mv: child.move_from_parent,
                visits: child.visit_count,
                prior: child.prior,
                value: child.value(),
            });
            child_id = child.next_sibling;
        }

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
        let best_puct = best_value + self.c_puct * children[best_idx].prior
            * n_total.sqrt() / (1.0 + best_visits as f32);

        // For each other child, subtract forced playouts as long as
        // it doesn't cause their PUCT to exceed the best child's PUCT
        let k = 2.0f32;
        let mut adjusted_visits: Vec<u32> = children.iter().map(|c| c.visits).collect();

        for (i, child) in children.iter().enumerate() {
            if i == best_idx || child.visits == 0 {
                continue;
            }
            let n_forced = (k * (child.prior * n_total).sqrt()) as u32;
            if n_forced == 0 {
                continue;
            }
            // Try subtracting up to n_forced visits
            let max_subtract = n_forced.min(child.visits.saturating_sub(1));
            for subtract in (1..=max_subtract).rev() {
                let new_visits = child.visits - subtract;
                // Check: would PUCT(child) with new_visits still be < best_puct?
                let child_puct = child.value + self.c_puct * child.prior
                    * n_total.sqrt() / (1.0 + new_visits as f32);
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

    /// Encode a game state for NN evaluation.
    pub fn encode_game(game: &G) -> (Vec<f32>, Vec<f32>) {
        let mut board = vec![0.0f32; G::BOARD_CHANNELS * G::GRID_SIZE * G::GRID_SIZE];
        let mut reserve = vec![0.0f32; G::RESERVE_SIZE];
        game.encode_board(&mut board, &mut reserve);
        (board, reserve)
    }
}

#[cfg(test)]
mod tests {
    // Tests require a concrete GameEngine implementation, so they live
    // in the hive crate (or other game crates) rather than here.
}
