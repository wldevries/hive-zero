/// MCTS search with lazy child creation and batch NN evaluation.
///
/// Key optimization: nodes are only created when first visited, not when
/// the parent is expanded. Expansion just stores (move, prior) edges.
/// This avoids cloning 1000+ boards for Zertz's large move space.

use core_game::game::{Game, Outcome, Player};

use super::arena::{NodeArena, NodeId};
use super::node::{Edge, MctsNode};
use crate::board_encoding::{encode_board, GRID_SIZE, NUM_CHANNELS, RESERVE_SIZE};
use crate::move_encoding::get_legal_move_mask;
#[cfg(test)]
use crate::move_encoding::POLICY_SIZE;
use crate::zertz::ZertzBoard;

const DEFAULT_C_PUCT: f32 = 1.5;

// ---------------------------------------------------------------------------
// Edge selection (UCB)
// ---------------------------------------------------------------------------

/// Pick the best edge index by UCB score.
fn pick_best_edge(arena: &NodeArena, node_id: NodeId, c_puct: f32, forced: bool) -> usize {
    let node = arena.get(node_id);
    let parent_visits = node.visit_count;
    let sqrt_parent = (parent_visits as f32).sqrt();
    let n_total = parent_visits as f32;

    let mut best_idx = 0;
    let mut best_score = f32::NEG_INFINITY;

    for (i, edge) in node.edges.iter().enumerate() {
        let (child_visits, child_value) = match edge.child_id {
            Some(cid) => {
                let child = arena.get(cid);
                (child.visit_count, child.value())
            }
            None => (0, 0.0),
        };

        let exploration = c_puct * edge.prior * sqrt_parent / (1.0 + child_visits as f32);

        let score = if forced && child_visits > 0 {
            let k = 2.0f32;
            let n_forced = (k * (edge.prior * n_total).sqrt()) as u32;
            if child_visits < n_forced {
                f32::INFINITY
            } else {
                child_value + exploration
            }
        } else {
            child_value + exploration
        };

        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    best_idx
}

// ---------------------------------------------------------------------------
// Tree traversal
// ---------------------------------------------------------------------------

/// Select a leaf node, creating child nodes lazily when first visited.
/// Returns the leaf NodeId (either newly created or existing unexpanded node).
fn select_leaf(
    arena: &mut NodeArena,
    root: NodeId,
    c_puct: f32,
    forced_playouts: bool,
) -> NodeId {
    let mut node_id = root;
    let mut is_root = true;
    loop {
        if !arena.get(node_id).is_expanded() {
            return node_id; // unexpanded leaf
        }

        let best_idx = pick_best_edge(arena, node_id, c_puct, is_root && forced_playouts);

        // Check if child node already exists
        let child_id = arena.get(node_id).edges[best_idx].child_id;

        if let Some(cid) = child_id {
            node_id = cid;
            is_root = false;
        } else {
            // Lazy child creation: clone parent board, play the move
            let mv = arena.get(node_id).edges[best_idx].mv;
            let child_board = {
                let mut b = arena.get(node_id).board.clone_light();
                b.play_mcts(mv).expect("legal move failed in MCTS selection");
                b
            };
            let new_node = MctsNode {
                board: child_board,
                parent: Some(node_id),
                parent_edge_idx: best_idx as u16,
                visit_count: 0,
                value_sum: 0.0,
                move_from_parent: Some(mv),
                edges: Vec::new(),
            };
            let cid = arena.alloc(new_node);
            arena.get_mut(node_id).edges[best_idx].child_id = Some(cid);
            return cid;
        }
    }
}

// ---------------------------------------------------------------------------
// Backpropagation
// ---------------------------------------------------------------------------

/// Backpropagate a value up the tree (alternating negation).
fn backpropagate(arena: &mut NodeArena, mut node_id: NodeId, mut value: f32) {
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

/// Apply virtual loss up the tree to deter batch re-selection of the same path.
fn apply_virtual_loss(arena: &mut NodeArena, mut node_id: NodeId) {
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

/// Replace virtual loss placeholder with the real NN value.
fn correct_virtual_loss(arena: &mut NodeArena, mut node_id: NodeId, mut real_value: f32) {
    let mut virtual_value = -1.0f32;
    loop {
        let node = arena.get_mut(node_id);
        node.value_sum += real_value - virtual_value;
        real_value = -real_value;
        virtual_value = -virtual_value;
        match node.parent {
            Some(parent) => node_id = parent,
            None => break,
        }
    }
}

/// Terminal game value from a player's perspective.
fn terminal_value(board: &ZertzBoard, perspective: Player) -> f32 {
    match board.outcome() {
        Outcome::WonBy(p) => {
            if p == perspective {
                1.0
            } else {
                -1.0
            }
        }
        Outcome::Draw | Outcome::Ongoing => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Expansion (cheap: just stores edges, no board clones)
// ---------------------------------------------------------------------------

/// Expand a node: populate its edges with legal moves and normalized priors.
/// No child boards are created here — that happens lazily during selection.
fn expand_with_policy(arena: &mut NodeArena, node_id: NodeId, policy: &[f32]) {
    let board = &arena.get(node_id).board;
    let (_mask, indexed_moves) = get_legal_move_mask(board);
    if indexed_moves.is_empty() {
        // Terminal or no moves — mark as expanded with empty edges.
        // This prevents re-selection as an unexpanded leaf.
        arena.get_mut(node_id).edges = Vec::new();
        // Use a sentinel: push a dummy edge so is_expanded() returns true.
        // Actually, let's handle this differently — see is_terminal check in select_leaves.
        return;
    }

    let mut total_prior = 0.0f32;
    let mut edges = Vec::with_capacity(indexed_moves.len());
    for &(idx, mv) in &indexed_moves {
        let prior = policy[idx];
        total_prior += prior;
        edges.push(Edge {
            mv,
            prior,
            child_id: None,
        });
    }

    // Normalize priors
    if total_prior > 0.0 {
        for edge in &mut edges {
            edge.prior /= total_prior;
        }
    }

    arena.get_mut(node_id).edges = edges;
}

// ---------------------------------------------------------------------------
// MctsSearch
// ---------------------------------------------------------------------------

/// Single-game MCTS search engine.
pub struct MctsSearch {
    pub arena: NodeArena,
    pub c_puct: f32,
    pub root: NodeId,
    pub use_forced_playouts: bool,
}

impl MctsSearch {
    pub fn new(capacity: usize) -> Self {
        MctsSearch {
            arena: NodeArena::new(capacity),
            c_puct: DEFAULT_C_PUCT,
            root: 0,
            use_forced_playouts: false,
        }
    }

    /// Initialize search for a game position. Only creates edges (fast).
    pub fn init(&mut self, board: &ZertzBoard, policy: &[f32]) {
        self.arena.reset();
        let root_node = MctsNode {
            board: board.clone(),
            parent: None,
            parent_edge_idx: 0,
            visit_count: 0,
            value_sum: 0.0,
            move_from_parent: None,
            edges: Vec::new(),
        };
        let root = self.arena.alloc(root_node);
        self.root = root;
        expand_with_policy(&mut self.arena, root, policy);
    }

    /// Select leaves for batch evaluation. Terminal nodes are handled immediately.
    pub fn select_leaves(&mut self, batch_size: usize) -> Vec<NodeId> {
        let root_player = self.arena.get(self.root).board.next_player();
        let mut leaves = Vec::new();

        for _ in 0..batch_size {
            let leaf =
                select_leaf(&mut self.arena, self.root, self.c_puct, self.use_forced_playouts);

            if self.arena.get(leaf).board.is_game_over() {
                let value = terminal_value(&self.arena.get(leaf).board, root_player);
                backpropagate(&mut self.arena, leaf, value);
            } else if self.arena.get(leaf).is_expanded() {
                // Expanded but no legal moves (shouldn't happen if outcome is None, but be safe)
                backpropagate(&mut self.arena, leaf, 0.0);
            } else {
                apply_virtual_loss(&mut self.arena, leaf);
                leaves.push(leaf);
            }
        }

        leaves
    }

    /// Expand leaf nodes with policies and backpropagate values.
    pub fn expand_and_backprop(
        &mut self,
        leaves: &[NodeId],
        policies: &[Vec<f32>],
        values: &[f32],
    ) {
        let root_player = self.arena.get(self.root).board.next_player();
        for (i, &leaf) in leaves.iter().enumerate() {
            expand_with_policy(&mut self.arena, leaf, &policies[i]);
            let mut value = values[i];
            if self.arena.get(leaf).board.next_player() != root_player {
                value = -value;
            }
            correct_virtual_loss(&mut self.arena, leaf, value);
        }
    }

    /// Apply Dirichlet noise to root edge priors.
    pub fn apply_root_dirichlet(&mut self, alpha: f32, epsilon: f32) {
        use rand::SeedableRng;
        use rand_distr::{Dirichlet, Distribution};

        let edge_count = self.arena.get(self.root).edges.len();
        if edge_count == 0 {
            return;
        }

        let alphas = vec![alpha; edge_count];
        let dirichlet = match Dirichlet::new(&alphas) {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut rng = rand::rngs::StdRng::from_entropy();
        let noise: Vec<f32> = dirichlet.sample(&mut rng);

        let edges = &mut self.arena.get_mut(self.root).edges;
        for (i, edge) in edges.iter_mut().enumerate() {
            edge.prior = (1.0 - epsilon) * edge.prior + epsilon * noise[i];
        }
    }

    /// Get the best move by visit count.
    pub fn best_move(&self) -> Option<crate::zertz::ZertzMove> {
        let root = self.arena.get(self.root);
        let mut best_visits = 0u32;
        let mut best_move = None;

        for edge in &root.edges {
            let visits = edge
                .child_id
                .map(|cid| self.arena.get(cid).visit_count)
                .unwrap_or(0);
            if visits > best_visits {
                best_visits = visits;
                best_move = Some(edge.mv);
            }
        }

        best_move
    }

    /// Get visit count distribution for training policy.
    pub fn get_visit_distribution(&self) -> Vec<(crate::zertz::ZertzMove, f32)> {
        let root = self.arena.get(self.root);
        let mut result = Vec::new();
        let mut total_visits = 0u32;

        for edge in &root.edges {
            let visits = edge
                .child_id
                .map(|cid| self.arena.get(cid).visit_count)
                .unwrap_or(0);
            if visits > 0 {
                result.push((edge.mv, visits as f32));
                total_visits += visits;
            }
        }

        if total_visits > 0 {
            for item in &mut result {
                item.1 /= total_visits as f32;
            }
        }

        result
    }

    /// Get visit distribution with KataGo-style policy target pruning.
    pub fn get_pruned_visit_distribution(&self) -> Vec<(crate::zertz::ZertzMove, f32)> {
        let root = self.arena.get(self.root);
        let parent_visits = root.visit_count;
        let n_total = parent_visits as f32;

        struct ChildInfo {
            mv: crate::zertz::ZertzMove,
            visits: u32,
            prior: f32,
            value: f32,
        }
        let mut children: Vec<ChildInfo> = Vec::new();
        for edge in &root.edges {
            let (visits, value) = match edge.child_id {
                Some(cid) => {
                    let c = self.arena.get(cid);
                    (c.visit_count, c.value())
                }
                None => continue, // never visited, skip
            };
            if visits == 0 {
                continue;
            }
            children.push(ChildInfo {
                mv: edge.mv,
                visits,
                prior: edge.prior,
                value,
            });
        }

        if children.is_empty() {
            return Vec::new();
        }

        let best_idx = children
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| c.visits)
            .map(|(i, _)| i)
            .unwrap();
        let best_visits = children[best_idx].visits;
        let best_value = children[best_idx].value;

        let best_puct = best_value
            + self.c_puct * children[best_idx].prior * n_total.sqrt()
                / (1.0 + best_visits as f32);

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
            let max_subtract = n_forced.min(child.visits.saturating_sub(1));
            for subtract in (1..=max_subtract).rev() {
                let new_visits = child.visits - subtract;
                let child_puct = child.value
                    + self.c_puct * child.prior * n_total.sqrt() / (1.0 + new_visits as f32);
                if child_puct < best_puct {
                    adjusted_visits[i] = new_visits;
                    break;
                }
            }
        }

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

        if total_visits > 0 {
            for item in &mut result {
                item.1 /= total_visits as f32;
            }
        }

        result
    }

    /// Encode a leaf node's board state for NN evaluation.
    /// Returns (board_flat, reserve_flat).
    pub fn encode_leaf(&self, leaf: NodeId) -> (Vec<f32>, Vec<f32>) {
        let board = &self.arena.get(leaf).board;
        let mut board_buf = vec![0.0f32; NUM_CHANNELS * GRID_SIZE * GRID_SIZE];
        let mut reserve_buf = vec![0.0f32; RESERVE_SIZE];
        encode_board(board, &mut board_buf, &mut reserve_buf);
        (board_buf, reserve_buf)
    }

    /// Root value estimate.
    pub fn root_value(&self) -> f32 {
        self.arena.get(self.root).value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcts_init() {
        let board = ZertzBoard::default();
        let uniform_policy = vec![1.0 / POLICY_SIZE as f32; POLICY_SIZE];
        let mut search = MctsSearch::new(1000);
        search.init(&board, &uniform_policy);

        let root = search.arena.get(search.root);
        assert!(root.is_expanded());
        assert!(!root.edges.is_empty());
        // Should have 1944 edges but only 1 node (the root — no children yet)
        assert_eq!(search.arena.len(), 1);
    }

    #[test]
    fn test_mcts_select_and_expand() {
        let board = ZertzBoard::default();
        let uniform_policy = vec![1.0 / POLICY_SIZE as f32; POLICY_SIZE];
        let mut search = MctsSearch::new(10000);
        search.init(&board, &uniform_policy);

        let leaves = search.select_leaves(8);
        assert!(!leaves.is_empty());
        // Should have created child nodes lazily (root + 8 children)
        assert_eq!(search.arena.len(), 1 + leaves.len());

        let policies: Vec<Vec<f32>> = leaves.iter().map(|_| uniform_policy.clone()).collect();
        let values: Vec<f32> = vec![0.0; leaves.len()];
        search.expand_and_backprop(&leaves, &policies, &values);

        assert!(search.arena.get(search.root).visit_count > 0);
    }

    #[test]
    fn test_best_move() {
        let board = ZertzBoard::default();
        let uniform_policy = vec![1.0 / POLICY_SIZE as f32; POLICY_SIZE];
        let mut search = MctsSearch::new(10000);
        search.init(&board, &uniform_policy);

        for _ in 0..10 {
            let leaves = search.select_leaves(8);
            if leaves.is_empty() {
                break;
            }
            let policies: Vec<Vec<f32>> = leaves.iter().map(|_| uniform_policy.clone()).collect();
            let values: Vec<f32> = vec![0.0; leaves.len()];
            search.expand_and_backprop(&leaves, &policies, &values);
        }

        let best = search.best_move();
        assert!(best.is_some());
    }

    #[test]
    fn test_visit_distribution() {
        let board = ZertzBoard::default();
        let uniform_policy = vec![1.0 / POLICY_SIZE as f32; POLICY_SIZE];
        let mut search = MctsSearch::new(10000);
        search.init(&board, &uniform_policy);

        for _ in 0..5 {
            let leaves = search.select_leaves(4);
            if leaves.is_empty() {
                break;
            }
            let policies: Vec<Vec<f32>> = leaves.iter().map(|_| uniform_policy.clone()).collect();
            let values: Vec<f32> = vec![0.0; leaves.len()];
            search.expand_and_backprop(&leaves, &policies, &values);
        }

        let dist = search.get_visit_distribution();
        assert!(!dist.is_empty());

        let total: f32 = dist.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_lazy_creation_efficiency() {
        // With 20 simulations, we should create far fewer than 1944 nodes.
        let board = ZertzBoard::default();
        let uniform_policy = vec![1.0 / POLICY_SIZE as f32; POLICY_SIZE];
        let mut search = MctsSearch::new(10000);
        search.init(&board, &uniform_policy);

        for _ in 0..3 {
            let leaves = search.select_leaves(8);
            if leaves.is_empty() {
                break;
            }
            let policies: Vec<Vec<f32>> = leaves.iter().map(|_| uniform_policy.clone()).collect();
            let values: Vec<f32> = vec![0.0; leaves.len()];
            search.expand_and_backprop(&leaves, &policies, &values);
        }

        // Should be much less than 1944
        assert!(search.arena.len() < 100, "arena has {} nodes", search.arena.len());
    }
}
