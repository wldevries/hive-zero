/// MCTS search implementation with arena allocation and batch NN evaluation.

use super::arena::{NodeArena, NodeId};
use super::node::MctsNode;
use crate::game::{Game, GameState, Move};
use crate::piece::PieceColor;
use crate::move_encoding::get_legal_move_mask;
use crate::board_encoding::{encode_board, NUM_CHANNELS, RESERVE_SIZE};
use crate::board::GRID_SIZE;

const DEFAULT_C_PUCT: f32 = 1.5;

/// UCB score for child selection.
fn ucb_score(node: &MctsNode, parent_visits: u32, c_puct: f32) -> f32 {
    let exploration = c_puct * node.prior * (parent_visits as f32).sqrt()
        / (1.0 + node.visit_count as f32);
    node.value() + exploration
}

/// Select the best child by UCB score.
fn best_child(arena: &NodeArena, node_id: NodeId, c_puct: f32) -> NodeId {
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
fn select_leaf(arena: &NodeArena, root: NodeId, c_puct: f32) -> NodeId {
    let mut node_id = root;
    loop {
        let node = arena.get(node_id);
        if !node.is_expanded || node.first_child.is_none() {
            return node_id;
        }
        node_id = best_child(arena, node_id, c_puct);
    }
}

/// Backpropagate a value up the tree.
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

/// Apply virtual loss: increment visit_count and subtract 1 from value_sum up the tree.
/// This deters subsequent selections from taking the same path within a batch.
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

/// Correct virtual loss by replacing the -1 placeholder with the real value.
/// Does NOT increment visit_count (already done by apply_virtual_loss).
fn correct_virtual_loss(arena: &mut NodeArena, mut node_id: NodeId, mut real_value: f32) {
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
fn terminal_value(game: &Game, perspective: PieceColor) -> f32 {
    match game.state {
        GameState::Draw => 0.0,
        GameState::WhiteWins => {
            if perspective == PieceColor::White { 1.0 } else { -1.0 }
        }
        GameState::BlackWins => {
            if perspective == PieceColor::Black { 1.0 } else { -1.0 }
        }
        _ => 0.0,
    }
}

/// Expand a node with a policy vector, adding children to the arena.
fn expand_with_policy(arena: &mut NodeArena, node_id: NodeId, policy: &[f32]) {
    arena.get_mut(node_id).is_expanded = true;
    let game = arena.get(node_id).game.clone();

    let valid_moves = game.valid_moves();
    if valid_moves.is_empty() {
        // Pass
        let mut child_game = game.clone();
        child_game.play_pass();
        let child_id = arena.alloc(child_game, Some(node_id), Move::pass(), 1.0);
        arena.get_mut(node_id).first_child = Some(child_id);
        arena.get_mut(node_id).child_count = 1;
        return;
    }

    let (_mask, indexed_moves) = get_legal_move_mask(&game);
    let mut total_prior = 0.0f32;
    let mut first_child_id: Option<NodeId> = None;
    let mut prev_child_id: Option<NodeId> = None;
    let mut child_count = 0u16;

    for &(idx, mv) in &indexed_moves {
        let prior = policy[idx];
        total_prior += prior;

        let mut child_game = game.clone();
        child_game.play_move(&mv);
        let child_id = arena.alloc(child_game, Some(node_id), mv, prior);

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

/// Single-game MCTS search engine.
pub struct MctsSearch {
    pub arena: NodeArena,
    pub c_puct: f32,
    pub root: NodeId,
    /// Temporary storage for leaf IDs between select and expand calls.
    pub stashed_leaves: Vec<NodeId>,
}

impl MctsSearch {
    pub fn new(capacity: usize) -> Self {
        MctsSearch {
            arena: NodeArena::new(capacity),
            c_puct: DEFAULT_C_PUCT,
            root: 0, // will be set in init
            stashed_leaves: Vec::new(),
        }
    }

    /// Initialize search for a game position.
    pub fn init(&mut self, game: &Game, policy: &[f32]) {
        self.arena.reset();
        let root = self.arena.alloc(game.clone(), None, Move::pass(), 0.0);
        self.root = root;
        expand_with_policy(&mut self.arena, root, policy);
    }

    /// Run one batch of simulations. Returns list of (leaf_id, game) pairs
    /// that need NN evaluation. Terminal nodes are handled immediately.
    pub fn select_leaves(&mut self, batch_size: usize) -> Vec<NodeId> {
        let root_turn = self.arena.get(self.root).game.turn_color;
        let mut leaves = Vec::new();

        for _ in 0..batch_size {
            let leaf = select_leaf(&self.arena, self.root, self.c_puct);

            if self.arena.get(leaf).game.is_game_over() {
                let value = terminal_value(&self.arena.get(leaf).game, root_turn);
                backpropagate(&mut self.arena, leaf, value);
            } else {
                // Apply virtual loss so subsequent selections in this batch diverge.
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
        let root_turn = self.arena.get(self.root).game.turn_color;
        for (i, &leaf) in leaves.iter().enumerate() {
            expand_with_policy(&mut self.arena, leaf, &policies[i]);
            let mut value = values[i];
            if self.arena.get(leaf).game.turn_color != root_turn {
                value = -value;
            }
            // Replace virtual loss placeholder with real value (visit_count already correct).
            correct_virtual_loss(&mut self.arena, leaf, value);
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
    pub fn best_move(&self) -> Option<Move> {
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
    pub fn get_visit_distribution(&self) -> Vec<(Move, f32)> {
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

    /// Encode a leaf node's game state for NN evaluation.
    pub fn encode_leaf(&self, leaf: NodeId) -> (Vec<f32>, Vec<f32>) {
        let game = &self.arena.get(leaf).game;
        let mut board = vec![0.0f32; NUM_CHANNELS * GRID_SIZE * GRID_SIZE];
        let mut reserve = vec![0.0f32; RESERVE_SIZE];
        encode_board(game, &mut board, &mut reserve);
        (board, reserve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::move_encoding::POLICY_SIZE;

    #[test]
    fn test_mcts_init() {
        let game = Game::new();
        let uniform_policy = vec![1.0 / POLICY_SIZE as f32; POLICY_SIZE];
        let mut search = MctsSearch::new(1000);
        search.init(&game, &uniform_policy);

        // Root should be expanded with 4 children (4 piece types, queen excluded, deduped)
        let root = search.arena.get(search.root);
        assert!(root.is_expanded);
        assert_eq!(root.child_count, 4);
    }

    #[test]
    fn test_mcts_select_and_expand() {
        let game = Game::new();
        let uniform_policy = vec![1.0 / POLICY_SIZE as f32; POLICY_SIZE];
        let mut search = MctsSearch::new(10000);
        search.init(&game, &uniform_policy);

        // Select leaves
        let leaves = search.select_leaves(8);
        assert!(!leaves.is_empty());

        // Expand with uniform policies
        let policies: Vec<Vec<f32>> = leaves.iter().map(|_| uniform_policy.clone()).collect();
        let values: Vec<f32> = vec![0.0; leaves.len()];
        search.expand_and_backprop(&leaves, &policies, &values);

        // Root should have visits now
        assert!(search.arena.get(search.root).visit_count > 0);
    }

    #[test]
    fn test_best_move() {
        let game = Game::new();
        let uniform_policy = vec![1.0 / POLICY_SIZE as f32; POLICY_SIZE];
        let mut search = MctsSearch::new(10000);
        search.init(&game, &uniform_policy);

        // Run some simulations
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
}
