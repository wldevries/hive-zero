/// Arena-based node allocation for MCTS.
/// Pre-allocates a pool of nodes to avoid per-node heap allocation.

use super::node::MctsNode;
use crate::game::Player;

/// Opaque node identifier.
pub type NodeId = u32;

/// Pre-allocated node pool.
pub struct NodeArena<M: Copy> {
    nodes: Vec<MctsNode<M>>,
    free_list: Vec<u32>,
}

impl<M: Copy> NodeArena<M> {
    pub fn new(capacity: usize, default_move: M) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        // Node 0 is reserved as "null"
        nodes.push(MctsNode::new(None, default_move, 0.0, Player::Player1));
        NodeArena {
            nodes,
            free_list: Vec::new(),
        }
    }

    /// Allocate a new node. Returns its ID.
    pub fn alloc(&mut self, parent: Option<NodeId>, mv: M, policy_prior: f32, turn_player: Player) -> NodeId {
        let node = MctsNode::new(parent, mv, policy_prior, turn_player);
        if let Some(id) = self.free_list.pop() {
            self.nodes[id as usize] = node;
            id
        } else {
            let id = self.nodes.len() as u32;
            self.nodes.push(node);
            id
        }
    }

    /// Get a reference to a node.
    #[inline]
    pub fn get(&self, id: NodeId) -> &MctsNode<M> {
        &self.nodes[id as usize]
    }

    /// Get a mutable reference to a node.
    #[inline]
    pub fn get_mut(&mut self, id: NodeId) -> &mut MctsNode<M> {
        &mut self.nodes[id as usize]
    }

    /// Reset the arena for reuse (keeps allocated memory).
    pub fn reset(&mut self) {
        self.nodes.truncate(1); // keep the null node
        self.free_list.clear();
    }

    /// Recursively free a subtree rooted at `root_id` into the free list.
    /// All descendants (including `root_id`) become available for reuse.
    pub fn free_subtree(&mut self, root_id: NodeId) {
        let mut stack = vec![root_id];
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id as usize];
            let mut child = node.first_child;
            while let Some(cid) = child {
                stack.push(cid);
                child = self.nodes[cid as usize].next_sibling;
            }
            self.free_list.push(id);
        }
    }

    /// Free a single node (no children) into the free list.
    pub fn free_node(&mut self, id: NodeId) {
        self.free_list.push(id);
    }

    /// Number of active nodes.
    pub fn len(&self) -> usize {
        self.nodes.len() - 1 // exclude null node
    }
}
