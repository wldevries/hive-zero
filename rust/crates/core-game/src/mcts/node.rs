/// MCTS Node with linked-list children for arena allocation.
/// Game states are NOT stored in nodes — they are reconstructed by replaying
/// moves from the root game during selection. This reduces per-node memory
/// from ~4KB+ to ~60 bytes.
///
/// # Value convention
/// `value_sum` accumulates values from the *parent's player's* perspective:
/// positive means the move that led here was good for whoever chose it.
/// UCB selection uses `node.value() + exploration` directly (no negation).
/// The root has no parent, so `root_value()` negates to recover the root
/// player's own expected return. See docs/mcts_value_convention.md.

use crate::game::Player;
use super::arena::NodeId;

#[derive(Clone)]
pub struct MctsNode<M: Copy> {
    pub parent: Option<NodeId>,
    pub first_child: Option<NodeId>,
    pub next_sibling: Option<NodeId>,
    /// Number of times this node has been selected (includes virtual-loss visits).
    pub visit_count: u32,
    /// Sum of backed-up values from the *parent's* player's perspective.
    /// Positive means the move that led here was good for whoever chose it.
    pub value_sum: f32,
    /// Neural network prior probability for the move that leads to this node.
    pub prior: f32,
    /// Whether this node's children have been added to the arena.
    pub is_expanded: bool,
    /// The move played from the parent to reach this node.
    pub move_from_parent: M,
    /// The player to move at this node (not the player who moved here).
    pub turn_player: Player,
    pub child_count: u16,
}

impl<M: Copy> MctsNode<M> {
    pub fn new(parent: Option<NodeId>, mv: M, prior: f32, turn_player: Player) -> Self {
        MctsNode {
            parent,
            first_child: None,
            next_sibling: None,
            visit_count: 0,
            value_sum: 0.0,
            prior,
            is_expanded: false,
            move_from_parent: mv,
            turn_player,
            child_count: 0,
        }
    }

    /// Mean backed-up value from the *parent's* player's perspective.
    /// Positive means this was a good move for whoever chose it.
    /// Returns 0 for unvisited nodes.
    /// For the root's own perspective, use `MctsSearch::root_value()` instead.
    #[inline]
    pub fn value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }
}
