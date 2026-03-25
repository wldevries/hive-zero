/// MCTS Node with linked-list children for arena allocation.
/// Game states are NOT stored in nodes — they are reconstructed by replaying
/// moves from the root game during selection. This reduces per-node memory
/// from ~4KB+ to ~60 bytes.

use crate::game::Player;
use super::arena::NodeId;

#[derive(Clone)]
pub struct MctsNode<M: Copy> {
    pub parent: Option<NodeId>,
    pub first_child: Option<NodeId>,
    pub next_sibling: Option<NodeId>,
    pub visit_count: u32,
    pub value_sum: f32,
    pub prior: f32,
    pub is_expanded: bool,
    pub move_from_parent: M,
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

    #[inline]
    pub fn value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }
}
