/// MCTS Node with linked-list children for arena allocation.

use crate::game::{Game, Move};
use super::arena::NodeId;

#[derive(Clone)]
pub struct MctsNode {
    pub game: Game,
    pub parent: Option<NodeId>,
    pub first_child: Option<NodeId>,
    pub next_sibling: Option<NodeId>,
    pub visit_count: u32,
    pub value_sum: f32,
    pub prior: f32,
    pub is_expanded: bool,
    pub move_from_parent: Move,
    pub child_count: u16,
}

impl MctsNode {
    pub fn new(game: Game, parent: Option<NodeId>, mv: Move, prior: f32) -> Self {
        MctsNode {
            game,
            parent,
            first_child: None,
            next_sibling: None,
            visit_count: 0,
            value_sum: 0.0,
            prior,
            is_expanded: false,
            move_from_parent: mv,
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

impl Default for MctsNode {
    fn default() -> Self {
        MctsNode::new(Game::new(), None, Move::pass(), 0.0)
    }
}
