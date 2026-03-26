/// MCTS Node with lazy child creation via edges.
///
/// Instead of pre-creating all child nodes during expansion (expensive when
/// there are 1000+ legal moves), we store edges with moves and priors.
/// Child nodes are created lazily when first selected.

use crate::zertz::{ZertzBoard, ZertzMove};
use super::arena::NodeId;

/// Edge from parent to potential child. Stores the move and NN prior.
/// The child node is created lazily on first visit.
pub struct Edge {
    pub mv: ZertzMove,
    pub prior: f32,
    pub child_id: Option<NodeId>,
}

pub struct MctsNode {
    pub board: ZertzBoard,
    pub parent: Option<NodeId>,
    /// Index of the edge in parent's edges Vec that points to this node.
    pub parent_edge_idx: u16,
    pub visit_count: u32,
    pub value_sum: f32,
    pub move_from_parent: Option<ZertzMove>,
    /// Populated during expansion. Empty means unexpanded.
    pub edges: Vec<Edge>,
}

impl MctsNode {
    #[inline]
    pub fn value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }

    pub fn is_expanded(&self) -> bool {
        !self.edges.is_empty()
    }
}

impl Default for MctsNode {
    fn default() -> Self {
        MctsNode {
            board: ZertzBoard::default(),
            parent: None,
            parent_edge_idx: 0,
            visit_count: 0,
            value_sum: 0.0,
            move_from_parent: None,
            edges: Vec::new(),
        }
    }
}
