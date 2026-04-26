/// MCTS Node with linked-list children for arena allocation.
/// Game states are NOT stored in nodes — they are reconstructed by replaying
/// moves from the root game during selection. This reduces per-node memory
/// from ~4KB+ to ~60 bytes.
///
/// # Value convention
/// `value_sum` accumulates the zero-sum (W − L) component from the *parent's
/// player's* perspective: positive means the move that led here was good for
/// whoever chose it. Sign-flips on every player boundary during backprop.
///
/// `draw_sum` accumulates the draw probability D *symmetrically* — never
/// sign-flipped, because both players see a draw with the same magnitude.
/// Combining them with `value(contempt) = (value_sum − contempt · draw_sum) / N`
/// gives the standard `W − L − contempt · D` per-player Q value for UCB.
///
/// UCB selection uses `node.value(contempt) + exploration` directly (no
/// negation). The root has no parent, so `root_value()` negates to recover
/// the root player's own expected return. See docs/mcts_value_convention.md.
///
/// # Prior convention
/// `policy_prior` is the raw NN softmax output and is never mutated after
/// node creation. `dirichlet_noise` is 0.0 for all non-root children and is
/// set (then cleared on reroot) only for the current root's direct children.
/// The effective prior used in UCB is computed via `prior(dir_epsilon)`.

use crate::game::Player;
use super::arena::NodeId;

#[derive(Clone)]
pub struct MctsNode<M: Copy> {
    pub parent: Option<NodeId>,
    pub first_child: Option<NodeId>,
    pub next_sibling: Option<NodeId>,
    /// Number of times this node has been selected (includes virtual-loss visits).
    pub visit_count: u32,
    /// Sum of zero-sum W−L components, from the *parent's* player's perspective.
    /// Positive means the move that led here was good for whoever chose it.
    /// Sign-flipped on every player boundary during backprop.
    pub value_sum: f32,
    /// Sum of draw-probability components. Symmetric — added unflipped at
    /// every ancestor regardless of player, because both players see a draw
    /// with the same magnitude. Combined with `value_sum` via `value(contempt)`.
    pub draw_sum: f32,
    /// Clean NN softmax probability — never mutated after node creation.
    pub policy_prior: f32,
    /// Dirichlet noise component; 0.0 for all non-root-child nodes.
    /// Set by `apply_root_dirichlet`, cleared on `reroot`.
    pub dirichlet_noise: f32,
    /// Whether this node's children have been added to the arena.
    pub is_expanded: bool,
    /// The move played from the parent to reach this node.
    pub move_from_parent: M,
    /// The player to move at this node (not the player who moved here).
    pub turn_player: Player,
    pub child_count: u16,
}

impl<M: Copy> MctsNode<M> {
    pub fn new(parent: Option<NodeId>, mv: M, policy_prior: f32, turn_player: Player) -> Self {
        MctsNode {
            parent,
            first_child: None,
            next_sibling: None,
            visit_count: 0,
            value_sum: 0.0,
            draw_sum: 0.0,
            policy_prior,
            dirichlet_noise: 0.0,
            is_expanded: false,
            move_from_parent: mv,
            turn_player,
            child_count: 0,
        }
    }

    /// Effective prior for UCB: mixes in Dirichlet noise when present.
    /// For non-root children (`dirichlet_noise == 0.0`) this is just `policy_prior`.
    #[inline]
    pub fn prior(&self, dir_epsilon: f32) -> f32 {
        if self.dirichlet_noise == 0.0 {
            self.policy_prior
        } else {
            (1.0 - dir_epsilon) * self.policy_prior + dir_epsilon * self.dirichlet_noise
        }
    }

    /// Mean backed-up Q value from the *parent's* player's perspective, with
    /// draw contempt applied: `(value_sum − contempt · draw_sum) / visit_count`.
    /// Positive means this was a good move for whoever chose it. Returns 0 for
    /// unvisited nodes. For the root's own perspective, use
    /// `MctsSearch::root_value()` instead.
    #[inline]
    pub fn value(&self, contempt: f32) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            (self.value_sum - contempt * self.draw_sum) / self.visit_count as f32
        }
    }
}
