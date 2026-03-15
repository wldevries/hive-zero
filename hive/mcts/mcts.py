"""Monte Carlo Tree Search with neural network evaluation."""

from __future__ import annotations
import math
import numpy as np
from typing import Optional

from ..core.game import Game, GameState
from ..core.pieces import PieceColor
from ..encoding.board_encoder import encode_board
from ..encoding.move_encoder import get_legal_move_mask, encode_move, POLICY_SIZE


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = ['game', 'parent', 'move', 'children', 'visit_count',
                 'value_sum', 'prior', 'is_expanded']

    def __init__(self, game: Game, parent: Optional[MCTSNode] = None,
                 move=None, prior: float = 0.0):
        self.game = game
        self.parent = parent
        self.move = move  # (piece, from_pos, to_pos) or None for pass
        self.children: list[MCTSNode] = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float = 1.5) -> float:
        """Upper Confidence Bound score for selection."""
        if self.parent is None:
            return 0.0
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value + exploration

    def best_child(self, c_puct: float = 1.5) -> MCTSNode:
        return max(self.children, key=lambda c: c.ucb_score(c_puct))

    def select_leaf(self, c_puct: float = 1.5) -> MCTSNode:
        """Traverse tree to find a leaf node."""
        node = self
        while node.is_expanded and node.children:
            node = node.best_child(c_puct)
        return node


class MCTS:
    """Monte Carlo Tree Search using neural network for evaluation."""

    def __init__(self, model=None, c_puct: float = 1.5, device: str = "cpu"):
        self.model = model
        self.c_puct = c_puct
        self.device = device

    def search(self, game: Game, max_simulations: int = 800) -> Optional[tuple]:
        """Run MCTS and return the best move.

        Returns (piece, from_pos, to_pos) or None for pass.
        """
        root = MCTSNode(game.copy())
        self._expand(root)

        if not root.children:
            return None  # pass

        for _ in range(max_simulations):
            leaf = root.select_leaf(self.c_puct)

            if leaf.game.is_game_over:
                value = self._terminal_value(leaf.game, root.game.turn_color)
            else:
                value = self._expand(leaf)
                # Flip value since it's from the perspective of the node's player
                if leaf.game.turn_color != root.game.turn_color:
                    value = -value

            self._backpropagate(leaf, value)

        # Select move with highest visit count
        best = max(root.children, key=lambda c: c.visit_count)
        return best.move

    def get_policy(self, game: Game, max_simulations: int = 800,
                   temperature: float = 1.0) -> tuple[list, np.ndarray]:
        """Run MCTS and return (moves, visit_count_distribution).

        Used for self-play training data generation.
        """
        root = MCTSNode(game.copy())
        self._expand(root)

        if not root.children:
            return [], np.array([])

        for _ in range(max_simulations):
            leaf = root.select_leaf(self.c_puct)

            if leaf.game.is_game_over:
                value = self._terminal_value(leaf.game, root.game.turn_color)
            else:
                value = self._expand(leaf)
                if leaf.game.turn_color != root.game.turn_color:
                    value = -value

            self._backpropagate(leaf, value)

        moves = [c.move for c in root.children]
        visit_counts = np.array([c.visit_count for c in root.children], dtype=np.float32)

        if temperature == 0:
            # Deterministic: pick the best
            probs = np.zeros_like(visit_counts)
            probs[np.argmax(visit_counts)] = 1.0
        else:
            # Apply temperature
            visit_counts = visit_counts ** (1.0 / temperature)
            probs = visit_counts / visit_counts.sum()

        return moves, probs

    def _expand(self, node: MCTSNode) -> float:
        """Expand a node: add children and return value estimate."""
        node.is_expanded = True
        game = node.game

        valid_moves = game.valid_moves()
        if not valid_moves:
            # Must pass
            child_game = game.copy()
            child_game.play_pass()
            child = MCTSNode(child_game, parent=node, move=None, prior=1.0)
            node.children.append(child)
            return self._evaluate(game)

        # Get neural network evaluation
        if self.model is not None:
            policy, value = self._nn_evaluate(game)
        else:
            # Uniform policy fallback
            policy = np.ones(POLICY_SIZE, dtype=np.float32) / POLICY_SIZE
            value = 0.0

        # Create children with prior probabilities
        mask, indexed_moves = get_legal_move_mask(game)
        total_prior = 0.0

        for idx, piece, from_pos, to_pos in indexed_moves:
            prior = policy[idx]
            total_prior += prior

            child_game = game.copy()
            child_game.play_move(piece, from_pos, to_pos)
            child = MCTSNode(child_game, parent=node,
                             move=(piece, from_pos, to_pos), prior=prior)
            node.children.append(child)

        # Renormalize priors
        if total_prior > 0:
            for child in node.children:
                child.prior /= total_prior

        return value

    def _evaluate(self, game: Game) -> float:
        """Get value estimate for a position."""
        if self.model is not None:
            _, value = self._nn_evaluate(game)
            return value
        return 0.0  # neutral evaluation without model

    def _nn_evaluate(self, game: Game) -> tuple[np.ndarray, float]:
        """Run neural network on game state."""
        import torch

        board_tensor, reserve_vector = encode_board(game)
        bt = torch.tensor(board_tensor).unsqueeze(0).to(self.device)
        rv = torch.tensor(reserve_vector).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(bt, rv)

        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        value = value.cpu().item()

        return policy, value

    def _terminal_value(self, game: Game, perspective_color: PieceColor) -> float:
        """Value of a terminal position from perspective_color's view."""
        if game.state == GameState.DRAW:
            return 0.0
        if game.state == GameState.WHITE_WINS:
            return 1.0 if perspective_color == PieceColor.WHITE else -1.0
        if game.state == GameState.BLACK_WINS:
            return 1.0 if perspective_color == PieceColor.BLACK else -1.0
        return 0.0

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            value = -value  # flip for opponent
            current = current.parent
