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
    """A node in the MCTS tree.

    Uses lazy game state creation - child game states are only materialized
    when the node is selected for expansion.
    """

    __slots__ = ['game', 'parent', 'move', 'children', 'visit_count',
                 'value_sum', 'prior', 'is_expanded', '_legal_moves']

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
        self._legal_moves = None  # cached

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
    """Monte Carlo Tree Search using neural network for evaluation.

    Collects multiple leaf nodes and evaluates them in a single batched
    GPU call to avoid per-simulation inference overhead.
    """

    def __init__(self, model=None, c_puct: float = 1.5, device: str = "cpu",
                 batch_size: int = 8):
        self.model = model
        self.c_puct = c_puct
        self.device = device
        self.batch_size = batch_size

    def search(self, game: Game, max_simulations: int = 800) -> Optional[tuple]:
        """Run MCTS and return the best move."""
        root = MCTSNode(game.copy())
        self._expand_single(root)

        if not root.children:
            return None

        sims_done = 0
        while sims_done < max_simulations:
            batch = min(self.batch_size, max_simulations - sims_done)
            self._run_batch(root, batch)
            sims_done += batch

        best = max(root.children, key=lambda c: c.visit_count)
        return best.move

    def get_policy(self, game: Game, max_simulations: int = 800,
                   temperature: float = 1.0) -> tuple[list, np.ndarray]:
        """Run MCTS and return (moves, visit_count_distribution)."""
        root = MCTSNode(game.copy())
        self._expand_single(root)

        if not root.children:
            return [], np.array([])

        sims_done = 0
        while sims_done < max_simulations:
            batch = min(self.batch_size, max_simulations - sims_done)
            self._run_batch(root, batch)
            sims_done += batch

        moves = [c.move for c in root.children]
        visit_counts = np.array([c.visit_count for c in root.children], dtype=np.float32)

        if temperature == 0:
            probs = np.zeros_like(visit_counts)
            probs[np.argmax(visit_counts)] = 1.0
        else:
            visit_counts = visit_counts ** (1.0 / temperature)
            total = visit_counts.sum()
            probs = visit_counts / total if total > 0 else np.ones_like(visit_counts) / len(visit_counts)

        return moves, probs

    def _run_batch(self, root: MCTSNode, batch_size: int):
        """Select multiple leaves, batch-evaluate them, then backpropagate."""
        leaves = []
        terminal_leaves = []

        for _ in range(batch_size):
            leaf = root.select_leaf(self.c_puct)

            if leaf.game.is_game_over:
                terminal_leaves.append(leaf)
            elif leaf in [l for l in leaves]:
                # Same leaf selected again - just treat as terminal with value 0
                terminal_leaves.append(leaf)
            else:
                leaves.append(leaf)

        # Handle terminal nodes immediately
        for leaf in terminal_leaves:
            value = self._terminal_value(leaf.game, root.game.turn_color)
            self._backpropagate(leaf, value)

        if not leaves:
            return

        # Batch neural network evaluation
        if self.model is not None:
            policies, values = self._nn_evaluate_batch([l.game for l in leaves])
        else:
            policies = [np.ones(POLICY_SIZE, dtype=np.float32) / POLICY_SIZE] * len(leaves)
            values = [0.0] * len(leaves)

        # Expand each leaf and backpropagate
        for leaf, policy, value in zip(leaves, policies, values):
            self._expand_with_policy(leaf, policy)
            if leaf.game.turn_color != root.game.turn_color:
                value = -value
            self._backpropagate(leaf, value)

    def _expand_single(self, node: MCTSNode):
        """Expand a single node (used for root)."""
        if self.model is not None:
            policy, value = self._nn_evaluate_single(node.game)
        else:
            policy = np.ones(POLICY_SIZE, dtype=np.float32) / POLICY_SIZE
        self._expand_with_policy(node, policy)

    def _expand_with_policy(self, node: MCTSNode, policy: np.ndarray):
        """Expand a node using a precomputed policy."""
        node.is_expanded = True
        game = node.game

        valid_moves = game.valid_moves()
        if not valid_moves:
            child_game = game.copy()
            child_game.play_pass()
            child = MCTSNode(child_game, parent=node, move=None, prior=1.0)
            node.children.append(child)
            return

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

        if total_prior > 0:
            for child in node.children:
                child.prior /= total_prior

    def _nn_evaluate_single(self, game: Game) -> tuple[np.ndarray, float]:
        """Single game NN evaluation."""
        import torch
        board_tensor, reserve_vector = encode_board(game)
        bt = torch.tensor(board_tensor).unsqueeze(0).to(self.device)
        rv = torch.tensor(reserve_vector).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(bt, rv)

        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        return policy, value.cpu().item()

    def _nn_evaluate_batch(self, games: list[Game]) -> tuple[list[np.ndarray], list[float]]:
        """Batch neural network evaluation - single GPU call for multiple positions."""
        import torch

        boards = []
        reserves = []
        for game in games:
            bt, rv = encode_board(game)
            boards.append(bt)
            reserves.append(rv)

        bt_batch = torch.tensor(np.stack(boards)).to(self.device)
        rv_batch = torch.tensor(np.stack(reserves)).to(self.device)

        with torch.no_grad():
            policy_logits, values = self.model(bt_batch, rv_batch)

        policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
        values = values.cpu().numpy().flatten()

        return [policies[i] for i in range(len(games))], values.tolist()

    def _terminal_value(self, game: Game, perspective_color: PieceColor) -> float:
        if game.state == GameState.DRAW:
            return 0.0
        if game.state == GameState.WHITE_WINS:
            return 1.0 if perspective_color == PieceColor.WHITE else -1.0
        if game.state == GameState.BLACK_WINS:
            return 1.0 if perspective_color == PieceColor.BLACK else -1.0
        return 0.0

    def _backpropagate(self, node: MCTSNode, value: float):
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            value = -value
            current = current.parent
