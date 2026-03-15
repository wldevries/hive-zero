"""MCTS module - imports Rust implementation if available, else Python fallback."""

try:
    from hive_engine import RustGame, RustMCTS
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

from .mcts import MCTS, MCTSNode
