"""Shared Python-side config dataclasses for AlphaZero-style self-play.

These mirror the Rust structs in `core_game::selfplay_config` and reduce
the long flat-kwargs explosion in each game's `SelfPlayTrainer.run(...)`
signature and CLI → orchestrator → pyo3-session forwarding chain.

Each dataclass has a `session_kwargs()` method that returns the dict
to splat into the corresponding `*SelfPlaySession(...)` pyo3 constructor,
so the boundary between Python config and Rust pyo3 kwargs is one place
to maintain.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class MctsConfig:
    """MCTS-related knobs that every game's self-play loop uses identically."""
    simulations: int = 200
    c_puct: float = 1.5
    dir_alpha: float = 0.3
    dir_epsilon: float = 0.25
    forced_playouts: bool = False
    draw_contempt: float = 0.0
    play_batch_size: int = 8
    temperature: float = 1.0
    temp_threshold: int = 20

    def session_kwargs(self) -> dict:
        # Pyo3 session kwargs match field names 1:1.
        return asdict(self)


@dataclass
class PlayoutCapConfig:
    """KataGo-style playout-cap randomization. p == 0.0 disables it."""
    p: float = 0.0
    fast_cap: int = 30

    def session_kwargs(self) -> dict:
        # Pyo3 session prefixes p as `playout_cap_p`.
        return {"playout_cap_p": self.p, "fast_cap": self.fast_cap}


@dataclass
class OpeningRandomConfig:
    """Random-opening-moves window. Each game samples uniformly in [min, max]."""
    min: int = 0
    max: int = 0

    @classmethod
    def from_cli_arg(cls, arg: int | tuple[int, int]) -> "OpeningRandomConfig":
        """Build from `--random-opening-moves N` (int) or `N-M` (tuple)."""
        if isinstance(arg, tuple):
            return cls(min=arg[0], max=arg[1])
        return cls(min=arg, max=arg)

    def session_kwargs(self) -> dict:
        return {
            "random_opening_moves_min": self.min,
            "random_opening_moves_max": self.max,
        }
