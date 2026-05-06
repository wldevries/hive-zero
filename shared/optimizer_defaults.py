"""Optimizer-aware defaults for `lr` and `weight_decay`.

AdamW decouples weight decay from the LR step, so its sensible default is
~100× higher than SGD's. AdamW also typically uses a smaller LR.
"""

from __future__ import annotations

# (lr, weight_decay) per optimizer kind, used when the CLI didn't supply a value.
_DEFAULTS: dict[str, tuple[float, float]] = {
    "sgd": (0.005, 1e-4),
    "adamw": (1e-3, 1e-2),
}


def resolve_optimizer_defaults(
    optimizer: str,
    lr: float | None,
    weight_decay: float | None,
) -> tuple[float, float]:
    """Resolve (lr, weight_decay), filling in optimizer-specific defaults for None values."""
    kind = optimizer.lower()
    if kind not in _DEFAULTS:
        raise ValueError(f"unknown optimizer {optimizer!r}; expected one of {list(_DEFAULTS)}")
    default_lr, default_wd = _DEFAULTS[kind]
    return (default_lr if lr is None else lr,
            default_wd if weight_decay is None else weight_decay)
