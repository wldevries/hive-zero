"""Optimizer-aware defaults for `lr` and `weight_decay`.

AdamW decouples weight decay from the LR step, so its sensible default is
~100× higher than SGD's. AdamW also typically uses a smaller LR.
"""

from __future__ import annotations

# (lr, weight_decay, warmup_steps) per optimizer kind. AdamW's running variance
# estimates are noisy until they've seen a few hundred batches, so a short
# linear warmup avoids large/random first-step updates. SGD doesn't need it.
_DEFAULTS: dict[str, tuple[float, float, int]] = {
    "sgd":   (0.005, 1e-4, 0),
    "adamw": (1e-3,  1e-2, 500),
}


def resolve_optimizer_defaults(
    optimizer: str,
    lr: float | None,
    weight_decay: float | None,
    warmup_steps: int | None = None,
) -> tuple[float, float, int]:
    """Resolve (lr, weight_decay, warmup_steps), filling per-optimizer defaults for None values."""
    kind = optimizer.lower()
    if kind not in _DEFAULTS:
        raise ValueError(f"unknown optimizer {optimizer!r}; expected one of {list(_DEFAULTS)}")
    default_lr, default_wd, default_warmup = _DEFAULTS[kind]
    return (
        default_lr if lr is None else lr,
        default_wd if weight_decay is None else weight_decay,
        default_warmup if warmup_steps is None else warmup_steps,
    )
