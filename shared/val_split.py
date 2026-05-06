"""Deterministic train/val split for supervised pretrain.

Picks the val subset by stable hash of `sgf_name` so the same game lands in
the same bucket on every run, regardless of random seed or game-list order.
"""

from __future__ import annotations

import zlib
from typing import Sequence, TypeVar

T = TypeVar("T", bound=tuple)


def split_train_val(
    games: Sequence[T],
    val_frac: float,
    *,
    sgf_name_index: int = 1,
) -> tuple[list[T], list[T]]:
    """Split `games` into (train, val) by hash of the sgf_name field.

    `games` is a list of tuples; element `sgf_name_index` is the SGF filename
    used as the hash key. CRC32 keeps the assignment stable across Python runs.
    """
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    if val_frac == 0.0:
        return list(games), []
    threshold = int(val_frac * 1000)
    train: list[T] = []
    val: list[T] = []
    for row in games:
        sgf_name = row[sgf_name_index]
        bucket = zlib.crc32(sgf_name.encode("utf-8")) % 1000
        (val if bucket < threshold else train).append(row)
    return train, val
