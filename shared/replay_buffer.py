"""Shared utilities for HDF5-backed ring-buffer replay buffers."""

from __future__ import annotations
import os
import numpy as np


def _ring_buffer_segments(
    stored_count: int, stored_max: int, stored_size: int, n_keep: int
) -> list[tuple[int, int]]:
    """Return contiguous (start, end) slices that yield the most recent n_keep samples
    in insertion order from the ring buffer."""
    if stored_count < stored_max:
        # Buffer not full; data occupies [0, stored_size) in order.
        n = min(stored_size, n_keep)
        return [(stored_size - n, stored_size)]
    else:
        # Full ring buffer; in-order: [ring_start, stored_max) then [0, ring_start)
        ring_start = int(stored_count % stored_max)
        n = min(stored_max, n_keep)
        skip = stored_max - n
        seg1_len = stored_max - ring_start
        if skip >= seg1_len:
            seg2_start = skip - seg1_len
            return [(seg2_start, ring_start)]
        else:
            return [(ring_start + skip, stored_max), (0, ring_start)]


def resize_h5_buffer(h5path: str, new_max_size: int) -> None:
    """Rebuild replay.h5 to new_max_size, preserving the most recent samples."""
    import h5py

    with h5py.File(h5path, "r") as old:
        stored_max   = int(old.attrs["max_size"])
        stored_count = int(old.attrs["count"])
        stored_size  = int(old.attrs["size"])

        segments = _ring_buffer_segments(stored_count, stored_max, stored_size, new_max_size)
        n = sum(e - s for s, e in segments)

        extra_attrs = {k: v for k, v in old.attrs.items()
                       if k not in ("max_size", "count", "size")}

        tmp_path = h5path + ".resize_tmp"
        with h5py.File(tmp_path, "w") as new:
            new.attrs["max_size"] = new_max_size
            new.attrs["count"] = n
            new.attrs["size"] = n
            for k, v in extra_attrs.items():
                new.attrs[k] = v

            ds_map = {}
            for name in old:
                shape = (new_max_size,) + old[name].shape[1:]
                ds_map[name] = new.create_dataset(name, shape=shape,
                                                   dtype=old[name].dtype)
            dst = 0
            for seg_start, seg_end in segments:
                seg_len = seg_end - seg_start
                for name in old:
                    ds_map[name][dst:dst + seg_len] = old[name][seg_start:seg_end]
                dst += seg_len

    os.replace(tmp_path, h5path)


def handle_buffer_size_mismatch(h5file, h5path: str, requested_max_size: int):
    """Handle a replay buffer max_size mismatch on resume.

    Prompts the user to resize (keeping most recent samples) or continue with
    the stored size. Closes h5file before acting, then returns a fresh handle.

    Returns (effective_max_size, reopened_h5file).
    """
    import h5py

    stored_max  = int(h5file.attrs["max_size"])
    stored_size = int(h5file.attrs["size"])

    print(f"\n  Replay buffer size mismatch:")
    print(f"    Stored:    {stored_max:,} max  ({stored_size:,} filled)")
    print(f"    Requested: {requested_max_size:,} max")

    h5file.close()
    choice = input("  Resize to new size? [y=resize / n=keep current] ").strip().lower()

    if choice == 'y':
        print(f"  Rebuilding {stored_max:,} → {requested_max_size:,} ...")
        resize_h5_buffer(h5path, requested_max_size)
        effective_max = requested_max_size
    else:
        effective_max = stored_max
        print(f"  Keeping current size ({stored_max:,})")

    return effective_max, h5py.File(h5path, "r+")
