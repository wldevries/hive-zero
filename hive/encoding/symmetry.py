"""On-the-fly symmetry augmentation for hex board tensors and policy targets.

Hex boards have 12 symmetries: 6 rotations × 2 (with/without mirror).
We apply a random transform per sample during training for free data augmentation.
"""

from __future__ import annotations
import numpy as np

from .board_encoder import GRID_SIZE, GRID_CENTER

# Rotation and mirror for THIS project's hex coord system.
# Directions: E(1,0), NE(0,-1), NW(-1,-1), W(-1,0), SW(0,1), SE(1,1)
# 60° rotation maps each direction to the next: (q,r) → (r, -q+r)
# Mirror swaps q and r: (q,r) → (r, q)

def _hex_rotate60(q: int, r: int) -> tuple[int, int]:
    """Rotate hex coord 60°: (q,r) → (r, -q+r)."""
    return (r, -q + r)


def _hex_mirror(q: int, r: int) -> tuple[int, int]:
    """Mirror hex coord: (q,r) → (r, q)."""
    return (r, q)


def _build_grid_permutation(transform_fn) -> np.ndarray:
    """Build a mapping: for each (row, col) in grid, where does it map to?

    Returns array of shape (GRID_SIZE, GRID_SIZE, 2) where [r,c] = (new_r, new_c).
    Cells that map outside the grid get (-1, -1).
    """
    perm = np.full((GRID_SIZE, GRID_SIZE, 2), -1, dtype=np.int32)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            # Grid to hex
            q = col - GRID_CENTER
            r = row - GRID_CENTER
            # Transform
            nq, nr = transform_fn(q, r)
            # Hex to grid
            new_col = nq + GRID_CENTER
            new_row = nr + GRID_CENTER
            if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE:
                perm[row, col] = (new_row, new_col)
    return perm


def _compose_transform(n_rotations: int, mirror: bool):
    """Create a composed transform function for n rotations + optional mirror."""
    def transform(q, r):
        if mirror:
            q, r = _hex_mirror(q, r)
        for _ in range(n_rotations):
            q, r = _hex_rotate60(q, r)
        return q, r
    return transform


# Direction channel permutation under rotation:
# Channels 0-5 are directions E, NE, NW, W, SW, SE.
# Rotating 60° CW shifts each direction by +1 (mod 6).
# Mirror (swap q,r) maps direction i → MIRROR_DIR[i].
#   E(1,0)→(0,1)=SW(4), NE(0,-1)→(-1,0)=W(3), NW(-1,-1)→(-1,-1)=NW(2),
#   W(-1,0)→(0,-1)=NE(1), SW(0,1)→(1,0)=E(0), SE(1,1)→(1,1)=SE(5)
MIRROR_DIR = [4, 3, 2, 1, 0, 5]

def _direction_channel_perm(n_rotations: int, mirror: bool) -> list[int]:
    """Compute how direction channels 0-5 permute under the transform.

    Returns a list where result[new_ch] = old_ch.
    """
    # Start with identity
    perm = list(range(6))
    if mirror:
        perm = [MIRROR_DIR[i] for i in perm]
    # Each 60° rotation shifts directions by +1, so new_ch i gets old_ch i-1
    for _ in range(n_rotations):
        perm = [(p - 1) % 6 for p in perm]
    return perm


# Precompute all 12 symmetry transforms with numpy fancy indexing arrays
_TRANSFORMS = []
for _mirror in [False, True]:
    for _nrot in range(6):
        grid_perm = _build_grid_permutation(_compose_transform(_nrot, _mirror))
        dir_perm = _direction_channel_perm(_nrot, _mirror)

        # Build fancy indexing arrays for valid cells only
        valid = grid_perm[:, :, 0] >= 0  # (GRID_SIZE, GRID_SIZE) bool mask
        src_rows, src_cols = np.where(valid)
        dst_rows = grid_perm[src_rows, src_cols, 0]
        dst_cols = grid_perm[src_rows, src_cols, 1]

        _TRANSFORMS.append((src_rows, src_cols, dst_rows, dst_cols, dir_perm))

# Clean up module-level loop variables
del _mirror, _nrot, grid_perm, dir_perm, valid, src_rows, src_cols, dst_rows, dst_cols


def apply_symmetry(board_tensor: np.ndarray, policy_target: np.ndarray,
                   sym_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply symmetry transform to board tensor and policy target.

    Args:
        board_tensor: shape (NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        policy_target: shape (POLICY_SIZE,) = (12 * GRID_SIZE * GRID_SIZE,)
        sym_index: 0-11, which symmetry to apply (0 = identity)

    Returns:
        Transformed (board_tensor, policy_target).
    """
    if sym_index == 0:
        return board_tensor, policy_target

    src_rows, src_cols, dst_rows, dst_cols, dir_perm = _TRANSFORMS[sym_index]
    gs = GRID_SIZE

    # Transform board tensor using fancy indexing
    new_board = np.zeros_like(board_tensor)
    new_board[:, dst_rows, dst_cols] = board_tensor[:, src_rows, src_cols]

    # Transform policy target
    policy_3d = policy_target.reshape(12, gs, gs)
    new_policy_3d = np.zeros_like(policy_3d)

    # Direction channels 0-5: spatial + channel permutation
    for new_ch in range(6):
        old_ch = dir_perm[new_ch]
        new_policy_3d[new_ch, dst_rows, dst_cols] = policy_3d[old_ch, src_rows, src_cols]

    # Non-direction channels 6-11: spatial only
    new_policy_3d[6:, dst_rows, dst_cols] = policy_3d[6:, src_rows, src_cols]

    return new_board, new_policy_3d.reshape(-1)
