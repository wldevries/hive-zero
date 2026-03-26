"""Tests for D6 hex symmetry augmentation in training."""

import numpy as np
import pytest

from hive.nn.training import (
    _load_sym_perms,
    HiveDataset,
)
from hive.encoding.board_encoder import NUM_CHANNELS, GRID_SIZE, RESERVE_SIZE
from hive.encoding.move_encoder import POLICY_SIZE, NUM_POLICY_CHANNELS

_SYM_PERMS = _load_sym_perms(GRID_SIZE)
_GRID_CELLS = GRID_SIZE * GRID_SIZE
_NUM_POLICY_CH = NUM_POLICY_CHANNELS

CENTER = GRID_SIZE // 2


def cell(row, col):
    """Grid (row, col) to flat cell index."""
    return row * GRID_SIZE + col


def hex_to_cell(q, r):
    """Axial hex to flat cell index."""
    return cell(r + CENTER, q + CENTER)


# --- Permutation tables ---

class TestPermutations:
    def test_identity_perm_is_identity(self):
        perm = _SYM_PERMS[0]
        for i in range(_GRID_CELLS):
            assert perm[i] == i

    def test_all_perms_injective(self):
        """No two output cells map to the same input cell."""
        for i, perm in enumerate(_SYM_PERMS):
            in_bounds = perm[perm < _GRID_CELLS]
            assert len(in_bounds) == len(set(in_bounds)), f"Symmetry {i} has duplicates"

    def test_center_fixed_by_all(self):
        """All symmetries fix the center cell."""
        center_cell = cell(CENTER, CENTER)
        for i, perm in enumerate(_SYM_PERMS):
            assert perm[center_cell] == center_cell, f"Symmetry {i} moves center"

    def test_r1_maps_known_point(self):
        """R1 (60 deg): hex (1,0) -> hex (0,1), i.e. grid (11,12) -> (12,11)."""
        src = hex_to_cell(1, 0)  # grid (11, 12)
        dst = hex_to_cell(0, 1)  # grid (12, 11)
        assert _SYM_PERMS[1][dst] == src

    def test_r3_maps_known_point(self):
        """R3 (180 deg): hex (2, 3) -> hex (-2, -3), grid (14,13) -> (8,9)."""
        src = hex_to_cell(2, 3)
        dst = hex_to_cell(-2, -3)
        assert _SYM_PERMS[3][dst] == src

    def test_compose_perm_r1_r1_equals_r2(self):
        """Composing R1 gather perm with itself equals R2 perm."""
        p1 = _SYM_PERMS[1]
        p2 = _SYM_PERMS[2]
        for out_cell in range(_GRID_CELLS):
            mid = p1[out_cell]
            composed = p1[mid] if mid < _GRID_CELLS else _GRID_CELLS
            expected = p2[out_cell]
            # Both OOB is OK
            if composed >= _GRID_CELLS and expected >= _GRID_CELLS:
                continue
            assert composed == expected, f"Cell {out_cell}: composed={composed}, R2={expected}"

    def test_perm_preserves_nonzero_count(self):
        """Applying any permutation to a board preserves the number of nonzero cells."""
        board = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        # Place some pieces near center (will stay in-bounds under rotation)
        board[0, 11, 11] = 1.0  # queen at center
        board[1, 11, 12] = 1.0  # spider adjacent
        board[11, 12, 11] = 1.0  # opponent queen
        board[38, 11, 11] = 2.0 / 7  # stack height
        original_nonzero = np.count_nonzero(board)

        for i, perm in enumerate(_SYM_PERMS):
            bf = board.reshape(NUM_CHANNELS, _GRID_CELLS)
            padded = np.concatenate([bf, np.zeros((NUM_CHANNELS, 1), dtype=np.float32)], axis=1)
            rotated = padded[:, perm].reshape(NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
            assert np.count_nonzero(rotated) == original_nonzero, f"Symmetry {i} changed nonzero count"


# --- Dataset augmentation ---

class TestDatasetAugmentation:
    def _make_dataset(self):
        ds = HiveDataset(max_size=10)
        board = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        board[0, CENTER, CENTER] = 1.0  # queen at center
        board[1, CENTER, CENTER + 1] = 1.0  # spider right of center
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        policy[0 * _GRID_CELLS + cell(CENTER, CENTER + 2)] = 0.7  # queen to (11,13)
        policy[1 * _GRID_CELLS + cell(CENTER, CENTER + 1)] = 0.3  # spider at (11,12)
        ds.add_sample(board, np.ones(RESERVE_SIZE, dtype=np.float32), policy, 1.0)
        return ds

    def test_no_augment_returns_copy(self):
        ds = self._make_dataset()
        ds.augment_symmetry = False
        b, _, p, *_ = ds[0]
        np.testing.assert_array_equal(b.numpy(), ds.board_tensors[0])
        np.testing.assert_array_equal(p.numpy(), ds.policy_targets[0])

    def test_augment_preserves_shapes(self):
        ds = self._make_dataset()
        ds.augment_symmetry = True
        b, rv, p, v, w, vo, po, aux = ds[0]
        assert b.shape == (NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        assert p.shape == (POLICY_SIZE,)
        assert rv.shape == (RESERVE_SIZE,)

    def test_augment_preserves_value(self):
        ds = self._make_dataset()
        ds.augment_symmetry = True
        _, _, _, v, *_ = ds[0]
        assert v.item() == 1.0

    def test_augment_preserves_board_sum(self):
        """Sum of all board values should be preserved (rotation is a permutation)."""
        ds = self._make_dataset()
        original_sum = ds.board_tensors[0].sum()
        ds.augment_symmetry = True
        # Try many times to hit non-identity symmetries
        for _ in range(50):
            b, *_ = ds[0]
            np.testing.assert_allclose(b.numpy().sum(), original_sum, rtol=1e-5)

    def test_augment_preserves_policy_sum(self):
        """Sum of policy target should be preserved."""
        ds = self._make_dataset()
        original_sum = ds.policy_targets[0].sum()
        ds.augment_symmetry = True
        for _ in range(50):
            _, _, p, *_ = ds[0]
            np.testing.assert_allclose(p.numpy().sum(), original_sum, rtol=1e-5)

    def test_augment_r3_flips_board(self):
        """Manually verify 180-degree rotation on a simple board."""
        ds = self._make_dataset()
        board = ds.board_tensors[0]
        perm = _SYM_PERMS[3]  # 180 degrees

        bf = board.reshape(NUM_CHANNELS, _GRID_CELLS)
        padded = np.concatenate([bf, np.zeros((NUM_CHANNELS, 1), dtype=np.float32)], axis=1)
        rotated = padded[:, perm].reshape(NUM_CHANNELS, GRID_SIZE, GRID_SIZE)

        # Queen was at (11,11) -> stays at (11,11) under 180
        assert rotated[0, CENTER, CENTER] == 1.0
        # Spider was at (11,12) i.e. hex(1,0) -> hex(-1,0) -> grid(11,10)
        assert rotated[1, CENTER, CENTER - 1] == 1.0
        # Original spider position should now be empty
        assert rotated[1, CENTER, CENTER + 1] == 0.0

    def test_augment_r3_flips_policy(self):
        """180-degree rotation moves policy targets to mirrored cells."""
        ds = self._make_dataset()
        policy = ds.policy_targets[0]
        perm = _SYM_PERMS[3]

        pf = policy.reshape(_NUM_POLICY_CH, _GRID_CELLS)
        padded = np.concatenate([pf, np.zeros((_NUM_POLICY_CH, 1), dtype=np.float32)], axis=1)
        rotated = padded[:, perm].reshape(-1)

        # Queen move was to (11,13) hex(2,0) -> hex(-2,0) -> grid(11,9)
        assert rotated[0 * _GRID_CELLS + cell(CENTER, CENTER - 2)] == pytest.approx(0.7)
        # Spider was at (11,12) hex(1,0) -> hex(-1,0) -> grid(11,10)
        assert rotated[1 * _GRID_CELLS + cell(CENTER, CENTER - 1)] == pytest.approx(0.3)
