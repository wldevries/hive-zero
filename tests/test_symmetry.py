"""Tests for hex symmetry augmentation."""

import numpy as np
import pytest
from hive.core.hex import Hex, DIRECTIONS
from hive.encoding.board_encoder import GRID_SIZE, GRID_CENTER
from hive.encoding.symmetry import (
    _hex_rotate60, _hex_mirror, apply_symmetry, _TRANSFORMS,
)


class TestHexRotation:
    """Verify the 60° rotation maps each direction to the next."""

    def test_rotation_maps_directions(self):
        for i, d in enumerate(DIRECTIONS):
            rq, rr = _hex_rotate60(d.q, d.r)
            expected = DIRECTIONS[(i + 1) % 6]
            assert (rq, rr) == (expected.q, expected.r), (
                f"Dir {i} ({d.q},{d.r}) -> ({rq},{rr}), "
                f"expected ({expected.q},{expected.r})"
            )

    def test_six_rotations_return_to_origin(self):
        q, r = 3, -2
        for _ in range(6):
            q, r = _hex_rotate60(q, r)
        assert (q, r) == (3, -2)

    def test_rotation_preserves_origin(self):
        assert _hex_rotate60(0, 0) == (0, 0)


class TestHexMirror:
    """Verify the mirror transform."""

    MIRROR_EXPECTED = [5, 4, 3, 2, 1, 0]  # E↔SE, NE↔SW, NW↔W

    def test_mirror_maps_directions(self):
        for i, d in enumerate(DIRECTIONS):
            mq, mr = _hex_mirror(d.q, d.r)
            expected = DIRECTIONS[self.MIRROR_EXPECTED[i]]
            assert (mq, mr) == (expected.q, expected.r)

    def test_double_mirror_is_identity(self):
        q, r = 5, -3
        mq, mr = _hex_mirror(*_hex_mirror(q, r))
        assert (mq, mr) == (q, r)


class TestTransformTable:
    """Verify the precomputed transform table."""

    def test_twelve_transforms(self):
        assert len(_TRANSFORMS) == 12

    def test_identity_is_first(self):
        src_r, src_c, dst_r, dst_c, dir_perm = _TRANSFORMS[0]
        assert np.array_equal(src_r, dst_r)
        assert np.array_equal(src_c, dst_c)
        assert dir_perm == [0, 1, 2, 3, 4, 5]


class TestApplySymmetry:
    """End-to-end tests for apply_symmetry."""

    def _make_data(self):
        board = np.zeros((23, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        policy = np.zeros(12 * GRID_SIZE * GRID_SIZE, dtype=np.float32)
        return board, policy

    def test_identity_returns_same(self):
        board, policy = self._make_data()
        board[0, 11, 12] = 1.0
        policy[42] = 1.0
        nb, np_ = apply_symmetry(board, policy, 0)
        assert nb is board  # identity returns same object
        assert np_ is policy

    def test_rotation_moves_piece(self):
        """A piece at E neighbor (hex 1,0) should rotate to NE (hex 1,-1)."""
        board, policy = self._make_data()
        # Place piece at hex (1,0) = grid (11,12)
        board[0, 11, 12] = 1.0
        nb, _ = apply_symmetry(board, policy, 1)  # 1 rotation

        # hex (1,0) -> rotated (1,-1) -> grid (10,12)
        assert nb[0, 10, 12] == 1.0
        assert nb[0, 11, 12] == 0.0

    def test_rotation_permutes_direction_channels(self):
        """Direction channel 0 (from E) at center should become channel 1 (from NE)."""
        board, policy = self._make_data()
        gs2 = GRID_SIZE * GRID_SIZE
        # Set direction 0 at center (11,11)
        policy[0 * gs2 + 11 * GRID_SIZE + 11] = 1.0
        _, np_ = apply_symmetry(board, policy, 1)

        # Should now be direction 1 at center
        assert np_[1 * gs2 + 11 * GRID_SIZE + 11] == 1.0
        assert np_[0 * gs2 + 11 * GRID_SIZE + 11] == 0.0

    def test_placement_channels_not_permuted(self):
        """Placement channels 7-11 should only get spatial transform, no channel swap."""
        board, policy = self._make_data()
        gs2 = GRID_SIZE * GRID_SIZE
        # Place a queen placement (channel 7) at hex (1,0) = grid (11,12)
        policy[7 * gs2 + 11 * GRID_SIZE + 12] = 1.0
        _, np_ = apply_symmetry(board, policy, 1)

        # Should still be channel 7, but at rotated position (10,12)
        assert np_[7 * gs2 + 10 * GRID_SIZE + 12] == 1.0

    def test_all_transforms_preserve_mass_near_center(self):
        """Policy mass near center is preserved (corners may clip off grid)."""
        board = np.zeros((23, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        policy = np.zeros(12 * GRID_SIZE * GRID_SIZE, dtype=np.float32)
        gs2 = GRID_SIZE * GRID_SIZE
        # Place data only near center (within radius 5) so rotations stay in grid
        c = GRID_CENTER
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                r, co = c + dr, c + dc
                board[0, r, co] = np.random.randn()
                policy[7 * gs2 + r * GRID_SIZE + co] = abs(np.random.randn())
        original_policy_sum = policy.sum()
        original_board_sum = board.sum()

        for i in range(12):
            nb, np_ = apply_symmetry(board, policy, i)
            assert abs(np_.sum() - original_policy_sum) < 1e-3, f"Transform {i} changed policy mass"
            assert abs(nb.sum() - original_board_sum) < 1e-3, f"Transform {i} changed board mass"

    def test_six_rotations_return_to_original(self):
        """Applying 1 rotation six times should return to original."""
        board = np.zeros((23, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        board[0, 11, 12] = 1.0
        board[3, 13, 11] = 2.5
        policy = np.zeros(12 * GRID_SIZE * GRID_SIZE, dtype=np.float32)
        gs2 = GRID_SIZE * GRID_SIZE
        policy[2 * gs2 + 11 * GRID_SIZE + 12] = 1.0

        b, p = board.copy(), policy.copy()
        for _ in range(6):
            b, p = apply_symmetry(b, p, 1)

        np.testing.assert_allclose(b, board, atol=1e-6)
        np.testing.assert_allclose(p, policy, atol=1e-6)

    def test_double_mirror_is_identity(self):
        """Applying mirror twice should return to original."""
        board = np.zeros((23, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        board[0, 11, 12] = 1.0
        policy = np.zeros(12 * GRID_SIZE * GRID_SIZE, dtype=np.float32)
        gs2 = GRID_SIZE * GRID_SIZE
        policy[0 * gs2 + 11 * GRID_SIZE + 12] = 1.0

        # Mirror is index 6 (0 rotations, mirror=True)
        b, p = apply_symmetry(board, policy, 6)
        b, p = apply_symmetry(b, p, 6)

        np.testing.assert_allclose(b, board, atol=1e-6)
        np.testing.assert_allclose(p, policy, atol=1e-6)
