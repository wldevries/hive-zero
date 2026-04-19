"""Tests for D6 hex symmetry augmentation for Zertz training."""

import numpy as np
import pytest

from zertz.nn.training import (
    _load_grid_perms,
    _load_hex_perms,
    ZertzDataset,
    _BOARD_SIZE,
)
from zertz.nn.model import NUM_CHANNELS, GRID_SIZE, POLICY_SIZE, RESERVE_SIZE

_GRID_PERMS = _load_grid_perms()
_HEX_PERMS = _load_hex_perms()
_GC = GRID_SIZE * GRID_SIZE  # 49


# ---------------------------------------------------------------------------
# Hex coordinate helpers (mirroring zertz_game::hex)
# ---------------------------------------------------------------------------

def _hex_to_index(q: int, r: int) -> int:
    """Compute hex_to_index(q, r) in Python, matching the Rust implementation."""
    RADIUS = 3
    idx = 0
    row = -RADIUS
    while row < r:
        q_min = max(-RADIUS, -RADIUS - row)
        q_max = min(RADIUS, RADIUS - row)
        idx += q_max - q_min + 1
        row += 1
    q_min = max(-RADIUS, -RADIUS - r)
    return idx + (q - q_min)


def _hex_to_grid_flat(q: int, r: int) -> int:
    """Compute flat grid index for hex (q, r), matching hex_to_grid."""
    RADIUS = 3
    grid_row = r + RADIUS
    q_min = max(-RADIUS, -RADIUS - r)
    grid_col = q - q_min
    return grid_row * GRID_SIZE + grid_col


# ---------------------------------------------------------------------------
# Grid permutation tests
# ---------------------------------------------------------------------------

class TestGridPermutations:
    def test_count(self):
        assert len(_GRID_PERMS) == 12

    def test_shape(self):
        for perm in _GRID_PERMS:
            assert perm.shape == (_GC,)

    def test_identity_valid_cells_fixed(self):
        """Identity perm: each valid cell maps to itself."""
        perm = _GRID_PERMS[0]
        # Enumerate all valid hex positions and check
        RADIUS = 3
        for r in range(-RADIUS, RADIUS + 1):
            q_min = max(-RADIUS, -RADIUS - r)
            q_max = min(RADIUS, RADIUS - r)
            for q in range(q_min, q_max + 1):
                flat = _hex_to_grid_flat(q, r)
                assert perm[flat] == flat, f"Identity moved ({q},{r}) flat={flat}"

    def test_identity_invalid_cells_sentinel(self):
        """Identity perm: invalid cells have sentinel value (49)."""
        perm = _GRID_PERMS[0]
        valid_flats = set()
        RADIUS = 3
        for r in range(-RADIUS, RADIUS + 1):
            q_min = max(-RADIUS, -RADIUS - r)
            q_max = min(RADIUS, RADIUS - r)
            for q in range(q_min, q_max + 1):
                valid_flats.add(_hex_to_grid_flat(q, r))
        for i in range(_GC):
            if i not in valid_flats:
                assert perm[i] == _GC, f"Invalid cell {i} should be sentinel"

    def test_exactly_37_valid_entries(self):
        """Each grid perm has exactly 37 non-sentinel values."""
        for i, perm in enumerate(_GRID_PERMS):
            valid = perm[perm < _GC]
            assert len(valid) == _BOARD_SIZE, f"Symmetry {i}: {len(valid)} valid entries"

    def test_bijective_on_valid_cells(self):
        """The 37 non-sentinel values in each perm are all distinct."""
        for i, perm in enumerate(_GRID_PERMS):
            valid = perm[perm < _GC]
            assert len(valid) == len(set(valid.tolist())), f"Symmetry {i} has duplicates"

    def test_r1_known_point(self):
        """R1 (60° CW): hex (0,-3) → (3,-3). grid (0,0) → (0,3)."""
        perm = _GRID_PERMS[1]
        src = _hex_to_grid_flat(0, -3)   # (0, 0) → flat 0
        dst = _hex_to_grid_flat(3, -3)   # (0, 3) → flat 3
        assert perm[dst] == src

    def test_r3_known_point(self):
        """R3 (180°): hex (1,0) → (-1,0). grid (3,4) → (3,2)."""
        perm = _GRID_PERMS[3]
        src = _hex_to_grid_flat(1, 0)    # row=3, col=4 → flat 25
        dst = _hex_to_grid_flat(-1, 0)   # row=3, col=2 → flat 23
        assert perm[dst] == src

    def test_invalid_cells_zero_after_augment(self):
        """After augmenting a board with any symmetry, invalid cells are still zero."""
        valid_flats = set()
        RADIUS = 3
        for r in range(-RADIUS, RADIUS + 1):
            q_min = max(-RADIUS, -RADIUS - r)
            q_max = min(RADIUS, RADIUS - r)
            for q in range(q_min, q_max + 1):
                valid_flats.add(_hex_to_grid_flat(q, r))

        board = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        # Place a marble at a valid cell
        board[0, 3, 3] = 1.0  # hex (0,0) center

        for sym_idx, perm in enumerate(_GRID_PERMS):
            bf = board.reshape(NUM_CHANNELS, _GC)
            padded = np.concatenate([bf, np.zeros((NUM_CHANNELS, 1), dtype=np.float32)], axis=1)
            result = padded[:, perm].reshape(NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
            rf = result.reshape(NUM_CHANNELS, _GC)
            for i in range(_GC):
                if i not in valid_flats:
                    assert np.all(rf[:, i] == 0.0), f"Sym {sym_idx}: invalid cell {i} nonzero"


# ---------------------------------------------------------------------------
# Hex permutation tests
# ---------------------------------------------------------------------------

class TestHexPermutations:
    def test_count(self):
        assert len(_HEX_PERMS) == 12

    def test_shape(self):
        for perm in _HEX_PERMS:
            assert perm.shape == (_BOARD_SIZE,)

    def test_identity(self):
        perm = _HEX_PERMS[0]
        for i in range(_BOARD_SIZE):
            assert perm[i] == i, f"Identity perm moved hex index {i}"

    def test_bijective(self):
        for i, perm in enumerate(_HEX_PERMS):
            assert len(set(perm.tolist())) == _BOARD_SIZE, f"Symmetry {i} not bijective"

    def test_values_in_range(self):
        for i, perm in enumerate(_HEX_PERMS):
            assert perm.min() >= 0 and perm.max() < _BOARD_SIZE, f"Symmetry {i} out of range"

    def test_r1_known_point(self):
        """R1 (60° CW): hex (0,-3) idx=0 → hex (3,-3) idx=3. hex_perm[3] = 0."""
        perm = _HEX_PERMS[1]
        src = _hex_to_index(0, -3)   # = 0
        dst = _hex_to_index(3, -3)   # = 3
        assert perm[dst] == src

    def test_r3_known_point(self):
        """R3 (180°): hex (1,0) → (-1,0). hex_perm[hex_to_index(-1,0)] = hex_to_index(1,0)."""
        perm = _HEX_PERMS[3]
        src = _hex_to_index(1, 0)
        dst = _hex_to_index(-1, 0)
        assert perm[dst] == src

    def test_compose_r1_r1_equals_r2(self):
        """Composing R1 gather perm with itself gives R2."""
        p1 = _HEX_PERMS[1]
        p2 = _HEX_PERMS[2]
        composed = p1[p1]  # apply p1 twice
        np.testing.assert_array_equal(composed, p2)


# ---------------------------------------------------------------------------
# Dataset augmentation tests
# ---------------------------------------------------------------------------

def _make_dataset() -> ZertzDataset:
    """Create a dataset with one position: white marble at hex (1,0), policy on Place move.

    490-format layout: [place_W(49), place_G(49), place_B(49), remove(49), cap_dirs(6*49)]
    Place(color=W, place_at=(1,0), remove=(2,-1)):
      - place_cp white: policy[0*49 + grid(1,0)] = policy[0*49 + 25] = policy[25]
      - place_rm:       policy[3*49 + grid(2,-1)] = policy[3*49 + 18] = policy[165]
    """
    ds = ZertzDataset(max_size=10)

    board = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    # White marble at hex (1,0): grid row=3, col=4
    board[0, 3, 4] = 1.0

    GS = GRID_SIZE * GRID_SIZE  # 49
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    place_at_grid = _hex_to_grid_flat(1, 0)   # row=3,col=4 → 25
    remove_grid   = _hex_to_grid_flat(2, -1)  # row=2,col=4 → 18
    policy[0 * GS + place_at_grid] = 0.8   # place_cp: white at (1,0)
    policy[3 * GS + remove_grid]   = 0.8   # place_rm: remove (2,-1)

    reserve = np.ones(RESERVE_SIZE, dtype=np.float32)
    ds.add_batch(
        board.reshape(1, -1),
        reserve.reshape(1, -1),
        policy.reshape(1, -1),
        np.array([1.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        [False],
    )
    return ds


class TestDatasetAugmentation:
    def test_no_augment_returns_copy(self):
        ds = _make_dataset()
        ds.augment_symmetry = False
        board, reserve, policy, *_ = ds[0]
        np.testing.assert_array_equal(board.numpy(), ds.board_tensors[0])
        np.testing.assert_array_equal(policy.numpy(), ds.policy_targets[0])

    def test_augment_preserves_shapes(self):
        ds = _make_dataset()
        ds.augment_symmetry = True
        board, reserve, policy, value, weight, vo, *_ = ds[0]
        assert board.shape == (NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        assert policy.shape == (POLICY_SIZE,)
        assert reserve.shape == (RESERVE_SIZE,)

    def test_augment_preserves_value(self):
        ds = _make_dataset()
        ds.augment_symmetry = True
        for _ in range(30):
            _, _, _, value, *_ = ds[0]
            assert value.item() == pytest.approx(1.0)

    def test_augment_preserves_board_sum(self):
        ds = _make_dataset()
        original_sum = float(ds.board_tensors[0].sum())
        ds.augment_symmetry = True
        for _ in range(50):
            board, *_ = ds[0]
            np.testing.assert_allclose(board.numpy().sum(), original_sum, rtol=1e-5)

    def test_augment_preserves_policy_sum(self):
        ds = _make_dataset()
        original_sum = float(ds.policy_targets[0].sum())
        ds.augment_symmetry = True
        for _ in range(50):
            _, _, policy, *_ = ds[0]
            np.testing.assert_allclose(policy.numpy().sum(), original_sum, rtol=1e-5)

    def test_augment_r3_board(self):
        """R3 (180°): white marble at hex (1,0) → hex (-1,0)."""
        ds = _make_dataset()
        board = ds.board_tensors[0]
        perm = _GRID_PERMS[3]

        bf = board.reshape(NUM_CHANNELS, _GC)
        padded = np.concatenate([bf, np.zeros((NUM_CHANNELS, 1), dtype=np.float32)], axis=1)
        result = padded[:, perm].reshape(NUM_CHANNELS, GRID_SIZE, GRID_SIZE)

        # Original marble at grid (3,4) should now be at grid (3,2)
        assert result[0, 3, 4] == pytest.approx(0.0)  # old position empty
        assert result[0, 3, 2] == pytest.approx(1.0)  # new position occupied

    def test_augment_r3_policy(self):
        """R3 (180°): Place white at (1,0) remove (2,-1) → place at (-1,0) remove (-2,1).

        490-format: check that place_cp and place_rm cells move correctly under R3.
        """
        ds = _make_dataset()
        GS = GRID_SIZE * GRID_SIZE  # 49
        grid_perm = _GRID_PERMS[3]

        policy = ds.policy_targets[0]
        p = policy.reshape(10, GS)
        padded = np.concatenate([p, np.zeros((10, 1), dtype=np.float32)], axis=1)
        p_new = padded[:, grid_perm]  # (10, 49)

        # R3: hex(1,0) → hex(-1,0), hex(2,-1) → hex(-2,1)
        old_place_grid = _hex_to_grid_flat(1, 0)    # 25
        new_place_grid = _hex_to_grid_flat(-1, 0)   # 23
        old_remove_grid = _hex_to_grid_flat(2, -1)  # 18
        new_remove_grid = _hex_to_grid_flat(-2, 1)  # 29

        # place_cp (channel 0 = white): value 0.8 should move from old to new grid cell
        assert p_new[0, new_place_grid]  == pytest.approx(0.8)
        assert p_new[0, old_place_grid]  == pytest.approx(0.0)
        # place_rm (channel 3): value 0.8 should move from old to new grid cell
        assert p_new[3, new_remove_grid] == pytest.approx(0.8)
        assert p_new[3, old_remove_grid] == pytest.approx(0.0)
