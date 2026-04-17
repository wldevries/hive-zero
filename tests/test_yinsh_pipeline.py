"""Smoke tests for the Yinsh AlphaZero pipeline."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def engine_zero():
    return pytest.importorskip("engine_zero")


# ---------------------------------------------------------------------------
# Symmetry permutation tables
# ---------------------------------------------------------------------------


def test_yinsh_d6_grid_perms_shape(engine_zero):
    perms = engine_zero.yinsh_d6_grid_permutations()
    assert len(perms) == 12
    expected_n = engine_zero.YINSH_GRID_SIZE * engine_zero.YINSH_GRID_SIZE
    for p in perms:
        arr = np.asarray(p)
        assert arr.shape == (expected_n,)


def test_yinsh_d6_dir_perms_shape(engine_zero):
    perms = engine_zero.yinsh_d6_dir_permutations()
    assert len(perms) == 12
    for p in perms:
        arr = np.asarray(p)
        assert arr.shape == (3,)
        # Each row-dir perm must be a permutation of {0,1,2}.
        assert sorted(arr.tolist()) == [0, 1, 2]


def test_yinsh_d6_grid_perm_identity_is_identity(engine_zero):
    perm = np.asarray(engine_zero.yinsh_d6_grid_permutations()[0])
    n = engine_zero.YINSH_GRID_SIZE * engine_zero.YINSH_GRID_SIZE
    # Identity sym (index 0): every valid cell maps to itself, invalid stays at sentinel n.
    valid_count = int((perm < n).sum())
    assert valid_count == engine_zero.YINSH_BOARD_SIZE
    # Confirm fixed-point property on valid cells.
    for i in range(n):
        v = int(perm[i])
        if v < n:
            assert v == i


def test_yinsh_valid_d6_indices_includes_identity(engine_zero):
    valid = list(engine_zero.yinsh_valid_d6_indices())
    assert 0 in valid
    assert all(0 <= i < 12 for i in valid)


def test_yinsh_d6_valid_perms_have_full_board(engine_zero):
    """Every Yinsh-valid D6 sym maps all 85 cells to valid positions."""
    valid = list(engine_zero.yinsh_valid_d6_indices())
    perms = engine_zero.yinsh_d6_grid_permutations()
    n = engine_zero.YINSH_GRID_SIZE * engine_zero.YINSH_GRID_SIZE
    for sym_idx in valid:
        arr = np.asarray(perms[sym_idx])
        valid_count = int((arr < n).sum())
        assert valid_count == engine_zero.YINSH_BOARD_SIZE, (
            f"sym {sym_idx} maps only {valid_count}/{engine_zero.YINSH_BOARD_SIZE} cells"
        )


# ---------------------------------------------------------------------------
# Dataset augmentation invariants
# ---------------------------------------------------------------------------


def test_dataset_augmentation_preserves_sums(engine_zero):
    """For every Yinsh-valid D6 sym, augmenting a sample preserves board and
    policy sums when the policy is only nonzero on legal moves (which always
    sit on valid board cells).
    """
    from yinsh.nn.model import GRID_SIZE, NUM_CHANNELS
    from yinsh.nn.training import _apply_symmetry

    valid_syms = list(engine_zero.yinsh_valid_d6_indices())

    # Build a non-trivial game state — past Setup so we exercise multiple phases
    # and several legal moves on valid cells.
    game = engine_zero.YinshGame()
    while game.phase() == "setup":
        moves = game.valid_moves()
        game.play(moves[0])
    board_2d, _ = game.encode()
    board = np.asarray(board_2d).reshape(NUM_CHANNELS, GRID_SIZE, GRID_SIZE).astype(np.float32)

    # Construct a policy uniform over the real legal moves, using the same
    # PolicyIndex::Sum convention the Rust selfplay loop uses for MoveRing.
    # We do this by running one step of best_move with a constant eval and
    # reading back the visit-distribution via training_data: simpler — just
    # build a valid-only policy by replaying the legal-move encoding through
    # the move_encoding module via a tiny self-play with zero search.
    # Easier still: just allocate uniform mass on legal-move policy indices
    # using the C-side YinshSelfPlaySession output.
    import torch

    from yinsh.nn.model import create_model

    model = create_model(num_blocks=2, channels=8).eval()

    def eval_fn(b_np, r_np):
        b = torch.from_numpy(np.asarray(b_np)).to(dtype=torch.float32)
        r = torch.from_numpy(np.asarray(r_np)).to(dtype=torch.float32)
        with torch.no_grad():
            policy, value = model(b, r)
        return policy.cpu().numpy(), value.cpu().numpy().squeeze(1)

    session = engine_zero.YinshSelfPlaySession(
        num_games=1, simulations=4, max_moves=20, c_puct=1.5, play_batch_size=1,
    )
    result = session.play_games(eval_fn)
    boards, _, policies, _, _, _ = result.training_data()
    boards = np.asarray(boards)
    policies = np.asarray(policies)
    assert boards.shape[0] > 0

    sample_board = boards[0].reshape(NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
    sample_policy = policies[0]
    base_board_sum = float(sample_board.sum())
    base_policy_sum = float(sample_policy.sum())

    for sym in valid_syms:
        b2, p2 = _apply_symmetry(sample_board.copy(), sample_policy.copy(), sym)
        assert b2.sum() == pytest.approx(base_board_sum, rel=1e-5)
        assert p2.sum() == pytest.approx(base_policy_sum, rel=1e-5), (
            f"sym {sym}: policy sum changed {base_policy_sum} -> {p2.sum()}"
        )


def test_dataset_augmentation_identity_is_noop(engine_zero):
    from yinsh.nn.model import GRID_SIZE, NUM_CHANNELS, POLICY_SIZE
    from yinsh.nn.training import _apply_symmetry

    rng = np.random.default_rng(1)
    board = rng.random((NUM_CHANNELS, GRID_SIZE, GRID_SIZE)).astype(np.float32)
    policy = rng.random(POLICY_SIZE).astype(np.float32)
    b2, p2 = _apply_symmetry(board, policy, 0)
    np.testing.assert_array_equal(b2, board)
    np.testing.assert_array_equal(p2, policy)


# ---------------------------------------------------------------------------
# End-to-end smoke: tiny self-play + train_epoch
# ---------------------------------------------------------------------------


def test_smoke_selfplay_then_train(engine_zero):
    torch = pytest.importorskip("torch")

    from yinsh.nn.model import (
        GRID_SIZE,
        NUM_CHANNELS,
        POLICY_SIZE,
        RESERVE_SIZE,
        YinshNet,
        create_model,
    )
    from yinsh.nn.training import Trainer, YinshDataset

    device = "cpu"
    model = create_model(num_blocks=2, channels=16).to(device)
    model.eval()

    def eval_fn(board_np, reserve_np):
        b = torch.from_numpy(np.asarray(board_np)).to(dtype=torch.float32)
        r = torch.from_numpy(np.asarray(reserve_np)).to(dtype=torch.float32)
        with torch.no_grad():
            policy, value = model(b, r)
        return policy.cpu().numpy(), value.cpu().numpy().squeeze(1)

    session = engine_zero.YinshSelfPlaySession(
        num_games=2,
        simulations=8,
        max_moves=80,
        temperature=1.0,
        temp_threshold=4,
        c_puct=1.5,
        play_batch_size=2,
    )
    result = session.play_games(eval_fn)
    assert result.num_samples > 0

    boards, reserves, policies, values, value_only, phase_flags = result.training_data()
    boards = np.asarray(boards)
    assert boards.shape[1] == NUM_CHANNELS * GRID_SIZE * GRID_SIZE
    assert np.asarray(reserves).shape[1] == RESERVE_SIZE
    assert np.asarray(policies).shape[1] == POLICY_SIZE

    dataset = YinshDataset(max_size=max(64, result.num_samples * 2))
    dataset.add_batch(
        board_tensors=boards,
        reserve_vectors=np.asarray(reserves),
        policy_targets=np.asarray(policies),
        value_targets=np.asarray(values),
        value_only=list(value_only),
        phase_flags=list(phase_flags),
    )
    assert len(dataset) == result.num_samples

    trainer = Trainer(model=model, device=device, lr=0.01)
    losses = trainer.train_epoch(dataset, batch_size=8)
    assert "policy_loss" in losses
    assert "value_loss" in losses
    assert losses["total_loss"] >= 0.0


def test_smoke_battle(engine_zero):
    torch = pytest.importorskip("torch")

    from yinsh.nn.model import create_model

    device = "cpu"
    m1 = create_model(num_blocks=2, channels=16).to(device).eval()
    m2 = create_model(num_blocks=2, channels=16).to(device).eval()

    def make_eval(model):
        def fn(board_np, reserve_np):
            b = torch.from_numpy(np.asarray(board_np)).to(dtype=torch.float32)
            r = torch.from_numpy(np.asarray(reserve_np)).to(dtype=torch.float32)
            with torch.no_grad():
                policy, value = model(b, r)
            return policy.cpu().numpy(), value.cpu().numpy().squeeze(1)

        return fn

    session = engine_zero.YinshSelfPlaySession(
        num_games=2,
        simulations=4,
        max_moves=60,
        c_puct=1.5,
        play_batch_size=1,
    )
    result = session.play_battle(make_eval(m1), make_eval(m2))
    total = result.wins_model1 + result.wins_model2 + result.draws
    assert total == 2
