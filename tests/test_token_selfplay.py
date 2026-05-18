"""End-to-end smoke test for token self-play.

Plays one short game with an untrained transformer, writes per-ply samples
to an in-memory `HiveDataset`, then runs one training step against that
buffer. If this passes, the full pipeline — tokenize → MCTS → callback →
training data → loss — works end to end.
"""
import numpy as np
import torch

from hive.nn.training import HiveDataset, Trainer, MAX_LEGAL_MOVES
from hive.nn.transformer import SEQ_LEN, F_FLAGS, create_model
from hive.selfplay.token_selfplay import (
    play_one_game, make_torch_eval_fn, _sample_move_index,
)


def test_sample_move_index_argmax_when_temperature_zero():
    import random
    probs = np.array([0.1, 0.7, 0.2])
    assert _sample_move_index(probs, 0.0, random.Random(0)) == 1


def test_sample_move_index_returns_valid_index_under_temperature():
    import random
    probs = np.array([0.1, 0.4, 0.3, 0.2])
    for seed in range(10):
        idx = _sample_move_index(probs, 1.0, random.Random(seed))
        assert 0 <= idx < probs.size


def test_play_one_game_writes_samples_to_buffer():
    """Untrained transformer, tiny game, low simulations: every legal-move
    ply should produce one buffer sample."""
    torch.manual_seed(0)
    model = create_model({"d_model": 64, "n_blocks": 2, "num_heads": 4})
    eval_fn = make_torch_eval_fn(model, device="cpu")

    ds = HiveDataset(max_size=128)
    result = play_one_game(
        eval_fn, ds,
        simulations=4,
        play_batch=2,
        max_moves=10,
        temperature_moves=4,
        grid_size=7,
        skip_timeout_data=False,  # we WILL hit max_moves; keep the data
        rng_seed=0,
    )

    assert result.move_count > 0, "game made no progress"
    assert result.samples_written > 0, "no samples written to buffer"
    assert ds.raw_size == result.samples_written

    # Spot-check a sample: (mover, dest) indices land inside the real
    # (non-pad) token range, and probs are normalized.
    cat = ds.tokens_categoricals[0]
    msk = ds.tokens_mask[0]
    msrc = ds.move_src[0]
    mdst = ds.move_dst[0]
    mprob = ds.move_probs[0]
    nmov = int(ds.num_moves[0])
    assert cat[0, 0] == 1, "slot 0 must be the [GAME] token"
    used = int(msk.sum())
    assert used > 1
    assert nmov > 0
    for k in range(nmov):
        assert 0 <= int(msrc[k]) < used
        assert 0 <= int(mdst[k]) < used
    s = float(mprob[:nmov].sum())
    assert abs(s - 1.0) < 1e-3, f"move_probs sum to {s} not 1"


def test_play_then_train_step_runs():
    """After one short game, a single training batch must not blow up."""
    torch.manual_seed(0)
    model = create_model({"d_model": 64, "n_blocks": 2, "num_heads": 4})
    eval_fn = make_torch_eval_fn(model, device="cpu")

    ds = HiveDataset(max_size=128)
    play_one_game(
        eval_fn, ds,
        simulations=4, play_batch=2, max_moves=8,
        grid_size=7, skip_timeout_data=False, rng_seed=1,
    )
    assert ds.raw_size > 0

    trainer = Trainer(model=model, device="cpu", lr=0.01)
    stats = trainer.train_epoch(ds, batch_size=2)
    assert "policy_loss" in stats
    assert "value_loss" in stats
    assert np.isfinite(stats["total_loss"])
