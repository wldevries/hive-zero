"""End-to-end test for the token transformer training pipeline.

Exercises:
  - HiveDataset in-memory ring buffer (add_sample / __getitem__ / __getitems__)
  - Trainer._compute_batch_losses on synthetic token batches
  - D6 token-position augmentation produces the expected (q, r) permutations
  - One train_epoch reduces the total loss when the same sample is repeated
    (model learns to overfit a tiny dataset, the standard smoke check)
"""
import numpy as np
import pytest
import torch

from hive.nn.transformer import (
    SEQ_LEN, F_FLAGS, BILINEAR_DIM,
    TOKEN_KIND_GAME, TOKEN_KIND_PIECE, TOKEN_KIND_RESERVE, TOKEN_KIND_CELL,
    create_model,
)
from hive.nn.training import (
    HiveDataset, Trainer, MAX_LEGAL_MOVES, _load_axial_transforms,
)


def _synthetic_sample(used_tokens: int = 14, n_legal_moves: int = 6):
    """Build one consistent training sample.

    Token layout: [GAME, PIECE, PIECE, PIECE, RESERVE, RESERVE, CELL, CELL, …].
    Legal moves are random (mover_slot, dest_slot) pairs from those slots,
    with a peaked probability distribution (one mode + uniform tail).
    """
    rng = np.random.default_rng(123)
    cat = np.zeros((SEQ_LEN, 5), dtype=np.uint8)
    pos = np.zeros((SEQ_LEN, 2), dtype=np.int8)
    flg = np.zeros((SEQ_LEN, F_FLAGS), dtype=np.float32)
    msk = np.zeros(SEQ_LEN, dtype=np.bool_)

    cat[0, 0] = TOKEN_KIND_GAME
    msk[0] = True

    # 3 PIECE tokens (slots 1-3)
    for i, (q, r) in enumerate([(0, 0), (1, 0), (0, 1)]):
        cat[1 + i, 0] = TOKEN_KIND_PIECE
        cat[1 + i, 1] = 1                  # piece_type Queen
        cat[1 + i, 2] = 1                  # COLOR_MINE
        cat[1 + i, 3] = 0                  # z (solo piece → bottom == top)
        pos[1 + i] = (q, r)
        msk[1 + i] = True

    # 2 RESERVE tokens (slots 4-5)
    for i, pt in enumerate([2, 5]):  # Spider, Ant
        cat[4 + i, 0] = TOKEN_KIND_RESERVE
        cat[4 + i, 1] = pt
        cat[4 + i, 2] = 1
        cat[4 + i, 4] = 2  # count
        msk[4 + i] = True

    # 8 CELL tokens (slots 6-13)
    cells = [(-1, 0), (-1, 1), (0, -1), (1, -1), (2, 0), (2, -1), (-2, 1), (-2, 2)]
    for i, (q, r) in enumerate(cells):
        cat[6 + i, 0] = TOKEN_KIND_CELL
        pos[6 + i] = (q, r)
        msk[6 + i] = True

    assert used_tokens == 14, "test sample assumes 14 used tokens"

    # Choose legal moves (mover ∈ {1..5}, dest ∈ {6..13}).
    mover_pool = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    dest_pool  = np.arange(6, 14, dtype=np.uint8)
    src = rng.choice(mover_pool, n_legal_moves, replace=True)
    dst = rng.choice(dest_pool,  n_legal_moves, replace=True)
    # peaked distribution: first move gets 0.5 mass, rest uniform on the tail
    probs = np.full(n_legal_moves, 0.5 / (n_legal_moves - 1), dtype=np.float32)
    probs[0] = 0.5

    return {
        "tokens_categoricals": cat,
        "tokens_positions": pos,
        "tokens_flags": flg,
        "tokens_mask": msk,
        "move_src": src,
        "move_dst": dst,
        "move_probs": probs,
        "value_target": 0.2,
        "value_only": False,
        "policy_only": False,
        "aux_targets": np.array([0.1, 0.0, 0.5, 0.4, 0.3, 0.3], dtype=np.float32),
    }


def test_dataset_add_and_read_roundtrip():
    ds = HiveDataset(max_size=8)
    sample = _synthetic_sample()
    ds.add_sample(**sample)
    ds.add_sample(**sample)

    assert ds.raw_size == 2
    assert len(ds) == 2  # augment_symmetry off

    (cat, pos, flg, msk, msrc, mdst, mprob, nmov,
     vtar, rqtar, vo, po, aux, sym) = ds[0]
    assert cat.shape == (SEQ_LEN, 5)
    assert pos.shape == (SEQ_LEN, 2)
    assert flg.shape == (SEQ_LEN, F_FLAGS)
    assert msk.shape == (SEQ_LEN,)
    assert msrc.shape == (MAX_LEGAL_MOVES,)
    assert int(nmov.item()) == sample["move_probs"].shape[0]
    assert float(vtar.item()) == pytest.approx(0.2)
    assert int(sym.item()) == 0
    assert torch.allclose(aux, torch.from_numpy(sample["aux_targets"]))


def test_d6_augmentation_permutes_token_positions():
    """sym_idx ∈ [0, 12) must produce 12 distinct position permutations of
    the same sample, each matching one of the 2×2 axial transforms exposed
    by `d6_axial_transforms` in Rust."""
    ds = HiveDataset(max_size=8)
    sample = _synthetic_sample()
    ds.add_sample(**sample)
    ds.augment_symmetry = True

    assert len(ds) == 12

    trainer = Trainer(device="cpu", lr=0.01)
    # Run augment for each sym_idx in turn and verify positions transform.
    mats = _load_axial_transforms()
    base_pos_np = sample["tokens_positions"].astype(np.int64)

    for s in range(12):
        (_, pos, _, _, _, _, _, _, _, _, _, _, _, sym_idx) = ds[s]
        assert int(sym_idx.item()) == s
        pos_batch = pos.unsqueeze(0).to(torch.int64)         # (1, L, 2)
        sym_batch = sym_idx.unsqueeze(0).to(torch.int64)     # (1,)
        out = trainer._apply_sym_gpu(pos_batch, sym_batch).cpu().numpy()[0]  # (L, 2)

        # Recompute expected: new = mat @ (q, r)
        mat = mats[s]
        expected = (base_pos_np @ mat.T).astype(np.int64)
        # Only check positional tokens (kind ∈ {PIECE, CELL}); non-positional
        # slots have (0, 0) and stay (0, 0) under any linear map.
        assert np.array_equal(out, expected), \
            f"sym_idx={s}: positions disagree with axial transform matrix"


def test_train_epoch_overfits_repeated_sample():
    """Classic smoke check: train on N copies of the same sample for a few
    epochs and total loss should drop materially."""
    torch.manual_seed(0)
    np.random.seed(0)
    sample = _synthetic_sample()
    ds = HiveDataset(max_size=32)
    for _ in range(16):
        ds.add_sample(**sample)

    model = create_model({"d_model": 64, "n_blocks": 2, "num_heads": 4})
    trainer = Trainer(model=model, device="cpu", lr=0.05, grad_clip=1.0)

    loss_history = []
    for _ in range(8):
        stats = trainer.train_epoch(ds, batch_size=4)
        loss_history.append(stats["total_loss"])

    assert loss_history[-1] < loss_history[0] - 0.05, (
        f"loss did not drop materially: history = {loss_history}"
    )


def test_validate_returns_top1_acc_and_uniform_ce():
    """After overfitting one sample, top1_acc should be 1.0 — the model
    learns to put argmax on the same move the target peaks at."""
    torch.manual_seed(0)
    np.random.seed(0)
    sample = _synthetic_sample()
    ds = HiveDataset(max_size=32)
    for _ in range(8):
        ds.add_sample(**sample)

    model = create_model({"d_model": 64, "n_blocks": 2, "num_heads": 4})
    trainer = Trainer(model=model, device="cpu", lr=0.1, grad_clip=1.0)
    for _ in range(20):
        trainer.train_epoch(ds, batch_size=4)
    stats = trainer.validate(ds, batch_size=4)
    assert "top1_acc" in stats
    assert "uniform_ce" in stats
    assert stats["top1_acc"] >= 0.99, f"top1_acc only {stats['top1_acc']:.2f}"


def test_value_only_mask_excludes_sample_from_policy_loss():
    """value_only=True samples must not contribute to policy loss — verify
    by setting all samples value_only and checking policy_loss == 0."""
    sample = _synthetic_sample()
    sample["value_only"] = True
    ds = HiveDataset(max_size=8)
    ds.add_sample(**sample)
    ds.add_sample(**sample)

    trainer = Trainer(device="cpu", lr=0.01)
    # value_only=True for every sample → policy_weight is 0 everywhere →
    # the divisor is clamped to 1 but numerator is 0 → policy_loss = 0.
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    losses = trainer._compute_batch_losses(batch, 1.0, 1.0)
    assert losses["policy"].item() == 0.0
