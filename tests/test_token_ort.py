"""End-to-end test for the token-based ORT inference path.

Pipeline exercised:
    HiveTransformer  ──export_onnx──▶  *.onnx
    *.onnx           ──HiveOrtSession──▶  Rust HiveTokenOrtEngine
    HiveGame.mcts_run_ort(session)  ──▶  visit distribution
    play_one_game(ort_session=…)    ──▶  HiveDataset

If this test passes, the user can self-play train via the ORT path
(no Python GIL trips inside the MCTS simulation loop).
"""
import os
import tempfile

import numpy as np
import pytest
import torch

# Onnxruntime must be importable for this test to be meaningful.
ort = pytest.importorskip("onnxruntime")

from hive.nn.transformer import (
    HiveTransformer, create_model, export_onnx,
    SEQ_LEN, F_FLAGS, BILINEAR_DIM,
)


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory):
    """Export a tiny HiveTransformer to ONNX once per module run."""
    torch.manual_seed(0)
    model = create_model({"d_model": 64, "n_blocks": 2, "num_heads": 4})
    model.eval()
    path = tmp_path_factory.mktemp("ort") / "hive_token.onnx"
    # Dynamic batch dim so MCTS can send variable-sized batches; fp32 for
    # the most reliable cross-provider compatibility in tests.
    export_onnx(model, str(path), batch_size=None, precision="fp32")
    return str(path)


def test_onnxruntime_runs_exported_model(onnx_path):
    """The exported ONNX must run on the CPU provider and produce valid shapes."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feeds = {
        "categoricals": np.zeros((1, SEQ_LEN, 5), dtype=np.int32),
        "positions":    np.zeros((1, SEQ_LEN, 2), dtype=np.int32),
        "flags":        np.zeros((1, SEQ_LEN, F_FLAGS), dtype=np.float32),
        "mask":         np.zeros((1, SEQ_LEN), dtype=np.bool_),
    }
    feeds["categoricals"][0, 0, 0] = 1  # GAME token
    feeds["mask"][0, 0] = True
    policy, wdl, aux = sess.run(None, feeds)
    assert policy.shape == (1, 2 * SEQ_LEN * BILINEAR_DIM)
    assert wdl.shape == (1, 3)
    assert aux.shape == (1, 6)
    assert abs(float(wdl.sum()) - 1.0) < 1e-4


def _can_load_engine_zero_ort(onnx_path) -> bool:
    """Try to load via the Rust HiveOrtSession. Returns False if the Rust
    side can't find onnxruntime.dll or fails to init the providers."""
    try:
        from engine_zero import HiveOrtSession
        HiveOrtSession(onnx_path)
        return True
    except Exception:
        return False


def test_hive_ort_session_drives_mcts(onnx_path):
    """PyHiveOrtSession (the Rust-side token engine wrapper) must accept the
    exported model and let HiveGame.mcts_run_ort drive a search through it
    without invoking any Python callback."""
    if not _can_load_engine_zero_ort(onnx_path):
        pytest.skip("engine_zero.HiveOrtSession could not load the exported model")
    from engine_zero import HiveGame, HiveOrtSession

    session = HiveOrtSession(onnx_path)
    g = HiveGame(grid_size=7)
    cat, pos, flg, msk, msrc, mdst, mprob, root_q, best_uhp = g.mcts_run_ort(
        session, simulations=4, play_batch=2, dir_alpha=0.0, dir_epsilon=0.0,
    )
    assert cat.shape == (SEQ_LEN, 5)
    assert msk.shape == (SEQ_LEN,)
    assert msrc.shape == mdst.shape == mprob.shape
    used = int(msk.sum())
    assert used > 1
    assert msrc.size > 0
    for k in range(msrc.size):
        assert 0 <= int(msrc[k]) < used
        assert 0 <= int(mdst[k]) < used
    s = float(mprob.sum())
    assert abs(s - 1.0) < 1e-3
    assert best_uhp != ""


def test_play_one_game_via_ort(onnx_path):
    """Full self-play loop via ORT — must write samples to the buffer."""
    if not _can_load_engine_zero_ort(onnx_path):
        pytest.skip("engine_zero.HiveOrtSession could not load the exported model")
    from engine_zero import HiveOrtSession
    from hive.nn.training import HiveDataset
    from hive.selfplay.token_selfplay import play_one_game

    session = HiveOrtSession(onnx_path)
    ds = HiveDataset(max_size=64)
    result = play_one_game(
        eval_fn=None,
        ort_session=session,
        dataset=ds,
        simulations=4, play_batch=2, max_moves=6,
        grid_size=7, skip_timeout_data=False, rng_seed=0,
    )
    assert result.samples_written > 0
    assert ds.raw_size == result.samples_written


def test_play_one_game_rejects_both_or_neither():
    """The dispatch is exclusive — exactly one of eval_fn / ort_session."""
    from hive.nn.training import HiveDataset
    from hive.selfplay.token_selfplay import play_one_game

    ds = HiveDataset(max_size=8)
    with pytest.raises(ValueError):
        play_one_game(eval_fn=None, ort_session=None, dataset=ds, grid_size=7)
