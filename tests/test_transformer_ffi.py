"""Integration test for the Hive transformer FFI round trip.

Pins the contract between three layers:
    HiveGame (Rust)   →   PyGame.encode_tokens / best_move (PyO3 FFI)
    Python callback   ←   model.forward (PyTorch)
    MCTS              ←   policy in (Q || K) D-major layout

If this test passes, the architecture is wired up end to end. Subsequent
training/self-play phases build on this foundation.
"""
import numpy as np
import pytest
import torch


def _eval_fn_factory(model, device: str):
    def eval_fn(categoricals, positions, flags, mask):
        cat = torch.from_numpy(np.asarray(categoricals)).to(device=device, dtype=torch.int64)
        pos = torch.from_numpy(np.asarray(positions)).to(device=device, dtype=torch.int64)
        flg = torch.from_numpy(np.asarray(flags)).to(device=device, dtype=torch.float32)
        msk = torch.from_numpy(np.asarray(mask)).to(device=device, dtype=torch.bool)
        with torch.no_grad():
            policy, wdl_logits, aux = model(cat, pos, flg, msk)
            wdl = torch.softmax(wdl_logits, dim=1)
        return (
            policy.float().cpu().numpy().astype(np.float32),
            wdl.cpu().numpy().astype(np.float32),
            aux.cpu().numpy().astype(np.float32),
        )
    return eval_fn


def test_encode_tokens_matches_python_shapes():
    from engine_zero import HiveGame, token_vocab
    from hive.nn.transformer import SEQ_LEN, F_FLAGS, BILINEAR_DIM

    L, F, D = token_vocab()
    assert (L, F, D) == (SEQ_LEN, F_FLAGS, BILINEAR_DIM), \
        "Rust and Python token vocab constants disagree"

    g = HiveGame(grid_size=7)
    cat, pos, flg, msk = g.encode_tokens()
    assert cat.shape == (SEQ_LEN, 5)
    assert pos.shape == (SEQ_LEN, 2)
    assert flg.shape == (SEQ_LEN, F_FLAGS)
    assert msk.shape == (SEQ_LEN,)
    # Empty board: 1 GAME + 0 PIECE + 5 my reserves + 5 opp reserves + 1 dest cell = 12
    assert int(msk.sum()) == 12
    assert cat[0, 0] == 1, "slot 0 must be the [GAME] token"


def test_best_move_emits_valid_first_move():
    from engine_zero import HiveGame
    from hive.nn.transformer import create_model

    torch.manual_seed(0)
    model = create_model()
    model.eval()

    g = HiveGame(grid_size=7)
    move_str = g.best_move(_eval_fn_factory(model, "cpu"),
                           simulations=8, play_batch=4)
    # First move is always a placement at the origin — exact piece type
    # depends on untrained-net priors, but the format must be valid UHP.
    assert move_str in {"wQ", "wS1", "wB1", "wG1", "wA1"}, \
        f"first move {move_str!r} is not a recognised first-move UHP string"


def test_two_ply_round_trip_does_not_panic():
    from engine_zero import HiveGame
    from hive.nn.transformer import create_model

    torch.manual_seed(0)
    model = create_model()
    model.eval()
    eval_fn = _eval_fn_factory(model, "cpu")

    g = HiveGame(grid_size=7)
    m1 = g.best_move(eval_fn, simulations=8, play_batch=4)
    assert g.play_move_uhp(m1), f"failed to play first move {m1!r}"
    m2 = g.best_move(eval_fn, simulations=8, play_batch=4)
    assert g.play_move_uhp(m2), f"failed to play second move {m2!r}"
    assert g.move_count == 2


def test_best_move_ort_is_not_implemented():
    """Phase G hasn't ported ORT yet; calling it should error clearly so
    callers can fall back to the Python callback path."""
    from engine_zero import HiveGame

    g = HiveGame(grid_size=7)
    with pytest.raises(NotImplementedError):
        g.best_move_ort(None)  # ort_session is ignored; defaults are fine


def test_encode_tokens_with_moves_returns_token_slot_pairs():
    """Each legal move must be tagged with a (mover_slot, dest_slot) pair
    that indexes into the same token batch's L positions."""
    from engine_zero import HiveGame
    from hive.nn.transformer import SEQ_LEN

    g = HiveGame(grid_size=7)
    cat, pos, flg, msk, moves = g.encode_tokens_with_moves()
    assert msk.shape == (SEQ_LEN,)
    used = int(msk.sum())
    # All slot indices must be inside the real (non-pad) range.
    for mover_slot, dest_slot, _piece, _from, _to in moves:
        assert 0 <= mover_slot < used, f"mover slot {mover_slot} out of [0, {used})"
        assert 0 <= dest_slot < used, f"dest slot {dest_slot} out of [0, {used})"
