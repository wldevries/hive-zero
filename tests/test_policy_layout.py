"""Regression tests for the Q/K bilinear movement policy layout.

`nn.Conv2d(C, D, 1)(p)` outputs shape `(B, D, G, G)`; `.flatten(1)` produces
a tensor with D-major memory layout: `flat[b, d*G² + cell]`. The Rust MCTS
(`core-game/src/mcts/search.rs` DotProduct branch) reads with that exact
stride: `policy[q_offset + d*g2 + src_cell]`.

The Python training path must reshape it the same way before gathering. A
prior bug used `.view(B, gs2, D)`, which silently re-interpreted the memory
as cell-major (`flat[b, cell*D + d]`) and scrambled channel<->cell. The
result was that movement priors trained against a transposed objective
while inference used the correct formula — gradients went nowhere useful.
"""
import math

import torch
import torch.nn as nn

from hive.encoding.move_encoder import NUM_PLACE_CHANNELS


def _build_policy_logits(B: int, D: int, G: int, seed: int = 0):
    """Mimic the model's policy head: `place | Q | K`, each .flatten(1) of
    a Conv2d output. Returns (policy_logits, q_out, k_out) so callers can
    cross-check against the un-flattened spatial tensors."""
    torch.manual_seed(seed)
    place_conv = nn.Conv2d(4, NUM_PLACE_CHANNELS, 1)
    q_conv = nn.Conv2d(4, D, 1)
    k_conv = nn.Conv2d(4, D, 1)
    feats = torch.randn(B, 4, G, G)
    place_out = place_conv(feats)
    q_out = q_conv(feats)
    k_out = k_conv(feats)
    policy_logits = torch.cat(
        [place_out.flatten(1), q_out.flatten(1), k_out.flatten(1)], dim=1
    )
    return policy_logits, q_out, k_out


def _train_path_qk_dot(policy_logits, B, D, G, m_src, m_dst):
    """Replicate the Q/K gather + dot product from training.py."""
    gs2 = G * G
    place_end = NUM_PLACE_CHANNELS * gs2
    q = policy_logits[:, place_end:place_end + D * gs2].view(B, D, gs2).transpose(1, 2)
    k = policy_logits[:, place_end + D * gs2:].view(B, D, gs2).transpose(1, 2)
    src_exp = m_src.unsqueeze(-1).expand(-1, -1, D)
    dst_exp = m_dst.unsqueeze(-1).expand(-1, -1, D)
    q_g = torch.gather(q, 1, src_exp)
    k_g = torch.gather(k, 1, dst_exp)
    return (q_g * k_g).sum(-1) / math.sqrt(D)


def test_qk_dot_matches_conv_output():
    """Training-path Q·K must agree with the dot product on the spatial
    Conv2d output `(B, D, G, G)` indexed at the (src_cell, dst_cell) hexes."""
    B, D, G = 2, 8, 7
    policy_logits, q_out, k_out = _build_policy_logits(B, D, G, seed=0)

    pairs = [(0, 0), (5, 17), (G * G - 1, G * G - 2), (12, 36)]
    M = len(pairs)
    m_src = torch.tensor([[s for s, _ in pairs]] * B)
    m_dst = torch.tensor([[d for _, d in pairs]] * B)

    actual = _train_path_qk_dot(policy_logits, B, D, G, m_src, m_dst)

    expected = torch.zeros(B, M)
    for m, (src, dst) in enumerate(pairs):
        sr, sc = divmod(src, G)
        dr, dc = divmod(dst, G)
        expected[:, m] = (q_out[:, :, sr, sc] * k_out[:, :, dr, dc]).sum(dim=1) / math.sqrt(D)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_qk_dot_matches_rust_flat_formula():
    """Training-path Q·K must agree with Rust's flat-index formula:
    `policy[q_offset + d*g2 + src_cell] * policy[k_offset + d*g2 + dst_cell]`
    summed over d, divided by sqrt(D)."""
    B, D, G = 1, 4, 5
    gs2 = G * G
    place_end = NUM_PLACE_CHANNELS * gs2
    q_offset = place_end
    k_offset = place_end + D * gs2

    torch.manual_seed(1)
    total_size = place_end + 2 * D * gs2
    policy_logits = torch.randn(B, total_size)

    src_cell, dst_cell = 7, 13
    m_src = torch.tensor([[src_cell]])
    m_dst = torch.tensor([[dst_cell]])

    actual = _train_path_qk_dot(policy_logits, B, D, G, m_src, m_dst)

    expected = torch.zeros(B, 1)
    for b in range(B):
        s = 0.0
        for d in range(D):
            s += (
                policy_logits[b, q_offset + d * gs2 + src_cell].item()
                * policy_logits[b, k_offset + d * gs2 + dst_cell].item()
            )
        expected[b, 0] = s / math.sqrt(D)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_buggy_view_disagrees_with_conv_output():
    """Sanity check: confirm the *old* `.view(B, gs2, D)` reshape produces
    different values from indexing into the Conv2d output. If this ever
    starts passing, either the test or the layout convention is wrong."""
    B, D, G = 2, 8, 7
    gs2 = G * G
    place_end = NUM_PLACE_CHANNELS * gs2
    policy_logits, q_out, k_out = _build_policy_logits(B, D, G, seed=2)

    # Old buggy reshape — cell-major reinterpretation of D-major memory.
    q_buggy = policy_logits[:, place_end:place_end + D * gs2].view(B, gs2, D)
    k_buggy = policy_logits[:, place_end + D * gs2:].view(B, gs2, D)

    src_cell, dst_cell = 12, 36
    m_src = torch.tensor([[src_cell]] * B)
    m_dst = torch.tensor([[dst_cell]] * B)
    src_exp = m_src.unsqueeze(-1).expand(-1, -1, D)
    dst_exp = m_dst.unsqueeze(-1).expand(-1, -1, D)
    q_g = torch.gather(q_buggy, 1, src_exp)
    k_g = torch.gather(k_buggy, 1, dst_exp)
    buggy = (q_g * k_g).sum(-1) / math.sqrt(D)

    sr, sc = divmod(src_cell, G)
    dr, dc = divmod(dst_cell, G)
    correct = (q_out[:, :, sr, sc] * k_out[:, :, dr, dc]).sum(dim=1, keepdim=True) / math.sqrt(D)

    diff = (buggy - correct).abs().max().item()
    assert diff > 1e-3, (
        f"Buggy reshape unexpectedly agrees with correct formula (max diff {diff:.2e}). "
        "Either the layout convention has changed or the test is no longer meaningful."
    )
