"""Regression tests for the token-based HiveTransformer.

The Rust MCTS reads the policy tensor via `PolicyIndex::DotProduct` which
expects D-major layout: `policy[q_offset + d * g2 + src]`. With
`g2 = SEQ_LEN` and `src = token_slot`, the math becomes:

    Q[d, token_slot] = policy[d * SEQ_LEN + token_slot]                # Q section
    K[d, token_slot] = policy[SEQ_LEN * D + d * SEQ_LEN + token_slot]  # K section

This test pins that contract: any future refactor of the policy head must
preserve D-major layout for both the Q and K sections.
"""
import math

import torch

from hive.nn.transformer import (
    HiveTransformer, SEQ_LEN, BILINEAR_DIM, F_FLAGS, create_model,
    TOKEN_KIND_GAME, TOKEN_KIND_PIECE, TOKEN_KIND_CELL,
)


def _synthetic_batch(B: int = 2, used: int = 14):
    cat = torch.zeros(B, SEQ_LEN, 5, dtype=torch.int32)
    pos = torch.zeros(B, SEQ_LEN, 2, dtype=torch.int32)
    flg = torch.zeros(B, SEQ_LEN, F_FLAGS, dtype=torch.float32)
    mask = torch.zeros(B, SEQ_LEN, dtype=torch.bool)
    cat[:, 0, 0] = TOKEN_KIND_GAME
    cat[:, 1:5, 0] = TOKEN_KIND_PIECE
    cat[:, 5:used, 0] = TOKEN_KIND_CELL
    pos[:, 1:5, 0] = torch.tensor([0, 1, 0, -1])
    pos[:, 1:5, 1] = torch.tensor([0, 0, 1, 1])
    pos[:, 5:used, 0] = torch.tensor([2, 1, -1, -2, 0, 1, 2, 3, 4][:used - 5])
    pos[:, 5:used, 1] = torch.tensor([0, -1, 2, 1, -1, 1, 0, 0, 0][:used - 5])
    mask[:, :used] = True
    return cat, pos, flg, mask


def test_forward_shapes():
    m = create_model()
    m.eval()
    cat, pos, flg, mask = _synthetic_batch()
    with torch.no_grad():
        policy, wdl, aux = m(cat, pos, flg, mask)
    B = cat.shape[0]
    assert policy.shape == (B, 2 * SEQ_LEN * BILINEAR_DIM)
    assert wdl.shape == (B, 3)
    assert aux.shape == (B, 6)
    assert torch.isfinite(policy).all()
    assert torch.isfinite(wdl).all()
    assert torch.isfinite(aux).all()


def test_policy_d_major_layout():
    """Reading policy as (D, L) per section must reproduce the Q/K projections.

    Rust's PolicyIndex::DotProduct math does:
        Q[d, slot] := policy[d * L + slot]
        K[d, slot] := policy[L*D + d * L + slot]
        logit(mover, dest) = Σ_d Q[d, mover] * K[d, dest] / sqrt(D)

    For this test we hook into the model's policy_q / policy_k linears and
    verify the layout produced by the forward pass matches a direct
    computation. This is the contract the Rust MCTS relies on.
    """
    torch.manual_seed(0)
    m = create_model()
    m.eval()
    cat, pos, flg, mask = _synthetic_batch()
    B = cat.shape[0]

    with torch.no_grad():
        # Run the trunk by reusing internals, then check the policy head.
        x, positional, piece_mask = m._embed(cat, pos, flg)
        z = cat[..., 3].long().clamp(0, 7)
        attn_bias = (
            m.rel_bias(pos[..., 0].long(), pos[..., 1].long(), positional)
            + m.vert_bias(z, piece_mask)
        )
        mask_f = mask.to(attn_bias.dtype)
        pad_bias = (1.0 - mask_f).unsqueeze(1).unsqueeze(1) * -1e9
        attn_mask = attn_bias + pad_bias
        h = x
        for block in m.blocks:
            h = block(h, attn_mask)
        h = m.final_norm(h)
        q_emb = m.policy_q(h)              # (B, L, D)
        k_emb = m.policy_k(h)              # (B, L, D)

        policy, _, _ = m(cat, pos, flg, mask)

        # Decode policy back into (B, D, L) per section.
        L, D = SEQ_LEN, BILINEAR_DIM
        q_decoded = policy[:, :L * D].view(B, D, L)
        k_decoded = policy[:, L * D:].view(B, D, L)

        # Q[token, d] from forward must equal Q[d, token] D-major lookup.
        assert torch.allclose(
            q_decoded, q_emb.permute(0, 2, 1).contiguous(), atol=1e-5
        ), "Q section is not D-major"
        assert torch.allclose(
            k_decoded, k_emb.permute(0, 2, 1).contiguous(), atol=1e-5
        ), "K section is not D-major"


def test_backward_produces_gradients():
    m = create_model()
    m.train()
    cat, pos, flg, mask = _synthetic_batch()
    flg = flg + 1e-3 * torch.randn_like(flg)  # break perfect symmetry
    policy, wdl, aux = m(cat, pos, flg, mask)
    (policy.mean() + wdl.mean() + aux.mean()).backward()
    has_grad = sum(
        1 for p in m.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    assert has_grad > 0, "no parameters received gradients"


def test_pad_tokens_dont_leak_via_attention():
    """A real token should produce the same embedding regardless of whether
    surrounding pad slots contain garbage values, because the attention mask
    excludes pads."""
    torch.manual_seed(0)
    m = create_model()
    m.eval()

    cat_a, pos_a, flg_a, mask_a = _synthetic_batch(B=1, used=10)
    cat_b = cat_a.clone()
    pos_b = pos_a.clone()
    flg_b = flg_a.clone()
    mask_b = mask_a.clone()
    # Corrupt pad slots in `b` only — these should be ignored.
    cat_b[:, 20:40, 0] = 2   # fake PIECE tokens
    pos_b[:, 20:40, 0] = torch.randint(-5, 6, (20,))
    pos_b[:, 20:40, 1] = torch.randint(-5, 6, (20,))
    flg_b[:, 20:40, :] = torch.randn(20, F_FLAGS)
    # mask_b stays the same — slots 20:40 are NOT real.

    with torch.no_grad():
        policy_a, wdl_a, aux_a = m(cat_a, pos_a, flg_a, mask_a)
        policy_b, wdl_b, aux_b = m(cat_b, pos_b, flg_b, mask_b)

    # The GAME token's contribution flows through self-attention over real
    # tokens only. Pads must not change it.
    assert torch.allclose(wdl_a, wdl_b, atol=1e-5), \
        "pad-slot garbage leaked into value head — attention mask broken"
    assert torch.allclose(aux_a, aux_b, atol=1e-5), \
        "pad-slot garbage leaked into aux head"
