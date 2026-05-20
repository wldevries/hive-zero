"""Token-based transformer for Hive.

This module replaces the CNN/bilinear-Q·K policy network in `hive/nn/model.py`.
The trunk is a stack of pre-norm transformer blocks operating on the
variable-length token sequence emitted by `rust/.../hive-game/src/tokenize.rs`.

Input contract (must match `rust/crates/hive-game/src/tokenize.rs`):
    categoricals : (B, L, 5)  uint8   — kind, piece_type, color, stack_depth, count
    positions    : (B, L, 2)  int8    — (q, r) axial coordinates
    flags        : (B, L, F)  float32 — per-token continuous/binary flag bundle
    mask         : (B, L)     bool    — True for real tokens, False for pad

Output contract:
    policy : (B, 2*L*D) float32 — [Q (L*D)  ||  K (L*D)], each in D-major
                                  order: `policy[d*L + token_slot]`. Matches
                                  `PolicyIndex::DotProduct { g2 = L,
                                  embed_dim = D, q_offset=0, k_offset=L*D }`.
    wdl    : (B, 3)     float32 — raw WDL logits (caller softmaxes)
    aux    : (B, 6)     float32 — sigmoid-activated auxiliary signals
                                  [my_qd, opp_qd, my_qe, opp_qe, my_mob, opp_mob]
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- vocabulary (mirror rust/crates/hive-game/src/tokenize.rs) ----------

SEQ_LEN: int = 128
"""Padded token sequence length."""

BILINEAR_DIM: int = 64
"""Bilinear Q/K embedding dim (D). Matches Rust `BILINEAR_DIM`."""

F_FLAGS: int = 8
"""Per-token continuous/binary flag dim."""

NUM_TOKEN_KINDS: int = 5      # PAD, GAME, PIECE, RESERVE, CELL
NUM_PIECE_TYPES: int = 6      # NONE + 5 base types
NUM_COLORS: int = 3           # NONE, MINE, OPP
NUM_Z_LEVELS: int = 8         # 0..=7 (per-piece distance-from-top: 0=top piece)
NUM_COUNT_BUCKETS: int = 12   # 0..=11 (reserve counts per type)

# Δz bucket range for the vertical attention bias. Signed offset in [-7, +7]
# → 15 buckets per head. Matches Rust `MAX_STACK_DEPTH = 7`.
Z_REL_RANGE: int = 2 * NUM_Z_LEVELS - 1
Z_REL_OFFSET: int = NUM_Z_LEVELS - 1

# Token-kind constants used to detect positional tokens in the model
TOKEN_KIND_PAD = 0
TOKEN_KIND_GAME = 1
TOKEN_KIND_PIECE = 2
TOKEN_KIND_RESERVE = 3
TOKEN_KIND_CELL = 4

# Axial coord offset for the absolute-position lookup. Board recentering
# keeps live pieces within q,r ∈ [-(GRID_SIZE//2), +(GRID_SIZE//2)] = [-11, 11]
# for the default 23×23 board.
POS_RANGE: int = 23
POS_OFFSET: int = POS_RANGE // 2  # 11 → maps q,r ∈ [-11,11] to [0, 22]

# Relative-bias signed-(Δq, Δr) range: 2 * POS_OFFSET + 1.
REL_RANGE: int = 2 * POS_RANGE - 1   # 45 buckets per axis
REL_OFFSET: int = POS_RANGE - 1      # 22


# ---------- positional embedding ---------------------------------------------


class HexAbsolutePosition(nn.Module):
    """Learned absolute (q, r) → d_model lookup. Applied only to positional
    tokens (`kind ∈ {PIECE, CELL}`); other tokens get the zero embedding.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # 23×23 table for (q, r). Slot (POS_OFFSET, POS_OFFSET) = origin.
        self.table = nn.Embedding(POS_RANGE * POS_RANGE, d_model)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, q: torch.Tensor, r: torch.Tensor, positional: torch.Tensor) -> torch.Tensor:
        """
        q, r:         (B, L) int (axial coords; pad-slot values are arbitrary)
        positional:   (B, L) bool — True for tokens that have a real position
        returns:      (B, L, d_model)
        """
        q_idx = (q + POS_OFFSET).clamp(0, POS_RANGE - 1)
        r_idx = (r + POS_OFFSET).clamp(0, POS_RANGE - 1)
        flat = q_idx * POS_RANGE + r_idx
        emb = self.table(flat)                                   # (B, L, d_model)
        emb = emb * positional.unsqueeze(-1).to(emb.dtype)
        return emb


class VerticalBias(nn.Module):
    """Learned per-head additive attention bias keyed on signed Δz (difference
    in stack position) between PIECE token pairs. Lets the trunk reason about
    "this token is two layers above me" in beetle stacks. Non-PIECE pairs get
    no contribution from this term.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.table = nn.Parameter(torch.zeros(num_heads, Z_REL_RANGE))
        # Zero init keeps the model identity-equivalent at start, same
        # convention as HexRelativeBias.

    def forward(self, z: torch.Tensor, piece_mask: torch.Tensor) -> torch.Tensor:
        """
        z:           (B, L) long — piece position in stack (0 elsewhere)
        piece_mask:  (B, L) bool — True for PIECE tokens, False otherwise
        returns:     (B, num_heads, L, L) additive bias tensor
        """
        B, L = z.shape
        dz = z.unsqueeze(2) - z.unsqueeze(1)                  # (B, L, L)
        idx = (dz + Z_REL_OFFSET).clamp(0, Z_REL_RANGE - 1)   # (B, L, L)
        idx_flat = idx.reshape(B, L * L)
        # Gather per head from (H, Z_REL_RANGE). `self.table[:, idx_flat]`
        # yields (H, B, L*L); permute → (B, H, L*L).
        bias = self.table[:, idx_flat].permute(1, 0, 2).view(B, self.num_heads, L, L)
        # Mask: only PIECE-PIECE pairs see the vertical bias. Non-PIECE
        # tokens have z=0 anyway, so the (0,0) bucket would otherwise apply
        # everywhere — gating keeps that channel meaningful only for stacks.
        both_pieces = piece_mask.unsqueeze(2) & piece_mask.unsqueeze(1)   # (B, L, L)
        bias = bias * both_pieces.unsqueeze(1).to(bias.dtype)
        return bias


class HexRelativeBias(nn.Module):
    """Learned per-head bias keyed on signed (Δq, Δr) between every token pair.

    Hex distance is symmetry-invariant under D6 and direction is permuted, so
    the bias naturally captures hex geometry. Non-positional token pairs use
    a single shared "null" bias scalar per head.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        # Signed (Δq, Δr) buckets — Δq, Δr ∈ [-22, 22] → 45 each.
        self.table = nn.Parameter(torch.zeros(num_heads, REL_RANGE, REL_RANGE))
        self.null_bias = nn.Parameter(torch.zeros(num_heads))
        # Zero init keeps the model identity-equivalent at start.

    def forward(self, q: torch.Tensor, r: torch.Tensor, positional: torch.Tensor) -> torch.Tensor:
        """
        q, r:        (B, L) int
        positional:  (B, L) bool — True for tokens with real positions
        returns:     (B, num_heads, L, L) additive bias tensor
        """
        B, L = q.shape
        dq = q.unsqueeze(2) - q.unsqueeze(1)           # (B, L, L)
        dr = r.unsqueeze(2) - r.unsqueeze(1)           # (B, L, L)
        dq_idx = (dq + REL_OFFSET).clamp(0, REL_RANGE - 1)
        dr_idx = (dr + REL_OFFSET).clamp(0, REL_RANGE - 1)
        flat = dq_idx * REL_RANGE + dr_idx              # (B, L, L)
        flat_idx = flat.reshape(B, L * L)
        # Gather per head: (num_heads, L*L). Avoid materializing a huge
        # broadcast — index per head then stack.
        per_head = self.table.view(self.num_heads, REL_RANGE * REL_RANGE)  # (H, R²)
        # (B, L*L, H) via gather along R²:
        bias = per_head[:, flat_idx].permute(1, 0, 2)   # (B, H, L*L)
        bias = bias.view(B, self.num_heads, L, L)

        # Replace bias entries where either token is non-positional with null_bias.
        both_positional = positional.unsqueeze(2) & positional.unsqueeze(1)  # (B, L, L)
        null = self.null_bias.view(1, self.num_heads, 1, 1).expand(B, -1, L, L)
        bias = torch.where(both_positional.unsqueeze(1), bias, null)
        return bias


# ---------- transformer block ------------------------------------------------


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with an explicit per-head additive bias."""

    def __init__(self, d_model: int, num_heads: int, ffn_mult: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # Precompute the 1/sqrt(head_dim) scale as a Python constant so it
        # gets folded into the ONNX graph as a static value rather than a
        # runtime `Slice → Cast → Sqrt → Cast → Div` chain. That chain
        # rounds through fp32 inside an otherwise-fp16 graph and pulls
        # host↔device Memcpy nodes (one per block).
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)

        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        x:          (B, L, d_model)
        attn_mask:  (B, num_heads, L, L) float — additive bias already
                    composed with the pad-key mask (pad columns set to
                    -inf upstream). Always float (no bool tensors enter
                    per-block code — keeps the graph on CUDA without the
                    Cast→Where→Memcpy fallback to CPU that hits when bool
                    ops have no GPU impl).
        """
        B, L, D = x.shape
        # ---- self-attention ----
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, L, head_dim)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=self.attn_scale,
        )
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, L, D)
        x = x + self.out_proj(attn_out)

        # ---- FFN ----
        x = x + self.ffn(self.ln2(x))
        return x


# ---------- model -----------------------------------------------------------


class HiveTransformer(nn.Module):
    """Token-based AlphaZero-style network for Hive.

    Architecture:
        embed (sum-of-categorical-embeddings + abs-pos + flag-projection)
        → N × TransformerBlock(d_model, num_heads, ffn_mult, attn_bias=HexRelativeBias)
        → final LayerNorm
        → policy head (linear → Q, linear → K, dot-product per legal move)
        → value head  (h[GAME] → MLP → 3 logits)
        → aux head    (h[GAME] → MLP → 6 sigmoids)
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        n_blocks: int = 8,
        ffn_mult: int = 4,
        bilinear_dim: int = BILINEAR_DIM,
    ):
        super().__init__()
        self.game = "hive"
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_blocks = n_blocks
        self.ffn_mult = ffn_mult
        self.bilinear_dim = bilinear_dim
        self.seq_len = SEQ_LEN

        # ---- categorical embeddings (summed) ----
        self.kind_emb = nn.Embedding(NUM_TOKEN_KINDS, d_model)
        self.piece_type_emb = nn.Embedding(NUM_PIECE_TYPES, d_model)
        self.color_emb = nn.Embedding(NUM_COLORS, d_model)
        # Per-piece distance-from-top in stack (0=top piece, height-1=bottom).
        # Anchoring z=0 to the top means solo pieces and tops-of-stacks share
        # z_emb(0) — the model sees "I'm the visible/mobile piece" as a single
        # signal regardless of stack height instead of having to compose
        # FLAG_TOP_OF_STACK with a height-dependent z. Gated to zero on
        # non-PIECE tokens in `_embed`. Replaces the prior top-only
        # `stack_depth_emb` now that buried pieces get their own tokens.
        self.z_emb = nn.Embedding(NUM_Z_LEVELS, d_model)
        self.count_emb = nn.Embedding(NUM_COUNT_BUCKETS, d_model)
        for emb in (self.kind_emb, self.piece_type_emb, self.color_emb,
                    self.z_emb, self.count_emb):
            nn.init.normal_(emb.weight, std=0.02)

        # ---- absolute position embedding ----
        self.abs_pos = HexAbsolutePosition(d_model)

        # ---- flag projection ----
        self.flag_proj = nn.Linear(F_FLAGS, d_model, bias=True)

        # ---- relative bias (shared across all blocks; one head bias per layer keeps
        # the trunk uniform and matches T5-style designs) ----
        self.rel_bias = HexRelativeBias(num_heads)
        # Vertical (Δz) bias for PIECE-PIECE pairs, composed additively with rel_bias.
        self.vert_bias = VerticalBias(num_heads)

        # ---- transformer trunk ----
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ffn_mult)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # ---- policy head ----
        self.policy_q = nn.Linear(d_model, bilinear_dim)
        self.policy_k = nn.Linear(d_model, bilinear_dim)

        # ---- value head (reads GAME token) ----
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )

        # ---- aux head (reads GAME token) ----
        self.aux_mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 6),
        )

    def _embed(
        self,
        categoricals: torch.Tensor,  # (B, L, 5) uint8
        positions: torch.Tensor,      # (B, L, 2) int8
        flags: torch.Tensor,          # (B, L, F_FLAGS) float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the per-token embedding, the positional-token mask, and the
        PIECE-token mask used for the Δz attention bias."""
        kind = categoricals[..., 0].long()
        piece_type = categoricals[..., 1].long()
        color = categoricals[..., 2].long()
        z = categoricals[..., 3].long().clamp(0, NUM_Z_LEVELS - 1)
        count = categoricals[..., 4].long().clamp(0, NUM_COUNT_BUCKETS - 1)

        q = positions[..., 0].long()
        r = positions[..., 1].long()
        positional = (kind == TOKEN_KIND_PIECE) | (kind == TOKEN_KIND_CELL)  # (B, L) bool
        piece_mask = (kind == TOKEN_KIND_PIECE)                              # (B, L) bool

        # z_emb is gated to zero on non-PIECE tokens — z=0 there is meaningless,
        # and a constant bias on every non-PIECE token would just be absorbed
        # into kind_emb anyway.
        z_contribution = self.z_emb(z) * piece_mask.unsqueeze(-1).to(self.z_emb.weight.dtype)

        x = (
            self.kind_emb(kind)
            + self.piece_type_emb(piece_type)
            + self.color_emb(color)
            + z_contribution
            + self.count_emb(count)
            + self.abs_pos(q, r, positional)
            + self.flag_proj(flags)
        )
        return x, positional, piece_mask

    def forward(
        self,
        categoricals: torch.Tensor,   # (B, L, 5) uint8/long
        positions: torch.Tensor,       # (B, L, 2) int8/long
        flags: torch.Tensor,           # (B, L, F_FLAGS) float
        mask: torch.Tensor,            # (B, L) bool — real tokens
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            policy_flat : (B, 2 * L * bilinear_dim) — `[Q || K]` D-major
            wdl_logits  : (B, 3)
            aux         : (B, 6) — sigmoid-activated
        """
        # NOTE: don't `assert L == self.seq_len` here — that lets torch.jit
        # tracer fold a tensor->bool comparison into a constant during ONNX
        # export and emits TracerWarning. The shape check happens upstream
        # (Rust tokenizer is the only producer of these tensors and always
        # emits SEQ_LEN-padded batches).
        x, positional, piece_mask = self._embed(categoricals, positions, flags)

        # Compute relative bias once; reused across all blocks.
        q_pos = positions[..., 0].long()
        r_pos = positions[..., 1].long()
        z = categoricals[..., 3].long().clamp(0, NUM_Z_LEVELS - 1)
        attn_bias = (
            self.rel_bias(q_pos, r_pos, positional)           # (B, H, L, L) hex Δq,Δr
            + self.vert_bias(z, piece_mask)                   # (B, H, L, L) Δz on PIECE pairs
        )

        # Compose attn_bias + pad-key mask once at the model entry as
        # PURE FLOAT math. Bool ops (~, masked_fill, Where on bool)
        # frequently lack CUDA kernels in ORT's CUDA EP and get
        # partitioned to CPU — each fallback per block adds a host↔device
        # Memcpy node and breaks the all-GPU graph. By doing one
        # bool→float Cast (which IS CUDA-supported) at the entry and
        # working in float thereafter, every per-block self-attention
        # call sees a plain float additive mask.
        #
        # The saturation constant is `-1e4`, NOT `-1e9` or `-inf`: in fp16
        # exports those overflow to `-inf`, and then `0 * -inf = NaN`
        # (IEEE 754) for the REAL token rows where (1 − mask_f) = 0 — and
        # the NaN propagates through every downstream attention block,
        # producing NaN priors and effectively breaking MCTS. `-1e4` fits
        # well inside fp16's representable range (max ≈ 65504), keeps
        # `0 · (−1e4) = 0` exact, and is still small enough that
        # `exp(−1e4)` underflows to 0 in softmax — i.e. masked positions
        # contribute zero weight as intended.
        mask_f = mask.to(attn_bias.dtype)                       # (B, L) 1.0 real, 0.0 pad
        pad_bias = (1.0 - mask_f).unsqueeze(1).unsqueeze(1) * -1e4  # (B, 1, 1, L)
        attn_mask = attn_bias + pad_bias                         # (B, H, L, L)

        for block in self.blocks:
            x = block(x, attn_mask)

        h = self.final_norm(x)

        # ---- policy: bilinear Q·K over (mover, dest) token pairs ----
        # MCTS reads `policy[d * L + token_slot]` (D-major). Python emits
        # `Q : (B, L, D)` and `K : (B, L, D)`; permute → flatten to match.
        q_emb = self.policy_q(h)                              # (B, L, D)
        k_emb = self.policy_k(h)                              # (B, L, D)
        q_flat = q_emb.permute(0, 2, 1).contiguous().flatten(1)   # (B, D*L)
        k_flat = k_emb.permute(0, 2, 1).contiguous().flatten(1)   # (B, D*L)
        policy_flat = torch.cat([q_flat, k_flat], dim=1)      # (B, 2*L*D)

        # ---- value + aux from GAME token (slot 0 by Rust convention) ----
        game_tok = h[:, 0, :]
        wdl_logits = self.value_mlp(game_tok)
        aux = torch.sigmoid(self.aux_mlp(game_tok))

        return policy_flat, wdl_logits, aux


# ---------- factories / checkpoint API ---------------------------------------


def create_model(model_config: dict | None = None) -> HiveTransformer:
    cfg = model_config or {}
    return HiveTransformer(
        d_model=cfg.get("d_model", 128),
        num_heads=cfg.get("num_heads", 4),
        n_blocks=cfg.get("n_blocks", 8),
        ffn_mult=cfg.get("ffn_mult", 4),
        bilinear_dim=cfg.get("bilinear_dim", BILINEAR_DIM),
    )


def save_checkpoint(
    model: HiveTransformer,
    path: str,
    generation: int = 0,
    metadata: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
):
    """Save model with architecture config and training metadata."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "game": "hive",
        "arch": "transformer",
        "d_model": model.d_model,
        "num_heads": model.num_heads,
        "n_blocks": model.n_blocks,
        "ffn_mult": model.ffn_mult,
        "bilinear_dim": model.bilinear_dim,
        "generation": generation,
        "metadata": metadata or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(path: str) -> tuple[HiveTransformer, dict]:
    """Load a transformer checkpoint. Refuses to load legacy CNN checkpoints
    (these became unusable when `hive/nn/model.py` was deleted)."""
    checkpoint = torch.load(path, weights_only=False)
    arch = checkpoint.get("arch")
    if arch != "transformer":
        raise ValueError(
            f"checkpoint at {path} has arch={arch!r}; transformer required"
        )
    cfg = {
        "d_model":      checkpoint.get("d_model", 128),
        "num_heads":    checkpoint.get("num_heads", 4),
        "n_blocks":     checkpoint.get("n_blocks", 8),
        "ffn_mult":     checkpoint.get("ffn_mult", 4),
        "bilinear_dim": checkpoint.get("bilinear_dim", BILINEAR_DIM),
    }
    model = create_model(cfg)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    compatible = {
        k: v for k, v in state_dict.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    model.load_state_dict(compatible, strict=False)
    if "model_state_dict" not in checkpoint:
        checkpoint = {"generation": 0, "metadata": {}}
    if "generation" not in checkpoint and "iteration" in checkpoint:
        checkpoint["generation"] = checkpoint.pop("iteration")
    model.eval()
    return model, checkpoint


class _OnnxExportWrapper(nn.Module):
    """Wraps HiveTransformer for ONNX export.

    - Softmaxes WDL so ORT consumers see probabilities.
    - With `precision="fp16"` (default), runs the internal model in float16
      while keeping fp32 inputs and outputs at the ONNX boundary. Categorical
      inputs stay int64 (embedding gather has no float16 cost) — only `flags`
      is cast.
    """

    def __init__(self, model: HiveTransformer, precision: str = "fp16"):
        super().__init__()
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"unknown precision {precision!r}")
        self.precision = precision
        m = copy.deepcopy(model)
        if precision == "fp16":
            # Cast float weights only; embedding lookup tables stay fp32 internally
            # but produce fp16 output via .to(dtype) at runtime. Simplest path:
            # cast the whole model and accept the slight init drift.
            m = m.to(torch.float16)
        self.model = m

    def forward(self, categoricals, positions, flags, mask):
        if self.precision == "fp16":
            flags = flags.to(torch.float16)
        policy, wdl_logits, aux = self.model(categoricals, positions, flags, mask)
        wdl = F.softmax(wdl_logits, dim=1)
        if self.precision == "fp16":
            policy = policy.to(torch.float32)
            wdl = wdl.to(torch.float32)
            aux = aux.to(torch.float32)
        return policy, wdl, aux


def export_onnx(model: HiveTransformer, path: str, batch_size: int | None = None,
                precision: str = "fp16"):
    """Export model to ONNX for Rust-native ORT inference.

    Inputs:
        categoricals (B, L, 5) int64
        positions    (B, L, 2) int64
        flags        (B, L, F) float32 (cast to fp16 inside wrapper if requested)
        mask         (B, L)    bool
    Outputs:
        policy (B, 2*L*D), wdl (B, 3) [W,D,L probs], aux (B, 6)
    """
    import os
    L = model.seq_len
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    b = batch_size or 1
    dummy_cat = torch.zeros(b, L, 5, dtype=torch.int64, device=device)
    dummy_pos = torch.zeros(b, L, 2, dtype=torch.int64, device=device)
    dummy_flags = torch.zeros(b, L, F_FLAGS, dtype=torch.float32, device=device)
    dummy_mask = torch.zeros(b, L, dtype=torch.bool, device=device)
    dummy_mask[:, 0] = True  # at least the GAME token is real

    wrapper = _OnnxExportWrapper(model, precision=precision).eval().to(device)
    dynamic_axes = None if batch_size else {
        "categoricals": {0: "batch"},
        "positions":    {0: "batch"},
        "flags":        {0: "batch"},
        "mask":         {0: "batch"},
        "policy":       {0: "batch"},
        "wdl":          {0: "batch"},
        "aux":          {0: "batch"},
    }
    torch.onnx.export(
        wrapper,
        (dummy_cat, dummy_pos, dummy_flags, dummy_mask),
        path,
        input_names=["categoricals", "positions", "flags", "mask"],
        output_names=["policy", "wdl", "aux"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )
    if was_training:
        model.train()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    batch_label = f" (batch={batch_size}, static)" if batch_size else ""
    print(f"  ONNX exported: {path} ({size_mb:.1f} MB, {precision}){batch_label}")
