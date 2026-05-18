"""Training loop for the Hive token-based transformer.

The policy target is a distribution over `(mover_slot, dest_slot)` token
pairs at each position. There is *one* softmax over the full legal-move set
(placements and movements unified), matching the bilinear Q·K head the
Rust MCTS consumes at inference. See `hive/nn/transformer.py` for the model
and `rust/crates/hive-game/src/tokenize.rs` for the token vocabulary.

Replay-buffer schema (HDF5):

    tokens_categoricals  (N, L, 5)  uint8
    tokens_positions     (N, L, 2)  int8
    tokens_flags         (N, L, F)  float16
    tokens_mask          (N, L)     bool
    move_src             (N, K)     uint8   — mover-token slot per legal move
    move_dst             (N, K)     uint8   — dest-token slot per legal move
    move_probs           (N, K)     float32 — MCTS visit probability
    num_moves            (N,)       int32   — number of legal moves (≤ K)
    value_targets        (N,)       float32 — scalar in [-1, 1]
    root_q_targets       (N,)       float32 — MCTS root W−L (diagnostics)
    value_only           (N,)       bool    — mask value loss (timeouts)
    policy_only          (N,)       bool    — mask policy loss (fast-cap turns)
    aux_targets          (N, 6)     float32 — [my_qd, opp_qd, my_qe, opp_qe,
                                               my_mob, opp_mob]

D6 symmetry augmentation: permutes per-token `(q, r)` positions through the
12 axial transforms. Policy targets are *invariant* — tokens keep their
slot identity under augmentation; only their spatial coordinates change.
This is materially simpler than the CNN-era cell-permutation flow.
"""

from __future__ import annotations
import contextlib
import math
import os

# Disable HDF5 advisory file locking so DataLoader workers can open the
# replay buffer 'r' while the main process is detached. libhdf5 reads this
# on every open, and the var propagates through spawn() on Windows.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional
from tqdm import tqdm

from .transformer import (
    HiveTransformer, create_model,
    SEQ_LEN, F_FLAGS, BILINEAR_DIM,
)
from shared.replay_buffer import handle_buffer_size_mismatch

# Per-sample cap on the (mover, dest) policy-target list. Hive mid-game
# rarely exceeds ~150 legal moves; 256 leaves comfortable headroom.
MAX_LEGAL_MOVES = 256


def _scalar_to_wdl(v: torch.Tensor) -> torch.Tensor:
    """Convert scalar value targets in [-1, 1] to (W, D, L) soft targets summing to 1."""
    w = v.clamp(min=0)
    l = (-v).clamp(min=0)
    d = 1.0 - w - l
    return torch.stack([w, d, l], dim=1)


# ---------------------------------------------------------------------------
# D6 token-position permutations: lazy-loaded once.
# ---------------------------------------------------------------------------
_AXIAL_TRANSFORMS: Optional[np.ndarray] = None  # (12, 2, 2) int8

def _load_axial_transforms() -> np.ndarray:
    global _AXIAL_TRANSFORMS
    if _AXIAL_TRANSFORMS is None:
        from engine_zero import d6_axial_transforms
        mats = d6_axial_transforms()
        _AXIAL_TRANSFORMS = np.stack([np.asarray(m, dtype=np.int8) for m in mats])
    return _AXIAL_TRANSFORMS


_H5_DATASETS = (
    "tokens_categoricals", "tokens_positions", "tokens_flags", "tokens_mask",
    "move_src", "move_dst", "move_probs", "num_moves",
    "value_targets", "root_q_targets",
    "value_only", "policy_only",
    "aux_targets",
)


def _h5_worker_init_fn(worker_id: int):
    """DataLoader worker init: each worker opens its own 'r' h5 handle."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    if hasattr(dataset, "_open_for_read"):
        dataset._open_for_read()


class HiveDataset(Dataset):
    """Replay buffer for the token transformer architecture.

    Per sample we store:
      - the full token batch (categoricals, positions, flags, mask), padded to L
      - the sparse legal-move list as (mover_slot, dest_slot, prob) triples
      - scalar value target + boolean policy/value masks
      - 6 aux signals (queen danger/escape/mobility per side)

    There is no separate placement target — every legal move (placement or
    movement) is a `(mover, dest)` token pair. The single policy CE softmax
    runs over the full legal-move list per sample, matching what MCTS
    consumes at inference.
    """

    def __init__(self, max_size: int = 50_000, buf_dir: str | None = None):
        self.max_size = max_size
        self._h5file = None
        self._h5path: str | None = None
        self.augment_symmetry = False

        def _zeros(shape, dtype):
            return np.zeros(shape, dtype=dtype)

        if buf_dir is not None:
            import h5py
            os.makedirs(buf_dir, exist_ok=True)
            h5path = os.path.join(buf_dir, "replay.h5")
            self._h5path = h5path
            resuming = os.path.exists(h5path)
            self._h5file = h5py.File(h5path, "r+" if resuming else "w")

            if resuming:
                stored_max = int(self._h5file.attrs["max_size"])
                if stored_max != max_size:
                    max_size, self._h5file = handle_buffer_size_mismatch(
                        self._h5file, h5path, max_size
                    )
                    self.max_size = max_size
                stored_arch = self._h5file.attrs.get("arch", "")
                if stored_arch != "transformer":
                    raise ValueError(
                        f"buffer at {h5path} has arch={stored_arch!r}; "
                        "transformer-arch buffer required (delete to start fresh)"
                    )
                self._count = int(self._h5file.attrs["count"])
                self._size = int(self._h5file.attrs["size"])
                print(f"  Replay buffer resumed: {self._size}/{max_size} samples from {h5path}")
            else:
                self._count = 0
                self._size = 0
                self._h5file.attrs["max_size"] = max_size
                self._h5file.attrs["arch"] = "transformer"
                self._h5file.attrs["count"] = 0
                self._h5file.attrs["size"] = 0

            def _ds(name, shape, dtype):
                if name in self._h5file:
                    return self._h5file[name]
                return self._h5file.create_dataset(name, shape=shape, dtype=dtype, track_order=False)
        else:
            self._count = 0
            self._size = 0
            _ds = lambda name, shape, dtype: _zeros(shape, dtype)

        self.tokens_categoricals = _ds("tokens_categoricals", (max_size, SEQ_LEN, 5),       np.uint8)
        self.tokens_positions    = _ds("tokens_positions",    (max_size, SEQ_LEN, 2),       np.int8)
        self.tokens_flags        = _ds("tokens_flags",        (max_size, SEQ_LEN, F_FLAGS), np.float16)
        self.tokens_mask         = _ds("tokens_mask",         (max_size, SEQ_LEN),          np.bool_)
        self.move_src            = _ds("move_src",            (max_size, MAX_LEGAL_MOVES),  np.uint8)
        self.move_dst            = _ds("move_dst",            (max_size, MAX_LEGAL_MOVES),  np.uint8)
        self.move_probs          = _ds("move_probs",          (max_size, MAX_LEGAL_MOVES),  np.float32)
        self.num_moves           = _ds("num_moves",           (max_size,),                  np.int32)
        self.value_targets       = _ds("value_targets",       (max_size,),                  np.float32)
        self.root_q_targets      = _ds("root_q_targets",      (max_size,),                  np.float32)
        self.value_only          = _ds("value_only",          (max_size,),                  np.bool_)
        self.policy_only         = _ds("policy_only",         (max_size,),                  np.bool_)
        self.aux_targets         = _ds("aux_targets",         (max_size, 6),                np.float32)

        if self._h5file is not None:
            self._h5file.flush()

    def close(self):
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None

    def __del__(self):
        self.close()

    # ---- multi-process read lifecycle (unchanged shape from CNN era) -------

    def _attach_h5_datasets(self):
        f = self._h5file
        for name in _H5_DATASETS:
            setattr(self, name, f[name])

    def _open_for_read(self):
        if self._h5path is None:
            return
        import h5py
        self._h5file = h5py.File(self._h5path, "r")
        self._attach_h5_datasets()

    def _reopen_h5(self, mode: str):
        import h5py
        if self._h5file is not None:
            self._h5file.flush()
            self._h5file.close()
            self._h5file = None
        self._h5file = h5py.File(self._h5path, mode)
        self._attach_h5_datasets()

    @contextlib.contextmanager
    def read_only_mode(self):
        if self._h5path is None:
            yield
            return
        self._reopen_h5("r")
        try:
            yield
        finally:
            self._reopen_h5("r+")

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._h5path is not None:
            state["_h5file"] = None
            for name in _H5_DATASETS:
                state[name] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # ---- write API ---------------------------------------------------------

    def add_sample(
        self,
        tokens_categoricals: np.ndarray,  # (L, 5) uint8
        tokens_positions: np.ndarray,      # (L, 2) int8
        tokens_flags: np.ndarray,          # (L, F) float
        tokens_mask: np.ndarray,           # (L,)  bool
        move_src: np.ndarray,              # (K,) uint8  — mover token slot per legal move
        move_dst: np.ndarray,              # (K,) uint8  — dest token slot
        move_probs: np.ndarray,            # (K,) float32
        value_target: float,
        value_only: bool = False, policy_only: bool = False,
        aux_targets: np.ndarray | None = None,
        root_q_target: float | None = None,
    ):
        idx = self._count % self.max_size
        self.tokens_categoricals[idx] = tokens_categoricals
        self.tokens_positions[idx] = tokens_positions
        self.tokens_flags[idx] = tokens_flags.astype(np.float16, copy=False)
        self.tokens_mask[idx] = tokens_mask

        nm = min(len(move_src), MAX_LEGAL_MOVES)
        self.move_src[idx, :nm] = move_src[:nm]
        self.move_dst[idx, :nm] = move_dst[:nm]
        self.move_probs[idx, :nm] = move_probs[:nm]
        if nm < MAX_LEGAL_MOVES:
            self.move_src[idx, nm:] = 0
            self.move_dst[idx, nm:] = 0
            self.move_probs[idx, nm:] = 0.0
        self.num_moves[idx] = nm

        self.value_targets[idx] = value_target
        self.root_q_targets[idx] = (
            root_q_target if root_q_target is not None else value_target
        )
        self.value_only[idx] = value_only
        self.policy_only[idx] = policy_only
        self.aux_targets[idx] = (
            np.asarray(aux_targets, dtype=np.float32)
            if aux_targets is not None
            else np.zeros(6, dtype=np.float32)
        )
        self._count += 1
        self._size = min(self._size + 1, self.max_size)

        if self._h5file is not None:
            self._h5file.attrs["count"] = self._count
            self._h5file.attrs["size"] = self._size

    def add_batch(
        self,
        tokens_categoricals: np.ndarray,  # (N, L, 5)
        tokens_positions: np.ndarray,      # (N, L, 2)
        tokens_flags: np.ndarray,          # (N, L, F)
        tokens_mask: np.ndarray,           # (N, L)
        move_src: np.ndarray,              # (N, K)
        move_dst: np.ndarray,              # (N, K)
        move_probs: np.ndarray,            # (N, K)
        num_moves: np.ndarray,             # (N,) int32
        value_targets: np.ndarray,         # (N,)
        value_only: np.ndarray,            # (N,) bool
        policy_only: np.ndarray,           # (N,) bool
        aux_targets: np.ndarray,           # (N, 6)
        root_q_targets: np.ndarray | None = None,
    ):
        """Bulk insert from contiguous arrays — the Rust self-play side will
        call this via the Phase F selfplay-result handoff."""
        n = tokens_categoricals.shape[0]
        rq = (
            np.asarray(root_q_targets, dtype=np.float32)
            if root_q_targets is not None
            else np.asarray(value_targets, dtype=np.float32)
        )
        pairs = [
            (self.tokens_categoricals, tokens_categoricals.astype(np.uint8, copy=False)),
            (self.tokens_positions,    tokens_positions.astype(np.int8,  copy=False)),
            (self.tokens_flags,        tokens_flags.astype(np.float16,   copy=False)),
            (self.tokens_mask,         tokens_mask.astype(np.bool_,      copy=False)),
            (self.move_src,            move_src.astype(np.uint8,         copy=False)),
            (self.move_dst,            move_dst.astype(np.uint8,         copy=False)),
            (self.move_probs,          move_probs.astype(np.float32,     copy=False)),
            (self.num_moves,           num_moves.astype(np.int32,        copy=False)),
            (self.value_targets,       value_targets.astype(np.float32,  copy=False)),
            (self.root_q_targets,      rq),
            (self.value_only,          np.asarray(value_only,  dtype=np.bool_)),
            (self.policy_only,         np.asarray(policy_only, dtype=np.bool_)),
            (self.aux_targets,         aux_targets.astype(np.float32,    copy=False)),
        ]

        start = self._count % self.max_size
        end = start + n
        if end <= self.max_size:
            for arr, data in pairs:
                arr[start:end] = data
        else:
            first = self.max_size - start
            for arr, data in pairs:
                arr[start:] = data[:first]
                arr[:n - first] = data[first:]

        self._count += n
        self._size = min(self._size + n, self.max_size)
        if self._h5file is not None:
            self._h5file.attrs["count"] = self._count
            self._h5file.attrs["size"] = self._size
            self._h5file.flush()

    def clear(self):
        self._count = 0
        self._size = 0
        if self._h5file is not None:
            self._h5file.attrs["count"] = 0
            self._h5file.attrs["size"] = 0
            self._h5file.flush()

    def __len__(self):
        return self._size * 12 if self.augment_symmetry else self._size

    @property
    def raw_size(self) -> int:
        return self._size

    # ---- read API ---------------------------------------------------------

    def __getitem__(self, idx):
        """Returns a per-sample tuple plus `sym_idx` ∈ [0, 12). The D6
        position permutation is applied on GPU in `Trainer._apply_sym_gpu`
        — `__getitem__` only carries the sym index through."""
        if self.augment_symmetry:
            sym = idx % 12
            base_idx = idx // 12
        else:
            sym = 0
            base_idx = idx

        cat = self.tokens_categoricals[base_idx]
        pos = self.tokens_positions[base_idx]
        flg = self.tokens_flags[base_idx]
        msk = self.tokens_mask[base_idx]
        if self._h5file is None:
            cat = cat.copy(); pos = pos.copy(); flg = flg.copy(); msk = msk.copy()

        return (
            torch.from_numpy(cat),
            torch.from_numpy(pos),
            torch.from_numpy(flg.astype(np.float32)),  # cast at load time so model.forward gets f32
            torch.from_numpy(msk),
            torch.from_numpy(self.move_src[base_idx].astype(np.int64)),
            torch.from_numpy(self.move_dst[base_idx].astype(np.int64)),
            torch.from_numpy(self.move_probs[base_idx].copy()),
            torch.tensor(int(self.num_moves[base_idx]), dtype=torch.int32),
            torch.tensor(float(self.value_targets[base_idx]), dtype=torch.float32),
            torch.tensor(float(self.root_q_targets[base_idx]), dtype=torch.float32),
            torch.tensor(bool(self.value_only[base_idx]), dtype=torch.bool),
            torch.tensor(bool(self.policy_only[base_idx]), dtype=torch.bool),
            torch.from_numpy(self.aux_targets[base_idx].astype(np.float32, copy=False)),
            torch.tensor(int(sym), dtype=torch.int64),
        )

    def __getitems__(self, indices):
        """Batched fetch: one h5py read per dataset rather than per sample.
        ~20-50× fetch speedup on h5-backed datasets."""
        n = len(indices)
        idx_arr = np.asarray(indices, dtype=np.int64)
        if self.augment_symmetry:
            base = idx_arr // 12
            syms = idx_arr % 12
        else:
            base = idx_arr
            syms = np.zeros(n, dtype=np.int64)

        if self._h5file is not None:
            unique, inverse = np.unique(base, return_inverse=True)
            sel = unique.tolist()
            def read(ds):
                return ds[sel][inverse]
        else:
            def read(ds):
                return ds[base]

        cat = read(self.tokens_categoricals)
        pos = read(self.tokens_positions)
        flg = read(self.tokens_flags).astype(np.float32, copy=False)
        msk = read(self.tokens_mask)
        msrc = read(self.move_src).astype(np.int64, copy=False)
        mdst = read(self.move_dst).astype(np.int64, copy=False)
        mprob = read(self.move_probs)
        nmov = read(self.num_moves)
        vtar = read(self.value_targets)
        rqtar = read(self.root_q_targets)
        vo = read(self.value_only)
        po = read(self.policy_only)
        aux = read(self.aux_targets).astype(np.float32, copy=False)

        out = []
        for i in range(n):
            out.append((
                torch.from_numpy(cat[i]),
                torch.from_numpy(pos[i]),
                torch.from_numpy(flg[i]),
                torch.from_numpy(msk[i]),
                torch.from_numpy(msrc[i]),
                torch.from_numpy(mdst[i]),
                torch.from_numpy(mprob[i]),
                torch.tensor(int(nmov[i]), dtype=torch.int32),
                torch.tensor(float(vtar[i]), dtype=torch.float32),
                torch.tensor(float(rqtar[i]), dtype=torch.float32),
                torch.tensor(bool(vo[i]), dtype=torch.bool),
                torch.tensor(bool(po[i]), dtype=torch.bool),
                torch.from_numpy(aux[i]),
                torch.tensor(int(syms[i]), dtype=torch.int64),
            ))
        return out


class Trainer:
    """Trains HiveTransformer on token-pair policy targets."""

    def __init__(
        self,
        model: Optional[HiveTransformer] = None,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        lr: float = 0.02,
        optimizer: str = "sgd",
        grad_clip: float | None = 1.0,
        warmup_steps: int = 0,
    ):
        self.device = torch.device(device)
        self.model = model or create_model()
        self.model.to(self.device)
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self._base_lr = lr
        self._step_count = 0
        kind = optimizer.lower()
        if kind == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr,
                momentum=0.9, weight_decay=weight_decay,
            )
        elif kind == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"unknown optimizer {optimizer!r}")
        self._compiled = (
            torch.compile(self.model, dynamic=False)
            if self.device.type == "cuda"
            else self.model
        )
        self._sym_mats_gpu: Optional[torch.Tensor] = None

    @property
    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _init_sym_buffers(self):
        """Move the 12 D6 axial transforms to GPU. One-time on first batch."""
        if self._sym_mats_gpu is not None:
            return
        mats = _load_axial_transforms()  # (12, 2, 2) int8
        self._sym_mats_gpu = torch.from_numpy(mats).to(self.device).to(torch.int64)

    def _apply_sym_gpu(self, positions: torch.Tensor, sym_idx: torch.Tensor) -> torch.Tensor:
        """Permute per-sample token positions through D6.

        `positions`:  (B, L, 2) int64 axial coordinates
        `sym_idx`:    (B,)      int64 in [0, 12)
        Returns:      (B, L, 2) int64 transformed positions.

        Token *identity* and the policy target are unchanged — only the
        spatial coordinates flow through the symmetry. This is materially
        simpler than the CNN era where cell permutations had to remap
        per-cell channels and per-cell policy targets.
        """
        self._init_sym_buffers()
        mats = self._sym_mats_gpu[sym_idx]      # (B, 2, 2)
        # positions: (B, L, 2). Apply mat per sample:
        #   new_q = mat[0, 0]*q + mat[0, 1]*r
        #   new_r = mat[1, 0]*q + mat[1, 1]*r
        # einsum: "bij,blj->bli"
        return torch.einsum("bij,blj->bli", mats, positions)

    def _compute_batch_losses(
        self, batch, value_loss_scale: float, aux_loss_scale: float
    ):
        (cat, pos, flg, msk,
         msrc, mdst, mprob, nmov,
         value_target, root_q, vo_mask, po_mask, aux_target, sym_idx) = batch

        device = self.device
        device_type = self.device.type
        nb = device_type == "cuda"

        cat = cat.to(device, non_blocking=nb).long()
        pos = pos.to(device, non_blocking=nb).long()
        flg = flg.to(device, non_blocking=nb)
        msk = msk.to(device, non_blocking=nb)
        msrc = msrc.to(device, non_blocking=nb)
        mdst = mdst.to(device, non_blocking=nb)
        mprob = mprob.to(device, non_blocking=nb)
        nmov = nmov.to(device, non_blocking=nb)
        value_target = value_target.to(device, non_blocking=nb)
        vo_mask = vo_mask.to(device, non_blocking=nb)
        po_mask = po_mask.to(device, non_blocking=nb)
        aux_target = aux_target.to(device, non_blocking=nb)
        sym_idx = sym_idx.to(device, non_blocking=nb)

        # D6 augmentation: permute per-token positions in-place.
        pos = self._apply_sym_gpu(pos, sym_idx)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            policy_flat, wdl_logits, aux = self._compiled(cat, pos, flg, msk)

        B = cat.size(0)
        L = SEQ_LEN
        D = BILINEAR_DIM

        # policy_flat: (B, 2*L*D) = [Q-section (L*D, D-major) || K-section (L*D, D-major)]
        # Reshape to (B, D, L) per section; transpose to (B, L, D) for indexed gather.
        q = policy_flat[:, : L * D].view(B, D, L).transpose(1, 2)   # (B, L, D)
        k = policy_flat[:, L * D :].view(B, D, L).transpose(1, 2)   # (B, L, D)

        # Gather Q[mover] and K[dest] for every legal-move slot.
        src_exp = msrc.unsqueeze(-1).expand(-1, -1, D)               # (B, K, D)
        dst_exp = mdst.unsqueeze(-1).expand(-1, -1, D)
        q_g = torch.gather(q, 1, src_exp)
        k_g = torch.gather(k, 1, dst_exp)
        m_logits = (q_g * k_g).sum(-1) / math.sqrt(D)                # (B, K)

        # Padding mask: only the first nmov[i] slots per sample are real.
        legal_mask = (
            torch.arange(msrc.size(1), device=device).unsqueeze(0)
            < nmov.unsqueeze(1).long()
        )
        masked_logits = m_logits.masked_fill(~legal_mask, float("-inf"))
        log_probs = torch.log_softmax(masked_logits.float(), dim=1)
        log_probs = log_probs.masked_fill(~legal_mask, 0.0)

        per_sample_policy = -(mprob * log_probs).sum(dim=1)
        has_target = nmov > 0
        per_sample_policy = per_sample_policy * has_target.float()

        policy_weight = (~vo_mask).float()
        policy_loss = (
            (per_sample_policy * policy_weight).sum()
            / policy_weight.sum().clamp(min=1.0)
        )

        # Value: 3-way WDL KL against soft target.
        wdl_target = _scalar_to_wdl(value_target)
        log_wdl = torch.log_softmax(wdl_logits.float(), dim=1)
        per_sample_value = -(wdl_target * log_wdl).sum(dim=1)
        value_weight = (~po_mask).float()
        value_loss = (
            (per_sample_value * value_weight).sum()
            / value_weight.sum().clamp(min=1.0)
        )

        # Aux: per-component MSE (queen danger, escape, mobility — 2 sides each).
        aux_mse = (aux - aux_target) ** 2
        qd_loss = aux_mse[:, 0:2].mean()
        qe_loss = aux_mse[:, 2:4].mean()
        mob_loss = aux_mse[:, 4:6].mean()
        aux_loss = qd_loss + qe_loss + mob_loss

        total = policy_loss + value_loss_scale * value_loss + aux_loss_scale * aux_loss

        # Diagnostics: top-1 accuracy + uniform-CE baseline.
        pol_active = has_target & (~vo_mask)
        argmax_pred = masked_logits.argmax(dim=1)
        argmax_true = mprob.argmax(dim=1)
        top1_hits = ((argmax_pred == argmax_true) & pol_active).sum()
        n_legal = legal_mask.sum(dim=1).clamp(min=1).float()
        uniform_ce_sum = (torch.log(n_legal) * pol_active.float()).sum()
        n_pol_active = pol_active.sum()

        return {
            "total": total, "policy": policy_loss, "value": value_loss,
            "aux": aux_loss, "qd": qd_loss, "qe": qe_loss, "mob": mob_loss,
            "top1_hits": top1_hits, "uniform_ce_sum": uniform_ce_sum,
            "n_pol_active": n_pol_active,
        }

    def train_epoch(
        self,
        dataset: HiveDataset,
        batch_size: int = 64,
        value_loss_scale: float = 1.0,
        aux_loss_scale: float = 1.0,
        subsample_frac: float = 1.0,
        num_workers: int = 0,
    ) -> dict:
        self.model.train()
        drop = self.device.type == "cuda"
        loader_kwargs = self._loader_kwargs(num_workers)
        if subsample_frac < 1.0:
            from torch.utils.data import RandomSampler
            n = max(1, int(len(dataset) * subsample_frac))
            sampler = RandomSampler(dataset, replacement=False, num_samples=n)
            loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                drop_last=drop, **loader_kwargs)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                drop_last=drop, **loader_kwargs)

        sums = {"total": 0.0, "policy": 0.0, "value": 0.0,
                "aux": 0.0, "qd": 0.0, "qe": 0.0, "mob": 0.0}
        num_batches = 0

        for batch in tqdm(loader, desc="  Training", leave=False, unit="batch"):
            losses = self._compute_batch_losses(batch, value_loss_scale, aux_loss_scale)
            loss = losses["total"]

            if self.warmup_steps > 0 and self._step_count < self.warmup_steps:
                scale = (self._step_count + 1) / self.warmup_steps
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self._base_lr * scale
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self._step_count += 1

            for k in sums:
                sums[k] += losses[k].item()
            num_batches += 1

        if num_batches == 0:
            return {k + "_loss": 0.0 for k in sums} | {"total_loss": 0.0}
        return {
            "policy_loss": sums["policy"] / num_batches,
            "value_loss": sums["value"] / num_batches,
            "aux_loss": sums["aux"] / num_batches,
            "qd_loss": sums["qd"] / num_batches,
            "qe_loss": sums["qe"] / num_batches,
            "mob_loss": sums["mob"] / num_batches,
            "total_loss": sums["total"] / num_batches,
        }

    def _loader_kwargs(self, num_workers: int) -> dict:
        kw: dict = {"num_workers": num_workers}
        if self.device.type == "cuda":
            kw["pin_memory"] = True
        if num_workers > 0:
            kw["worker_init_fn"] = _h5_worker_init_fn
        return kw

    @torch.no_grad()
    def validate(
        self,
        dataset: HiveDataset,
        batch_size: int = 64,
        value_loss_scale: float = 1.0,
        aux_loss_scale: float = 1.0,
        num_workers: int = 0,
    ) -> dict:
        self.model.eval()
        drop = self.device.type == "cuda"
        loader_kwargs = self._loader_kwargs(num_workers)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            drop_last=drop, **loader_kwargs)

        sums = {"total": 0.0, "policy": 0.0, "value": 0.0,
                "aux": 0.0, "qd": 0.0, "qe": 0.0, "mob": 0.0}
        top1_hits = 0
        uniform_ce_sum = 0.0
        n_pol_active = 0
        num_batches = 0

        for batch in tqdm(loader, desc="  Validating", leave=False, unit="batch"):
            losses = self._compute_batch_losses(batch, value_loss_scale, aux_loss_scale)
            for k in sums:
                sums[k] += losses[k].item()
            top1_hits += int(losses["top1_hits"].item())
            uniform_ce_sum += float(losses["uniform_ce_sum"].item())
            n_pol_active += int(losses["n_pol_active"].item())
            num_batches += 1

        if num_batches == 0:
            return {k + "_loss": 0.0 for k in sums} | {"total_loss": 0.0,
                                                       "top1_acc": 0.0,
                                                       "uniform_ce": 0.0}
        denom = max(n_pol_active, 1)
        return {
            "policy_loss": sums["policy"] / num_batches,
            "value_loss": sums["value"] / num_batches,
            "aux_loss": sums["aux"] / num_batches,
            "qd_loss": sums["qd"] / num_batches,
            "qe_loss": sums["qe"] / num_batches,
            "mob_loss": sums["mob"] / num_batches,
            "total_loss": sums["total"] / num_batches,
            "top1_acc": top1_hits / denom,
            "uniform_ce": uniform_ce_sum / denom,
        }
