"""Training loop for the Hive neural network.

The policy target is a **joint** distribution over every legal move at a
position — placements and movements share one softmax, matching what MCTS
consumes at inference. See docs/policy_heads.md.

Both heads are trained **masked**: the softmax denominator covers only the
legal placement and movement logits at this position. An earlier experiment
unmasked the placement head (illegal placements in the denom as a 0-target
suppression signal) but caused legal placement priors to undershoot their
MCTS targets by 6-9pp and drove the decisive-game rate down — see
docs/hive_ideas.md (#placement-policy-masking).
"""

from __future__ import annotations
import math
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional
from tqdm import tqdm

from .model import HiveNet, create_model
from ..encoding.board_encoder import NUM_CHANNELS, DEFAULT_GRID_SIZE, RESERVE_SIZE
from ..encoding.move_encoder import NUM_PLACE_CHANNELS
from shared.replay_buffer import handle_buffer_size_mismatch

# Per-sample caps on the sparse legal-action target. Must match the Rust side
# (hive_selfplay.rs training_data()).
MAX_PLACEMENTS = 128
MAX_MOVE_PAIRS = 256


def _scalar_to_wdl(v: torch.Tensor) -> torch.Tensor:
    """Convert scalar value targets in [-1, 1] to (W, D, L) soft targets summing to 1."""
    w = v.clamp(min=0)
    l = (-v).clamp(min=0)
    d = 1.0 - w - l
    return torch.stack([w, d, l], dim=1)

# ---------------------------------------------------------------------------
# Hex D6 symmetry: lazy-loaded per grid_size.
# ---------------------------------------------------------------------------
_SYM_PERMS_CACHE: dict[int, list] = {}
_BAD_CELLS_CACHE: dict[int, np.ndarray] = {}

def _load_sym_perms(grid_size: int):
    if grid_size not in _SYM_PERMS_CACHE:
        from engine_zero import d6_grid_permutations
        _SYM_PERMS_CACHE[grid_size] = [np.array(p) for p in d6_grid_permutations(grid_size)]
    return _SYM_PERMS_CACHE[grid_size]

def _load_bad_cells(grid_size: int) -> np.ndarray:
    """Flat mask of cells where |s|=|q+r| > half_grid — these clip under non-{0°,180°} D6 rotations."""
    if grid_size not in _BAD_CELLS_CACHE:
        half = grid_size // 2
        rows, cols = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
        _BAD_CELLS_CACHE[grid_size] = (np.abs(2 * half - rows - cols) > half).ravel()
    return _BAD_CELLS_CACHE[grid_size]


class HiveDataset(Dataset):
    """Replay buffer holding sparse joint (placement + movement) policy targets.

    Per sample we store the full legal-action list so the training softmax has
    the same denominator MCTS uses at inference — placements and movements
    are NOT trained as two independent heads.

    Placement target: flat indices into [0, 5*G²) (type*G² + cell).
    Movement target : (src_cell, dst_cell) pairs with dst computed via Q[src]·K[dst].
    Probabilities sum to 1 across the union of both per sample.
    """

    def __init__(self, max_size: int = 50_000, grid_size: int = DEFAULT_GRID_SIZE,
                 buf_dir: str | None = None):
        self.max_size = max_size
        self.grid_size = grid_size
        self._h5file = None
        self.augment_symmetry = False

        def _zeros(shape, dtype):
            return np.zeros(shape, dtype=dtype)

        if buf_dir is not None:
            import h5py
            os.makedirs(buf_dir, exist_ok=True)
            h5path = os.path.join(buf_dir, "replay.h5")
            resuming = os.path.exists(h5path)
            self._h5file = h5py.File(h5path, "r+" if resuming else "w")

            if resuming:
                stored_max = int(self._h5file.attrs["max_size"])
                stored_gs = int(self._h5file.attrs.get("grid_size", grid_size))
                if stored_max != max_size:
                    max_size, self._h5file = handle_buffer_size_mismatch(
                        self._h5file, h5path, max_size
                    )
                    self.max_size = max_size
                if stored_gs != grid_size:
                    raise ValueError(
                        f"Buffer grid_size mismatch: stored {stored_gs} vs requested {grid_size}. "
                        f"Delete {h5path} to start fresh."
                    )
                self._count = int(self._h5file.attrs["count"])
                self._size = int(self._h5file.attrs["size"])
                print(f"  Replay buffer resumed: {self._size}/{max_size} samples from {h5path}")
            else:
                self._count = 0
                self._size = 0
                self._h5file.attrs["max_size"] = max_size
                self._h5file.attrs["grid_size"] = grid_size
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

        self.board_tensors   = _ds("board_tensors",   (max_size, NUM_CHANNELS, grid_size, grid_size), np.float32)
        self.reserve_vectors = _ds("reserve_vectors", (max_size, RESERVE_SIZE),                       np.float32)
        self.place_idx       = _ds("place_idx",       (max_size, MAX_PLACEMENTS),                     np.uint16)
        self.place_probs     = _ds("place_probs",     (max_size, MAX_PLACEMENTS),                     np.float32)
        self.num_placements  = _ds("num_placements",  (max_size,),                                    np.int32)
        self.movement_src    = _ds("movement_src",    (max_size, MAX_MOVE_PAIRS),                     np.uint16)
        self.movement_dst    = _ds("movement_dst",    (max_size, MAX_MOVE_PAIRS),                     np.uint16)
        self.movement_probs  = _ds("movement_probs",  (max_size, MAX_MOVE_PAIRS),                     np.float32)
        self.num_movements   = _ds("num_movements",   (max_size,),                                    np.int32)
        self.value_targets   = _ds("value_targets",   (max_size,),                                    np.float32)
        # MCTS root W−L (no contempt) per sample, retained for diagnostics
        # and inspection scripts. Q-mix on timeouts is now baked into
        # `value_targets` Rust-side (see TimeoutTarget::QMix in search.rs),
        # so this dataset is no longer read by the loss path.
        self.root_q_targets  = _ds("root_q_targets",  (max_size,),                                    np.float32)
        self.value_only      = _ds("value_only",      (max_size,),                                    np.bool_)
        self.policy_only     = _ds("policy_only",     (max_size,),                                    np.bool_)
        self.my_queen_danger  = _ds("my_queen_danger",  (max_size,), np.float32)
        self.opp_queen_danger = _ds("opp_queen_danger", (max_size,), np.float32)
        self.my_queen_escape  = _ds("my_queen_escape",  (max_size,), np.float32)
        self.opp_queen_escape = _ds("opp_queen_escape", (max_size,), np.float32)
        self.my_mobility      = _ds("my_mobility",      (max_size,), np.float32)
        self.opp_mobility     = _ds("opp_mobility",     (max_size,), np.float32)

        if self._h5file is not None:
            if "root_q_targets_initialized" not in self._h5file.attrs and self._size > 0:
                self.root_q_targets[: self._size] = self.value_targets[: self._size]
            self._h5file.attrs["root_q_targets_initialized"] = True
            self._h5file.flush()

    def close(self):
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None

    def __del__(self):
        self.close()

    def add_sample(self, board_tensor: np.ndarray, reserve_vector: np.ndarray,
                   place_idx_arr,   # list/array of int, length <= MAX_PLACEMENTS
                   place_probs_arr, # list/array of float
                   movement_src_cells,  # list/array of int, length <= MAX_MOVE_PAIRS
                   movement_dst_cells,
                   movement_probs_arr,
                   value_target: float,
                   value_only: bool = False, policy_only: bool = False,
                   my_queen_danger: float = 0.0, opp_queen_danger: float = 0.0,
                   my_queen_escape: float = 0.0, opp_queen_escape: float = 0.0,
                   my_mobility: float = 0.0, opp_mobility: float = 0.0):
        idx = self._count % self.max_size
        self.board_tensors[idx] = board_tensor
        self.reserve_vectors[idx] = reserve_vector

        np_ = min(len(place_idx_arr), MAX_PLACEMENTS)
        self.place_idx[idx, :np_] = place_idx_arr[:np_]
        self.place_probs[idx, :np_] = place_probs_arr[:np_]
        if np_ < MAX_PLACEMENTS:
            self.place_idx[idx, np_:] = 0
            self.place_probs[idx, np_:] = 0.0
        self.num_placements[idx] = np_

        nm = min(len(movement_src_cells), MAX_MOVE_PAIRS)
        self.movement_src[idx, :nm] = movement_src_cells[:nm]
        self.movement_dst[idx, :nm] = movement_dst_cells[:nm]
        self.movement_probs[idx, :nm] = movement_probs_arr[:nm]
        if nm < MAX_MOVE_PAIRS:
            self.movement_src[idx, nm:] = 0
            self.movement_dst[idx, nm:] = 0
            self.movement_probs[idx, nm:] = 0.0
        self.num_movements[idx] = nm

        self.value_targets[idx] = value_target
        # No MCTS root available in single-sample API; fall back to z so
        # q-mix is a no-op for these samples.
        self.root_q_targets[idx] = value_target
        self.value_only[idx] = value_only
        self.policy_only[idx] = policy_only
        self.my_queen_danger[idx] = my_queen_danger
        self.opp_queen_danger[idx] = opp_queen_danger
        self.my_queen_escape[idx] = my_queen_escape
        self.opp_queen_escape[idx] = opp_queen_escape
        self.my_mobility[idx] = my_mobility
        self.opp_mobility[idx] = opp_mobility
        self._count += 1
        self._size = min(self._size + 1, self.max_size)

    def add_batch(self, board_tensors: np.ndarray, reserve_vectors: np.ndarray,
                  place_idx_data: np.ndarray,   # (N, MAX_PLACEMENTS) uint16
                  place_prob_data: np.ndarray,  # (N, MAX_PLACEMENTS) float32
                  num_placements_arr: np.ndarray,  # (N,) int32
                  movement_src_data: np.ndarray,   # (N, MAX_MOVE_PAIRS) uint16
                  movement_dst_data: np.ndarray,   # (N, MAX_MOVE_PAIRS) uint16
                  movement_prob_data: np.ndarray,  # (N, MAX_MOVE_PAIRS) float32
                  num_movements_arr: np.ndarray,   # (N,) int32
                  value_targets: np.ndarray,
                  value_only: list[bool], policy_only: list[bool],
                  my_queen_danger: np.ndarray | None = None,
                  opp_queen_danger: np.ndarray | None = None,
                  my_queen_escape: np.ndarray | None = None,
                  opp_queen_escape: np.ndarray | None = None,
                  my_mobility: np.ndarray | None = None,
                  opp_mobility: np.ndarray | None = None,
                  root_q_targets: np.ndarray | None = None):
        """Bulk insert from contiguous arrays."""
        n = board_tensors.shape[0]
        boards = board_tensors.reshape(n, NUM_CHANNELS, self.grid_size, self.grid_size)
        vo_arr  = np.asarray(value_only,  dtype=np.bool_)
        po_arr  = np.asarray(policy_only, dtype=np.bool_)
        _z = lambda: np.zeros(n, dtype=np.float32)
        mqd  = my_queen_danger  if my_queen_danger  is not None else _z()
        oqd  = opp_queen_danger if opp_queen_danger is not None else _z()
        mqe  = my_queen_escape  if my_queen_escape  is not None else _z()
        oqe  = opp_queen_escape if opp_queen_escape is not None else _z()
        mmob = my_mobility      if my_mobility      is not None else _z()
        omob = opp_mobility     if opp_mobility     is not None else _z()
        # Fall back to value_targets when caller doesn't supply root_q (e.g.
        # supervised pretrain), so q-mix becomes a no-op for those samples.
        rq = (
            np.asarray(root_q_targets, dtype=np.float32)
            if root_q_targets is not None
            else np.asarray(value_targets, dtype=np.float32)
        )

        pairs = [
            (self.board_tensors,   boards),
            (self.reserve_vectors, reserve_vectors),
            (self.place_idx,       place_idx_data),
            (self.place_probs,     place_prob_data),
            (self.num_placements,  num_placements_arr),
            (self.movement_src,    movement_src_data),
            (self.movement_dst,    movement_dst_data),
            (self.movement_probs,  movement_prob_data),
            (self.num_movements,   num_movements_arr),
            (self.value_targets,   value_targets),
            (self.root_q_targets,  rq),
            (self.value_only,      vo_arr),
            (self.policy_only,     po_arr),
            (self.my_queen_danger,  mqd),
            (self.opp_queen_danger, oqd),
            (self.my_queen_escape,  mqe),
            (self.opp_queen_escape, oqe),
            (self.my_mobility,      mmob),
            (self.opp_mobility,     omob),
        ]

        start = self._count % self.max_size
        end   = start + n
        if end <= self.max_size:
            for arr, data in pairs:
                arr[start:end] = data
        else:
            first = self.max_size - start
            for arr, data in pairs:
                arr[start:]     = data[:first]
                arr[:n - first] = data[first:]

        self._count += n
        self._size = min(self._size + n, self.max_size)

        if self._h5file is not None:
            self._h5file.attrs["count"] = self._count
            self._h5file.attrs["size"]  = self._size
            self._h5file.flush()

    def clear(self):
        self._count = 0
        self._size = 0
        if self._h5file is not None:
            self._h5file.attrs["count"] = 0
            self._h5file.attrs["size"]  = 0
            self._h5file.flush()

    def __len__(self):
        return self._size * 12 if self.augment_symmetry else self._size

    @property
    def raw_size(self) -> int:
        return self._size

    def __getitem__(self, idx):
        """Returns the raw position plus a `sym_idx` integer in [0, 12).

        The D6 permutation is applied on GPU in Trainer._apply_sym_gpu after
        the batch lands on device — moves the per-sample numpy gather/filter/
        renormalize work off the main thread, which is the data-loading
        bottleneck on Windows where multi-worker DataLoaders are unreliable.
        """
        if self.augment_symmetry:
            sym = idx % 12
            base_idx = idx // 12
        else:
            sym = 0
            base_idx = idx

        board = self.board_tensors[base_idx]
        # h5py slicing materializes a fresh array; for the in-memory backing
        # store .copy() is needed to detach from the underlying buffer.
        if self._h5file is None:
            board = board.copy()

        aux_targets = np.array([
            self.my_queen_danger[base_idx], self.opp_queen_danger[base_idx],
            self.my_queen_escape[base_idx], self.opp_queen_escape[base_idx],
            self.my_mobility[base_idx], self.opp_mobility[base_idx],
        ], dtype=np.float32)

        return (
            torch.from_numpy(board),
            torch.from_numpy(self.reserve_vectors[base_idx].copy()),
            torch.from_numpy(self.place_idx[base_idx].astype(np.int64)),
            torch.from_numpy(self.place_probs[base_idx].copy()),
            torch.tensor(int(self.num_placements[base_idx]), dtype=torch.int32),
            torch.from_numpy(self.movement_src[base_idx].astype(np.int64)),
            torch.from_numpy(self.movement_dst[base_idx].astype(np.int64)),
            torch.from_numpy(self.movement_probs[base_idx].copy()),
            torch.tensor(int(self.num_movements[base_idx]), dtype=torch.int32),
            torch.tensor(self.value_targets[base_idx], dtype=torch.float32),
            torch.tensor(self.root_q_targets[base_idx], dtype=torch.float32),
            torch.tensor(self.value_only[base_idx], dtype=torch.bool),
            torch.tensor(self.policy_only[base_idx], dtype=torch.bool),
            torch.from_numpy(aux_targets),
            torch.tensor(sym, dtype=torch.int64),
        )


class Trainer:
    """Trains HiveNet on joint-softmax policy targets."""

    def __init__(self, model: Optional[HiveNet] = None,
                 weight_decay: float = 1e-4, device: str = "cpu",
                 lr: float = 0.02, optimizer: str = "sgd",
                 grad_clip: float | None = 1.0,
                 warmup_steps: int = 0):
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
            raise ValueError(f"unknown optimizer {optimizer!r}; expected 'sgd' or 'adamw'")
        self._compiled = torch.compile(self.model, dynamic=False) if self.device.type == "cuda" else self.model
        self._sym_perms_gpu: Optional[torch.Tensor] = None
        self._bad_cells_gpu: Optional[torch.Tensor] = None
        self._sym_buffers_grid_size: Optional[int] = None

    @property
    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def _init_sym_buffers(self, grid_size: int):
        """Move the 12 D6 cell permutations and the corner-clip mask to GPU.

        Lazy: keyed on grid_size, rebuilt only when the model's grid changes.
        Each permutation entry is in [0, gs²]; the value gs² signals 'out of
        bounds' and is used as a gather sentinel into a zero-padded buffer.
        """
        if (self._sym_perms_gpu is not None
                and self._sym_buffers_grid_size == grid_size):
            return
        perms = _load_sym_perms(grid_size)  # list of 12 np.int arrays, shape (gs²,)
        stacked = np.stack(perms).astype(np.int64)
        self._sym_perms_gpu = torch.from_numpy(stacked).to(self.device)
        bad = _load_bad_cells(grid_size)
        self._bad_cells_gpu = torch.from_numpy(bad.astype(np.bool_)).to(self.device)
        self._sym_buffers_grid_size = grid_size

    def _apply_sym_gpu(self, board, p_idx, p_prob, n_p,
                       m_src, m_dst, m_prob, n_m, sym_idx):
        """Apply per-sample D6 augmentation on GPU.

        Inputs are already on `self.device`. `sym_idx` is (B,) in [0, 12).
        Mirrors the CPU augmentation that used to live in HiveDataset.__getitem__:
        permutes board cells, decodes/permutes placement indices, permutes
        movement (src, dst) pairs, drops slots whose permuted cell falls in
        the corner-clip zone, and renormalizes the joint probability target
        across surviving placements + movements.

        Returns the augmented tensors plus combined (n_p/n_m × aug-validity)
        boolean masks for placements and movements. Callers must use those
        masks in place of recomputing from arange/n_p.
        """
        B, C = board.size(0), board.size(1)
        gs = board.size(-1)
        gs2 = gs * gs
        self._init_sym_buffers(gs)
        sym_perms = self._sym_perms_gpu  # (12, gs²)
        bad_mask = self._bad_cells_gpu   # (gs²,) bool

        # Bad-cell override: any piece in the corner-clip zone forces sym → 0
        # for rotations/mirrors that would clip (everything except 0° and 180°).
        piece_present = (board.view(B, C, gs2)[:, :10, :] > 0).any(dim=1)  # (B, gs²)
        has_corner = (piece_present & bad_mask.unsqueeze(0)).any(dim=1)    # (B,)
        unsafe_sym = (sym_idx != 0) & (sym_idx != 3)
        sym_idx = torch.where(has_corner & unsafe_sym,
                              torch.zeros_like(sym_idx), sym_idx)

        perm_b = sym_perms[sym_idx]  # (B, gs²)

        # Augment board: gather cells with a zero-padded sentinel column so
        # any perm entry == gs² (OOB) reads as 0.
        board_flat = board.view(B, C, gs2)
        zero_col = torch.zeros(B, C, 1, device=board.device, dtype=board.dtype)
        board_padded = torch.cat([board_flat, zero_col], dim=2)  # (B, C, gs²+1)
        perm_bc = perm_b.unsqueeze(1).expand(-1, C, -1)
        new_board = torch.gather(board_padded, 2, perm_bc).view(B, C, gs, gs)

        # Augment placement indices (type * gs² + cell). Slots whose new cell
        # is OOB are clamped to cell 0 and masked out in p_mask below.
        p_idx_l = p_idx.long()
        p_cell = p_idx_l % gs2
        p_type = p_idx_l // gs2
        new_p_cell = torch.gather(perm_b, 1, p_cell)
        p_aug_valid = (new_p_cell < gs2)
        safe_p_cell = torch.where(p_aug_valid, new_p_cell,
                                  torch.zeros_like(new_p_cell))
        new_p_idx = p_type * gs2 + safe_p_cell

        # Augment movement (src, dst).
        m_src_l = m_src.long()
        m_dst_l = m_dst.long()
        new_m_src = torch.gather(perm_b, 1, m_src_l)
        new_m_dst = torch.gather(perm_b, 1, m_dst_l)
        m_aug_valid = (new_m_src < gs2) & (new_m_dst < gs2)
        safe_m_src = torch.where(m_aug_valid, new_m_src,
                                 torch.zeros_like(new_m_src))
        safe_m_dst = torch.where(m_aug_valid, new_m_dst,
                                 torch.zeros_like(new_m_dst))

        # Combined masks: original padding bound × augmentation validity.
        device = board.device
        p_mask_orig = (torch.arange(p_idx.size(1), device=device).unsqueeze(0)
                       < n_p.unsqueeze(1).long())
        m_mask_orig = (torch.arange(m_src.size(1), device=device).unsqueeze(0)
                       < n_m.unsqueeze(1).long())
        p_mask = p_mask_orig & p_aug_valid
        m_mask = m_mask_orig & m_aug_valid

        # Renormalize joint target over surviving placements + movements.
        p_prob_z = p_prob * p_mask.to(p_prob.dtype)
        m_prob_z = m_prob * m_mask.to(m_prob.dtype)
        total = (p_prob_z.sum(dim=1) + m_prob_z.sum(dim=1)).clamp(min=1e-8)
        new_p_prob = p_prob_z / total.unsqueeze(1)
        new_m_prob = m_prob_z / total.unsqueeze(1)

        return (new_board, new_p_idx, new_p_prob,
                safe_m_src, safe_m_dst, new_m_prob,
                p_mask, m_mask)

    def _compute_batch_losses(self, batch, value_loss_scale: float, aux_loss_scale: float):
        """Compute per-batch loss tensors. Shared by train_epoch and validate.

        Q-target mixing for timeout games is now baked into `value_target` at
        Rust-side collection time (see `TimeoutTarget::QMix` in search.rs), so
        no per-batch mixing is needed here.
        """
        (board, reserve, p_idx, p_prob, n_p,
         m_src, m_dst, m_prob, n_m,
         value_target, root_q, vo_mask, po_mask, aux_target, sym_idx) = batch

        device = self.device
        device_type = self.device.type
        D = self.model.bilinear_dim

        board = board.to(device)
        reserve = reserve.to(device)
        p_idx = p_idx.to(device)
        p_prob = p_prob.to(device)
        n_p = n_p.to(device)
        m_src = m_src.to(device)
        m_dst = m_dst.to(device)
        m_prob = m_prob.to(device)
        n_m = n_m.to(device)
        value_target = value_target.to(device).unsqueeze(1)
        root_q = root_q.to(device).unsqueeze(1)
        vo_mask = vo_mask.to(device)
        po_mask = po_mask.to(device)
        aux_target = aux_target.to(device)
        sym_idx = sym_idx.to(device)

        # D6 symmetry augmentation runs here on GPU rather than per-sample in
        # the DataLoader worker. Returned p_mask/m_mask combine the padding
        # bound (arange < n_p) with augmentation validity, so the loss code
        # below uses them directly instead of recomputing from n_p/n_m.
        (board, p_idx, p_prob, m_src, m_dst, m_prob,
         p_mask, m_mask) = self._apply_sym_gpu(
            board, p_idx, p_prob, n_p, m_src, m_dst, m_prob, n_m, sym_idx
        )

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            policy_logits, wdl_logits, aux, src_logits, legal_place_logits = self._compiled(board, reserve)

        B = board.size(0)
        gs = board.size(-1)
        gs2 = gs * gs
        place_end = NUM_PLACE_CHANNELS * gs2

        place_logits_flat = policy_logits[:, :place_end]
        # Q/K come from `nn.Conv2d(C, D, 1)(p).flatten(1)`, which has D-major
        # memory layout: flat[b, d*gs2 + cell]. The Rust MCTS reads with this
        # same stride (search.rs DotProduct: q_offset + d*g2 + src_cell).
        # `.view(B, gs2, D)` would silently re-interpret the memory as
        # cell-major and scramble channel<->cell, breaking gradient flow on
        # movement priors (placements were unaffected since they use a flat
        # gather). Keep the D-major layout, then transpose for `gather` over
        # cells along dim 1.
        q = policy_logits[:, place_end:place_end + D * gs2].view(B, D, gs2).transpose(1, 2)
        k = policy_logits[:, place_end + D * gs2:].view(B, D, gs2).transpose(1, 2)

        p_gather = torch.gather(place_logits_flat, 1, p_idx)

        src_exp = m_src.unsqueeze(-1).expand(-1, -1, D)
        dst_exp = m_dst.unsqueeze(-1).expand(-1, -1, D)
        q_g = torch.gather(q, 1, src_exp)
        k_g = torch.gather(k, 1, dst_exp)
        m_logits = (q_g * k_g).sum(-1) / math.sqrt(D)

        combined_legal_logits = torch.cat([p_gather, m_logits], dim=1)
        combined_probs = torch.cat([p_prob, m_prob], dim=1)

        # p_mask, m_mask were produced by _apply_sym_gpu and already combine
        # the padding bound with augmentation validity.
        c_mask = torch.cat([p_mask, m_mask], dim=1)

        # Softmax denominator: legal placements + legal movements only.
        # Padded slots are -inf so they drop out of logsumexp.
        masked_logits = combined_legal_logits.masked_fill(~c_mask, float('-inf'))
        log_probs = torch.log_softmax(masked_logits.float(), dim=1)
        log_probs = log_probs.masked_fill(~c_mask, 0.0)  # avoid 0 * -inf = NaN

        per_sample_policy = -(combined_probs * log_probs).sum(dim=1)
        has_target = ((n_p + n_m) > 0)
        per_sample_policy = per_sample_policy * has_target.float()

        policy_weight = (~vo_mask).float()
        # Normalize by active-sample count, not full batch: .mean() ties the
        # policy gradient (and the logged number) to the mask fraction, so
        # --playout-cap-p secretly scales the effective policy LR. Sum-over-
        # active matches AlphaZero/KataGo and decouples LR from masking.
        policy_loss = (per_sample_policy * policy_weight).sum() / policy_weight.sum().clamp(min=1.0)

        wdl_target = _scalar_to_wdl(value_target.squeeze(1))
        log_wdl = torch.log_softmax(wdl_logits.float(), dim=1)
        per_sample_value = -(wdl_target * log_wdl).sum(dim=1)
        value_weight = (~po_mask).float()
        value_loss = (per_sample_value * value_weight).sum() / value_weight.sum().clamp(min=1.0)

        aux_mse = (aux - aux_target) ** 2
        qd_loss = aux_mse[:, 0:2].mean()
        qe_loss = aux_mse[:, 2:4].mean()
        mob_loss = aux_mse[:, 4:6].mean()

        # Source-marginal aux: per-cell BCE against Σ_dst π_MCTS(src, dst).
        # Padded movement slots have m_prob == 0, so scatter_add over them is
        # a no-op regardless of m_src content.
        src_target = torch.zeros(B, gs2, device=device, dtype=src_logits.dtype)
        src_target.scatter_add_(1, m_src.long(), m_prob.to(src_logits.dtype))
        src_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            src_logits, src_target
        )

        # Legal-placement aux: per-cell BCE against the binary "any of my piece
        # types can place here" mask derived from place_idx. p_idx[i, k] is the
        # flat (channel × gs² + cell) index of the k-th legal placement;
        # cell = idx % gs², OR'd across channels. p_mask already excludes both
        # padded slots and aug-invalid slots.
        p_cells = (p_idx.long() % gs2)                                 # (B, MAX_PLACEMENTS)
        p_valid = p_mask.to(legal_place_logits.dtype)
        lp_count = torch.zeros(B, gs2, device=device, dtype=legal_place_logits.dtype)
        lp_count.scatter_add_(1, p_cells, p_valid)
        lp_target = (lp_count > 0).to(legal_place_logits.dtype)
        lp_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            legal_place_logits, lp_target
        )

        aux_loss = qd_loss + qe_loss + mob_loss + src_loss + lp_loss

        total = policy_loss + value_loss_scale * value_loss + aux_loss_scale * aux_loss

        pol_active = has_target & (~vo_mask)
        argmax_pred = masked_logits.argmax(dim=1)
        argmax_true = combined_probs.argmax(dim=1)
        top1_hits = ((argmax_pred == argmax_true) & pol_active).sum()
        n_legal = c_mask.sum(dim=1).clamp(min=1).float()
        uniform_ce_sum = (torch.log(n_legal) * pol_active.float()).sum()
        n_pol_active = pol_active.sum()

        return {
            "total": total, "policy": policy_loss, "value": value_loss,
            "aux": aux_loss, "qd": qd_loss, "qe": qe_loss, "mob": mob_loss,
            "src": src_loss, "lp": lp_loss,
            "top1_hits": top1_hits, "uniform_ce_sum": uniform_ce_sum,
            "n_pol_active": n_pol_active,
        }

    def train_epoch(self, dataset: HiveDataset, batch_size: int = 64, value_loss_scale: float = 1.0,
                    aux_loss_scale: float = 1.0, subsample_frac: float = 1.0) -> dict:
        """Train one epoch. Returns loss dict.

        `subsample_frac`: if < 1.0, draw a uniform random subset of size
        int(len(dataset) * frac) without replacement instead of iterating the
        full dataset. With symmetry augmentation on, this samples uniformly
        from the (position × sym) flat pool — a fresh subset each epoch.
        """
        self.model.train()
        drop = self.device.type == "cuda"
        if subsample_frac < 1.0:
            from torch.utils.data import RandomSampler
            n = max(1, int(len(dataset) * subsample_frac))
            sampler = RandomSampler(dataset, replacement=False, num_samples=n)
            loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=drop)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop)

        sums = {"total": 0.0, "policy": 0.0, "value": 0.0,
                "aux": 0.0, "qd": 0.0, "qe": 0.0, "mob": 0.0, "src": 0.0, "lp": 0.0}
        num_batches = 0

        for batch in tqdm(loader, desc="  Training", leave=False, unit="batch"):
            losses = self._compute_batch_losses(batch, value_loss_scale, aux_loss_scale)
            loss = losses["total"]

            if self.warmup_steps > 0 and self._step_count < self.warmup_steps:
                scale = (self._step_count + 1) / self.warmup_steps
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self._base_lr * scale
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
            return {"policy_loss": 0, "value_loss": 0, "qd_loss": 0,
                    "qe_loss": 0, "mob_loss": 0, "src_loss": 0, "lp_loss": 0,
                    "aux_loss": 0, "total_loss": 0}

        return {
            "policy_loss": sums["policy"] / num_batches,
            "value_loss": sums["value"] / num_batches,
            "aux_loss": sums["aux"] / num_batches,
            "qd_loss": sums["qd"] / num_batches,
            "qe_loss": sums["qe"] / num_batches,
            "mob_loss": sums["mob"] / num_batches,
            "src_loss": sums["src"] / num_batches,
            "lp_loss": sums["lp"] / num_batches,
            "total_loss": sums["total"] / num_batches,
        }

    @torch.no_grad()
    def validate(self, dataset: HiveDataset, batch_size: int = 64,
                 value_loss_scale: float = 1.0, aux_loss_scale: float = 1.0) -> dict:
        """No-grad pass over `dataset`. Returns loss dict + top1_acc and uniform_ce diagnostics."""
        self.model.eval()
        drop = self.device.type == "cuda"
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=drop)

        sums = {"total": 0.0, "policy": 0.0, "value": 0.0,
                "aux": 0.0, "qd": 0.0, "qe": 0.0, "mob": 0.0, "src": 0.0, "lp": 0.0}
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
            return {"policy_loss": 0, "value_loss": 0, "qd_loss": 0,
                    "qe_loss": 0, "mob_loss": 0, "src_loss": 0, "lp_loss": 0,
                    "aux_loss": 0, "total_loss": 0,
                    "top1_acc": 0.0, "uniform_ce": 0.0}

        denom = max(n_pol_active, 1)
        return {
            "policy_loss": sums["policy"] / num_batches,
            "value_loss": sums["value"] / num_batches,
            "aux_loss": sums["aux"] / num_batches,
            "qd_loss": sums["qd"] / num_batches,
            "qe_loss": sums["qe"] / num_batches,
            "mob_loss": sums["mob"] / num_batches,
            "src_loss": sums["src"] / num_batches,
            "lp_loss": sums["lp"] / num_batches,
            "total_loss": sums["total"] / num_batches,
            "top1_acc": top1_hits / denom,
            "uniform_ce": uniform_ce_sum / denom,
        }
