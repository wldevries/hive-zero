"""Training loop for YinshNet — single flat 59-channel policy head, WDL value.

The dataset optionally augments samples with the subset of D6 hex symmetries
that map every Yinsh valid cell to another valid cell (computed in Rust by
`yinsh_valid_d6_indices`). Two policy channel groups need direction permutation:
  - ClaimRow row channels (1-3): 3 row directions, via `yinsh_d6_dir_permutations`
  - MoveRing channels (5-58): 6 movement directions, via `yinsh_d6_movement_dir_permutations`
The ClaimRow ring channel (4) and PlaceRing channel (0) only need spatial gather.
"""

from __future__ import annotations

import contextlib
import os

# Disable HDF5 advisory file locking so DataLoader workers can open the
# replay buffer 'r' while the main process is detached. libhdf5 reads this
# on every open, and the var propagates through spawn() on Windows. Safe
# because training and selfplay phases never overlap by design (the main
# process closes 'r+' before workers open 'r').
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import (
    GRID_SIZE,
    NUM_CHANNELS,
    POLICY_CHANNELS,
    POLICY_SIZE,
    RESERVE_SIZE,
    YinshNet,
    create_model,
)
from shared.replay_buffer import handle_buffer_size_mismatch

_GS = GRID_SIZE * GRID_SIZE  # 121

# Layout of the 59 policy channels (must match move_encoding.rs).
_CH_PLACE_RING = 0
_CH_CLAIM_ROW_BASE = 1   # channels 1, 2, 3 — ClaimRow row partner (joint A)
_CH_CLAIM_RING = 4       # channel 4 — ClaimRow ring partner (joint B)
_CH_MOVE_BASE = 5        # channels 5..58: 6 dirs × 9 distances
_MAX_RING_DIST = 9
_NUM_DIRS = 6

_GRID_PERM_CACHE: list | None = None
_DIR_PERM_CACHE: list | None = None
_MOVE_DIR_PERM_CACHE: list | None = None
_VALID_SYM_CACHE: list | None = None


def _load_grid_perms() -> list[np.ndarray]:
    global _GRID_PERM_CACHE
    if _GRID_PERM_CACHE is None:
        from engine_zero import yinsh_d6_grid_permutations
        _GRID_PERM_CACHE = [np.asarray(p, dtype=np.int64) for p in yinsh_d6_grid_permutations()]
    return _GRID_PERM_CACHE


def _load_dir_perms() -> list[np.ndarray]:
    global _DIR_PERM_CACHE
    if _DIR_PERM_CACHE is None:
        from engine_zero import yinsh_d6_dir_permutations
        _DIR_PERM_CACHE = [np.asarray(p, dtype=np.int64) for p in yinsh_d6_dir_permutations()]
    return _DIR_PERM_CACHE


def _load_move_dir_perms() -> list[np.ndarray]:
    global _MOVE_DIR_PERM_CACHE
    if _MOVE_DIR_PERM_CACHE is None:
        from engine_zero import yinsh_d6_movement_dir_permutations
        _MOVE_DIR_PERM_CACHE = [np.asarray(p, dtype=np.int64) for p in yinsh_d6_movement_dir_permutations()]
    return _MOVE_DIR_PERM_CACHE


def _load_valid_syms() -> list[int]:
    global _VALID_SYM_CACHE
    if _VALID_SYM_CACHE is None:
        from engine_zero import yinsh_valid_d6_indices
        _VALID_SYM_CACHE = list(yinsh_valid_d6_indices())
    return _VALID_SYM_CACHE


# Dataset attribute names that map 1:1 to h5py datasets. Used by the
# detach/reattach lifecycle and the pickling/worker-init plumbing so we don't
# have to repeat the list in three places.
_H5_DATASETS = (
    "board_tensors", "reserve_vectors",
    "policy_targets",
    "value_targets", "root_q_targets",
    "value_only", "phase_flags", "generations",
)


def _h5_worker_init_fn(worker_id: int):
    """DataLoader worker init: open the unpickled dataset's h5 file 'r'.

    h5py.File handles don't pickle, so workers arrive with _h5file=None and
    all dataset attrs stripped (see YinshDataset.__getstate__). Each worker
    opens its own libhdf5 handle; concurrent 'r' opens are safe given the
    HDF5_USE_FILE_LOCKING=FALSE module-level setting and the main process
    having closed its 'r+' handle via YinshDataset.read_only_mode().
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    if hasattr(dataset, "_open_for_read"):
        dataset._open_for_read()


def _apply_symmetry(
    board: np.ndarray,
    policy: np.ndarray,
    sym: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply D6 transform `sym` to (board[4, 11, 11], policy[7139]).

    Board channels are all rotation-symmetric (rings/markers, current-player-relative)
    and need only spatial gather. Two policy channel groups also need a direction
    permutation:
      - ClaimRow row (ch 1-3): via row-direction table
      - MoveRing  (ch 5-58): 6-direction blocks of 9 distances each
    The ClaimRow ring channel (4) and PlaceRing channel (0) only need spatial gather.
    The reserve vector is unaffected by the symmetry (its phase one-hot is a
    rotation-invariant scalar).
    """
    if sym == 0:
        return board, policy

    grid_perm = _load_grid_perms()[sym]
    dir_perm = _load_dir_perms()[sym]
    move_dir_perm = _load_move_dir_perms()[sym]

    # Board: pad with one zero column so out-of-board positions gather to 0.
    flat = board.reshape(NUM_CHANNELS, _GS)
    padded = np.concatenate([flat, np.zeros((NUM_CHANNELS, 1), dtype=np.float32)], axis=1)
    new_board = padded[:, grid_perm].reshape(NUM_CHANNELS, GRID_SIZE, GRID_SIZE)

    # Policy: spatial gather first, then channel permutation for direction-sensitive groups.
    pol_flat = policy.reshape(POLICY_CHANNELS, _GS)
    pol_padded = np.concatenate([pol_flat, np.zeros((POLICY_CHANNELS, 1), dtype=np.float32)], axis=1)
    spatially_perm = pol_padded[:, grid_perm]  # (59, 121)

    new_policy = np.empty_like(spatially_perm)
    # PlaceRing (ch 0) and ClaimRow ring (ch 4): spatial only.
    new_policy[_CH_PLACE_RING] = spatially_perm[_CH_PLACE_RING]
    new_policy[_CH_CLAIM_RING] = spatially_perm[_CH_CLAIM_RING]
    # ClaimRow row (ch 1-3): permute direction channels via row-dir table.
    for d in range(3):
        old_d = int(dir_perm[d])
        new_policy[_CH_CLAIM_ROW_BASE + d] = spatially_perm[_CH_CLAIM_ROW_BASE + old_d]
    # MoveRing (ch 5-58): permute the 6 direction-blocks of 9 distances.
    for new_d in range(_NUM_DIRS):
        old_d = int(move_dir_perm[new_d])
        new_base = _CH_MOVE_BASE + new_d * _MAX_RING_DIST
        old_base = _CH_MOVE_BASE + old_d * _MAX_RING_DIST
        new_policy[new_base : new_base + _MAX_RING_DIST] = spatially_perm[old_base : old_base + _MAX_RING_DIST]

    return new_board, new_policy.reshape(POLICY_SIZE)


class YinshDataset(Dataset):
    """Ring-buffer replay dataset for Yinsh self-play positions.

    When `buf_dir` is given, all arrays are stored in a single HDF5 file
    (`replay.h5`) so the buffer survives process restarts. `_count`/`_size`
    live as HDF5 attributes in the same file — no separate metadata file.

    h5py datasets support the same slice-assignment syntax as numpy arrays, so
    `add_batch` and `__getitem__` use a single code path for both cases.
    """

    def __init__(self, max_size: int = 100_000, buf_dir: str | None = None):
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
                self._count = int(self._h5file.attrs["count"])
                self._size = int(self._h5file.attrs["size"])
                print(f"  Replay buffer resumed: {self._size} samples from {h5path}")
            else:
                self._count = 0
                self._size = 0
                self._h5file.attrs["max_size"] = max_size
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

        self.board_tensors  = _ds("board_tensors",  (max_size, NUM_CHANNELS, GRID_SIZE, GRID_SIZE), np.float32)
        self.reserve_vectors = _ds("reserve_vectors", (max_size, RESERVE_SIZE), np.float32)
        self.policy_targets  = _ds("policy_targets",  (max_size, POLICY_SIZE),  np.float32)
        self.value_targets   = _ds("value_targets",   (max_size,),               np.float32)
        # MCTS root W−L (no contempt) per sample, used for q-target value
        # mixing in `Trainer.train_epoch`. NaN on any sample disables mixing
        # for that sample (falls back to z). Older buffers without this
        # dataset get one initialized from value_targets so q-mix is a no-op
        # for those samples on resume.
        self.root_q_targets  = _ds("root_q_targets",  (max_size,),               np.float32)
        self.value_only      = _ds("value_only",      (max_size,),               np.bool_)
        self.phase_flags     = _ds("phase_flags",     (max_size,),               np.uint8)
        self.generations     = _ds("generations",     (max_size,),               np.int32)

        if self._h5file is not None:
            # Backward-compat: a buffer written before root_q existed will
            # have just created the dataset filled with zeros, but the live
            # samples (indices < self._size) were trained on real outcomes.
            # Seed root_q for those samples from value_targets so q-mix
            # collapses to z (no change in behavior) until they cycle out.
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

    # ---------------------------------------------------------------------
    # Multi-process read lifecycle.
    # ---------------------------------------------------------------------
    # The main process holds an 'r+' handle during selfplay writes; for
    # training we want DataLoader workers to read in parallel. h5py.File
    # doesn't pickle, so workers can't inherit the main handle — they each
    # open a fresh 'r' handle via worker_init_fn. The main process must
    # release its handle before workers open theirs (and reopen afterwards
    # for the next selfplay phase). read_only_mode() encapsulates that.

    def _attach_h5_datasets(self):
        f = self._h5file
        for name in _H5_DATASETS:
            setattr(self, name, f[name])

    def _open_for_read(self):
        """Worker-side: open this dataset's h5 file 'r' and attach handles."""
        if self._h5path is None:
            return
        import h5py
        self._h5file = h5py.File(self._h5path, "r")
        self._attach_h5_datasets()

    def _reopen_h5(self, mode: str):
        """Flush+close any current handle, then open `mode` and re-attach."""
        import h5py
        if self._h5file is not None:
            self._h5file.flush()
            self._h5file.close()
            self._h5file = None
        self._h5file = h5py.File(self._h5path, mode)
        self._attach_h5_datasets()

    @contextlib.contextmanager
    def read_only_mode(self):
        """Swap main's 'r+' handle to 'r' so DataLoader workers can also open 'r'.

        Main keeps a working handle (so num_workers=0 still reads), and
        worker processes open their own 'r' handles in worker_init_fn.
        Concurrent 'r' opens coexist because HDF5_USE_FILE_LOCKING=FALSE is
        set at module load. On exit, main reopens 'r+' for the next write
        phase. No-op for in-memory backing.
        """
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
        # h5py.File / h5py.Dataset objects can't pickle. Workers will reopen
        # via _h5_worker_init_fn. For the in-memory backing case (no _h5path),
        # the dataset attrs are numpy arrays — let them pickle normally.
        if self._h5path is not None:
            state["_h5file"] = None
            for name in _H5_DATASETS:
                state[name] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # h5-backed: worker_init_fn will open the file. In-memory: arrays
        # came through pickle intact; nothing to do.

    def add_batch(
        self,
        board_tensors: np.ndarray,
        reserve_vectors: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        value_only: list[bool],
        phase_flags: list[int] | np.ndarray,
        generation: int = 0,
        root_q_targets: np.ndarray | None = None,
    ):
        n = board_tensors.shape[0]
        boards  = board_tensors.reshape(n, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        vo_arr  = np.asarray(value_only, dtype=np.bool_)
        pf_arr  = np.asarray(phase_flags, dtype=np.uint8)
        gen_arr = np.full(n, generation, dtype=np.int32)
        # When root_q is not provided (e.g. supervised pretrain that has no
        # MCTS), fall back to value_targets so q-mix is a no-op.
        rq_arr = (
            np.asarray(root_q_targets, dtype=np.float32)
            if root_q_targets is not None
            else np.asarray(value_targets, dtype=np.float32)
        )

        pairs = [
            (self.board_tensors,   boards),
            (self.reserve_vectors, reserve_vectors),
            (self.policy_targets,  policy_targets),
            (self.value_targets,   value_targets),
            (self.root_q_targets,  rq_arr),
            (self.value_only,      vo_arr),
            (self.phase_flags,     pf_arr),
            (self.generations,     gen_arr),
        ]

        start = self._count % self.max_size
        end   = start + n
        if end <= self.max_size:
            for arr, data in pairs:
                arr[start:end] = data
        else:
            first = self.max_size - start
            for arr, data in pairs:
                arr[start:]      = data[:first]
                arr[:n - first]  = data[first:]

        self._count += n
        self._size = min(self._size + n, self.max_size)

        if self._h5file is not None:
            self._h5file.attrs["count"] = self._count
            self._h5file.attrs["size"]  = self._size
            self._h5file.flush()

    def clear(self):
        self._count = 0
        self._size  = 0
        if self._h5file is not None:
            self._h5file.attrs["count"] = 0
            self._h5file.attrs["size"]  = 0
            self._h5file.flush()

    def __len__(self):
        if self.augment_symmetry:
            return self._size * len(_load_valid_syms())
        return self._size

    def __getitem__(self, idx):
        """Returns the raw position plus a `sym_idx` integer in [0, 12).

        The D6 permutation is applied on GPU in Trainer._apply_sym_gpu after
        the batch lands on device — moves the per-sample numpy gather/channel-
        permutation work off the main thread, which is the data-loading
        bottleneck on Windows where multi-worker DataLoaders are unreliable.
        """
        if self.augment_symmetry:
            valid_syms = _load_valid_syms()
            sym = valid_syms[idx % len(valid_syms)]
            base_idx = idx // len(valid_syms)
        else:
            sym = 0
            base_idx = idx

        return (
            torch.from_numpy(self.board_tensors[base_idx].copy()),
            torch.from_numpy(self.reserve_vectors[base_idx].copy()),
            torch.from_numpy(self.policy_targets[base_idx].copy()),
            torch.tensor(self.value_targets[base_idx], dtype=torch.float32),
            torch.tensor(self.root_q_targets[base_idx], dtype=torch.float32),
            torch.tensor(bool(self.value_only[base_idx]), dtype=torch.bool),
            torch.tensor(sym, dtype=torch.int64),
        )

    def __getitems__(self, indices):
        """Batched fetch — one read per dataset instead of one per sample.

        PyTorch's DataLoader (>=1.13) calls this with the batch's full index
        list when defined, falling back to __getitem__ otherwise. Each
        __getitem__ call triggers ~7 random h5py reads; collapsing the batch
        into one fancy-index read per dataset hits h5py's sorted-unique fast
        path and amortises per-call overhead across the whole batch.

        Returns a list[tuple] in the same shape as __getitem__, in the same
        order as `indices`.
        """
        n = len(indices)
        idx_arr = np.asarray(indices, dtype=np.int64)
        if self.augment_symmetry:
            valid_syms = _load_valid_syms()
            L = len(valid_syms)
            base = idx_arr // L
            syms = np.asarray(valid_syms, dtype=np.int64)[idx_arr % L]
        else:
            base = idx_arr
            syms = np.zeros(n, dtype=np.int64)

        if self._h5file is not None:
            # h5py fast-path: sorted-unique selection list + numpy fan-out
            # via the inverse permutation. Dedupes in-batch collisions for free.
            unique, inverse = np.unique(base, return_inverse=True)
            sel = unique.tolist()
            def read(ds):
                return ds[sel][inverse]
        else:
            # In-memory backing: fancy indexing already copies, no disk concerns.
            def read(ds):
                return ds[base]

        boards     = read(self.board_tensors)
        reserves   = read(self.reserve_vectors)
        policies   = read(self.policy_targets)
        v_target   = read(self.value_targets)
        rq_target  = read(self.root_q_targets)
        vo         = read(self.value_only)

        out = []
        for i in range(n):
            out.append((
                torch.from_numpy(boards[i]),
                torch.from_numpy(reserves[i]),
                torch.from_numpy(policies[i]),
                torch.tensor(float(v_target[i]), dtype=torch.float32),
                torch.tensor(float(rq_target[i]), dtype=torch.float32),
                torch.tensor(bool(vo[i]), dtype=torch.bool),
                torch.tensor(int(syms[i]), dtype=torch.int64),
            ))
        return out


def _policy_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample CE over legal moves only.

    Of the 7139 policy logits, ~30 correspond to legal moves at any state and
    the rest are unreachable. We mask illegal cells to ~-inf before softmax so
    the partition function sums only over legals — the network gets zero
    gradient on illegal logits and is free to emit garbage there. The Rust
    MCTS path (`expand_with_policy`) likewise reads logits only at legal-move
    indices and softmaxes over those, so training and inference stay aligned.

    The legal mask is derived from `target > 0`: every legal move's encoding
    receives nonzero visit probability (including ClaimRow's `Sum(a, b)` 50/50
    split, which writes to both factor cells).
    """
    legal = (target > 0).to(logits.dtype)
    masked_logits = logits + (legal - 1.0) * 1e9
    t_sum = target.sum(dim=1, keepdim=True).clamp(min=1e-8)
    t_norm = target / t_sum
    return -(t_norm * torch.log_softmax(masked_logits, dim=1)).sum(dim=1)


def _scalar_to_wdl(v: torch.Tensor) -> torch.Tensor:
    """Convert scalar value targets in [-1, 1] to (W, D, L) soft targets summing to 1."""
    w = v.clamp(min=0)
    l = (-v).clamp(min=0)
    d = 1.0 - w - l
    return torch.stack([w, d, l], dim=1)


class Trainer:
    """SGD+momentum trainer for YinshNet."""

    def __init__(
        self,
        model: Optional[YinshNet] = None,
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
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        elif kind == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"unknown optimizer {optimizer!r}; expected 'sgd' or 'adamw'")
        self._compiled = torch.compile(self.model, dynamic=False) if self.device.type == "cuda" else self.model
        # Lazy-built GPU permutation tables (see _init_sym_buffers).
        self._grid_perms_gpu: Optional[torch.Tensor] = None
        self._ch_perms_gpu: Optional[torch.Tensor] = None

    @property
    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _init_sym_buffers(self):
        """Move the D6 permutation tables to GPU.

        Two tables are needed:
          - grid_perms (12, GRID_SIZE²): spatial cell permutation. Entries can
            equal GRID_SIZE² (sentinel for off-board cells); the caller pads
            the input with a zero column to absorb them. Even within the
            valid-sym subset, the 11×11 grid has 36 off-board cells whose
            perm value is the sentinel.
          - ch_perms (12, POLICY_CHANNELS=59): policy-channel permutation that
            combines spatial-only channels (PlaceRing ch 0, ClaimRow ring ch 4)
            with direction-permuted channels (ClaimRow row dirs ch 1-3 via
            yinsh_d6_dir_permutations, MoveRing dirs × distances ch 5-58 via
            yinsh_d6_movement_dir_permutations).
        """
        if self._grid_perms_gpu is not None:
            return
        grid = np.stack(_load_grid_perms()).astype(np.int64)  # (12, _GS)
        self._grid_perms_gpu = torch.from_numpy(grid).to(self.device)

        dir_perms = _load_dir_perms()        # list of 12 arrays of len 3
        mv_perms = _load_move_dir_perms()    # list of 12 arrays of len 6
        ch_table = np.empty((12, POLICY_CHANNELS), dtype=np.int64)
        for s in range(12):
            ch_table[s, _CH_PLACE_RING] = _CH_PLACE_RING
            ch_table[s, _CH_CLAIM_RING] = _CH_CLAIM_RING
            for d in range(3):
                old_d = int(dir_perms[s][d])
                ch_table[s, _CH_CLAIM_ROW_BASE + d] = _CH_CLAIM_ROW_BASE + old_d
            for new_d in range(_NUM_DIRS):
                old_d = int(mv_perms[s][new_d])
                new_base = _CH_MOVE_BASE + new_d * _MAX_RING_DIST
                old_base = _CH_MOVE_BASE + old_d * _MAX_RING_DIST
                for dist in range(_MAX_RING_DIST):
                    ch_table[s, new_base + dist] = old_base + dist
        self._ch_perms_gpu = torch.from_numpy(ch_table).to(self.device)

    def _apply_sym_gpu(self, board: torch.Tensor, policy: torch.Tensor,
                       sym_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply per-sample D6 augmentation on GPU.

        Inputs are already on `self.device`. `sym_idx` is (B,) in [0, 12).
        Mirrors the CPU augmentation that used to live in YinshDataset.__getitem__:
        permutes board cells (with zero-padded sentinel for the 36 off-board
        positions of the 11×11 grid) and permutes the 59 policy channels both
        spatially (cell gather) and by direction (channel gather).
        Sym=0 is identity so `augment_symmetry=False` is a no-op pass through
        this helper.
        """
        self._init_sym_buffers()
        B, C, gs, _ = board.shape
        gs2 = gs * gs
        grid_perm_b = self._grid_perms_gpu[sym_idx]   # (B, gs²)
        ch_perm_b = self._ch_perms_gpu[sym_idx]       # (B, POLICY_CHANNELS)

        # Board cell permutation: pad with a zero column so any sentinel
        # entry == gs² reads as 0.
        board_flat = board.view(B, C, gs2)
        zero_col_b = torch.zeros(B, C, 1, device=board.device, dtype=board.dtype)
        board_padded = torch.cat([board_flat, zero_col_b], dim=2)
        gp_bc = grid_perm_b.unsqueeze(1).expand(-1, C, -1)
        new_board = torch.gather(board_padded, 2, gp_bc).view(B, C, gs, gs)

        # Policy: spatial cell gather first, then channel-direction gather.
        pol_flat = policy.view(B, POLICY_CHANNELS, gs2)
        zero_col_p = torch.zeros(B, POLICY_CHANNELS, 1,
                                 device=policy.device, dtype=policy.dtype)
        pol_padded = torch.cat([pol_flat, zero_col_p], dim=2)
        gp_pc = grid_perm_b.unsqueeze(1).expand(-1, POLICY_CHANNELS, -1)
        spatially_permuted = torch.gather(pol_padded, 2, gp_pc)  # (B, C_P, gs²)

        ch_idx = ch_perm_b.unsqueeze(2).expand(-1, -1, gs2)
        new_policy = torch.gather(spatially_permuted, 1, ch_idx)
        return new_board, new_policy.view(B, POLICY_SIZE)

    def _loader_kwargs(self, num_workers: int) -> dict:
        """Shared DataLoader kwargs. Enables pin_memory on CUDA and worker
        plumbing when num_workers > 0. With num_workers == 0, falls back to
        the pre-existing single-threaded fetch path.

        persistent_workers is deliberately off: each train_epoch creates a
        fresh DataLoader, so workers would never be reused across iters;
        leaving it on would only delay their teardown non-deterministically
        and keep 'r' h5 handles alive past read_only_mode's __exit__."""
        kw: dict = {"num_workers": num_workers}
        if self.device.type == "cuda":
            kw["pin_memory"] = True
        if num_workers > 0:
            kw["worker_init_fn"] = _h5_worker_init_fn
        return kw

    def train_epoch(
        self,
        dataset: YinshDataset,
        batch_size: int = 256,
        value_loss_scale: float = 1.0,
        q_mix_lambda: float = 0.0,
        num_workers: int = 0,
    ) -> dict:
        """Run one training epoch.

        `q_mix_lambda`: KataGo-style q-target mixing weight in [0, 1]. The
        per-sample value target becomes `(1-λ)·z + λ·root_q`, where `z` is the
        game outcome and `root_q` is the MCTS root W−L (no contempt) at the
        played turn. Samples with NaN root_q (e.g. supervised pretrain or
        bot-played turns in hive) bypass mixing and use `z` directly. λ=0
        recovers the original outcome-only behavior.

        `num_workers`: if > 0, the caller must wrap this call in
        `dataset.read_only_mode()` so workers can open the h5 file 'r'.
        """
        self.model.train()
        drop = self.device.type == "cuda"
        loader_kwargs = self._loader_kwargs(num_workers)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=drop, **loader_kwargs)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        device_type = self.device.type

        nb = self.device.type == "cuda"  # non_blocking only helps with pinned source
        for board, reserve, policy_target, value_target, root_q, value_only_mask, sym_idx in tqdm(
            loader, desc="  Training", leave=False, unit="batch"
        ):
            board = board.to(self.device, non_blocking=nb)
            reserve = reserve.to(self.device, non_blocking=nb)
            policy_target = policy_target.to(self.device, non_blocking=nb)
            value_target = value_target.to(self.device, non_blocking=nb).unsqueeze(1)
            root_q = root_q.to(self.device, non_blocking=nb).unsqueeze(1)
            value_only_mask = value_only_mask.to(self.device, non_blocking=nb)
            sym_idx = sym_idx.to(self.device, non_blocking=nb)

            # D6 symmetry augmentation runs here on GPU rather than per-sample
            # in the DataLoader worker. sym=0 is identity so augment_symmetry=
            # False is a no-op pass through this helper.
            board, policy_target = self._apply_sym_gpu(board, policy_target, sym_idx)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                policy_logits, wdl_logits = self._compiled(board, reserve)

            # Policy loss: skip value-only (fast-cap) turns.
            policy_active = ~value_only_mask
            if policy_active.any():
                p_loss_per = _policy_ce(
                    policy_logits[policy_active], policy_target[policy_active]
                )
                policy_loss = p_loss_per.mean()
            else:
                policy_loss = torch.tensor(0.0, device=self.device)

            # Q-target mixing: blend the search-improved root value into the
            # outcome target. NaN root_q (no MCTS at that sample) keeps z.
            if q_mix_lambda > 0.0:
                rq_safe = torch.where(torch.isnan(root_q), value_target, root_q)
                effective_target = (1.0 - q_mix_lambda) * value_target + q_mix_lambda * rq_safe
            else:
                effective_target = value_target

            # Value loss: cross-entropy on WDL soft targets.
            wdl_target = _scalar_to_wdl(effective_target.squeeze(1))  # (B, 3)
            log_wdl = torch.log_softmax(wdl_logits.float(), dim=1)
            per_sample_value = -(wdl_target * log_wdl).sum(dim=1)
            value_loss = per_sample_value.mean()

            loss = policy_loss + value_loss_scale * value_loss

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

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_loss += float(loss.item())
            num_batches += 1

        if num_batches == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "total_loss": total_loss / num_batches,
        }

    @torch.no_grad()
    def validate(
        self,
        dataset: YinshDataset,
        batch_size: int = 256,
        value_loss_scale: float = 1.0,
        num_workers: int = 0,
    ) -> dict:
        """Mirror of `train_epoch` with no backward/step. Returns loss dict + top1_acc and uniform_ce."""
        self.model.eval()
        drop = self.device.type == "cuda"
        loader_kwargs = self._loader_kwargs(num_workers)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            drop_last=drop, **loader_kwargs)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        top1_hits = 0
        uniform_ce_sum = 0.0
        n_pol_active = 0
        num_batches = 0
        device_type = self.device.type

        nb = self.device.type == "cuda"
        for board, reserve, policy_target, value_target, _root_q, value_only_mask, sym_idx in tqdm(
            loader, desc="  Validating", leave=False, unit="batch"
        ):
            board = board.to(self.device, non_blocking=nb)
            reserve = reserve.to(self.device, non_blocking=nb)
            policy_target = policy_target.to(self.device, non_blocking=nb)
            value_target = value_target.to(self.device, non_blocking=nb).unsqueeze(1)
            value_only_mask = value_only_mask.to(self.device, non_blocking=nb)
            sym_idx = sym_idx.to(self.device, non_blocking=nb)

            board, policy_target = self._apply_sym_gpu(board, policy_target, sym_idx)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                policy_logits, wdl_logits = self._compiled(board, reserve)

            policy_active = ~value_only_mask
            if policy_active.any():
                pa_logits = policy_logits[policy_active]
                pa_target = policy_target[policy_active]
                p_loss_per = _policy_ce(pa_logits, pa_target)
                policy_loss = p_loss_per.mean()

                pa_legal = (pa_target > 0).to(pa_logits.dtype)
                pa_masked = pa_logits + (pa_legal - 1.0) * 1e9
                argmax_pred = pa_masked.argmax(dim=1)
                argmax_true = pa_target.argmax(dim=1)
                top1_hits += int((argmax_pred == argmax_true).sum().item())
                n_legal_per = pa_legal.sum(dim=1).clamp(min=1)
                uniform_ce_sum += float(torch.log(n_legal_per).sum().item())
                n_pol_active += int(policy_active.sum().item())
            else:
                policy_loss = torch.tensor(0.0, device=self.device)

            wdl_target = _scalar_to_wdl(value_target.squeeze(1))
            log_wdl = torch.log_softmax(wdl_logits.float(), dim=1)
            per_sample_value = -(wdl_target * log_wdl).sum(dim=1)
            value_loss = per_sample_value.mean()

            loss = policy_loss + value_loss_scale * value_loss

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_loss += float(loss.item())
            num_batches += 1

        if num_batches == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0,
                    "top1_acc": 0.0, "uniform_ce": 0.0}

        denom = max(n_pol_active, 1)
        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "total_loss": total_loss / num_batches,
            "top1_acc": top1_hits / denom,
            "uniform_ce": uniform_ce_sum / denom,
        }
