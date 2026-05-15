"""Training loop for ZertzNet with factorized conv1x1 policy heads.

The dataset stores raw positions plus a `sym_idx` integer in [0, 12); the D6
symmetry permutation is applied on GPU in `Trainer._apply_sym_gpu` after the
batch lands on device. Mirrors the Yinsh layout: spatial-only board cells +
fused channel-direction table that handles the cap_dir direction channels
(4-9) and leaves the place/remove channels (0-3) spatial-only.
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

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional
from tqdm import tqdm

from .model import ZertzNet, create_model, NUM_CHANNELS, GRID_SIZE, POLICY_SIZE, RESERVE_SIZE
from shared.replay_buffer import handle_buffer_size_mismatch

_BOARD_SIZE = 37  # number of valid hex cells on the Zertz board
_NUM_DIRS = 6
_RADIUS = 3
_GS = GRID_SIZE * GRID_SIZE  # 49
_POLICY_CHANNELS = 10  # 4 place (W, G, B, remove) + 6 cap_dir
_CH_CAP_BASE = 4

# Inverse gather permutations for direction channels under each D6 symmetry.
# _D6_DIR_PERMS[sym][new_d] = old_d: "to build new channel new_d, gather from old channel old_d"
_D6_DIR_PERMS = [
    [0, 1, 2, 3, 4, 5],  # sym 0: identity
    [5, 0, 1, 2, 3, 4],  # sym 1: rot 1
    [4, 5, 0, 1, 2, 3],  # sym 2: rot 2
    [3, 4, 5, 0, 1, 2],  # sym 3: rot 3 (180°)
    [2, 3, 4, 5, 0, 1],  # sym 4: rot 4
    [1, 2, 3, 4, 5, 0],  # sym 5: rot 5
    [0, 5, 4, 3, 2, 1],  # sym 6: mirror (self-inverse)
    [1, 0, 5, 4, 3, 2],  # sym 7: mirror+rot1 (self-inverse)
    [2, 1, 0, 5, 4, 3],  # sym 8: mirror+rot2 (self-inverse)
    [3, 2, 1, 0, 5, 4],  # sym 9: mirror+rot3 (self-inverse)
    [4, 3, 2, 1, 0, 5],  # sym 10: mirror+rot4 (self-inverse)
    [5, 4, 3, 2, 1, 0],  # sym 11: mirror+rot5 (self-inverse)
]

_GRID_PERM_CACHE: list | None = None
_HEX_PERM_CACHE: list | None = None


def _load_grid_perms() -> list:
    global _GRID_PERM_CACHE
    if _GRID_PERM_CACHE is None:
        from engine_zero import zertz_d6_grid_permutations
        _GRID_PERM_CACHE = [np.array(p, dtype=np.int64) for p in zertz_d6_grid_permutations()]
    return _GRID_PERM_CACHE


def _load_hex_perms() -> list:
    global _HEX_PERM_CACHE
    if _HEX_PERM_CACHE is None:
        from engine_zero import zertz_d6_hex_permutations
        _HEX_PERM_CACHE = [np.array(p, dtype=np.int64) for p in zertz_d6_hex_permutations()]
    return _HEX_PERM_CACHE


# Dataset attribute names that map 1:1 to h5py datasets. Used by the
# detach/reattach lifecycle and the pickling/worker-init plumbing so we don't
# have to repeat the list in three places.
_H5_DATASETS = (
    "board_tensors", "reserve_vectors",
    "policy_targets",
    "value_targets",
    "value_only", "capture_turn", "mid_capture_turn",
)


def _h5_worker_init_fn(worker_id: int):
    """DataLoader worker init: open the unpickled dataset's h5 file 'r'.

    h5py.File handles don't pickle, so workers arrive with _h5file=None and
    all dataset attrs stripped (see ZertzDataset.__getstate__). Each worker
    opens its own libhdf5 handle; concurrent 'r' opens are safe given the
    HDF5_USE_FILE_LOCKING=FALSE module-level setting and the main process
    having closed its 'r+' handle via ZertzDataset.read_only_mode().
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    if hasattr(dataset, "_open_for_read"):
        dataset._open_for_read()


class ZertzDataset(Dataset):
    """Ring-buffer replay dataset for Zertz self-play positions.

    When `buf_dir` is given, all arrays are stored in a single HDF5 file
    (`replay.h5`) so the buffer survives process restarts. `_count`/`_size`
    live as HDF5 attributes in the same file — no separate metadata file.
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
                stored_policy_size = int(self._h5file["policy_targets"].shape[1]) if "policy_targets" in self._h5file else 4440
                if stored_policy_size != POLICY_SIZE:
                    raise ValueError(
                        f"Replay buffer policy format changed: stored {stored_policy_size} vs current {POLICY_SIZE}. "
                        f"Delete {h5path} to start fresh with the new format."
                    )
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

        self.board_tensors    = _ds("board_tensors",    (max_size, NUM_CHANNELS, GRID_SIZE, GRID_SIZE), np.float32)
        self.reserve_vectors  = _ds("reserve_vectors",  (max_size, RESERVE_SIZE),                       np.float32)
        self.policy_targets   = _ds("policy_targets",   (max_size, POLICY_SIZE),                        np.float32)
        self.value_targets    = _ds("value_targets",    (max_size,),                                    np.float32)
        self.value_only       = _ds("value_only",       (max_size,),                                    np.bool_)
        self.capture_turn     = _ds("capture_turn",     (max_size,),                                    np.bool_)
        self.mid_capture_turn = _ds("mid_capture_turn", (max_size,),                                    np.bool_)

        if self._h5file is not None:
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

    def add_batch(self, board_tensors: np.ndarray, reserve_vectors: np.ndarray,
                  policy_targets: np.ndarray, value_targets: np.ndarray,
                  value_only: list[bool],
                  capture_turn: list[bool] | None = None,
                  mid_capture_turn: list[bool] | None = None):
        n = board_tensors.shape[0]
        boards   = board_tensors.reshape(n, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        vo_arr   = np.asarray(value_only, dtype=np.bool_)
        ct_arr   = np.asarray(capture_turn, dtype=np.bool_) if capture_turn is not None else np.zeros(n, dtype=np.bool_)
        mct_arr  = np.asarray(mid_capture_turn, dtype=np.bool_) if mid_capture_turn is not None else np.zeros(n, dtype=np.bool_)

        pairs = [
            (self.board_tensors,    boards),
            (self.reserve_vectors,  reserve_vectors),
            (self.policy_targets,   policy_targets),
            (self.value_targets,    value_targets),
            (self.value_only,       vo_arr),
            (self.capture_turn,     ct_arr),
            (self.mid_capture_turn, mct_arr),
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

    def __getitem__(self, idx):
        """Returns the raw position plus a `sym_idx` integer in [0, 12).

        The D6 permutation is applied on GPU in Trainer._apply_sym_gpu after
        the batch lands on device — moves the per-sample numpy gather/channel-
        permutation work off the main thread.
        """
        if self.augment_symmetry:
            sym = idx % 12
            base_idx = idx // 12
        else:
            sym = 0
            base_idx = idx

        return (
            torch.from_numpy(self.board_tensors[base_idx].copy()),
            torch.from_numpy(self.reserve_vectors[base_idx].copy()),
            torch.from_numpy(self.policy_targets[base_idx].copy()),
            torch.tensor(self.value_targets[base_idx], dtype=torch.float32),
            torch.tensor(bool(self.value_only[base_idx]), dtype=torch.bool),
            torch.tensor(bool(self.capture_turn[base_idx]), dtype=torch.bool),
            torch.tensor(bool(self.mid_capture_turn[base_idx]), dtype=torch.bool),
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
            base = idx_arr // 12
            syms = idx_arr % 12
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
        vo         = read(self.value_only)
        cap        = read(self.capture_turn)
        mcap       = read(self.mid_capture_turn)

        out = []
        for i in range(n):
            out.append((
                torch.from_numpy(boards[i]),
                torch.from_numpy(reserves[i]),
                torch.from_numpy(policies[i]),
                torch.tensor(float(v_target[i]), dtype=torch.float32),
                torch.tensor(bool(vo[i]), dtype=torch.bool),
                torch.tensor(bool(cap[i]), dtype=torch.bool),
                torch.tensor(bool(mcap[i]), dtype=torch.bool),
                torch.tensor(int(syms[i]), dtype=torch.int64),
            ))
        return out


def _marginalize_policy(flat_policy: torch.Tensor):
    """Split flat policy[B, 490] into per-head targets.

    Layout: [place_W(49), place_G(49), place_B(49), remove(49),
             cap_E(49), cap_NE(49), cap_NW(49), cap_W(49), cap_SW(49), cap_SE(49)]

    Returns:
        place_cp: [B, 3*49] color/position targets (channels 0-2)
        place_rm: [B, 49] remove-ring targets (channel 3)
        cap_dir:  [B, 6*49] direction targets (6 channels × 49 grid cells)
    """
    place_cp = flat_policy[:, :3 * _GS]           # [B, 147]
    place_rm = flat_policy[:, 3 * _GS:4 * _GS]    # [B, 49]
    cap_dir  = flat_policy[:, 4 * _GS:]            # [B, 294]
    return place_cp, place_rm, cap_dir


def _head_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample cross-entropy: -sum(normalized_target * log_softmax(logits), dim=1).

    Normalizes target to sum to 1 per sample. Returns [N] losses.
    """
    t_sum = target.sum(dim=1, keepdim=True).clamp(min=1e-8)
    t_norm = target / t_sum
    return -(t_norm * torch.log_softmax(logits, dim=1)).sum(dim=1)


class Trainer:
    """Trains ZertzNet on self-play data using SGD with momentum."""

    def __init__(self, model: Optional[ZertzNet] = None,
                 weight_decay: float = 1e-4, device: str = "cpu",
                 lr: float = 0.02):
        self.device = torch.device(device)
        self.model = model or create_model()
        self.model.to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr,
            momentum=0.9, weight_decay=weight_decay,
        )
        self._compiled = torch.compile(self.model, dynamic=False) if self.device.type == "cuda" else self.model
        # Lazy-built GPU permutation tables (see _init_sym_buffers).
        self._grid_perms_gpu: Optional[torch.Tensor] = None
        self._ch_perms_gpu: Optional[torch.Tensor] = None

    @property
    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def _init_sym_buffers(self):
        """Move the D6 permutation tables to GPU.

        Two tables are needed:
          - grid_perms (12, GRID_SIZE²=49): spatial cell permutation. Entries
            equal to 49 are sentinels for off-board cells (the 7×7 grid has
            37 valid hex cells); the caller pads the input with a zero column
            to absorb them.
          - ch_perms (12, _POLICY_CHANNELS=10): policy-channel permutation that
            combines spatial-only place/remove channels (0-3) with direction-
            permuted cap_dir channels (4-9 via _D6_DIR_PERMS).
        """
        if self._grid_perms_gpu is not None:
            return
        grid = np.stack(_load_grid_perms()).astype(np.int64)  # (12, _GS)
        self._grid_perms_gpu = torch.from_numpy(grid).to(self.device)

        ch_table = np.empty((12, _POLICY_CHANNELS), dtype=np.int64)
        for s in range(12):
            # Place/remove channels (0-3): spatial-only, identity in channel space.
            for c in range(_CH_CAP_BASE):
                ch_table[s, c] = c
            # Cap_dir channels (4-9): permute directions via _D6_DIR_PERMS.
            dir_perm = _D6_DIR_PERMS[s]
            for new_d in range(_NUM_DIRS):
                old_d = int(dir_perm[new_d])
                ch_table[s, _CH_CAP_BASE + new_d] = _CH_CAP_BASE + old_d
        self._ch_perms_gpu = torch.from_numpy(ch_table).to(self.device)

    def _apply_sym_gpu(self, board: torch.Tensor, policy: torch.Tensor,
                       sym_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply per-sample D6 augmentation on GPU.

        Inputs are already on `self.device`. `sym_idx` is (B,) in [0, 12).
        Sym=0 is identity so `augment_symmetry=False` is a no-op pass through
        this helper. Mirrors the CPU augmentation: pads the spatial dim with a
        zero column so sentinel cells (== GRID_SIZE²) gather to 0; gathers
        cells per the grid perm; then gathers policy channels per the fused
        channel perm (place/remove are identity, cap_dir is direction-permuted).
        """
        self._init_sym_buffers()
        B, C, gs, _ = board.shape
        gs2 = gs * gs
        grid_perm_b = self._grid_perms_gpu[sym_idx]   # (B, gs²)
        ch_perm_b = self._ch_perms_gpu[sym_idx]       # (B, _POLICY_CHANNELS)

        # Board cell permutation: pad with a zero column so any sentinel
        # entry == gs² reads as 0.
        board_flat = board.view(B, C, gs2)
        zero_col_b = torch.zeros(B, C, 1, device=board.device, dtype=board.dtype)
        board_padded = torch.cat([board_flat, zero_col_b], dim=2)
        gp_bc = grid_perm_b.unsqueeze(1).expand(-1, C, -1)
        new_board = torch.gather(board_padded, 2, gp_bc).view(B, C, gs, gs)

        # Policy: spatial cell gather first, then channel-direction gather.
        pol_flat = policy.view(B, _POLICY_CHANNELS, gs2)
        zero_col_p = torch.zeros(B, _POLICY_CHANNELS, 1,
                                 device=policy.device, dtype=policy.dtype)
        pol_padded = torch.cat([pol_flat, zero_col_p], dim=2)
        gp_pc = grid_perm_b.unsqueeze(1).expand(-1, _POLICY_CHANNELS, -1)
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

    def train_epoch(self, dataset: ZertzDataset, batch_size: int = 256,
                    value_loss_scale: float = 1.0, num_workers: int = 0) -> dict:
        """Train one epoch with factorized policy heads. Returns loss dict.

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
        total_place_value_loss = 0.0
        total_capture_value_loss = 0.0
        total_place_policy_loss = 0.0
        total_capture_policy_loss = 0.0
        place_value_batches = 0
        capture_value_batches = 0
        place_policy_batches = 0
        capture_policy_batches = 0
        num_batches = 0

        device_type = self.device.type
        nb = self.device.type == "cuda"  # non_blocking only helps with pinned source
        for board, reserve, policy_target, value_target, vo_mask, cap_mask, mid_cap_mask, sym_idx in tqdm(
                loader, desc="  Training", leave=False, unit="batch"):
            board = board.to(self.device, non_blocking=nb)
            reserve = reserve.to(self.device, non_blocking=nb)
            policy_target = policy_target.to(self.device, non_blocking=nb)
            value_target = value_target.to(self.device, non_blocking=nb).unsqueeze(1)
            vo_mask = vo_mask.to(self.device, non_blocking=nb)
            cap_mask = cap_mask.to(self.device, non_blocking=nb)
            mid_cap_mask = mid_cap_mask.to(self.device, non_blocking=nb)  # kept for value logging split
            sym_idx = sym_idx.to(self.device, non_blocking=nb)

            # D6 symmetry augmentation runs here on GPU rather than per-sample
            # in the DataLoader worker. sym=0 is identity so augment_symmetry=
            # False is a no-op pass through this helper.
            board, policy_target = self._apply_sym_gpu(board, policy_target, sym_idx)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                place_logits, cap_dir_logits, value = self._compiled(board, reserve)

            # Convert flat policy to per-head targets
            place_cp_t, place_rm_t, cap_dir_t = _marginalize_policy(policy_target)

            # --- Policy loss ---
            policy_mask = (~vo_mask).float()
            policy_loss = torch.tensor(0.0, device=self.device)

            # Place turns: train place color/position (ch 0-2) and remove (ch 3) heads
            place_active = ~cap_mask & ~vo_mask
            if place_active.any():
                pw = policy_mask[place_active]
                n_place = place_active.sum().item()
                pp = torch.tensor(0.0, device=self.device)

                # Color/position head (channels 0-2, flat 3*49=147)
                cp_logits = place_logits[place_active, :3 * _GS]
                cp_target = place_cp_t[place_active]
                has_cp = cp_target.sum(dim=1) > 0
                if has_cp.any():
                    cp_loss = _head_ce(cp_logits[has_cp], cp_target[has_cp])
                    policy_loss = policy_loss + (cp_loss * pw[has_cp]).mean()
                    pp = pp + (cp_loss * pw[has_cp]).sum() / n_place

                # Remove head (channel 3, flat 49)
                rm_logits = place_logits[place_active, 3 * _GS:]
                rm_target = place_rm_t[place_active]
                has_rm = rm_target.sum(dim=1) > 0
                if has_rm.any():
                    rm_loss = _head_ce(rm_logits[has_rm], rm_target[has_rm])
                    policy_loss = policy_loss + (rm_loss * pw[has_rm]).mean()
                    pp = pp + (rm_loss * pw[has_rm]).sum() / n_place

                total_place_policy_loss += pp.item()
                place_policy_batches += 1

            # Capture turns (first-hop and mid-capture): train cap_dir head uniformly
            cap_active = cap_mask & ~vo_mask
            if cap_active.any():
                cw = policy_mask[cap_active]
                n_cap = cap_active.sum().item()
                c_loss = _head_ce(cap_dir_logits[cap_active], cap_dir_t[cap_active])
                policy_loss = policy_loss + (c_loss * cw).mean()
                total_capture_policy_loss += ((c_loss * cw).sum() / n_cap).item()
                capture_policy_batches += 1

            # --- Value loss ---
            per_sample_value = (value.squeeze(1) - value_target.squeeze(1)) ** 2
            value_loss = per_sample_value.mean()

            # --- Split logging ---
            place_mask = ~cap_mask
            if place_mask.any():
                total_place_value_loss += per_sample_value[place_mask].mean().item()
                place_value_batches += 1
            if cap_mask.any():
                total_capture_value_loss += per_sample_value[cap_mask].mean().item()
                capture_value_batches += 1

            loss = policy_loss + value_loss_scale * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0,
                    "place_value_loss": 0.0, "capture_value_loss": 0.0,
                    "place_policy_loss": 0.0, "capture_policy_loss": 0.0}

        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "total_loss": total_loss / num_batches,
            "place_value_loss": total_place_value_loss / place_value_batches if place_value_batches else 0.0,
            "capture_value_loss": total_capture_value_loss / capture_value_batches if capture_value_batches else 0.0,
            "place_policy_loss": total_place_policy_loss / place_policy_batches if place_policy_batches else 0.0,
            "capture_policy_loss": total_capture_policy_loss / capture_policy_batches if capture_policy_batches else 0.0,
        }
