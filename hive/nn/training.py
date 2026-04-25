"""Training loop for the Hive neural network.

The policy target is a **joint** distribution over every legal move at a
position — placements and movements share one softmax, matching what MCTS
consumes at inference. See docs/policy_heads.md.
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
        self.value_only      = _ds("value_only",      (max_size,),                                    np.bool_)
        self.policy_only     = _ds("policy_only",     (max_size,),                                    np.bool_)
        self.my_queen_danger  = _ds("my_queen_danger",  (max_size,), np.float32)
        self.opp_queen_danger = _ds("opp_queen_danger", (max_size,), np.float32)
        self.my_queen_escape  = _ds("my_queen_escape",  (max_size,), np.float32)
        self.opp_queen_escape = _ds("opp_queen_escape", (max_size,), np.float32)
        self.my_mobility      = _ds("my_mobility",      (max_size,), np.float32)
        self.opp_mobility     = _ds("opp_mobility",     (max_size,), np.float32)

        if self._h5file is not None:
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
                  opp_mobility: np.ndarray | None = None):
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
        if self.augment_symmetry:
            sym = idx % 12
            base_idx = idx // 12
        else:
            sym = 0
            base_idx = idx

        board = self.board_tensors[base_idx]
        p_idx = self.place_idx[base_idx].copy()
        p_prob = self.place_probs[base_idx].copy()
        n_p = int(self.num_placements[base_idx])
        m_src = self.movement_src[base_idx].copy()
        m_dst = self.movement_dst[base_idx].copy()
        m_prob = self.movement_probs[base_idx].copy()
        n_m = int(self.num_movements[base_idx])

        # Rotations 1,2,4,5 and all 6 mirrors involve s=-(q+r) and clip pieces
        # in the square-grid corners (|s|>half_grid). Skip those augmentations.
        # Identity (sym=0) and 180° (sym=3) never clip.
        if sym not in (0, 3) and self.augment_symmetry:
            bad = _load_bad_cells(self.grid_size)
            if board[:10].reshape(10, -1)[:, bad].any():
                sym = 0

        if sym != 0:
            gs = self.grid_size
            gc = gs * gs
            sym_perms = _load_sym_perms(gs)
            perm = sym_perms[sym]

            # Board: (NUM_CHANNELS, gs, gs) → permute spatial cells
            bf = board.reshape(NUM_CHANNELS, gc)
            padded = np.concatenate([bf, np.zeros((NUM_CHANNELS, 1), dtype=np.float32)], axis=1)
            board = padded[:, perm].reshape(NUM_CHANNELS, gs, gs)

            # Placement indices: idx = type * gc + cell. Permute the cell part.
            if n_p > 0:
                flat = p_idx[:n_p].astype(np.int64)
                type_part = flat // gc
                cell_part = flat % gc
                new_cell = perm[cell_part]
                valid_p = new_cell < gc
                new_flat = (type_part * gc + new_cell).astype(np.uint16)
            else:
                valid_p = np.zeros(0, dtype=bool)
                new_flat = np.zeros(0, dtype=np.uint16)

            # Movement pairs: permute src and dst; drop pairs with OOB src or dst.
            if n_m > 0:
                new_src = perm[m_src[:n_m]]
                new_dst = perm[m_dst[:n_m]]
                valid_m = (new_src < gc) & (new_dst < gc)
            else:
                valid_m = np.zeros(0, dtype=bool)
                new_src = np.zeros(0, dtype=np.int64)
                new_dst = np.zeros(0, dtype=np.int64)

            # Renormalize the joint target across surviving placements + movements.
            kept_p_prob = p_prob[:n_p][valid_p] if n_p > 0 else np.zeros(0, dtype=np.float32)
            kept_m_prob = m_prob[:n_m][valid_m] if n_m > 0 else np.zeros(0, dtype=np.float32)
            total = float(kept_p_prob.sum() + kept_m_prob.sum())
            if total > 0:
                kept_p_prob = kept_p_prob / total
                kept_m_prob = kept_m_prob / total

            # Write back (pad with zeros).
            n_p_new = len(kept_p_prob)
            p_idx = np.zeros(MAX_PLACEMENTS, dtype=np.uint16)
            p_prob = np.zeros(MAX_PLACEMENTS, dtype=np.float32)
            if n_p_new > 0:
                p_idx[:n_p_new] = new_flat[valid_p][:MAX_PLACEMENTS]
                p_prob[:n_p_new] = kept_p_prob[:MAX_PLACEMENTS]
            n_p = min(n_p_new, MAX_PLACEMENTS)

            n_m_new = len(kept_m_prob)
            m_src = np.zeros(MAX_MOVE_PAIRS, dtype=np.uint16)
            m_dst = np.zeros(MAX_MOVE_PAIRS, dtype=np.uint16)
            m_prob = np.zeros(MAX_MOVE_PAIRS, dtype=np.float32)
            if n_m_new > 0:
                m_src[:n_m_new] = new_src[valid_m][:MAX_MOVE_PAIRS].astype(np.uint16)
                m_dst[:n_m_new] = new_dst[valid_m][:MAX_MOVE_PAIRS].astype(np.uint16)
                m_prob[:n_m_new] = kept_m_prob[:MAX_MOVE_PAIRS]
            n_m = min(n_m_new, MAX_MOVE_PAIRS)
        else:
            board = board.copy()

        aux_targets = np.array([
            self.my_queen_danger[base_idx], self.opp_queen_danger[base_idx],
            self.my_queen_escape[base_idx], self.opp_queen_escape[base_idx],
            self.my_mobility[base_idx], self.opp_mobility[base_idx],
        ], dtype=np.float32)

        return (
            torch.from_numpy(board),
            torch.from_numpy(self.reserve_vectors[base_idx].copy()),
            torch.from_numpy(p_idx.astype(np.int64)),
            torch.from_numpy(p_prob),
            torch.tensor(n_p, dtype=torch.int32),
            torch.from_numpy(m_src.astype(np.int64)),
            torch.from_numpy(m_dst.astype(np.int64)),
            torch.from_numpy(m_prob),
            torch.tensor(n_m, dtype=torch.int32),
            torch.tensor(self.value_targets[base_idx], dtype=torch.float32),
            torch.tensor(self.value_only[base_idx], dtype=torch.bool),
            torch.tensor(self.policy_only[base_idx], dtype=torch.bool),
            torch.from_numpy(aux_targets),
        )


class Trainer:
    """Trains HiveNet on joint-softmax policy targets."""

    def __init__(self, model: Optional[HiveNet] = None,
                 weight_decay: float = 1e-4, device: str = "cpu",
                 lr: float = 0.02):
        self.device = torch.device(device)
        self.model = model or create_model()
        self.model.to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr,
            momentum=0.9, weight_decay=weight_decay,
        )
        self._compiled = torch.compile(self.model, dynamic=True, backend="cudagraphs") if self.device.type == "cuda" else self.model

    @property
    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def train_epoch(self, dataset: HiveDataset, batch_size: int = 64, value_loss_scale: float = 1.0, aux_loss_scale: float = 1.0) -> dict:
        """Train one epoch. Returns loss dict."""
        self.model.train()
        drop = self.device.type == "cuda"
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_aux_loss = 0.0
        total_qd_loss = 0.0
        total_qe_loss = 0.0
        total_mob_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        D = self.model.bilinear_dim
        device_type = self.device.type
        device = self.device

        for (board, reserve, p_idx, p_prob, n_p,
             m_src, m_dst, m_prob, n_m,
             value_target, vo_mask, po_mask, aux_target) in tqdm(
                loader, desc="  Training", leave=False, unit="batch"):
            board = board.to(device)
            reserve = reserve.to(device)
            p_idx = p_idx.to(device)     # (B, MAX_PLACEMENTS) int64
            p_prob = p_prob.to(device)   # (B, MAX_PLACEMENTS) float32
            n_p = n_p.to(device)         # (B,) int32
            m_src = m_src.to(device)
            m_dst = m_dst.to(device)
            m_prob = m_prob.to(device)
            n_m = n_m.to(device)
            value_target = value_target.to(device).unsqueeze(1)
            vo_mask = vo_mask.to(device)
            po_mask = po_mask.to(device)
            aux_target = aux_target.to(device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                policy_logits, wdl, aux = self._compiled(board, reserve)

            B = board.size(0)
            gs = board.size(-1)
            gs2 = gs * gs
            place_end = NUM_PLACE_CHANNELS * gs2

            place_logits_flat = policy_logits[:, :place_end]                   # (B, 5*G²)
            q = policy_logits[:, place_end:place_end + D * gs2].view(B, gs2, D)  # (B, G², D)
            k = policy_logits[:, place_end + D * gs2:].view(B, gs2, D)           # (B, G², D)

            # Gather placement logits at legal indices.
            p_gather = torch.gather(place_logits_flat, 1, p_idx)               # (B, MAX_PLACEMENTS)

            # Bilinear movement logits.
            src_exp = m_src.unsqueeze(-1).expand(-1, -1, D)
            dst_exp = m_dst.unsqueeze(-1).expand(-1, -1, D)
            q_g = torch.gather(q, 1, src_exp)                                  # (B, MAX_MOVE_PAIRS, D)
            k_g = torch.gather(k, 1, dst_exp)                                  # (B, MAX_MOVE_PAIRS, D)
            m_logits = (q_g * k_g).sum(-1) / math.sqrt(D)                      # (B, MAX_MOVE_PAIRS)

            # One softmax over the combined legal-action set — matches MCTS.
            combined_logits = torch.cat([p_gather, m_logits], dim=1)
            combined_probs = torch.cat([p_prob, m_prob], dim=1)

            p_mask = (torch.arange(MAX_PLACEMENTS, device=device).unsqueeze(0)
                      < n_p.unsqueeze(1).long())
            m_mask = (torch.arange(MAX_MOVE_PAIRS, device=device).unsqueeze(0)
                      < n_m.unsqueeze(1).long())
            c_mask = torch.cat([p_mask, m_mask], dim=1)

            combined_logits = combined_logits.masked_fill(~c_mask, float('-inf'))
            log_probs = torch.log_softmax(combined_logits, dim=1)
            log_probs = log_probs.masked_fill(~c_mask, 0.0)  # avoid 0 * -inf = NaN

            per_sample_policy = -(combined_probs * log_probs).sum(dim=1)
            has_target = ((n_p + n_m) > 0)
            per_sample_policy = per_sample_policy * has_target.float()

            policy_weight = (~vo_mask).float()
            policy_loss = (per_sample_policy * policy_weight).mean()

            # Value loss: cross-entropy on WDL soft targets, masked for policy-only samples.
            wdl_target = _scalar_to_wdl(value_target.squeeze(1))  # (B, 3)
            log_wdl = torch.log(wdl.float().clamp(min=1e-7))
            per_sample_value = -(wdl_target * log_wdl).sum(dim=1)
            value_weight = (~po_mask).float()
            value_loss = (per_sample_value * value_weight).mean()

            # Auxiliary losses: MSE on all 6 outputs, always active.
            aux_mse = (aux - aux_target) ** 2
            qd_loss = aux_mse[:, 0:2].mean()
            qe_loss = aux_mse[:, 2:4].mean()
            mob_loss = aux_mse[:, 4:6].mean()
            aux_loss = qd_loss + qe_loss + mob_loss

            loss = policy_loss + value_loss_scale * value_loss + aux_loss_scale * aux_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_aux_loss += aux_loss.item()
            total_qd_loss += qd_loss.item()
            total_qe_loss += qe_loss.item()
            total_mob_loss += mob_loss.item()
            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            return {"policy_loss": 0, "value_loss": 0, "qd_loss": 0,
                    "qe_loss": 0, "mob_loss": 0, "aux_loss": 0, "total_loss": 0}

        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "aux_loss": total_aux_loss / num_batches,
            "qd_loss": total_qd_loss / num_batches,
            "qe_loss": total_qe_loss / num_batches,
            "mob_loss": total_mob_loss / num_batches,
            "total_loss": total_loss / num_batches,
        }
