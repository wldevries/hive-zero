"""Training loop for the Hive neural network."""

from __future__ import annotations
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional
from tqdm import tqdm

from .model import HiveNet, create_model
from ..encoding.board_encoder import NUM_CHANNELS, DEFAULT_GRID_SIZE, RESERVE_SIZE
from ..encoding.move_encoder import (
    NUM_PLACE_CHANNELS,
    place_section_size,
)

MAX_MOVE_PAIRS = 256  # max legal movement pairs per position (src, dst, prob)

# ---------------------------------------------------------------------------
# Hex D6 symmetry: lazy-loaded per grid_size.
# ---------------------------------------------------------------------------
_SYM_PERMS_CACHE: dict[int, list] = {}

def _load_sym_perms(grid_size: int):
    if grid_size not in _SYM_PERMS_CACHE:
        from engine_zero import d6_grid_permutations
        _SYM_PERMS_CACHE[grid_size] = [np.array(p) for p in d6_grid_permutations(grid_size)]
    return _SYM_PERMS_CACHE[grid_size]


class HiveDataset(Dataset):
    """Dataset of training samples for the bilinear Q·K policy head.

    Policy targets are split into:
      - place_targets: (max_size, 5*G²) — placement visit distribution
      - movement_src/dst/probs: (max_size, MAX_MOVE_PAIRS) — sparse joint movement distribution
      - num_movements: (max_size,) — actual pair count per sample

    Acts as a replay buffer with ring-buffer eviction.
    """

    def __init__(self, max_size: int = 50_000, grid_size: int = DEFAULT_GRID_SIZE):
        self.max_size = max_size
        self.grid_size = grid_size
        self._place_size = place_section_size(grid_size)
        self._count = 0
        self._size = 0
        self.augment_symmetry = False

        # Pre-allocate contiguous arrays
        self.board_tensors = np.zeros((max_size, NUM_CHANNELS, grid_size, grid_size), dtype=np.float32)
        self.reserve_vectors = np.zeros((max_size, RESERVE_SIZE), dtype=np.float32)
        self.place_targets = np.zeros((max_size, self._place_size), dtype=np.float32)
        self.movement_src = np.zeros((max_size, MAX_MOVE_PAIRS), dtype=np.uint16)
        self.movement_dst = np.zeros((max_size, MAX_MOVE_PAIRS), dtype=np.uint16)
        self.movement_probs = np.zeros((max_size, MAX_MOVE_PAIRS), dtype=np.float32)
        self.num_movements = np.zeros(max_size, dtype=np.int32)
        self.value_targets = np.zeros(max_size, dtype=np.float32)
        self.value_only = np.zeros(max_size, dtype=np.bool_)
        self.policy_only = np.zeros(max_size, dtype=np.bool_)
        self.my_queen_danger = np.zeros(max_size, dtype=np.float32)
        self.opp_queen_danger = np.zeros(max_size, dtype=np.float32)
        self.my_queen_escape = np.zeros(max_size, dtype=np.float32)
        self.opp_queen_escape = np.zeros(max_size, dtype=np.float32)
        self.my_mobility = np.zeros(max_size, dtype=np.float32)
        self.opp_mobility = np.zeros(max_size, dtype=np.float32)

    def add_sample(self, board_tensor: np.ndarray, reserve_vector: np.ndarray,
                   place_target: np.ndarray,
                   movement_src_cells,  # list/array of int, length <= MAX_MOVE_PAIRS
                   movement_dst_cells,  # list/array of int
                   movement_probs_arr,  # list/array of float
                   value_target: float,
                   value_only: bool = False, policy_only: bool = False,
                   my_queen_danger: float = 0.0, opp_queen_danger: float = 0.0,
                   my_queen_escape: float = 0.0, opp_queen_escape: float = 0.0,
                   my_mobility: float = 0.0, opp_mobility: float = 0.0):
        idx = self._count % self.max_size
        self.board_tensors[idx] = board_tensor
        self.reserve_vectors[idx] = reserve_vector
        self.place_targets[idx] = place_target
        n = min(len(movement_src_cells), MAX_MOVE_PAIRS)
        self.movement_src[idx, :n] = movement_src_cells[:n]
        self.movement_dst[idx, :n] = movement_dst_cells[:n]
        self.movement_probs[idx, :n] = movement_probs_arr[:n]
        if n < MAX_MOVE_PAIRS:
            self.movement_src[idx, n:] = 0
            self.movement_dst[idx, n:] = 0
            self.movement_probs[idx, n:] = 0.0
        self.num_movements[idx] = n
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
                  place_targets: np.ndarray,
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
        boards_flat = board_tensors.reshape(n, NUM_CHANNELS, self.grid_size, self.grid_size)
        for i in range(n):
            idx = self._count % self.max_size
            self.board_tensors[idx] = boards_flat[i]
            self.reserve_vectors[idx] = reserve_vectors[i]
            self.place_targets[idx] = place_targets[i]
            self.movement_src[idx] = movement_src_data[i]
            self.movement_dst[idx] = movement_dst_data[i]
            self.movement_probs[idx] = movement_prob_data[i]
            self.num_movements[idx] = num_movements_arr[i]
            self.value_targets[idx] = value_targets[i]
            self.value_only[idx] = value_only[i]
            self.policy_only[idx] = policy_only[i]
            self.my_queen_danger[idx] = my_queen_danger[i] if my_queen_danger is not None else 0.0
            self.opp_queen_danger[idx] = opp_queen_danger[i] if opp_queen_danger is not None else 0.0
            self.my_queen_escape[idx] = my_queen_escape[i] if my_queen_escape is not None else 0.0
            self.opp_queen_escape[idx] = opp_queen_escape[i] if opp_queen_escape is not None else 0.0
            self.my_mobility[idx] = my_mobility[i] if my_mobility is not None else 0.0
            self.opp_mobility[idx] = opp_mobility[i] if opp_mobility is not None else 0.0
            self._count += 1
            self._size = min(self._size + 1, self.max_size)

    def clear(self):
        self._count = 0
        self._size = 0

    def __len__(self):
        return self._size * 12 if self.augment_symmetry else self._size

    def __getitem__(self, idx):
        if self.augment_symmetry:
            sym = idx % 12
            base_idx = idx // 12
        else:
            sym = 0
            base_idx = idx

        board = self.board_tensors[base_idx]
        place = self.place_targets[base_idx]
        mv_src = self.movement_src[base_idx].copy()
        mv_dst = self.movement_dst[base_idx].copy()
        mv_prob = self.movement_probs[base_idx].copy()
        n_mv = int(self.num_movements[base_idx])

        if sym != 0:
            gs = self.grid_size
            gc = gs * gs
            sym_perms = _load_sym_perms(gs)
            perm = sym_perms[sym]

            # Board: (NUM_CHANNELS, gs, gs) → permute spatial cells
            bf = board.reshape(NUM_CHANNELS, gc)
            padded = np.concatenate([bf, np.zeros((NUM_CHANNELS, 1), dtype=np.float32)], axis=1)
            board = padded[:, perm].reshape(NUM_CHANNELS, gs, gs)

            # Placement: (5, gc) → permute → flatten
            pf = place.reshape(NUM_PLACE_CHANNELS, gc)
            padded_p = np.concatenate([pf, np.zeros((NUM_PLACE_CHANNELS, 1), dtype=np.float32)], axis=1)
            place = padded_p[:, perm].reshape(-1)

            # Movement pairs: apply perm to src and dst cells
            if n_mv > 0:
                mv_src[:n_mv] = perm[mv_src[:n_mv]]
                mv_dst[:n_mv] = perm[mv_dst[:n_mv]]
        else:
            board = board.copy()
            place = place.copy()

        aux_targets = np.array([
            self.my_queen_danger[base_idx], self.opp_queen_danger[base_idx],
            self.my_queen_escape[base_idx], self.opp_queen_escape[base_idx],
            self.my_mobility[base_idx], self.opp_mobility[base_idx],
        ], dtype=np.float32)

        return (
            torch.from_numpy(board),
            torch.from_numpy(self.reserve_vectors[base_idx].copy()),
            torch.from_numpy(place),
            torch.from_numpy(mv_src.astype(np.int64)),
            torch.from_numpy(mv_dst.astype(np.int64)),
            torch.from_numpy(mv_prob),
            torch.tensor(n_mv, dtype=torch.int32),
            torch.tensor(self.value_targets[base_idx], dtype=torch.float32),
            torch.tensor(self.value_only[base_idx], dtype=torch.bool),
            torch.tensor(self.policy_only[base_idx], dtype=torch.bool),
            torch.from_numpy(aux_targets),
        )


class Trainer:
    """Trains the HiveNet model on self-play data using SGD with momentum."""

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

    @property
    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def train_epoch(self, dataset: HiveDataset, batch_size: int = 64, value_loss_scale: float = 1.0) -> dict:
        """Train one epoch. Returns loss dict."""
        self.model.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

        for board, reserve, place_target, mv_src, mv_dst, mv_probs, n_mv, \
                value_target, vo_mask, po_mask, aux_target in tqdm(loader, desc="  Training", leave=False, unit="batch"):
            board = board.to(self.device)
            reserve = reserve.to(self.device)
            place_target = place_target.to(self.device)
            mv_src = mv_src.to(self.device)       # (B, MAX_MOVE_PAIRS) int64
            mv_dst = mv_dst.to(self.device)       # (B, MAX_MOVE_PAIRS) int64
            mv_probs = mv_probs.to(self.device)   # (B, MAX_MOVE_PAIRS) float32
            n_mv = n_mv.to(self.device)           # (B,) int32
            value_target = value_target.to(self.device).unsqueeze(1)
            vo_mask = vo_mask.to(self.device)
            po_mask = po_mask.to(self.device)
            aux_target = aux_target.to(self.device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                policy_logits, value, aux = self.model(board, reserve)

            gs = board.size(-1)
            gs2 = gs * gs
            place_end = NUM_PLACE_CHANNELS * gs2

            place_logits = policy_logits[:, :place_end]                   # (B, 5*G²)
            q = policy_logits[:, place_end:place_end + D * gs2]           # (B, D*G²)
            k = policy_logits[:, place_end + D * gs2:]                    # (B, D*G²)

            B = board.size(0)
            q = q.view(B, gs2, D)  # (B, G², D)
            k = k.view(B, gs2, D)  # (B, G², D)

            # Placement loss: soft CE
            def soft_ce(logits, target):
                return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1)

            place_loss = soft_ce(place_logits, place_target)

            # Bilinear movement loss: gather Q[src] and K[dst] per pair, dot product
            src_idx = mv_src.unsqueeze(-1).expand(-1, -1, D)   # (B, MAX_MOVE_PAIRS, D)
            dst_idx = mv_dst.unsqueeze(-1).expand(-1, -1, D)   # (B, MAX_MOVE_PAIRS, D)
            q_gathered = torch.gather(q, 1, src_idx)            # (B, MAX_MOVE_PAIRS, D)
            k_gathered = torch.gather(k, 1, dst_idx)            # (B, MAX_MOVE_PAIRS, D)
            move_logits = (q_gathered * k_gathered).sum(-1) / (D ** 0.5)  # (B, MAX_MOVE_PAIRS)

            # Mask padded entries (i >= num_movements)
            move_mask = (torch.arange(MAX_MOVE_PAIRS, device=self.device)
                         .unsqueeze(0) < n_mv.unsqueeze(1))
            move_logits = move_logits.masked_fill(~move_mask, float('-inf'))

            # CE loss; zero out padded positions to avoid 0 * -inf = NaN
            move_log_probs = torch.log_softmax(move_logits, dim=1)
            move_log_probs = move_log_probs.masked_fill(~move_mask, 0.0)
            move_loss = -(mv_probs * move_log_probs).sum(dim=1)
            move_loss = move_loss * (n_mv > 0).float()

            per_sample_policy = place_loss + move_loss
            policy_weight = (~vo_mask).float()
            policy_loss = (per_sample_policy * policy_weight).mean()

            # Value loss: MSE, masked for policy-only samples
            per_sample_value = (value.squeeze(1) - value_target.squeeze(1)) ** 2
            value_weight = (~po_mask).float()
            value_loss = (per_sample_value * value_weight).mean()

            # Auxiliary losses: MSE on all 6 outputs, always active
            aux_mse = (aux - aux_target) ** 2
            qd_loss = aux_mse[:, 0:2].mean()
            qe_loss = aux_mse[:, 2:4].mean()
            mob_loss = aux_mse[:, 4:6].mean()
            aux_loss = qd_loss + qe_loss + mob_loss

            loss = policy_loss + value_loss_scale * value_loss + aux_loss

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
