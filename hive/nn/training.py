"""Training loop for the Hive neural network."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional
from tqdm import tqdm

from .model import HiveNet, create_model, save_model
from ..encoding.board_encoder import NUM_CHANNELS, DEFAULT_GRID_SIZE, RESERVE_SIZE
from ..encoding.move_encoder import (
    NUM_POLICY_CHANNELS, NUM_PLACE_CHANNELS,
    policy_size as compute_policy_size,
    src_section_offset, dst_section_offset,
)

# ---------------------------------------------------------------------------
# Hex D6 symmetry: lazy-loaded per grid_size.
# ---------------------------------------------------------------------------
_SYM_PERMS_CACHE: dict[int, list] = {}

def _load_sym_perms(grid_size: int):
    if grid_size not in _SYM_PERMS_CACHE:
        from hive_engine import d6_grid_permutations
        _SYM_PERMS_CACHE[grid_size] = [np.array(p) for p in d6_grid_permutations(grid_size)]
    return _SYM_PERMS_CACHE[grid_size]


class HiveDataset(Dataset):
    """Dataset of (board_tensor, reserve_vector, policy_target, value_target) tuples.

    Acts as a replay buffer with a max capacity using pre-allocated numpy arrays
    and a ring buffer to avoid GC pressure from thousands of individual arrays.
    """

    def __init__(self, max_size: int = 50_000, grid_size: int = DEFAULT_GRID_SIZE):
        """Args:
            max_size: Maximum number of samples to keep.
            grid_size: Spatial grid dimension for NN encoding.
        """
        self.max_size = max_size
        self.grid_size = grid_size
        self._ps = compute_policy_size(grid_size)
        self._count = 0  # total samples added (for ring buffer index)
        self._size = 0   # current number of valid samples
        self.augment_symmetry = False  # set True to apply random D6 augmentation
        # Pre-allocate contiguous arrays
        self.board_tensors = np.zeros((max_size, NUM_CHANNELS, grid_size, grid_size), dtype=np.float32)
        self.reserve_vectors = np.zeros((max_size, RESERVE_SIZE), dtype=np.float32)
        self.policy_targets = np.zeros((max_size, self._ps), dtype=np.float32)
        self.value_targets = np.zeros(max_size, dtype=np.float32)
        self.weights = np.ones(max_size, dtype=np.float32)
        self.value_only = np.zeros(max_size, dtype=np.bool_)
        self.policy_only = np.zeros(max_size, dtype=np.bool_)
        self.my_queen_danger = np.zeros(max_size, dtype=np.float32)
        self.opp_queen_danger = np.zeros(max_size, dtype=np.float32)
        self.my_queen_escape = np.zeros(max_size, dtype=np.float32)
        self.opp_queen_escape = np.zeros(max_size, dtype=np.float32)
        self.my_mobility = np.zeros(max_size, dtype=np.float32)
        self.opp_mobility = np.zeros(max_size, dtype=np.float32)

    def add_sample(self, board_tensor: np.ndarray, reserve_vector: np.ndarray,
                   policy_target: np.ndarray, value_target: float,
                   weight: float = 1.0, value_only: bool = False, policy_only: bool = False,
                   my_queen_danger: float = 0.0, opp_queen_danger: float = 0.0,
                   my_queen_escape: float = 0.0, opp_queen_escape: float = 0.0,
                   my_mobility: float = 0.0, opp_mobility: float = 0.0):
        idx = self._count % self.max_size
        self.board_tensors[idx] = board_tensor
        self.reserve_vectors[idx] = reserve_vector
        self.policy_targets[idx] = policy_target
        self.value_targets[idx] = value_target
        self.weights[idx] = weight
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
                  policy_targets: np.ndarray, value_targets: np.ndarray,
                  weights: np.ndarray, value_only: list[bool], policy_only: list[bool],
                  my_queen_danger: np.ndarray | None = None,
                  opp_queen_danger: np.ndarray | None = None,
                  my_queen_escape: np.ndarray | None = None,
                  opp_queen_escape: np.ndarray | None = None,
                  my_mobility: np.ndarray | None = None,
                  opp_mobility: np.ndarray | None = None):
        """Bulk insert from contiguous arrays. Much faster than per-sample add."""
        n = board_tensors.shape[0]
        boards_flat = board_tensors.reshape(n, NUM_CHANNELS, self.grid_size, self.grid_size)
        for i in range(n):
            idx = self._count % self.max_size
            self.board_tensors[idx] = boards_flat[i]
            self.reserve_vectors[idx] = reserve_vectors[i]
            self.policy_targets[idx] = policy_targets[i]
            self.value_targets[idx] = value_targets[i]
            self.weights[idx] = weights[i]
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
        policy = self.policy_targets[base_idx]

        if sym != 0:
            gs = self.grid_size
            gc = gs * gs
            sym_perms = _load_sym_perms(gs)
            perm = sym_perms[sym]
            # Board: (NUM_CHANNELS, gs, gs) → permute spatial cells
            bf = board.reshape(NUM_CHANNELS, gc)
            padded = np.concatenate([bf, np.zeros((NUM_CHANNELS, 1), dtype=np.float32)], axis=1)
            board = padded[:, perm].reshape(NUM_CHANNELS, gs, gs)
            # Policy: (7*gs*gs,) → (7, gc) → permute → flatten
            pf = policy.reshape(NUM_POLICY_CHANNELS, gc)
            padded_p = np.concatenate([pf, np.zeros((NUM_POLICY_CHANNELS, 1), dtype=np.float32)], axis=1)
            policy = padded_p[:, perm].reshape(-1)
        else:
            board = board.copy()
            policy = policy.copy()

        # Auxiliary targets as a single tensor [qd, qd, qe, qe, mob, mob]
        aux_targets = np.array([
            self.my_queen_danger[base_idx], self.opp_queen_danger[base_idx],
            self.my_queen_escape[base_idx], self.opp_queen_escape[base_idx],
            self.my_mobility[base_idx], self.opp_mobility[base_idx],
        ], dtype=np.float32)
        return (
            torch.from_numpy(board),
            torch.from_numpy(self.reserve_vectors[base_idx].copy()),
            torch.from_numpy(policy),
            torch.tensor(self.value_targets[base_idx], dtype=torch.float32),
            torch.tensor(self.weights[base_idx], dtype=torch.float32),
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

        device_type = self.device.type
        for board, reserve, policy_target, value_target, weight, vo_mask, po_mask, aux_target in tqdm(loader, desc="  Training", leave=False, unit="batch"):
            board = board.to(self.device)
            reserve = reserve.to(self.device)
            policy_target = policy_target.to(self.device)
            value_target = value_target.to(self.device).unsqueeze(1)
            weight = weight.to(self.device)
            vo_mask = vo_mask.to(self.device)
            po_mask = po_mask.to(self.device)
            aux_target = aux_target.to(self.device)  # (batch, 6)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                policy_logits, value, aux = self.model(board, reserve)

            # Policy loss: factorized 3-head cross-entropy.
            # Flat policy = [place(5*G²) | src(G²) | dst(G²)]
            gs = board.size(-1)
            gs2 = gs * gs
            src_off = NUM_PLACE_CHANNELS * gs2
            dst_off = (NUM_PLACE_CHANNELS + 1) * gs2

            place_logits = policy_logits[:, :src_off]       # (B, 5*G²)
            src_logits   = policy_logits[:, src_off:dst_off] # (B, G²)
            dst_logits   = policy_logits[:, dst_off:]        # (B, G²)

            place_target = policy_target[:, :src_off]
            src_target   = policy_target[:, src_off:dst_off]
            dst_target   = policy_target[:, dst_off:]

            # Soft CE per head: -sum(target * log_softmax(logits))
            # Zero when all targets are zero (no placements or no movements in this position).
            def soft_ce(logits, target):
                return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1)

            per_sample_policy = soft_ce(place_logits, place_target) \
                               + soft_ce(src_logits, src_target) \
                               + soft_ce(dst_logits, dst_target)
            policy_weight = weight * (~vo_mask).float()
            policy_loss = (per_sample_policy * policy_weight).mean()

            # Value loss: weighted MSE, masked for policy-only samples (zero-heuristic draws)
            per_sample_value = (value.squeeze(1) - value_target.squeeze(1)) ** 2
            value_weight = weight * (~po_mask).float()
            value_loss = (per_sample_value * value_weight).mean()

            # Auxiliary losses: MSE on all 6 outputs, always active
            # aux[:, 0:2] = queen danger, aux[:, 2:4] = queen escape, aux[:, 4:6] = mobility
            aux_mse = (aux - aux_target) ** 2
            qd_loss = aux_mse[:, 0:2].mean()
            qe_loss = aux_mse[:, 2:4].mean()
            mob_loss = aux_mse[:, 4:6].mean()
            aux_loss = qd_loss + qe_loss + mob_loss

            # Combined loss
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
