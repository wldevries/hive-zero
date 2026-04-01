"""Constants for Hive move encoding (factorized policy layout for neural network output).

Actual encoding is done in Rust (hive_engine). This file only exports the
constants needed by the Python NN model and training code.

Policy layout: 11 conceptual channels x grid_size x grid_size = 11*G*G flat vector.

  [0 .. 5*G*G)       placement head: piece_type (0-4) x dest
  [5*G*G .. 6*G*G)   movement source head
  [6*G*G .. 11*G*G)  movement destination head: piece_type (0-4) x dest

MCTS scoring:
  Placement (type, dest):  prior = policy[type * G² + dest_cell]
  Movement  (src,  dest):  prior = policy[5*G² + src_cell] + policy[6*G² + type*G² + dest_cell]
"""

from .board_encoder import DEFAULT_GRID_SIZE, GRID_SIZE

NUM_POLICY_CHANNELS = 11     # 5 place + 1 src + 5 dst
NUM_PLACE_CHANNELS = 5       # placement head channels
POLICY_SIZE = NUM_POLICY_CHANNELS * GRID_SIZE * GRID_SIZE  # default 5,819


def policy_size(grid_size: int = DEFAULT_GRID_SIZE) -> int:
    return NUM_POLICY_CHANNELS * grid_size * grid_size


def place_section_size(grid_size: int) -> int:
    return NUM_PLACE_CHANNELS * grid_size * grid_size


def src_section_offset(grid_size: int) -> int:
    return NUM_PLACE_CHANNELS * grid_size * grid_size


def dst_section_offset(grid_size: int) -> int:
    return (NUM_PLACE_CHANNELS + 1) * grid_size * grid_size
