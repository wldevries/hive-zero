"""Constants for Hive move encoding (bilinear Q·K policy layout for neural network output).

Actual encoding is done in Rust (engine_zero). This file only exports the
constants needed by the Python NN model and training code.

Policy layout: (5 + 2*BILINEAR_DIM) conceptual channels x grid_size x grid_size.

  [0 .. 5*G*G)              placement head: piece_type (0-4) x dest
  [5*G*G .. (5+D)*G*G)      Q embeddings: D channels x G*G cells
  [(5+D)*G*G .. (5+2D)*G*G) K embeddings: D channels x G*G cells

MCTS scoring:
  Placement (type, dest):  prior = policy[type * G² + dest_cell]
  Movement  (src,  dest):  prior = Q[src] · K[dst] / sqrt(D)
    where Q[src] = policy[5*G² + 0*G² + src .. (5+D)*G² + src, step=G²]
"""

from .board_encoder import DEFAULT_GRID_SIZE, GRID_SIZE

BILINEAR_DIM = 32           # D — embedding dimension for Q·K movement head
NUM_POLICY_CHANNELS = 5 + 2 * BILINEAR_DIM   # = 21
NUM_PLACE_CHANNELS = 5       # placement head channels
POLICY_SIZE = NUM_POLICY_CHANNELS * GRID_SIZE * GRID_SIZE  # default 21*23*23 = 11109


def policy_size(grid_size: int = DEFAULT_GRID_SIZE) -> int:
    return NUM_POLICY_CHANNELS * grid_size * grid_size


def place_section_size(grid_size: int) -> int:
    return NUM_PLACE_CHANNELS * grid_size * grid_size


def q_section_offset(grid_size: int) -> int:
    return NUM_PLACE_CHANNELS * grid_size * grid_size


def k_section_offset(grid_size: int) -> int:
    return (NUM_PLACE_CHANNELS + BILINEAR_DIM) * grid_size * grid_size
