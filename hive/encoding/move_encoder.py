"""Constants for Hive move encoding (policy layout for neural network output).

Actual encoding is done in Rust (hive_engine). This file only exports the
constants needed by the Python NN model and training code.

Policy layout: 11 channels x grid_size x grid_size total policy logits.
Channel = piece index within current player (0-10).
"""

from .board_encoder import DEFAULT_GRID_SIZE, GRID_SIZE

NUM_POLICY_CHANNELS = 11
POLICY_SIZE = NUM_POLICY_CHANNELS * GRID_SIZE * GRID_SIZE  # default 5,819


def policy_size(grid_size: int = DEFAULT_GRID_SIZE) -> int:
    return NUM_POLICY_CHANNELS * grid_size * grid_size
