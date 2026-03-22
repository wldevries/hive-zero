"""Constants for Hive move encoding (policy layout for neural network output).

Actual encoding is done in Rust (hive_engine). This file only exports the
constants needed by the Python NN model and training code.

Policy layout: 11 channels x 23 x 23 grid = 5,819 total policy logits.
Channel = piece index within current player (0-10).
"""

from .board_encoder import GRID_SIZE

NUM_POLICY_CHANNELS = 11
POLICY_SIZE = NUM_POLICY_CHANNELS * GRID_SIZE * GRID_SIZE  # 5,819
