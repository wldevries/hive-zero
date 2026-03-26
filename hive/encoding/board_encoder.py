"""Constants for Hive board encoding (tensor layout for neural network input).

Actual encoding is done in Rust (hive_engine). This file only exports the
constants needed by the Python NN model and training code.
"""

# Default grid dimensions: 23x23 centered hex grid
DEFAULT_GRID_SIZE = 23

# Legacy alias — code that doesn't yet use parametric grid_size
GRID_SIZE = DEFAULT_GRID_SIZE

# 39 channels: 11 current pieces + 11 opponent pieces + 16 stacked beetles + 1 stack height
NUM_CHANNELS = 39

# Reserve vector: 5 piece types x 2 colors = 10 values
RESERVE_SIZE = 10
