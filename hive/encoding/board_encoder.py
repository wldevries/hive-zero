"""Constants for Hive board encoding (tensor layout for neural network input).

Actual encoding is done in Rust (hive_engine). This file only exports the
constants needed by the Python NN model and training code.
"""

# Default grid dimensions: 23x23 centered hex grid
DEFAULT_GRID_SIZE = 23

# Legacy alias — code that doesn't yet use parametric grid_size
GRID_SIZE = DEFAULT_GRID_SIZE

# 19 channels: 5 current piece types + 5 opponent piece types + 8 stacker depths + 1 stack height
NUM_CHANNELS = 19

# Reserve vector: 5 piece types x 2 colors = 10 values
RESERVE_SIZE = 10
