"""Zertz self-play package.

Module-level constants shared by `selfplay.py` and `battle.py`.
"""

# Typical post-split-ply max plies per game. Used as the initial pbar
# total (auto-extends if exceeded) and to size the replay buffer at
# `games_per_gen * TYPICAL_GAME_PLIES * replay_window`.
#
# Pre-split-ply this was 65; the placement-into-two-halves refactor
# (commit 55471c5) roughly doubled placement plies, pushing typical
# max game lengths to the low 80s. 85 covers most observed games
# without auto-extending.
TYPICAL_GAME_PLIES = 85
