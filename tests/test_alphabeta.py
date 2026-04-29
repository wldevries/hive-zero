"""Smoke tests for the Hive alphabeta search."""

import random
import time

import pytest

from engine_zero import HiveGame


def test_returns_valid_uhp_move():
    g = HiveGame()
    mv = g.best_move_alphabeta(2)
    assert isinstance(mv, str)
    assert len(mv) > 0


def test_depth_3_within_budget():
    """Plan budget: depth 3 < 2s on a representative mid-opening position."""
    random.seed(1)
    g = HiveGame()
    for _ in range(14):
        moves = g.valid_moves()
        if not moves:
            break
        piece, frm, to = random.choice(moves)
        g.play_move(piece, frm, to)

    t = time.time()
    g.best_move_alphabeta(3)
    elapsed = time.time() - t
    assert elapsed < 2.0, f"depth 3 took {elapsed:.2f}s, exceeds 2s budget"


def _play_game_alphabeta_vs_random(ab_is_white: bool, depth: int, seed: int, max_moves: int = 120):
    random.seed(seed)
    g = HiveGame()
    while not g.is_game_over and g.move_count < max_moves:
        ab_turn = (g.move_count % 2 == 0) == ab_is_white
        if ab_turn:
            uhp = g.best_move_alphabeta(depth)
            if uhp == "pass":
                g.play_pass()
            else:
                ok = g.play_move_uhp(uhp)
                assert ok, f"alphabeta returned unplayable UHP move: {uhp}"
        else:
            moves = g.valid_moves()
            if not moves:
                g.play_pass()
            else:
                piece, frm, to = random.choice(moves)
                g.play_move(piece, frm, to)
    return g.state


def test_selfplay_with_bot_produces_decisive_games():
    """End-to-end: bot_frac=1.0 self-play produces decisive games (or repetition
    draws) instead of timeouts. Verifies value_only flag is set for bot turns
    and the value-target signal is non-trivial.
    """
    import numpy as np
    from engine_zero import RustSelfPlaySession

    POLICY_SIZE = 69 * 23 * 23  # NUM_POLICY_CHANNELS * grid_size^2

    session = RustSelfPlaySession(
        num_games=4,
        simulations=20,
        max_moves=120,
        grid_size=23,
        bot_frac=1.0,
        bot_depth=2,
    )

    def eval_fn(boards_bf16, reserves_bf16):
        n = boards_bf16.shape[0]
        policy = np.ones((n, POLICY_SIZE), dtype=np.float32) / POLICY_SIZE
        wdl = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
        return policy, wdl

    result = session.play_games(eval_fn=eval_fn)
    assert result.draws_timeout == 0, "bot opponent should prevent timeout draws"
    assert result.num_samples > 0, "no training samples emitted"

    _, _, _, values, vo, _, _, _ = result.training_data()
    assert sum(vo) > 0, "no bot turns in emitted samples (expected ~half)"
    assert sum(vo) < len(vo), "every emitted turn was bot-only — MCTS turns missing"
    assert float(values.std()) > 0.0, "value targets have zero variance — head won't learn"


@pytest.mark.parametrize("n_games", [10])
def test_beats_random_overwhelmingly(n_games):
    """Depth-2 alphabeta should crush a random-move opponent (>90% win rate).

    Plays alternating colors so a side bias in either alphabeta or the random
    opponent doesn't skew the result.
    """
    ab_wins = 0
    decisive = 0
    for i in range(n_games):
        ab_is_white = i % 2 == 0
        state = _play_game_alphabeta_vs_random(ab_is_white, depth=2, seed=i)
        ab_won = (
            (ab_is_white and state == "WhiteWins")
            or (not ab_is_white and state == "BlackWins")
        )
        if state in ("WhiteWins", "BlackWins"):
            decisive += 1
        if ab_won:
            ab_wins += 1
    assert decisive >= int(0.9 * n_games), f"only {decisive}/{n_games} games decisive"
    assert ab_wins >= int(0.9 * n_games), f"alphabeta only won {ab_wins}/{n_games}"
