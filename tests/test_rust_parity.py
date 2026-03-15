"""Cross-validation tests: Python Game vs Rust RustGame.

Ensures both implementations produce identical results for:
- valid_moves() at every step
- Board encoding (board tensor + reserve vector)
- Move encoding (policy indices)
"""

import pytest
import numpy as np
import random

try:
    from hive_engine import RustGame
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

from hive.core.game import Game
from hive.core.pieces import Piece, PieceColor, PieceType
from hive.encoding.board_encoder import encode_board
from hive.encoding.move_encoder import encode_move, get_legal_move_mask


def skip_if_no_rust():
    if not HAS_RUST:
        pytest.skip("Rust extension not built")


def py_moves_to_set(moves):
    """Convert Python valid_moves list to comparable set."""
    result = set()
    for piece, from_pos, to_pos in moves:
        piece_str = str(piece)
        from_tuple = (from_pos.q, from_pos.r) if from_pos is not None else None
        to_tuple = (to_pos.q, to_pos.r)
        result.add((piece_str, from_tuple, to_tuple))
    return result


def rust_moves_to_set(moves):
    """Convert Rust valid_moves list to comparable set."""
    return set((p, f, t) for p, f, t in moves)


class TestValidMovesParity:
    """Test that Python and Rust produce identical valid_moves."""

    def test_initial_moves(self):
        skip_if_no_rust()
        py_game = Game()
        rs_game = RustGame()

        py_moves = py_moves_to_set(py_game.valid_moves())
        rs_moves = rust_moves_to_set(rs_game.valid_moves())
        assert py_moves == rs_moves, f"Initial moves differ:\nPy: {py_moves}\nRs: {rs_moves}"

    def test_after_first_move(self):
        skip_if_no_rust()
        py_game = Game()
        rs_game = RustGame()

        # Play wQ1 at origin
        py_piece = Piece(PieceColor.WHITE, PieceType.QUEEN, 1)
        from hive.core.hex import Hex
        py_game.play_move(py_piece, None, Hex(0, 0))
        rs_game.play_move("wQ1", None, (0, 0))

        py_moves = py_moves_to_set(py_game.valid_moves())
        rs_moves = rust_moves_to_set(rs_game.valid_moves())
        assert py_moves == rs_moves

    def test_random_game_parity(self):
        """Play random games and verify moves match at every step."""
        skip_if_no_rust()
        random.seed(42)

        for game_idx in range(5):
            py_game = Game()
            rs_game = RustGame()

            for move_num in range(60):
                py_valid = py_game.valid_moves()
                py_set = py_moves_to_set(py_valid)
                rs_set = rust_moves_to_set(rs_game.valid_moves())

                assert py_set == rs_set, (
                    f"Game {game_idx}, move {move_num}: valid_moves differ.\n"
                    f"Py only: {py_set - rs_set}\n"
                    f"Rs only: {rs_set - py_set}"
                )

                if not py_valid:
                    # Both must pass
                    py_game.play_pass()
                    rs_game.play_pass()
                    continue

                # Pick a random move
                idx = random.randint(0, len(py_valid) - 1)
                piece, from_pos, to_pos = py_valid[idx]
                piece_str = str(piece)
                from_tuple = (from_pos.q, from_pos.r) if from_pos is not None else None
                to_tuple = (to_pos.q, to_pos.r)

                py_game.play_move(piece, from_pos, to_pos)
                rs_game.play_move(piece_str, from_tuple, to_tuple)

                if py_game.is_game_over:
                    assert rs_game.is_game_over
                    break


class TestEncodingParity:
    """Test that Python and Rust encoding produce identical tensors."""

    def test_board_encoding_empty(self):
        skip_if_no_rust()
        py_game = Game()
        rs_game = RustGame()

        py_board, py_reserve = encode_board(py_game)
        rs_board, rs_reserve = rs_game.encode_board()

        np.testing.assert_array_almost_equal(py_board, rs_board, decimal=6,
            err_msg="Board tensor mismatch on empty board")
        np.testing.assert_array_almost_equal(py_reserve, rs_reserve, decimal=6,
            err_msg="Reserve vector mismatch on empty board")

    def test_board_encoding_after_moves(self):
        skip_if_no_rust()
        from hive.core.hex import Hex

        py_game = Game()
        rs_game = RustGame()

        moves = [
            ("wQ1", None, (0, 0)),
            ("bQ1", None, (1, 0)),
            ("wS1", None, (-1, 0)),
            ("bA1", None, (2, 0)),
        ]

        for piece_str, from_pos, to_pos in moves:
            py_piece = Piece.from_str(piece_str)
            py_from = Hex(*from_pos) if from_pos is not None else None
            py_to = Hex(*to_pos)
            py_game.play_move(py_piece, py_from, py_to)
            rs_game.play_move(piece_str, from_pos, to_pos)

        py_board, py_reserve = encode_board(py_game)
        rs_board, rs_reserve = rs_game.encode_board()

        np.testing.assert_array_almost_equal(py_board, rs_board, decimal=6,
            err_msg="Board tensor mismatch after moves")
        np.testing.assert_array_almost_equal(py_reserve, rs_reserve, decimal=6,
            err_msg="Reserve vector mismatch after moves")

    def test_encoding_random_positions(self):
        """Verify encoding parity across 20 random game positions."""
        skip_if_no_rust()
        random.seed(123)
        from hive.core.hex import Hex

        for _ in range(20):
            py_game = Game()
            rs_game = RustGame()

            for _ in range(random.randint(1, 40)):
                py_valid = py_game.valid_moves()
                if not py_valid:
                    py_game.play_pass()
                    rs_game.play_pass()
                    continue

                idx = random.randint(0, len(py_valid) - 1)
                piece, from_pos, to_pos = py_valid[idx]
                piece_str = str(piece)
                from_tuple = (from_pos.q, from_pos.r) if from_pos is not None else None
                to_tuple = (to_pos.q, to_pos.r)

                py_game.play_move(piece, from_pos, to_pos)
                rs_game.play_move(piece_str, from_tuple, to_tuple)

                if py_game.is_game_over:
                    break

            py_board, py_reserve = encode_board(py_game)
            rs_board, rs_reserve = rs_game.encode_board()

            np.testing.assert_array_almost_equal(py_board, rs_board, decimal=6)
            np.testing.assert_array_almost_equal(py_reserve, rs_reserve, decimal=6)


class TestMoveEncodingParity:
    """Test that move encoding produces identical indices."""

    def test_move_encoding_matches(self):
        skip_if_no_rust()
        from hive.core.hex import Hex

        py_game = Game()
        rs_game = RustGame()

        # After some moves, compare move encoding
        moves = [
            ("wQ1", None, (0, 0)),
            ("bQ1", None, (1, 0)),
        ]

        for piece_str, from_pos, to_pos in moves:
            py_piece = Piece.from_str(piece_str)
            py_from = Hex(*from_pos) if from_pos is not None else None
            py_to = Hex(*to_pos)
            py_game.play_move(py_piece, py_from, py_to)
            rs_game.play_move(piece_str, from_pos, to_pos)

        # Get legal move masks from both
        py_mask, py_indexed = get_legal_move_mask(py_game)
        rs_mask, rs_indexed = rs_game.get_legal_move_mask()

        np.testing.assert_array_equal(py_mask, rs_mask,
            err_msg="Legal move masks differ")

        py_indices = {idx for idx, _, _, _ in py_indexed}
        rs_indices = {idx for idx, _, _, _ in rs_indexed}
        assert py_indices == rs_indices, (
            f"Move indices differ.\n"
            f"Py only: {py_indices - rs_indices}\n"
            f"Rs only: {rs_indices - py_indices}"
        )
