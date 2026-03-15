"""Tests for core game logic."""

import pytest
from hive.core.hex import Hex, DIRECTIONS
from hive.core.pieces import Piece, PieceColor, PieceType, player_pieces
from hive.core.board import Board
from hive.core.rules import get_moves, get_placements
from hive.core.game import Game, GameState


class TestHex:
    def test_add(self):
        assert Hex(1, 2) + Hex(3, 4) == Hex(4, 6)

    def test_sub(self):
        assert Hex(3, 4) - Hex(1, 2) == Hex(2, 2)

    def test_neighbors(self):
        h = Hex(0, 0)
        assert len(h.neighbors()) == 6

    def test_distance(self):
        assert Hex(0, 0).distance(Hex(2, -1)) == 2
        assert Hex(0, 0).distance(Hex(0, 0)) == 0


class TestPieces:
    def test_from_str(self):
        p = Piece.from_str("wQ1")
        assert p == Piece(PieceColor.WHITE, PieceType.QUEEN, 1)

    def test_from_str_no_number(self):
        p = Piece.from_str("bQ")
        assert p == Piece(PieceColor.BLACK, PieceType.QUEEN, 1)

    def test_str(self):
        p = Piece(PieceColor.WHITE, PieceType.SPIDER, 2)
        assert str(p) == "wS2"

    def test_player_pieces(self):
        whites = player_pieces(PieceColor.WHITE)
        assert len(whites) == 11  # Q=1, S=2, B=2, G=3, A=3


class TestBoard:
    def test_place_and_find(self):
        board = Board()
        p = Piece(PieceColor.WHITE, PieceType.QUEEN, 1)
        board.place_piece(p, Hex(0, 0))
        assert board.piece_position(p) == Hex(0, 0)
        assert board.top_piece(Hex(0, 0)) == p

    def test_stacking(self):
        board = Board()
        p1 = Piece(PieceColor.WHITE, PieceType.SPIDER, 1)
        p2 = Piece(PieceColor.WHITE, PieceType.BEETLE, 1)
        board.place_piece(p1, Hex(0, 0))
        board.place_piece(p2, Hex(0, 0))
        assert board.top_piece(Hex(0, 0)) == p2
        assert board.stack_height(Hex(0, 0)) == 2

    def test_remove_top(self):
        board = Board()
        p1 = Piece(PieceColor.WHITE, PieceType.SPIDER, 1)
        p2 = Piece(PieceColor.WHITE, PieceType.BEETLE, 1)
        board.place_piece(p1, Hex(0, 0))
        board.place_piece(p2, Hex(0, 0))
        board.remove_piece(p2)
        assert board.top_piece(Hex(0, 0)) == p1

    def test_connectivity(self):
        board = Board()
        p1 = Piece(PieceColor.WHITE, PieceType.SPIDER, 1)
        p2 = Piece(PieceColor.WHITE, PieceType.QUEEN, 1)
        p3 = Piece(PieceColor.BLACK, PieceType.QUEEN, 1)
        board.place_piece(p1, Hex(0, 0))
        board.place_piece(p2, Hex(1, 0))
        board.place_piece(p3, Hex(2, 0))
        assert board.is_connected()
        # Removing p2 should disconnect p1 and p3
        assert not board.is_connected(exclude=p2)

    def test_can_slide(self):
        board = Board()
        # Create a gate: two pieces at (0,-1) and (0,1) block sliding from (0,0) to (-1,0)
        board.place_piece(Piece(PieceColor.WHITE, PieceType.ANT, 1), Hex(0, 0))
        board.place_piece(Piece(PieceColor.WHITE, PieceType.SPIDER, 1), Hex(-1, -1))  # NW of (0,0)
        board.place_piece(Piece(PieceColor.BLACK, PieceType.SPIDER, 1), Hex(-1, 1))   # one neighbor
        # Check specific slide paths
        # Without a full gate, should be able to slide
        assert board.can_slide(Hex(0, 0), Hex(-1, 0))


class TestPlacements:
    def test_first_placement(self):
        board = Board()
        places = get_placements(PieceColor.WHITE, board)
        assert places == {Hex(0, 0)}

    def test_second_placement(self):
        board = Board()
        board.place_piece(Piece(PieceColor.WHITE, PieceType.SPIDER, 1), Hex(0, 0))
        places = get_placements(PieceColor.BLACK, board)
        # Should be all 6 neighbors of (0,0)
        assert len(places) == 6

    def test_third_placement_no_enemy_adjacency(self):
        board = Board()
        ws1 = Piece(PieceColor.WHITE, PieceType.SPIDER, 1)
        bs1 = Piece(PieceColor.BLACK, PieceType.SPIDER, 1)
        board.place_piece(ws1, Hex(0, 0))
        board.place_piece(bs1, Hex(1, 0))
        places = get_placements(PieceColor.WHITE, board)
        # White must place adjacent to white but not adjacent to black
        assert Hex(1, 0) not in places  # occupied
        # Positions adjacent to white but not black
        for p in places:
            # Check not adjacent to black
            assert not any(
                board.top_piece(n) is not None and board.top_piece(n).color == PieceColor.BLACK
                for n in p.neighbors()
            )


class TestMoves:
    def _setup_basic(self):
        """Set up a basic board for movement tests."""
        board = Board()
        # Line of pieces: wQ at (0,0), bQ at (1,0), wS1 at (-1,0), bS1 at (2,0)
        wq = Piece(PieceColor.WHITE, PieceType.QUEEN, 1)
        bq = Piece(PieceColor.BLACK, PieceType.QUEEN, 1)
        ws1 = Piece(PieceColor.WHITE, PieceType.SPIDER, 1)
        bs1 = Piece(PieceColor.BLACK, PieceType.SPIDER, 1)
        board.place_piece(ws1, Hex(-1, 0))
        board.place_piece(wq, Hex(0, 0))
        board.place_piece(bq, Hex(1, 0))
        board.place_piece(bs1, Hex(2, 0))
        return board, wq, bq, ws1, bs1

    def test_queen_moves(self):
        board, wq, bq, ws1, bs1 = self._setup_basic()
        # In a straight line A-B-C-D, interior pieces are articulation points
        # wq is at (0,0) between ws1(-1,0) and bq(1,0) - can't move (One Hive)
        moves = get_moves(wq, board)
        assert len(moves) == 0

        # But end pieces can move: ws1 at (-1,0)
        moves = get_moves(ws1, board)
        assert len(moves) > 0

    def test_grasshopper_jump(self):
        board = Board()
        wg = Piece(PieceColor.WHITE, PieceType.GRASSHOPPER, 1)
        ws1 = Piece(PieceColor.WHITE, PieceType.SPIDER, 1)
        bs1 = Piece(PieceColor.BLACK, PieceType.SPIDER, 1)
        board.place_piece(wg, Hex(0, 0))
        board.place_piece(ws1, Hex(1, 0))
        board.place_piece(bs1, Hex(2, 0))
        moves = get_moves(wg, board)
        # Grasshopper jumps over pieces in a line, landing at (3,0)
        assert Hex(3, 0) in moves

    def test_beetle_climb(self):
        board = Board()
        wb = Piece(PieceColor.WHITE, PieceType.BEETLE, 1)
        ws1 = Piece(PieceColor.WHITE, PieceType.SPIDER, 1)
        board.place_piece(ws1, Hex(0, 0))
        board.place_piece(wb, Hex(1, 0))
        moves = get_moves(wb, board)
        # Beetle should be able to climb on top of ws1
        assert Hex(0, 0) in moves

    def test_ant_movement(self):
        board = Board()
        wa = Piece(PieceColor.WHITE, PieceType.ANT, 1)
        wq = Piece(PieceColor.WHITE, PieceType.QUEEN, 1)
        bq = Piece(PieceColor.BLACK, PieceType.QUEEN, 1)
        board.place_piece(wa, Hex(0, 0))
        board.place_piece(wq, Hex(1, 0))
        board.place_piece(bq, Hex(2, 0))
        moves = get_moves(wa, board)
        # Ant should be able to reach many positions around the hive
        assert len(moves) > 2

    def test_one_hive_rule(self):
        board = Board()
        # Chain: A - B - C, removing B disconnects
        a = Piece(PieceColor.WHITE, PieceType.QUEEN, 1)
        b = Piece(PieceColor.WHITE, PieceType.SPIDER, 1)
        c = Piece(PieceColor.BLACK, PieceType.QUEEN, 1)
        board.place_piece(a, Hex(0, 0))
        board.place_piece(b, Hex(1, 0))
        board.place_piece(c, Hex(2, 0))
        # b is an articulation point, should have no moves
        moves = get_moves(b, board)
        assert len(moves) == 0


class TestGame:
    def test_new_game(self):
        game = Game()
        assert game.state == GameState.NOT_STARTED
        assert game.turn_color == PieceColor.WHITE

    def test_first_moves(self):
        game = Game()
        valid = game.valid_moves()
        # First move: can place any piece except queen at origin (tournament rule)
        assert len(valid) == 10

    def test_game_string(self):
        game = Game()
        gs = game.game_string
        assert gs.startswith("Base;NotStarted;White[1]")

    def test_queen_must_be_placed_by_turn_4(self):
        game = Game()
        # Play 3 moves each without placing queen
        ws1 = Piece(PieceColor.WHITE, PieceType.SPIDER, 1)
        game.play_move(ws1, None, Hex(0, 0))

        bs1 = Piece(PieceColor.BLACK, PieceType.SPIDER, 1)
        game.play_move(bs1, None, Hex(1, 0))

        ws2 = Piece(PieceColor.WHITE, PieceType.SPIDER, 2)
        pos = list(get_placements(PieceColor.WHITE, game.board))[0]
        game.play_move(ws2, None, pos)

        bs2 = Piece(PieceColor.BLACK, PieceType.SPIDER, 2)
        pos = list(get_placements(PieceColor.BLACK, game.board))[0]
        game.play_move(bs2, None, pos)

        wa1 = Piece(PieceColor.WHITE, PieceType.ANT, 1)
        pos = list(get_placements(PieceColor.WHITE, game.board))[0]
        game.play_move(wa1, None, pos)

        ba1 = Piece(PieceColor.BLACK, PieceType.ANT, 1)
        pos = list(get_placements(PieceColor.BLACK, game.board))[0]
        game.play_move(ba1, None, pos)

        # White's 4th move: must place queen
        valid = game.valid_moves()
        for piece, _, _ in valid:
            assert piece.piece_type == PieceType.QUEEN

    def test_undo(self):
        game = Game()
        ws1 = Piece(PieceColor.WHITE, PieceType.SPIDER, 1)
        game.play_move(ws1, None, Hex(0, 0))
        game.undo()
        assert game.move_count == 0
        assert game.turn_color == PieceColor.WHITE
        assert game.board.piece_position(ws1) is None

    def test_win_detection(self):
        game = Game()
        board = game.board
        # Manually surround a queen
        wq = Piece(PieceColor.WHITE, PieceType.QUEEN, 1)
        board.place_piece(wq, Hex(0, 0))
        game._white_reserve.discard(wq)

        # Place 6 pieces around the white queen
        surrounding = [
            Piece(PieceColor.BLACK, PieceType.QUEEN, 1),
            Piece(PieceColor.BLACK, PieceType.SPIDER, 1),
            Piece(PieceColor.BLACK, PieceType.SPIDER, 2),
            Piece(PieceColor.BLACK, PieceType.ANT, 1),
            Piece(PieceColor.BLACK, PieceType.ANT, 2),
            Piece(PieceColor.BLACK, PieceType.ANT, 3),
        ]
        for piece, direction in zip(surrounding, Hex(0, 0).neighbors()):
            board.place_piece(piece, direction)
            game._black_reserve.discard(piece)

        game.state = GameState.IN_PROGRESS
        game._check_game_end()
        assert game.state == GameState.BLACK_WINS


class TestUHP:
    def test_parse_first_move(self):
        from hive.uhp.engine import UHPEngine
        engine = UHPEngine()
        engine.game = Game()
        piece, from_pos, to_pos = engine._parse_move("wS1")
        assert piece == Piece(PieceColor.WHITE, PieceType.SPIDER, 1)
        assert from_pos is None
        assert to_pos == Hex(0, 0)

    def test_parse_relative_position(self):
        from hive.uhp.engine import UHPEngine
        engine = UHPEngine()
        engine.game = Game()
        ws1 = Piece(PieceColor.WHITE, PieceType.SPIDER, 1)
        engine.game.board.place_piece(ws1, Hex(0, 0))
        engine.game._white_reserve.discard(ws1)

        # "bS1 wS1-" means black Spider 1 to the right of white Spider 1
        piece, from_pos, to_pos = engine._parse_move("bS1 wS1-")
        assert piece == Piece(PieceColor.BLACK, PieceType.SPIDER, 1)
        assert to_pos == Hex(1, 0)  # right of (0,0)
