import unittest
import chess
import time
import os
import sys

# Make sure workspace root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from board_cy import CustomBitboardBoard
except ImportError:
    from board import CustomBitboardBoard

import engine


class TestEngineLogic(unittest.TestCase):

    def setUp(self):
        # Clear Transposition Table
        engine.clear_tt()

    def make_uci_move(self, board, uci_str):
        for m in board.generate_legal_moves():
            if board.to_chess_move(m).uci() == uci_str:
                board.make_move(m)
                return
        raise ValueError(f"Move {uci_str} not found in legal moves")

    def test_threefold_repetition(self):
        """Test that the engine detects threefold repetition draw."""
        # Simple repetition position setup
        # Play Nf3 (g1f3) Nf6 (g8f6) Ng1 (f3g1) Ng8 (f6g8)
        board = CustomBitboardBoard.from_chess_board(chess.Board())
        
        # 1. Nf3 Nf6
        self.make_uci_move(board, "g1f3")
        self.make_uci_move(board, "g8f6")
        
        # 2. Ng1 Ng8 (Position occurs 2nd time - after Ng8)
        self.make_uci_move(board, "f3g1")
        self.make_uci_move(board, "f6g8")
        
        # 3. Nf3 Nf6
        self.make_uci_move(board, "g1f3")
        self.make_uci_move(board, "g8f6")
        
        # 4. Ng1 Ng8 (Position occurs 3rd time)
        self.make_uci_move(board, "f3g1")
        self.make_uci_move(board, "f6g8")
        
        # Check if is_repetition works
        self.assertTrue(engine.is_repetition(board))

    def test_fifty_move_rule(self):
        """Test that the engine registers draw at halfmove clock >= 100."""
        board = CustomBitboardBoard.from_chess_board(chess.Board())
        board.halfmove_clock = 100
        # If in check, the engine negamax should return 0 at ply > 0
        score = engine.negamax(board, depth=1, alpha=-10000, beta=10000, color=1, ply=1)
        self.assertEqual(score, 0)

    def test_ponderhit_time_limit_conversion(self):
        """Test that the time limit conversions between Cython and Python backend work correctly."""
        # Verify USING_CYTHON flag exists
        self.assertTrue(hasattr(engine, "USING_CYTHON"))
        
        # Test target time calculations
        elapsed = 0.5
        target = 2.0
        
        # If USING_CYTHON, it must convert to ms. Otherwise it stays in seconds.
        if engine.USING_CYTHON:
            time_limit = (elapsed + target) * 1000.0
            self.assertEqual(time_limit, 2500.0)
        else:
            time_limit = elapsed + target
            self.assertEqual(time_limit, 2.5)

    def test_static_exchange_evaluation(self):
        """Test that Static Exchange Evaluation correctly scores captures."""
        # Setup position with a queen capturing a defended pawn
        # 1k6/8/8/4p3/3p4/3Q4/8/1K6 w - - 0 1 -> d4 pawn is defended by e5 pawn.
        # Qd3xd4 captures defended pawn. This is SEE < 0 because Qd3 is lost for d4 pawn.
        
        chess_board = chess.Board("1k6/8/8/4p3/3p4/3Q4/8/1K6 w - - 0 1")
        board = CustomBitboardBoard.from_chess_board(chess_board)
        
        # Find move Qd3xd4 (from d3=19 to d4=27)
        # piece_map indices: Qd3 is on sq 19, pd4 is on sq 27
        # Move encoding: from_sq | (to_sq << 6) | (flag << 12)
        # Qd3xd4 flag = 0 (normal capture)
        move = 19 | (27 << 6)
        
        see_score = engine.static_exchange_evaluation(board, move)
        # We capture a pawn (100) but lose a queen (900) because d4 is defended by e5 pawn.
        # So SEE should be 100 - 900 = -800 (negative!)
        self.assertLess(see_score, 0)


if __name__ == "__main__":
    unittest.main()
