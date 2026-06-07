"""Quick perft correctness test."""
import chess
import time
from board_cy import CustomBitboardBoard

PERFT_EXPECTED = {1: 20, 2: 400, 3: 8902, 4: 197281, 5: 4865609}

board = chess.Board()

all_passed = True
for depth, expected in PERFT_EXPECTED.items():
    b = CustomBitboardBoard.from_chess_board(board)
    t0 = time.perf_counter()
    result = b.run_perft_recursive(depth)
    elapsed = time.perf_counter() - t0
    status = "PASSED" if result == expected else "FAILED"
    if result != expected:
        all_passed = False
    nps = int(result / elapsed) if elapsed > 0 else 0
    print(f"  Perft {depth}: {result:>10,} (expected {expected:>10,}) [{status}]  {elapsed:.3f}s  {nps:>12,} nps")
    if depth == 4 and elapsed > 1.0:
        # don't run depth 5 if we're already slow
        break

print()
print("=== PERFT:", "ALL PASSED" if all_passed else "SOME FAILED", "===")
