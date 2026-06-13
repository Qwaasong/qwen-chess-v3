import chess
import os
import sys

# Load baseline engine/board
sys.path.insert(0, os.path.abspath("baseline"))
import board_cy as baseline_board
import engine_cy as baseline_engine
sys.path.pop(0)

# Load optimized engine/board
sys.path.insert(0, os.path.abspath("."))
import board_cy as opt_board
import engine_cy as opt_engine
sys.path.pop(0)

positions = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", # Ruy Lopez / Open
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", # Sicilian
    "k7/8/8/8/8/8/8/K7 w - - 0 1", # King only
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" # Kiwipete
]

print("=" * 60)
print("COMPARING EVALUATION SCORES (Baseline vs Optimized)")
print("=" * 60)

for idx, fen in enumerate(positions, 1):
    print(f"\nPosition {idx}: {fen}")
    
    # Baseline
    b_board = chess.Board(fen)
    base_b = baseline_board.CustomBitboardBoard.from_chess_board(b_board)
    base_eval = (base_b.score_mg, base_b.score_eg, base_b.phase)
    
    # Optimized
    opt_b = opt_board.CustomBitboardBoard.from_chess_board(b_board)
    opt_eval = (opt_b.score_mg, opt_b.score_eg, opt_b.phase)
    
    print(f"  Baseline (mg, eg, phase) : {base_eval}")
    print(f"  Optimized (mg, eg, phase): {opt_eval}")
    if base_eval != opt_eval:
        print("  [!] WARNING: EVALUATION MISMATCH!")
        
print("=" * 60)
