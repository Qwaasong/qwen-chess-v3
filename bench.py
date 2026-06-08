"""Benchmark for move generation: Make/Unmake vs Copy-Make."""

import time
import chess
from board_cy import CustomBitboardBoard
try:
    from engine_cy import generate_legal_moves_copymake
except ImportError:
    generate_legal_moves_copymake = None


def benchmark_move_generation(iterations: int = 50000) -> dict:
    positions = [
        chess.Board(),
        chess.Board("2rq1r2/pb2bpp1/1pnppn1p/2p1k3/4p2B/1PPBP3/PP2QPPP/R2NRNK1 w - - 0 1"),
        chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ]

    results = {}
    for idx, pos in enumerate(positions):
        board = CustomBitboardBoard.from_chess_board(pos)

        # 1. Benchmark Make/Unmake
        moves_std = board.generate_legal_moves()
        results[f"position_{idx}_make_unmake"] = {
            "mode": "Make/Unmake",
            "move_count": len(moves_std),
            "iterations": iterations,
        }
        start = time.perf_counter()
        for _ in range(iterations):
            board.generate_legal_moves()
        elapsed = time.perf_counter() - start
        results[f"position_{idx}_make_unmake"]["elapsed_ms"] = elapsed * 1000
        results[f"position_{idx}_make_unmake"]["moves_per_sec"] = (
            iterations * len(moves_std) / elapsed if elapsed > 0 else float("inf")
        )

        # 2. Benchmark Copy-Make
        if generate_legal_moves_copymake is not None:
            moves_cp = generate_legal_moves_copymake(board)
            results[f"position_{idx}_copy_make"] = {
                "mode": "Copy-Make",
                "move_count": len(moves_cp),
                "iterations": iterations,
            }
            start = time.perf_counter()
            for _ in range(iterations):
                generate_legal_moves_copymake(board)
            elapsed = time.perf_counter() - start
            results[f"position_{idx}_copy_make"]["elapsed_ms"] = elapsed * 1000
            results[f"position_{idx}_copy_make"]["moves_per_sec"] = (
                iterations * len(moves_cp) / elapsed if elapsed > 0 else float("inf")
            )

            # Sanity check move count
            if len(moves_std) != len(moves_cp):
                print(f"WARNING: Move count mismatch in Position {idx}! Make/Unmake={len(moves_std)}, Copy-Make={len(moves_cp)}")

    return results


if __name__ == "__main__":
    if generate_legal_moves_copymake is None:
        print("Error: Could not import generate_legal_moves_copymake from engine_cy. Please compile first.")
        exit(1)

    print("Running move generation benchmark (Make/Unmake vs Copy-Make)...")
    res = benchmark_move_generation()
    
    # Print comparison
    for i in range(3):
        print(f"\n--- Position {i + 1} ---")
        k_mu = f"position_{i}_make_unmake"
        k_cm = f"position_{i}_copy_make"
        
        v_mu = res[k_mu]
        v_cm = res[k_cm]
        
        print(f"Make/Unmake: {v_mu['move_count']} legal moves, {v_mu['elapsed_ms']:.1f}ms, {v_mu['moves_per_sec']:.0f} moves/sec")
        print(f"Copy-Make  : {v_cm['move_count']} legal moves, {v_cm['elapsed_ms']:.1f}ms, {v_cm['moves_per_sec']:.0f} moves/sec")
        
        speedup = (v_cm['moves_per_sec'] / v_mu['moves_per_sec'] - 1) * 100
        print(f"Speed difference: {speedup:+.1f}%")
