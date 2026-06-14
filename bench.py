"""Benchmark for move generation and search performance of qwen-chess-v3."""

import time
import chess
from board_cy import CustomBitboardBoard
import engine

def benchmark_move_generation(iterations: int = 50000) -> dict:
    positions = [
        chess.Board(),
        chess.Board("2rq1r2/pb2bpp1/1pnppn1p/2p1k3/4p2B/1PPBP3/PP2QPPP/R2NRNK1 w - - 0 1"),
        chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ]

    results = {}
    for idx, pos in enumerate(positions):
        board = CustomBitboardBoard.from_chess_board(pos)

        # Benchmark Make/Unmake
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

    return results

def benchmark_search(depth: int = 7) -> None:
    print(f"\nRunning search benchmark to depth {depth}...")
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    
    t0 = time.perf_counter()
    # Use depth_limit to restrict search depth
    best_move = engine.get_best_move(board, time_limit=10.0, depth_limit=depth, print_info=False)
    elapsed = time.perf_counter() - t0
    
    nodes = engine.info.nodes
    nps = int(nodes / elapsed) if elapsed > 0.0 else 0
    print(f"Search completed in {elapsed:.3f}s")
    print(f"Nodes searched: {nodes:,}")
    print(f"Speed: {nps:,} NPS")
    print(f"Best Move: {best_move}")

if __name__ == "__main__":
    print("Running move generation benchmark...")
    res = benchmark_move_generation()
    
    # Print results
    for i in range(3):
        print(f"\n--- Position {i + 1} ---")
        k_mu = f"position_{i}_make_unmake"
        v_mu = res[k_mu]
        print(f"Make/Unmake: {v_mu['move_count']} legal moves, {v_mu['elapsed_ms']:.1f}ms, {v_mu['moves_per_sec']:.0f} moves/sec")

    benchmark_search()
