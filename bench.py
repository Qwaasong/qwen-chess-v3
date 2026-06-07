"""Benchmark for move generation from board.py."""

import time

import chess

from board_cy import CustomBitboardBoard


def benchmark_move_generation(iterations: int = 100000) -> dict:
    positions = [
        chess.Board(),
        chess.Board("2rq1r2/pb2bpp1/1pnppn1p/2p1k3/4p2B/1PPBP3/PP2QPPP/R2NRNK1 w - - 0 1"),
        chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ]

    results = {}
    for idx, pos in enumerate(positions):
        board = CustomBitboardBoard.from_chess_board(pos)
        moves = board.generate_legal_moves()
        results[f"position_{idx}"] = {
            "move_count": len(moves),
            "iterations": iterations,
        }
        start = time.perf_counter()
        for _ in range(iterations):
            board.generate_legal_moves()
        elapsed = time.perf_counter() - start
        results[f"position_{idx}"]["elapsed_ms"] = elapsed * 1000
        results[f"position_{idx}"]["moves_per_sec"] = (
            iterations * len(moves) / elapsed if elapsed > 0 else float("inf")
        )

    return results


if __name__ == "__main__":
    res = benchmark_move_generation()
    for k, v in res.items():
        print(
            f"{k}: {v['move_count']} legal moves, "
            f"{v['elapsed_ms']:.2f}ms over {v['iterations']} iterations, "
            f"{v['moves_per_sec']:.0f} legal moves/sec"
        )
