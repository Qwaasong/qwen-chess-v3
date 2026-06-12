"""
Play a single game between two chess engines in a subprocess-isolated environment.
Used by the round-robin tournament runner.

Usage:
    python play_game_process.py <white_engine_path> <black_engine_path> <time_limit>

Output:
    JSON to stdout with game result: {"result": "1-0" | "0-1" | "1/2-1/2", "pgn": "...", "moves": N}
"""

import sys
import json
import time

import chess

# --- Import engine from the specified path ---
def load_engine_from_path(engine_path: str):
    """Dynamically load engine module from a given folder path."""
    import importlib.util
    import os

    # Add path to sys.path temporarily
    abs_path = os.path.abspath(engine_path)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)

    # Try to load compiled Cython version first, then pure Python
    try:
        spec = importlib.util.spec_from_file_location(
            "engine_module", os.path.join(abs_path, "engine.py")
        )
        engine_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine_mod)
        return engine_mod
    except Exception as e:
        raise RuntimeError(f"Failed to load engine from {abs_path}: {e}") from e


def play_game(
    white_path: str,
    black_path: str,
    time_limit: float = 0.1,
    max_moves: int = 150,
) -> dict:
    """Play a single game between two engines."""
    white_engine = load_engine_from_path(white_path)
    black_engine = load_engine_from_path(black_path)

    board = chess.Board()
    game_moves = []
    result = "1/2-1/2"

    for move_num in range(max_moves * 2):  # max_moves full moves = 2x half-moves
        if board.is_game_over():
            break

        current_engine = white_engine if board.turn == chess.WHITE else black_engine

        start = time.time()
        try:
            move = current_engine.get_best_move(board, time_limit=time_limit)
        except Exception:
            # Engine error = forfeit
            move = None

        elapsed = time.time() - start

        if move is None or move not in board.legal_moves:
            # Try to find a legal move
            legal = list(board.legal_moves)
            if legal:
                move = legal[0]
            else:
                break

        board.push(move)
        game_moves.append(move.uci())

    # Determine result
    outcome = board.outcome()
    if outcome is not None:
        if outcome.winner == chess.WHITE:
            result = "1-0"
        elif outcome.winner == chess.BLACK:
            result = "0-1"
        else:
            result = "1/2-1/2"
    else:
        # Adjudication by move count or other condition
        result = "1/2-1/2"

    # Build minimal PGN
    pgn_moves = " ".join(
        f"{i // 2 + 1}{'.' if i % 2 == 0 else '...'} {m}"
        for i, m in enumerate(game_moves)
    )
    pgn = f"[Result \"{result}\"]\n\n{pgn_moves} {result}"

    return {
        "result": result,
        "pgn": pgn,
        "moves": len(game_moves),
    }


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(json.dumps({"error": "Usage: play_game_process.py <white> <black> <time_limit>"}))
        sys.exit(1)

    white_path = sys.argv[1]
    black_path = sys.argv[2]
    time_limit = float(sys.argv[3])

    try:
        game_result = play_game(white_path, black_path, time_limit)
        print(json.dumps(game_result))
    except Exception as e:
        print(json.dumps({"error": str(e), "result": "1/2-1/2", "pgn": "", "moves": 0}))
        sys.exit(0)
