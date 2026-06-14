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
    """Play a single game between two engines using standard UCI process isolation."""
    import subprocess
    import os

    def start_engine(path: str):
        # We run "python -u uci.py" in the engine's directory
        p = subprocess.Popen(
            [sys.executable, "-u", "uci.py"],
            cwd=os.path.abspath(path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        # Initialize UCI
        p.stdin.write("uci\n")
        p.stdin.flush()
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f"Engine at {path} exited prematurely during uci handshake.")
            line = line.strip()
            if line == "uciok":
                break
        p.stdin.write("isready\n")
        p.stdin.flush()
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f"Engine at {path} exited prematurely during isready check.")
            line = line.strip()
            if line == "readyok":
                break
        return p

    white_proc = start_engine(white_path)
    black_proc = start_engine(black_path)

    board = chess.Board()
    game_moves = []
    result = "1/2-1/2"

    try:
        # Tell both engines a new game is starting
        white_proc.stdin.write("ucinewgame\n")
        white_proc.stdin.flush()
        black_proc.stdin.write("ucinewgame\n")
        black_proc.stdin.flush()

        for move_num in range(max_moves * 2):
            if board.is_game_over():
                break

            current_proc = white_proc if board.turn == chess.WHITE else black_proc

            # Send position and go command
            moves_str = " ".join(game_moves)
            pos_cmd = f"position startpos moves {moves_str}\n" if moves_str else "position startpos\n"
            current_proc.stdin.write(pos_cmd)
            current_proc.stdin.flush()

            # We use movetime in milliseconds
            go_cmd = f"go movetime {int(time_limit * 1000)}\n"
            current_proc.stdin.write(go_cmd)
            current_proc.stdin.flush()

            # Read engine output until bestmove
            move_uci = None
            while True:
                line = current_proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2:
                        move_uci = parts[1]
                    break

            if not move_uci or move_uci == "(none)":
                break

            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                break

            board.push(move)
            game_moves.append(move_uci)

    finally:
        # Quit processes safely
        try:
            white_proc.stdin.write("quit\n")
            white_proc.stdin.flush()
            white_proc.terminate()
        except Exception:
            pass
        try:
            black_proc.stdin.write("quit\n")
            black_proc.stdin.flush()
            black_proc.terminate()
        except Exception:
            pass

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
        result = "1/2-1/2"

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
        sys.argv = [sys.argv[0], "d:\\Coding\\Research\\qwen-chess-engine\\qwen-chess-v3", "d:\\Coding\\Research\\qwen-chess-engine\\qwen-chess-v3\\baseline", "0.1"]
        # sys.exit(1)

    white_path = sys.argv[1]
    black_path = sys.argv[2]
    time_limit = float(sys.argv[3])

    try:
        game_result = play_game(white_path, black_path, time_limit)
        print(json.dumps(game_result))
    except Exception as e:
        print(json.dumps({"error": str(e), "result": "1/2-1/2", "pgn": "", "moves": 0}))
        sys.exit(0)
