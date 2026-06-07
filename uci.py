"""UCI wrapper for the qwen-chess-v3 engine."""

import sys
import time
import threading
from typing import Optional

import chess

import engine

try:
    from board_cy import CustomBitboardBoard
except ImportError:
    from board import CustomBitboardBoard

# Track hash and threads options set by GUI
HASH_SIZE_MB: int = 16
THREADS_COUNT: int = 1

# Active search thread
search_thread: Optional[threading.Thread] = None


def search_task(
    chess_board: chess.Board,
    time_limit: float,
    depth_limit: Optional[int],
) -> None:
    """Run search iteratively and print info to stdout.

    This function is intended to run in a separate thread.
    """
    d_limit = depth_limit if depth_limit is not None else 0
    best_move = engine.get_best_move(
        chess_board,
        time_limit=time_limit,
        depth_limit=d_limit,
        print_info=True,
    )
    if best_move is None:
        print("bestmove (none)", flush=True)
    else:
        print(f"bestmove {best_move.uci()}", flush=True)


def parse_position(line: str) -> chess.Board:
    """Parse position command into a chess.Board object."""
    words = line.split()
    if not words or words[0] != "position":
        return chess.Board()

    board = chess.Board()

    if "startpos" in words:
        moves_idx = words.index("moves") if "moves" in words else -1
        if moves_idx != -1:
            moves = words[moves_idx + 1 :]
        else:
            moves = []
        board.reset()
    elif "fen" in words:
        fen_idx = words.index("fen")
        moves_idx = words.index("moves") if "moves" in words else -1

        if moves_idx != -1:
            fen_words = words[fen_idx + 1 : moves_idx]
            moves = words[moves_idx + 1 :]
        else:
            fen_words = words[fen_idx + 1 :]
            moves = []

        fen_str = " ".join(fen_words)
        board = chess.Board(fen_str)
    else:
        return board

    for move_str in moves:
        try:
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                board.push(move)
            else:
                board.push(board.parse_uci(move_str))
        except Exception:
            pass

    return board


def stop_search() -> None:
    """Interrupt the running search thread, if any."""
    global search_thread
    if search_thread is not None and search_thread.is_alive():
        engine.info.stop = True
        search_thread.join()
    search_thread = None


def main() -> None:
    """Main UCI loop processing stdin."""
    global search_thread, HASH_SIZE_MB, THREADS_COUNT
    board = chess.Board()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        words = line.split()
        command = words[0]

        if command == "uci":
            print("id name Qwen Chess Engine v3", flush=True)
            print("id author Qwen / AI", flush=True)
            print(
                "option name Hash type spin default 16 min 1 max 1024",
                flush=True,
            )
            print(
                "option name Threads type spin default 1 min 1 max 1",
                flush=True,
            )
            print("uciok", flush=True)

        elif command == "isready":
            print("readyok", flush=True)

        elif command == "setoption":
            # setoption name Hash value 32
            # setoption name Threads value 1
            try:
                name_idx = words.index("name")
                value_idx = words.index("value")
                opt_name = words[name_idx + 1]
                opt_val = words[value_idx + 1]

                if opt_name.lower() == "hash":
                    HASH_SIZE_MB = int(opt_val)
                elif opt_name.lower() == "threads":
                    THREADS_COUNT = int(opt_val)
            except (ValueError, IndexError):
                pass

        elif command == "ucinewgame":
            stop_search()
            engine.clear_tt()
            board = chess.Board()

        elif command == "position":
            stop_search()
            board = parse_position(line)

        elif command == "go":
            stop_search()

            # Parse go parameters
            wtime: Optional[int] = None
            btime: Optional[int] = None
            winc: Optional[int] = None
            binc: Optional[int] = None
            movetime: Optional[int] = None
            depth_limit: Optional[int] = None
            infinite = False

            i = 1
            while i < len(words):
                token = words[i]
                if token == "wtime" and i + 1 < len(words):
                    wtime = int(words[i + 1])
                    i += 2
                elif token == "btime" and i + 1 < len(words):
                    btime = int(words[i + 1])
                    i += 2
                elif token == "winc" and i + 1 < len(words):
                    winc = int(words[i + 1])
                    i += 2
                elif token == "binc" and i + 1 < len(words):
                    binc = int(words[i + 1])
                    i += 2
                elif token == "movetime" and i + 1 < len(words):
                    movetime = int(words[i + 1])
                    i += 2
                elif token == "depth" and i + 1 < len(words):
                    depth_limit = int(words[i + 1])
                    i += 2
                elif token == "infinite":
                    infinite = True
                    i += 1
                else:
                    i += 1

            # Determine time limit
            if movetime is not None:
                time_limit = movetime / 1000.0
            elif infinite:
                time_limit = 86400.0
            else:
                # Time management formula
                time_left = wtime if board.turn == chess.WHITE else btime
                inc = winc if board.turn == chess.WHITE else binc

                if time_left is not None:
                    time_left_sec = time_left / 1000.0
                    inc_sec = (inc / 1000.0) if inc is not None else 0.0
                    time_limit = max(0.05, time_left_sec / 20.0 + inc_sec / 2.0)
                else:
                    time_limit = 2.0

            search_thread = threading.Thread(
                target=search_task,
                args=(board.copy(), time_limit, depth_limit),
            )
            search_thread.start()

        elif command == "stop":
            stop_search()

        elif command == "quit":
            stop_search()
            sys.exit(0)


if __name__ == "__main__":
    main()
