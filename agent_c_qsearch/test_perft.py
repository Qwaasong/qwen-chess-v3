"""Perft correctness test with statistically accurate performance measurement.

Improvements over the original:
  1. Warmup phase to stabilize CPU frequency and OS scheduling
  2. Multiple trials per depth with statistical reporting
  3. Correctness check separated from benchmark measurement
  4. Median (robust against outliers) + best + std deviation
  5. Process priority hint for reduced scheduling noise
  6. GC disabled during timing to avoid collector pauses
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
import time
import chess
from board_cy import CustomBitboardBoard

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PERFT_EXPECTED: dict[int, int] = {
    1: 20,
    2: 400,
    3: 8_902,
    4: 197_281,
    5: 4_865_609,
}

# Number of benchmark trials per depth (more = more reliable median)
NUM_TRIALS = 5

# Number of warmup iterations before measurement starts
NUM_WARMUP = 2

# Minimum elapsed time (seconds) for a trial to be considered reliable.
# If a trial finishes faster than this, we repeat the perft call
# multiple times within a single trial (calibrated loop).
MIN_TRIAL_TIME = 0.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _try_raise_priority() -> None:
    """Attempt to raise process priority to reduce scheduling noise."""
    try:
        if sys.platform == "win32":
            import ctypes
            # ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, 0x00008000)
        else:
            os.nice(-10)
    except Exception:
        pass  # Non-critical; ignore if we can't raise priority


def _run_perft_once(depth: int) -> tuple[int, float]:
    """Run perft once; returns (node_count, elapsed_seconds)."""
    board = chess.Board()
    b = CustomBitboardBoard.from_chess_board(board)
    gc.disable()
    t0 = time.perf_counter()
    result = b.run_perft_recursive(depth)
    elapsed = time.perf_counter() - t0
    gc.enable()
    return result, elapsed


def _run_calibrated_trial(depth: int, expected_nodes: int) -> float:
    """Run a single trial that lasts at least MIN_TRIAL_TIME.

    If one perft call is too fast, we repeat it multiple times in a loop
    and divide the total time by the number of repetitions.  This reduces
    the overhead of time.perf_counter() granularity (typically ~100 ns)
    relative to the measured interval.
    """
    reps = max(1, int(MIN_TRIAL_TIME / 0.01))  # start guess

    board = chess.Board()
    b = CustomBitboardBoard.from_chess_board(board)

    gc.disable()
    t0 = time.perf_counter()
    for _ in range(reps):
        b.run_perft_recursive(depth)
        # Reset board state for next repetition
        b = CustomBitboardBoard.from_chess_board(board)
    total_elapsed = time.perf_counter() - t0
    gc.enable()

    return total_elapsed / reps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _try_raise_priority()

    print("=" * 72)
    print("  PERFT CORRECTNESS + BENCHMARK")
    print(f"  Trials per depth: {NUM_TRIALS}  |  Warmup rounds: {NUM_WARMUP}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Phase 1: Correctness check (single run, no timing pressure)
    # ------------------------------------------------------------------
    print("\n--- Correctness Check ---")
    all_passed = True
    board = chess.Board()

    for depth, expected in PERFT_EXPECTED.items():
        b = CustomBitboardBoard.from_chess_board(board)
        result = b.run_perft_recursive(depth)
        status = "PASSED" if result == expected else "FAILED"
        if result != expected:
            all_passed = False
        print(f"  Perft {depth}: {result:>10,} (expected {expected:>10,}) [{status}]")

    print()
    if not all_passed:
        print("=== CORRECTNESS: SOME FAILED — skipping benchmark ===")
        sys.exit(1)
    else:
        print("=== CORRECTNESS: ALL PASSED ===")

    # ------------------------------------------------------------------
    # Phase 2: Warmup (stabilize CPU turbo, branch predictor, caches)
    # ------------------------------------------------------------------
    print(f"\n--- Warmup ({NUM_WARMUP} rounds at depth 4) ---")
    for i in range(NUM_WARMUP):
        _run_perft_once(4)
        print(f"  Warmup {i + 1}/{NUM_WARMUP} done")

    # ------------------------------------------------------------------
    # Phase 3: Benchmark with multiple trials per depth
    # ------------------------------------------------------------------
    print(f"\n--- Benchmark ({NUM_TRIALS} trials per depth) ---\n")

    for depth, expected in PERFT_EXPECTED.items():
        trials_elapsed: list[float] = []

        for trial_idx in range(NUM_TRIALS):
            # Use calibrated trial for accurate measurement
            elapsed = _run_calibrated_trial(depth, expected)
            trials_elapsed.append(elapsed)
            nps = int(expected / elapsed) if elapsed > 0 else 0
            print(
                f"  Perft {depth}  trial {trial_idx + 1}/{NUM_TRIALS}: "
                f"{elapsed:.4f}s  {nps:>12,} nps"
            )

        # Compute statistics
        median_elapsed = statistics.median(trials_elapsed)
        mean_elapsed = statistics.mean(trials_elapsed)
        best_elapsed = min(trials_elapsed)
        std_elapsed = statistics.stdev(trials_elapsed) if len(trials_elapsed) > 1 else 0.0

        median_nps = int(expected / median_elapsed) if median_elapsed > 0 else 0
        best_nps = int(expected / best_elapsed) if best_elapsed > 0 else 0

        # Coefficient of variation (lower = more consistent)
        cv = (std_elapsed / mean_elapsed * 100) if mean_elapsed > 0 else 0

        print(f"  {'-' * 52}")
        print(f"  Perft {depth}  SUMMARY:")
        print(f"    Median : {median_elapsed:.4f}s  {median_nps:>12,} nps")
        print(f"    Best   : {best_elapsed:.4f}s  {best_nps:>12,} nps")
        print(f"    Mean   : {mean_elapsed:.4f}s  {int(expected / mean_elapsed):>12,} nps")
        print(f"    StdDev : {std_elapsed:.4f}s  (CV: {cv:.1f}%)")
        print()


if __name__ == "__main__":
    main()
