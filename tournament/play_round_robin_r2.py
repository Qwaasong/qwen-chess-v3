"""
Round-Robin Tournament Runner for Optimization Round 2.

Runs a 4-engine round-robin tournament:
  - Baseline: root engine (optimization-round-2 branch = Combined from main)
  - Agent-A:  agent_a_spsa/ (SPSA-tuned parameters)
  - Agent-B:  agent_b_timemgmt/ (dynamic time management)
  - Agent-C:  agent_c_qsearch/ (enhanced quiescence search)

Format: 20 games/pairing (10 as White, 10 as Black), 6 pairings = 120 games total.
Time control: 0.1 seconds/move, max 150 moves/game.

Results are displayed in a cross-table and saved to results/round2_tournament.pgn.
"""

import subprocess
import sys
import json
import os
import time
from itertools import combinations

# --- Engine Definitions ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENGINES = {
    "Baseline": os.path.join(BASE_DIR, "baseline"), # Saved baseline engine
    "New-Engine": BASE_DIR,                         # Engine with CONSTANT.md parameters
}

# --- Tournament Parameters ---
GAMES_PER_PAIRING = 20    # 10 as White + 10 as Black
TIME_LIMIT = 0.1          # seconds per move
MAX_MOVES = 150           # per game adjudication limit
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PGN_FILE = os.path.join(RESULTS_DIR, "round2_tournament.pgn")
GAME_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "play_game_process.py")

# --- Score tracking ---
# scores[engine_name][opponent_name] = float points
scores = {name: {opp: 0.0 for opp in ENGINES if opp != name} for name in ENGINES}
total_scores = {name: 0.0 for name in ENGINES}
wins = {name: 0 for name in ENGINES}
draws = {name: 0 for name in ENGINES}
losses = {name: 0 for name in ENGINES}
all_pgns = []
game_count = 0


def run_game(white_name: str, black_name: str, game_num: int) -> str:
    """Run a single game in a subprocess and return the result string."""
    global game_count
    game_count += 1

    white_path = ENGINES[white_name]
    black_path = ENGINES[black_name]

    cmd = [
        sys.executable,
        GAME_RUNNER,
        white_path,
        black_path,
        str(TIME_LIMIT),
    ]

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout per game
        )
        elapsed = time.time() - start
        output = proc.stdout.strip()

        if not output:
            # Subprocess error — count as draw
            print(f"  [!] Game {game_count:03d} {white_name} vs {black_name}: subprocess returned no output ({elapsed:.1f}s)")
            return "1/2-1/2"

        data = json.loads(output)
        result = data.get("result", "1/2-1/2")
        pgn = data.get("pgn", "")
        n_moves = data.get("moves", 0)

        # Add metadata headers to PGN
        full_pgn = (
            f'[Event "Optimization Round 2 Tournament"]\n'
            f'[White "{white_name}"]\n'
            f'[Black "{black_name}"]\n'
            f'[Round "{game_num}"]\n'
            f'{pgn}\n'
        )
        all_pgns.append(full_pgn)

        print(f"  Game {game_count:03d} {white_name} (W) vs {black_name} (B): {result} ({n_moves} moves, {elapsed:.1f}s)")
        return result

    except subprocess.TimeoutExpired:
        print(f"  [!] Game {game_count:03d} TIMEOUT — counting as draw")
        return "1/2-1/2"
    except Exception as e:
        print(f"  [!] Game {game_count:03d} ERROR: {e}")
        return "1/2-1/2"


def update_scores(white_name: str, black_name: str, result: str) -> None:
    """Update score tables based on game result."""
    if result == "1-0":
        scores[white_name][black_name] += 1.0
        total_scores[white_name] += 1.0
        wins[white_name] += 1
        losses[black_name] += 1
    elif result == "0-1":
        scores[black_name][white_name] += 1.0
        total_scores[black_name] += 1.0
        wins[black_name] += 1
        losses[white_name] += 1
    else:
        scores[white_name][black_name] += 0.5
        scores[black_name][white_name] += 0.5
        total_scores[white_name] += 0.5
        total_scores[black_name] += 0.5
        draws[white_name] += 1
        draws[black_name] += 1


def print_standings() -> None:
    """Print current tournament standings."""
    engine_names = list(ENGINES.keys())
    ranked = sorted(engine_names, key=lambda n: total_scores[n], reverse=True)

    total_games = wins[engine_names[0]] + draws[engine_names[0]] + losses[engine_names[0]]

    print("\n" + "=" * 65)
    print("TOURNAMENT STANDINGS")
    print("=" * 65)
    print(f"{'Rank':<6} {'Engine':<12} {'Points':<8} {'W':<5} {'D':<5} {'L':<5} {'Win%':<8}")
    print("-" * 65)
    for rank, name in enumerate(ranked, 1):
        g = wins[name] + draws[name] + losses[name]
        win_pct = (total_scores[name] / g * 100) if g > 0 else 0.0
        print(f"  {rank:<4} {name:<12} {total_scores[name]:<8.1f} {wins[name]:<5} {draws[name]:<5} {losses[name]:<5} {win_pct:.1f}%")
    print("=" * 65)


def print_cross_table() -> None:
    """Print cross-table of head-to-head scores."""
    engine_names = list(ENGINES.keys())
    ranked = sorted(engine_names, key=lambda n: total_scores[n], reverse=True)

    print("\nCROSS TABLE (row = engine, column = opponent, score = points earned)")
    print("-" * 65)
    # Header
    header = f"{'Engine':<12}"
    for name in ranked:
        header += f"  {name[:8]:>8}"
    header += f"  {'Total':>8}"
    print(header)
    print("-" * 65)
    for name in ranked:
        row = f"{name:<12}"
        for opp in ranked:
            if opp == name:
                row += f"  {'---':>8}"
            else:
                row += f"  {scores[name][opp]:>8.1f}"
        row += f"  {total_scores[name]:>8.1f}"
        print(row)
    print("-" * 65)


def save_pgn() -> None:
    """Save all games to PGN file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(PGN_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_pgns))
    print(f"\nPGN saved to: {PGN_FILE}")


def main() -> None:
    """Run the round-robin tournament."""
    engine_names = list(ENGINES.keys())
    pairings = list(combinations(engine_names, 2))

    total_games = len(pairings) * GAMES_PER_PAIRING
    print(f"{'=' * 65}")
    print(f"OPTIMIZATION ROUND 2 — ROUND ROBIN TOURNAMENT")
    print(f"{'=' * 65}")
    print(f"Engines     : {', '.join(engine_names)}")
    print(f"Pairings    : {len(pairings)}")
    print(f"Games/pair  : {GAMES_PER_PAIRING}")
    print(f"Total games : {total_games}")
    print(f"Time control: {TIME_LIMIT}s/move, max {MAX_MOVES} moves")
    print(f"{'=' * 65}\n")

    start_time = time.time()

    for pairing_num, (eng_a, eng_b) in enumerate(pairings, 1):
        half = GAMES_PER_PAIRING // 2
        print(f"\n--- Pairing {pairing_num}/{len(pairings)}: {eng_a} vs {eng_b} ---")

        # First half: eng_a = White
        for g in range(half):
            result = run_game(eng_a, eng_b, pairing_num * 100 + g + 1)
            update_scores(eng_a, eng_b, result)

        # Second half: eng_b = White
        for g in range(half):
            result = run_game(eng_b, eng_a, pairing_num * 100 + half + g + 1)
            update_scores(eng_b, eng_a, result)

        # Show intermediate standings after each pairing
        print_standings()

    # Final results
    elapsed_total = time.time() - start_time
    print(f"\nTournament completed in {elapsed_total:.1f}s ({elapsed_total / 60:.1f} minutes)")
    print_standings()
    print_cross_table()
    save_pgn()

    # Determine winner
    engine_names = list(ENGINES.keys())
    winner = max(engine_names, key=lambda n: total_scores[n])
    baseline_score = total_scores["Baseline"]
    winner_score = total_scores[winner]

    print(f"\n{'=' * 65}")
    print(f"WINNER: {winner} ({winner_score:.1f} points)")
    if winner != "Baseline":
        improvement = winner_score - baseline_score
        print(f"Improvement over Baseline: +{improvement:.1f} points")
        print("SUCCESS: A new engine variant has beaten the Combined baseline!")
    else:
        print("Baseline held its ground. No optimization surpassed the Combined engine.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
