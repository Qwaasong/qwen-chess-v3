import subprocess
import sys
import json
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_PATH = os.path.join(BASE_DIR, "baseline")
OPTIMIZED_PATH = BASE_DIR
GAME_RUNNER = os.path.join(BASE_DIR, "tournament", "play_game_process.py")

GAMES = 20
TIME_LIMIT = 0.1

scores = {"Optimized": 0.0, "Baseline": 0.0}
wins = {"Optimized": 0, "Baseline": 0}
draws = {"Optimized": 0, "Baseline": 0}
losses = {"Optimized": 0, "Baseline": 0}

print("=" * 60)
print("TOURNAMENT: OPTIMIZED VS BASELINE (20 Games, 0.1s Time Control)")
print("=" * 60)

for i in range(GAMES):
    # Alternate colors
    if i % 2 == 0:
        white_name, white_path = "Optimized", OPTIMIZED_PATH
        black_name, black_path = "Baseline", BASELINE_PATH
    else:
        white_name, white_path = "Baseline", BASELINE_PATH
        black_name, black_path = "Optimized", OPTIMIZED_PATH

    cmd = [sys.executable, GAME_RUNNER, white_path, black_path, str(TIME_LIMIT)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = proc.stdout.strip()
        if not output:
            result = "1/2-1/2"
        else:
            data = json.loads(output)
            result = data.get("result", "1/2-1/2")
    except Exception as e:
        print(f"Error in game {i+1}: {e}")
        result = "1/2-1/2"

    # Update scores
    if result == "1-0":
        scores[white_name] += 1.0
        wins[white_name] += 1
        losses[black_name] += 1
    elif result == "0-1":
        scores[black_name] += 1.0
        wins[black_name] += 1
        losses[white_name] += 1
    else:
        scores["Optimized"] += 0.5
        scores["Baseline"] += 0.5
        draws["Optimized"] += 1
        draws["Baseline"] += 1

    print(f"Game {i+1:02d}: White ({white_name}) vs Black ({black_name}) -> {result} | Scores: Optimized {scores['Optimized']:.1f} - {scores['Baseline']:.1f} Baseline")

print("=" * 60)
print("FINAL RESULTS:")
print(f"Optimized: {scores['Optimized']:.1f} pts ({wins['Optimized']} W, {draws['Optimized']} D, {losses['Optimized']} L)")
print(f"Baseline: {scores['Baseline']:.1f} pts ({wins['Baseline']} W, {draws['Baseline']} D, {losses['Baseline']} L)")
print("=" * 60)
