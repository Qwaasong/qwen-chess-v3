# qwen-chess-v3

A high-performance chess engine written in Cython and Python, featuring an optimized Make/Unmake board representation and state-of-the-art search optimizations.

## Key Features & Architecture

### 1. Board Representation & Move Generation
- **Bitboards**: Custom bitboards for piece representation, yielding extremely fast occupancy checks.
- **Magic Bitboards**: Used for sliding pieces (bishops, rooks, queens) for O(1) attack lookup.
- **Incremental Make/Unmake**: An optimized, O(1) copy-free Make/Unmake board representation. Reversible changes are applied directly to the board state, while non-reversible changes (such as castling rights, en passant squares, halfmove clocks, and captured pieces) are recorded in a lightweight stack slot.
- **Zobrist Hashing**: Fully incremental hashing to track board states and enable Transposition Table lookup.

### 2. Search Engine
- **Alpha-Beta Negamax Search**: Principal Variation Search (PVS) with iterative deepening.
- **Transposition Table (TT)**: Stores evaluated positions, depths, bounds, and best moves to prune search trees.
- **Late Move Reductions (LMR)**: Safely reduces search depth for less promising moves.
- **Null Move Pruning (NMP)**: Prunes safe positions by checking if the opponent cannot create a threat even with a free turn.
- **Quiescence Search**: Evaluates tactical captures and checks at leaf nodes to avoid the horizon effect.
- **Heuristics**:
  - MVV-LVA (Most Valuable Victim - Least Valuable Attacker) for capture ordering.
  - Killer Move Heuristic for prioritizing quiet moves that caused cutoffs.
  - History Heuristic for scoring quiet moves based on historical cutoffs.
  - Countermove Heuristic to prioritize moves commonly played in response to the opponent's previous move.

### 3. Evaluation
- **PeSTO's Piece-Square Tables (PST)**: Uses tapered evaluation interpolating between Middlegame and Endgame phases.
- **King Safety & Pawn Structure**: Evaluates pawn shield integrity, passed pawns, and king vulnerabilities.

---

## Getting Started

### Prerequisites
- Python 3.10+
- C Compiler (MSVC Build Tools on Windows, GCC/Clang on Linux/macOS)
- `python-chess` library for interfacing and testing

### Compilation
Build the Cython extensions in-place:
```bash
python setup.py build_ext --inplace
```

### Running Tests and Benchmarks
1. **Chess Logic and Repetition Tests**:
   ```bash
   python -m unittest test_engine_logic.py
   ```

2. **Perft Correctness & Speed Verification**:
   ```bash
   python test_perft.py
   ```

3. **Evaluation Verification (against baseline)**:
   ```bash
   python verify_eval.py
   ```

4. **NPS Benchmarks**:
   ```bash
   python bench.py
   ```

---

## UCI Protocol Support
This engine implements standard UCI (Universal Chess Interface) commands to connect with any compatible chess GUI (such as Arena, Cute Chess, or BanksiaGUI).
```bash
python uci.py
```
Supported UCI features include:
- `position fen [FEN]` and `position startpos`
- `go depth [D]`, `go wtime [W] btime [B]`, `go infinite`, etc.
- `ponder` and `ponderhit`
- `stop` and `quit`