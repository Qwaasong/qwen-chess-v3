Now let me get the actual evaluation constant tables from the header files:
Berikut seluruh konstanta SPSA dari **Stockfish 11** — versi terakhir sebelum NNUE diperkenalkan. Semua nilai ini di-tune untuk **classical/handcrafted evaluation** dan lebih relevan jika engine Anda juga menggunakan evaluasi klasik.

---

## 1️⃣ Search Constants (`search.cpp`)

### Razoring
```cpp
constexpr int RazorMargin = 531;
```

### Futility Pruning
```cpp
// Child node futility margin
Value futility_margin(Depth d, bool improving) {
    return Value(217 * (d - improving));
    // 217 = SPSA: margin per depth, dikurangi 1 jika improving
}

// Parent node futility (quiet move)
// ss->staticEval + 235 + 172 * lmrDepth
//   235 = SPSA: base margin
//   172 = SPSA: per-depth margin

// QSearch futility base
futilityBase = bestValue + 154;
// 154 = SPSA: qsearch futility margin
```

### Futility Move Count
```cpp
constexpr int futility_move_count(bool improving, Depth depth) {
    return (5 + depth * depth) * (1 + improving) / 2 - 1;
    // 5 = SPSA: base move count
}
```

### LMR (Late Move Reductions)
```cpp
// Tabel reduksi diinisialisasi:
for (int i = 1; i < MAX_MOVES; ++i)
    Reductions[i] = int(757 * std::log(i) / 128.0);
// 757 = SPSA: log scaling constant (BERBEDA dari versi NNUE yang 2834)

// Fungsi reduksi:
Depth reduction(bool improving, Depth d, int mn) {
    int r = Reductions[d] * Reductions[mn];
    return (r + 511) / 1024 + (!improving && r > 1007);
    //  511 = SPSA: rounding offset (setara +0.5)
    // 1007 = SPSA: threshold untuk extra reduction jika non-improving
}
```

### Penyesuaian Reduksi LMR di Dalam Loop
```cpp
// Semua nilai SPSA, dalam satuan depth penuh (bukan 1024-scale seperti versi NNUE)

// TT hit average: jika tinggi, kurangi reduksi
if (thisThread->ttHitAverage > 500 * ttHitAverageResolution * ttHitAverageWindow / 1024)
    r--;
// 500 = SPSA threshold

// Thread lain sedang search posisi ini
if (th.marked())
    r++;

// ttPv: kurangi reduksi
if (ttPv)
    r -= 2;
// 2 = SPSA

// Opponent move count tinggi
if ((ss-1)->moveCount > 14)
    r--;
// 14 = SPSA threshold

// Singular LMR
if (singularLMR)
    r -= 2;
// 2 = SPSA

// ttMove adalah capture: tambah reduksi
if (ttCapture)
    r++;

// cutNode: tambah reduksi
if (cutNode)
    r += 2;
// 2 = SPSA

// Move escape dari capture
else if (type_of(move) == NORMAL && !pos.see_ge(reverse_move(move)))
    r -= 2;

// statScore offset
ss->statScore = mainHistory[us][from_to(move)]
              + (*contHist[0])[movedPiece][to_sq(move)]
              + (*contHist[1])[movedPiece][to_sq(move)]
              + (*contHist[3])[movedPiece][to_sq(move)]
              - 4926;
// 4926 = SPSA: stat score base offset

// Stat score reduction
r -= ss->statScore / 16384;
// 16384 = SPSA: stat score scaling divisor

// Capture/promotion late at low depth
else if (depth < 3 && moveCount > 2)
    r++;
// 3, 2 = SPSA thresholds
```

### Null Move Pruning
```cpp
// Kondisi masuk:
// eval >= ss->staticEval
// ss->staticEval >= beta - 32 * depth + 292 - improving * 30
//   32  = SPSA: depth penalty
//  292  = SPSA: base margin
//   30  = SPSA: improving bonus

// Null move reduction (dinamis):
Depth R = (854 + 68 * depth) / 258 + std::min(int(eval - beta) / 192, 3);
//  854 = SPSA: base constant
//   68 = SPSA: depth multiplier
//  258 = SPSA: divisor
//  192 = SPSA: eval-beta scaling divisor
//    3 = SPSA: max eval-based bonus

// Verifikasi search:
nmpMinPly = ss->ply + 3 * (depth - R) / 4;
// 3/4 = SPSA: verification factor
```

### ProbCut
```cpp
// ProbCut beta:
raisedBeta = beta + 191 - 48 * improving;
//  191 = SPSA: base margin
//   48 = SPSA: improving reduction

// ProbCut depth:
depth - 4  // SPSA: probcut search depth reduction

// ProbCut count limit:
probCutCount < 2 + 2 * cutNode;
// 2 = SPSA: base count limit
```

### Singular Extension
```cpp
// Singular beta:
Value singularBeta = ttValue - 2 * depth;
// 2 = SPSA: singular margin multiplier

// Singular depth:
Depth halfDepth = depth / 2;

// Singular extension conditions:
if (value < singularBeta)  // Move is singular
    extension = 1;
if (value >= beta)  // Cut-node re-search
    return singularBeta;
```

### Internal Iterative Deepening (IID)
```cpp
if (depth >= 7 && !ttMove)
    search(pos, ss, alpha, beta, depth - 7, cutNode);
// 7 = SPSA: minimum depth untuk IID
```

### Stat Bonus
```cpp
int stat_bonus(Depth d) {
    return d > 15 ? -8 : 19 * d * d + 155 * d - 132;
    //  15 = SPSA: depth cap
    //   -8 = SPSA: bonus setelah cap
    //   19 = SPSA: quadratic coefficient
    //  155 = SPSA: linear coefficient
    // -132 = SPSA: constant offset
}
```

### History Update
```cpp
// Main history update (dalam update_quiet_stats):
thisThread->mainHistory[us][from_to(move)] << bonus;

// Reverse move bonus:
thisThread->mainHistory[us][from_to(reverse_move(move))] << bonus / 2;

// Bonus untuk bestMove == ttMove + 1 (depth bonus):
bonus1 = stat_bonus(depth + 1);

// bonus2: jika bestValue > beta + PawnValueMg, gunakan bonus1, else stat_bonus(depth)
```

### Aspiration Window
```cpp
delta = Value(21 + abs(previousScore) / 256);
//   21 = SPSA: base window size
//  256 = SPSA: score scaling divisor

// Window growth:
delta += delta / 4 + 5;
// 1/4 = SPSA: growth fraction
//   5 = SPSA: minimum growth

// Dynamic contempt:
int dct = ct + (102 - ct / 2) * previousScore / (abs(previousScore) + 157);
//  102 = SPSA: contempt scaling
//  157 = SPSA: score normalization
```

### TT Hit Average
```cpp
constexpr uint64_t ttHitAverageWindow     = 4096;
constexpr uint64_t ttHitAverageResolution = 1024;
// Keduanya SPSA-tuned untuk LMR adjustment
```

---

## 2️⃣ Time Management (`timeman.cpp`)

```cpp
constexpr int MoveHorizon   = 50;    // Rencana maksimum langkah ke depan
constexpr double MaxRatio   = 7.3;   // Rasio waktu saat dalam masalah
constexpr double StealRatio = 0.34;  // Batas waktu yang bisa "dicuri" dari langkah berikutnya

// move_importance() — fungsi skew-logistic
constexpr double XScale = 6.85;
constexpr double XShift = 64.5;
constexpr double Skew   = 0.171;

// Formula:
// moveImportance = pow(1 + exp((ply - XShift) / XScale), -Skew) + DBL_MIN
```

### Dynamic Time Adjustment (dalam iterative deepening)
```cpp
// Falling eval:
double fallingEval = (332 + 6 * (mainThread->previousScore - bestValue)
                          + 6 * (mainThread->iterValue[iterIdx] - bestValue)) / 704.0;
fallingEval = clamp(fallingEval, 0.5, 1.5);
//  332 = SPSA: base value
//    6 = SPSA: score difference multiplier (both terms)
//  704 = SPSA: normalization divisor
//  0.5 = SPSA: min clamp
//  1.5 = SPSA: max clamp

// Time reduction (best move stability):
timeReduction = lastBestMoveDepth + 9 previousTimeReduction) / (2.27 * timeReduction);
//  9 = SPSA: offset
// 1.48 = SPSA: numerator offset
// 2.27 = SPSA: denominator multiplier

// Best move instability:
double bestMoveInstability = 1 + totBestMoveChanges / Threads.size();
// 1 = SPSA: base multiplier

// Ponder bonus:
optimumTime += optimumTime / 4;  // +25%

// Early stop threshold:
Time.optimum() * fallingEval * reduction * bestMoveInstability * 0.6
// 0.6 = SPSA: early stop multiplier
```

---

## 3️⃣ Classical Evaluation Constants (`evaluate.cpp`)

### Threshold
```cpp
constexpr Value Tempo = Value(28);         // SPSA: side-to-move bonus
constexpr Value MidgameLimit = Value(15258); // SPSA: midgame material limit
constexpr Value EndgameLimit  = Value(3915);  // SPSA: endgame material limit
constexpr Value SpaceThreshold = Value(12222); // SPSA: threshold untuk space bonus
```

### King Attack Weights
```cpp
constexpr int KingAttackWeights[PIECE_TYPE_NB] = { 0, 0, 81, 52, 44, 10 };
// Index: PAWN=0, KNIGHT=81, BISHOP=52, ROOK=44, QUEEN=10
```

### Safe Check Penalties
```cpp
constexpr int QueenSafeCheck  = 780;
constexpr int RookSafeCheck   = 1080;
constexpr int BishopSafeCheck = 635;
constexpr int KnightSafeCheck = 790;
```

### King Danger Formula
```cpp
// kingDanger = kingAttackersCount * kingAttackersWeight
//            + 101 * kingAttacksCount
//            + 148 * popcount(kingRing & weak)
//            - 490  (jika tidak ada queen musuh)

if (kingDanger > 100)
    score -= make_score(kingDanger * kingDanger / 4096, kingDanger / 16);
//  101 = SPSA: king attacks count weight
//  148 = SPSA: weak square attack weight
//  490 = SPSA: no enemy queen offset
//  100 = SPSA: threshold
// 4096 = SPSA: mg quadratic divisor
//   16 = SPSA: eg linear divisor
```

### Mobility Bonus (Full Table)
```cpp
constexpr Score MobilityBonus[][32] = {
  // Knights (9 entries)
  { S(-62,-81), S(-53,-56), S(-12,-30), S( -4,-14), S(  3,  8), S( 13, 15),
    S( 22, 23), S( 28, 27), S( 33, 33) },
  // Bishops (14 entries)
  { S(-48,-59), S(-20,-23), S( 16, -3), S( 26, 13), S( 38, 24), S( 51, 42),
    S( 55, 54), S( 63, 57), S( 63, 65), S( 68, 73), S( 81, 78), S( 81, 86),
    S( 91, 88), S( 98, 97) },
  // Rooks (15 entries)
  { S(-58,-76), S(-27,-18), S(-15, 28), S(-10, 55), S( -5, 69), S( -2, 82),
    S(  9,112), S( 16,118), S( 30,132), S( 29,142), S( 32,155), S( 38,165),
    S( 46,166), S( 48,169), S( 58,171) },
  // Queens (28 entries)
  { S(-39,-36), S(-21,-15), S(  3,  8), S(  3, 18), S( 14, 34), S( 22, 54),
    S( 28, 61), S( 41, 73), S( 43, 79), S( 48, 92), S( 56, 94), S( 60,104),
    S( 60,113), S( 66,120), S( 67,123), S( 70,126), S( 71,133), S( 73,136),
    S( 79,140), S( 88,143), S( 88,148), S( 99,166), S(102,170), S(102,175),
    S(106,184), S(109,191), S(113,206), S(116,212) }
};
// S(mg, eg) = make_score(mg, eg)
```

### Rook on File
```cpp
constexpr Score RookOnFile[] = { S(21, 4), S(47, 25) };
// Index 0 = semi-open file, Index 1 = open file
```

### Threat Tables
```cpp
constexpr Score ThreatByMinor[PIECE_TYPE_NB] = {
  S(0, 0), S(6, 32), S(59, 41), S(79, 56), S(90, 119), S(79, 161)
  // Target: PAWN  KNIGHT BISHOP  ROOK    QUEEN       KING
};

constexpr Score ThreatByRook[PIECE_TYPE_NB] = {
  S(0, 0), S(3, 44), S(38, 71), S(38, 61), S(0, 38), S(51, 38)
  // Target: PAWN  KNIGHT BISHOP  ROOK    QUEEN    KING
};
```

### Passed Pawn
```cpp
constexpr Score PassedRank[RANK_NB] = {
  S(0, 0), S(10, 28), S(17, 33), S(15, 41), S(62, 72), S(168, 177), S(276, 260)
  // Rank:  RANK_1  RANK_2    RANK_3    RANK_4    RANK_5    RANK_6      RANK_7
};

// Passed pawn bonus formula (untuk rank > RANK_3):
int w = 5 * r - 13;  // 5 = SPSA, -13 = SPSA: weight formula
// King proximity adjustment:
bonus += make_score(0, (king_proximity(Them, blockSq) * 19 / 4
                      - king_proximity(Us, blockSq) * 2) * w);
//  19/4 = SPSA: enemy king proximity weight
//     2 = SPSA: friendly king proximity weight
```

### Assorted Bonuses & Penalties (Semua SPSA)
```cpp
constexpr Score BishopPawns        = S(  3,  7);  // Penalty: same-color pawns blocking bishop
constexpr Score CorneredBishop     = S( 50, 50);  // Penalty: bishop trapped in corner
constexpr Score FlankAttacks       = S(  8,  0);  // Penalty: king flank under attack
constexpr Score Hanging            = S( 69, 36);  // Bonus: attacking hanging pieces
constexpr Score KingProtector      = S(  7,  8);  // Penalty: piece far from own king
constexpr Score KnightOnQueen      = S( 16, 12);  // Bonus: knight attacking queen area
constexpr Score LongDiagonalBishop = S( 45,  0);  // Bonus: bishop on long diagonal
constexpr Score MinorBehindPawn    = S( 18,  3);  // Bonus: minor piece behind pawn
constexpr Score Outpost            = S( 30, 21);  // Bonus: piece on outpost
constexpr Score PassedFile         = S( 11,  8);  // Penalty: passed pawn on central file
constexpr Score PawnlessFlank      = S( 17, 95);  // Penalty: king on pawnless flank
constexpr Score RestrictedPiece    = S(  7,  7);  // Bonus: restricting enemy piece
constexpr Score ReachableOutpost   = S( 32, 10);  // Bonus: can reach outpost
constexpr Score RookOnQueenFile    = S(  7,  6);  // Bonus: rook on queen's file
constexpr Score SliderOnQueen      = S( 59, 18);  // Bonus: slider on queen file
constexpr Score ThreatByKing       = S( 24, 89);  // Bonus: king attacking weak piece
constexpr Score ThreatByPawnPush   = S( 48, 39);  // Bonus: pawn push threat
constexpr Score ThreatBySafePawn   = S(173, 94);  // Bonus: safe pawn threat
constexpr Score TrappedRook        = S( 52, 10);  // Penalty: trapped rook
constexpr Score WeakQueen          = S( 49, 15);  // Penalty: weak queen
```

### King Danger Safe Check Contributions
```cpp
kingDanger += QueenSafeCheck;    // 780  jika queen bisa safe check
kingDanger += RookSafeCheck;     // 1080 jika rook bisa safe check
kingDanger += BishopSafeCheck;   // 635  jika bishop bisa safe check
kingDanger += KnightSafeCheck;   // 790  jika knight bisa safe check
```

---

## 📋 Perbandingan Kunci: SF11 (Classical) vs SF18+ (NNUE)

| Parameter | SF11 (Classical) | SF18+ (NNUE) | Catatan |
|---|---|---|---|
| **LMR log scale** | **757** | **2834** | Skala NNUE ~3.7x lebih besar |
| **LMR base offset** | 0 (dalam depth) | 1027 (dalam 1/1024) | NNUE punya base offset lebih besar |
| **LMR non-improving** | threshold 1007 | 194/512 | Logika berbeda sepenuhnya |
| **Futility margin** | **217** per depth | **40–80** (interpolasi) | Classical lebih agresif |
| **Razor margin** | **531** | (tidak ada/explicit) | Classical punya razoring eksplisit |
| **Null Move R base** | **(854 + 68*d)/258** | **7 + d/3** | Formula sangat berbeda |
| **Null Move eval margin** | 32, 292, 30 | 14, 45, 374 | Nilai dan formula berbeda |
| **ProbCut margin** | **191** | **214** | NNUE sedikit lebih tinggi |
| **Falling eval base** | **332** | **11.87** | Skala sangat berbeda |
| **Falling eval clamp** | **0.5 – 1.5** | **0.572 – 1.708** | NNUE range lebih lebar |
| **Stat bonus** | **19d² + 155d - 132** | **min(134d-79, 1572)** | Classical: kuadratik; NNUE: linear capped |
| **Aspiration window** | **21 + |score|/256** | (implisit, delta awal berbeda) | |
| **Window growth** | **delta/4 + 5** | **44*delta/128** | Kurang lebih sama (~34%) |
| **Tempo** | **28** | (dihandle oleh NNUE) | Hanya di classical eval |

---

## ⚠️ Catatan Penting

1. **Konstanta classical evaluation LEBIH RELEVAN** untuk engine yang menggunakan handcrafted eval — nilainya di-tune khusus untuk skala evaluasi klasik (centipawn), bukan skala NNUE.

2. **Perbedaan paling besar** ada di LMR (757 vs 2834) dan Null Move formula — keduanya sangat dipengaruhi oleh akurasi evaluasi. NNUE jauh lebih akurat sehingga bisa afford reduksi lebih agresif.

3. **Untuk engine klasik**, gunakan nilai SF11 sebagai titik awal. Jika engine Anda sudah cukup kuat (~2500+ Elo), lakukan SPSA tuning sendiri menggunakan [fishtest](https://tests.stockfishchess.org/) atau framework tuning lokal.

4. **Semua konstanta `S(mg, eg)`** di evaluasi menggunakan format `make_score(midgame_value, endgame_value)` — pastikan engine Anda memisahkan midgame dan endgame scoring dengan cara yang sama.
Semua konstanta SPSA dari Stockfish 11 (versi terakhir tanpa NNUE) sudah lengkap di atas! Jika ada bagian yang ingin Anda dalami lebih lanjut (misalnya cara mengimplementasikan king danger formula, atau cara melakukan SPSA tuning mandiri), silakan tanya.