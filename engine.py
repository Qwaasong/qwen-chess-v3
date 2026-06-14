"""Search Engine and Evaluator for qwen-chess-v3."""

import time

import chess

try:
    from board_cy import (
        CustomBitboardBoard,
        get_move_source,
        get_move_dest,
        get_move_flag,
        FLAG_EP,
        FLAG_PROMOTE_N,
        FLAG_PROMOTE_Q,
        FLAG_PROMOTE_R,
        FLAG_PROMOTE_B,
        lsb,
        clear_bit,
        WHITE,
        BLACK,
        KNIGHT_ATTACKS,
        KING_ATTACKS,
        PAWN_ATTACKS,
        get_bishop_attacks,
        get_rook_attacks,
        get_queen_attacks,
    )
except ImportError:
    from board import (
        CustomBitboardBoard,
        get_move_source,
        get_move_dest,
        get_move_flag,
        FLAG_EP,
        FLAG_PROMOTE_N,
        FLAG_PROMOTE_Q,
        FLAG_PROMOTE_R,
        FLAG_PROMOTE_B,
        lsb,
        clear_bit,
        WHITE,
        BLACK,
        KNIGHT_ATTACKS,
        KING_ATTACKS,
        PAWN_ATTACKS,
        get_bishop_attacks,
        get_rook_attacks,
        get_queen_attacks,
    )

# --- Configuration ---
PIECE_VALUES = [100, 320, 330, 500, 900, 0]

PIECE_VALUES_MG = [82, 337, 365, 477, 1025, 0]
PIECE_VALUES_EG = [94, 281, 297, 512, 936, 0]

PAWN_TABLE_MG = (
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0
)

PAWN_TABLE_EG = (
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0
)

KNIGHT_TABLE_MG = (
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23
)

KNIGHT_TABLE_EG = (
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64
)

BISHOP_TABLE_MG = (
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21
)

BISHOP_TABLE_EG = (
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17
)

ROOK_TABLE_MG = (
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26
)

ROOK_TABLE_EG = (
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20
)

QUEEN_TABLE_MG = (
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50
)

QUEEN_TABLE_EG = (
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41
)

KING_TABLE_MG = (
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14
)

KING_TABLE_EG = (
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
      -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43
)

PST_MG = {
    0: PAWN_TABLE_MG,
    1: KNIGHT_TABLE_MG,
    2: BISHOP_TABLE_MG,
    3: ROOK_TABLE_MG,
    4: QUEEN_TABLE_MG,
    5: KING_TABLE_MG
}

PST_EG = {
    0: PAWN_TABLE_EG,
    1: KNIGHT_TABLE_EG,
    2: BISHOP_TABLE_EG,
    3: ROOK_TABLE_EG,
    4: QUEEN_TABLE_EG,
    5: KING_TABLE_EG
}

MOBILITY_KNIGHT_MG = [-62, -53, -12, -4, 3, 13, 22, 28, 33]
MOBILITY_KNIGHT_EG = [-81, -56, -30, -14, 8, 15, 23, 27, 33]
MOBILITY_BISHOP_MG = [-48, -20, 16, 26, 38, 51, 55, 63, 63, 68, 81, 81, 91, 98]
MOBILITY_BISHOP_EG = [-59, -23, -3, 13, 24, 42, 54, 57, 65, 73, 78, 86, 88, 97]
MOBILITY_ROOK_MG = [-58, -27, -15, -10, -5, -2, 9, 16, 30, 29, 32, 38, 46, 48, 58]
MOBILITY_ROOK_EG = [-76, -18, 28, 55, 69, 82, 112, 118, 132, 142, 155, 165, 166, 169, 171]
MOBILITY_QUEEN_MG = [-39, -21, 3, 3, 14, 22, 28, 41, 43, 48, 56, 60, 60, 66, 67, 70, 71, 73, 79, 88, 88, 99, 102, 102, 106, 109, 113, 116]
MOBILITY_QUEEN_EG = [-36, -15, 8, 18, 34, 54, 61, 73, 79, 92, 94, 104, 113, 120, 123, 126, 133, 136, 140, 143, 148, 166, 170, 175, 184, 191, 206, 212]

# Precompute material and PST values for fast evaluation
white_piece_values_mg = {}
black_piece_values_mg = {}
white_piece_values_eg = {}
black_piece_values_eg = {}

for p_type in range(6):
    white_piece_values_mg[p_type] = [0] * 64
    black_piece_values_mg[p_type] = [0] * 64
    white_piece_values_eg[p_type] = [0] * 64
    black_piece_values_eg[p_type] = [0] * 64
    for sq in range(64):
        rank = sq // 8
        file = sq % 8

        # White perspective (A8 is top-left, index 0 in PST)
        w_idx = (7 - rank) * 8 + file
        white_piece_values_mg[p_type][sq] = (
            PIECE_VALUES_MG[p_type] + PST_MG[p_type][w_idx]
        )
        white_piece_values_eg[p_type][sq] = (
            PIECE_VALUES_EG[p_type] + PST_EG[p_type][w_idx]
        )

        # Black perspective (A1 is bottom-left, index 56 in PST)
        b_idx = rank * 8 + file
        black_piece_values_mg[p_type][sq] = (
            PIECE_VALUES_MG[p_type] + PST_MG[p_type][b_idx]
        )
        black_piece_values_eg[p_type][sq] = (
            PIECE_VALUES_EG[p_type] + PST_EG[p_type][b_idx]
        )


# --- Search State ---
class SearchInfo:
    """Class to keep track of search parameters and node counts."""

    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit = 0.0
        self.stop = False
        self.current_depth = 0
        self.root_ponder_move = -1


info = SearchInfo()
tt = {}  # Transposition Table
killer_moves = [[0] * 64 for _ in range(2)]
history_moves = [[0] * 64 for _ in range(12)]
counter_moves = [[-1] * 64 for _ in range(12)]
LMR_REDUCTIONS = [[0] * 64 for _ in range(64)]

def init_lmr_reductions():
    import math
    reductions_1d = [0] * 64
    for i in range(1, 64):
        reductions_1d[i] = int(757 * math.log(i) / 128.0)
    for depth in range(64):
        for move_count in range(64):
            if depth == 0 or move_count == 0:
                LMR_REDUCTIONS[depth][move_count] = 0
            else:
                r = reductions_1d[depth] * reductions_1d[move_count]
                reduction_depth = (r + 511) // 1024 + (1 if r > 1007 else 0)
                LMR_REDUCTIONS[depth][move_count] = reduction_depth * 1024

init_lmr_reductions()


# --- Pawn evaluation masks ---
PASSED_PAWN_MASK_W = [0] * 64
PASSED_PAWN_MASK_B = [0] * 64
ADJACENT_FILES_MASK = [0] * 8

def init_pawn_masks():
    for f in range(8):
        mask = 0
        if f > 0:
            mask |= (0x0101010101010101 << (f - 1))
        if f < 7:
            mask |= (0x0101010101010101 << (f + 1))
        ADJACENT_FILES_MASK[f] = mask

    for sq in range(64):
        r_coord = sq // 8
        f_coord = sq % 8

        min_f = f_coord - 1
        if min_f < 0:
            min_f = 0
        max_f = f_coord + 1
        if max_f > 7:
            max_f = 7

        # White passed pawn mask
        mask = 0
        for f in range(min_f, max_f + 1):
            for r in range(r_coord + 1, 8):
                mask |= (1 << (r * 8 + f))
        PASSED_PAWN_MASK_W[sq] = mask

        # Black passed pawn mask
        mask = 0
        for f in range(min_f, max_f + 1):
            for r in range(0, r_coord):
                mask |= (1 << (r * 8 + f))
        PASSED_PAWN_MASK_B[sq] = mask

init_pawn_masks()


# --- Evaluation Function ---
def evaluate(board: CustomBitboardBoard) -> int:
    """Static evaluation of the board with Tapered Evaluation (Middlegame vs Endgame)."""
    # Count non-pawn material to determine phase (Knights=1, Bishops=1, Rooks=2, Queens=4)
    knights = board.bitboards[1].bit_count() + board.bitboards[7].bit_count()
    bishops = board.bitboards[2].bit_count() + board.bitboards[8].bit_count()
    rooks = board.bitboards[3].bit_count() + board.bitboards[9].bit_count()
    queens = board.bitboards[4].bit_count() + board.bitboards[10].bit_count()
    
    phase = knights * 1 + bishops * 1 + rooks * 2 + queens * 4
    if phase > 24:
        phase = 24

    score_mg = 0
    score_eg = 0

    # 1. Bishop Pair Bonus (30 cp MG, 50 cp EG)
    w_bishops = board.bitboards[2].bit_count()
    b_bishops = board.bitboards[8].bit_count()
    if w_bishops >= 2:
        score_mg += 30
        score_eg += 50
    if b_bishops >= 2:
        score_mg -= 30
        score_eg -= 50

    friendly_occ_w = board.occupancies[0]
    friendly_occ_b = board.occupancies[1]
    both_occ = board.occupancies[2]

    # White pieces (0-5)
    for p_idx in range(6):
        bb = board.bitboards[p_idx]
        while bb:
            sq_idx = lsb(bb)
            bb = clear_bit(bb, sq_idx)
            score_mg += white_piece_values_mg[p_idx][sq_idx]
            score_eg += white_piece_values_eg[p_idx][sq_idx]
            
            # Mobility (White Knights=1, Bishops=2, Rooks=3, Queens=4)
            if p_idx == 1:
                mobility = (KNIGHT_ATTACKS[sq_idx] & ~friendly_occ_w).bit_count()
                score_mg += mobility * 3
                score_eg += mobility * 3
            elif p_idx == 2:
                mobility = (get_bishop_attacks(sq_idx, both_occ) & ~friendly_occ_w).bit_count()
                score_mg += mobility * 3
                score_eg += mobility * 3
            elif p_idx == 3:
                mobility = (get_rook_attacks(sq_idx, both_occ) & ~friendly_occ_w).bit_count()
                score_mg += mobility * 3
                score_eg += mobility * 3
            elif p_idx == 4:
                mobility = (get_queen_attacks(sq_idx, both_occ) & ~friendly_occ_w).bit_count()
                score_mg += mobility * 3
                score_eg += mobility * 3

    # Black pieces (6-11)
    for p_idx in range(6):
        bb = board.bitboards[p_idx + 6]
        while bb:
            sq_idx = lsb(bb)
            bb = clear_bit(bb, sq_idx)
            score_mg -= black_piece_values_mg[p_idx][sq_idx]
            score_eg -= black_piece_values_eg[p_idx][sq_idx]
            
            # Mobility (Black Knights=1, Bishops=2, Rooks=3, Queens=4)
            if p_idx == 1:
                mobility = (KNIGHT_ATTACKS[sq_idx] & ~friendly_occ_b).bit_count()
                score_mg -= mobility * 3
                score_eg -= mobility * 3
            elif p_idx == 2:
                mobility = (get_bishop_attacks(sq_idx, both_occ) & ~friendly_occ_b).bit_count()
                score_mg -= mobility * 3
                score_eg -= mobility * 3
            elif p_idx == 3:
                mobility = (get_rook_attacks(sq_idx, both_occ) & ~friendly_occ_b).bit_count()
                score_mg -= mobility * 3
                score_eg -= mobility * 3
            elif p_idx == 4:
                mobility = (get_queen_attacks(sq_idx, both_occ) & ~friendly_occ_b).bit_count()
                score_mg -= mobility * 3
                score_eg -= mobility * 3

    # 3. Doubled Pawns penalty
    w_pawns = board.bitboards[0]  # P_P = 0
    b_pawns = board.bitboards[6]  # P_p = 6
    
    for f in range(8):
        file_mask = 0x0101010101010101 << f
        w_count = (w_pawns & file_mask).bit_count()
        if w_count > 1:
            score_mg += (w_count - 1) * -15
            score_eg += (w_count - 1) * -15
        
        b_count = (b_pawns & file_mask).bit_count()
        if b_count > 1:
            score_mg -= (b_count - 1) * -15
            score_eg -= (b_count - 1) * -15

    # 4. Passed & Isolated Pawns
    passed_pawn_mg = [0, 10, 17, 15, 62, 168, 276, 0]
    passed_pawn_eg = [0, 28, 33, 41, 72, 177, 260, 0]

    # White Pawns
    bb = w_pawns
    while bb:
        sq_idx = lsb(bb)
        bb = clear_bit(bb, sq_idx)
        r = sq_idx // 8
        f = sq_idx % 8

        # Isolated
        if (w_pawns & ADJACENT_FILES_MASK[f]) == 0:
            score_mg -= 15
            score_eg -= 15
        
        # Passed
        if (b_pawns & PASSED_PAWN_MASK_W[sq_idx]) == 0:
            score_mg += passed_pawn_mg[r]
            score_eg += passed_pawn_eg[r]

    # Black Pawns
    bb = b_pawns
    while bb:
        sq_idx = lsb(bb)
        bb = clear_bit(bb, sq_idx)
        r = sq_idx // 8
        f = sq_idx % 8

        # Isolated
        if (b_pawns & ADJACENT_FILES_MASK[f]) == 0:
            score_mg += 15
            score_eg += 15

        # Passed
        if (w_pawns & PASSED_PAWN_MASK_B[sq_idx]) == 0:
            score_mg -= passed_pawn_mg[7 - r]
            score_eg -= passed_pawn_eg[7 - r]

    # 5. King Safety (files g/h or b/c) on first rank (White) / eighth rank (Black)
    w_king_bb = board.bitboards[5] # P_K = 5
    if w_king_bb:
        sq_idx = lsb(w_king_bb)
        r = sq_idx // 8
        f = sq_idx % 8
        if r == 0:
            w_ks_penalty = 0
            if f == 6 or f == 7: # King side (G1/H1)
                for file_idx in range(5, 8):
                    if (w_pawns & (1 << (8 + file_idx))):
                        pass
                    elif (w_pawns & (1 << (16 + file_idx))):
                        w_ks_penalty += 10
                    else:
                        w_ks_penalty += 25
            elif f == 1 or f == 2: # Queen side (B1/C1)
                for file_idx in range(0, 3):
                    if (w_pawns & (1 << (8 + file_idx))):
                        pass
                    elif (w_pawns & (1 << (16 + file_idx))):
                        w_ks_penalty += 10
                    else:
                        w_ks_penalty += 25
            score_mg -= w_ks_penalty

    b_king_bb = board.bitboards[11] # P_k = 11
    if b_king_bb:
        sq_idx = lsb(b_king_bb)
        r = sq_idx // 8
        f = sq_idx % 8
        if r == 7:
            b_ks_penalty = 0
            if f == 6 or f == 7: # King side (G8/H8)
                for file_idx in range(5, 8):
                    if (b_pawns & (1 << (48 + file_idx))):
                        pass
                    elif (b_pawns & (1 << (40 + file_idx))):
                        b_ks_penalty += 10
                    else:
                        b_ks_penalty += 25
            elif f == 1 or f == 2: # Queen side (B8/C8)
                for file_idx in range(0, 3):
                    if (b_pawns & (1 << (48 + file_idx))):
                        pass
                    elif (b_pawns & (1 << (40 + file_idx))):
                        b_ks_penalty += 10
                    else:
                        b_ks_penalty += 25
            score_mg += b_ks_penalty

    # Interpolate between Middlegame and Endgame and return as C-like rounded integer
    return int((score_mg * phase + score_eg * (24 - phase)) / 24)


# --- Move Ordering / Scoring ---
def get_mvv_lva_score(board: CustomBitboardBoard, move: int) -> int:
    """Calculates MVV-LVA score for a move."""
    from_sq = get_move_source(move)
    to_sq = get_move_dest(move)
    flag = get_move_flag(move)

    attacker = board.get_piece_at(from_sq)
    if attacker is None:
        return 0

    attacker_val = PIECE_VALUES[attacker % 6]

    if flag == FLAG_EP:
        victim_val = PIECE_VALUES[0]  # Pawn
    else:
        victim = board.get_piece_at(to_sq)
        if victim is None:
            return 0
        victim_val = PIECE_VALUES[victim % 6]

    score = victim_val * 10 - attacker_val
    if flag >= FLAG_PROMOTE_N:
        if flag == FLAG_PROMOTE_Q:
            score += 9000
        elif flag == FLAG_PROMOTE_R:
            score += 5000
        elif flag == FLAG_PROMOTE_B:
            score += 3300
        else:
            score += 3200
    return score


# Piece values for SEE (simpler, material-only)
SEE_VALUES = [100, 320, 330, 500, 900, 20000, 0]  # P, N, B, R, Q, K, none

def static_exchange_evaluation(board: CustomBitboardBoard, move: int) -> int:
    """Estimate the material exchange value of a capture move.
    Returns positive if winning capture, negative if losing.
    """
    to_sq = get_move_dest(move)
    from_sq = get_move_source(move)
    flag = get_move_flag(move)

    if flag == FLAG_EP:
        captured_val = SEE_VALUES[0]  # pawn
    else:
        victim = board.get_piece_at(to_sq)
        if victim is None:
            return 0
        captured_val = SEE_VALUES[victim % 6]

    attacker = board.get_piece_at(from_sq)
    if attacker is None:
        return 0
    attacker_val = SEE_VALUES[attacker % 6]

    if captured_val >= attacker_val:
        return captured_val - attacker_val

    opp_side = BLACK if board.side_to_move == WHITE else WHITE
    if board.is_square_attacked(to_sq, opp_side):
        return captured_val - attacker_val

    return captured_val


# --- Quiescence Search ---
def quiescence(
    board: CustomBitboardBoard,
    alpha: int,
    beta: int,
    color: int,
    ply: int,
    qdepth: int = 0,
) -> int:
    """Quiescence search to avoid horizon effect on captures and checks."""
    info.nodes += 1

    in_check = board.in_check()
    stand_pat = 0
    if not in_check:
        stand_pat = color * evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

    pseudo_moves = board.generate_pseudo_legal_moves()
    moves_to_search = []
    legal_moves_searched = 0

    if in_check:
        for move in pseudo_moves:
            score = get_mvv_lva_score(board, move)
            to_sq = get_move_dest(move)
            flag = get_move_flag(move)
            is_cap = flag == FLAG_EP or (board.get_piece_at(to_sq) is not None)
            if is_cap:
                score += 10000
            elif flag >= FLAG_PROMOTE_N:
                score += 9000
            moves_to_search.append((score, move))
    else:
        for move in pseudo_moves:
            to_sq = get_move_dest(move)
            flag = get_move_flag(move)
            is_cap = flag == FLAG_EP or (board.get_piece_at(to_sq) is not None)

            if is_cap:
                # Prune captures that are static losses (SEE < 0)
                if static_exchange_evaluation(board, move) < 0:
                    continue
                score = get_mvv_lva_score(board, move)
                moves_to_search.append((score, move))
            elif qdepth < 1:
                if not board.make_move(move):
                    board.unmake_move()
                    continue
                gives_check = board.in_check()
                board.unmake_move()
                if gives_check:
                    moves_to_search.append((0, move))

    moves_to_search.sort(key=lambda x: x[0], reverse=True)

    for _, move in moves_to_search:
        if not board.make_move(move):
            board.unmake_move()
            continue
        legal_moves_searched += 1
        val = -quiescence(board, -beta, -alpha, -color, ply + 1, qdepth + 1)
        board.unmake_move()

        if val >= beta:
            return beta
        alpha = max(alpha, val)

    if in_check and legal_moves_searched == 0:
        return -99999 + ply

    return alpha


def is_repetition(board: CustomBitboardBoard) -> bool:
    key = board.zobrist_key
    halfmove = board.halfmove_clock
    keys = board.history_keys
    hist_len = len(keys)
    limit = hist_len - halfmove
    if limit < 0:
        limit = 0
    # Search backwards from hist_len - 2 down to limit with step -2
    for i in range(hist_len - 2, limit - 1, -2):
        if keys[i] == key:
            return True
    return False


# --- Negamax Search ---
def negamax(
    board: CustomBitboardBoard,
    depth: int,
    alpha: int,
    beta: int,
    color: int,
    ply: int,
    extensions: int = 0,
    prev_move: int = -1,
    allow_nmp: bool = True,
) -> int:
    """Recursively searches the game tree using Alpha-Beta Negamax algorithm."""
    info.nodes += 1

    # --- Draw detection ---
    if ply > 0:
        if board.halfmove_clock >= 100 or is_repetition(board):
            return 0

    if info.nodes % 4096 == 0:
        if time.time() - info.start_time > info.time_limit:
            info.stop = True
            return 0

    alpha_orig = alpha
    in_check = board.in_check()
    extended = 0

    # Transposition Table lookup (performed BEFORE check extension)
    key = board.zobrist_key
    tt_move = None
    if key in tt:
        tt_entry = tt[key]
        if tt_entry['depth'] >= depth:
            val = tt_entry['val']
            # TT Mate Score Adjustment when reading
            if val > MATE_THRESHOLD:
                val -= ply
            elif val < -MATE_THRESHOLD:
                val += ply

            flag = tt_entry['flag']
            if flag == 0:  # Exact
                return val
            elif flag == 1:  # Lower bound
                alpha = max(alpha, val)
            elif flag == 2:  # Upper bound
                beta = min(beta, val)

            if alpha >= beta:
                return val
        tt_move = tt_entry.get('move')

    # Check Extensions
    if in_check and extensions < 8:
        depth += 1
        extended = 1

    # --- Null Move Pruning (NMP) ---
    has_non_pawn = False
    static_eval = color * evaluate(board)
    if (allow_nmp and
        depth >= 3 and 
        not in_check and 
        ply > 0 and 
        not info.stop and
        static_eval >= beta - 32 * depth + 292):
        
        # Check if side to move has non-pawn material (zugzwang safety)
        if color == 1:
            for p in range(1, 5):
                if board.bitboards[p] != 0:
                    has_non_pawn = True
                    break
        else:
            for p in range(7, 11):
                if board.bitboards[p] != 0:
                    has_non_pawn = True
                    break

        if has_non_pawn:
            # Dynamic reduction R matching Stockfish 11
            R = (854 + 68 * depth) // 258 + min(int((static_eval - beta) / 192), 3)
            R = max(1, min(depth - 1, R))
            
            if board.make_null_move():
                val = -negamax(board, depth - 1 - R, -beta, -beta + 1, -color, ply + 1, extensions + extended, -1, False)
                board.unmake_move()
                
                if info.stop:
                    return 0

                if val >= beta:
                    # Verification search for deep cuts (anti-zugzwang verification)
                    if depth >= 6:
                        val = negamax(board, depth - 1 - R, beta - 1, beta, color, ply, extensions + extended, prev_move, False)
                        if val >= beta:
                            return val
                    else:
                        return val

    if depth <= 0:
        val = quiescence(board, alpha, beta, color, ply, 0)
        tt_val = val
        if tt_val > MATE_THRESHOLD:
            tt_val += ply
        elif tt_val < -MATE_THRESHOLD:
            tt_val -= ply

        q_flag = 0
        if val <= alpha:
            q_flag = 2
        elif val >= beta:
            q_flag = 1

        if key not in tt or 0 >= tt[key]['depth']:
            tt[key] = {'depth': 0, 'val': tt_val, 'flag': q_flag, 'move': None}
        return val

    pseudo_moves = board.generate_pseudo_legal_moves()

    # Resolve Countermove Heuristic
    counter_move = -1
    if prev_move != -1:
        prev_to = get_move_dest(prev_move)
        prev_piece = board.get_piece_at(prev_to)
        if prev_piece is not None:
            counter_move = counter_moves[prev_piece][prev_to]

    # Move Ordering
    moves_scored = []
    for move in pseudo_moves:
        if move == tt_move:
            score = 1000000
        else:
            from_sq = get_move_source(move)
            to_sq = get_move_dest(move)
            flag = get_move_flag(move)
            is_cap = flag == FLAG_EP or (board.get_piece_at(to_sq) is not None)
            
            if is_cap:
                score = 100000 + get_mvv_lva_score(board, move)
            elif flag >= FLAG_PROMOTE_N:
                score = 90000 + PIECE_VALUES[0]
            else:
                if move == killer_moves[0][ply]:
                    score = 9000
                elif move == killer_moves[1][ply]:
                    score = 8000
                elif move == counter_move:
                    score = 7500
                else:
                    p_type = board.get_piece_at(from_sq)
                    if p_type is not None:
                        score = history_moves[p_type][to_sq]
                    else:
                        score = 0
        moves_scored.append((score, move))

    moves_scored.sort(key=lambda x: x[0], reverse=True)

    legal_moves_searched = 0
    pv_node = (beta - alpha) > 1

    best_val = -INFINITE
    best_move = None

    for _, move in moves_scored:
        from_sq = get_move_source(move)
        to_sq = get_move_dest(move)
        flag = get_move_flag(move)
        is_cap = flag == FLAG_EP or (board.get_piece_at(to_sq) is not None)

        if not board.make_move(move):
            board.unmake_move()
            continue
        
        legal_moves_searched += 1

        if legal_moves_searched == 1:
            val = -negamax(board, depth - 1, -beta, -alpha, -color, ply + 1, extensions + extended, move, True)
        else:
            # --- Late Move Reductions (LMR) ---
            if (depth >= 2 and 
                legal_moves_searched > 4 and 
                not is_cap and 
                flag < FLAG_PROMOTE_N and 
                not in_check):
                
                # Lookup reduction from precomputed table
                r_val = LMR_REDUCTIONS[depth][legal_moves_searched] if legal_moves_searched < 64 else LMR_REDUCTIONS[depth][63]
                r_int = r_val // 1024

                # Reductions tuning
                if pv_node:
                    r_int -= 1
                if move == killer_moves[0][ply] or move == killer_moves[1][ply]:
                    r_int -= 1
                
                p_type = board.get_piece_at(to_sq) # the piece is already moved to to_sq
                if p_type is not None:
                    if history_moves[p_type][to_sq] > 3000:
                        r_int -= 1
                    elif history_moves[p_type][to_sq] < 500:
                        r_int += 1

                if r_int < 1:
                    r_int = 1
                if r_int >= depth:
                    r_int = depth - 1

                # Search at reduced depth with null window
                val = -negamax(board, depth - 1 - r_int, -alpha - 1, -alpha, -color, ply + 1, extensions + extended, move, True)
                
                # Re-search at full depth with null window if reduced search failed high
                if val > alpha and r_int > 0:
                    val = -negamax(board, depth - 1, -alpha - 1, -alpha, -color, ply + 1, extensions + extended, move, True)
            else:
                # Search at full depth with null window
                val = -negamax(board, depth - 1, -alpha - 1, -alpha, -color, ply + 1, extensions + extended, move, True)
            
            # Re-search with full window if null window search failed high
            if val > alpha and val < beta:
                val = -negamax(board, depth - 1, -beta, -alpha, -color, ply + 1, extensions + extended, move, True)
        
        board.unmake_move()

        if info.stop:
            return 0

        if val > best_val:
            best_val = val
            best_move = move

        if val > alpha:
            alpha = val

        if alpha >= beta:
            # Beta cutoff: update Killer & History heuristics for quiet moves
            if not is_cap and flag < FLAG_PROMOTE_N:
                if killer_moves[0][ply] != move:
                    killer_moves[1][ply] = killer_moves[0][ply]
                    killer_moves[0][ply] = move
                
                p_type = board.get_piece_at(from_sq)
                if p_type is not None:
                    history_moves[p_type][to_sq] += depth * depth
                    if history_moves[p_type][to_sq] > 5000:
                        history_moves[p_type][to_sq] = 5000
                
                # Countermove update
                if prev_move != -1:
                    prev_to = get_move_dest(prev_move)
                    prev_piece = board.get_piece_at(prev_to)
                    if prev_piece is not None:
                        counter_moves[prev_piece][prev_to] = move
            break

    if info.stop:
        return 0

    if legal_moves_searched == 0:
        if board.in_check():
            return -99999 + ply
        return 0

    tt_flag = 0
    if best_val <= alpha_orig:
        tt_flag = 2
    elif best_val >= beta:
        tt_flag = 1

    tt_val = best_val
    if tt_val > MATE_THRESHOLD:
        tt_val += ply
    elif tt_val < -MATE_THRESHOLD:
        tt_val -= ply

    if key not in tt or depth >= tt[key]['depth']:
        tt[key] = {'depth': depth, 'val': tt_val, 'flag': tt_flag, 'move': best_move}
    return best_val



# --- Main Engine API ---
def get_best_move(
    chess_board: chess.Board,
    time_limit: float = 1.0,
    depth_limit: int = 0,
    print_info: bool = False,
) -> chess.Move | None:
    """Finds the best move using iterative deepening search."""
    # 2. Setup Search state
    info.start_time = time.time()
    info.time_limit = time_limit
    info.stop = False
    info.nodes = 0

    # 3. Convert python-chess Board to CustomBitboardBoard
    board = CustomBitboardBoard.from_chess_board(chess_board)

    global killer_moves, history_moves, counter_moves
    killer_moves = [[0] * 64 for _ in range(2)]
    history_moves = [[0] * 64 for _ in range(12)]
    counter_moves = [[-1] * 64 for _ in range(12)]

    legal_moves = board.generate_legal_moves()
    if not legal_moves:
        return None

    best_move_so_far = legal_moves[0]
    color = 1 if board.side_to_move == WHITE else -1

    depth = 1
    last_score = 0
    while not info.stop:
        if depth_limit > 0 and depth > depth_limit:
            break

        # Age history moves (decay by 50% at the start of each iteration)
        for i in range(12):
            for j in range(64):
                history_moves[i][j] //= 2

        info.current_depth = depth
        if depth < 5:
            negamax(board, depth, -INFINITE, INFINITE, color, 0, 0, -1, True)
            key_tuple = board.zobrist_key
            if key_tuple in tt:
                last_score = tt[key_tuple]['val']
        else:
            delta = 21 + abs(last_score) // 256
            alpha_aw = last_score - delta
            beta_aw = last_score + delta
            while not info.stop:
                if alpha_aw < -INFINITE: alpha_aw = -INFINITE
                if beta_aw > INFINITE: beta_aw = INFINITE
                
                score = negamax(board, depth, alpha_aw, beta_aw, color, 0, 0, -1, True)
                
                if info.stop:
                    break
                
                key_tuple = board.zobrist_key
                if key_tuple in tt:
                    last_score = tt[key_tuple]['val']
                else:
                    last_score = score
                
                if last_score <= alpha_aw:
                    beta_aw = alpha_aw
                    delta += delta // 4 + 5
                    alpha_aw = last_score - delta
                elif last_score >= beta_aw:
                    alpha_aw = beta_aw
                    delta += delta // 4 + 5
                    beta_aw = last_score + delta
                else:
                    break

        if info.stop:
            break

        # Retrieve best move from Transposition Table
        key_tuple = board.zobrist_key
        cdef_score = 0
        if key_tuple in tt and tt[key_tuple]['move'] is not None:
            best_move_so_far = tt[key_tuple]['move']
            cdef_score = tt[key_tuple]['val']

        if print_info:
            score = cdef_score
            if abs(score) > MATE_THRESHOLD:
                mate_plies = 99999 - abs(score)
                mate_moves = (mate_plies + 1) // 2
                if score < 0:
                    mate_moves = -mate_moves
                score_str = f"mate {mate_moves}"
            else:
                score_str = f"cp {score}"

            elapsed = time.time() - info.start_time
            elapsed_ms = int(elapsed * 1000)
            nps = int(info.nodes / elapsed) if elapsed > 0.0 else 0
            best_move_obj = board.to_chess_move(best_move_so_far)
            best_move_uci = best_move_obj.uci()
            safe_print(
                f"info depth {depth} score {score_str} nodes {info.nodes} "
                f"nps {nps} time {elapsed_ms} pv {best_move_uci}"
            )

        if time.time() - info.start_time > time_limit * 0.95:
            break

        depth += 1

    return board.to_chess_move(best_move_so_far)

# --- Cython Engine Wrapper & Fallback ---
try:
    import engine_cy
    USING_CYTHON = True

    _get_best_move_cy = engine_cy.get_best_move_cy
    safe_print = engine_cy.safe_print

    def get_best_move(
        chess_board: chess.Board,
        time_limit: float = 1.0,
        depth_limit: int = 0,
        print_info: bool = False,
    ) -> chess.Move | None:
        info.root_ponder_move = -1
        return _get_best_move_cy(chess_board, time_limit, depth_limit, print_info)

    class CythonInfoWrapper:
        @property
        def nodes(self):
            return engine_cy.get_nodes()
        @nodes.setter
        def nodes(self, value):
            engine_cy.set_nodes(value)
        @property
        def current_depth(self):
            return engine_cy.get_depth()
        @current_depth.setter
        def current_depth(self, value):
            engine_cy.set_depth(value)
        @property
        def stop(self):
            return engine_cy.get_stop()
        @stop.setter
        def stop(self, value):
            engine_cy.set_stop(value)
        @property
        def start_time(self):
            return engine_cy.get_start_time()
        @start_time.setter
        def start_time(self, value):
            engine_cy.set_start_time(value)
        @property
        def time_limit(self):
            return engine_cy.get_time_limit()
        @time_limit.setter
        def time_limit(self, value):
            engine_cy.set_time_limit(value)
        @property
        def root_ponder_move(self):
            return engine_cy.get_ponder_move()
        @root_ponder_move.setter
        def root_ponder_move(self, value):
            engine_cy.set_ponder_move(value)

    info = CythonInfoWrapper()

    def clear_tt():
        engine_cy.clear_tt()

    def get_ponder_move():
        return engine_cy.get_ponder_move()

    def get_ponder_move_uci(chess_board):
        return engine_cy.get_ponder_move_uci(chess_board)
except ImportError:
    USING_CYTHON = False
    
    import threading
    stdout_lock = threading.Lock()
    def safe_print(msg):
        with stdout_lock:
            print(msg, flush=True)

    def clear_tt():
        global tt
        tt.clear()

    def get_ponder_move():
        return -1

    def get_ponder_move_uci(chess_board):
        return None


def main() -> None:
    """Main function to test the chess engine v3 on a specific position."""
    test_board = chess.Board(
        "7k/8/1K2PPPP/8/B7/8/4pppp/8 w - - 0 1"
    )
    print("Testing Engine v3 on starting position:")
    print(test_board)
    start = time.time()
    test_move = get_best_move(test_board, time_limit=30.0)
    end = time.time()
    print(f"\nBest move: {test_move}")
    print(f"Time taken: {end - start:.4f}s")
    print(f"Nodes searched: {info.nodes}")
    print(f"Depth reached: {info.current_depth}")


if __name__ == "__main__":
    main()
