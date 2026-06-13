# cython: language_level=3
# type: ignore
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
"""Search Engine and Evaluator for qwen-chess-v3 in Cython."""

cdef extern from *:
    """
    #ifdef _WIN32
    #include <windows.h>
    static double get_time_ms() {
        return (double)GetTickCount();
    }
    #else
    #include <sys/time.h>
    static double get_time_ms() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
    }
    #endif
    """
    double get_time_ms() noexcept nogil

import time
import random
import chess
from libc.string cimport memset, memcpy
from libc.math cimport log
from board_cy cimport (
    CustomBitboardBoard,
    CMoveList,
    CGameState,
    ZOBRIST_PIECES,
    ZOBRIST_CASTLING,
    ZOBRIST_EP,
    ZOBRIST_SIDE,
    cy_get_evaluation_bonuses,
    cy_is_square_attacked,
)

# --- Constants ---
cdef int WHITE = 0
cdef int BLACK = 1
cdef int INFINITE = 10000000
cdef int MATE_THRESHOLD = 90000

cdef int PIECE_VALUES[6]
PIECE_VALUES[0] = 100
PIECE_VALUES[1] = 320
PIECE_VALUES[2] = 330
PIECE_VALUES[3] = 500
PIECE_VALUES[4] = 900
PIECE_VALUES[5] = 0

# Tapered piece values from PeSTO
cdef int PIECE_VALUES_MG[6]
PIECE_VALUES_MG[0] = 82
PIECE_VALUES_MG[1] = 337
PIECE_VALUES_MG[2] = 365
PIECE_VALUES_MG[3] = 477
PIECE_VALUES_MG[4] = 1025
PIECE_VALUES_MG[5] = 0

cdef int PIECE_VALUES_EG[6]
PIECE_VALUES_EG[0] = 94
PIECE_VALUES_EG[1] = 281
PIECE_VALUES_EG[2] = 297
PIECE_VALUES_EG[3] = 512
PIECE_VALUES_EG[4] = 936
PIECE_VALUES_EG[5] = 0

# PST Tables from PeSTO
cdef int PAWN_TABLE_MG[64]
PAWN_TABLE_MG[:] = [
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0
]

cdef int PAWN_TABLE_EG[64]
PAWN_TABLE_EG[:] = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0
]

cdef int KNIGHT_TABLE_MG[64]
KNIGHT_TABLE_MG[:] = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23
]

cdef int KNIGHT_TABLE_EG[64]
KNIGHT_TABLE_EG[:] = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64
]

cdef int BISHOP_TABLE_MG[64]
BISHOP_TABLE_MG[:] = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21
]

cdef int BISHOP_TABLE_EG[64]
BISHOP_TABLE_EG[:] = [
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17
]

cdef int ROOK_TABLE_MG[64]
ROOK_TABLE_MG[:] = [
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26
]

cdef int ROOK_TABLE_EG[64]
ROOK_TABLE_EG[:] = [
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20
]

cdef int QUEEN_TABLE_MG[64]
QUEEN_TABLE_MG[:] = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50
]

cdef int QUEEN_TABLE_EG[64]
QUEEN_TABLE_EG[:] = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41
]

cdef int KING_TABLE_MG[64]
KING_TABLE_MG[:] = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14
]

cdef int KING_TABLE_EG[64]
KING_TABLE_EG[:] = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
      -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43
]

cdef const int* PST_MG[6]
PST_MG[0] = PAWN_TABLE_MG
PST_MG[1] = KNIGHT_TABLE_MG
PST_MG[2] = BISHOP_TABLE_MG
PST_MG[3] = ROOK_TABLE_MG
PST_MG[4] = QUEEN_TABLE_MG
PST_MG[5] = KING_TABLE_MG

cdef const int* PST_EG[6]
PST_EG[0] = PAWN_TABLE_EG
PST_EG[1] = KNIGHT_TABLE_EG
PST_EG[2] = BISHOP_TABLE_EG
PST_EG[3] = ROOK_TABLE_EG
PST_EG[4] = QUEEN_TABLE_EG
PST_EG[5] = KING_TABLE_EG

# Precomputed static evaluation tables for board_cy
cdef int white_piece_values_mg[6][64]
cdef int black_piece_values_mg[6][64]
cdef int white_piece_values_eg[6][64]
cdef int black_piece_values_eg[6][64]

cdef void init_piece_values() noexcept:
    cdef int p_type, sq, rank, file, w_idx, b_idx
    for p_type in range(6):
        for sq in range(64):
            rank = sq // 8
            file = sq % 8
            # White perspective
            w_idx = (7 - rank) * 8 + file
            white_piece_values_mg[p_type][sq] = PIECE_VALUES_MG[p_type] + PST_MG[p_type][w_idx]
            white_piece_values_eg[p_type][sq] = PIECE_VALUES_EG[p_type] + PST_EG[p_type][w_idx]
            # Black perspective
            b_idx = rank * 8 + file
            black_piece_values_mg[p_type][sq] = PIECE_VALUES_MG[p_type] + PST_MG[p_type][b_idx]
            black_piece_values_eg[p_type][sq] = PIECE_VALUES_EG[p_type] + PST_EG[p_type][b_idx]

init_piece_values()

cdef inline void add_piece_eval(int piece, int sq, int *score_mg, int *score_eg, int *phase) noexcept nogil:
    if piece < 6:
        score_mg[0] += white_piece_values_mg[piece][sq]
        score_eg[0] += white_piece_values_eg[piece][sq]
    else:
        score_mg[0] -= black_piece_values_mg[piece - 6][sq]
        score_eg[0] -= black_piece_values_eg[piece - 6][sq]
    
    cdef int p_type = piece % 6
    if p_type == 1:   # Knight
        phase[0] += 1
    elif p_type == 2: # Bishop
        phase[0] += 1
    elif p_type == 3: # Rook
        phase[0] += 2
    elif p_type == 4: # Queen
        phase[0] += 4

cdef inline void remove_piece_eval(int piece, int sq, int *score_mg, int *score_eg, int *phase) noexcept nogil:
    if piece < 6:
        score_mg[0] -= white_piece_values_mg[piece][sq]
        score_eg[0] -= white_piece_values_eg[piece][sq]
    else:
        score_mg[0] += black_piece_values_mg[piece - 6][sq]
        score_eg[0] += black_piece_values_eg[piece - 6][sq]
    
    cdef int p_type = piece % 6
    if p_type == 1:   # Knight
        phase[0] -= 1
    elif p_type == 2: # Bishop
        phase[0] -= 1
    elif p_type == 3: # Rook
        phase[0] -= 2
    elif p_type == 4: # Queen
        phase[0] -= 4

# --- Portable LSB and Popcount inline ---
cdef extern from *:
    """
    #ifdef _MSC_VER
    #include <intrin.h>
    static __forceinline int _cy_lsb_impl2(unsigned long long bb) {
        if (!bb) return -1;
        unsigned long idx;
        _BitScanForward64(&idx, bb);
        return (int)idx;
    }
    #if defined(_M_AMD64) || defined(_M_ARM64)
    static __forceinline int _cy_popcount_impl2(unsigned long long bb) {
        return (int)__popcnt64(bb);
    }
    #else
    static __forceinline int _cy_popcount_impl2(unsigned long long bb) {
        bb = bb - ((bb >> 1) & 0x5555555555555555ULL);
        bb = (bb & 0x3333333333333333ULL) + ((bb >> 2) & 0x3333333333333333ULL);
        return (int)((((bb + (bb >> 4)) & 0xF0F0F0F0F0F0F0FULL) * 0x101010101010101ULL) >> 56);
    }
    #endif
    #else
    static __inline__ int _cy_lsb_impl2(unsigned long long bb) {
        if (!bb) return -1;
        return __builtin_ctzll(bb);
    }
    static __inline__ int _cy_popcount_impl2(unsigned long long bb) {
        return __builtin_popcountll(bb);
    }
    #endif
    """
    int _cy_lsb_impl2(unsigned long long bb) nogil
    int _cy_popcount_impl2(unsigned long long bb) nogil

cdef inline int cy_lsb(unsigned long long bb) nogil:
    return _cy_lsb_impl2(bb)

cdef inline int cy_popcount(unsigned long long bb) nogil:
    return _cy_popcount_impl2(bb)


cdef inline unsigned long long cy_clear_bit(unsigned long long bb, int sq) nogil:
    return bb & ~(<unsigned long long>1 << sq)

cdef inline int cy_get_move_source(int move) nogil:
    return move & 0x3F

cdef inline int cy_get_move_dest(int move) nogil:
    return (move >> 6) & 0x3F

cdef inline int cy_get_move_flag(int move) nogil:
    return (move >> 12) & 0x0F

# --- Search State ---
cdef struct SearchInfo:
    long long nodes
    double start_time
    double time_limit
    bint stop
    int current_depth
    unsigned long long root_key
    int root_best_move
    int root_ponder_move

cdef SearchInfo info

# --- Transposition Table (Raw C Array) ---
cdef struct TTEntry:
    unsigned long long key
    int depth
    int val
    int flag
    int move

# 1,048,576 entries = 24 MB
DEF TT_SIZE = 1048576
cdef TTEntry _tt[TT_SIZE]

# Killer Moves, History, and Countermove Heuristics
cdef int killer_moves[2][64]
cdef int history_moves[12][64]
cdef int counter_moves[12][64]
cdef int static_evals[256]

# LMR Reductions
cdef int LMR_REDUCTIONS[64][64]

cdef void init_lmr_reductions() noexcept:
    cdef int depth, move_count, i, r, reduction_depth
    cdef int reductions_1d[64]
    reductions_1d[0] = 0
    for i in range(1, 64):
        reductions_1d[i] = <int>(757 * log(i) / 128.0)
    for depth in range(64):
        for move_count in range(64):
            if depth == 0 or move_count == 0:
                LMR_REDUCTIONS[depth][move_count] = 0
            else:
                r = reductions_1d[depth] * reductions_1d[move_count]
                reduction_depth = (r + 511) // 1024 + (1 if r > 1007 else 0)
                LMR_REDUCTIONS[depth][move_count] = reduction_depth * 1024

cdef bint has_sufficient_material_bb(unsigned long long *bitboards) noexcept nogil:
    cdef int w_knights, b_knights, w_bishops, b_bishops, total_pieces, w_sq, b_sq
    
    # Pawns
    if bitboards[0] != 0 or bitboards[6] != 0:
        return True
    # Rooks
    if bitboards[3] != 0 or bitboards[9] != 0:
        return True
    # Queens
    if bitboards[4] != 0 or bitboards[10] != 0:
        return True
        
    w_knights = cy_popcount(bitboards[1])
    b_knights = cy_popcount(bitboards[7])
    w_bishops = cy_popcount(bitboards[2])
    b_bishops = cy_popcount(bitboards[8])
    
    total_pieces = w_knights + b_knights + w_bishops + b_bishops
    
    if total_pieces == 0:
        return False
        
    if total_pieces == 1:
        return False
        
    if w_bishops == 1 and b_bishops == 1 and w_knights == 0 and b_knights == 0:
        w_sq = cy_lsb(bitboards[2])
        b_sq = cy_lsb(bitboards[8])
        if (((w_sq >> 3) ^ (w_sq & 7)) & 1) == (((b_sq >> 3) ^ (b_sq & 7)) & 1):
            return False
            
    return True

cdef bint is_repetition(unsigned long long *search_history, int root_history_len, int ply, unsigned long long key, int halfmove_clock) noexcept nogil:
    cdef int current_len = root_history_len + ply
    cdef int limit = current_len - halfmove_clock
    if limit < 0:
        limit = 0
    cdef int i = current_len - 2
    while i >= limit:
        if search_history[i] == key:
            return True
        i -= 2
    return False

cdef list extract_pv(CGameState root_state, int depth, CustomBitboardBoard shell):
    cdef list pv_moves = []
    cdef CGameState state = root_state
    cdef CGameState child_state
    cdef CMoveList pseudo
    cdef int i, move, idx
    cdef TTEntry *entry
    cdef list visited_keys = []
    cdef bint found = False
    
    for d in range(depth):
        key = state.zobrist_key
        if key in visited_keys:
            break
        visited_keys.append(key)
        
        idx = key & (TT_SIZE - 1)
        entry = &_tt[idx]
        if entry.key != key or entry.move == -1:
            break
            
        move = entry.move
        
        pseudo.count = 0
        load_state_to_shell(&state, shell)
        shell._generate_pseudo_legal_moves_c(&pseudo)
        
        found = False
        for i in range(pseudo.count):
            if pseudo.moves[i] == move:
                found = True
                break
                
        if not found:
            break
            
        child_state = state
        make_move_on_state(&child_state, move)
        if not is_state_legal(&child_state, shell, state.side_to_move):
            break
            
        pv_moves.append(move)
        state = child_state
        
    return pv_moves

# Scored Moves Structures and Sorting
cdef struct CScoredMove:
    int move
    int score

cdef struct CScoredMoveList:
    CScoredMove moves[256]
    int count

cdef inline void sort_moves(CScoredMoveList *mlist) noexcept nogil:
    cdef int i, j
    cdef CScoredMove temp
    for i in range(1, mlist.count):
        temp = mlist.moves[i]
        j = i - 1
        while j >= 0 and mlist.moves[j].score < temp.score:
            mlist.moves[j + 1] = mlist.moves[j]
            j -= 1
        mlist.moves[j + 1] = temp

# --- Move Picker Stages ---
cdef enum MovePickerStage:
    STAGE_TT_MOVE = 0
    STAGE_GEN_CAPTURES = 1
    STAGE_CAPTURES = 2
    STAGE_KILLERS = 3
    STAGE_COUNTERMOVE = 4
    STAGE_GEN_QUIETS = 5
    STAGE_QUIETS = 6
    STAGE_DONE = 7

cdef enum QMovePickerStage:
    QSTAGE_GEN_CAPTURES = 0
    QSTAGE_CAPTURES = 1
    QSTAGE_QUIET_CHECKS = 2
    QSTAGE_DONE = 3

cdef struct MovePicker:
    int stage
    int tt_move
    int killer_0
    int killer_1
    int counter_move
    CScoredMoveList captures
    CScoredMoveList quiets
    bint captures_generated
    bint quiets_generated
    int capture_idx
    int quiet_idx

cdef struct QMovePicker:
    int stage
    int index
    bint in_check
    int qdepth
    CScoredMoveList moves

cdef inline void init_move_picker(
    MovePicker *mp,
    int tt_move,
    int killer_0,
    int killer_1,
    int counter_move
) noexcept nogil:
    mp.stage = STAGE_TT_MOVE
    mp.tt_move = tt_move
    mp.killer_0 = killer_0
    mp.killer_1 = killer_1
    mp.counter_move = counter_move
    mp.captures.count = 0
    mp.quiets.count = 0
    mp.captures_generated = False
    mp.quiets_generated = False
    mp.capture_idx = 0
    mp.quiet_idx = 0

cdef inline void generate_and_score_captures(MovePicker *mp, CustomBitboardBoard board) noexcept nogil:
    cdef CMoveList raw_moves
    raw_moves.count = 0
    board._generate_captures_c(&raw_moves)
    
    mp.captures.count = 0
    cdef int i, move
    for i in range(raw_moves.count):
        move = raw_moves.moves[i]
        mp.captures.moves[i].move = move
        mp.captures.moves[i].score = get_mvv_lva_score(board, move)
    mp.captures.count = raw_moves.count
    sort_moves(&mp.captures)
    mp.captures_generated = True

cdef inline void generate_and_score_quiets(MovePicker *mp, CustomBitboardBoard board, int ply) noexcept nogil:
    cdef CMoveList raw_moves
    raw_moves.count = 0
    board._generate_quiets_c(&raw_moves)
    
    mp.quiets.count = 0
    cdef int i, move, from_sq, p_type
    for i in range(raw_moves.count):
        move = raw_moves.moves[i]
        mp.quiets.moves[i].move = move
        from_sq = move & 0x3F
        p_type = board.piece_map[from_sq]
        if p_type != -1:
            mp.quiets.moves[i].score = history_moves[p_type][(move >> 6) & 0x3F]
        else:
            mp.quiets.moves[i].score = 0
    mp.quiets.count = raw_moves.count
    sort_moves(&mp.quiets)
    mp.quiets_generated = True

cdef int next_move(MovePicker *mp, CustomBitboardBoard board, int ply) noexcept nogil:
    cdef int move = -1
    cdef int i
    cdef bint found = False

    while True:
        if mp.stage == STAGE_TT_MOVE:
            mp.stage = STAGE_GEN_CAPTURES
            if mp.tt_move != -1:
                # Verify if the TT move is pseudo-legal
                if cy_get_move_flag(mp.tt_move) == 3 or board.piece_map[cy_get_move_dest(mp.tt_move)] != -1:
                    if not mp.captures_generated:
                        generate_and_score_captures(mp, board)
                    for i in range(mp.captures.count):
                        if mp.captures.moves[i].move == mp.tt_move:
                            found = True
                            break
                else:
                    if not mp.quiets_generated:
                        generate_and_score_quiets(mp, board, ply)
                    for i in range(mp.quiets.count):
                        if mp.quiets.moves[i].move == mp.tt_move:
                            found = True
                            break
                if found:
                    return mp.tt_move

        elif mp.stage == STAGE_GEN_CAPTURES:
            mp.stage = STAGE_CAPTURES
            if not mp.captures_generated:
                generate_and_score_captures(mp, board)

        elif mp.stage == STAGE_CAPTURES:
            if mp.capture_idx < mp.captures.count:
                move = mp.captures.moves[mp.capture_idx].move
                mp.capture_idx += 1
                if move != mp.tt_move:
                    return move
            else:
                mp.stage = STAGE_KILLERS

        elif mp.stage == STAGE_KILLERS:
            mp.stage = STAGE_COUNTERMOVE
            if mp.killer_0 != -1 and mp.killer_0 != mp.tt_move:
                if not mp.quiets_generated:
                    generate_and_score_quiets(mp, board, ply)
                for i in range(mp.quiets.count):
                    if mp.quiets.moves[i].move == mp.killer_0:
                        return mp.killer_0
            if mp.killer_1 != -1 and mp.killer_1 != mp.tt_move:
                if not mp.quiets_generated:
                    generate_and_score_quiets(mp, board, ply)
                for i in range(mp.quiets.count):
                    if mp.quiets.moves[i].move == mp.killer_1:
                        return mp.killer_1

        elif mp.stage == STAGE_COUNTERMOVE:
            mp.stage = STAGE_GEN_QUIETS
            if mp.counter_move != -1 and mp.counter_move != mp.tt_move and mp.counter_move != mp.killer_0 and mp.counter_move != mp.killer_1:
                if not mp.quiets_generated:
                    generate_and_score_quiets(mp, board, ply)
                for i in range(mp.quiets.count):
                    if mp.quiets.moves[i].move == mp.counter_move:
                        return mp.counter_move

        elif mp.stage == STAGE_GEN_QUIETS:
            mp.stage = STAGE_QUIETS
            if not mp.quiets_generated:
                generate_and_score_quiets(mp, board, ply)

        elif mp.stage == STAGE_QUIETS:
            if mp.quiet_idx < mp.quiets.count:
                move = mp.quiets.moves[mp.quiet_idx].move
                mp.quiet_idx += 1
                if (move != mp.tt_move and 
                    move != mp.killer_0 and 
                    move != mp.killer_1 and 
                    move != mp.counter_move):
                    return move
            else:
                mp.stage = STAGE_DONE

        elif mp.stage == STAGE_DONE:
            return -1

cdef inline void init_qmove_picker(
    QMovePicker *qmp,
    bint in_check,
    int qdepth
) noexcept nogil:
    qmp.stage = QSTAGE_GEN_CAPTURES
    qmp.index = 0
    qmp.in_check = in_check
    qmp.qdepth = qdepth
    qmp.moves.count = 0

cdef int next_qmove(QMovePicker *qmp, CustomBitboardBoard board) noexcept nogil:
    cdef int move = -1
    cdef CMoveList raw_moves
    cdef int i, score, to_sq, flag, is_cap
    cdef bint gives_check

    while True:
        if qmp.stage == QSTAGE_GEN_CAPTURES:
            qmp.stage = QSTAGE_CAPTURES
            qmp.moves.count = 0
            if qmp.in_check:
                raw_moves.count = 0
                board._generate_pseudo_legal_moves_c(&raw_moves)
                for i in range(raw_moves.count):
                    move = raw_moves.moves[i]
                    score = get_mvv_lva_score(board, move)
                    to_sq = (move >> 6) & 0x3F
                    flag = (move >> 12) & 0x0F
                    is_cap = flag == 3 or (board.piece_map[to_sq] != -1)
                    if is_cap:
                        score += 10000
                    elif flag >= 8:
                        score += 9000
                    qmp.moves.moves[i].move = move
                    qmp.moves.moves[i].score = score
                qmp.moves.count = raw_moves.count
                sort_moves(&qmp.moves)
            else:
                raw_moves.count = 0
                board._generate_captures_c(&raw_moves)
                for i in range(raw_moves.count):
                    move = raw_moves.moves[i]
                    qmp.moves.moves[i].move = move
                    qmp.moves.moves[i].score = get_mvv_lva_score(board, move)
                qmp.moves.count = raw_moves.count
                sort_moves(&qmp.moves)

        elif qmp.stage == QSTAGE_CAPTURES:
            if qmp.index < qmp.moves.count:
                move = qmp.moves.moves[qmp.index].move
                qmp.index += 1
                return move
            else:
                if not qmp.in_check and qmp.qdepth < 1:
                    qmp.stage = QSTAGE_QUIET_CHECKS
                    qmp.index = 0
                    qmp.moves.count = 0
                    
                    raw_moves.count = 0
                    board._generate_quiets_c(&raw_moves)
                    for i in range(raw_moves.count):
                        move = raw_moves.moves[i]
                        if not board.make_move_c(move):
                            board.unmake_move_c()
                            continue
                        gives_check = board.in_check_c()
                        board.unmake_move_c()
                        if gives_check:
                            qmp.moves.moves[qmp.moves.count].move = move
                            qmp.moves.moves[qmp.moves.count].score = 0
                            qmp.moves.count += 1
                else:
                    qmp.stage = QSTAGE_DONE

        elif qmp.stage == QSTAGE_QUIET_CHECKS:
            if qmp.index < qmp.moves.count:
                move = qmp.moves.moves[qmp.index].move
                qmp.index += 1
                return move
            else:
                qmp.stage = QSTAGE_DONE

        elif qmp.stage == QSTAGE_DONE:
            return -1

def clear_tt():
    """Clears the Transposition Table memory."""
    memset(_tt, 0, sizeof(_tt))

# --- Copy-Make State Helpers ---
cdef void load_state_to_shell(CGameState *state, CustomBitboardBoard shell) noexcept nogil:
    memcpy(shell._bb,       state.bitboards,    12 * sizeof(unsigned long long))
    memcpy(shell._occ,      state.occupancies,   3 * sizeof(unsigned long long))
    memcpy(shell.piece_map, state.piece_map,    64 * sizeof(int))
    shell.side_to_move    = state.side_to_move
    shell.castling_rights = state.castling_rights
    shell.en_passant_sq   = state.en_passant_sq
    shell.halfmove_clock  = state.halfmove_clock
    shell.fullmove_number = state.fullmove_number
    shell.zobrist_key     = state.zobrist_key
    shell.score_mg        = state.score_mg
    shell.score_eg        = state.score_eg
    shell.phase           = state.phase
    shell._history_len    = 0

cdef void make_null_move_on_state(CGameState *state) noexcept nogil:
    cdef int old_ep = state.en_passant_sq
    state.en_passant_sq = 64
    state.halfmove_clock += 1

    # XOR out old EP file
    if old_ep != 64:
        state.zobrist_key ^= ZOBRIST_EP[old_ep % 8]

    # Toggle side and XOR side
    state.side_to_move = BLACK if state.side_to_move == WHITE else WHITE
    state.zobrist_key ^= ZOBRIST_SIDE

    if state.side_to_move == WHITE:
        state.fullmove_number += 1

cdef int negamax_copymake(
    CGameState *state,
    int depth,
    int alpha,
    int beta,
    int color,
    int ply,
    int extensions,
    int prev_move,
    CustomBitboardBoard shell,
    unsigned long long *search_history,
    int root_history_len,
) noexcept nogil:
    info.nodes += 1

    if info.nodes % 4096 == 0:
        if get_time_ms() - info.start_time > info.time_limit:
            info.stop = True
            return 0

    search_history[root_history_len + ply] = state.zobrist_key

    if ply > 0:
        if state.halfmove_clock >= 100:
            return 0
        if not has_sufficient_material_bb(state.bitboards):
            return 0
        if is_repetition(search_history, root_history_len, ply, state.zobrist_key, state.halfmove_clock):
            return 0

    cdef int alpha_orig = alpha
    cdef bint in_check, improving
    cdef int extended = 0, rbeta = 0
    cdef unsigned long long key
    cdef unsigned int idx
    cdef TTEntry *entry
    cdef int val, tt_move = -1, tt_val, q_flag
    cdef bint has_non_pawn = False
    cdef int p, R
    cdef CGameState null_state
    cdef CGameState child_state
    cdef int counter_move = -1
    cdef int prev_to, prev_piece
    cdef MovePicker mp
    cdef int legal_moves_searched = 0
    cdef int r_val, r_int, margin = 0, futility_limit = 0, lmrDepth = 0
    cdef bint pv_node = (beta - alpha) > 1
    cdef int move, from_sq, to_sq, flag, best_val = -INFINITE, best_move = -1, static_eval
    cdef bint is_cap
    cdef int p_type

    # Check check status
    load_state_to_shell(state, shell)
    in_check = shell.in_check_c()

    # Check Extensions
    if in_check and extensions < 8:
        depth += 1
        extended = 1

    # Transposition Table lookup (O(1) raw pointer, GIL-free)
    key = state.zobrist_key
    idx = key & (TT_SIZE - 1)
    entry = &_tt[idx]

    if entry.key == key:
        if entry.depth >= depth:
            val = entry.val
            if val > MATE_THRESHOLD:
                val -= ply
            elif val < -MATE_THRESHOLD:
                val += ply

            if entry.flag == 0:  # Exact
                return val
            elif entry.flag == 1:  # Lower bound
                if val > alpha:
                    alpha = val
            elif entry.flag == 2:  # Upper bound
                if val < beta:
                    beta = val

            if alpha >= beta:
                return val
        tt_move = entry.move

    static_eval = color * evaluate_state(state)
    if ply < 256:
        static_evals[ply] = static_eval

    improving = False
    if ply >= 2 and not in_check:
        improving = static_eval > static_evals[ply - 2]

    # --- Razoring ---
    if (not pv_node and 
        not in_check and 
        depth <= 3 and 
        static_eval + 531 <= alpha):
        
        load_state_to_shell(state, shell)
        if depth <= 1:
            return quiescence(shell, alpha - 1, alpha, color, ply, 0)
        
        rbeta = alpha - 531
        v = quiescence(shell, rbeta - 1, rbeta, color, ply, 0)
        if v < rbeta:
            return v

    # --- Null Move Pruning (NMP) ---
    has_non_pawn = False
    if (depth >= 3 and 
        not in_check and 
        ply > 0 and 
        not info.stop and
        static_eval >= beta - 32 * depth + 292):
        
        # Check if side to move has non-pawn material (zugzwang safety)
        if color == 1:
            for p in range(1, 5): # P_N=1, P_B=2, P_R=3, P_Q=4
                if state.bitboards[p] != 0:
                    has_non_pawn = True
                    break
        else:
            for p in range(7, 11): # P_n=7, P_b=8, P_r=9, P_q=10
                if state.bitboards[p] != 0:
                    has_non_pawn = True
                    break

        if has_non_pawn:
            # Dynamic reduction R matching Stockfish 11
            R = (854 + 68 * depth) / 258 + ((static_eval - beta) / 192 if (static_eval - beta) / 192 < 3 else 3)
            if R < 1:
                R = 1
            elif R >= depth:
                R = depth - 1
            null_state = state[0]
            make_null_move_on_state(&null_state)
            val = -negamax_copymake(&null_state, depth - 1 - R, -beta, -beta + 1, -color, ply + 1, extensions + extended, -1, shell, search_history, root_history_len)
            
            if info.stop:
                return 0

            if val >= beta:
                # Verification search for deep cuts (anti-zugzwang verification)
                if depth >= 6:
                    val = negamax_copymake(state, depth - 1 - R, beta - 1, beta, color, ply, extensions + extended, prev_move, shell, search_history, root_history_len)
                    if val >= beta:
                        return val
                else:
                    return val

    if depth <= 0:
        # Load state into shell board and run quiescence search (fallback to Make/Unmake)
        load_state_to_shell(state, shell)
        val = quiescence(shell, alpha, beta, color, ply, 0)
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

        entry.key = key
        entry.depth = 0
        entry.val = tt_val
        entry.flag = q_flag
        entry.move = -1
        return val

    # Resolve Countermove Heuristic
    if prev_move != -1:
        prev_to = (prev_move >> 6) & 0x3F
        prev_piece = state.piece_map[prev_to]
        if prev_piece != -1:
            counter_move = counter_moves[prev_piece][prev_to]

    # Initialize MovePicker
    init_move_picker(&mp, tt_move, killer_moves[0][ply], killer_moves[1][ply], counter_move)

    while True:
        # Load current parent state to shell so next_move generates/queries moves correctly
        load_state_to_shell(state, shell)
        move = next_move(&mp, shell, ply)
        if move == -1:
            break

        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        flag = (move >> 12) & 0x0F
        is_cap = flag == 3 or (state.piece_map[to_sq] != -1)

        child_state = state[0]
        make_move_on_state(&child_state, move)
        if not is_state_legal(&child_state, shell, state.side_to_move):
            continue
        
        legal_moves_searched += 1

        # --- Futility Pruning ---
        if (legal_moves_searched > 1 and
            not in_check and
            not is_cap and
            flag < 8 and
            depth < 16):
            
            # Simple Futility Pruning (Child node futility margin)
            if depth < 4:
                margin = 217 * (depth - (1 if improving else 0))
                if static_eval + margin <= alpha:
                    continue
            
            # Parent Node Futility Pruning
            r_val = LMR_REDUCTIONS[depth][legal_moves_searched] if legal_moves_searched < 64 else LMR_REDUCTIONS[depth][63]
            r_int = r_val // 1024
            if pv_node:
                r_int -= 1
            if move == killer_moves[0][ply] or move == killer_moves[1][ply]:
                r_int -= 1
            
            p_type = child_state.piece_map[to_sq]
            if p_type != -1:
                if history_moves[p_type][to_sq] > 2800:
                    r_int -= 1
                elif history_moves[p_type][to_sq] < 600:
                    r_int += 1
            
            if r_int < 0:
                r_int = 0
            
            lmrDepth = depth - 1 - r_int
            if lmrDepth < 0:
                lmrDepth = 0
            
            futility_limit = (5 + depth * depth) * (1 + (1 if improving else 0)) // 2 - 1
            if (static_eval + 235 + 172 * lmrDepth <= alpha and 
                legal_moves_searched >= futility_limit):
                continue

        if legal_moves_searched == 1:
            val = -negamax_copymake(&child_state, depth - 1, -beta, -alpha, -color, ply + 1, extensions + extended, move, shell, search_history, root_history_len)
        else:
            # --- Late Move Reductions (LMR) ---
            if (depth >= 2 and 
                legal_moves_searched > 4 and 
                not is_cap and 
                flag < 8 and 
                not in_check):
                
                # Lookup reduction from precomputed table
                r_val = LMR_REDUCTIONS[depth][legal_moves_searched] if legal_moves_searched < 64 else LMR_REDUCTIONS[depth][63]
                r_int = r_val // 1024

                # Reductions tuning
                if pv_node:
                    r_int -= 1
                if move == killer_moves[0][ply] or move == killer_moves[1][ply]:
                    r_int -= 1
                
                p_type = child_state.piece_map[to_sq] # the piece is already moved to to_sq
                if p_type != -1:
                    if history_moves[p_type][to_sq] > 2800:
                        r_int -= 1
                    elif history_moves[p_type][to_sq] < 600:
                        r_int += 1

                if r_int < 1:
                    r_int = 1
                if r_int >= depth:
                    r_int = depth - 1

                # Search at reduced depth with null window
                val = -negamax_copymake(&child_state, depth - 1 - r_int, -alpha - 1, -alpha, -color, ply + 1, extensions + extended, move, shell, search_history, root_history_len)
                
                # Re-search at full depth with null window if reduced search failed high
                if val > alpha and r_int > 0:
                    val = -negamax_copymake(&child_state, depth - 1, -alpha - 1, -alpha, -color, ply + 1, extensions + extended, move, shell, search_history, root_history_len)
            else:
                # Search at full depth with null window
                val = -negamax_copymake(&child_state, depth - 1, -alpha - 1, -alpha, -color, ply + 1, extensions + extended, move, shell, search_history, root_history_len)
            
            # Re-search with full window if null window search failed high
            if val > alpha and val < beta:
                val = -negamax_copymake(&child_state, depth - 1, -beta, -alpha, -color, ply + 1, extensions + extended, move, shell, search_history, root_history_len)

        if info.stop:
            return 0

        if val > best_val:
            best_val = val
            best_move = move
            if ply == 0:
                info.root_best_move = move

        if val > alpha:
            alpha = val

        if alpha >= beta:
            # Beta cutoff: update Killer, History, and Countermove heuristics for quiet moves
            if not is_cap and flag < 8:
                if killer_moves[0][ply] != move:
                    killer_moves[1][ply] = killer_moves[0][ply]
                    killer_moves[0][ply] = move
                
                p_type = state.piece_map[from_sq]
                if p_type != -1:
                    history_moves[p_type][to_sq] += depth * depth
                    if history_moves[p_type][to_sq] > 5000:
                        history_moves[p_type][to_sq] = 5000

                # Countermove update
                if prev_move != -1:
                    prev_to = (prev_move >> 6) & 0x3F
                    prev_piece = state.piece_map[prev_to]
                    if prev_piece != -1:
                        counter_moves[prev_piece][prev_to] = move
            break

    if info.stop:
        return 0

    if legal_moves_searched == 0:
        if in_check:
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

    if ((entry.key == 0 or entry.key == key or depth >= entry.depth) and
        (entry.key != info.root_key or key == info.root_key)):
        entry.key = key
        entry.depth = depth
        entry.val = tt_val
        entry.flag = tt_flag
        entry.move = best_move

    return best_val
cdef int evaluate_state(CGameState *state) noexcept nogil:
    cdef int phase = state.phase
    if phase > 24:
        phase = 24
    return <int>((state.score_mg * phase + state.score_eg * (24 - phase)) / 24)

cdef void make_move_on_state(CGameState *state, int move) noexcept nogil:
    cdef int from_sq = move & 0x3F
    cdef int to_sq   = (move >> 6) & 0x3F
    cdef int flag    = (move >> 12) & 0x0F
    cdef int side    = state.side_to_move
    cdef int opp_side = 1 if side == 0 else 0

    cdef int mp = state.piece_map[from_sq]
    if mp == -1:
        return

    # Update evaluation: remove moving piece from source square
    remove_piece_eval(mp, from_sq, &state.score_mg, &state.score_eg, &state.phase)

    # XOR out the moving piece from the source square
    state.zobrist_key ^= ZOBRIST_PIECES[mp][from_sq]
    state.bitboards[mp] = state.bitboards[mp] & ~(<unsigned long long>1 << from_sq)
    state.piece_map[from_sq] = -1

    cdef int cap = state.piece_map[to_sq]

    cdef int old_ep = state.en_passant_sq
    cdef int old_castle = state.castling_rights

    state.en_passant_sq = 64
    state.halfmove_clock += 1
    if side == BLACK:
        state.fullmove_number += 1

    if cap != -1:
        # Update evaluation: remove captured piece from destination square
        remove_piece_eval(cap, to_sq, &state.score_mg, &state.score_eg, &state.phase)

        state.zobrist_key ^= ZOBRIST_PIECES[cap][to_sq]
        state.bitboards[cap] = state.bitboards[cap] & ~(<unsigned long long>1 << to_sq)
        state.piece_map[to_sq] = -1
        state.halfmove_clock = 0

    cdef int cap_ep_sq, ep_pawn_idx, prom_offset, p_prom

    if flag == 1:  # FLAG_DOUBLE_PUSH = 1
        state.en_passant_sq = from_sq + 8 if side == WHITE else from_sq - 8
        state.halfmove_clock = 0
    elif flag == 3:  # FLAG_EP = 3
        cap_ep_sq   = to_sq - 8 if side == WHITE else to_sq + 8
        ep_pawn_idx = 6 if side == WHITE else 0 # P_p=6, P_P=0
        # Update evaluation: remove captured EP pawn
        remove_piece_eval(ep_pawn_idx, cap_ep_sq, &state.score_mg, &state.score_eg, &state.phase)

        state.zobrist_key ^= ZOBRIST_PIECES[ep_pawn_idx][cap_ep_sq]
        state.bitboards[ep_pawn_idx] = state.bitboards[ep_pawn_idx] & ~(<unsigned long long>1 << cap_ep_sq)
        state.piece_map[cap_ep_sq] = -1
        state.halfmove_clock = 0
    elif flag == 2:  # FLAG_CASTLE = 2
        if to_sq == 6: # G1 = 6
            # Update evaluation: White castle rook from H1 to F1
            remove_piece_eval(3, 7, &state.score_mg, &state.score_eg, &state.phase)
            add_piece_eval(3, 5, &state.score_mg, &state.score_eg, &state.phase)

            state.zobrist_key ^= ZOBRIST_PIECES[3][7] ^ ZOBRIST_PIECES[3][5] # P_R=3, H1=7, F1=5
            state.bitboards[3] = state.bitboards[3] & ~(<unsigned long long>1 << 7)
            state.bitboards[3] = state.bitboards[3] | (<unsigned long long>1 << 5)
            state.piece_map[7] = -1; state.piece_map[5] = 3
        elif to_sq == 2: # C1 = 2
            # Update evaluation: White castle rook from A1 to D1
            remove_piece_eval(3, 0, &state.score_mg, &state.score_eg, &state.phase)
            add_piece_eval(3, 3, &state.score_mg, &state.score_eg, &state.phase)

            state.zobrist_key ^= ZOBRIST_PIECES[3][0] ^ ZOBRIST_PIECES[3][3] # P_R=3, A1=0, D1=3
            state.bitboards[3] = state.bitboards[3] & ~(<unsigned long long>1 << 0)
            state.bitboards[3] = state.bitboards[3] | (<unsigned long long>1 << 3)
            state.piece_map[0] = -1; state.piece_map[3] = 3
        elif to_sq == 62: # G8 = 62
            # Update evaluation: Black castle rook from H8 to F8
            remove_piece_eval(9, 63, &state.score_mg, &state.score_eg, &state.phase)
            add_piece_eval(9, 61, &state.score_mg, &state.score_eg, &state.phase)

            state.zobrist_key ^= ZOBRIST_PIECES[9][63] ^ ZOBRIST_PIECES[9][61] # P_r=9, H8=63, F8=61
            state.bitboards[9] = state.bitboards[9] & ~(<unsigned long long>1 << 63)
            state.bitboards[9] = state.bitboards[9] | (<unsigned long long>1 << 61)
            state.piece_map[63] = -1; state.piece_map[61] = 9
        elif to_sq == 58: # C8 = 58
            # Update evaluation: Black castle rook from A8 to D8
            remove_piece_eval(9, 56, &state.score_mg, &state.score_eg, &state.phase)
            add_piece_eval(9, 59, &state.score_mg, &state.score_eg, &state.phase)

            state.zobrist_key ^= ZOBRIST_PIECES[9][56] ^ ZOBRIST_PIECES[9][59] # P_r=9, A8=56, D8=59
            state.bitboards[9] = state.bitboards[9] & ~(<unsigned long long>1 << 56)
            state.bitboards[9] = state.bitboards[9] | (<unsigned long long>1 << 59)
            state.piece_map[56] = -1; state.piece_map[59] = 9

    # Place piece at destination (with promotion)
    if flag >= 8:  # FLAG_PROMOTE_N = 8
        prom_offset = 0 if side == WHITE else 6
        if   flag == 11: p_prom = 4 + prom_offset # P_Q = 4
        elif flag == 10: p_prom = 3 + prom_offset # P_R = 3
        elif flag == 9:  p_prom = 2 + prom_offset # P_B = 2
        else:            p_prom = 1 + prom_offset # P_N = 1

        # Update evaluation: add promoted piece to destination square
        add_piece_eval(p_prom, to_sq, &state.score_mg, &state.score_eg, &state.phase)

        state.zobrist_key ^= ZOBRIST_PIECES[p_prom][to_sq]
        state.bitboards[p_prom] = state.bitboards[p_prom] | (<unsigned long long>1 << to_sq)
        state.piece_map[to_sq] = p_prom
        state.halfmove_clock = 0
    else:
        # Update evaluation: add moving piece to destination square
        add_piece_eval(mp, to_sq, &state.score_mg, &state.score_eg, &state.phase)

        state.zobrist_key ^= ZOBRIST_PIECES[mp][to_sq]
        state.bitboards[mp] = state.bitboards[mp] | (<unsigned long long>1 << to_sq)
        state.piece_map[to_sq] = mp
        if mp == 0 or mp == 6: # P_P=0, P_p=6
            state.halfmove_clock = 0

    # Update castling rights
    if   mp == 5:      state.castling_rights &= ~(1 | 2) # P_K=5, WK=1, WQ=2
    elif mp == 11:     state.castling_rights &= ~(4 | 8) # P_k=11, BK=4, BQ=8
    if   from_sq == 7:  state.castling_rights &= ~1
    elif from_sq == 0:  state.castling_rights &= ~2
    elif from_sq == 63: state.castling_rights &= ~4
    elif from_sq == 56: state.castling_rights &= ~8
    if   to_sq == 7:    state.castling_rights &= ~1
    elif to_sq == 0:    state.castling_rights &= ~2
    elif to_sq == 63:   state.castling_rights &= ~4
    elif to_sq == 56:   state.castling_rights &= ~8

    # Incremental occupancy update
    state.occupancies[side] = state.occupancies[side] & ~(<unsigned long long>1 << from_sq)
    if cap != -1:
        state.occupancies[opp_side] = state.occupancies[opp_side] & ~(<unsigned long long>1 << to_sq)
    if flag == 3:  # FLAG_EP = 3
        cap_ep_sq = to_sq - 8 if side == WHITE else to_sq + 8
        state.occupancies[opp_side] = state.occupancies[opp_side] & ~(<unsigned long long>1 << cap_ep_sq)
    elif flag == 2:  # FLAG_CASTLE = 2
        if to_sq == 6: # G1 = 6
            state.occupancies[0] = state.occupancies[0] & ~(<unsigned long long>1 << 7)
            state.occupancies[0] = state.occupancies[0] | (<unsigned long long>1 << 5)
        elif to_sq == 2: # C1 = 2
            state.occupancies[0] = state.occupancies[0] & ~(<unsigned long long>1 << 0)
            state.occupancies[0] = state.occupancies[0] | (<unsigned long long>1 << 3)
        elif to_sq == 62: # G8 = 62
            state.occupancies[1] = state.occupancies[1] & ~(<unsigned long long>1 << 63)
            state.occupancies[1] = state.occupancies[1] | (<unsigned long long>1 << 61)
        elif to_sq == 58: # C8 = 58
            state.occupancies[1] = state.occupancies[1] & ~(<unsigned long long>1 << 56)
            state.occupancies[1] = state.occupancies[1] | (<unsigned long long>1 << 59)

    state.occupancies[side] = state.occupancies[side] | (<unsigned long long>1 << to_sq)
    state.occupancies[2]    = state.occupancies[0] | state.occupancies[1]

    # Toggle side
    state.side_to_move = opp_side

    # XOR castling/EP/side zobrist updates
    if old_castle != state.castling_rights:
        state.zobrist_key ^= ZOBRIST_CASTLING[old_castle] ^ ZOBRIST_CASTLING[state.castling_rights]
    if old_ep != 64:
        state.zobrist_key ^= ZOBRIST_EP[old_ep % 8]
    if state.en_passant_sq != 64:
        state.zobrist_key ^= ZOBRIST_EP[state.en_passant_sq % 8]
    state.zobrist_key ^= ZOBRIST_SIDE

cdef bint is_state_legal(CGameState *state, CustomBitboardBoard shell, int side_that_moved) noexcept nogil:
    cdef int king_piece_idx = 5 if side_that_moved == WHITE else 11
    cdef unsigned long long king_bb = state.bitboards[king_piece_idx]
    cdef int king_sq = cy_lsb(king_bb)
    if king_sq == -1:
        return True
    
    cdef int opponent = BLACK if side_that_moved == WHITE else WHITE
    return not cy_is_square_attacked(state.bitboards, state.occupancies[2], king_sq, opponent)

def clear_tt():
    """Clears the Transposition Table memory."""
    memset(_tt, 0, sizeof(_tt))

# --- Evaluation ---
cdef int evaluate(CustomBitboardBoard board) nogil:
    """Static evaluation of the board with Tapered Evaluation (Middlegame vs Endgame)."""
    cdef int phase = board.phase
    if phase > 24:
        phase = 24
    cdef int bonus_mg = 0
    cdef int bonus_eg = 0
    cy_get_evaluation_bonuses(board, &bonus_mg, &bonus_eg)
    return <int>(((board.score_mg + bonus_mg) * phase + (board.score_eg + bonus_eg) * (24 - phase)) / 24)


# --- MVV-LVA Scoring ---
cdef int get_mvv_lva_score(CustomBitboardBoard board, int move) nogil:
    cdef int from_sq = move & 0x3F
    cdef int to_sq = (move >> 6) & 0x3F
    cdef int flag = (move >> 12) & 0x0F

    cdef int attacker = board.piece_map[from_sq]
    if attacker == -1:
        return 0

    cdef int attacker_val = PIECE_VALUES[attacker % 6]
    cdef int victim_val = 0
    cdef int victim

    if flag == 3:  # FLAG_EP = 3
        victim_val = PIECE_VALUES[0]
    else:
        victim = board.piece_map[to_sq]
        if victim == -1:
            return 0
        victim_val = PIECE_VALUES[victim % 6]

    cdef int score = victim_val * 10 - attacker_val
    if flag >= 8:  # FLAG_PROMOTE_N
        if flag == 11:    # FLAG_PROMOTE_Q
            score += 9000
        elif flag == 10:  # FLAG_PROMOTE_R
            score += 5000
        elif flag == 9:   # FLAG_PROMOTE_B
            score += 3300
        else:
            score += 3200
    return score

# --- Quiescence Search ---
cdef int quiescence(
    CustomBitboardBoard board,
    int alpha,
    int beta,
    int color,
    int ply,
    int qdepth = 0,
) noexcept nogil:
    info.nodes += 1

    cdef bint in_check = board.in_check_c()
    cdef int stand_pat = 0
    if not in_check:
        stand_pat = color * evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

    cdef QMovePicker qmp
    init_qmove_picker(&qmp, in_check, qdepth)

    cdef int move, val
    cdef int legal_moves_searched = 0

    cdef int to_sq, flag, captured_piece, captured_val
    cdef bint is_cap
    while True:
        move = next_qmove(&qmp, board)
        if move == -1:
            break

        if not in_check:
            to_sq = (move >> 6) & 0x3F
            flag = (move >> 12) & 0x0F
            is_cap = flag == 3 or (board.piece_map[to_sq] != -1)
            if is_cap:
                captured_piece = board.piece_map[to_sq]
                captured_val = PIECE_VALUES[captured_piece % 6] if captured_piece != -1 else 100
                if stand_pat + captured_val + 154 <= alpha:
                    continue

        if not board.make_move_c(move):
            board.unmake_move_c()
            continue
        legal_moves_searched += 1
        val = -quiescence(board, -beta, -alpha, -color, ply + 1, qdepth + 1)
        board.unmake_move_c()

        if val >= beta:
            return beta
        if val > alpha:
            alpha = val

    if in_check and legal_moves_searched == 0:
        return -99999 + ply

    return alpha

# --- Negamax Search ---
cdef int negamax(
    CustomBitboardBoard board,
    int depth,
    int alpha,
    int beta,
    int color,
    int ply,
    int extensions,
    int prev_move,
    unsigned long long *search_history,
    int root_history_len,
) noexcept nogil:
    info.nodes += 1

    if info.nodes % 4096 == 0:
        if get_time_ms() - info.start_time > info.time_limit:
            info.stop = True
            return 0

    search_history[root_history_len + ply] = board.zobrist_key

    if ply > 0:
        if board.halfmove_clock >= 100:
            return 0
        if not has_sufficient_material_bb(board._bb):
            return 0
        if is_repetition(search_history, root_history_len, ply, board.zobrist_key, board.halfmove_clock):
            return 0

    cdef int alpha_orig = alpha
    cdef bint in_check = board.in_check_c()
    cdef bint improving = False
    cdef bint pv_node = (beta - alpha) > 1
    cdef int rbeta = 0
    cdef int margin = 0
    cdef int futility_limit = 0
    cdef int lmrDepth = 0
    cdef int static_eval = 0

    # Check Extensions
    cdef int extended = 0
    if in_check and extensions < 8:
        depth += 1
        extended = 1

    # Transposition Table lookup (O(1) raw pointer, GIL-free)
    cdef unsigned long long key = board.zobrist_key
    cdef unsigned int idx = key & (TT_SIZE - 1)
    cdef TTEntry *entry = &_tt[idx]
    cdef int val, tt_move = -1, tt_val, q_flag

    if entry.key == key:
        if entry.depth >= depth:
            val = entry.val
            if val > MATE_THRESHOLD:
                val -= ply
            elif val < -MATE_THRESHOLD:
                val += ply

            if entry.flag == 0:  # Exact
                return val
            elif entry.flag == 1:  # Lower bound
                if val > alpha:
                    alpha = val
            elif entry.flag == 2:  # Upper bound
                if val < beta:
                    beta = val

            if alpha >= beta:
                return val
        tt_move = entry.move

    static_eval = color * evaluate(board)
    if ply < 256:
        static_evals[ply] = static_eval

    improving = False
    if ply >= 2 and not in_check:
        improving = static_eval > static_evals[ply - 2]

    # --- Razoring ---
    if (not pv_node and 
        not in_check and 
        depth <= 3 and 
        static_eval + 531 <= alpha):
        
        if depth <= 1:
            return quiescence(board, alpha - 1, alpha, color, ply, 0)
        
        rbeta = alpha - 531
        v = quiescence(board, rbeta - 1, rbeta, color, ply, 0)
        if v < rbeta:
            return v

    # --- Null Move Pruning (NMP) ---
    cdef bint has_non_pawn = False
    cdef int p, R
    if (depth >= 3 and 
        not in_check and 
        ply > 0 and 
        not info.stop and
        static_eval >= beta - 32 * depth + 292):
        
        # Check if side to move has non-pawn material (zugzwang safety)
        if color == 1:
            for p in range(1, 5): # P_N=1, P_B=2, P_R=3, P_Q=4
                if board._bb[p] != 0:
                    has_non_pawn = True
                    break
        else:
            for p in range(7, 11): # P_n=7, P_b=8, P_r=9, P_q=10
                if board._bb[p] != 0:
                    has_non_pawn = True
                    break

        if has_non_pawn:
            # Dynamic reduction R matching Stockfish 11
            R = (854 + 68 * depth) / 258 + ((static_eval - beta) / 192 if (static_eval - beta) / 192 < 3 else 3)
            if R < 1:
                R = 1
            elif R >= depth:
                R = depth - 1
            if board.make_null_move_c():
                val = -negamax(board, depth - 1 - R, -beta, -beta + 1, -color, ply + 1, extensions + extended, -1, search_history, root_history_len)
                board.unmake_move_c()
                
                if info.stop:
                    return 0

                if val >= beta:
                    # Verification search for deep cuts (anti-zugzwang verification)
                    if depth >= 6:
                        val = negamax(board, depth - 1 - R, beta - 1, beta, color, ply, extensions + extended, prev_move, search_history, root_history_len)
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

        entry.key = key
        entry.depth = 0
        entry.val = tt_val
        entry.flag = q_flag
        entry.move = -1
        return val

    # Resolve Countermove Heuristic
    cdef int counter_move = -1
    cdef int prev_to, prev_piece
    if prev_move != -1:
        prev_to = (prev_move >> 6) & 0x3F
        prev_piece = board.piece_map[prev_to]
        if prev_piece != -1:
            counter_move = counter_moves[prev_piece][prev_to]

    # Initialize MovePicker
    cdef MovePicker mp
    init_move_picker(&mp, tt_move, killer_moves[0][ply], killer_moves[1][ply], counter_move)

    cdef int legal_moves_searched = 0
    cdef int r_val, r_int
    pv_node = (beta - alpha) > 1

    cdef int move, from_sq, to_sq, flag, best_val = -INFINITE, best_move = -1
    cdef bint is_cap
    cdef int p_type

    while True:
        move = next_move(&mp, board, ply)
        if move == -1:
            break

        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        flag = (move >> 12) & 0x0F
        is_cap = flag == 3 or (board.piece_map[to_sq] != -1)

        if not board.make_move_c(move):
            board.unmake_move_c()
            continue
        
        legal_moves_searched += 1

        # --- Futility Pruning ---
        if (legal_moves_searched > 1 and
            not in_check and
            not is_cap and
            flag < 8 and
            depth < 16):
            
            # Simple Futility Pruning (Child node futility margin)
            if depth < 4:
                margin = 217 * (depth - (1 if improving else 0))
                if static_eval + margin <= alpha:
                    board.unmake_move_c()
                    continue
            
            # Parent Node Futility Pruning
            r_val = LMR_REDUCTIONS[depth][legal_moves_searched] if legal_moves_searched < 64 else LMR_REDUCTIONS[depth][63]
            r_int = r_val // 1024
            if pv_node:
                r_int -= 1
            if move == killer_moves[0][ply] or move == killer_moves[1][ply]:
                r_int -= 1
            
            p_type = board.piece_map[to_sq]
            if p_type != -1:
                if history_moves[p_type][to_sq] > 2800:
                    r_int -= 1
                elif history_moves[p_type][to_sq] < 600:
                    r_int += 1
            
            if r_int < 0:
                r_int = 0
            
            lmrDepth = depth - 1 - r_int
            if lmrDepth < 0:
                lmrDepth = 0
            
            futility_limit = (5 + depth * depth) * (1 + (1 if improving else 0)) // 2 - 1
            if (static_eval + 235 + 172 * lmrDepth <= alpha and 
                legal_moves_searched >= futility_limit):
                board.unmake_move_c()
                continue

        if legal_moves_searched == 1:
            val = -negamax(board, depth - 1, -beta, -alpha, -color, ply + 1, extensions + extended, move, search_history, root_history_len)
        else:
            # --- Late Move Reductions (LMR) ---
            if (depth >= 2 and 
                legal_moves_searched > 4 and 
                not is_cap and 
                flag < 8 and 
                not in_check):
                
                # Lookup reduction from precomputed table
                r_val = LMR_REDUCTIONS[depth][legal_moves_searched] if legal_moves_searched < 64 else LMR_REDUCTIONS[depth][63]
                r_int = r_val // 1024

                # Reductions tuning
                if pv_node:
                    r_int -= 1
                if move == killer_moves[0][ply] or move == killer_moves[1][ply]:
                    r_int -= 1
                
                p_type = board.piece_map[to_sq] # the piece is already moved to to_sq
                if p_type != -1:
                    if history_moves[p_type][to_sq] > 2800:
                        r_int -= 1
                    elif history_moves[p_type][to_sq] < 600:
                        r_int += 1

                if r_int < 1:
                    r_int = 1
                if r_int >= depth:
                    r_int = depth - 1

                # Search at reduced depth with null window
                val = -negamax(board, depth - 1 - r_int, -alpha - 1, -alpha, -color, ply + 1, extensions + extended, move, search_history, root_history_len)
                
                # Re-search at full depth with null window if reduced search failed high
                if val > alpha and r_int > 0:
                    val = -negamax(board, depth - 1, -alpha - 1, -alpha, -color, ply + 1, extensions + extended, move, search_history, root_history_len)
            else:
                # Search at full depth with null window
                val = -negamax(board, depth - 1, -alpha - 1, -alpha, -color, ply + 1, extensions + extended, move, search_history, root_history_len)
            
            # Re-search with full window if null window search failed high
            if val > alpha and val < beta:
                val = -negamax(board, depth - 1, -beta, -alpha, -color, ply + 1, extensions + extended, move, search_history, root_history_len)
        
        board.unmake_move_c()

        if info.stop:
            return 0

        if val > best_val:
            best_val = val
            best_move = move
            if ply == 0:
                info.root_best_move = move

        if val > alpha:
            alpha = val

        if alpha >= beta:
            # Beta cutoff: update Killer, History, and Countermove heuristics for quiet moves
            if not is_cap and flag < 8:
                if killer_moves[0][ply] != move:
                    killer_moves[1][ply] = killer_moves[0][ply]
                    killer_moves[0][ply] = move
                
                p_type = board.piece_map[from_sq]
                if p_type != -1:
                    history_moves[p_type][to_sq] += depth * depth
                    if history_moves[p_type][to_sq] > 5000:
                        history_moves[p_type][to_sq] = 5000

                # Countermove update
                if prev_move != -1:
                    prev_to = (prev_move >> 6) & 0x3F
                    prev_piece = board.piece_map[prev_to]
                    if prev_piece != -1:
                        counter_moves[prev_piece][prev_to] = move
            break

    if info.stop:
        return 0

    if legal_moves_searched == 0:
        if board.in_check_c():
            return -99999 + ply
        return 0

    cdef int tt_flag = 0
    if best_val <= alpha_orig:
        tt_flag = 2
    elif best_val >= beta:
        tt_flag = 1

    tt_val = best_val
    if tt_val > MATE_THRESHOLD:
        tt_val += ply
    elif tt_val < -MATE_THRESHOLD:
        tt_val -= ply

    if ((entry.key == 0 or entry.key == key or depth >= entry.depth) and
        (entry.key != info.root_key or key == info.root_key)):
        entry.key = key
        entry.depth = depth
        entry.val = tt_val
        entry.flag = tt_flag
        entry.move = best_move

    return best_val

# --- Main Engine API ---
def get_best_move_cy(
    object chess_board,
    double time_limit = 1.0,
    int depth_limit = 0,
    bint print_info = False,
    int search_mode = 1,
) -> object:
    """Finds the best move using iterative deepening search (Cython compiled)."""
    global info
    info.start_time = get_time_ms()
    info.time_limit = time_limit * 1000.0
    info.stop = False
    info.nodes = 0

    cdef CustomBitboardBoard board = CustomBitboardBoard.from_chess_board(chess_board)
    cdef bint stable = True

    memset(killer_moves, 0, sizeof(killer_moves))
    memset(history_moves, 0, sizeof(history_moves))
    memset(counter_moves, -1, sizeof(counter_moves))
    memset(static_evals, 0, sizeof(static_evals))

    cdef CMoveList legal_moves
    legal_moves.count = 0
    board._generate_legal_moves_c(&legal_moves)
    if legal_moves.count == 0:
        return None
    if legal_moves.count == 1:
        # Instantly return the only legal move (UCI standard / modern chess engine optimization)
        return board.to_chess_move(legal_moves.moves[0])

    info.root_key = board.zobrist_key
    info.root_best_move = legal_moves.moves[0]
    info.root_ponder_move = -1

    cdef int best_move_so_far = legal_moves.moves[0]
    cdef int color = 1 if board.side_to_move == WHITE else -1

    # Initialize and populate game history array for repetition check
    cdef unsigned long long search_history[1024]
    cdef int root_history_len = board._history_len
    cdef int i_hist
    for i_hist in range(root_history_len):
        search_history[i_hist] = board._history[i_hist].zobrist_key

    # Populate root state for copy-make
    cdef CGameState root_state
    memcpy(root_state.bitboards,    board._bb,       12 * sizeof(unsigned long long))
    memcpy(root_state.occupancies,  board._occ,       3 * sizeof(unsigned long long))
    memcpy(root_state.piece_map,    board.piece_map, 64 * sizeof(int))
    root_state.side_to_move    = board.side_to_move
    root_state.castling_rights = board.castling_rights
    root_state.en_passant_sq   = board.en_passant_sq
    root_state.halfmove_clock  = board.halfmove_clock
    root_state.fullmove_number = board.fullmove_number
    root_state.zobrist_key     = board.zobrist_key
    root_state.score_mg        = board.score_mg
    root_state.score_eg        = board.score_eg
    root_state.phase           = board.phase

    cdef int depth = 1
    cdef unsigned long long key = board.zobrist_key
    cdef unsigned int idx
    cdef TTEntry *entry
    cdef int score
    cdef int delta = 15
    cdef int last_score = 0
    cdef int alpha_aw, beta_aw
    
    # Best moves history for stability detection (Dynamic Time Management)
    cdef int best_moves_history[64]
    memset(best_moves_history, 0, sizeof(best_moves_history))

    while not info.stop:
        if depth_limit > 0 and depth > depth_limit:
            break

        info.current_depth = depth
        if depth < 5:
            if search_mode == 1:
                negamax_copymake(&root_state, depth, -INFINITE, INFINITE, color, 0, 0, -1, board, search_history, root_history_len)
            else:
                load_state_to_shell(&root_state, board)
                negamax(board, depth, -INFINITE, INFINITE, color, 0, 0, -1, search_history, root_history_len)
            idx = key & (TT_SIZE - 1)
            entry = &_tt[idx]
            if entry.key == key:
                last_score = entry.val
        else:
            delta = 21 + abs(last_score) // 256
            alpha_aw = last_score - delta
            beta_aw = last_score + delta
            while not info.stop:
                if alpha_aw < -INFINITE: alpha_aw = -INFINITE
                if beta_aw > INFINITE: beta_aw = INFINITE
                
                if search_mode == 1:
                    score = negamax_copymake(&root_state, depth, alpha_aw, beta_aw, color, 0, 0, -1, board, search_history, root_history_len)
                else:
                    load_state_to_shell(&root_state, board)
                    score = negamax(board, depth, alpha_aw, beta_aw, color, 0, 0, -1, search_history, root_history_len)
                
                if info.stop:
                    break
                
                idx = key & (TT_SIZE - 1)
                entry = &_tt[idx]
                if entry.key == key:
                    last_score = entry.val
                else:
                    last_score = score
                
                if last_score <= alpha_aw:
                    beta_aw = alpha_aw
                    delta += delta // 4 + 5
                    alpha_aw = last_score - delta
                    if alpha_aw < -INFINITE: alpha_aw = -INFINITE
                elif last_score >= beta_aw:
                    alpha_aw = beta_aw
                    delta += delta // 4 + 5
                    beta_aw = last_score + delta
                    if beta_aw > INFINITE: beta_aw = INFINITE
                else:
                    break

        if info.stop:
            break

        best_move_so_far = info.root_best_move

        if print_info:
            score = 0
            if entry.key == key:
                score = entry.val

            if abs(score) > MATE_THRESHOLD:
                mate_plies = 99999 - abs(score)
                mate_moves = (mate_plies + 1) // 2
                if score < 0:
                    mate_moves = -mate_moves
                score_str = f"mate {mate_moves}"
            else:
                score_str = f"cp {score}"

            elapsed_ms = int(get_time_ms() - info.start_time)
            elapsed = elapsed_ms / 1000.0
            nps = int(info.nodes / elapsed) if elapsed > 0.0 else 0
            
            # Extract full PV line
            pv_list = extract_pv(root_state, depth, board)
            if len(pv_list) > 1:
                info.root_ponder_move = pv_list[1]
            else:
                info.root_ponder_move = -1
            
            pv_str_list = []
            for m in pv_list:
                pv_str_list.append(board.to_chess_move(m).uci())
            pv_line = " ".join(pv_str_list)
            if not pv_line and best_move_so_far != -1:
                pv_line = board.to_chess_move(best_move_so_far).uci()
                
            print(
                f"info depth {depth} score {score_str} nodes {info.nodes} "
                f"nps {nps} time {elapsed_ms} pv {pv_line}",
                flush=True,
            )

        # Dynamic Time Management: stable best move check
        if depth < 64:
            best_moves_history[depth] = best_move_so_far
        stable = True
        if depth >= 6:
            for i_hist in range(depth - 3, depth + 1):
                if best_moves_history[i_hist] != best_move_so_far:
                    stable = False
                    break
            if stable and (get_time_ms() - info.start_time) > (time_limit * 1000.0 * 0.5):
                break

        if (get_time_ms() - info.start_time) > (time_limit * 1000.0 * 0.95):
            break

        depth += 1

    return board.to_chess_move(best_move_so_far)

def get_nodes():
    return info.nodes

def set_nodes(long long val):
    global info
    info.nodes = val

def get_depth():
    return info.current_depth

def set_depth(int val):
    global info
    info.current_depth = val

def get_stop():
    return info.stop

def set_stop(bint val):
    global info
    info.stop = val

def set_ponder_move(int val):
    global info
    info.root_ponder_move = val

def get_start_time():
    return info.start_time

def set_start_time(double val):
    global info
    info.start_time = val

def get_time_limit():
    return info.time_limit

def set_time_limit(double val):
    global info
    info.time_limit = val

def generate_legal_moves_copymake(CustomBitboardBoard board):
    """Generates legal moves using Copy-Make style validation."""
    cdef CGameState root_state
    memcpy(root_state.bitboards,    board._bb,       12 * sizeof(unsigned long long))
    memcpy(root_state.occupancies,  board._occ,       3 * sizeof(unsigned long long))
    memcpy(root_state.piece_map,    board.piece_map, 64 * sizeof(int))
    root_state.side_to_move    = board.side_to_move
    root_state.castling_rights = board.castling_rights
    root_state.en_passant_sq   = board.en_passant_sq
    root_state.halfmove_clock  = board.halfmove_clock
    root_state.fullmove_number = board.fullmove_number
    root_state.zobrist_key     = board.zobrist_key
    root_state.score_mg        = board.score_mg
    root_state.score_eg        = board.score_eg
    root_state.phase           = board.phase

    cdef CMoveList pseudo
    pseudo.count = 0
    board._generate_pseudo_legal_moves_c(&pseudo)

    cdef list legal_moves = []
    cdef int i, move
    cdef CGameState child_state

    for i in range(pseudo.count):
        move = pseudo.moves[i]
        child_state = root_state
        make_move_on_state(&child_state, move)
        if is_state_legal(&child_state, board, root_state.side_to_move):
            legal_moves.append(move)

    # Restore board to original state before returning
    load_state_to_shell(&root_state, board)
    return legal_moves

cdef long long run_perft_copymake_recursive(CGameState *state, int depth, CustomBitboardBoard shell) noexcept nogil:
    if depth == 0:
        return 1

    cdef CMoveList pseudo
    pseudo.count = 0
    load_state_to_shell(state, shell)
    shell._generate_pseudo_legal_moves_c(&pseudo)

    cdef int i, move
    cdef long long nodes = 0
    cdef CGameState child_state

    if depth == 1:
        for i in range(pseudo.count):
            move = pseudo.moves[i]
            child_state = state[0]
            make_move_on_state(&child_state, move)
            if is_state_legal(&child_state, shell, state.side_to_move):
                nodes += 1
        return nodes

    for i in range(pseudo.count):
        move = pseudo.moves[i]
        child_state = state[0]
        make_move_on_state(&child_state, move)
        if is_state_legal(&child_state, shell, state.side_to_move):
            nodes += run_perft_copymake_recursive(&child_state, depth - 1, shell)

    return nodes

def run_perft_copymake(CustomBitboardBoard board, int depth):
    cdef CGameState root_state
    memcpy(root_state.bitboards,    board._bb,       12 * sizeof(unsigned long long))
    memcpy(root_state.occupancies,  board._occ,       3 * sizeof(unsigned long long))
    memcpy(root_state.piece_map,    board.piece_map, 64 * sizeof(int))
    root_state.side_to_move    = board.side_to_move
    root_state.castling_rights = board.castling_rights
    root_state.en_passant_sq   = board.en_passant_sq
    root_state.halfmove_clock  = board.halfmove_clock
    root_state.fullmove_number = board.fullmove_number
    root_state.zobrist_key     = board.zobrist_key
    root_state.score_mg        = board.score_mg
    root_state.score_eg        = board.score_eg
    root_state.phase           = board.phase

    cdef long long total_nodes = run_perft_copymake_recursive(&root_state, depth, board)

    # Restore board
    load_state_to_shell(&root_state, board)
    return total_nodes

def run_perft_copymake_divide(CustomBitboardBoard board, int depth):
    cdef CGameState root_state
    memcpy(root_state.bitboards,    board._bb,       12 * sizeof(unsigned long long))
    memcpy(root_state.occupancies,  board._occ,       3 * sizeof(unsigned long long))
    memcpy(root_state.piece_map,    board.piece_map, 64 * sizeof(int))
    root_state.side_to_move    = board.side_to_move
    root_state.castling_rights = board.castling_rights
    root_state.en_passant_sq   = board.en_passant_sq
    root_state.halfmove_clock  = board.halfmove_clock
    root_state.fullmove_number = board.fullmove_number
    root_state.zobrist_key     = board.zobrist_key
    root_state.score_mg        = board.score_mg
    root_state.score_eg        = board.score_eg
    root_state.phase           = board.phase

    cdef CMoveList pseudo
    pseudo.count = 0
    board._generate_pseudo_legal_moves_c(&pseudo)

    cdef dict result = {}
    cdef int i, move
    cdef CGameState child_state
    cdef long long nodes

    for i in range(pseudo.count):
        move = pseudo.moves[i]
        child_state = root_state
        make_move_on_state(&child_state, move)
        if is_state_legal(&child_state, board, root_state.side_to_move):
            if depth > 1:
                nodes = run_perft_copymake_recursive(&child_state, depth - 1, board)
            else:
                nodes = 1
            result[move] = nodes

    # Restore board
    load_state_to_shell(&root_state, board)
    return result

def get_ponder_move():
    return info.root_ponder_move

def get_ponder_move_uci(object chess_board):
    cdef CustomBitboardBoard board = CustomBitboardBoard.from_chess_board(chess_board)
    cdef int m = get_ponder_move()
    if m == -1:
        return None
    return board.to_chess_move(m).uci()

