# cython: language_level=3
# type: ignore
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
"""Search Engine and Evaluator for qwen-chess-v3 in Cython."""

import time
import random
import chess
from libc.string cimport memset
from libc.math cimport log
from board_cy cimport CustomBitboardBoard, CMoveList

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

# PST Tables
cdef int PAWN_TABLE[64]
PAWN_TABLE[:] = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0
]

cdef int KNIGHT_TABLE[64]
KNIGHT_TABLE[:] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

cdef int BISHOP_TABLE[64]
BISHOP_TABLE[:] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

cdef int ROOK_TABLE[64]
ROOK_TABLE[:] = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

cdef int QUEEN_TABLE[64]
QUEEN_TABLE[:] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

cdef int KING_TABLE[64]
KING_TABLE[:] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

cdef const int* PST[6]
PST[0] = PAWN_TABLE
PST[1] = KNIGHT_TABLE
PST[2] = BISHOP_TABLE
PST[3] = ROOK_TABLE
PST[4] = QUEEN_TABLE
PST[5] = KING_TABLE

# Precomputed static evaluation tables
cdef int white_piece_values[6][64]
cdef int black_piece_values[6][64]

cdef void init_piece_values() noexcept:
    cdef int p_type, sq, rank, file, w_idx, b_idx
    for p_type in range(6):
        for sq in range(64):
            rank = sq // 8
            file = sq % 8
            # White perspective
            w_idx = (7 - rank) * 8 + file
            white_piece_values[p_type][sq] = PIECE_VALUES[p_type] + PST[p_type][w_idx]
            # Black perspective
            b_idx = rank * 8 + file
            black_piece_values[p_type][sq] = PIECE_VALUES[p_type] + PST[p_type][b_idx]

init_piece_values()

# --- Portable LSB (BitScanForward) inline ---
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
    #else
    static __inline__ int _cy_lsb_impl2(unsigned long long bb) {
        if (!bb) return -1;
        return __builtin_ctzll(bb);
    }
    #endif
    """
    int _cy_lsb_impl2(unsigned long long bb) nogil

cdef inline int cy_lsb(unsigned long long bb) nogil:
    return _cy_lsb_impl2(bb)

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

# Killer Moves & History Heuristics
cdef int killer_moves[2][64]
cdef int history_moves[12][64]

# LMR Reductions
cdef int LMR_REDUCTIONS[64][64]

cdef void init_lmr_reductions() noexcept:
    cdef int depth, move_count
    for depth in range(64):
        for move_count in range(64):
            if depth == 0 or move_count == 0:
                LMR_REDUCTIONS[depth][move_count] = 0
            else:
                LMR_REDUCTIONS[depth][move_count] = <int>(0.5 + log(depth) * log(move_count) / 1.95 * 1024)

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

def clear_tt():
    """Clears the Transposition Table memory."""
    memset(_tt, 0, sizeof(_tt))

# --- Evaluation ---
cdef int evaluate(CustomBitboardBoard board) nogil:
    """Static evaluation of the board from White's perspective."""
    cdef int score = 0
    cdef int p_idx, sq_idx
    cdef unsigned long long bb
    
    # White pieces (0-5)
    for p_idx in range(6):
        bb = board._bb[p_idx]
        while bb:
            sq_idx = _cy_lsb_impl2(bb)
            bb = bb & ~(<unsigned long long>1 << sq_idx)
            score += white_piece_values[p_idx][sq_idx]

    # Black pieces (6-11)
    for p_idx in range(6):
        bb = board._bb[p_idx + 6]
        while bb:
            sq_idx = _cy_lsb_impl2(bb)
            bb = bb & ~(<unsigned long long>1 << sq_idx)
            score -= black_piece_values[p_idx][sq_idx]

    return score

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
) except *:
    info.nodes += 1

    cdef bint in_check = board.in_check()
    cdef int stand_pat = 0
    if not in_check:
        stand_pat = color * evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

    cdef CMoveList pseudo_moves
    pseudo_moves.count = 0
    board._generate_pseudo_legal_moves_c(&pseudo_moves)

    cdef CScoredMoveList moves_to_search
    moves_to_search.count = 0

    cdef int i, move, to_sq, flag, score, val, qdepth
    cdef bint is_cap, gives_check
    cdef int legal_moves_searched = 0

    if in_check:
        for i in range(pseudo_moves.count):
            move = pseudo_moves.moves[i]
            score = get_mvv_lva_score(board, move)
            to_sq = (move >> 6) & 0x3F
            flag = (move >> 12) & 0x0F
            is_cap = flag == 3 or (board.piece_map[to_sq] != -1)
            if is_cap:
                score += 10000
            elif flag >= 8:
                score += 9000
            moves_to_search.moves[moves_to_search.count].move = move
            moves_to_search.moves[moves_to_search.count].score = score
            moves_to_search.count += 1
    else:
        qdepth = ply - info.current_depth
        for i in range(pseudo_moves.count):
            move = pseudo_moves.moves[i]
            to_sq = (move >> 6) & 0x3F
            flag = (move >> 12) & 0x0F
            is_cap = flag == 3 or (board.piece_map[to_sq] != -1)

            if is_cap:
                score = get_mvv_lva_score(board, move)
                moves_to_search.moves[moves_to_search.count].move = move
                moves_to_search.moves[moves_to_search.count].score = score
                moves_to_search.count += 1
            elif qdepth < 1:
                if not board.make_move_c(move):
                    board.unmake_move_c()
                    continue
                gives_check = board.in_check()
                board.unmake_move_c()
                if gives_check:
                    moves_to_search.moves[moves_to_search.count].move = move
                    moves_to_search.moves[moves_to_search.count].score = 0
                    moves_to_search.count += 1

    sort_moves(&moves_to_search)

    for i in range(moves_to_search.count):
        move = moves_to_search.moves[i].move
        if not board.make_move_c(move):
            board.unmake_move_c()
            continue
        legal_moves_searched += 1
        val = -quiescence(board, -beta, -alpha, -color, ply + 1)
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
) except *:
    info.nodes += 1

    if info.nodes % 4096 == 0:
        with nogil:
            pass
        if time.time() - info.start_time > info.time_limit:
            info.stop = True
            return 0

    cdef int alpha_orig = alpha
    cdef bint in_check = board.in_check()

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
            if entry.flag == 1:  # Lower bound
                if val >= beta:
                    return val
                if val > alpha:
                    alpha = val
            if entry.flag == 2:  # Upper bound
                if val <= alpha:
                    return val
                if val < beta:
                    beta = val
            if alpha >= beta:
                return val
        tt_move = entry.move

    # --- Null Move Pruning (NMP) ---
    cdef bint has_non_pawn = False
    cdef int p, R
    if (depth >= 3 and 
        not in_check and 
        ply > 0 and 
        not info.stop):
        
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
            R = 3 + depth // 4
            if board.make_null_move():
                val = -negamax(board, depth - 1 - R, -beta, -beta + 1, -color, ply + 1)
                board.unmake_move_c()
                
                if info.stop:
                    return 0

                if val >= beta:
                    # Verification search for deep cuts (anti-zugzwang verification)
                    if depth >= 6:
                        val = negamax(board, depth - 1 - R, beta - 1, beta, color, ply)
                        if val >= beta:
                            return val
                    else:
                        return val

    if depth <= 0:
        val = quiescence(board, alpha, beta, color, ply)
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

    cdef CMoveList pseudo_moves
    pseudo_moves.count = 0
    board._generate_pseudo_legal_moves_c(&pseudo_moves)

    # Move Ordering
    cdef CScoredMoveList moves_scored
    moves_scored.count = 0
    
    cdef int i, move, from_sq, to_sq, flag, score, best_val = -INFINITE, best_move = -1
    cdef bint is_cap
    cdef int p_type

    for i in range(pseudo_moves.count):
        move = pseudo_moves.moves[i]
        if move == tt_move:
            score = 1000000
        else:
            from_sq = move & 0x3F
            to_sq = (move >> 6) & 0x3F
            flag = (move >> 12) & 0x0F
            is_cap = flag == 3 or (board.piece_map[to_sq] != -1)
            
            if is_cap:
                score = 100000 + get_mvv_lva_score(board, move)
            elif flag >= 8:
                score = 90000 + PIECE_VALUES[0]
            else:
                if move == killer_moves[0][ply]:
                    score = 9000
                elif move == killer_moves[1][ply]:
                    score = 8000
                else:
                    p_type = board.piece_map[from_sq]
                    if p_type != -1:
                        score = history_moves[p_type][to_sq]
                    else:
                        score = 0
        moves_scored.moves[moves_scored.count].move = move
        moves_scored.moves[moves_scored.count].score = score
        moves_scored.count += 1

    sort_moves(&moves_scored)

    cdef int legal_moves_searched = 0
    cdef int r_val, r_int
    cdef bint pv_node = (beta - alpha) > 1

    for i in range(moves_scored.count):
        move = moves_scored.moves[i].move
        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        flag = (move >> 12) & 0x0F
        is_cap = flag == 3 or (board.piece_map[to_sq] != -1)

        if not board.make_move_c(move):
            board.unmake_move_c()
            continue
        
        legal_moves_searched += 1

        if legal_moves_searched == 1:
            val = -negamax(board, depth - 1, -beta, -alpha, -color, ply + 1)
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
                    if history_moves[p_type][to_sq] > 2000:
                        r_int -= 1
                    elif history_moves[p_type][to_sq] < 500:
                        r_int += 1

                if r_int < 1:
                    r_int = 1
                if r_int >= depth:
                    r_int = depth - 1

                # Search at reduced depth with null window
                val = -negamax(board, depth - 1 - r_int, -alpha - 1, -alpha, -color, ply + 1)
                
                # Re-search at full depth with null window if reduced search failed high
                if val > alpha and r_int > 0:
                    val = -negamax(board, depth - 1, -alpha - 1, -alpha, -color, ply + 1)
            else:
                # Search at full depth with null window
                val = -negamax(board, depth - 1, -alpha - 1, -alpha, -color, ply + 1)
            
            # Re-search with full window if null window search failed high
            if val > alpha and val < beta:
                val = -negamax(board, depth - 1, -beta, -alpha, -color, ply + 1)
        
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
            # Beta cutoff: update Killer & History heuristics for quiet moves
            if not is_cap and flag < 8:
                if killer_moves[0][ply] != move:
                    killer_moves[1][ply] = killer_moves[0][ply]
                    killer_moves[0][ply] = move
                
                p_type = board.piece_map[from_sq]
                if p_type != -1:
                    history_moves[p_type][to_sq] += depth * depth
                    if history_moves[p_type][to_sq] > 5000:
                        history_moves[p_type][to_sq] = 5000
            break

    if info.stop:
        return 0

    if legal_moves_searched == 0:
        if board.in_check():
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

    if key == info.root_key or entry.key != info.root_key:
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
) -> object:
    """Finds the best move using iterative deepening search (Cython compiled)."""
    global info
    info.start_time = time.time()
    info.time_limit = time_limit
    info.stop = False
    info.nodes = 0

    cdef CustomBitboardBoard board = CustomBitboardBoard.from_chess_board(chess_board)

    memset(killer_moves, 0, sizeof(killer_moves))
    memset(history_moves, 0, sizeof(history_moves))

    cdef CMoveList legal_moves
    legal_moves.count = 0
    board._generate_legal_moves_c(&legal_moves)
    if legal_moves.count == 0:
        return None

    info.root_key = board.zobrist_key
    info.root_best_move = legal_moves.moves[0]

    cdef int best_move_so_far = legal_moves.moves[0]
    cdef int color = 1 if board.side_to_move == WHITE else -1

    cdef int depth = 1
    cdef unsigned long long key
    cdef unsigned int idx
    cdef TTEntry *entry
    cdef int score
    cdef int delta = 15
    cdef int last_score = 0
    cdef int alpha_aw, beta_aw

    while not info.stop:
        if depth_limit > 0 and depth > depth_limit:
            break

        info.current_depth = depth
        if depth < 5:
            negamax(board, depth, -INFINITE, INFINITE, color, 0)
            key = board.zobrist_key
            idx = key & (TT_SIZE - 1)
            entry = &_tt[idx]
            if entry.key == key:
                last_score = entry.val
        else:
            delta = 15
            alpha_aw = last_score - delta
            beta_aw = last_score + delta
            while not info.stop:
                if alpha_aw < -INFINITE: alpha_aw = -INFINITE
                if beta_aw > INFINITE: beta_aw = INFINITE
                
                score = negamax(board, depth, alpha_aw, beta_aw, color, 0)
                
                if info.stop:
                    break
                
                key = board.zobrist_key
                idx = key & (TT_SIZE - 1)
                entry = &_tt[idx]
                if entry.key == key:
                    last_score = entry.val
                else:
                    last_score = score
                
                if last_score <= alpha_aw:
                    beta_aw = alpha_aw
                    delta += delta // 3 + 5
                    alpha_aw = last_score - delta
                elif last_score >= beta_aw:
                    alpha_aw = beta_aw
                    delta += delta // 3 + 5
                    beta_aw = last_score + delta
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

            elapsed = time.time() - info.start_time
            elapsed_ms = int(elapsed * 1000)
            nps = int(info.nodes / elapsed) if elapsed > 0.0 else 0
            best_move_obj = board.to_chess_move(best_move_so_far)
            best_move_uci = best_move_obj.uci()
            print(
                f"info depth {depth} score {score_str} nodes {info.nodes} "
                f"nps {nps} time {elapsed_ms} pv {best_move_uci}",
                flush=True,
            )

        if time.time() - info.start_time > time_limit * 0.95:
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
