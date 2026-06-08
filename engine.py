"""Search Engine and Evaluator for qwen-chess-v3."""

import random
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
    )

# --- Configuration ---
PIECE_VALUES = [100, 320, 330, 500, 900, 0]
INFINITE = 10000000
MATE_THRESHOLD = 90000

PAWN_TABLE_MG = (
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0
)

PAWN_TABLE_EG = (
     0,   0,   0,   0,   0,   0,   0,   0,
    50,  50,  50,  50,  50,  50,  50,  50,
    30,  30,  30,  30,  30,  30,  30,  30,
    20,  20,  20,  20,  20,  20,  20,  20,
    10,  10,  10,  10,  10,  10,  10,  10,
     5,   5,   5,   5,   5,   5,   5,   5,
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0
)

KNIGHT_TABLE = (
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
)

BISHOP_TABLE = (
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
)

ROOK_TABLE = (
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
)

QUEEN_TABLE = (
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
)

KING_TABLE_MG = (
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
)

KING_TABLE_EG = (
    -50,-30,-30,-30,-30,-30,-30,-50,
    -30,-10,-10,-10,-10,-10,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10,-10,-10,-10,-10,-10,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
)

PST_MG = {
    0: PAWN_TABLE_MG,
    1: KNIGHT_TABLE,
    2: BISHOP_TABLE,
    3: ROOK_TABLE,
    4: QUEEN_TABLE,
    5: KING_TABLE_MG
}

PST_EG = {
    0: PAWN_TABLE_EG,
    1: KNIGHT_TABLE,
    2: BISHOP_TABLE,
    3: ROOK_TABLE,
    4: QUEEN_TABLE,
    5: KING_TABLE_EG
}

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
            PIECE_VALUES[p_type] + PST_MG[p_type][w_idx]
        )
        white_piece_values_eg[p_type][sq] = (
            PIECE_VALUES[p_type] + PST_EG[p_type][w_idx]
        )

        # Black perspective (A1 is bottom-left, index 56 in PST)
        b_idx = rank * 8 + file
        black_piece_values_mg[p_type][sq] = (
            PIECE_VALUES[p_type] + PST_MG[p_type][b_idx]
        )
        black_piece_values_eg[p_type][sq] = (
            PIECE_VALUES[p_type] + PST_EG[p_type][b_idx]
        )

# Remove temporary module-level variables to avoid redefining outer scope warnings
# del p_type, sq, rank, file, w_idx, b_idx


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
LMR_REDUCTIONS = [[0] * 64 for _ in range(64)]

def init_lmr_reductions():
    import math
    for depth in range(64):
        for move_count in range(64):
            if depth == 0 or move_count == 0:
                LMR_REDUCTIONS[depth][move_count] = 0
            else:
                LMR_REDUCTIONS[depth][move_count] = int(0.5 + math.log(depth) * math.log(move_count) / 1.95 * 1024)

init_lmr_reductions()


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

    # White pieces (0-5)
    for p_idx in range(6):
        bb = board.bitboards[p_idx]
        while bb:
            sq_idx = lsb(bb)
            bb = clear_bit(bb, sq_idx)
            score_mg += white_piece_values_mg[p_idx][sq_idx]
            score_eg += white_piece_values_eg[p_idx][sq_idx]

    # Black pieces (6-11)
    for p_idx in range(6):
        bb = board.bitboards[p_idx + 6]
        while bb:
            sq_idx = lsb(bb)
            bb = clear_bit(bb, sq_idx)
            score_mg -= black_piece_values_mg[p_idx][sq_idx]
            score_eg -= black_piece_values_eg[p_idx][sq_idx]

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


# --- Negamax Search ---
def negamax(
    board: CustomBitboardBoard,
    depth: int,
    alpha: int,
    beta: int,
    color: int,
    ply: int,
    extensions: int = 0,
) -> int:
    """Recursively searches the game tree using Alpha-Beta Negamax algorithm."""
    info.nodes += 1

    if info.nodes % 4096 == 0:
        if time.time() - info.start_time > info.time_limit:
            info.stop = True
            return 0

    alpha_orig = alpha
    in_check = board.in_check()

    # Check Extensions
    extended = 0
    if in_check and extensions < 8:
        depth += 1
        extended = 1

    # Transposition Table lookup
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

    # --- Null Move Pruning (NMP) ---
    has_non_pawn = False
    if (depth >= 3 and 
        not in_check and 
        ply > 0 and 
        not info.stop):
        
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
            R = 3 + depth // 4
            if board.make_null_move():
                val = -negamax(board, depth - 1 - R, -beta, -beta + 1, -color, ply + 1, extensions + extended)
                board.unmake_move()
                
                if info.stop:
                    return 0

                if val >= beta:
                    # Verification search for deep cuts (anti-zugzwang verification)
                    if depth >= 6:
                        val = negamax(board, depth - 1 - R, beta - 1, beta, color, ply, extensions + extended)
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
            val = -negamax(board, depth - 1, -beta, -alpha, -color, ply + 1, extensions + extended)
        else:
            # --- Late Move Reductions (LMR) ---
            if (depth >= 2 and 
                legal_moves_searched > 1 and 
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
                    if history_moves[p_type][to_sq] > 2000:
                        r_int -= 1
                    elif history_moves[p_type][to_sq] < 500:
                        r_int += 1

                if r_int < 1:
                    r_int = 1
                if r_int >= depth:
                    r_int = depth - 1

                # Search at reduced depth with null window
                val = -negamax(board, depth - 1 - r_int, -alpha - 1, -alpha, -color, ply + 1, extensions + extended)
                
                # Re-search at full depth with null window if reduced search failed high
                if val > alpha and r_int > 0:
                    val = -negamax(board, depth - 1, -alpha - 1, -alpha, -color, ply + 1, extensions + extended)
            else:
                # Search at full depth with null window
                val = -negamax(board, depth - 1, -alpha - 1, -alpha, -color, ply + 1, extensions + extended)
            
            # Re-search with full window if null window search failed high
            if val > alpha and val < beta:
                val = -negamax(board, depth - 1, -beta, -alpha, -color, ply + 1, extensions + extended)
        
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


# --- Opening Book Setup ---
def build_opening_book() -> dict[str, list[chess.Move]]:
    """Builds a simple opening book database from standard opening lines."""
    book = {}
    lines = [
        ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"],
        ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"],
        ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O"],
        ["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Bf5", "Ng3", "Bg6"],
        ["e4", "e6", "d4", "d5", "Nc3", "Nf6", "Bg5", "Be7", "e5", "Nfd7"],
        ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3", "d6"],
        ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qa5", "d4", "Nf6"],
        ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "Nf3", "O-O"],
        ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "e6", "e3", "Nbd7"],
        ["c4", "e5", "Nc3", "Nf6", "g3", "d5", "cxd5", "Nxd5", "Bg2"]
    ]
    for line in lines:
        temp_board = chess.Board()
        for move_str in line:
            try:
                move = temp_board.parse_san(move_str)
                key = " ".join(temp_board.fen().split()[:4])
                if key not in book:
                    book[key] = []
                if move not in book[key]:
                    book[key].append(move)
                temp_board.push(move)
            except ValueError:
                continue
    return book


OPENING_BOOK = build_opening_book()


# --- Main Engine API ---
def get_best_move(
    chess_board: chess.Board,
    time_limit: float = 1.0,
    depth_limit: int = 0,
    print_info: bool = False,
) -> chess.Move | None:
    """Finds the best move using iterative deepening search."""
    # 1. Opening Book Check
    key = " ".join(chess_board.fen().split()[:4])
    if key in OPENING_BOOK:
        moves = OPENING_BOOK[key]
        if moves:
            return random.choice(moves)

    # 2. Setup Search state
    info.start_time = time.time()
    info.time_limit = time_limit
    info.stop = False
    info.nodes = 0

    # 3. Convert python-chess Board to CustomBitboardBoard
    board = CustomBitboardBoard.from_chess_board(chess_board)

    global killer_moves, history_moves
    killer_moves = [[0] * 64 for _ in range(2)]
    history_moves = [[0] * 64 for _ in range(12)]

    legal_moves = board.generate_legal_moves()
    if not legal_moves:
        return None

    best_move_so_far = legal_moves[0]
    color = 1 if board.side_to_move == WHITE else -1

    depth = 1
    while not info.stop:
        if depth_limit > 0 and depth > depth_limit:
            break
        info.current_depth = depth
        negamax(board, depth, -INFINITE, INFINITE, color, 0)

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
            print(
                f"info depth {depth} score {score_str} nodes {info.nodes} "
                f"nps {nps} time {elapsed_ms} pv {best_move_uci}",
                flush=True,
            )

        if time.time() - info.start_time > time_limit * 0.95:
            break

        depth += 1

    return board.to_chess_move(best_move_so_far)

# --- Cython Engine Wrapper & Fallback ---
try:
    import engine_cy

    _get_best_move_cy = engine_cy.get_best_move_cy

    def get_best_move(
        chess_board: chess.Board,
        time_limit: float = 1.0,
        depth_limit: int = 0,
        print_info: bool = False,
        search_mode: int = 1,
    ) -> chess.Move | None:
        info.root_ponder_move = -1
        key = " ".join(chess_board.fen().split()[:4])
        if key in OPENING_BOOK:
            moves = OPENING_BOOK[key]
            if moves:
                return random.choice(moves)
        return _get_best_move_cy(chess_board, time_limit, depth_limit, print_info, search_mode)

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
