# board_cy.pxd
# Cython declarations for CustomBitboardBoard to allow direct C-level access from engine_cy.

cdef struct CGameState:
    unsigned long long bitboards[12]
    unsigned long long occupancies[3]
    unsigned long long zobrist_key
    int score_mg
    int score_eg
    int phase
    unsigned short fullmove_number
    unsigned char side_to_move
    unsigned char castling_rights
    unsigned char en_passant_sq
    unsigned char halfmove_clock
    signed char piece_map[64]

cdef struct CMoveList:
    int moves[256]
    int count

cdef class CustomBitboardBoard:
    cdef public int side_to_move
    cdef public int castling_rights
    cdef public int en_passant_sq
    cdef public int halfmove_clock
    cdef public int fullmove_number
    cdef public unsigned long long zobrist_key
    cdef signed char piece_map[64]
    cdef CGameState _history[512]
    cdef int _history_len
    cdef unsigned long long _bb[12]
    cdef unsigned long long _occ[3]
    cdef public int score_mg
    cdef public int score_eg
    cdef public int phase

    cdef unsigned long long _recompute_zobrist(self) noexcept nogil
    cdef void _generate_pseudo_legal_moves_c(self, CMoveList *move_list) noexcept nogil
    cdef void _generate_captures_c(self, CMoveList *move_list) noexcept nogil
    cdef void _generate_quiets_c(self, CMoveList *move_list) noexcept nogil
    cdef void _generate_legal_moves_c(self, CMoveList *move_list) noexcept nogil
    cdef bint make_move_c(self, int move) noexcept nogil
    cdef void unmake_move_c(self) noexcept nogil
    cdef bint is_square_attacked_c(self, int sq, int attacker_color) noexcept nogil
    cdef bint in_check_c(self) noexcept nogil
    cdef bint make_null_move_c(self) noexcept nogil
    cdef long long _run_perft_recursive_c(self, int depth) noexcept nogil

    cpdef object get_piece_at(self, int square)
    cpdef bint is_square_attacked(self, int sq, int attacker_color)
    cpdef bint in_check(self)
    cpdef list generate_pseudo_legal_moves(self)
    cpdef list generate_legal_moves(self)
    cpdef bint make_move(self, int move)
    cpdef void unmake_move(self)
    cpdef bint make_null_move(self)
    cpdef long long run_perft_recursive(self, int depth)

cdef public unsigned long long ZOBRIST_PIECES[12][64]
cdef public unsigned long long ZOBRIST_CASTLING[16]
cdef public unsigned long long ZOBRIST_EP[8]
cdef public unsigned long long ZOBRIST_SIDE

cdef void cy_evaluate_pawns(CustomBitboardBoard board, int *mg_score, int *eg_score) noexcept nogil
cdef void cy_evaluate_king_safety(CustomBitboardBoard board, int *mg_score, int *eg_score) noexcept nogil
cdef void cy_get_evaluation_bonuses(CustomBitboardBoard board, int *mg_score, int *eg_score) noexcept nogil

cdef bint cy_is_square_attacked(unsigned long long *bb, unsigned long long occupancy, int sq, int attacker_color) noexcept nogil

