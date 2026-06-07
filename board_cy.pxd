# board_cy.pxd
# Cython declarations for CustomBitboardBoard to allow direct C-level access from engine_cy.

cdef struct CGameState:
    unsigned long long bitboards[12]
    unsigned long long occupancies[3]
    int side_to_move
    int castling_rights
    int en_passant_sq
    int halfmove_clock
    int fullmove_number
    int piece_map[64]
    unsigned long long zobrist_key

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
    cdef int piece_map[64]
    cdef CGameState _history[512]
    cdef int _history_len
    cdef unsigned long long _bb[12]
    cdef unsigned long long _occ[3]

    cdef unsigned long long _recompute_zobrist(self)
    cdef void _generate_pseudo_legal_moves_c(self, CMoveList *move_list)
    cdef void _generate_legal_moves_c(self, CMoveList *move_list)
    cdef bint make_move_c(self, int move)
    cdef void unmake_move_c(self)

    cpdef object get_piece_at(self, int square)
    cpdef bint is_square_attacked(self, int sq, int attacker_color)
    cpdef bint in_check(self)
    cpdef list generate_pseudo_legal_moves(self)
    cpdef list generate_legal_moves(self)
    cpdef bint make_move(self, int move)
    cpdef void unmake_move(self)
    cpdef bint make_null_move(self)
    cpdef long long run_perft_recursive(self, int depth)
