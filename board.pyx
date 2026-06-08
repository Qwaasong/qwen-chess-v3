# cython: language_level=3
# type: ignore
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""Board representation and move generation using Cython-accelerated Bitboards."""

from __future__ import annotations
import chess
from libc.string cimport memcpy, memset

# Portable LSB via inline C — uses MSVC _BitScanForward64 or GCC __builtin_ctzll
cdef extern from *:
    """
    #ifdef _MSC_VER
    #include <intrin.h>
    static __forceinline int _cy_lsb_impl(unsigned long long bb) {
        if (!bb) return -1;
        unsigned long idx;
        _BitScanForward64(&idx, bb);
        return (int)idx;
    }
    #else
    static __inline__ int _cy_lsb_impl(unsigned long long bb) {
        if (!bb) return -1;
        return __builtin_ctzll(bb);
    }
    #endif
    """
    int _cy_lsb_impl(unsigned long long bb) nogil

# ---------------------------------------------------------------------------
# Constants for square indices (A1=0 ... H8=63)
# ---------------------------------------------------------------------------
A1, B1, C1, D1, E1, F1, G1, H1 = 0, 1, 2, 3, 4, 5, 6, 7
A2, B2, C2, D2, E2, F2, G2, H2 = 8, 9, 10, 11, 12, 13, 14, 15
A3, B3, C3, D3, E3, F3, G3, H3 = 16, 17, 18, 19, 20, 21, 22, 23
A4, B4, C4, D4, E4, F4, G4, H4 = 24, 25, 26, 27, 28, 29, 30, 31
A5, B5, C5, D5, E5, F5, G5, H5 = 32, 33, 34, 35, 36, 37, 38, 39
A6, B6, C6, D6, E6, F6, G6, H6 = 40, 41, 42, 43, 44, 45, 46, 47
A7, B7, C7, D7, E7, F7, G7, H7 = 48, 49, 50, 51, 52, 53, 54, 55
A8, B8, C8, D8, E8, F8, G8, H8 = 56, 57, 58, 59, 60, 61, 62, 63

# Colors
WHITE = 0
BLACK = 1

# Piece types (0-11): white P N B R Q K, black p n b r q k
P_P = 0;  P_N = 1;  P_B = 2;  P_R = 3;  P_Q = 4;  P_K = 5
P_p = 6;  P_n = 7;  P_b = 8;  P_r = 9;  P_q = 10; P_k = 11

# Castling rights
WK = 1; WQ = 2; BK = 4; BQ = 8

# Move flags
FLAG_NORMAL      = 0
FLAG_DOUBLE_PUSH = 1
FLAG_CASTLE      = 2
FLAG_EP          = 3
FLAG_PROMOTE_N   = 8
FLAG_PROMOTE_B   = 9
FLAG_PROMOTE_R   = 10
FLAG_PROMOTE_Q   = 11

# Sentinel value for empty squares in piece_map (-1 = no piece)
EMPTY_SQUARE = -1

# Maximum search/game depth for the C-level history stack
DEF MAX_HISTORY = 512

# ---------------------------------------------------------------------------
# Magic Numbers (Tord Romstad / Stockfish)
# ---------------------------------------------------------------------------
BISHOP_MAGICS = [
    0x40040844404084, 0x2004208a004208, 0x10190041080202,
    0x108060845042010, 0x581104180800210, 0x2112080446200010,
    0x1080820820060210, 0x3c0808410220200, 0x4050404440404,
    0x21001420088, 0x24d0080801082102, 0x1020a0a020400,
    0x40308200402, 0x4011002100800, 0x401484104104005,
    0x801010402020200, 0x400210c3880100, 0x404022024108200,
    0x810018200204102, 0x4002801a02003, 0x85040820080400,
    0x810102c808880400, 0xe900410884800, 0x8002020480840102,
    0x220200865090201, 0x2010100a02021202, 0x152048408022401,
    0x20080002081110, 0x4001001021004000, 0x800040400a011002,
    0x44004081011002, 0x1c004001012080, 0x8004200962a00220,
    0x8422100208500202, 0x2000402200300c08, 0x8646020080080080,
    0x80020a0200100808, 0x2010004880111000, 0x623000a080011400,
    0x42008c0340209202, 0x209188240001000, 0x400408a884001800,
    0x110400a6080400, 0x1840060a44020800, 0x90080104000041,
    0x201011000808101, 0x1a2208080504f080, 0x8012020600211212,
    0x500861011240000, 0x180806108200800, 0x4000020e01040044,
    0x300000261044000a, 0x802241102020002, 0x20906061210001,
    0x5a84841004010310, 0x4010801011c04, 0xa010109502200,
    0x4a02012000, 0x500201010098b028, 0x8040002811040900,
    0x28000010020204, 0x6000020202d0240, 0x8918844842082200,
    0x4010011029020020,
]

ROOK_MAGICS = [
    0x8a80104000800020, 0x140002000100040, 0x2801880a0017001,
    0x100081001000420, 0x200020010080420, 0x3001c0002010008,
    0x8480008002000100, 0x2080088004402900, 0x800098204000,
    0x2024401000200040, 0x100802000801000, 0x120800800801000,
    0x208808088000400, 0x2802200800400, 0x2200800100020080,
    0x801000060821100, 0x80044006422000, 0x100808020004000,
    0x12108a0010204200, 0x140848010000802, 0x481828014002800,
    0x8094004002004100, 0x4010040010010802, 0x20008806104,
    0x100400080208000, 0x2040002120081000, 0x21200680100081,
    0x20100080080080, 0x2000a00200410, 0x20080800400,
    0x80088400100102, 0x80004600042881, 0x4040008040800020,
    0x440003000200801, 0x4200011004500, 0x188020010100100,
    0x14800401802800, 0x2080040080800200, 0x124080204001001,
    0x200046502000484, 0x480400080088020, 0x1000422010034000,
    0x30200100110040, 0x100021010009, 0x2002080100110004,
    0x202008004008002, 0x20020004010100, 0x2048440040820001,
    0x101002200408200, 0x40802000401080, 0x4008142004410100,
    0x2060820c0120200, 0x1001004080100, 0x20c020080040080,
    0x2935610830022400, 0x44440041009200, 0x280001040802101,
    0x2100190040002085, 0x80c0084100102001, 0x4024081001000421,
    0x20030a0244872, 0x12001008414402, 0x2006104900a0804,
    0x1004081002402,
]

BISHOP_RELEVANT_BITS = [
    6, 5, 5, 5, 5, 5, 5, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 5, 5, 5, 5, 5, 5, 6,
]

ROOK_RELEVANT_BITS = [
    12, 11, 11, 11, 11, 11, 11, 12,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    12, 11, 11, 11, 11, 11, 11, 12,
]

# ---------------------------------------------------------------------------
# C-level bitboard helpers — all pure C, nogil-safe
# ---------------------------------------------------------------------------
cdef inline int cy_lsb(unsigned long long bb) nogil:
    """Index of least-significant set bit. Returns -1 if bb==0."""
    return _cy_lsb_impl(bb)

cdef inline unsigned long long cy_clear_bit(unsigned long long bb, int sq) nogil:
    return bb & ~(<unsigned long long>1 << sq)

cdef inline unsigned long long cy_set_bit(unsigned long long bb, int sq) nogil:
    return bb | (<unsigned long long>1 << sq)

cdef inline int cy_get_bit(unsigned long long bb, int sq) nogil:
    return <int>((bb >> sq) & 1)

# ---------------------------------------------------------------------------
# Public Python-visible wrappers
# ---------------------------------------------------------------------------
def lsb(bb):
    return cy_lsb(<unsigned long long>bb)

def pop_count(bb):
    return bin(bb).count('1')

def clear_bit(bb, sq):
    return cy_clear_bit(<unsigned long long>bb, sq)

def set_bit(bb, sq):
    return cy_set_bit(<unsigned long long>bb, sq)

def get_bit(bb, sq):
    return cy_get_bit(<unsigned long long>bb, sq)

# ---------------------------------------------------------------------------
# Sliding attack table init helpers (called once at import)
# ---------------------------------------------------------------------------
def mask_bishop_attacks(int square):
    """Generates bishop attack mask for a given square, excluding edges."""
    cdef unsigned long long attacks = 0
    cdef int tr = square // 8, tf = square % 8, r, f
    r = tr + 1; f = tf + 1
    while r < 7 and f < 7:
        attacks |= (<unsigned long long>1 << (r * 8 + f)); r += 1; f += 1
    r = tr + 1; f = tf - 1
    while r < 7 and f > 0:
        attacks |= (<unsigned long long>1 << (r * 8 + f)); r += 1; f -= 1
    r = tr - 1; f = tf + 1
    while r > 0 and f < 7:
        attacks |= (<unsigned long long>1 << (r * 8 + f)); r -= 1; f += 1
    r = tr - 1; f = tf - 1
    while r > 0 and f > 0:
        attacks |= (<unsigned long long>1 << (r * 8 + f)); r -= 1; f -= 1
    return attacks

def mask_rook_attacks(int square):
    """Generates rook attack mask for a given square, excluding edges."""
    cdef unsigned long long attacks = 0
    cdef int tr = square // 8, tf = square % 8, r, f
    for r in range(tr + 1, 7):
        attacks |= (<unsigned long long>1 << (r * 8 + tf))
    for r in range(tr - 1, 0, -1):
        attacks |= (<unsigned long long>1 << (r * 8 + tf))
    for f in range(tf + 1, 7):
        attacks |= (<unsigned long long>1 << (tr * 8 + f))
    for f in range(tf - 1, 0, -1):
        attacks |= (<unsigned long long>1 << (tr * 8 + f))
    return attacks

def bishop_attacks_on_the_fly(int square, unsigned long long block):
    """Generates bishop attacks on-the-fly considering blockers."""
    cdef unsigned long long attacks = 0, sq_bb
    cdef int tr = square // 8, tf = square % 8, r, f
    r = tr + 1; f = tf + 1
    while r < 8 and f < 8:
        sq_bb = <unsigned long long>1 << (r * 8 + f)
        attacks |= sq_bb
        if sq_bb & block: break
        r += 1; f += 1
    r = tr + 1; f = tf - 1
    while r < 8 and f >= 0:
        sq_bb = <unsigned long long>1 << (r * 8 + f)
        attacks |= sq_bb
        if sq_bb & block: break
        r += 1; f -= 1
    r = tr - 1; f = tf + 1
    while r >= 0 and f < 8:
        sq_bb = <unsigned long long>1 << (r * 8 + f)
        attacks |= sq_bb
        if sq_bb & block: break
        r -= 1; f += 1
    r = tr - 1; f = tf - 1
    while r >= 0 and f >= 0:
        sq_bb = <unsigned long long>1 << (r * 8 + f)
        attacks |= sq_bb
        if sq_bb & block: break
        r -= 1; f -= 1
    return attacks

def rook_attacks_on_the_fly(int square, unsigned long long block):
    """Generates rook attacks on-the-fly considering blockers."""
    cdef unsigned long long attacks = 0, sq_bb
    cdef int tr = square // 8, tf = square % 8, r, f
    for r in range(tr + 1, 8):
        sq_bb = <unsigned long long>1 << (r * 8 + tf)
        attacks |= sq_bb
        if sq_bb & block: break
    for r in range(tr - 1, -1, -1):
        sq_bb = <unsigned long long>1 << (r * 8 + tf)
        attacks |= sq_bb
        if sq_bb & block: break
    for f in range(tf + 1, 8):
        sq_bb = <unsigned long long>1 << (tr * 8 + f)
        attacks |= sq_bb
        if sq_bb & block: break
    for f in range(tf - 1, -1, -1):
        sq_bb = <unsigned long long>1 << (tr * 8 + f)
        attacks |= sq_bb
        if sq_bb & block: break
    return attacks

def set_occupancy(int index, int bits_in_mask, unsigned long long attack_mask):
    """Configures occupancy bitboard based on index and attack mask."""
    cdef unsigned long long occupancy = 0, temp_mask = attack_mask
    cdef int i, square
    for i in range(bits_in_mask):
        square = cy_lsb(temp_mask)
        temp_mask = cy_clear_bit(temp_mask, square)
        if index & (1 << i):
            occupancy = cy_set_bit(occupancy, square)
    return occupancy

# ---------------------------------------------------------------------------
# Precomputed attack tables
# Python-level lists (for public API) + C-level arrays (for hot path)
# ---------------------------------------------------------------------------
BISHOP_MASKS = [mask_bishop_attacks(sq) for sq in range(64)]
ROOK_MASKS   = [mask_rook_attacks(sq)   for sq in range(64)]

KNIGHT_ATTACKS = [0] * 64
KING_ATTACKS   = [0] * 64
PAWN_ATTACKS   = [[0] * 64 for _ in range(2)]
BISHOP_ATTACKS = [[0] * 512  for _ in range(64)]
ROOK_ATTACKS   = [[0] * 4096 for _ in range(64)]

# C-level arrays for the hot path (no GIL overhead on access)
cdef unsigned long long _KNIGHT_ATTACKS[64]
cdef unsigned long long _KING_ATTACKS[64]
cdef unsigned long long _PAWN_ATTACKS_W[64]
cdef unsigned long long _PAWN_ATTACKS_B[64]
cdef unsigned long long _BISHOP_MASKS[64]
cdef unsigned long long _ROOK_MASKS[64]
cdef unsigned long long _BISHOP_MAGICS[64]
cdef unsigned long long _ROOK_MAGICS[64]
cdef int               _BISHOP_BITS[64]
cdef int               _ROOK_BITS[64]
cdef unsigned long long _BISHOP_ATTACKS[64][512]
cdef unsigned long long _ROOK_ATTACKS[64][4096]


def _init_attack_tables():
    """Initialises all precomputed attack tables (Python lists + C arrays)."""
    cdef int sq_idx, r_coord, f_coord, dr, df, nr_coord, nf_coord
    cdef unsigned long long k_att, king_att, wp_att, bp_att, occ, atk
    cdef int b_bits, r_bits, i_val
    cdef unsigned long long b_mask, r_mask, b_magic, r_magic, idx_val

    for sq_idx in range(64):
        r_coord = sq_idx // 8
        f_coord = sq_idx % 8

        # Knight
        k_att = 0
        for dr, df in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            nr_coord = r_coord + dr; nf_coord = f_coord + df
            if 0 <= nr_coord < 8 and 0 <= nf_coord < 8:
                k_att |= (<unsigned long long>1 << (nr_coord * 8 + nf_coord))
        KNIGHT_ATTACKS[sq_idx] = k_att
        _KNIGHT_ATTACKS[sq_idx] = k_att

        # King
        king_att = 0
        for dr in range(-1, 2):
            for df in range(-1, 2):
                if dr == 0 and df == 0: continue
                nr_coord = r_coord + dr; nf_coord = f_coord + df
                if 0 <= nr_coord < 8 and 0 <= nf_coord < 8:
                    king_att |= (<unsigned long long>1 << (nr_coord * 8 + nf_coord))
        KING_ATTACKS[sq_idx] = king_att
        _KING_ATTACKS[sq_idx] = king_att

        # White pawn attacks
        wp_att = 0
        if f_coord > 0 and sq_idx + 7 < 64: wp_att |= (<unsigned long long>1 << (sq_idx + 7))
        if f_coord < 7 and sq_idx + 9 < 64: wp_att |= (<unsigned long long>1 << (sq_idx + 9))
        PAWN_ATTACKS[WHITE][sq_idx] = wp_att
        _PAWN_ATTACKS_W[sq_idx] = wp_att

        # Black pawn attacks
        bp_att = 0
        if f_coord > 0 and sq_idx - 9 >= 0: bp_att |= (<unsigned long long>1 << (sq_idx - 9))
        if f_coord < 7 and sq_idx - 7 >= 0: bp_att |= (<unsigned long long>1 << (sq_idx - 7))
        PAWN_ATTACKS[BLACK][sq_idx] = bp_att
        _PAWN_ATTACKS_B[sq_idx] = bp_att

    for sq_idx in range(64):
        b_mask  = <unsigned long long>BISHOP_MASKS[sq_idx]
        b_bits  = BISHOP_RELEVANT_BITS[sq_idx]
        b_magic = <unsigned long long>BISHOP_MAGICS[sq_idx]
        _BISHOP_MASKS[sq_idx]  = b_mask
        _BISHOP_MAGICS[sq_idx] = b_magic
        _BISHOP_BITS[sq_idx]   = b_bits

        for i_val in range(1 << b_bits):
            occ     = set_occupancy(i_val, b_bits, b_mask)
            idx_val = (occ * b_magic) >> (64 - b_bits)
            atk     = bishop_attacks_on_the_fly(sq_idx, occ)
            BISHOP_ATTACKS[sq_idx][idx_val] = atk
            _BISHOP_ATTACKS[sq_idx][idx_val] = atk

        r_mask  = <unsigned long long>ROOK_MASKS[sq_idx]
        r_bits  = ROOK_RELEVANT_BITS[sq_idx]
        r_magic = <unsigned long long>ROOK_MAGICS[sq_idx]
        _ROOK_MASKS[sq_idx]  = r_mask
        _ROOK_MAGICS[sq_idx] = r_magic
        _ROOK_BITS[sq_idx]   = r_bits

        for i_val in range(1 << r_bits):
            occ     = set_occupancy(i_val, r_bits, r_mask)
            idx_val = (occ * r_magic) >> (64 - r_bits)
            atk     = rook_attacks_on_the_fly(sq_idx, occ)
            ROOK_ATTACKS[sq_idx][idx_val] = atk
            _ROOK_ATTACKS[sq_idx][idx_val] = atk


_init_attack_tables()


# ---------------------------------------------------------------------------
# Hot-path attack lookups (C arrays, no Python object overhead)
# ---------------------------------------------------------------------------
cdef inline unsigned long long cy_get_bishop_attacks(
        int square, unsigned long long occupancy) nogil:
    cdef unsigned long long masked_occ = occupancy & _BISHOP_MASKS[square]
    cdef unsigned long long idx = (masked_occ * _BISHOP_MAGICS[square]) >> (64 - _BISHOP_BITS[square])
    return _BISHOP_ATTACKS[square][idx]

cdef inline unsigned long long cy_get_rook_attacks(
        int square, unsigned long long occupancy) nogil:
    cdef unsigned long long masked_occ = occupancy & _ROOK_MASKS[square]
    cdef unsigned long long idx = (masked_occ * _ROOK_MAGICS[square]) >> (64 - _ROOK_BITS[square])
    return _ROOK_ATTACKS[square][idx]

cdef inline unsigned long long cy_get_queen_attacks(
        int square, unsigned long long occupancy) nogil:
    return cy_get_bishop_attacks(square, occupancy) | cy_get_rook_attacks(square, occupancy)

# Public wrappers
def get_bishop_attacks(int square, unsigned long long occupancy):
    return cy_get_bishop_attacks(square, occupancy)

def get_rook_attacks(int square, unsigned long long occupancy):
    return cy_get_rook_attacks(square, occupancy)

def get_queen_attacks(int square, unsigned long long occupancy):
    return cy_get_queen_attacks(square, occupancy)

# ---------------------------------------------------------------------------
# Move packing / unpacking
# ---------------------------------------------------------------------------
cdef inline int cy_encode_move(int from_sq, int to_sq, int flag) nogil:
    return from_sq | (to_sq << 6) | (flag << 12)

cdef inline int cy_get_move_source(int move) nogil:
    return move & 0x3F

cdef inline int cy_get_move_dest(int move) nogil:
    return (move >> 6) & 0x3F

cdef inline int cy_get_move_flag(int move) nogil:
    return (move >> 12) & 0x0F

def encode_move(int from_sq, int to_sq, int flag):
    return cy_encode_move(from_sq, to_sq, flag)

def get_move_source(int move):
    return cy_get_move_source(move)

def get_move_dest(int move):
    return cy_get_move_dest(move)

def get_move_flag(int move):
    return cy_get_move_flag(move)

# ---------------------------------------------------------------------------
# Zobrist Keys & Initialization
# ---------------------------------------------------------------------------
cdef public unsigned long long ZOBRIST_PIECES[12][64]
cdef public unsigned long long ZOBRIST_CASTLING[16]
cdef public unsigned long long ZOBRIST_EP[8]
cdef public unsigned long long ZOBRIST_SIDE

cdef void _init_zobrist() noexcept:
    cdef unsigned long long state = 1070372ULL
    cdef int i, j
    for i in range(12):
        for j in range(64):
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            ZOBRIST_PIECES[i][j] = state
    for i in range(16):
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        ZOBRIST_CASTLING[i] = state
    for i in range(8):
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        ZOBRIST_EP[i] = state
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    global ZOBRIST_SIDE
    ZOBRIST_SIDE = state

_init_zobrist()


# ---------------------------------------------------------------------------
# CGameState — C struct for O(1) stack-based make/unmake history
#
# Stores board snapshots entirely in stack-allocated C memory.
# No Python allocator involvement, no reference counting, no GIL for copy.
# ---------------------------------------------------------------------------
cdef struct CGameState:
    unsigned long long bitboards[12]   # piece bitboards (white P..K, black p..k)
    unsigned long long occupancies[3]  # [WHITE, BLACK, BOTH]
    int side_to_move
    int castling_rights
    int en_passant_sq
    int halfmove_clock
    int fullmove_number
    int piece_map[64]                  # piece index per square; -1 = empty
    unsigned long long zobrist_key


# ---------------------------------------------------------------------------
# Helper: copy CGameState fields to/from CustomBitboardBoard
# ---------------------------------------------------------------------------
cdef inline void _save_state(CGameState *slot,
                              unsigned long long *bbs_py,
                              unsigned long long *occ_py,
                              int side, int castle, int ep,
                              int half, int full,
                              int *pm,
                              unsigned long long zobrist) noexcept nogil:
    """Write board state into a CGameState slot (pure C, no GIL)."""
    memcpy(slot.bitboards,    bbs_py, 12 * sizeof(unsigned long long))
    memcpy(slot.occupancies,  occ_py,  3 * sizeof(unsigned long long))
    memcpy(slot.piece_map,    pm,     64 * sizeof(int))
    slot.side_to_move    = side
    slot.castling_rights = castle
    slot.en_passant_sq   = ep
    slot.halfmove_clock  = half
    slot.fullmove_number = full
    slot.zobrist_key     = zobrist


cdef inline void cy_add_move(CMoveList *ml, int m) nogil:
    ml.moves[ml.count] = m
    ml.count += 1

# ---------------------------------------------------------------------------
# CustomBitboardBoard — main board class
# ---------------------------------------------------------------------------
cdef class CustomBitboardBoard:
    """High-performance chess board using Cython-accelerated bitboards.

    Key layout decisions:
    - bitboards / occupancies: Python list (public API, accessed from engine.py)
    - piece_map: cdef int[64] C array, sentinel -1 = empty (no Python object overhead)
    - _history: cdef CGameState[MAX_HISTORY] C stack, zero allocator overhead
    """

    @property
    def bitboards(self):
        return [self._bb[i] for i in range(12)]

    @property
    def occupancies(self):
        return [self._occ[i] for i in range(3)]

    def __init__(self):
        self.side_to_move    = WHITE
        self.castling_rights = 0
        self.en_passant_sq   = 64
        self.halfmove_clock  = 0
        self.fullmove_number = 1
        self._history_len    = 0
        self.zobrist_key     = 0

        # Initialise piece_map to -1 (empty) using libc memset
        memset(self.piece_map, 0xFF, 64 * sizeof(int))  # 0xFF fill → -1 for int32
        memset(self._bb,  0, 12 * sizeof(unsigned long long))
        memset(self._occ, 0,  3 * sizeof(unsigned long long))

    cdef unsigned long long _recompute_zobrist(self):
        cdef unsigned long long key = 0
        cdef int sq, p, ep_file
        for sq in range(64):
            p = self.piece_map[sq]
            if p != -1:
                key ^= ZOBRIST_PIECES[p][sq]
        if self.side_to_move == 1:  # BLACK = 1
            key ^= ZOBRIST_SIDE
        key ^= ZOBRIST_CASTLING[self.castling_rights]
        if self.en_passant_sq != 64:
            ep_file = self.en_passant_sq % 8
            key ^= ZOBRIST_EP[ep_file]
        return key

    @classmethod
    def from_chess_board(cls, chess_board):
        """Converts a python-chess Board to CustomBitboardBoard."""
        cdef CustomBitboardBoard board = CustomBitboardBoard()
        board.side_to_move = WHITE if chess_board.turn == chess.WHITE else BLACK

        cdef int castle = 0
        if chess_board.has_kingside_castling_rights(chess.WHITE):  castle |= WK
        if chess_board.has_queenside_castling_rights(chess.WHITE): castle |= WQ
        if chess_board.has_kingside_castling_rights(chess.BLACK):  castle |= BK
        if chess_board.has_queenside_castling_rights(chess.BLACK): castle |= BQ
        board.castling_rights = castle

        board.en_passant_sq = (
            chess_board.ep_square if chess_board.ep_square is not None else 64
        )
        board.halfmove_clock  = chess_board.halfmove_clock
        board.fullmove_number = chess_board.fullmove_number

        # piece_map is already -1 (memset in __init__)
        cdef int square, color_offset, p_type, idx
        for square in range(64):
            piece = chess_board.piece_at(square)
            if piece is not None:
                color_offset = 0 if piece.color == chess.WHITE else 6
                p_type = piece.piece_type
                idx = p_type - 1 + color_offset
                board._bb[idx] = cy_set_bit(board._bb[idx], square)
                board.piece_map[square] = idx

        cdef int i
        for i in range(6):
            board._occ[WHITE] |= board._bb[i]
            board._occ[BLACK] |= board._bb[i + 6]
        board._occ[2] = board._occ[WHITE] | board._occ[BLACK]

        board.zobrist_key = board._recompute_zobrist()
        return board

    cpdef object get_piece_at(self, int square):
        """Returns piece index (0-11) at square, or None if empty. O(1)."""
        cdef int p = self.piece_map[square]
        return None if p == -1 else p

    cpdef bint is_square_attacked(self, int sq, int attacker_color):
        """Returns True if sq is attacked by any piece of attacker_color."""
        cdef unsigned long long occupancy = self._occ[2]
        cdef unsigned long long pawn_bb, knight_bb, king_bb, bq_bb, rq_bb
        cdef int f = sq % 8, friendly_offset

        # 1. Pawn attacks
        if attacker_color == WHITE:
            pawn_bb = self._bb[P_P]
            if f > 0 and sq - 9 >= 0 and cy_get_bit(pawn_bb, sq - 9): return True
            if f < 7 and sq - 7 >= 0 and cy_get_bit(pawn_bb, sq - 7): return True
        else:
            pawn_bb = self._bb[P_p]
            if f > 0 and sq + 7 < 64 and cy_get_bit(pawn_bb, sq + 7): return True
            if f < 7 and sq + 9 < 64 and cy_get_bit(pawn_bb, sq + 9): return True

        # 2. Knight attacks
        knight_bb = self._bb[P_N if attacker_color == WHITE else P_n]
        if _KNIGHT_ATTACKS[sq] & knight_bb: return True

        # 3. King attacks
        king_bb = self._bb[P_K if attacker_color == WHITE else P_k]
        if _KING_ATTACKS[sq] & king_bb: return True

        # 4. Bishop / Queen (diagonals)
        friendly_offset = 0 if attacker_color == WHITE else 6
        bq_bb = self._bb[P_B + friendly_offset] | self._bb[P_Q + friendly_offset]
        if cy_get_bishop_attacks(sq, occupancy) & bq_bb: return True

        # 5. Rook / Queen (orthogonals)
        rq_bb = self._bb[P_R + friendly_offset] | self._bb[P_Q + friendly_offset]
        if cy_get_rook_attacks(sq, occupancy) & rq_bb: return True

        return False

    cpdef bint in_check(self):
        """Returns True if the side to move's king is in check."""
        cdef unsigned long long king_bb = self._bb[P_K if self.side_to_move == WHITE else P_k]
        cdef int king_sq = cy_lsb(king_bb)
        if king_sq == -1:
            return False
        return self.is_square_attacked(king_sq, BLACK if self.side_to_move == WHITE else WHITE)

    cdef void _generate_pseudo_legal_moves_c(self, CMoveList *move_list):
        """Generates pseudo-legal moves for the current side to move (C-level)."""
        cdef int side = self.side_to_move
        cdef unsigned long long friendly_occ = self._occ[WHITE if side == WHITE else BLACK]
        cdef unsigned long long opponent_occ = self._occ[BLACK if side == WHITE else WHITE]
        cdef unsigned long long both_occ = self._occ[2]

        cdef int offset = 0 if side == WHITE else 6
        cdef unsigned long long pawn_bb   = self._bb[P_P + offset]
        cdef unsigned long long knight_bb = self._bb[P_N + offset]
        cdef unsigned long long bishop_bb = self._bb[P_B + offset]
        cdef unsigned long long rook_bb   = self._bb[P_R + offset]
        cdef unsigned long long queen_bb  = self._bb[P_Q + offset]
        cdef unsigned long long king_bb   = self._bb[P_K + offset]

        cdef unsigned long long pawn_sqs, dest_sqs, knights, bishops, rooks, queens, king_iter
        cdef int from_sq, to_sq, to_sq2, r, f, dest_sq, flag

        # --- Pawn Moves ---
        pawn_sqs = pawn_bb
        while pawn_sqs:
            from_sq = cy_lsb(pawn_sqs)
            pawn_sqs = cy_clear_bit(pawn_sqs, from_sq)
            r = from_sq // 8
            f = from_sq % 8

            if side == WHITE:
                to_sq = from_sq + 8
                if to_sq < 64 and not cy_get_bit(both_occ, to_sq):
                    if r == 6:
                        for flag in [FLAG_PROMOTE_Q, FLAG_PROMOTE_R, FLAG_PROMOTE_B, FLAG_PROMOTE_N]:
                            cy_add_move(move_list, cy_encode_move(from_sq, to_sq, flag))
                    else:
                        cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))
                    to_sq2 = from_sq + 16
                    if r == 1 and not cy_get_bit(both_occ, to_sq2):
                        cy_add_move(move_list, cy_encode_move(from_sq, to_sq2, FLAG_DOUBLE_PUSH))

                if f > 0:
                    dest_sq = from_sq + 7
                    if dest_sq < 64:
                        if cy_get_bit(opponent_occ, dest_sq):
                            if r == 6:
                                for flag in [FLAG_PROMOTE_Q, FLAG_PROMOTE_R, FLAG_PROMOTE_B, FLAG_PROMOTE_N]:
                                    cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, flag))
                            else:
                                cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_NORMAL))
                        elif dest_sq == self.en_passant_sq:
                            cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_EP))
                if f < 7:
                    dest_sq = from_sq + 9
                    if dest_sq < 64:
                        if cy_get_bit(opponent_occ, dest_sq):
                            if r == 6:
                                for flag in [FLAG_PROMOTE_Q, FLAG_PROMOTE_R, FLAG_PROMOTE_B, FLAG_PROMOTE_N]:
                                    cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, flag))
                            else:
                                cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_NORMAL))
                        elif dest_sq == self.en_passant_sq:
                            cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_EP))
            else:
                to_sq = from_sq - 8
                if to_sq >= 0 and not cy_get_bit(both_occ, to_sq):
                    if r == 1:
                        for flag in [FLAG_PROMOTE_Q, FLAG_PROMOTE_R, FLAG_PROMOTE_B, FLAG_PROMOTE_N]:
                            cy_add_move(move_list, cy_encode_move(from_sq, to_sq, flag))
                    else:
                        cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))
                    to_sq2 = from_sq - 16
                    if r == 6 and not cy_get_bit(both_occ, to_sq2):
                        cy_add_move(move_list, cy_encode_move(from_sq, to_sq2, FLAG_DOUBLE_PUSH))

                if f > 0:
                    dest_sq = from_sq - 9
                    if dest_sq >= 0:
                        if cy_get_bit(opponent_occ, dest_sq):
                            if r == 1:
                                for flag in [FLAG_PROMOTE_Q, FLAG_PROMOTE_R, FLAG_PROMOTE_B, FLAG_PROMOTE_N]:
                                    cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, flag))
                            else:
                                cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_NORMAL))
                        elif dest_sq == self.en_passant_sq:
                            cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_EP))
                if f < 7:
                    dest_sq = from_sq - 7
                    if dest_sq >= 0:
                        if cy_get_bit(opponent_occ, dest_sq):
                            if r == 1:
                                for flag in [FLAG_PROMOTE_Q, FLAG_PROMOTE_R, FLAG_PROMOTE_B, FLAG_PROMOTE_N]:
                                    cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, flag))
                            else:
                                cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_NORMAL))
                        elif dest_sq == self.en_passant_sq:
                            cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_EP))

        # --- Knight Moves ---
        knights = knight_bb
        while knights:
            from_sq = cy_lsb(knights)
            knights = cy_clear_bit(knights, from_sq)
            dest_sqs = _KNIGHT_ATTACKS[from_sq] & ~friendly_occ
            while dest_sqs:
                to_sq = cy_lsb(dest_sqs)
                dest_sqs = cy_clear_bit(dest_sqs, to_sq)
                cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- Bishop Moves ---
        bishops = bishop_bb
        while bishops:
            from_sq = cy_lsb(bishops)
            bishops = cy_clear_bit(bishops, from_sq)
            dest_sqs = cy_get_bishop_attacks(from_sq, both_occ) & ~friendly_occ
            while dest_sqs:
                to_sq = cy_lsb(dest_sqs)
                dest_sqs = cy_clear_bit(dest_sqs, to_sq)
                cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- Rook Moves ---
        rooks = rook_bb
        while rooks:
            from_sq = cy_lsb(rooks)
            rooks = cy_clear_bit(rooks, from_sq)
            dest_sqs = cy_get_rook_attacks(from_sq, both_occ) & ~friendly_occ
            while dest_sqs:
                to_sq = cy_lsb(dest_sqs)
                dest_sqs = cy_clear_bit(dest_sqs, to_sq)
                cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- Queen Moves ---
        queens = queen_bb
        while queens:
            from_sq = cy_lsb(queens)
            queens = cy_clear_bit(queens, from_sq)
            dest_sqs = cy_get_queen_attacks(from_sq, both_occ) & ~friendly_occ
            while dest_sqs:
                to_sq = cy_lsb(dest_sqs)
                dest_sqs = cy_clear_bit(dest_sqs, to_sq)
                cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- King Moves ---
        king_iter = king_bb
        if king_iter:
            from_sq = cy_lsb(king_iter)
            dest_sqs = _KING_ATTACKS[from_sq] & ~friendly_occ
            while dest_sqs:
                to_sq = cy_lsb(dest_sqs)
                dest_sqs = cy_clear_bit(dest_sqs, to_sq)
                cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))

            # Castling
            if side == WHITE:
                if (
                    (self.castling_rights & WK)
                    and not cy_get_bit(both_occ, F1)
                    and not cy_get_bit(both_occ, G1)
                    and not self.is_square_attacked(E1, BLACK)
                    and not self.is_square_attacked(F1, BLACK)
                ):
                    cy_add_move(move_list, cy_encode_move(E1, G1, FLAG_CASTLE))
                if (
                    (self.castling_rights & WQ)
                    and not cy_get_bit(both_occ, D1)
                    and not cy_get_bit(both_occ, C1)
                    and not cy_get_bit(both_occ, B1)
                    and not self.is_square_attacked(E1, BLACK)
                    and not self.is_square_attacked(D1, BLACK)
                ):
                    cy_add_move(move_list, cy_encode_move(E1, C1, FLAG_CASTLE))
            else:
                if (
                    (self.castling_rights & BK)
                    and not cy_get_bit(both_occ, F8)
                    and not cy_get_bit(both_occ, G8)
                    and not self.is_square_attacked(E8, WHITE)
                    and not self.is_square_attacked(F8, WHITE)
                ):
                    cy_add_move(move_list, cy_encode_move(E8, G8, FLAG_CASTLE))
                if (
                    (self.castling_rights & BQ)
                    and not cy_get_bit(both_occ, D8)
                    and not cy_get_bit(both_occ, C8)
                    and not cy_get_bit(both_occ, B8)
                    and not self.is_square_attacked(E8, WHITE)
                    and not self.is_square_attacked(D8, WHITE)
                ):
                    cy_add_move(move_list, cy_encode_move(E8, C8, FLAG_CASTLE))

    cdef void _generate_captures_c(self, CMoveList *move_list):
        """Generates pseudo-legal captures and promotions for the current side to move (C-level)."""
        cdef int side = self.side_to_move
        cdef unsigned long long friendly_occ = self._occ[WHITE if side == WHITE else BLACK]
        cdef unsigned long long opponent_occ = self._occ[BLACK if side == WHITE else WHITE]
        cdef unsigned long long both_occ = self._occ[2]

        cdef int offset = 0 if side == WHITE else 6
        cdef unsigned long long pawn_bb   = self._bb[P_P + offset]
        cdef unsigned long long knight_bb = self._bb[P_N + offset]
        cdef unsigned long long bishop_bb = self._bb[P_B + offset]
        cdef unsigned long long rook_bb   = self._bb[P_R + offset]
        cdef unsigned long long queen_bb  = self._bb[P_Q + offset]
        cdef unsigned long long king_bb   = self._bb[P_K + offset]

        cdef unsigned long long pawn_sqs, dest_sqs, knights, bishops, rooks, queens, king_iter
        cdef int from_sq, to_sq, r, f, dest_sq, flag

        # --- Pawn Captures & Promotions ---
        pawn_sqs = pawn_bb
        while pawn_sqs:
            from_sq = cy_lsb(pawn_sqs)
            pawn_sqs = cy_clear_bit(pawn_sqs, from_sq)
            r = from_sq // 8
            f = from_sq % 8

            if side == WHITE:
                # Promotion without capture
                if r == 6:
                    to_sq = from_sq + 8
                    if to_sq < 64 and not cy_get_bit(both_occ, to_sq):
                        for flag in [FLAG_PROMOTE_Q, FLAG_PROMOTE_R, FLAG_PROMOTE_B, FLAG_PROMOTE_N]:
                            cy_add_move(move_list, cy_encode_move(from_sq, to_sq, flag))
                # Captures (with or without promotion)
                if f > 0:
                    dest_sq = from_sq + 7
                    if dest_sq < 64:
                        if cy_get_bit(opponent_occ, dest_sq):
                            if r == 6:
                                for flag in [FLAG_PROMOTE_Q, FLAG_PROMOTE_R, FLAG_PROMOTE_B, FLAG_PROMOTE_N]:
                                    cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, flag))
                            else:
                                cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_NORMAL))
                        elif dest_sq == self.en_passant_sq:
                            cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_EP))
                if f < 7:
                    dest_sq = from_sq + 9
                    if dest_sq < 64:
                        if cy_get_bit(opponent_occ, dest_sq):
                            if r == 6:
                                for flag in [FLAG_PROMOTE_Q, FLAG_PROMOTE_R, FLAG_PROMOTE_B, FLAG_PROMOTE_N]:
                                    cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, flag))
                            else:
                                cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_NORMAL))
                        elif dest_sq == self.en_passant_sq:
                            cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_EP))
            else:
                # Promotion without capture
                if r == 1:
                    to_sq = from_sq - 8
                    if to_sq >= 0 and not cy_get_bit(both_occ, to_sq):
                        for flag in [FLAG_PROMOTE_Q, FLAG_PROMOTE_R, FLAG_PROMOTE_B, FLAG_PROMOTE_N]:
                            cy_add_move(move_list, cy_encode_move(from_sq, to_sq, flag))
                # Captures (with or without promotion)
                if f > 0:
                    dest_sq = from_sq - 9
                    if dest_sq >= 0:
                        if cy_get_bit(opponent_occ, dest_sq):
                            if r == 1:
                                for flag in [FLAG_PROMOTE_Q, FLAG_PROMOTE_R, FLAG_PROMOTE_B, FLAG_PROMOTE_N]:
                                    cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, flag))
                            else:
                                cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_NORMAL))
                        elif dest_sq == self.en_passant_sq:
                            cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_EP))
                if f < 7:
                    dest_sq = from_sq - 7
                    if dest_sq >= 0:
                        if cy_get_bit(opponent_occ, dest_sq):
                            if r == 1:
                                for flag in [FLAG_PROMOTE_Q, FLAG_PROMOTE_R, FLAG_PROMOTE_B, FLAG_PROMOTE_N]:
                                    cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, flag))
                            else:
                                cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_NORMAL))
                        elif dest_sq == self.en_passant_sq:
                            cy_add_move(move_list, cy_encode_move(from_sq, dest_sq, FLAG_EP))

        # --- Knight Captures ---
        knights = knight_bb
        while knights:
            from_sq = cy_lsb(knights)
            knights = cy_clear_bit(knights, from_sq)
            dest_sqs = _KNIGHT_ATTACKS[from_sq] & opponent_occ
            while dest_sqs:
                to_sq = cy_lsb(dest_sqs)
                dest_sqs = cy_clear_bit(dest_sqs, to_sq)
                cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- Bishop Captures ---
        bishops = bishop_bb
        while bishops:
            from_sq = cy_lsb(bishops)
            bishops = cy_clear_bit(bishops, from_sq)
            dest_sqs = cy_get_bishop_attacks(from_sq, both_occ) & opponent_occ
            while dest_sqs:
                to_sq = cy_lsb(dest_sqs)
                dest_sqs = cy_clear_bit(dest_sqs, to_sq)
                cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- Rook Captures ---
        rooks = rook_bb
        while rooks:
            from_sq = cy_lsb(rooks)
            rooks = cy_clear_bit(rooks, from_sq)
            dest_sqs = cy_get_rook_attacks(from_sq, both_occ) & opponent_occ
            while dest_sqs:
                to_sq = cy_lsb(dest_sqs)
                dest_sqs = cy_clear_bit(dest_sqs, to_sq)
                cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- Queen Captures ---
        queens = queen_bb
        while queens:
            from_sq = cy_lsb(queens)
            queens = cy_clear_bit(queens, from_sq)
            dest_sqs = cy_get_queen_attacks(from_sq, both_occ) & opponent_occ
            while dest_sqs:
                to_sq = cy_lsb(dest_sqs)
                dest_sqs = cy_clear_bit(dest_sqs, to_sq)
                cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- King Captures ---
        king_iter = king_bb
        if king_iter:
            from_sq = cy_lsb(king_iter)
            dest_sqs = _KING_ATTACKS[from_sq] & opponent_occ
            while dest_sqs:
                to_sq = cy_lsb(dest_sqs)
                dest_sqs = cy_clear_bit(dest_sqs, to_sq)
                cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))

    cdef void _generate_quiets_c(self, CMoveList *move_list):
        """Generates pseudo-legal quiet moves (non-captures) for the current side to move (C-level)."""
        cdef int side = self.side_to_move
        cdef unsigned long long friendly_occ = self._occ[WHITE if side == WHITE else BLACK]
        cdef unsigned long long opponent_occ = self._occ[BLACK if side == WHITE else WHITE]
        cdef unsigned long long both_occ = self._occ[2]

        cdef int offset = 0 if side == WHITE else 6
        cdef unsigned long long pawn_bb   = self._bb[P_P + offset]
        cdef unsigned long long knight_bb = self._bb[P_N + offset]
        cdef unsigned long long bishop_bb = self._bb[P_B + offset]
        cdef unsigned long long rook_bb   = self._bb[P_R + offset]
        cdef unsigned long long queen_bb  = self._bb[P_Q + offset]
        cdef unsigned long long king_bb   = self._bb[P_K + offset]

        cdef unsigned long long pawn_sqs, dest_sqs, knights, bishops, rooks, queens, king_iter
        cdef int from_sq, to_sq, to_sq2, r, f

        # --- Pawn Quiet Moves ---
        pawn_sqs = pawn_bb
        while pawn_sqs:
            from_sq = cy_lsb(pawn_sqs)
            pawn_sqs = cy_clear_bit(pawn_sqs, from_sq)
            r = from_sq // 8

            if side == WHITE:
                if r < 6: # promotions are handled in captures
                    to_sq = from_sq + 8
                    if to_sq < 64 and not cy_get_bit(both_occ, to_sq):
                        cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))
                        to_sq2 = from_sq + 16
                        if r == 1 and not cy_get_bit(both_occ, to_sq2):
                            cy_add_move(move_list, cy_encode_move(from_sq, to_sq2, FLAG_DOUBLE_PUSH))
            else:
                if r > 1: # promotions are handled in captures
                    to_sq = from_sq - 8
                    if to_sq >= 0 and not cy_get_bit(both_occ, to_sq):
                        cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))
                        to_sq2 = from_sq - 16
                        if r == 6 and not cy_get_bit(both_occ, to_sq2):
                            cy_add_move(move_list, cy_encode_move(from_sq, to_sq2, FLAG_DOUBLE_PUSH))

        # --- Knight Quiets ---
        knights = knight_bb
        while knights:
            from_sq = cy_lsb(knights)
            knights = cy_clear_bit(knights, from_sq)
            dest_sqs = _KNIGHT_ATTACKS[from_sq] & ~both_occ
            while dest_sqs:
                to_sq = cy_lsb(dest_sqs)
                dest_sqs = cy_clear_bit(dest_sqs, to_sq)
                cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- Bishop Quiets ---
        bishops = bishop_bb
        while bishops:
            from_sq = cy_lsb(bishops)
            bishops = cy_clear_bit(bishops, from_sq)
            dest_sqs = cy_get_bishop_attacks(from_sq, both_occ) & ~both_occ
            while dest_sqs:
                to_sq = cy_lsb(dest_sqs)
                dest_sqs = cy_clear_bit(dest_sqs, to_sq)
                cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- Rook Quiets ---
        rooks = rook_bb
        while rooks:
            from_sq = cy_lsb(rooks)
            rooks = cy_clear_bit(rooks, from_sq)
            dest_sqs = cy_get_rook_attacks(from_sq, both_occ) & ~both_occ
            while dest_sqs:
                to_sq = cy_lsb(dest_sqs)
                dest_sqs = cy_clear_bit(dest_sqs, to_sq)
                cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- Queen Quiets ---
        queens = queen_bb
        while queens:
            from_sq = cy_lsb(queens)
            queens = cy_clear_bit(queens, from_sq)
            dest_sqs = cy_get_queen_attacks(from_sq, both_occ) & ~both_occ
            while dest_sqs:
                to_sq = cy_lsb(dest_sqs)
                dest_sqs = cy_clear_bit(dest_sqs, to_sq)
                cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- King Quiets & Castling ---
        king_iter = king_bb
        if king_iter:
            from_sq = cy_lsb(king_iter)
            dest_sqs = _KING_ATTACKS[from_sq] & ~both_occ
            while dest_sqs:
                to_sq = cy_lsb(dest_sqs)
                dest_sqs = cy_clear_bit(dest_sqs, to_sq)
                cy_add_move(move_list, cy_encode_move(from_sq, to_sq, FLAG_NORMAL))

            # Castling (Quiet moves)
            if side == WHITE:
                if (
                    (self.castling_rights & WK)
                    and not cy_get_bit(both_occ, F1)
                    and not cy_get_bit(both_occ, G1)
                    and not self.is_square_attacked(E1, BLACK)
                    and not self.is_square_attacked(F1, BLACK)
                ):
                    cy_add_move(move_list, cy_encode_move(E1, G1, FLAG_CASTLE))
                if (
                    (self.castling_rights & WQ)
                    and not cy_get_bit(both_occ, D1)
                    and not cy_get_bit(both_occ, C1)
                    and not cy_get_bit(both_occ, B1)
                    and not self.is_square_attacked(E1, BLACK)
                    and not self.is_square_attacked(D1, BLACK)
                ):
                    cy_add_move(move_list, cy_encode_move(E1, C1, FLAG_CASTLE))
            else:
                if (
                    (self.castling_rights & BK)
                    and not cy_get_bit(both_occ, F8)
                    and not cy_get_bit(both_occ, G8)
                    and not self.is_square_attacked(E8, WHITE)
                    and not self.is_square_attacked(F8, WHITE)
                ):
                    cy_add_move(move_list, cy_encode_move(E8, G8, FLAG_CASTLE))
                if (
                    (self.castling_rights & BQ)
                    and not cy_get_bit(both_occ, D8)
                    and not cy_get_bit(both_occ, C8)
                    and not cy_get_bit(both_occ, B8)
                    and not self.is_square_attacked(E8, WHITE)
                    and not self.is_square_attacked(D8, WHITE)
                ):
                    cy_add_move(move_list, cy_encode_move(E8, C8, FLAG_CASTLE))

    cpdef list generate_pseudo_legal_moves(self):
        """Generates pseudo-legal moves for the current side to move (list wrapper)."""
        cdef CMoveList move_list
        move_list.count = 0
        self._generate_pseudo_legal_moves_c(&move_list)
        cdef list moves = []
        cdef int i
        for i in range(move_list.count):
            moves.append(move_list.moves[i])
        return moves



    cdef void _generate_legal_moves_c(self, CMoveList *legal_moves):
        """Generates strictly legal moves (C-level)."""
        cdef CMoveList pseudo
        pseudo.count = 0
        self._generate_pseudo_legal_moves_c(&pseudo)
        
        cdef int i, m
        legal_moves.count = 0
        for i in range(pseudo.count):
            m = pseudo.moves[i]
            if self.make_move_c(m):
                cy_add_move(legal_moves, m)
            self.unmake_move_c()

    cpdef list generate_legal_moves(self):
        """Generates strictly legal moves (list wrapper)."""
        cdef CMoveList legal_moves
        legal_moves.count = 0
        self._generate_legal_moves_c(&legal_moves)
        cdef list legal = []
        cdef int i
        for i in range(legal_moves.count):
            legal.append(legal_moves.moves[i])
        return legal

    cpdef bint make_move(self, int move):
        """Makes a move (Python wrapper)."""
        return self.make_move_c(move)

    cdef bint make_move_c(self, int move):
        """Makes a move. Pushes board state onto C stack; returns True if legal."""
        if self._history_len >= MAX_HISTORY:
            return False  # stack overflow guard

        # --- Save current state into C struct (no Python allocator) ---
        cdef CGameState *slot = &self._history[self._history_len]
        memcpy(slot.bitboards,    self._bb,       12 * sizeof(unsigned long long))
        memcpy(slot.occupancies,  self._occ,       3 * sizeof(unsigned long long))
        memcpy(slot.piece_map,    self.piece_map, 64 * sizeof(int))
        slot.side_to_move    = self.side_to_move
        slot.castling_rights = self.castling_rights
        slot.en_passant_sq   = self.en_passant_sq
        slot.halfmove_clock  = self.halfmove_clock
        slot.fullmove_number = self.fullmove_number
        slot.zobrist_key     = self.zobrist_key
        self._history_len += 1

        # Record values to compute incremental changes
        cdef int old_ep = self.en_passant_sq
        cdef int old_castle = self.castling_rights

        # --- Decode move ---
        cdef int from_sq = cy_get_move_source(move)
        cdef int to_sq   = cy_get_move_dest(move)
        cdef int flag    = cy_get_move_flag(move)
        cdef int side    = self.side_to_move
        cdef int opp_side = 1 if side == 0 else 0

        # Direct C-array access — no Python object, no isinstance check
        cdef int mp = self.piece_map[from_sq]
        if mp == -1:
            return False

        # XOR out the moving piece from the source square
        self.zobrist_key ^= ZOBRIST_PIECES[mp][from_sq]

        self._bb[mp] = cy_clear_bit(self._bb[mp], from_sq)
        self.piece_map[from_sq] = -1

        cdef int cap = self.piece_map[to_sq]  # -1 if empty

        self.en_passant_sq = 64
        self.halfmove_clock += 1
        if side == BLACK:
            self.fullmove_number += 1

        if cap != -1:
            # XOR out the captured piece from the destination square
            self.zobrist_key ^= ZOBRIST_PIECES[cap][to_sq]
            self._bb[cap] = cy_clear_bit(self._bb[cap], to_sq)
            self.piece_map[to_sq] = -1
            self.halfmove_clock = 0

        cdef int cap_ep_sq, ep_pawn_idx, prom_offset, p_prom

        if flag == FLAG_DOUBLE_PUSH:
            self.en_passant_sq = from_sq + 8 if side == WHITE else from_sq - 8
            self.halfmove_clock = 0
        elif flag == FLAG_EP:
            cap_ep_sq   = to_sq - 8 if side == WHITE else to_sq + 8
            ep_pawn_idx = P_p if side == WHITE else P_P
            # XOR out the captured en passant pawn
            self.zobrist_key ^= ZOBRIST_PIECES[ep_pawn_idx][cap_ep_sq]
            self._bb[ep_pawn_idx] = cy_clear_bit(self._bb[ep_pawn_idx], cap_ep_sq)
            self.piece_map[cap_ep_sq] = -1
            self.halfmove_clock = 0
        elif flag == FLAG_CASTLE:
            if to_sq == G1:
                # XOR out the rook at H1, XOR in the rook at F1
                self.zobrist_key ^= ZOBRIST_PIECES[P_R][H1] ^ ZOBRIST_PIECES[P_R][F1]
                self._bb[P_R] = cy_clear_bit(self._bb[P_R], H1)
                self._bb[P_R] = cy_set_bit(self._bb[P_R], F1)
                self.piece_map[H1] = -1; self.piece_map[F1] = P_R
            elif to_sq == C1:
                # XOR out the rook at A1, XOR in the rook at D1
                self.zobrist_key ^= ZOBRIST_PIECES[P_R][A1] ^ ZOBRIST_PIECES[P_R][D1]
                self._bb[P_R] = cy_clear_bit(self._bb[P_R], A1)
                self._bb[P_R] = cy_set_bit(self._bb[P_R], D1)
                self.piece_map[A1] = -1; self.piece_map[D1] = P_R
            elif to_sq == G8:
                # XOR out the rook at H8, XOR in the rook at F8
                self.zobrist_key ^= ZOBRIST_PIECES[P_r][H8] ^ ZOBRIST_PIECES[P_r][F8]
                self._bb[P_r] = cy_clear_bit(self._bb[P_r], H8)
                self._bb[P_r] = cy_set_bit(self._bb[P_r], F8)
                self.piece_map[H8] = -1; self.piece_map[F8] = P_r
            elif to_sq == C8:
                # XOR out the rook at A8, XOR in the rook at D8
                self.zobrist_key ^= ZOBRIST_PIECES[P_r][A8] ^ ZOBRIST_PIECES[P_r][D8]
                self._bb[P_r] = cy_clear_bit(self._bb[P_r], A8)
                self._bb[P_r] = cy_set_bit(self._bb[P_r], D8)
                self.piece_map[A8] = -1; self.piece_map[D8] = P_r

        # Place piece at destination (with promotion)
        if flag >= FLAG_PROMOTE_N:
            prom_offset = 0 if side == WHITE else 6
            if   flag == FLAG_PROMOTE_Q: p_prom = P_Q + prom_offset
            elif flag == FLAG_PROMOTE_R: p_prom = P_R + prom_offset
            elif flag == FLAG_PROMOTE_B: p_prom = P_B + prom_offset
            else:                         p_prom = P_N + prom_offset
            # XOR in the promoted piece at the destination square
            self.zobrist_key ^= ZOBRIST_PIECES[p_prom][to_sq]
            self._bb[p_prom] = cy_set_bit(self._bb[p_prom], to_sq)
            self.piece_map[to_sq] = p_prom
            self.halfmove_clock = 0
        else:
            # XOR in the moving piece at the destination square
            self.zobrist_key ^= ZOBRIST_PIECES[mp][to_sq]
            self._bb[mp] = cy_set_bit(self._bb[mp], to_sq)
            self.piece_map[to_sq] = mp
            if mp == P_P or mp == P_p:
                self.halfmove_clock = 0

        # Update castling rights
        if   mp == P_K:      self.castling_rights &= ~(WK | WQ)
        elif mp == P_k:      self.castling_rights &= ~(BK | BQ)
        if   from_sq == H1:  self.castling_rights &= ~WK
        elif from_sq == A1:  self.castling_rights &= ~WQ
        elif from_sq == H8:  self.castling_rights &= ~BK
        elif from_sq == A8:  self.castling_rights &= ~BQ
        if   to_sq == H1:    self.castling_rights &= ~WK
        elif to_sq == A1:    self.castling_rights &= ~WQ
        elif to_sq == H8:    self.castling_rights &= ~BK
        elif to_sq == A8:    self.castling_rights &= ~BQ

        # Incremental occupancy update
        self._occ[side] = cy_clear_bit(self._occ[side], from_sq)
        if cap != -1:
            self._occ[opp_side] = cy_clear_bit(self._occ[opp_side], to_sq)
        if flag == FLAG_EP:
            cap_ep_sq = to_sq - 8 if side == WHITE else to_sq + 8
            self._occ[opp_side] = cy_clear_bit(self._occ[opp_side], cap_ep_sq)
        elif flag == FLAG_CASTLE:
            if to_sq == G1:
                self._occ[0] = cy_clear_bit(self._occ[0], H1)
                self._occ[0] = cy_set_bit(self._occ[0], F1)
            elif to_sq == C1:
                self._occ[0] = cy_clear_bit(self._occ[0], A1)
                self._occ[0] = cy_set_bit(self._occ[0], D1)
            elif to_sq == G8:
                self._occ[1] = cy_clear_bit(self._occ[1], H8)
                self._occ[1] = cy_set_bit(self._occ[1], F8)
            elif to_sq == C8:
                self._occ[1] = cy_clear_bit(self._occ[1], A8)
                self._occ[1] = cy_set_bit(self._occ[1], D8)

        self._occ[side] = cy_set_bit(self._occ[side], to_sq)
        self._occ[2]    = self._occ[0] | self._occ[1]

        # Toggle side
        self.side_to_move = opp_side

        # --- Zobrist Rights and Side Updates ---
        # XOR out old castle rights, XOR in new
        if old_castle != self.castling_rights:
            self.zobrist_key ^= ZOBRIST_CASTLING[old_castle] ^ ZOBRIST_CASTLING[self.castling_rights]

        # XOR out old EP file, XOR in new
        if old_ep != 64:
            self.zobrist_key ^= ZOBRIST_EP[old_ep % 8]
        if self.en_passant_sq != 64:
            self.zobrist_key ^= ZOBRIST_EP[self.en_passant_sq % 8]

        # XOR side to move
        self.zobrist_key ^= ZOBRIST_SIDE

        # Legality check: was our king left in check?
        cdef int friendly_side = opp_side ^ 1
        cdef int king_piece_idx = 5 if friendly_side == 0 else 11
        cdef unsigned long long king_bb2 = self._bb[king_piece_idx]
        cdef int king_sq = cy_lsb(king_bb2)
        if king_sq != -1 and self.is_square_attacked(king_sq, opp_side):
            return False

        return True

    cpdef bint make_null_move(self):
        """Makes a null move (skips turn). Pushes state onto stack."""
        if self._history_len >= MAX_HISTORY:
            return False  # stack overflow guard

        # --- Save current state into C struct (no Python allocator) ---
        cdef CGameState *slot = &self._history[self._history_len]
        memcpy(slot.bitboards,    self._bb,       12 * sizeof(unsigned long long))
        memcpy(slot.occupancies,  self._occ,       3 * sizeof(unsigned long long))
        memcpy(slot.piece_map,    self.piece_map, 64 * sizeof(int))
        slot.side_to_move    = self.side_to_move
        slot.castling_rights = self.castling_rights
        slot.en_passant_sq   = self.en_passant_sq
        slot.halfmove_clock  = self.halfmove_clock
        slot.fullmove_number = self.fullmove_number
        slot.zobrist_key     = self.zobrist_key
        self._history_len += 1

        cdef int old_ep = self.en_passant_sq
        self.en_passant_sq = 64
        self.halfmove_clock += 1

        # XOR out old EP file
        if old_ep != 64:
            self.zobrist_key ^= ZOBRIST_EP[old_ep % 8]

        # Toggle side and XOR side
        self.side_to_move = BLACK if self.side_to_move == WHITE else WHITE
        self.zobrist_key ^= ZOBRIST_SIDE

        if self.side_to_move == WHITE:
            self.fullmove_number += 1

        return True

    cpdef void unmake_move(self):
        """Pops the C-stack to restore the previous board state (Python wrapper)."""
        self.unmake_move_c()

    cdef void unmake_move_c(self):
        """Pops the C-stack to restore the previous board state. O(1) allocation."""
        if self._history_len == 0:
            return
        self._history_len -= 1
        cdef CGameState *slot = &self._history[self._history_len]

        # Restore scalars
        self.side_to_move    = slot.side_to_move
        self.castling_rights = slot.castling_rights
        self.en_passant_sq   = slot.en_passant_sq
        self.halfmove_clock  = slot.halfmove_clock
        self.fullmove_number = slot.fullmove_number
        self.zobrist_key     = slot.zobrist_key

        # Restore piece_map via memcpy — 256-byte copy, entirely in C
        memcpy(self.piece_map, slot.piece_map, 64 * sizeof(int))

        # Restore bitboards and occupancies (C array)
        memcpy(self._bb,  slot.bitboards,   12 * sizeof(unsigned long long))
        memcpy(self._occ, slot.occupancies,  3 * sizeof(unsigned long long))

    def to_chess_move(self, int move):
        """Converts packed move to python-chess Move."""
        cdef int from_sq = cy_get_move_source(move)
        cdef int to_sq   = cy_get_move_dest(move)
        cdef int flag    = cy_get_move_flag(move)

        promotion = None
        if flag >= FLAG_PROMOTE_N:
            if   flag == FLAG_PROMOTE_Q: promotion = chess.QUEEN
            elif flag == FLAG_PROMOTE_R: promotion = chess.ROOK
            elif flag == FLAG_PROMOTE_B: promotion = chess.BISHOP
            else:                         promotion = chess.KNIGHT

        return chess.Move(from_sq, to_sq, promotion=promotion)

    def print_board(self):
        """Prints the board in ASCII."""
        piece_symbols = ["P","N","B","R","Q","K","p","n","b","r","q","k"]
        print("-" * 17)
        for r in range(7, -1, -1):
            row_chars = []
            for f in range(8):
                sq = r * 8 + f
                p_idx = self.piece_map[sq]
                row_chars.append(piece_symbols[p_idx] if p_idx != -1 else ".")
            print(" ".join(row_chars))
        print("-" * 17)

    @property
    def history(self):
        """Exposes history depth for compatibility (read-only)."""
        return list(range(self._history_len))  # length proxy, not the real structs

    cpdef long long run_perft_recursive(self, int depth):
        """Runs perft recursion fully in Cython."""
        cdef CMoveList moves
        moves.count = 0
        self._generate_pseudo_legal_moves_c(&moves)
        cdef int i, m
        cdef long long nodes = 0

        if depth == 1:
            for i in range(moves.count):
                m = moves.moves[i]
                if self.make_move_c(m):
                    nodes += 1
                self.unmake_move_c()
            return nodes

        for i in range(moves.count):
            m = moves.moves[i]
            if self.make_move_c(m):
                nodes += self.run_perft_recursive(depth - 1)
            self.unmake_move_c()
        return nodes
