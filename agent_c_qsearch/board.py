"""Board representation and move generation using custom Bitboards."""

from __future__ import annotations
import chess

# Constants for square indices (A1=0, B1=1, ..., H8=63)
A1, B1, C1, D1, E1, F1, G1, H1 = range(8)
A2, B2, C2, D2, E2, F2, G2, H2 = range(8, 16)
A3, B3, C3, D3, E3, F3, G3, H3 = range(16, 24)
A4, B4, C4, D4, E4, F4, G4, H4 = range(24, 32)
A5, B5, C5, D5, E5, F5, G5, H5 = range(32, 40)
A6, B6, C6, D6, E6, F6, G6, H6 = range(40, 48)
A7, B7, C7, D7, E7, F7, G7, H7 = range(48, 56)
A8, B8, C8, D8, E8, F8, G8, H8 = range(56, 64)

# Colors
WHITE = 0
BLACK = 1

# Piece types
P_P = 0  # White Pawn
P_N = 1  # White Knight
P_B = 2  # White Bishop
P_R = 3  # White Rook
P_Q = 4  # White Queen
P_K = 5  # White King
P_p = 6  # Black Pawn  # pylint: disable=invalid-name
P_n = 7  # Black Knight  # pylint: disable=invalid-name
P_b = 8  # Black Bishop  # pylint: disable=invalid-name
P_r = 9  # Black Rook  # pylint: disable=invalid-name
P_q = 10 # Black Queen  # pylint: disable=invalid-name
P_k = 11 # Black King  # pylint: disable=invalid-name

# Castling rights
WK = 1
WQ = 2
BK = 4
BQ = 8

# Move flags
FLAG_NORMAL = 0
FLAG_DOUBLE_PUSH = 1
FLAG_CASTLE = 2
FLAG_EP = 3
FLAG_PROMOTE_N = 8
FLAG_PROMOTE_B = 9
FLAG_PROMOTE_R = 10
FLAG_PROMOTE_Q = 11

# Magic Numbers and Relevant Bits from Stockfish/Tord Romstad
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
    0x4010011029020020
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
    0x1004081002402
]

BISHOP_RELEVANT_BITS = [
    6, 5, 5, 5, 5, 5, 5, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 5, 5, 5, 5, 5, 5, 6
]

ROOK_RELEVANT_BITS = [
    12, 11, 11, 11, 11, 11, 11, 12,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    12, 11, 11, 11, 11, 11, 11, 12
]

# --- Bitboard Helpers ---
def lsb(bb: int) -> int:
    """Returns the index of the least significant bit (0-63). Returns -1 if 0."""
    if bb == 0:
        return -1
    return (bb & -bb).bit_length() - 1

def pop_count(bb: int) -> int:
    """Returns the number of set bits (1s) in the bitboard using Python's native popcount."""
    return bb.bit_count()

def clear_bit(bb: int, sq: int) -> int:
    """Clears the bit at the given square index (sets it to 0)."""
    return bb & ~(1 << sq)

def set_bit(bb: int, sq: int) -> int:
    """Sets the bit at the given square index to 1."""
    return bb | (1 << sq)

def get_bit(bb: int, sq: int) -> int:
    """Returns the value of the bit at the given square index (0 or 1)."""
    return (bb >> sq) & 1


# --- Sliding Attacks Generation (Masks and On-The-Fly) ---
def mask_bishop_attacks(square: int) -> int:
    """Generates attack mask for a bishop at a given square, excluding edges."""
    attacks = 0
    tr, tf = square // 8, square % 8
    for r, f in zip(range(tr + 1, 7), range(tf + 1, 7)):
        attacks |= (1 << (r * 8 + f))
    for r, f in zip(range(tr + 1, 7), range(tf - 1, 0, -1)):
        attacks |= (1 << (r * 8 + f))
    for r, f in zip(range(tr - 1, 0, -1), range(tf + 1, 7)):
        attacks |= (1 << (r * 8 + f))
    for r, f in zip(range(tr - 1, 0, -1), range(tf - 1, 0, -1)):
        attacks |= (1 << (r * 8 + f))
    return attacks

def mask_rook_attacks(square: int) -> int:
    """Generates attack mask for a rook at a given square, excluding edges."""
    attacks = 0
    tr, tf = square // 8, square % 8
    for r in range(tr + 1, 7):
        attacks |= (1 << (r * 8 + tf))
    for r in range(tr - 1, 0, -1):
        attacks |= (1 << (r * 8 + tf))
    for f in range(tf + 1, 7):
        attacks |= (1 << (tr * 8 + f))
    for f in range(tf - 1, 0, -1):
        attacks |= (1 << (tr * 8 + f))
    return attacks

def bishop_attacks_on_the_fly(square: int, block: int) -> int:
    """Generates bishop attacks on-the-fly, considering blocking pieces."""
    attacks = 0
    tr, tf = square // 8, square % 8
    for r, f in zip(range(tr + 1, 8), range(tf + 1, 8)):
        attacks |= (1 << (r * 8 + f))
        if (1 << (r * 8 + f)) & block:
            break
    for r, f in zip(range(tr + 1, 8), range(tf - 1, -1, -1)):
        attacks |= (1 << (r * 8 + f))
        if (1 << (r * 8 + f)) & block:
            break
    for r, f in zip(range(tr - 1, -1, -1), range(tf + 1, 8)):
        attacks |= (1 << (r * 8 + f))
        if (1 << (r * 8 + f)) & block:
            break
    for r, f in zip(range(tr - 1, -1, -1), range(tf - 1, -1, -1)):
        attacks |= (1 << (r * 8 + f))
        if (1 << (r * 8 + f)) & block:
            break
    return attacks

def rook_attacks_on_the_fly(square: int, block: int) -> int:
    """Generates rook attacks on-the-fly, considering blocking pieces."""
    attacks = 0
    tr, tf = square // 8, square % 8
    for r in range(tr + 1, 8):
        attacks |= (1 << (r * 8 + tf))
        if (1 << (r * 8 + tf)) & block:
            break
    for r in range(tr - 1, -1, -1):
        attacks |= (1 << (r * 8 + tf))
        if (1 << (r * 8 + tf)) & block:
            break
    for f in range(tf + 1, 8):
        attacks |= (1 << (tr * 8 + f))
        if (1 << (tr * 8 + f)) & block:
            break
    for f in range(tf - 1, -1, -1):
        attacks |= (1 << (tr * 8 + f))
        if (1 << (tr * 8 + f)) & block:
            break
    return attacks

def set_occupancy(index: int, bits_in_mask: int, attack_mask: int) -> int:
    """Configures occupancy bitboard based on an index and attack mask."""
    occupancy = 0
    temp_mask = attack_mask
    for i in range(bits_in_mask):
        square = lsb(temp_mask)
        temp_mask = clear_bit(temp_mask, square)
        if index & (1 << i):
            occupancy = set_bit(occupancy, square)
    return occupancy


# --- Magic Bishop/Rook Masks Precomputed ---
BISHOP_MASKS = [mask_bishop_attacks(sq) for sq in range(64)]
ROOK_MASKS = [mask_rook_attacks(sq) for sq in range(64)]


# --- Precomputed Attack Tables ---
KNIGHT_ATTACKS = [0] * 64
KING_ATTACKS = [0] * 64
PAWN_ATTACKS = [[0] * 64 for _ in range(2)]
BISHOP_ATTACKS = [[0] * 512 for _ in range(64)]
ROOK_ATTACKS = [[0] * 4096 for _ in range(64)]


# pylint: disable=too-many-locals, too-many-branches
def _init_attack_tables() -> None:
    """Initializes the precomputed non-sliding and sliding attack tables."""
    # Populate non-sliding attack tables
    for sq_idx in range(64):
        r_coord = sq_idx // 8
        f_coord = sq_idx % 8

        # Knight attacks
        k_att = 0
        for dr, df in [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]:
            nr_coord = r_coord + dr
            nf_coord = f_coord + df
            if 0 <= nr_coord < 8 and 0 <= nf_coord < 8:
                k_att |= (1 << (nr_coord * 8 + nf_coord))
        KNIGHT_ATTACKS[sq_idx] = k_att

        # King attacks
        king_att = 0
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                if dr == 0 and df == 0:
                    continue
                nr_coord = r_coord + dr
                nf_coord = f_coord + df
                if 0 <= nr_coord < 8 and 0 <= nf_coord < 8:
                    king_att |= (1 << (nr_coord * 8 + nf_coord))
        KING_ATTACKS[sq_idx] = king_att

        # Pawn attacks
        # White pawns (capturing up-left and up-right)
        wp_att = 0
        if f_coord > 0 and sq_idx + 7 < 64:
            wp_att |= (1 << (sq_idx + 7))
        if f_coord < 7 and sq_idx + 9 < 64:
            wp_att |= (1 << (sq_idx + 9))
        PAWN_ATTACKS[WHITE][sq_idx] = wp_att

        # Black pawns (capturing down-left and down-right)
        bp_att = 0
        if f_coord > 0 and sq_idx - 9 >= 0:
            bp_att |= (1 << (sq_idx - 9))
        if f_coord < 7 and sq_idx - 7 >= 0:
            bp_att |= (1 << (sq_idx - 7))
        PAWN_ATTACKS[BLACK][sq_idx] = bp_att

    for sq_idx in range(64):
        # Bishop lookup table initialization
        b_mask = BISHOP_MASKS[sq_idx]
        b_bits = BISHOP_RELEVANT_BITS[sq_idx]
        b_magic = BISHOP_MAGICS[sq_idx]
        for i_val in range(1 << b_bits):
            occ = set_occupancy(i_val, b_bits, b_mask)
            idx_val = ((occ * b_magic) & 0xFFFFFFFFFFFFFFFF) >> (64 - b_bits)
            BISHOP_ATTACKS[sq_idx][idx_val] = bishop_attacks_on_the_fly(sq_idx, occ)

        # Rook lookup table initialization
        r_mask = ROOK_MASKS[sq_idx]
        r_bits = ROOK_RELEVANT_BITS[sq_idx]
        r_magic = ROOK_MAGICS[sq_idx]
        for i_val in range(1 << r_bits):
            occ = set_occupancy(i_val, r_bits, r_mask)
            idx_val = ((occ * r_magic) & 0xFFFFFFFFFFFFFFFF) >> (64 - r_bits)
            ROOK_ATTACKS[sq_idx][idx_val] = rook_attacks_on_the_fly(sq_idx, occ)


_init_attack_tables()


def get_bishop_attacks(square: int, occupancy: int) -> int:
    """Retrieves bishop attacks using magic bitboards."""
    masked_occ = occupancy & BISHOP_MASKS[square]
    magic = BISHOP_MAGICS[square]
    bits = BISHOP_RELEVANT_BITS[square]
    idx = ((masked_occ * magic) & 0xFFFFFFFFFFFFFFFF) >> (64 - bits)
    return BISHOP_ATTACKS[square][idx]

def get_rook_attacks(square: int, occupancy: int) -> int:
    """Retrieves rook attacks using magic bitboards."""
    masked_occ = occupancy & ROOK_MASKS[square]
    magic = ROOK_MAGICS[square]
    bits = ROOK_RELEVANT_BITS[square]
    idx = ((masked_occ * magic) & 0xFFFFFFFFFFFFFFFF) >> (64 - bits)
    return ROOK_ATTACKS[square][idx]

def get_queen_attacks(square: int, occupancy: int) -> int:
    """Retrieves queen attacks (combination of bishop and rook attacks)."""
    return get_bishop_attacks(square, occupancy) | get_rook_attacks(square, occupancy)


# --- Move Packing / Unpacking (16-bit) ---
# bits 0-5  : source square (0-63)
# bits 6-11 : destination square (0-63)
# bits 12-15: move flags
def encode_move(from_sq: int, to_sq: int, flag: int) -> int:
    """Encodes move details (source, destination, flags) into a 16-bit int."""
    return from_sq | (to_sq << 6) | (flag << 12)

def get_move_source(move: int) -> int:
    """Extracts source square from the encoded 16-bit move."""
    return move & 0x3F

def get_move_dest(move: int) -> int:
    """Extracts destination square from the encoded 16-bit move."""
    return (move >> 6) & 0x3F

def get_move_flag(move: int) -> int:
    """Extracts move flag from the encoded 16-bit move."""
    return (move >> 12) & 0x0F


# --- Zobrist Hashing Constants ---
ZOBRIST_PIECES = [[0] * 64 for _ in range(12)]
ZOBRIST_CASTLING = [0] * 16
ZOBRIST_EP = [0] * 8
ZOBRIST_SIDE = 0

def _init_zobrist_py():
    global ZOBRIST_SIDE
    state = 1070372
    MASK = 0xFFFFFFFFFFFFFFFF
    
    def next_rand(s):
        s = (s ^ ((s << 13) & MASK)) & MASK
        s = (s ^ (s >> 7)) & MASK
        s = (s ^ ((s << 17) & MASK)) & MASK
        return s

    for i in range(12):
        for j in range(64):
            state = next_rand(state)
            ZOBRIST_PIECES[i][j] = state
    for i in range(16):
        state = next_rand(state)
        ZOBRIST_CASTLING[i] = state
    for i in range(8):
        state = next_rand(state)
        ZOBRIST_EP[i] = state
    state = next_rand(state)
    ZOBRIST_SIDE = state

_init_zobrist_py()


# --- Saved Game State for Unmaking Moves ---
# pylint: disable=too-few-public-methods, too-many-instance-attributes
class GameState:
    """Saves the relevant chess board state fields to support unmaking moves."""

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        bitboards: list[int],
        occupancies: list[int],
        side_to_move: int,
        castling_rights: int,
        en_passant_sq: int,
        halfmove_clock: int,
        fullmove_number: int,
        piece_map: list[int | None],
        zobrist_key: int = 0,
    ) -> None:
        self.bitboards = list(bitboards)
        self.occupancies = list(occupancies)
        self.side_to_move = side_to_move
        self.castling_rights = castling_rights
        self.en_passant_sq = en_passant_sq
        self.halfmove_clock = halfmove_clock
        self.fullmove_number = fullmove_number
        self.piece_map = list(piece_map)
        self.zobrist_key = zobrist_key


# --- Board Representation ---
# pylint: disable=too-many-instance-attributes
class CustomBitboardBoard:
    """A custom high-performance chess board representation using 64-bit integers."""

    def __init__(self) -> None:
        self.bitboards = [0] * 12
        self.occupancies = [0] * 3
        self.side_to_move = WHITE
        self.castling_rights = 0
        self.en_passant_sq = 64  # 64 means no en passant square
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.history: list[GameState] = []
        self.piece_map: list[int | None] = [None] * 64
        self.zobrist_key = 0

    def _recompute_zobrist(self) -> int:
        key = 0
        for sq in range(64):
            p = self.piece_map[sq]
            if p is not None:
                key ^= ZOBRIST_PIECES[p][sq]
        if self.side_to_move == BLACK:
            key ^= ZOBRIST_SIDE
        key ^= ZOBRIST_CASTLING[self.castling_rights]
        if self.en_passant_sq != 64:
            key ^= ZOBRIST_EP[self.en_passant_sq % 8]
        return key

    @classmethod
    def from_chess_board(cls, chess_board: chess.Board) -> CustomBitboardBoard:
        """Converts a python-chess Board to CustomBitboardBoard."""
        board = cls()
        board.side_to_move = WHITE if chess_board.turn == chess.WHITE else BLACK

        # Castling rights
        castle = 0
        if chess_board.has_kingside_castling_rights(chess.WHITE):
            castle |= WK
        if chess_board.has_queenside_castling_rights(chess.WHITE):
            castle |= WQ
        if chess_board.has_kingside_castling_rights(chess.BLACK):
            castle |= BK
        if chess_board.has_queenside_castling_rights(chess.BLACK):
            castle |= BQ
        board.castling_rights = castle

        # En passant
        board.en_passant_sq = (
            chess_board.ep_square
            if chess_board.ep_square is not None
            else 64
        )
        board.halfmove_clock = chess_board.halfmove_clock
        board.fullmove_number = chess_board.fullmove_number

        # Piece bitboards
        # White: P, N, B, R, Q, K (0-5)
        # Black: p, n, b, r, q, k (6-11)
        for square in range(64):
            piece = chess_board.piece_at(square)
            if piece is not None:
                color_offset = 0 if piece.color == chess.WHITE else 6
                p_type = piece.piece_type
                # Map python-chess piece types to 0-5 index
                idx = p_type - 1 + color_offset
                board.bitboards[idx] = set_bit(board.bitboards[idx], square)
                board.piece_map[square] = idx

        # Populate occupancies
        for i in range(6):
            board.occupancies[WHITE] |= board.bitboards[i]
            board.occupancies[BLACK] |= board.bitboards[i + 6]
        board.occupancies[2] = (
            board.occupancies[WHITE] | board.occupancies[BLACK]
        )

        board.zobrist_key = board._recompute_zobrist()
        return board

    def get_piece_at(self, square: int) -> int | None:
        """Returns the piece type/color index at the square (0-11), or None if empty (O(1))."""
        return self.piece_map[square]

    # pylint: disable=too-many-return-statements
    def is_square_attacked(self, sq: int, attacker_color: int) -> bool:
        """Checks if a square is attacked by any piece of attacker_color."""
        occupancy = self.occupancies[2]

        # 1. Attacked by pawns
        if attacker_color == WHITE:
            pawn_bb = self.bitboards[P_P]
            f = sq % 8
            if f > 0 and sq - 9 >= 0 and get_bit(pawn_bb, sq - 9):
                return True
            if f < 7 and sq - 7 >= 0 and get_bit(pawn_bb, sq - 7):
                return True
        else:
            pawn_bb = self.bitboards[P_p]
            f = sq % 8
            if f > 0 and sq + 7 < 64 and get_bit(pawn_bb, sq + 7):
                return True
            if f < 7 and sq + 9 < 64 and get_bit(pawn_bb, sq + 9):
                return True

        # 2. Attacked by knights
        knight_bb = self.bitboards[P_N if attacker_color == WHITE else P_n]
        if KNIGHT_ATTACKS[sq] & knight_bb:
            return True

        # 3. Attacked by king
        king_bb = self.bitboards[P_K if attacker_color == WHITE else P_k]
        if KING_ATTACKS[sq] & king_bb:
            return True

        # 4. Attacked by bishops or queens (diagonals)
        friendly_offset = 0 if attacker_color == WHITE else 6
        bishop_queen_bb = (
            self.bitboards[P_B + friendly_offset] |
            self.bitboards[P_Q + friendly_offset]
        )
        if get_bishop_attacks(sq, occupancy) & bishop_queen_bb:
            return True

        # 5. Attacked by rooks or queens (orthogonals)
        rook_queen_bb = (
            self.bitboards[P_R + friendly_offset] |
            self.bitboards[P_Q + friendly_offset]
        )
        if get_rook_attacks(sq, occupancy) & rook_queen_bb:
            return True

        return False

    def in_check(self) -> bool:
        """Returns True if the side to move's king is in check."""
        king_bb = self.bitboards[P_K if self.side_to_move == WHITE else P_k]
        king_sq = lsb(king_bb)
        if king_sq == -1:
            return False
        return self.is_square_attacked(
            king_sq,
            BLACK if self.side_to_move == WHITE else WHITE
        )

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    # pylint: disable=too-many-nested-blocks
    def generate_pseudo_legal_moves(self) -> list[int]:
        """Generates pseudo-legal moves for the current side to move."""
        moves = []
        side = self.side_to_move
        friendly_occ = self.occupancies[WHITE if side == WHITE else BLACK]
        opponent_occ = self.occupancies[BLACK if side == WHITE else WHITE]
        both_occ = self.occupancies[2]

        # White pieces: 0-5; Black pieces: 6-11
        offset = 0 if side == WHITE else 6
        pawn_bb = self.bitboards[P_P + offset]
        knight_bb = self.bitboards[P_N + offset]
        bishop_bb = self.bitboards[P_B + offset]
        rook_bb = self.bitboards[P_R + offset]
        queen_bb = self.bitboards[P_Q + offset]
        king_bb = self.bitboards[P_K + offset]

        # --- Pawn Moves ---
        pawn_sqs = pawn_bb
        while pawn_sqs:
            from_sq = lsb(pawn_sqs)
            pawn_sqs = clear_bit(pawn_sqs, from_sq)
            r, f = from_sq // 8, from_sq % 8

            # White pawns
            if side == WHITE:
                to_sq = from_sq + 8
                # 1. Single Push
                if to_sq < 64 and not get_bit(both_occ, to_sq):
                    # Promotion check
                    if r == 6:
                        for flag in [
                            FLAG_PROMOTE_Q, FLAG_PROMOTE_R,
                            FLAG_PROMOTE_B, FLAG_PROMOTE_N
                        ]:
                            moves.append(encode_move(from_sq, to_sq, flag))
                    else:
                        moves.append(encode_move(from_sq, to_sq, FLAG_NORMAL))

                    # 2. Double Push
                    to_sq2 = from_sq + 16
                    if r == 1 and not get_bit(both_occ, to_sq2):
                        moves.append(encode_move(from_sq, to_sq2, FLAG_DOUBLE_PUSH))

                # 3. Pawn Captures
                # Diagonals
                for dest_sq, cond in [(from_sq + 7, f > 0), (from_sq + 9, f < 7)]:
                    if cond and dest_sq < 64:
                        if get_bit(opponent_occ, dest_sq):
                            if r == 6:
                                for flag in [
                                    FLAG_PROMOTE_Q, FLAG_PROMOTE_R,
                                    FLAG_PROMOTE_B, FLAG_PROMOTE_N
                                ]:
                                    moves.append(encode_move(from_sq, dest_sq, flag))
                            else:
                                moves.append(encode_move(from_sq, dest_sq, FLAG_NORMAL))
                        elif dest_sq == self.en_passant_sq:
                            moves.append(encode_move(from_sq, dest_sq, FLAG_EP))

            # Black pawns
            else:
                to_sq = from_sq - 8
                # 1. Single Push
                if to_sq >= 0 and not get_bit(both_occ, to_sq):
                    if r == 1:
                        for flag in [
                            FLAG_PROMOTE_Q, FLAG_PROMOTE_R,
                            FLAG_PROMOTE_B, FLAG_PROMOTE_N
                        ]:
                            moves.append(encode_move(from_sq, to_sq, flag))
                    else:
                        moves.append(encode_move(from_sq, to_sq, FLAG_NORMAL))

                    # 2. Double Push
                    to_sq2 = from_sq - 16
                    if r == 6 and not get_bit(both_occ, to_sq2):
                        moves.append(encode_move(from_sq, to_sq2, FLAG_DOUBLE_PUSH))

                # 3. Pawn Captures
                for dest_sq, cond in [(from_sq - 9, f > 0), (from_sq - 7, f < 7)]:
                    if cond and dest_sq >= 0:
                        if get_bit(opponent_occ, dest_sq):
                            if r == 1:
                                for flag in [
                                    FLAG_PROMOTE_Q, FLAG_PROMOTE_R,
                                    FLAG_PROMOTE_B, FLAG_PROMOTE_N
                                ]:
                                    moves.append(encode_move(from_sq, dest_sq, flag))
                            else:
                                moves.append(encode_move(from_sq, dest_sq, FLAG_NORMAL))
                        elif dest_sq == self.en_passant_sq:
                            moves.append(encode_move(from_sq, dest_sq, FLAG_EP))

        # --- Knight Moves ---
        knights = knight_bb
        while knights:
            from_sq = lsb(knights)
            knights = clear_bit(knights, from_sq)
            dest_sqs = KNIGHT_ATTACKS[from_sq] & ~friendly_occ
            while dest_sqs:
                to_sq = lsb(dest_sqs)
                dest_sqs = clear_bit(dest_sqs, to_sq)
                moves.append(encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- Bishop Moves ---
        bishops = bishop_bb
        while bishops:
            from_sq = lsb(bishops)
            bishops = clear_bit(bishops, from_sq)
            dest_sqs = get_bishop_attacks(from_sq, both_occ) & ~friendly_occ
            while dest_sqs:
                to_sq = lsb(dest_sqs)
                dest_sqs = clear_bit(dest_sqs, to_sq)
                moves.append(encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- Rook Moves ---
        rooks = rook_bb
        while rooks:
            from_sq = lsb(rooks)
            rooks = clear_bit(rooks, from_sq)
            dest_sqs = get_rook_attacks(from_sq, both_occ) & ~friendly_occ
            while dest_sqs:
                to_sq = lsb(dest_sqs)
                dest_sqs = clear_bit(dest_sqs, to_sq)
                moves.append(encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- Queen Moves ---
        queens = queen_bb
        while queens:
            from_sq = lsb(queens)
            queens = clear_bit(queens, from_sq)
            dest_sqs = get_queen_attacks(from_sq, both_occ) & ~friendly_occ
            while dest_sqs:
                to_sq = lsb(dest_sqs)
                dest_sqs = clear_bit(dest_sqs, to_sq)
                moves.append(encode_move(from_sq, to_sq, FLAG_NORMAL))

        # --- King Moves ---
        king = king_bb
        if king:
            from_sq = lsb(king)
            dest_sqs = KING_ATTACKS[from_sq] & ~friendly_occ
            while dest_sqs:
                to_sq = lsb(dest_sqs)
                dest_sqs = clear_bit(dest_sqs, to_sq)
                moves.append(encode_move(from_sq, to_sq, FLAG_NORMAL))

            # --- Castling Moves ---
            if side == WHITE:
                # King side (WK)
                if (
                    (self.castling_rights & WK)
                    and not get_bit(both_occ, F1)
                    and not get_bit(both_occ, G1)
                ):
                    # King cannot pass through check
                    if (
                        not self.is_square_attacked(E1, BLACK)
                        and not self.is_square_attacked(F1, BLACK)
                    ):
                        moves.append(encode_move(E1, G1, FLAG_CASTLE))
                # Queen side (WQ)
                if (
                    (self.castling_rights & WQ)
                    and not get_bit(both_occ, D1)
                    and not get_bit(both_occ, C1)
                    and not get_bit(both_occ, B1)
                ):
                    if (
                        not self.is_square_attacked(E1, BLACK)
                        and not self.is_square_attacked(D1, BLACK)
                    ):
                        moves.append(encode_move(E1, C1, FLAG_CASTLE))
            else:
                # King side (BK)
                if (
                    (self.castling_rights & BK)
                    and not get_bit(both_occ, F8)
                    and not get_bit(both_occ, G8)
                ):
                    if (
                        not self.is_square_attacked(E8, WHITE)
                        and not self.is_square_attacked(F8, WHITE)
                    ):
                        moves.append(encode_move(E8, G8, FLAG_CASTLE))
                # Queen side (BQ)
                if (
                    (self.castling_rights & BQ)
                    and not get_bit(both_occ, D8)
                    and not get_bit(both_occ, C8)
                    and not get_bit(both_occ, B8)
                ):
                    if (
                        not self.is_square_attacked(E8, WHITE)
                        and not self.is_square_attacked(D8, WHITE)
                    ):
                        moves.append(encode_move(E8, C8, FLAG_CASTLE))

        return moves

    def generate_legal_moves(self) -> list[int]:
        """Generates strictly legal moves by filtering pseudo-legal moves."""
        legal = []
        pseudo = self.generate_pseudo_legal_moves()
        for m in pseudo:
            if self.make_move(m):
                legal.append(m)
            self.unmake_move()
        return legal

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def make_move(self, move: int) -> bool:
        """Makes a move on the board."""
        # Save state to history for unmake
        self.history.append(GameState(
            self.bitboards, self.occupancies, self.side_to_move,
            self.castling_rights, self.en_passant_sq, self.halfmove_clock,
            self.fullmove_number, self.piece_map, self.zobrist_key
        ))

        from_sq = get_move_source(move)
        to_sq = get_move_dest(move)
        flag = get_move_flag(move)
        side = self.side_to_move
        opp_side = BLACK if side == WHITE else WHITE

        # Identify piece being moved using piece_map
        moving_p_idx = self.piece_map[from_sq]
        if moving_p_idx is None:
            return False  # Empty square moved from

        old_castle = self.castling_rights
        old_ep = self.en_passant_sq

        # XOR out the moving piece from the source square
        self.zobrist_key ^= ZOBRIST_PIECES[moving_p_idx][from_sq]

        # Remove moving piece from source
        self.bitboards[moving_p_idx] = clear_bit(self.bitboards[moving_p_idx], from_sq)
        self.piece_map[from_sq] = None

        # Capture logic: check if there is an opponent piece at destination using piece_map
        captured_p_idx = self.piece_map[to_sq]

        # Default flags reset
        self.en_passant_sq = 64
        self.halfmove_clock += 1
        if side == BLACK:
            self.fullmove_number += 1

        if captured_p_idx is not None:
            # XOR out the captured piece from the destination square
            self.zobrist_key ^= ZOBRIST_PIECES[captured_p_idx][to_sq]
            # Remove captured piece from target
            self.bitboards[captured_p_idx] = clear_bit(self.bitboards[captured_p_idx], to_sq)
            self.piece_map[to_sq] = None
            self.halfmove_clock = 0  # Capture resets halfmove clock

        # Pawn double push sets EP square
        if flag == FLAG_DOUBLE_PUSH:
            self.en_passant_sq = from_sq + 8 if side == WHITE else from_sq - 8
            self.halfmove_clock = 0

        # En Passant capture logic
        elif flag == FLAG_EP:
            captured_ep_sq = to_sq - 8 if side == WHITE else to_sq + 8
            ep_pawn_idx = P_p if side == WHITE else P_P
            # XOR out the captured en passant pawn
            self.zobrist_key ^= ZOBRIST_PIECES[ep_pawn_idx][captured_ep_sq]
            self.bitboards[ep_pawn_idx] = clear_bit(self.bitboards[ep_pawn_idx], captured_ep_sq)
            self.piece_map[captured_ep_sq] = None
            self.halfmove_clock = 0

        # Castling rook movement
        elif flag == FLAG_CASTLE:
            if to_sq == G1:
                self.zobrist_key ^= ZOBRIST_PIECES[P_R][H1] ^ ZOBRIST_PIECES[P_R][F1]
                self.bitboards[P_R] = clear_bit(self.bitboards[P_R], H1)
                self.bitboards[P_R] = set_bit(self.bitboards[P_R], F1)
                self.piece_map[H1] = None
                self.piece_map[F1] = P_R
            elif to_sq == C1:
                self.zobrist_key ^= ZOBRIST_PIECES[P_R][A1] ^ ZOBRIST_PIECES[P_R][D1]
                self.bitboards[P_R] = clear_bit(self.bitboards[P_R], A1)
                self.bitboards[P_R] = set_bit(self.bitboards[P_R], D1)
                self.piece_map[A1] = None
                self.piece_map[D1] = P_R
            elif to_sq == G8:
                self.zobrist_key ^= ZOBRIST_PIECES[P_r][H8] ^ ZOBRIST_PIECES[P_r][F8]
                self.bitboards[P_r] = clear_bit(self.bitboards[P_r], H8)
                self.bitboards[P_r] = set_bit(self.bitboards[P_r], F8)
                self.piece_map[H8] = None
                self.piece_map[F8] = P_r
            elif to_sq == C8:
                self.zobrist_key ^= ZOBRIST_PIECES[P_r][A8] ^ ZOBRIST_PIECES[P_r][D8]
                self.bitboards[P_r] = clear_bit(self.bitboards[P_r], A8)
                self.bitboards[P_r] = set_bit(self.bitboards[P_r], D8)
                self.piece_map[A8] = None
                self.piece_map[D8] = P_r

        # Move piece to destination (with promotion check)
        if flag >= FLAG_PROMOTE_N:
            # Map promotion flag to piece type index
            promoted_offset = 0 if side == WHITE else 6
            if flag == FLAG_PROMOTE_Q:
                p_prom = P_Q + promoted_offset
            elif flag == FLAG_PROMOTE_R:
                p_prom = P_R + promoted_offset
            elif flag == FLAG_PROMOTE_B:
                p_prom = P_B + promoted_offset
            else:
                p_prom = P_N + promoted_offset
            # XOR in the promoted piece at destination
            self.zobrist_key ^= ZOBRIST_PIECES[p_prom][to_sq]
            self.bitboards[p_prom] = set_bit(self.bitboards[p_prom], to_sq)
            self.piece_map[to_sq] = p_prom
            self.halfmove_clock = 0
        else:
            # XOR in the moving piece at destination
            self.zobrist_key ^= ZOBRIST_PIECES[moving_p_idx][to_sq]
            self.bitboards[moving_p_idx] = set_bit(self.bitboards[moving_p_idx], to_sq)
            self.piece_map[to_sq] = moving_p_idx
            if moving_p_idx in [P_P, P_p]:
                self.halfmove_clock = 0

        # --- Update Castling Rights ---
        # If king moves or rooks are captured/moved, update rights
        if moving_p_idx == P_K:
            self.castling_rights &= ~(WK | WQ)
        elif moving_p_idx == P_k:
            self.castling_rights &= ~(BK | BQ)

        # Rook moves
        if from_sq == H1:
            self.castling_rights &= ~WK
        elif from_sq == A1:
            self.castling_rights &= ~WQ
        elif from_sq == H8:
            self.castling_rights &= ~BK
        elif from_sq == A8:
            self.castling_rights &= ~BQ

        # Rook captures
        if to_sq == H1:
            self.castling_rights &= ~WK
        elif to_sq == A1:
            self.castling_rights &= ~WQ
        elif to_sq == H8:
            self.castling_rights &= ~BK
        elif to_sq == A8:
            self.castling_rights &= ~BQ

        # --- Incremental update of occupancies ---
        # Remove moving piece from from_sq
        self.occupancies[side] = clear_bit(self.occupancies[side], from_sq)

        # If there was a capture, remove it from opp_side
        if captured_p_idx is not None:
            self.occupancies[opp_side] = clear_bit(self.occupancies[opp_side], to_sq)

        # En Passant capture: remove captured pawn from captured_ep_sq
        if flag == FLAG_EP:
            captured_ep_sq = to_sq - 8 if side == WHITE else to_sq + 8
            self.occupancies[opp_side] = clear_bit(self.occupancies[opp_side], captured_ep_sq)

        # Castling: update rook's occupancy
        elif flag == FLAG_CASTLE:
            if to_sq == G1:
                self.occupancies[WHITE] = clear_bit(self.occupancies[WHITE], H1)
                self.occupancies[WHITE] = set_bit(self.occupancies[WHITE], F1)
            elif to_sq == C1:
                self.occupancies[WHITE] = clear_bit(self.occupancies[WHITE], A1)
                self.occupancies[WHITE] = set_bit(self.occupancies[WHITE], D1)
            elif to_sq == G8:
                self.occupancies[BLACK] = clear_bit(self.occupancies[BLACK], H8)
                self.occupancies[BLACK] = set_bit(self.occupancies[BLACK], F8)
            elif to_sq == C8:
                self.occupancies[BLACK] = clear_bit(self.occupancies[BLACK], A8)
                self.occupancies[BLACK] = set_bit(self.occupancies[BLACK], D8)

        # Set moving/promoted piece at to_sq
        self.occupancies[side] = set_bit(self.occupancies[side], to_sq)

        # Combine white and black occupancies
        self.occupancies[2] = self.occupancies[WHITE] | self.occupancies[BLACK]

        # Toggle turn
        self.side_to_move = opp_side

        # --- Zobrist Rights and Side Updates ---
        if old_castle != self.castling_rights:
            self.zobrist_key ^= ZOBRIST_CASTLING[old_castle] ^ ZOBRIST_CASTLING[self.castling_rights]

        if old_ep != 64:
            self.zobrist_key ^= ZOBRIST_EP[old_ep % 8]
        if self.en_passant_sq != 64:
            self.zobrist_key ^= ZOBRIST_EP[self.en_passant_sq % 8]

        self.zobrist_key ^= ZOBRIST_SIDE

        # Legality check
        friendly_side = opp_side ^ 1
        king_bb = self.bitboards[P_K if friendly_side == WHITE else P_k]
        king_sq = lsb(king_bb)
        if king_sq != -1 and self.is_square_attacked(king_sq, opp_side):
            return False

        return True

    def unmake_move(self) -> None:
        """Restores the board to the previous state from history."""
        if not self.history:
            return
        state = self.history.pop()
        self.bitboards = state.bitboards
        self.occupancies = state.occupancies
        self.side_to_move = state.side_to_move
        self.castling_rights = state.castling_rights
        self.en_passant_sq = state.en_passant_sq
        self.halfmove_clock = state.halfmove_clock
        self.fullmove_number = state.fullmove_number
        self.piece_map = state.piece_map
        self.zobrist_key = state.zobrist_key

    def make_null_move(self) -> bool:
        """Makes a null move (skips turn). Pushes state onto stack."""
        self.history.append(GameState(
            self.bitboards, self.occupancies, self.side_to_move,
            self.castling_rights, self.en_passant_sq, self.halfmove_clock,
            self.fullmove_number, self.piece_map, self.zobrist_key
        ))

        old_ep = self.en_passant_sq
        self.en_passant_sq = 64
        self.halfmove_clock += 1

        if old_ep != 64:
            self.zobrist_key ^= ZOBRIST_EP[old_ep % 8]

        opp_side = BLACK if self.side_to_move == WHITE else WHITE
        self.side_to_move = opp_side
        self.zobrist_key ^= ZOBRIST_SIDE

        if self.side_to_move == WHITE:
            self.fullmove_number += 1

        return True

    def to_chess_move(self, move: int) -> chess.Move:
        """Converts packed move representation to python-chess Move object."""
        from_sq = get_move_source(move)
        to_sq = get_move_dest(move)
        flag = get_move_flag(move)

        promotion = None
        if flag >= FLAG_PROMOTE_N:
            if flag == FLAG_PROMOTE_Q:
                promotion = chess.QUEEN
            elif flag == FLAG_PROMOTE_R:
                promotion = chess.ROOK
            elif flag == FLAG_PROMOTE_B:
                promotion = chess.BISHOP
            else:
                promotion = chess.KNIGHT

        return chess.Move(from_sq, to_sq, promotion=promotion)

    def print_board(self) -> None:
        """Prints the board in ASCII."""
        piece_symbols = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
        print("-" * 17)
        for r in range(7, -1, -1):
            row_chars = []
            for f in range(8):
                sq = r * 8 + f
                p_idx = self.get_piece_at(sq)
                if p_idx is not None:
                    row_chars.append(piece_symbols[p_idx])
                else:
                    row_chars.append(".")
            print(" ".join(row_chars))
        print("-" * 17)
