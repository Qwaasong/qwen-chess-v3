import chess
from typing import ClassVar

# Constants
A1: int
B1: int
C1: int
D1: int
E1: int
F1: int
G1: int
H1: int
A2: int
B2: int
C2: int
D2: int
E2: int
F2: int
G2: int
H2: int
A3: int
B3: int
C3: int
D3: int
E3: int
F3: int
G3: int
H3: int
A4: int
B4: int
C4: int
D4: int
E4: int
F4: int
G4: int
H4: int
A5: int
B5: int
C5: int
D5: int
E5: int
F5: int
G5: int
H5: int
A6: int
B6: int
C6: int
D6: int
E6: int
F6: int
G6: int
H6: int
A7: int
B7: int
C7: int
D7: int
E7: int
F7: int
G7: int
H7: int
A8: int
B8: int
C8: int
D8: int
E8: int
F8: int
G8: int
H8: int

WHITE: int
BLACK: int

P_P: int
P_N: int
P_B: int
P_R: int
P_Q: int
P_K: int
P_p: int
P_n: int
P_b: int
P_r: int
P_q: int
P_k: int

WK: int
WQ: int
BK: int
BQ: int

FLAG_NORMAL: int
FLAG_DOUBLE_PUSH: int
FLAG_CASTLE: int
FLAG_EP: int
FLAG_PROMOTE_N: int
FLAG_PROMOTE_B: int
FLAG_PROMOTE_R: int
FLAG_PROMOTE_Q: int

EMPTY_SQUARE: int
MAX_HISTORY: ClassVar[int]

def lsb(bb: int) -> int: ...
def pop_count(bb: int) -> int: ...
def clear_bit(bb: int, sq: int) -> int: ...
def set_bit(bb: int, sq: int) -> int: ...
def get_bit(bb: int, sq: int) -> int: ...

def mask_bishop_attacks(square: int) -> int: ...
def mask_rook_attacks(square: int) -> int: ...
def bishop_attacks_on_the_fly(square: int, block: int) -> int: ...
def rook_attacks_on_the_fly(square: int, block: int) -> int: ...
def set_occupancy(index: int, bits_in_mask: int, attack_mask: int) -> int: ...

def get_bishop_attacks(square: int, occupancy: int) -> int: ...
def get_rook_attacks(square: int, occupancy: int) -> int: ...
def get_queen_attacks(square: int, occupancy: int) -> int: ...

def encode_move(from_sq: int, to_sq: int, flag: int) -> int: ...
def get_move_source(move: int) -> int: ...
def get_move_dest(move: int) -> int: ...
def get_move_flag(move: int) -> int: ...

class CustomBitboardBoard:
    bitboards: list[int]
    occupancies: list[int]
    side_to_move: int
    castling_rights: int
    en_passant_sq: int
    halfmove_clock: int
    fullmove_number: int
    
    def __init__(self) -> None: ...
    @classmethod
    def from_chess_board(cls, chess_board: chess.Board) -> CustomBitboardBoard: ...
    def get_piece_at(self, square: int) -> int | None: ...
    def is_square_attacked(self, sq: int, attacker_color: int) -> bool: ...
    def in_check(self) -> bool: ...
    def generate_pseudo_legal_moves(self) -> list[int]: ...
    def generate_legal_moves(self) -> list[int]: ...
    def make_move(self, move: int) -> bool: ...
    def unmake_move(self) -> None: ...
    def to_chess_move(self, move: int) -> chess.Move: ...
    def print_board(self) -> None: ...
    @property
    def history(self) -> list[int]: ...
