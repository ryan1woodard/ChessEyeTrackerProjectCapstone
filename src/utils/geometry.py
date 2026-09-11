"""Pure geometric helpers.

Everything in this module is deliberately free of Qt, OpenCV and MediaPipe so
that it can be unit tested with synthetic data and no hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

FILES = "abcdefgh"
WHITE_BOTTOM = "white_bottom"
BLACK_BOTTOM = "black_bottom"


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle in screen (virtual desktop) pixel space."""

    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def top(self) -> float:
        return self.y

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def contains(self, x: float, y: float) -> bool:
        """Half-open containment: left/top inclusive, right/bottom exclusive."""
        return self.left <= x < self.right and self.top <= y < self.bottom

    def inflated(self, margin: float) -> "Rect":
        return Rect(self.x - margin, self.y - margin,
                    self.width + 2 * margin, self.height + 2 * margin)

    def intersection(self, other: "Rect") -> "Rect":
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return Rect(left, top, 0.0, 0.0)
        return Rect(left, top, right - left, bottom - top)

    def iou(self, other: "Rect") -> float:
        inter = self.intersection(other).area
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)

    @classmethod
    def from_corners(cls, x0: float, y0: float, x1: float, y1: float) -> "Rect":
        return cls(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def square_indices_from_gaze(
    x: float,
    y: float,
    board_x: float,
    board_y: float,
    board_width: float,
    board_height: float,
) -> Optional[Tuple[int, int]]:
    """Return zero-based ``(column, row)`` of the board cell under ``(x, y)``.

    Column 0 is the left-most cell and row 0 is the top-most cell, in *screen*
    space -- orientation is applied later by :func:`square_name`.
    Returns ``None`` when the point lies outside the board.
    """
    if board_width <= 0 or board_height <= 0:
        return None
    if not (board_x <= x < board_x + board_width and board_y <= y < board_y + board_height):
        return None
    col = int((x - board_x) / (board_width / 8.0))
    row = int((y - board_y) / (board_height / 8.0))
    return (min(max(col, 0), 7), min(max(row, 0), 7))


def square_name(col: int, row: int, orientation: str = WHITE_BOTTOM) -> str:
    """Convert screen-space cell indices to algebraic notation."""
    if not (0 <= col <= 7 and 0 <= row <= 7):
        raise ValueError(f"cell indices out of range: {(col, row)}")
    if orientation == BLACK_BOTTOM:
        return f"{FILES[7 - col]}{row + 1}"
    return f"{FILES[col]}{8 - row}"


def square_from_gaze(
    x: float,
    y: float,
    board_x: float,
    board_y: float,
    board_width: float,
    board_height: float,
    orientation: str = WHITE_BOTTOM,
) -> Optional[str]:
    """Map a screen gaze point to a chess square such as ``"e4"``.

    >>> square_from_gaze(450, 150, 50, 50, 800, 800)
    'e7'
    """
    indices = square_indices_from_gaze(x, y, board_x, board_y, board_width, board_height)
    if indices is None:
        return None
    return square_name(indices[0], indices[1], orientation)


def square_rect(
    square: str,
    board_x: float,
    board_y: float,
    board_width: float,
    board_height: float,
    orientation: str = WHITE_BOTTOM,
) -> Rect:
    """Inverse of :func:`square_from_gaze`: the screen rectangle of a square."""
    square = square.strip().lower()
    if len(square) != 2 or square[0] not in FILES or not square[1].isdigit():
        raise ValueError(f"invalid square: {square!r}")
    file_index = FILES.index(square[0])
    rank = int(square[1])
    if not 1 <= rank <= 8:
        raise ValueError(f"invalid rank in square: {square!r}")
    if orientation == BLACK_BOTTOM:
        col, row = 7 - file_index, rank - 1
    else:
        col, row = file_index, 8 - rank
    cell_w = board_width / 8.0
    cell_h = board_height / 8.0
    return Rect(board_x + col * cell_w, board_y + row * cell_h, cell_w, cell_h)


def distance_to_square_center(
    x: float,
    y: float,
    square: str,
    board: Rect,
    orientation: str = WHITE_BOTTOM,
) -> float:
    """Distance in pixels from a point to the centre of a named square."""
    rect = square_rect(square, board.x, board.y, board.width, board.height, orientation)
    return distance((x, y), rect.center)


def bounding_rect(points: Iterable[Tuple[float, float]]) -> Rect:
    """Smallest axis-aligned rectangle containing all points."""
    pts = list(points)
    if not pts:
        raise ValueError("bounding_rect requires at least one point")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return Rect.from_corners(min(xs), min(ys), max(xs), max(ys))
