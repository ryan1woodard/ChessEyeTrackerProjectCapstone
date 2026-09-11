"""Chessboard detection tests using synthetic, deterministic images."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.screen.chessboard import (ChessboardDetector, ChessboardRegion,
                                   cell_levels, checker_score)
from src.utils.geometry import Rect


def make_board(size: int = 400, light: int = 235, dark: int = 118) -> np.ndarray:
    """A clean 8x8 grayscale board."""
    board = np.zeros((size, size), dtype=np.uint8)
    step = size // 8
    for row in range(8):
        for col in range(8):
            value = light if (row + col) % 2 == 0 else dark
            board[row * step:(row + 1) * step, col * step:(col + 1) * step] = value
    return board


def make_screen(board_size: int = 480, screen=(1280, 800),
                origin=(300, 140)) -> np.ndarray:
    """A grey desktop with a board pasted onto it, as BGR."""
    screen_image = np.full((screen[1], screen[0]), 60, dtype=np.uint8)
    board = make_board(board_size)
    x, y = origin
    screen_image[y:y + board_size, x:x + board_size] = board
    return cv2.cvtColor(screen_image, cv2.COLOR_GRAY2BGR)


class TestCheckerScore:
    def test_a_clean_board_scores_almost_perfectly(self):
        assert checker_score(make_board(256)) > 0.95

    def test_a_flat_image_scores_zero(self):
        assert checker_score(np.full((256, 256), 128, dtype=np.uint8)) == 0.0

    def test_noise_scores_low(self):
        rng = np.random.default_rng(11)
        noise = rng.integers(0, 255, (256, 256), dtype=np.uint8)
        assert checker_score(noise) < 0.35

    def test_inverted_colours_score_the_same(self):
        """The detector must be theme independent."""
        board = make_board(256)
        assert checker_score(255 - board) == pytest.approx(checker_score(board), abs=0.02)

    def test_pieces_degrade_but_do_not_destroy_the_score(self):
        board = make_board(256)
        rng = np.random.default_rng(12)
        step = 256 // 8
        # Occlude the back two ranks at each end, like a starting position.
        for row in list(range(2)) + list(range(6, 8)):
            for col in range(8):
                cx, cy = col * step + step // 2, row * step + step // 2
                colour = int(rng.choice([25, 240]))
                cv2.circle(board, (cx, cy), int(step * 0.40), colour, -1)
        # Sampling the cell borders means even large pieces barely dent it.
        assert checker_score(board) > 0.85

    def test_a_piece_covering_an_entire_cell_still_degrades_gracefully(self):
        """The score must fall, not stay falsely high, when cells are unreadable."""
        board = make_board(256)
        board[:64, :] = 200   # top two ranks completely obscured
        score = checker_score(board)
        assert 0.0 < score < 0.95

    def test_too_small_an_image_is_rejected(self):
        assert checker_score(np.zeros((16, 16), dtype=np.uint8)) == 0.0

    def test_cell_levels_shape_and_alternation(self):
        means = cell_levels(make_board(256))
        assert means.shape == (8, 8)
        assert means[0, 0] > means[0, 1]


class TestDetector:
    def test_finds_a_board_on_a_plain_desktop(self):
        detector = ChessboardDetector()
        region = detector.detect(make_screen())
        assert region is not None
        assert region.x == pytest.approx(300, abs=12)
        assert region.y == pytest.approx(140, abs=12)
        assert region.width == pytest.approx(480, abs=20)
        assert region.confidence > 0.3

    def test_coordinates_are_offset_by_the_capture_rectangle(self):
        detector = ChessboardDetector()
        frame_rect = Rect(1920, 0, 1280, 800)  # a second monitor to the right
        region = detector.detect(make_screen(), frame_rect)
        assert region is not None
        assert region.x == pytest.approx(1920 + 300, abs=12)

    def test_returns_none_when_there_is_no_board(self):
        rng = np.random.default_rng(13)
        noise = rng.integers(0, 255, (600, 900, 3), dtype=np.uint8)
        assert ChessboardDetector(score_threshold=0.6).detect(noise) is None

    def test_empty_input_is_handled(self):
        assert ChessboardDetector().detect(np.zeros((0, 0, 3), dtype=np.uint8)) is None

    def test_detects_boards_of_different_sizes(self):
        detector = ChessboardDetector()
        for size in (320, 480, 640):
            region = detector.detect(make_screen(board_size=size))
            assert region is not None, f"missed a {size}px board"
            assert region.width == pytest.approx(size, rel=0.08)


class TestChessboardRegion:
    def test_dict_round_trip(self):
        region = ChessboardRegion(10, 20, 300, 300, 0.8, "auto")
        assert ChessboardRegion.from_dict(region.to_dict()) == region

    def test_from_rect(self):
        region = ChessboardRegion.from_rect(Rect(5, 6, 100, 100))
        assert region.source == "manual"
        assert region.rect == Rect(5, 6, 100, 100)
