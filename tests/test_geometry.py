"""Geometry and chess-square mapping tests."""

from __future__ import annotations

import pytest

from src.utils.geometry import (BLACK_BOTTOM, Rect, WHITE_BOTTOM, bounding_rect,
                                distance_to_square_center, square_from_gaze,
                                square_indices_from_gaze, square_name, square_rect)

BOARD = (50.0, 50.0, 800.0, 800.0)  # x, y, w, h -> 100 px squares


class TestRect:
    def test_edges_and_area(self):
        rect = Rect(10, 20, 30, 40)
        assert (rect.left, rect.top, rect.right, rect.bottom) == (10, 20, 40, 60)
        assert rect.area == 1200
        assert rect.center == (25.0, 40.0)

    def test_containment_is_half_open(self):
        rect = Rect(0, 0, 10, 10)
        assert rect.contains(0, 0)
        assert rect.contains(9.99, 9.99)
        assert not rect.contains(10, 5)   # right edge excluded
        assert not rect.contains(5, 10)   # bottom edge excluded
        assert not rect.contains(-0.01, 5)

    def test_intersection_and_iou(self):
        a = Rect(0, 0, 10, 10)
        b = Rect(5, 5, 10, 10)
        assert a.intersection(b) == Rect(5, 5, 5, 5)
        assert a.iou(a) == pytest.approx(1.0)
        assert a.iou(b) == pytest.approx(25 / 175)
        assert a.iou(Rect(100, 100, 5, 5)) == 0.0

    def test_from_corners_normalises(self):
        assert Rect.from_corners(30, 40, 10, 20) == Rect(10, 20, 20, 20)

    def test_bounding_rect(self):
        assert bounding_rect([(1, 2), (5, 7), (3, 0)]) == Rect(1, 0, 4, 7)


class TestSquareMapping:
    def test_documented_example(self):
        assert square_from_gaze(450, 150, *BOARD) == "e7"

    @pytest.mark.parametrize("x,y,expected", [
        (55, 55, "a8"),      # top-left corner
        (845, 845, "h1"),    # bottom-right corner
        (450, 450, "e4"),    # just past the centre line
        (449, 449, "d5"),    # just before it
        (55, 845, "a1"),
        (845, 55, "h8"),
    ])
    def test_white_bottom(self, x, y, expected):
        assert square_from_gaze(x, y, *BOARD) == expected

    @pytest.mark.parametrize("x,y,expected", [
        (450, 150, "d2"),
        (55, 55, "h1"),
        (845, 845, "a8"),
    ])
    def test_black_bottom_is_a_180_degree_rotation(self, x, y, expected):
        assert square_from_gaze(x, y, *BOARD, orientation=BLACK_BOTTOM) == expected

    @pytest.mark.parametrize("x,y", [(49, 400), (851, 400), (400, 49), (400, 851)])
    def test_points_outside_the_board(self, x, y):
        assert square_from_gaze(x, y, *BOARD) is None

    def test_zero_sized_board(self):
        assert square_from_gaze(10, 10, 0, 0, 0, 0) is None

    def test_indices_are_clamped_inside_range(self):
        assert square_indices_from_gaze(849.999, 849.999, *BOARD) == (7, 7)

    def test_every_square_round_trips(self):
        for file_char in "abcdefgh":
            for rank in range(1, 9):
                square = f"{file_char}{rank}"
                for orientation in (WHITE_BOTTOM, BLACK_BOTTOM):
                    rect = square_rect(square, *BOARD, orientation=orientation)
                    cx, cy = rect.center
                    assert square_from_gaze(cx, cy, *BOARD,
                                            orientation=orientation) == square

    def test_square_rect_size(self):
        rect = square_rect("e4", *BOARD)
        assert rect.width == 100 and rect.height == 100

    def test_invalid_square_names_rejected(self):
        for bad in ("", "e", "e9", "i4", "44", "e0"):
            with pytest.raises(ValueError):
                square_rect(bad, *BOARD)

    def test_square_name_bounds(self):
        assert square_name(0, 0) == "a8"
        assert square_name(7, 7) == "h1"
        with pytest.raises(ValueError):
            square_name(8, 0)

    def test_distance_to_square_centre(self):
        board = Rect(*BOARD)
        centre = square_rect("e4", *BOARD).center
        assert distance_to_square_center(*centre, "e4", board) == pytest.approx(0.0)
        assert distance_to_square_center(centre[0] + 30, centre[1] + 40,
                                         "e4", board) == pytest.approx(50.0)

    def test_scaling_the_board_does_not_change_the_answer(self):
        """A resized browser window must map the same relative point identically."""
        for width in (320, 640, 800, 1201):
            board = (0.0, 0.0, float(width), float(width))
            # 4.5/8 across, 1.5/8 down -> e7
            x = width * 4.5 / 8
            y = width * 1.5 / 8
            assert square_from_gaze(x, y, *board) == "e7"
