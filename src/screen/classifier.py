"""Turning a gaze point into a semantic answer: which region, which square."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..utils.geometry import (Rect, WHITE_BOTTOM, square_from_gaze, square_rect)
from .regions import Region, RegionManager

OFF_SCREEN = "Off screen"
UNKNOWN = "Unknown"


@dataclass(frozen=True)
class GazeClassification:
    """Where a gaze point landed."""

    region_name: str
    square: Optional[str] = None
    kind: str = "other"
    on_screen: bool = True
    on_board: bool = False


class GazeRegionClassifier:
    """Classifies gaze points against the region set and the chessboard grid.

    Square assignment uses hysteresis: once a square is reported, the gaze has
    to move a configurable fraction of a cell past the boundary before a
    different square is reported. Without this, a gaze resting exactly on a
    cell edge flickers between two squares many times per second and floods
    the event log.
    """

    def __init__(self, regions: RegionManager, screen: Rect,
                 orientation: str = WHITE_BOTTOM,
                 off_screen_margin: float = 60.0,
                 square_hysteresis: float = 0.18) -> None:
        self.regions = regions
        self.screen = screen
        self.orientation = orientation
        self.off_screen_margin = off_screen_margin
        self.square_hysteresis = square_hysteresis
        self._last_square: Optional[str] = None

    def set_orientation(self, orientation: str) -> None:
        if orientation != self.orientation:
            self.orientation = orientation
            self._last_square = None

    def set_screen(self, screen: Rect) -> None:
        self.screen = screen

    def reset(self) -> None:
        self._last_square = None

    # ------------------------------------------------------------- squares
    def square_at(self, x: float, y: float, board: Rect) -> Optional[str]:
        """Board square under the point, with hysteresis against the previous one."""
        candidate = square_from_gaze(x, y, board.x, board.y, board.width,
                                     board.height, self.orientation)
        if candidate is None:
            self._last_square = None
            return None
        if self._last_square is None or candidate == self._last_square:
            self._last_square = candidate
            return candidate

        # Stay on the previous square until the point clears its border by a
        # margin, so boundary noise cannot cause rapid flip-flopping.
        previous = square_rect(self._last_square, board.x, board.y,
                               board.width, board.height, self.orientation)
        margin_x = previous.width * self.square_hysteresis
        margin_y = previous.height * self.square_hysteresis
        if previous.inflated(0).intersection(board).area > 0 and \
                previous.inflated(max(margin_x, margin_y)).contains(x, y):
            return self._last_square

        self._last_square = candidate
        return candidate

    # ------------------------------------------------------------ classify
    def classify(self, x: float, y: float) -> GazeClassification:
        if not self.screen.inflated(self.off_screen_margin).contains(x, y):
            self._last_square = None
            return GazeClassification(OFF_SCREEN, None, "away", on_screen=False)

        region: Optional[Region] = self.regions.classify(x, y)
        if region is None:
            self._last_square = None
            return GazeClassification(UNKNOWN, None, "other", on_screen=True)

        if region.name == RegionManager.BOARD_NAME:
            square = self.square_at(x, y, region.rect)
            return GazeClassification(region.name, square, region.kind,
                                      on_screen=True, on_board=True)

        self._last_square = None
        return GazeClassification(region.name, None, region.kind, on_screen=True)
