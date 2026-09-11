"""The full-screen gaze overlay.

This is a frameless, translucent, always-on-top window that draws the red gaze
dot and (optionally) debug information. Two flags make it usable while the user
plays chess normally:

``Qt.WindowTransparentForInput``
    The compositor never routes input to this window, so clicks, drags and
    scrolls all pass straight through to whatever is underneath.

``Qt.WA_TranslucentBackground``
    Only the drawn pixels are visible; the rest of the window is truly
    transparent rather than painted black.

The window is deliberately *not* a child of the main window, so minimising or
closing the main window does not disturb it.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QGuiApplication, QPainter, QPen,
                           QPixmap)
from PySide6.QtWidgets import QWidget

from ..utils.geometry import Rect, WHITE_BOTTOM, square_rect
from .heatmap import MODE_BOARD, MODE_BOTH, MODE_SCREEN, square_colour

logger = logging.getLogger(__name__)


class GazeOverlay(QWidget):
    """Draws the estimated gaze position on top of everything else."""

    def __init__(self, screen_rect: Rect, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
            | Qt.BypassWindowManagerHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.NoFocus)

        self._screen_rect = screen_rect
        self._point: Optional[Tuple[float, float]] = None
        self._confidence = 0.0
        self._valid = False

        # appearance
        self._dot_radius = 12
        self._dot_opacity = 0.75
        self._dot_color = QColor("#ff2d2d")

        # debug layers
        self._show_dot = True
        self._show_debug = False
        self._show_board = False
        self._show_regions = False
        self._board_rect: Optional[Rect] = None
        self._regions: List[Tuple[str, Rect]] = []
        self._debug_lines: List[str] = []
        self._square_rect: Optional[Rect] = None

        # heatmap layers
        self._show_heatmap = False
        self._heatmap_mode = MODE_BOTH
        self._heatmap_pixmap: Optional[QPixmap] = None
        self._board_cells: dict = {}
        self._board_orientation = WHITE_BOTTOM
        self._heatmap_labels = True

        self.set_screen_rect(screen_rect)

    # ------------------------------------------------------------- geometry
    def set_screen_rect(self, rect: Rect) -> None:
        """Position the overlay to cover exactly one monitor."""
        self._screen_rect = rect
        self.setGeometry(QRect(int(rect.x), int(rect.y),
                               int(rect.width), int(rect.height)))

    def _to_local(self, x: float, y: float) -> QPoint:
        return QPoint(int(x - self._screen_rect.x), int(y - self._screen_rect.y))

    # ------------------------------------------------------------ appearance
    def configure(self, dot_radius: int, dot_opacity: float, dot_color: str) -> None:
        self._dot_radius = max(2, int(dot_radius))
        self._dot_opacity = max(0.05, min(1.0, float(dot_opacity)))
        self._dot_color = QColor(dot_color)
        self.update()

    def set_layers(self, show_dot: bool, show_debug: bool,
                   show_board: bool, show_regions: bool,
                   show_heatmap: bool = False) -> None:
        self._show_dot = show_dot
        self._show_debug = show_debug
        self._show_board = show_board
        self._show_regions = show_regions
        self._show_heatmap = show_heatmap
        self.update()

    def set_heatmap_mode(self, mode: str) -> None:
        self._heatmap_mode = mode
        self.update()

    def set_heatmap_image(self, pixmap: Optional[QPixmap]) -> None:
        """Install the pre-rendered screen heatmap (already scaled by caller)."""
        self._heatmap_pixmap = pixmap
        if self._show_heatmap:
            self.update()

    def set_board_heatmap(self, cells: dict, orientation: str = WHITE_BOTTOM,
                          show_labels: bool = True) -> None:
        self._board_cells = cells or {}
        self._board_orientation = orientation
        self._heatmap_labels = show_labels
        if self._show_heatmap:
            self.update()

    @property
    def any_layer_visible(self) -> bool:
        return (self._show_dot or self._show_debug or self._show_board
                or self._show_regions or self._show_heatmap)

    # ----------------------------------------------------------------- data
    def set_gaze(self, x: float, y: float, confidence: float, valid: bool) -> None:
        self._point = (x, y)
        self._confidence = confidence
        self._valid = valid
        self.update()

    def clear_gaze(self) -> None:
        self._point = None
        self._valid = False
        self.update()

    def set_board(self, rect: Optional[Rect]) -> None:
        self._board_rect = rect
        self.update()

    def set_square_rect(self, rect: Optional[Rect]) -> None:
        self._square_rect = rect

    def set_regions(self, regions: List[Tuple[str, Rect]]) -> None:
        self._regions = regions
        self.update()

    def set_debug_lines(self, lines: List[str]) -> None:
        self._debug_lines = lines
        if self._show_debug:
            self.update()

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if self._show_heatmap:
            if self._heatmap_mode in (MODE_SCREEN, MODE_BOTH):
                self._paint_screen_heatmap(painter)
            if self._heatmap_mode in (MODE_BOARD, MODE_BOTH):
                self._paint_board_heatmap(painter)

        if self._show_regions:
            self._paint_regions(painter)
        if self._show_board and self._board_rect is not None:
            self._paint_board(painter)
        if self._show_dot and self._point is not None:
            self._paint_dot(painter)
        if self._show_debug:
            self._paint_debug(painter)
        painter.end()

    def _paint_screen_heatmap(self, painter: QPainter) -> None:
        if self._heatmap_pixmap is None or self._heatmap_pixmap.isNull():
            return
        painter.drawPixmap(self.rect(), self._heatmap_pixmap)

    def _paint_board_heatmap(self, painter: QPainter) -> None:
        """Shade each board square by how long it was looked at."""
        board = self._board_rect
        if board is None or not self._board_cells:
            return
        peak = max(self._board_cells.values())
        if peak <= 0:
            return

        painter.setFont(QFont("Segoe UI", 8, QFont.DemiBold))
        for square, seconds in self._board_cells.items():
            try:
                cell = square_rect(square, board.x, board.y, board.width,
                                   board.height, self._board_orientation)
            except ValueError:  # pragma: no cover - defensive
                continue
            fraction = seconds / peak
            red, green, blue = square_colour(fraction)
            # Alpha tracks dwell time so lightly-viewed squares stay readable.
            colour = QColor(red, green, blue, int(30 + 165 * fraction))

            origin = self._to_local(cell.x, cell.y)
            rect = QRect(origin.x(), origin.y(), int(cell.width), int(cell.height))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(colour))
            painter.drawRect(rect)

            if self._heatmap_labels and seconds >= 0.5:
                painter.setPen(QPen(QColor(255, 255, 255, 230)))
                painter.drawText(rect, Qt.AlignCenter, f"{seconds:.0f}s")

    def _paint_dot(self, painter: QPainter) -> None:
        assert self._point is not None
        center = self._to_local(*self._point)
        radius = self._dot_radius

        # Fade the dot as confidence drops so the user can see at a glance
        # when the estimate should not be trusted.
        opacity = self._dot_opacity * (0.35 + 0.65 * self._confidence)
        color = QColor(self._dot_color)
        color.setAlphaF(max(0.0, min(1.0, opacity)))

        if not self._valid:
            painter.setPen(QPen(QColor(255, 255, 255, 90), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(center, radius, radius)
            return

        halo = QColor(color)
        halo.setAlphaF(color.alphaF() * 0.28)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(halo))
        painter.drawEllipse(center, int(radius * 2.1), int(radius * 2.1))

        painter.setBrush(QBrush(color))
        painter.drawEllipse(center, radius, radius)

        painter.setPen(QPen(QColor(255, 255, 255, 200), 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, radius, radius)

        if self._square_rect is not None:
            painter.setPen(QPen(QColor(60, 200, 255, 170), 2))
            rect = self._square_rect
            painter.drawRect(QRect(*self._to_local(rect.x, rect.y).toTuple(),
                                   int(rect.width), int(rect.height)))

    def _paint_board(self, painter: QPainter) -> None:
        rect = self._board_rect
        assert rect is not None
        origin = self._to_local(rect.x, rect.y)
        painter.setPen(QPen(QColor(80, 220, 120, 200), 3))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRect(origin.x(), origin.y(), int(rect.width), int(rect.height)))

        painter.setPen(QPen(QColor(80, 220, 120, 70), 1))
        for i in range(1, 8):
            dx = int(rect.width * i / 8)
            dy = int(rect.height * i / 8)
            painter.drawLine(origin.x() + dx, origin.y(),
                             origin.x() + dx, origin.y() + int(rect.height))
            painter.drawLine(origin.x(), origin.y() + dy,
                             origin.x() + int(rect.width), origin.y() + dy)

    def _paint_regions(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 9))
        for name, rect in self._regions:
            origin = self._to_local(rect.x, rect.y)
            painter.setPen(QPen(QColor(255, 190, 60, 180), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRect(origin.x(), origin.y(),
                                   int(rect.width), int(rect.height)))
            painter.setPen(QPen(QColor(255, 190, 60, 220)))
            painter.drawText(origin.x() + 6, origin.y() + 16, name)

    def _paint_debug(self, painter: QPainter) -> None:
        if not self._debug_lines:
            return
        painter.setFont(QFont("Consolas", 10))
        metrics = painter.fontMetrics()
        width = max(metrics.horizontalAdvance(line) for line in self._debug_lines) + 24
        height = metrics.height() * len(self._debug_lines) + 18
        box = QRect(16, 16, width, height)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(12, 14, 18, 195)))
        painter.drawRoundedRect(box, 8, 8)

        painter.setPen(QPen(QColor(235, 240, 245)))
        y = box.top() + metrics.ascent() + 9
        for line in self._debug_lines:
            painter.drawText(box.left() + 12, y, line)
            y += metrics.height()


def primary_screen_rect() -> Rect:
    """Fallback screen geometry when no monitor has been chosen yet."""
    screen = QGuiApplication.primaryScreen()
    if screen is None:  # pragma: no cover - headless
        return Rect(0, 0, 1920, 1080)
    geometry = screen.geometry()
    return Rect(geometry.x(), geometry.y(), geometry.width(), geometry.height())
