"""Manual board selection.

A dimmed, full-screen overlay in which the user drags a rectangle around the
chessboard. Because boards are square, the selection snaps to a square by
default -- holding Shift allows a free-form rectangle for unusual layouts.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..utils.geometry import Rect


class BoardSelector(QWidget):
    """Lets the user drag out the board rectangle on a dimmed screen."""

    selected = Signal(object)   # Rect in virtual-desktop coordinates
    cancelled = Signal()

    def __init__(self, screen_rect: Rect, initial: Optional[Rect] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._screen_rect = screen_rect
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setCursor(Qt.CrossCursor)
        self.setWindowTitle("Select the chessboard")
        self.setGeometry(int(screen_rect.x), int(screen_rect.y),
                         int(screen_rect.width), int(screen_rect.height))

        self._origin: Optional[QPoint] = None
        self._current: Optional[QPoint] = None
        self._square_lock = True
        self._initial = initial

    def start(self) -> None:
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.OtherFocusReason)

    # ---------------------------------------------------------------- input
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._origin is not None:
            self._square_lock = not (event.modifiers() & Qt.ShiftModifier)
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton or self._origin is None:
            return
        rect = self._selection()
        self._origin = None
        if rect is None or rect.width() < 40 or rect.height() < 40:
            self.update()
            return
        result = Rect(rect.x() + self._screen_rect.x, rect.y() + self._screen_rect.y,
                      rect.width(), rect.height())
        self.close()
        self.selected.emit(result)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self.close()
            self.cancelled.emit()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------- geometry
    def _selection(self) -> Optional[QRect]:
        """The current drag rectangle, anchored at the point the drag started.

        The arithmetic is done on raw coordinates rather than via
        ``QRect.normalized()`` because ``QRect`` treats its edges as inclusive
        (``right() == left() + width() - 1``), which quietly introduces
        off-by-one errors in the width -- and a board rectangle that is one
        pixel wrong shifts every square boundary.
        """
        if self._origin is None or self._current is None:
            return None
        x0, y0 = self._origin.x(), self._origin.y()
        x1, y1 = self._current.x(), self._current.y()
        width, height = abs(x1 - x0), abs(y1 - y0)
        if self._square_lock:
            width = height = min(width, height)
        left = x0 if x1 >= x0 else x0 - width
        top = y0 if y1 >= y0 else y0 - height
        return QRect(left, top, width, height)

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 130))

        selection = self._selection()
        if selection is not None and selection.width() > 4:
            # Punch a clear hole so the board is fully visible while dragging.
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(selection, Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

            painter.setPen(QPen(QColor(90, 220, 140), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(selection)

            painter.setPen(QPen(QColor(90, 220, 140, 110), 1))
            for i in range(1, 8):
                dx = int(selection.width() * i / 8)
                dy = int(selection.height() * i / 8)
                painter.drawLine(selection.left() + dx, selection.top(),
                                 selection.left() + dx, selection.bottom())
                painter.drawLine(selection.left(), selection.top() + dy,
                                 selection.right(), selection.top() + dy)

            painter.setPen(QPen(QColor(235, 240, 245)))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(selection.left(), max(selection.top() - 8, 14),
                             f"{selection.width()} x {selection.height()}")

        painter.setPen(QPen(QColor(235, 240, 245)))
        painter.setFont(QFont("Segoe UI", 15))
        painter.drawText(self.rect().adjusted(0, 40, 0, 0), Qt.AlignHCenter | Qt.AlignTop,
                         "Drag a box around the chessboard\n"
                         "Hold Shift for a free-form rectangle  -  Esc to cancel")
        painter.end()
