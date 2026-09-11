"""Quick recentre.

Even a good calibration develops a bias over a long session: people slouch,
lean in, and shift in their chair, and a calibration is tied to the geometry it
was recorded at. Redoing all 13 points to correct a constant offset is
disproportionate.

This shows a single target in the middle of the screen, measures the difference
between where the tracker thinks the user is looking and where they actually
are, and applies that difference as a bias correction. It takes about three
seconds.

A pure offset cannot fix a mapping that has genuinely changed shape -- if the
error varies across the screen, a full recalibration is the honest remedy, and
the dialog says so when the samples disagree too much among themselves.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QPointF, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..tracking.pipeline import GazeSample
from ..utils.geometry import Rect

logger = logging.getLogger(__name__)

PHASE_SETTLE = "settle"
PHASE_COLLECT = "collect"


class RecentreWindow(QWidget):
    """Measures a constant gaze bias from a single centre target."""

    completed = Signal(float, float)   # dx, dy to add to future estimates
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, screen_rect: Rect, settle_ms: int = 700,
                 collect_ms: int = 1400, minimum_samples: int = 12,
                 parent: Optional[QWidget] = None, qt_screen=None) -> None:
        super().__init__(parent)
        self._screen_rect = screen_rect
        self._qt_screen = qt_screen
        self._settle_ms = settle_ms
        self._collect_ms = collect_ms
        self._minimum_samples = minimum_samples

        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setCursor(Qt.BlankCursor)
        self.setWindowTitle("Recentre")

        self._samples: List[Tuple[float, float]] = []
        self._phase = PHASE_SETTLE
        self._elapsed = 0
        self._animation = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._tick)

    # ---------------------------------------------------------------- start
    def start(self) -> None:
        if self._qt_screen is not None:
            handle = self.windowHandle()
            if handle is None:
                self.create()
                handle = self.windowHandle()
            if handle is not None:
                handle.setScreen(self._qt_screen)
                self.setGeometry(self._qt_screen.geometry())
        else:
            self.setGeometry(int(self._screen_rect.x), int(self._screen_rect.y),
                             int(self._screen_rect.width), int(self._screen_rect.height))
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.OtherFocusReason)
        self._timer.start()

    @property
    def _target(self) -> Tuple[float, float]:
        return (self.x() + self.width() / 2.0, self.y() + self.height() / 2.0)

    # -------------------------------------------------------------- samples
    def on_sample(self, sample: GazeSample) -> None:
        """Slot for ``TrackerWorker.sample_ready``."""
        if self._phase != PHASE_COLLECT or not sample.valid or sample.holding:
            return
        if sample.confidence < 0.3:
            return
        self._samples.append((sample.x, sample.y))

    # ----------------------------------------------------------------- loop
    def _tick(self) -> None:
        self._elapsed += self._timer.interval()
        self._animation += 0.09
        if self._phase == PHASE_SETTLE and self._elapsed >= self._settle_ms:
            self._phase = PHASE_COLLECT
            self._elapsed = 0
            self._samples.clear()
        elif self._phase == PHASE_COLLECT and self._elapsed >= self._collect_ms:
            self._finish()
        self.update()

    def _finish(self) -> None:
        self._timer.stop()
        self.close()

        if len(self._samples) < self._minimum_samples:
            self.failed.emit(
                "Not enough usable samples to recentre.\n\n"
                "Check the lighting and that your face is visible, then try again."
            )
            return

        xs = sorted(point[0] for point in self._samples)
        ys = sorted(point[1] for point in self._samples)
        middle = len(xs) // 2
        # The median resists the odd frame where the user glanced away.
        median_x = xs[middle] if len(xs) % 2 else 0.5 * (xs[middle - 1] + xs[middle])
        median_y = ys[middle] if len(ys) % 2 else 0.5 * (ys[middle - 1] + ys[middle])

        spread = math.hypot(
            (xs[int(len(xs) * 0.84)] - xs[int(len(xs) * 0.16)]) / 2.0,
            (ys[int(len(ys) * 0.84)] - ys[int(len(ys) * 0.16)]) / 2.0,
        )
        target_x, target_y = self._target
        dx, dy = target_x - median_x, target_y - median_y
        magnitude = math.hypot(dx, dy)
        diagonal = math.hypot(self.width(), self.height())

        if spread > diagonal * 0.10:
            self.failed.emit(
                "The estimates were too unsteady to measure a reliable offset.\n\n"
                "Improve the lighting, sit still for the three seconds, and try "
                "again -- or run a full calibration."
            )
            return
        if magnitude > diagonal * 0.35:
            self.failed.emit(
                f"The measured offset ({magnitude:.0f} px) is too large to be a "
                f"simple drift.\n\nA full calibration will fix this properly."
            )
            return

        logger.info("Recentre offset: (%.0f, %.0f) px from %d samples",
                    dx, dy, len(self._samples))
        self.completed.emit(dx, dy)

    # ---------------------------------------------------------------- input
    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self._timer.stop()
            self.close()
            self.cancelled.emit()
        else:
            super().keyPressEvent(event)

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(16, 18, 22, 225))

        centre = QPointF(self.width() / 2.0, self.height() / 2.0)
        pulse = 1.0 + 0.16 * math.sin(self._animation * 2.4)
        outer = 44 * (pulse if self._phase == PHASE_SETTLE else 1.0)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 70, 70, 40))
        painter.drawEllipse(centre, outer, outer)

        if self._phase == PHASE_COLLECT:
            progress = min(1.0, self._elapsed / max(self._collect_ms, 1))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(80, 220, 140), 4, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(int(centre.x() - outer), int(centre.y() - outer),
                            int(outer * 2), int(outer * 2), 90 * 16,
                            int(-progress * 360 * 16))

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 60, 60))
        painter.drawEllipse(centre, 16, 16)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(centre, 4, 4)

        painter.setPen(QPen(QColor(225, 232, 240)))
        painter.setFont(QFont("Segoe UI", 17))
        painter.drawText(self.rect().adjusted(0, 90, 0, 0), Qt.AlignHCenter | Qt.AlignTop,
                         "Look straight at the dot")
        painter.setFont(QFont("Segoe UI", 12))
        painter.setPen(QPen(QColor(150, 160, 175)))
        painter.drawText(self.rect().adjusted(0, 0, 0, -80),
                         Qt.AlignHCenter | Qt.AlignBottom,
                         "Measuring drift, about three seconds  -  Esc to cancel")
        painter.end()
