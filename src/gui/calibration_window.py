"""The calibration screen.

A full-screen window shows one target at a time. Each target runs through two
phases:

``settle``
    The target appears and pulses. Nothing is recorded yet -- the user needs a
    moment to find it and fixate.

``collect``
    Samples are recorded until either the sample quota or the dwell time is
    reached. A ring around the target fills up so the user can see progress and
    knows to keep looking.

Samples arrive on the ``sample_ready`` signal from the tracking worker, i.e.
already fully processed on the worker thread; this window only stores feature
vectors and never touches the camera itself.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..tracking.calibration import (CalibrationSession, calibration_pattern,
                                    pattern_to_pixels)
from ..tracking.pipeline import GazeSample
from ..utils.config import Config
from ..utils.geometry import Rect

logger = logging.getLogger(__name__)

PHASE_INTRO = "intro"
PHASE_SETTLE = "settle"
PHASE_COLLECT = "collect"
PHASE_DONE = "done"


class CalibrationWindow(QWidget):
    """Runs the calibration procedure over one monitor."""

    completed = Signal(object)   # CalibrationProfile
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, config: Config, screen_rect: Rect, monitor_index: int,
                 camera_index: int, parent: Optional[QWidget] = None,
                 qt_screen=None) -> None:
        super().__init__(parent)
        self._qt_screen = qt_screen
        self._config = config
        self._screen_rect = screen_rect
        self._monitor_index = monitor_index
        self._camera_index = camera_index

        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Calibration")
        self.setCursor(Qt.BlankCursor)
        self.setGeometry(int(screen_rect.x), int(screen_rect.y),
                         int(screen_rect.width), int(screen_rect.height))

        self._pattern_name = str(config.get("calibration.pattern", "13point"))
        pattern = calibration_pattern(self._pattern_name)
        self._targets: List[Tuple[float, float]] = pattern_to_pixels(
            pattern, int(screen_rect.width), int(screen_rect.height),
            (int(screen_rect.x), int(screen_rect.y))
        )

        self._session = CalibrationSession(
            feature_names=list(config.get("calibration.model_features", [])),
            screen_size=(int(screen_rect.width), int(screen_rect.height)),
            screen_origin=(int(screen_rect.x), int(screen_rect.y)),
            degree=int(config.get("calibration.poly_degree", 2)),
            alphas=list(config.get("calibration.ridge_alphas", [0.1, 1.0, 10.0])),
            max_yaw=float(config.get("tracking.max_head_yaw_deg", 40.0)),
            max_pitch=float(config.get("tracking.max_head_pitch_deg", 30.0)),
            min_openness=float(config.get("tracking.blink_ear_threshold", 0.16)),
        )

        self._settle_ms = int(config.get("calibration.settle_ms", 700))
        self._dwell_ms = int(config.get("calibration.dwell_ms", 900))
        self._quota = int(config.get("calibration.samples_per_point", 20))
        self._radius = int(config.get("calibration.target_radius_px", 18))
        self._head_motion_hint = bool(config.get("calibration.head_motion_guidance", True))

        self._index = 0
        self._phase = PHASE_INTRO
        self._phase_elapsed = 0
        self._animation = 0.0
        self._collected = 0
        self._face_present = False
        self._message = ""

        self._timer = QTimer(self)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._tick)

    def _rebuild_targets(self) -> None:
        """Recompute targets from the widget's real on-screen geometry.

        The window is what the user actually looks at, so its size is the
        authority on where the dots are. Deriving the targets from anything
        else risks training the model on coordinates that do not match where
        the dots appeared -- which produces a model whose predictions are
        uniformly scaled wrong and land off screen.
        """
        width, height = self.width(), self.height()
        origin = (self.x(), self.y())
        if width < 100 or height < 100:
            return
        pattern = calibration_pattern(self._pattern_name)
        self._targets = pattern_to_pixels(pattern, width, height, origin)
        self._session.screen_size = (width, height)
        self._session.screen_origin = origin
        logger.info("Calibration targets built for %dx%d at %s", width, height, origin)

    # ---------------------------------------------------------------- start
    def start(self) -> None:
        # Bind to the chosen monitor before going full screen, otherwise a
        # multi-monitor setup can open the calibration on the wrong display and
        # every recorded target coordinate would be meaningless.
        if self._qt_screen is not None:
            handle = self.windowHandle()
            if handle is None:
                self.create()
                handle = self.windowHandle()
            if handle is not None:
                handle.setScreen(self._qt_screen)
                geometry = self._qt_screen.geometry()
                self.setGeometry(geometry)
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.OtherFocusReason)
        self._rebuild_targets()
        self._phase = PHASE_INTRO
        self._phase_elapsed = 0
        self._timer.start()

    # ------------------------------------------------------------- samples
    def on_sample(self, sample: GazeSample) -> None:
        """Slot for ``TrackerWorker.sample_ready`` (queued to the GUI thread)."""
        self._face_present = sample.face_detected
        if self._phase != PHASE_COLLECT or sample.features is None:
            return
        if self._collected >= self._quota:
            return
        target = self._targets[self._index]
        if self._session.add(self._index, target, sample.features):
            self._collected += 1

    # ----------------------------------------------------------------- loop
    def _tick(self) -> None:
        self._phase_elapsed += self._timer.interval()
        self._animation += 0.08

        if self._phase == PHASE_INTRO:
            if self._phase_elapsed >= 2200:
                self._begin_point(0)
        elif self._phase == PHASE_SETTLE:
            if self._phase_elapsed >= self._settle_ms:
                self._phase = PHASE_COLLECT
                self._phase_elapsed = 0
                self._collected = 0
        elif self._phase == PHASE_COLLECT:
            done = self._collected >= self._quota or self._phase_elapsed >= self._dwell_ms
            if done:
                self._finish_point()
        self.update()

    def _begin_point(self, index: int) -> None:
        self._index = index
        self._phase = PHASE_SETTLE
        self._phase_elapsed = 0
        self._collected = 0

    def _finish_point(self) -> None:
        usable = self._session.count_for_point(self._index)
        logger.debug("Calibration point %d collected %d samples", self._index + 1, usable)
        if self._index + 1 < len(self._targets):
            self._begin_point(self._index + 1)
        else:
            self._complete()

    # -------------------------------------------------------------- finish
    def _complete(self) -> None:
        self._phase = PHASE_DONE
        self._timer.stop()
        try:
            estimator = self._session.fit()
        except ValueError as exc:
            logger.error("Calibration failed: %s", exc)
            self.close()
            self.failed.emit(str(exc))
            return
        profile = self._session.to_profile(
            estimator, self._monitor_index, self._camera_index, self._pattern_name
        )
        self.close()
        self.completed.emit(profile)

    def _cancel(self) -> None:
        self._timer.stop()
        self.close()
        self.cancelled.emit()

    # ---------------------------------------------------------------- input
    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self._cancel()
        elif event.key() == Qt.Key_Space and self._phase == PHASE_INTRO:
            self._begin_point(0)
        elif event.key() == Qt.Key_R and self._phase in (PHASE_SETTLE, PHASE_COLLECT):
            # Redo the current point.
            self._session.clear_point(self._index)
            self._begin_point(self._index)
        else:
            super().keyPressEvent(event)

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(16, 18, 22))

        if self._phase == PHASE_INTRO:
            self._paint_intro(painter)
        elif self._phase in (PHASE_SETTLE, PHASE_COLLECT):
            self._paint_target(painter)
            self._paint_status(painter)
        painter.end()

    def _paint_intro(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(240, 244, 250)))
        painter.setFont(QFont("Segoe UI", 30, QFont.DemiBold))
        painter.drawText(self.rect().adjusted(0, -120, 0, -120), Qt.AlignCenter,
                         "Calibration")

        painter.setFont(QFont("Segoe UI", 15))
        painter.setPen(QPen(QColor(180, 190, 205)))
        lines = (
            "Look directly at each dot until its ring fills.\n\n"
            "While you look, keep your eyes locked on the dot and let your\n"
            "head drift gently -- a little left and right, nearer and further.\n"
            "This is important: it is what teaches the tracker to stay accurate\n"
            "when you shift in your chair later. Holding perfectly still makes\n"
            "the tracker fragile.\n\n"
            "Sit at your normal playing distance.\n\n"
            f"{len(self._targets)} points, about "
            f"{len(self._targets) * (self._settle_ms + self._dwell_ms) / 1000:.0f} seconds.\n\n"
            "Space to begin now  -  R to redo a point  -  Esc to cancel"
        )
        painter.drawText(self.rect().adjusted(0, 60, 0, 60), Qt.AlignCenter, lines)

    def _paint_target(self, painter: QPainter) -> None:
        import math

        target = self._targets[self._index]
        center = QPointF(target[0] - self._screen_rect.x, target[1] - self._screen_rect.y)

        pulse = 1.0 + 0.18 * math.sin(self._animation * 2.2)
        outer = self._radius * 2.6 * (pulse if self._phase == PHASE_SETTLE else 1.0)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 70, 70, 40))
        painter.drawEllipse(center, outer, outer)

        if self._phase == PHASE_COLLECT:
            progress = min(1.0, self._collected / max(self._quota, 1))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(80, 220, 140), 4, Qt.SolidLine, Qt.RoundCap))
            span = int(-progress * 360 * 16)
            painter.drawArc(int(center.x() - outer), int(center.y() - outer),
                            int(outer * 2), int(outer * 2), 90 * 16, span)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 60, 60))
        painter.drawEllipse(center, self._radius, self._radius)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(center, max(2.0, self._radius * 0.22),
                            max(2.0, self._radius * 0.22))

    def _paint_status(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 14))
        painter.setPen(QPen(QColor(200, 208, 220)))
        text = f"Calibration {self._index + 1} of {len(self._targets)}"
        painter.drawText(self.rect().adjusted(0, 0, 0, -60), Qt.AlignHCenter | Qt.AlignBottom,
                         text)

        if self._phase == PHASE_COLLECT and self._head_motion_hint:
            painter.setPen(QPen(QColor(150, 205, 255)))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect().adjusted(0, 0, 0, -88),
                             Qt.AlignHCenter | Qt.AlignBottom,
                             "Eyes on the dot - let your head move gently")

        if not self._face_present:
            painter.setPen(QPen(QColor(255, 170, 70)))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(self.rect().adjusted(0, 0, 0, -30),
                             Qt.AlignHCenter | Qt.AlignBottom,
                             "Face not detected - check your lighting and camera")
