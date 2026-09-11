"""Screen analysis worker.

Board detection is far more expensive than gaze estimation and the board only
moves when the user resizes or scrolls the window, so this runs on its own
slow thread (a couple of frames per second) instead of per webcam frame. The
last successful detection is cached and re-used in between.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal

from ..utils.geometry import Rect
from ..utils.screens import ScreenInfo
from .capture import ScreenCaptureError, ScreenCaptureManager
from .chessboard import ChessboardDetector, ChessboardRegion

logger = logging.getLogger(__name__)


class ScreenWorker(QThread):
    """Periodically captures the selected monitor and looks for the board."""

    board_detected = Signal(object)   # ChessboardRegion
    board_lost = Signal()
    error_occurred = Signal(str)

    def __init__(self, screen: ScreenInfo, interval: float = 0.5,
                 redetect_seconds: float = 20.0, parent=None) -> None:
        super().__init__(parent)
        self._mutex = QMutex()
        self._running = False
        self._screen = screen
        self._monitor_index = screen.index
        self._interval = max(interval, 0.1)
        self._redetect_seconds = redetect_seconds
        self._detect_requested = True
        self._enabled = True
        self._current: Optional[ChessboardRegion] = None
        self._detector = ChessboardDetector()

    # -------------------------------------------------------------- controls
    def request_detection(self) -> None:
        """Force a detection on the next cycle, ignoring the cache."""
        with QMutexLocker(self._mutex):
            self._detect_requested = True

    def set_enabled(self, enabled: bool) -> None:
        with QMutexLocker(self._mutex):
            self._enabled = bool(enabled)

    def set_screen(self, screen: ScreenInfo) -> None:
        with QMutexLocker(self._mutex):
            self._screen = screen
            self._monitor_index = screen.index
            self._detect_requested = True

    def _current_screen(self) -> ScreenInfo:
        with QMutexLocker(self._mutex):
            return self._screen

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False

    def _snapshot(self):
        with QMutexLocker(self._mutex):
            return (self._running, self._enabled, self._monitor_index,
                    self._detect_requested)

    def _clear_request(self) -> None:
        with QMutexLocker(self._mutex):
            self._detect_requested = False

    # ------------------------------------------------------------------ loop
    def run(self) -> None:  # noqa: D102
        with QMutexLocker(self._mutex):
            self._running = True

        capture = ScreenCaptureManager(self._monitor_index)
        last_detection = 0.0
        try:
            while True:
                running, enabled, _monitor_index, forced = self._snapshot()
                if not running:
                    break
                if not enabled:
                    self.msleep(int(self._interval * 1000))
                    continue

                now = time.monotonic()
                due = forced or self._current is None or \
                    (now - last_detection) >= self._redetect_seconds
                if not due:
                    self.msleep(int(self._interval * 1000))
                    continue

                self._clear_request()
                last_detection = now
                screen = self._current_screen()
                try:
                    # Capture in physical pixels, then report the result in
                    # logical pixels so the board rectangle lands in the same
                    # coordinate space as the gaze estimates.
                    frame = capture.grab_region(screen.physical_rect)
                    rect = screen.physical_rect
                except ScreenCaptureError as exc:
                    logger.warning("Screen capture failed: %s", exc)
                    self.error_occurred.emit(str(exc))
                    self.msleep(2000)
                    continue

                region = self._detector.detect(frame, rect)
                if region is None:
                    region = self._detector.detect_with_corner_finder(frame, rect)
                if region is not None:
                    logical = screen.to_logical(region.rect)
                    region = ChessboardRegion.from_rect(logical, region.confidence,
                                                        region.source)

                if region is not None:
                    self._current = region
                    self.board_detected.emit(region)
                elif self._current is not None:
                    self._current = None
                    self.board_lost.emit()
                else:
                    self.board_lost.emit()

                self.msleep(int(self._interval * 1000))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Screen worker crashed")
            self.error_occurred.emit(f"Screen analysis stopped: {exc}")
        finally:
            capture.close()
            with QMutexLocker(self._mutex):
                self._running = False
            logger.info("Screen worker finished")

    @property
    def current_board(self) -> Optional[Rect]:
        return self._current.rect if self._current else None
