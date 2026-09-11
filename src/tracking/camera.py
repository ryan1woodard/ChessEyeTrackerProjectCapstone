"""Webcam access.

``CameraManager`` owns exactly one :class:`cv2.VideoCapture` and is only ever
used from the tracking worker thread -- never from the GUI thread.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or has permanently failed."""


@dataclass(frozen=True)
class CameraInfo:
    index: int
    name: str
    width: int
    height: int

    def label(self) -> str:
        return f"{self.name} ({self.width}x{self.height})"


def _preferred_backend() -> int:
    """DirectShow is markedly faster to open than MSMF on Windows."""
    if platform.system() == "Windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def enumerate_cameras(max_index: int = 6) -> List[CameraInfo]:
    """Probe device indices and return the ones that yield a frame.

    OpenCV exposes no portable way to read a friendly device name, so the name
    is synthesised. Probing is comparatively slow (a few hundred ms per index),
    so callers should cache the result.
    """
    found: List[CameraInfo] = []
    backend = _preferred_backend()

    # Probing absent indices makes OpenCV print a wall of driver errors to the
    # console. They are expected and harmless, so silence them for the probe.
    try:
        previous_log_level = cv2.getLogLevel()
        cv2.setLogLevel(0)
    except AttributeError:  # pragma: no cover - very old OpenCV
        previous_log_level = None

    for index in range(max_index):
        capture = cv2.VideoCapture(index, backend)
        try:
            if not capture.isOpened():
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            found.append(CameraInfo(index, f"Camera {index}", int(width), int(height)))
        except cv2.error as exc:  # pragma: no cover - driver dependent
            logger.debug("Probe of camera %d failed: %s", index, exc)
        finally:
            capture.release()

    if previous_log_level is not None:
        cv2.setLogLevel(previous_log_level)
    logger.info("Detected %d camera(s)", len(found))
    return found


class CameraManager:
    """Opens a webcam, delivers frames, and recovers from transient failures."""

    MAX_CONSECUTIVE_FAILURES = 30

    def __init__(self, device_index: int = 0, width: int = 1280,
                 height: int = 720, fps: int = 30) -> None:
        self.device_index = device_index
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self._capture: Optional[cv2.VideoCapture] = None
        self._failures = 0
        self._actual_size: Tuple[int, int] = (0, 0)

    # ------------------------------------------------------------- lifecycle
    def open(self) -> None:
        self.release()
        capture = cv2.VideoCapture(self.device_index, _preferred_backend())
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open camera {self.device_index}. It may be missing, "
                f"disabled, or in use by another application."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        capture.set(cv2.CAP_PROP_FPS, self.requested_fps)
        # A small buffer keeps latency low; not all drivers honour this.
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:  # pragma: no cover - backend dependent
            pass

        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise CameraError(
                f"Camera {self.device_index} opened but returned no frames. "
                f"Another application may be holding it."
            )
        self._actual_size = (int(frame.shape[1]), int(frame.shape[0]))
        self._capture = capture
        self._failures = 0
        logger.info("Camera %d opened at %dx%d", self.device_index, *self._actual_size)

    def release(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except cv2.error:  # pragma: no cover
                pass
            logger.info("Camera %d released", self.device_index)
        self._capture = None

    def __enter__(self) -> "CameraManager":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    # ----------------------------------------------------------------- state
    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def frame_size(self) -> Tuple[int, int]:
        return self._actual_size

    # ---------------------------------------------------------------- frames
    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or ``None`` if this read failed.

        Transient failures are tolerated; after ``MAX_CONSECUTIVE_FAILURES`` a
        reconnect is attempted, and if that fails a :class:`CameraError` is
        raised so the worker can report a hard disconnect to the UI.
        """
        if self._capture is None:
            raise CameraError("Camera is not open")
        ok, frame = self._capture.read()
        if ok and frame is not None:
            self._failures = 0
            return frame

        self._failures += 1
        if self._failures >= self.MAX_CONSECUTIVE_FAILURES:
            logger.warning("Camera %d unresponsive; attempting reconnect", self.device_index)
            time.sleep(0.4)
            try:
                self.open()
            except CameraError as exc:
                raise CameraError(f"Camera disconnected: {exc}") from exc
        return None
