"""Screen capture.

Captured images live in memory only. No screenshot is ever written to disk:
frames are grabbed, analysed for the board, and discarded.

``mss`` instances are not thread-safe, so :class:`ScreenCaptureManager` creates
its capture object lazily inside whichever thread first calls :meth:`grab`.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..utils.geometry import Rect

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonitorInfo:
    """A physical monitor in virtual-desktop coordinates.

    ``index`` matches the key used by ``mss``: 0 is the union of all monitors,
    1 is the primary monitor, and so on.
    """

    index: int
    x: int
    y: int
    width: int
    height: int

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.width, self.height)

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def origin(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def label(self) -> str:
        kind = "All monitors" if self.index == 0 else f"Monitor {self.index}"
        return f"{kind} - {self.width}x{self.height} at ({self.x}, {self.y})"

    def to_mss_dict(self) -> dict:
        return {"left": self.x, "top": self.y, "width": self.width, "height": self.height}


class ScreenCaptureError(RuntimeError):
    """Raised when the desktop cannot be captured."""


def list_monitors() -> List[MonitorInfo]:
    """Enumerate monitors. Returns an empty list if capture is unavailable."""
    try:
        import mss
    except ImportError as exc:  # pragma: no cover - dependency missing
        logger.error("mss is not installed: %s", exc)
        return []
    try:
        with mss.mss() as sct:
            return [
                MonitorInfo(index, int(m["left"]), int(m["top"]),
                            int(m["width"]), int(m["height"]))
                for index, m in enumerate(sct.monitors)
            ]
    except Exception as exc:  # pragma: no cover - platform dependent
        logger.error("Could not enumerate monitors: %s", exc)
        return []


class ScreenCaptureManager:
    """Grabs frames from one monitor, or from a sub-region of it."""

    def __init__(self, monitor_index: int = 1) -> None:
        self.monitor_index = monitor_index
        self._sct = None
        self._owner_thread: Optional[int] = None
        self._monitor: Optional[MonitorInfo] = None

    # ------------------------------------------------------------- lifecycle
    def _ensure_open(self) -> None:
        current = threading.get_ident()
        if self._sct is not None and self._owner_thread == current:
            return
        self.close()
        try:
            import mss
            self._sct = mss.mss()
        except Exception as exc:
            raise ScreenCaptureError(f"Screen capture unavailable: {exc}") from exc
        self._owner_thread = current

        monitors = self._sct.monitors
        if self.monitor_index >= len(monitors):
            logger.warning("Monitor %d not present; falling back to primary",
                           self.monitor_index)
            self.monitor_index = 1 if len(monitors) > 1 else 0
        entry = monitors[self.monitor_index]
        self._monitor = MonitorInfo(self.monitor_index, int(entry["left"]), int(entry["top"]),
                                    int(entry["width"]), int(entry["height"]))

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:  # pragma: no cover
                pass
        self._sct = None
        self._owner_thread = None

    def __enter__(self) -> "ScreenCaptureManager":
        self._ensure_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ info
    @property
    def monitor(self) -> MonitorInfo:
        self._ensure_open()
        assert self._monitor is not None
        return self._monitor

    def set_monitor(self, index: int) -> None:
        if index != self.monitor_index:
            self.monitor_index = index
            self.close()

    def grab_region(self, physical_rect: Rect,
                    max_width: Optional[int] = None) -> np.ndarray:
        """Capture an explicit region given in PHYSICAL pixels.

        Used by the screen worker, which already knows the monitor's physical
        geometry from Qt and does not need this class to resolve it. Keeping
        the physical rectangle explicit avoids relying on mss and Qt agreeing
        on monitor ordering, which they do not always do.
        """
        self._ensure_open()
        assert self._sct is not None
        box = {"left": int(physical_rect.x), "top": int(physical_rect.y),
               "width": max(int(physical_rect.width), 1),
               "height": max(int(physical_rect.height), 1)}
        try:
            shot = self._sct.grab(box)
        except Exception as exc:
            raise ScreenCaptureError(f"Screen capture failed: {exc}") from exc

        frame = np.asarray(shot)[:, :, :3]
        if max_width and frame.shape[1] > max_width:
            import cv2
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, (max_width, max(int(frame.shape[0] * scale), 1)),
                               interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame)

    # ---------------------------------------------------------------- frames
    def grab(self, region: Optional[Rect] = None,
             max_width: Optional[int] = None) -> Tuple[np.ndarray, Rect]:
        """Capture a BGR frame.

        Returns the image together with the virtual-desktop rectangle it covers,
        so detections can be mapped straight back to screen coordinates. When
        ``max_width`` is given the image is downscaled, but the returned
        rectangle still describes the original screen area.
        """
        self._ensure_open()
        assert self._sct is not None and self._monitor is not None

        area = region if region is not None else self._monitor.rect
        area = area.intersection(self._monitor.rect)
        if area.width < 2 or area.height < 2:
            raise ScreenCaptureError("Capture region is empty")

        box = {"left": int(area.x), "top": int(area.y),
               "width": int(area.width), "height": int(area.height)}
        try:
            shot = self._sct.grab(box)
        except Exception as exc:
            raise ScreenCaptureError(f"Screen capture failed: {exc}") from exc

        frame = np.asarray(shot)[:, :, :3]  # BGRA -> BGR
        if max_width and frame.shape[1] > max_width:
            import cv2
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, (max_width, max(int(frame.shape[0] * scale), 1)),
                               interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame), area
