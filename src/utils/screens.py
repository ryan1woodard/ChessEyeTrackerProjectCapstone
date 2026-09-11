"""Screen enumeration and the logical/physical coordinate split.

Why this module exists
----------------------
Windows display scaling (125%, 150%, 175%) makes "screen coordinates"
ambiguous, and mixing the two conventions silently destroys gaze accuracy.

* **Logical pixels** are what Qt uses. Everything Qt draws or positions --
  the calibration targets, the gaze overlay, the board selector -- lives here.
  On a 1920x1080 monitor at 150% scaling, Qt reports 1280x720.
* **Physical pixels** are what the display actually has, and what ``mss``
  returns when capturing. The same monitor reports 1920x1080.

The application uses **logical pixels everywhere** for gaze, calibration,
regions and squares, and converts to physical only at the moment of screen
capture. That single convention is what keeps calibration targets, the red dot
and the detected board in the same space.

Getting this wrong is not a subtle error: at 150% scaling a model calibrated in
physical coordinates predicts positions 1.5x too large, so almost every
estimate lands off screen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from .geometry import Rect

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScreenInfo:
    """One monitor, in both coordinate systems.

    ``index`` starts at 1 to match how monitors are labelled in the UI and how
    ``mss`` numbers them (0 being the union of all monitors).
    """

    index: int
    name: str
    rect: Rect                  # logical pixels (Qt)
    device_pixel_ratio: float   # logical -> physical multiplier

    @property
    def physical_rect(self) -> Rect:
        """The same monitor in physical pixels, for screen capture."""
        ratio = self.device_pixel_ratio
        return Rect(self.rect.x * ratio, self.rect.y * ratio,
                    self.rect.width * ratio, self.rect.height * ratio)

    @property
    def size(self) -> tuple[int, int]:
        return (int(self.rect.width), int(self.rect.height))

    @property
    def origin(self) -> tuple[int, int]:
        return (int(self.rect.x), int(self.rect.y))

    @property
    def is_scaled(self) -> bool:
        return abs(self.device_pixel_ratio - 1.0) > 0.01

    def to_physical(self, rect: Rect) -> Rect:
        """Convert a logical rectangle to physical pixels."""
        ratio = self.device_pixel_ratio
        return Rect(rect.x * ratio, rect.y * ratio,
                    rect.width * ratio, rect.height * ratio)

    def to_logical(self, rect: Rect) -> Rect:
        """Convert a physical rectangle back to logical pixels."""
        ratio = self.device_pixel_ratio or 1.0
        return Rect(rect.x / ratio, rect.y / ratio,
                    rect.width / ratio, rect.height / ratio)

    def label(self) -> str:
        scaling = f"  @{self.device_pixel_ratio * 100:.0f}%" if self.is_scaled else ""
        return (f"Monitor {self.index} - {int(self.rect.width)}x{int(self.rect.height)}"
                f" at ({int(self.rect.x)}, {int(self.rect.y)}){scaling}")


def list_screens() -> List[ScreenInfo]:
    """Enumerate monitors through Qt. Requires a running QGuiApplication."""
    from PySide6.QtGui import QGuiApplication

    application = QGuiApplication.instance()
    if application is None:  # pragma: no cover - misuse
        logger.error("list_screens() called before a QApplication exists")
        return []

    primary = QGuiApplication.primaryScreen()
    screens = list(QGuiApplication.screens())
    # Keep the primary monitor first so it becomes "Monitor 1".
    if primary is not None and primary in screens:
        screens.remove(primary)
        screens.insert(0, primary)

    infos: List[ScreenInfo] = []
    for index, screen in enumerate(screens, start=1):
        geometry = screen.geometry()
        info = ScreenInfo(
            index=index,
            name=screen.name() or f"Monitor {index}",
            rect=Rect(geometry.x(), geometry.y(), geometry.width(), geometry.height()),
            device_pixel_ratio=float(screen.devicePixelRatio()),
        )
        infos.append(info)
        if info.is_scaled:
            logger.info("Monitor %d is scaled at %.0f%% (logical %dx%d, physical %dx%d)",
                        info.index, info.device_pixel_ratio * 100,
                        int(info.rect.width), int(info.rect.height),
                        int(info.physical_rect.width), int(info.physical_rect.height))
    return infos


def find_screen(screens: List[ScreenInfo], index: int) -> Optional[ScreenInfo]:
    for screen in screens:
        if screen.index == index:
            return screen
    return screens[0] if screens else None


def qt_screen_for(index: int):
    """Return the ``QScreen`` matching a :class:`ScreenInfo` index."""
    from PySide6.QtGui import QGuiApplication

    screens = list_screens()
    target = find_screen(screens, index)
    if target is None:
        return QGuiApplication.primaryScreen()
    for screen in QGuiApplication.screens():
        geometry = screen.geometry()
        if (geometry.x(), geometry.y()) == (int(target.rect.x), int(target.rect.y)):
            return screen
    return QGuiApplication.primaryScreen()
