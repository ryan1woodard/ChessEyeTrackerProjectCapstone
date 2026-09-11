"""Live gaze heatmap accumulation and rendering.

Two heatmaps are maintained side by side, because they answer different
questions:

* a **screen** heatmap, accumulated on a coarse grid by splatting a Gaussian at
  each gaze sample, which shows broadly where attention pools;
* a **board** heatmap, accumulated as seconds per square, which is the one that
  is actually useful for chess.

Design notes
------------
The grid is deliberately coarse (96 cells wide by default) and scaled up when
painted. Accumulating at full screen resolution would be wasteful and, worse,
misleading: gaze estimates carry 50-150 px of error, so resolving the heatmap
more finely than that would imply precision the data does not have. The
Gaussian splat radius is tied to that same error.

Rendering is throttled and cached by the caller -- a repaint of the overlay
must stay cheap, because it happens on every frame.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np

from ..utils.geometry import Rect

logger = logging.getLogger(__name__)

MODE_SCREEN = "screen"
MODE_BOARD = "board"
MODE_BOTH = "both"
HEATMAP_MODES = (MODE_SCREEN, MODE_BOARD, MODE_BOTH)

#: Gradient stops as (position, R, G, B). Alpha is derived from intensity, so
#: cold areas fade out rather than washing the screen in blue.
_GRADIENT = (
    (0.00, 0, 40, 190),
    (0.35, 0, 190, 200),
    (0.60, 90, 220, 90),
    (0.80, 250, 210, 60),
    (1.00, 255, 45, 45),
)


def _build_colour_table(size: int = 256) -> np.ndarray:
    """Precompute an RGB lookup table for the gradient above."""
    table = np.zeros((size, 3), dtype=np.float64)
    positions = np.linspace(0.0, 1.0, size)
    stops = list(_GRADIENT)
    for channel in range(3):
        table[:, channel] = np.interp(
            positions,
            [stop[0] for stop in stops],
            [stop[channel + 1] for stop in stops],
        )
    return table.astype(np.uint8)


_COLOUR_TABLE = _build_colour_table()


class GazeHeatmap:
    """Accumulates gaze density over a screen region and over board squares."""

    def __init__(self, rect: Rect, grid_width: int = 96, sigma_px: float = 55.0,
                 half_life_seconds: float = 0.0) -> None:
        self.sigma_px = sigma_px
        self.half_life_seconds = half_life_seconds
        self._square_seconds: Dict[str, float] = {}
        self._last_decay: Optional[float] = None
        self._dirty = True
        self.set_rect(rect, grid_width)

    # ------------------------------------------------------------- geometry
    def set_rect(self, rect: Rect, grid_width: Optional[int] = None) -> None:
        """Point the heatmap at a new region, discarding all existing data.

        Board dwell times are cleared as well as the grid: a different monitor
        means a different board rectangle, so the old per-square totals no
        longer refer to anything real.
        """
        self.rect = rect
        if grid_width is not None:
            self.grid_width = max(int(grid_width), 8)
        aspect = rect.height / rect.width if rect.width > 0 else 0.6
        self.grid_height = max(int(round(self.grid_width * aspect)), 8)
        self._grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float64)
        self._kernel = self._build_kernel()
        self._square_seconds.clear()
        self._last_decay = None
        self._dirty = True

    def _build_kernel(self) -> np.ndarray:
        """A normalised Gaussian sized from the gaze error, in grid cells."""
        cells_per_px = self.grid_width / max(self.rect.width, 1.0)
        sigma = max(self.sigma_px * cells_per_px, 0.6)
        radius = max(int(math.ceil(sigma * 2.5)), 1)
        axis = np.arange(-radius, radius + 1, dtype=np.float64)
        gauss = np.exp(-(axis ** 2) / (2.0 * sigma ** 2))
        kernel = np.outer(gauss, gauss)
        total = kernel.sum()
        return kernel / total if total > 0 else kernel

    # ---------------------------------------------------------- accumulation
    def add(self, x: float, y: float, weight: float = 1.0,
            timestamp: Optional[float] = None) -> None:
        """Splat one gaze sample onto the grid."""
        if weight <= 0 or self.rect.width <= 0 or self.rect.height <= 0:
            return
        self._apply_decay(timestamp)

        col = (x - self.rect.x) / self.rect.width * self.grid_width
        row = (y - self.rect.y) / self.rect.height * self.grid_height
        if not (math.isfinite(col) and math.isfinite(row)):
            return
        col, row = int(round(col)), int(round(row))

        radius = self._kernel.shape[0] // 2
        top, left = row - radius, col - radius
        bottom, right = top + self._kernel.shape[0], left + self._kernel.shape[1]

        # Clip the kernel against the grid edges so looking near a corner still
        # accumulates rather than being dropped.
        grid_top, grid_left = max(top, 0), max(left, 0)
        grid_bottom = min(bottom, self.grid_height)
        grid_right = min(right, self.grid_width)
        if grid_top >= grid_bottom or grid_left >= grid_right:
            return

        patch = self._kernel[grid_top - top:grid_bottom - top,
                             grid_left - left:grid_right - left]
        self._grid[grid_top:grid_bottom, grid_left:grid_right] += patch * weight
        self._dirty = True

    def add_square_time(self, square: Optional[str], seconds: float) -> None:
        """Accumulate dwell time for a board square."""
        if not square or seconds <= 0:
            return
        self._square_seconds[square] = self._square_seconds.get(square, 0.0) + seconds
        self._dirty = True

    def _apply_decay(self, timestamp: Optional[float]) -> None:
        """Optionally fade older data so the map tracks recent attention."""
        if self.half_life_seconds <= 0 or timestamp is None:
            self._last_decay = timestamp
            return
        if self._last_decay is None:
            self._last_decay = timestamp
            return
        elapsed = timestamp - self._last_decay
        if elapsed <= 0:
            return
        self._last_decay = timestamp
        factor = 0.5 ** (elapsed / self.half_life_seconds)
        self._grid *= factor
        for key in list(self._square_seconds):
            self._square_seconds[key] *= factor

    def clear(self) -> None:
        self._grid.fill(0.0)
        self._square_seconds.clear()
        self._last_decay = None
        self._dirty = True

    # -------------------------------------------------------------- readout
    @property
    def is_empty(self) -> bool:
        return not self._square_seconds and self._grid.max() <= 0.0

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_clean(self) -> None:
        self._dirty = False

    @property
    def square_seconds(self) -> Dict[str, float]:
        return dict(self._square_seconds)

    @property
    def total_samples(self) -> float:
        return float(self._grid.sum())

    def normalised_grid(self, percentile: float = 99.0,
                        gamma: float = 0.5) -> np.ndarray:
        """Scale the grid into 0-1 for display.

        Two competing failure modes have to be balanced. Normalising by the
        maximum lets one long fixation crush everything else to invisibility.
        Normalising purely by a high percentile does the opposite: the
        reference falls well below the peak, so a broad region saturates and
        the hotspot renders as a flat, oversized blob that overstates how
        precisely the position is known.

        So the reference is the percentile, floored at a fraction of the peak,
        and the gamma then lifts mid-range values so moderately-viewed areas
        remain legible beside a hotspot.
        """
        peak = float(self._grid.max())
        if peak <= 0:
            return np.zeros_like(self._grid)
        reference = max(float(np.percentile(self._grid, percentile)), peak * 0.4)
        scaled = np.clip(self._grid / reference, 0.0, 1.0)
        return np.power(scaled, gamma)

    def peak_cell(self) -> Optional[Tuple[int, int]]:
        """Grid ``(row, column)`` of the most-looked-at cell."""
        if self._grid.max() <= 0:
            return None
        row, col = np.unravel_index(int(np.argmax(self._grid)), self._grid.shape)
        return int(row), int(col)

    def peak_position(self) -> Optional[Tuple[float, float]]:
        """Screen coordinates of the most-looked-at point."""
        cell = self.peak_cell()
        if cell is None:
            return None
        row, col = cell
        x = self.rect.x + (col + 0.5) / self.grid_width * self.rect.width
        y = self.rect.y + (row + 0.5) / self.grid_height * self.rect.height
        return float(x), float(y)

    # -------------------------------------------------------------- colours
    def to_rgba(self, opacity: float = 0.55, floor: float = 0.06) -> np.ndarray:
        """Render the grid to an ``(H, W, 4)`` RGBA array.

        Cells below ``floor`` are left fully transparent so the untouched parts
        of the screen stay completely clear.
        """
        intensity = self.normalised_grid()
        indices = np.clip((intensity * 255).astype(np.int32), 0, 255)
        rgba = np.zeros((self.grid_height, self.grid_width, 4), dtype=np.uint8)
        rgba[..., :3] = _COLOUR_TABLE[indices]

        alpha = np.clip((intensity - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0)
        rgba[..., 3] = (alpha * np.clip(opacity, 0.0, 1.0) * 255).astype(np.uint8)
        return rgba


def rgba_to_qimage(rgba: np.ndarray):
    """Convert an RGBA array to a QImage that owns its own memory."""
    from PySide6.QtGui import QImage

    rgba = np.ascontiguousarray(rgba)
    height, width = rgba.shape[:2]
    image = QImage(rgba.data, width, height, 4 * width, QImage.Format_RGBA8888)
    return image.copy()


def square_colour(fraction: float) -> Tuple[int, int, int]:
    """Gradient colour for a board square, given its share of the maximum."""
    index = int(np.clip(fraction, 0.0, 1.0) * 255)
    r, g, b = _COLOUR_TABLE[index]
    return int(r), int(g), int(b)
