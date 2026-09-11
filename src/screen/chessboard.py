"""Visual chessboard detection.

The application is a passive observer: it never touches chess.com, reads no
game state and injects nothing. The board is located purely from pixels.

Algorithm
---------
1. Downscale the screenshot (detection does not need full resolution).
2. Find edges and extract quadrilateral contours that are roughly square and
   large enough to plausibly be a board.
3. Score every candidate by warping it to a fixed square and correlating the
   64 cell brightnesses against the ideal alternating pattern. This is what
   makes the detector theme-independent: it does not care about colours, only
   that light and dark cells alternate.
4. Return the best-scoring candidate above a threshold.

Pieces sit on top of the squares and degrade the correlation, which is why the
acceptance threshold is fairly forgiving. Detection is a *convenience*: when it
fails the user selects the board by hand, which is always exact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..utils.geometry import Rect

logger = logging.getLogger(__name__)

DETECTION_WIDTH = 960
MIN_AREA_FRACTION = 0.01
MAX_AREA_FRACTION = 0.85
MIN_ASPECT = 0.80
MAX_ASPECT = 1.25
DEFAULT_SCORE_THRESHOLD = 0.32


@dataclass(frozen=True)
class ChessboardRegion:
    """A detected board in virtual-desktop pixel coordinates."""

    x: float
    y: float
    width: float
    height: float
    confidence: float
    source: str = "auto"

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.width, self.height)

    @classmethod
    def from_rect(cls, rect: Rect, confidence: float = 1.0,
                  source: str = "manual") -> "ChessboardRegion":
        return cls(rect.x, rect.y, rect.width, rect.height, confidence, source)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height,
                "confidence": self.confidence, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict) -> "ChessboardRegion":
        return cls(float(data["x"]), float(data["y"]), float(data["width"]),
                   float(data["height"]), float(data.get("confidence", 1.0)),
                   str(data.get("source", "manual")))


def cell_levels(gray: np.ndarray) -> np.ndarray:
    """Representative intensity of each of the 8x8 cells.

    Sampling the *centre* of a cell is the obvious approach and the wrong one:
    on a real board the centre is exactly where the piece stands, so half the
    cells would report the piece's colour rather than the square's.

    Instead each cell is sampled around its border -- inside the grid line but
    outside the area a piece occupies -- and reduced with a median so that any
    part of a piece which does overlap the ring is ignored.
    """
    size = gray.shape[0]
    step = size / 8.0
    levels = np.zeros((8, 8), dtype=np.float64)
    edge = max(int(step * 0.10), 1)   # skip grid lines and cell borders
    band = max(int(step * 0.20), 1)   # thickness of the sampled ring

    for row in range(8):
        for col in range(8):
            y0 = int(row * step) + edge
            y1 = int((row + 1) * step) - edge
            x0 = int(col * step) + edge
            x1 = int((col + 1) * step) - edge
            if y1 - y0 < 3 or x1 - x0 < 3:
                patch = gray[max(y0, 0):max(y1, y0 + 1), max(x0, 0):max(x1, x0 + 1)]
                levels[row, col] = float(np.median(patch)) if patch.size else 0.0
                continue
            cell = gray[y0:y1, x0:x1]
            ring = np.concatenate([
                cell[:band, :].ravel(),
                cell[-band:, :].ravel(),
                cell[:, :band].ravel(),
                cell[:, -band:].ravel(),
            ])
            levels[row, col] = float(np.median(ring)) if ring.size else 0.0
    return levels


def checker_score(gray_square: np.ndarray) -> float:
    """Correlation between cell brightnesses and an alternating pattern.

    Returns a value in ``[0, 1]``; a clean empty board scores close to 1.0 and
    unstructured content scores near 0.
    """
    if gray_square.shape[0] < 32 or gray_square.shape[1] < 32:
        return 0.0
    means = cell_levels(gray_square)
    rows, cols = np.indices((8, 8))
    pattern = np.where((rows + cols) % 2 == 0, 1.0, -1.0)

    flat_means = means.flatten() - means.mean()
    flat_pattern = pattern.flatten()
    denominator = np.linalg.norm(flat_means) * np.linalg.norm(flat_pattern)
    if denominator < 1e-9:
        return 0.0
    return float(abs(np.dot(flat_means, flat_pattern) / denominator))


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = points.sum(axis=1)
    diff = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(total)]
    ordered[2] = points[np.argmax(total)]
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]
    return ordered


def _warp_to_square(gray: np.ndarray, corners: np.ndarray, size: int = 256) -> np.ndarray:
    destination = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
                           dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(_order_corners(corners), destination)
    return cv2.warpPerspective(gray, matrix, (size, size))


def _quad_candidates(gray: np.ndarray) -> List[np.ndarray]:
    """Extract square-ish convex quadrilateral contours."""
    height, width = gray.shape[:2]
    image_area = float(width * height)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(blurred, lower, upper)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * MIN_AREA_FRACTION or area > image_area * MAX_AREA_FRACTION:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        points = approx.reshape(4, 2).astype(np.float32)
        x, y, w, h = cv2.boundingRect(approx)
        if w < 40 or h < 40:
            continue
        aspect = w / float(h)
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        # Reject contours that fill their bounding box poorly (not a rectangle).
        if area < 0.7 * w * h:
            continue
        candidates.append(points)
    return candidates


class ChessboardDetector:
    """Locates a chessboard in a screenshot."""

    def __init__(self, score_threshold: float = DEFAULT_SCORE_THRESHOLD,
                 detection_width: int = DETECTION_WIDTH) -> None:
        self.score_threshold = score_threshold
        self.detection_width = detection_width

    def detect(self, frame_bgr: np.ndarray,
               frame_rect: Optional[Rect] = None) -> Optional[ChessboardRegion]:
        """Detect a board and return it in virtual-desktop coordinates.

        ``frame_rect`` is the screen area the frame covers; when omitted the
        result is returned in frame-local coordinates.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        scale = 1.0
        gray = gray_full
        if gray.shape[1] > self.detection_width:
            scale = self.detection_width / gray.shape[1]
            gray = cv2.resize(gray, (self.detection_width, int(gray.shape[0] * scale)),
                              interpolation=cv2.INTER_AREA)

        best_rect: Optional[Tuple[float, float, float, float]] = None
        best_score = 0.0
        for corners in _quad_candidates(gray):
            warped = _warp_to_square(gray, corners)
            score = checker_score(warped)
            if score > best_score:
                x, y, w, h = cv2.boundingRect(corners.astype(np.int32))
                best_score = score
                best_rect = (float(x), float(y), float(w), float(h))

        if best_rect is None or best_score < self.score_threshold:
            logger.debug("No chessboard found (best score %.2f)", best_score)
            return None

        inv = 1.0 / scale
        x, y, w, h = (value * inv for value in best_rect)
        if frame_rect is not None:
            x += frame_rect.x
            y += frame_rect.y
        confidence = float(min(1.0, best_score / 0.8))
        logger.info("Chessboard detected at (%.0f, %.0f) %.0fx%.0f, score %.2f",
                    x, y, w, h, best_score)
        return ChessboardRegion(x, y, w, h, confidence, source="auto")

    def detect_with_corner_finder(self, frame_bgr: np.ndarray,
                                  frame_rect: Optional[Rect] = None
                                  ) -> Optional[ChessboardRegion]:
        """Secondary detector using OpenCV's inner-corner finder.

        This is precise on a mostly empty board but fails once pieces occlude
        the inner corners, so it is only tried as a fallback.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (7, 7),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK,
        )
        if not found or corners is None:
            return None
        points = corners.reshape(-1, 2)
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        # The 7x7 inner corners span 6 cells; extend by one cell on each side.
        cell_w = (x1 - x0) / 6.0
        cell_h = (y1 - y0) / 6.0
        rect = Rect(float(x0 - cell_w), float(y0 - cell_h),
                    float(x1 - x0 + 2 * cell_w), float(y1 - y0 + 2 * cell_h))
        if frame_rect is not None:
            rect = Rect(rect.x + frame_rect.x, rect.y + frame_rect.y, rect.width, rect.height)
        return ChessboardRegion.from_rect(rect, confidence=0.9, source="corners")
