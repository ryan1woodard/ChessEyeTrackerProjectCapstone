"""Debug rendering of the webcam preview.

Drawing happens on a copy of the frame in the GUI thread; the worker's own
buffers are never touched.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ..tracking import face_tracker as fl
from ..tracking.face_tracker import FaceLandmarks
from ..tracking.pipeline import GazeSample

_GREEN = (110, 230, 120)
_CYAN = (255, 210, 80)
_RED = (60, 60, 255)
_WHITE = (245, 245, 245)


def draw_debug_frame(
    frame_bgr: np.ndarray,
    sample: Optional[GazeSample],
    landmarks: Optional[FaceLandmarks] = None,
    show_face: bool = True,
    show_eyes: bool = True,
    show_pose: bool = True,
    mirror: bool = True,
) -> np.ndarray:
    """Return an annotated copy of a webcam frame."""
    image = frame_bgr.copy()

    if landmarks is not None:
        if show_face:
            points = landmarks.pixels(range(0, 468, 4)).astype(int)
            for x, y in points:
                cv2.circle(image, (int(x), int(y)), 1, _GREEN, -1, cv2.LINE_AA)
        if show_eyes and landmarks.has_iris:
            _draw_eyes(image, landmarks)
        if show_pose and sample is not None and sample.head_pose.valid:
            _draw_pose_axes(image, landmarks, sample)

    if sample is not None:
        _draw_hud(image, sample)

    if mirror:
        image = cv2.flip(image, 1)
    return image


def _draw_eyes(image: np.ndarray, landmarks: FaceLandmarks) -> None:
    for corners, lids, iris in (
        ((fl.EYE_LEFT_OUTER, fl.EYE_LEFT_INNER),
         (fl.EYE_LEFT_TOP, fl.EYE_LEFT_BOTTOM), fl.IRIS_LEFT),
        ((fl.EYE_RIGHT_OUTER, fl.EYE_RIGHT_INNER),
         (fl.EYE_RIGHT_TOP, fl.EYE_RIGHT_BOTTOM), fl.IRIS_RIGHT),
    ):
        a = landmarks.pixel(corners[0]).astype(int)
        b = landmarks.pixel(corners[1]).astype(int)
        top = landmarks.pixel(lids[0]).astype(int)
        bottom = landmarks.pixel(lids[1]).astype(int)
        cv2.line(image, tuple(a), tuple(b), _CYAN, 1, cv2.LINE_AA)
        cv2.line(image, tuple(top), tuple(bottom), _CYAN, 1, cv2.LINE_AA)

        iris_points = landmarks.pixels(iris)
        center = iris_points.mean(axis=0).astype(int)
        radius = int(np.linalg.norm(iris_points - iris_points.mean(axis=0), axis=1).max())
        cv2.circle(image, tuple(center), max(radius, 2), _RED, 1, cv2.LINE_AA)
        cv2.circle(image, tuple(center), 2, _RED, -1, cv2.LINE_AA)


def _draw_pose_axes(image: np.ndarray, landmarks: FaceLandmarks,
                    sample: GazeSample) -> None:
    """Draw a short yaw/pitch indicator from the nose tip."""
    origin = landmarks.pixel(fl.NOSE_TIP)
    length = landmarks.frame_width * 0.12
    yaw = np.radians(sample.head_pose.yaw)
    pitch = np.radians(sample.head_pose.pitch)
    tip = origin + np.array([np.sin(yaw), -np.sin(pitch)]) * length
    cv2.arrowedLine(image, tuple(origin.astype(int)), tuple(tip.astype(int)),
                    _WHITE, 2, cv2.LINE_AA, tipLength=0.25)


def _draw_hud(image: np.ndarray, sample: GazeSample) -> None:
    lines: List[str] = [
        f"FPS {sample.fps:4.1f}",
        f"Face {'yes' if sample.face_detected else 'no'}",
        f"Conf {sample.confidence * 100:5.1f}%",
    ]
    if sample.head_pose.valid:
        lines.append(f"Y/P/R {sample.head_pose.yaw:+.0f}/"
                     f"{sample.head_pose.pitch:+.0f}/{sample.head_pose.roll:+.0f}")
    if sample.eyes_closed:
        lines.append("EYES CLOSED")

    y = 20
    for line in lines:
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    _WHITE, 1, cv2.LINE_AA)
        y += 20


def bgr_to_qimage(frame_bgr: np.ndarray):
    """Convert a BGR array to a QImage that owns its own memory."""
    from PySide6.QtGui import QImage

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width, _ = rgb.shape
    image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888)
    return image.copy()
