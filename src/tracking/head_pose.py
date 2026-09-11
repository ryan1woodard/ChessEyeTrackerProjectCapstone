"""Head orientation estimation.

Gaze direction is not determined by the eyes alone: when someone looks at the
edge of a large monitor they usually turn their head as well. The head pose
below is fed into the calibration model alongside the iris features.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .face_tracker import (CHIN, EYE_LEFT_OUTER, EYE_RIGHT_OUTER, FaceLandmarks,
                           MOUTH_LEFT, MOUTH_RIGHT, NOSE_TIP)

logger = logging.getLogger(__name__)

# A coarse canonical face in millimetres, origin at the nose tip.
# Absolute scale is irrelevant here -- only the resulting angles are used.
_MODEL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],        # nose tip
        [0.0, -63.6, -12.5],    # chin
        [-43.3, 32.7, -26.0],   # image-left eye outer corner
        [43.3, 32.7, -26.0],    # image-right eye outer corner
        [-28.9, -28.9, -24.1],  # mouth left
        [28.9, -28.9, -24.1],   # mouth right
    ],
    dtype=np.float64,
)
_LANDMARK_IDS = (NOSE_TIP, CHIN, EYE_LEFT_OUTER, EYE_RIGHT_OUTER, MOUTH_LEFT, MOUTH_RIGHT)


@dataclass(frozen=True)
class HeadPose:
    """Head orientation in degrees.

    ``yaw`` is positive when the head turns towards the right of the image,
    ``pitch`` is positive when the chin lifts, ``roll`` is positive for a
    clockwise head tilt in the image.
    """

    pitch: float
    yaw: float
    roll: float
    valid: bool = True

    @classmethod
    def invalid(cls) -> "HeadPose":
        return cls(0.0, 0.0, 0.0, valid=False)

    def is_extreme(self, max_yaw: float = 40.0, max_pitch: float = 30.0) -> bool:
        return abs(self.yaw) > max_yaw or abs(self.pitch) > max_pitch


def _matrix_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    """Decompose a rotation matrix into (pitch, yaw, roll) degrees."""
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    if sy < 1e-6:  # gimbal lock
        pitch = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = math.atan2(-rotation[2, 0], sy)
        roll = 0.0
    else:
        pitch = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(-rotation[2, 0], sy)
        roll = math.atan2(rotation[1, 0], rotation[0, 0])
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


class HeadPoseEstimator:
    """Estimates head orientation from six stable facial landmarks."""

    def __init__(self) -> None:
        self._last_rvec: Optional[np.ndarray] = None
        self._last_tvec: Optional[np.ndarray] = None

    @staticmethod
    def camera_matrix(width: int, height: int) -> np.ndarray:
        """A generic pinhole intrinsic matrix; focal length approximated by width."""
        focal = float(width)
        return np.array(
            [[focal, 0.0, width / 2.0],
             [0.0, focal, height / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def estimate(self, landmarks: FaceLandmarks) -> HeadPose:
        if landmarks.transform_matrix is not None:
            rotation = landmarks.transform_matrix[:3, :3]
            pitch, yaw, roll = _matrix_to_euler(rotation)
            # The Tasks matrix is camera-to-face; flip signs for our convention.
            return HeadPose(pitch=-pitch, yaw=-yaw, roll=roll, valid=True)

        image_points = landmarks.pixels(_LANDMARK_IDS)
        camera = self.camera_matrix(landmarks.frame_width, landmarks.frame_height)
        distortion = np.zeros((4, 1), dtype=np.float64)

        use_guess = self._last_rvec is not None
        try:
            ok, rvec, tvec = cv2.solvePnP(
                _MODEL_POINTS,
                image_points,
                camera,
                distortion,
                rvec=self._last_rvec.copy() if use_guess else None,
                tvec=self._last_tvec.copy() if use_guess else None,
                useExtrinsicGuess=use_guess,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error as exc:  # pragma: no cover - degenerate geometry
            logger.debug("solvePnP failed: %s", exc)
            return HeadPose.invalid()
        if not ok:
            return HeadPose.invalid()

        self._last_rvec, self._last_tvec = rvec, tvec
        rotation, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = _matrix_to_euler(rotation)

        # solvePnP returns pitch near +/-180 for an upright head; normalise.
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
        return HeadPose(pitch=-pitch, yaw=yaw, roll=roll, valid=True)

    def reset(self) -> None:
        self._last_rvec = None
        self._last_tvec = None
