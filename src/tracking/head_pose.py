"""Head orientation estimation.

Gaze direction is not determined by the eyes alone: when someone looks at the
edge of a large monitor they usually turn their head as well. The head pose
below is fed into the calibration model alongside the iris features.

Two backends can supply it -- ``solvePnP`` on six landmarks, or the facial
transformation matrix from MediaPipe Tasks -- and both are normalised to one
convention here, because a calibration profile is a set of numbers fitted to
whatever the angles meant on the day. If the sign of an angle depends on which
MediaPipe build is installed, a profile silently stops meaning what it did, and
the failure looks like "the tracker got worse" rather than like a bug.

Roll is special. It is read off the eye-corner landmarks rather than from
either pose solver: the eye corners are among the most stable points on the
face, the measurement is a single ``arctan2`` with nothing to diverge, and it
is the same rotation the image-aligned iris features are expressed in. Keeping
one source for it means the tilt the model compensates for and the tilt it is
told about can never disagree.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .face_tracker import (CHIN, EYE_LEFT_INNER, EYE_LEFT_OUTER,
                           EYE_RIGHT_INNER, EYE_RIGHT_OUTER, FaceLandmarks,
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

#: Below this eye width in pixels the corner landmarks are too close together
#: for their angle to mean anything.
_MIN_EYE_WIDTH_PX = 6.0


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

    def is_extreme(self, max_yaw: float = 40.0, max_pitch: float = 30.0,
                   max_roll: float = 35.0) -> bool:
        """Whether the head is turned too far for the estimate to be trusted.

        Roll counts too. A head tilted past the range calibration covered is
        just as far outside the fitted model as one turned away, and treating
        tilt as always acceptable is what let a badly tilted head produce a
        confident, wrong answer.
        """
        return (abs(self.yaw) > max_yaw or abs(self.pitch) > max_pitch
                or abs(self.roll) > max_roll)


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

    @staticmethod
    def roll_from_landmarks(landmarks: FaceLandmarks) -> Optional[float]:
        """Head roll in degrees, straight off the two eye-corner lines.

        Returns ``None`` if the eyes are too small in frame to measure.
        """
        angles = []
        for outer, inner in ((EYE_LEFT_OUTER, EYE_LEFT_INNER),
                             (EYE_RIGHT_INNER, EYE_RIGHT_OUTER)):
            axis = landmarks.pixel(inner) - landmarks.pixel(outer)
            if float(np.linalg.norm(axis)) < _MIN_EYE_WIDTH_PX:
                continue
            angles.append(math.atan2(float(axis[1]), float(axis[0])))
        if not angles:
            return None
        # Averaged as unit vectors so the mean survives the +/-180 wrap.
        mean_sin = sum(math.sin(a) for a in angles) / len(angles)
        mean_cos = sum(math.cos(a) for a in angles) / len(angles)
        return math.degrees(math.atan2(mean_sin, mean_cos))

    def estimate(self, landmarks: FaceLandmarks) -> HeadPose:
        pose = self._orientation(landmarks)
        roll = self.roll_from_landmarks(landmarks)
        if roll is None or not pose.valid:
            return pose
        return HeadPose(pitch=pose.pitch, yaw=pose.yaw, roll=roll, valid=True)

    def _orientation(self, landmarks: FaceLandmarks) -> HeadPose:
        if landmarks.transform_matrix is not None:
            # The Tasks matrix maps the canonical face into camera space using
            # a y-up, z-towards-the-viewer frame, which is the mirror of the
            # y-down image frame used here on two axes. Decomposing it yields
            # all three angles negated -- ALL three. Negating only pitch and
            # yaw, as this once did, left roll reading backwards on this
            # backend and correctly on the other, so head tilt was compensated
            # in the wrong direction depending on which MediaPipe was
            # installed.
            rotation = landmarks.transform_matrix[:3, :3]
            pitch, yaw, roll = _matrix_to_euler(rotation)
            return HeadPose(pitch=-pitch, yaw=yaw, roll=-roll, valid=True)

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

        rotation, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = _matrix_to_euler(rotation)

        # solvePnP returns pitch near +/-180 for an upright head; normalise.
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180

        # Only seed the next frame from a solution that is physically
        # plausible. Keeping a wild rvec as the extrinsic guess is how a
        # single bad frame turns into a run of bad ones: the solver starts
        # from the nonsense and converges straight back to it.
        if abs(yaw) > 89.0 or abs(pitch) > 89.0:
            self.reset()
            return HeadPose.invalid()
        self._last_rvec, self._last_tvec = rvec, tvec
        # Yaw is negated so the sign matches the documented convention:
        # solvePnP's own yaw is positive when the face turns towards the LEFT
        # of the image, which is the opposite of what HeadPose promises. The
        # mismatch never showed up in accuracy -- the model is fitted per user,
        # so a consistently flipped input is simply absorbed -- but it made the
        # angle wrong for everything that reads it as a number, the confidence
        # penalties and the debug overlay included.
        return HeadPose(pitch=-pitch, yaw=-yaw, roll=roll, valid=True)

    def reset(self) -> None:
        self._last_rvec = None
        self._last_tvec = None
