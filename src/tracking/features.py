"""Eye, iris and face feature extraction.

The features below are the *only* thing the calibration model ever sees. Two
properties matter more than anything else here:

1. **Scale invariance.** Every eye measurement is divided by that eye's own
   corner-to-corner width, so moving 10 cm closer to the webcam does not shift
   the features.
2. **Roll invariance.** The iris offset is expressed in the eye's own local
   frame (along the eye axis and perpendicular to it) rather than in image
   axes, so tilting the head does not rotate the features.

Vertical offsets are normalised by eye *width*, not eye height: eye height
collapses during a blink and would make the vertical feature explode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import face_tracker as fl
from .face_tracker import FaceLandmarks
from .head_pose import HeadPose

FEATURE_NAMES: tuple[str, ...] = (
    "iris_l_x", "iris_l_y",
    "iris_r_x", "iris_r_y",
    "iris_mean_x", "iris_mean_y",
    "iris_vergence",
    "yaw", "pitch", "roll",
    "face_x", "face_y", "face_scale",
    "ear_l", "ear_r",
)

_MIN_EYE_WIDTH_PX = 6.0

#: Nominal half-range of each feature, used to scale the model's inputs.
#:
#: The obvious choice -- dividing by each feature's standard deviation across
#: the calibration samples -- is actively harmful here. A user who sits still
#: during calibration produces almost no variation in ``face_scale`` or
#: ``face_x``, so their standard deviation is tiny and dividing by it amplifies
#: those features by 50x or more. The model then keys off posture noise, and
#: the moment the user leans in slightly during play the prediction flies off
#: the screen.
#:
#: These features are already physically normalised (iris offsets are fractions
#: of eye width, angles are divided by a reference angle), so fixed nominal
#: scales are both meaningful and stable regardless of how still the user sat.
NOMINAL_SCALES: Dict[str, float] = {
    "iris_l_x": 0.18, "iris_l_y": 0.14,
    "iris_r_x": 0.18, "iris_r_y": 0.14,
    "iris_mean_x": 0.18, "iris_mean_y": 0.14,
    "iris_vergence": 0.10,
    "yaw": 0.45, "pitch": 0.45, "roll": 0.45,
    "face_x": 0.12, "face_y": 0.10, "face_scale": 0.04,
    "ear_l": 0.12, "ear_r": 0.12,
}


def nominal_scales(names) -> np.ndarray:
    """Fixed input scales for an ordered list of feature names."""
    return np.array([NOMINAL_SCALES.get(name, 1.0) for name in names], dtype=np.float64)


@dataclass
class EyeFeatures:
    """Per-eye measurements in the eye's local, scale-normalised frame."""

    iris_x: float
    iris_y: float
    openness: float
    width_px: float
    center_px: np.ndarray
    iris_center_px: np.ndarray
    valid: bool = True


@dataclass
class FeatureVector:
    """A single frame's normalised features, addressable by name."""

    values: Dict[str, float] = field(default_factory=dict)
    valid: bool = True
    left: Optional[EyeFeatures] = None
    right: Optional[EyeFeatures] = None
    head_pose: Optional[HeadPose] = None

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    def get(self, name: str, default: float = 0.0) -> float:
        return self.values.get(name, default)

    def to_array(self, names: Sequence[str]) -> np.ndarray:
        """Project onto an ordered subset of feature names."""
        return np.array([self.values.get(n, 0.0) for n in names], dtype=np.float64)

    @property
    def eye_openness(self) -> float:
        return 0.5 * (self.get("ear_l") + self.get("ear_r"))

    @classmethod
    def invalid(cls) -> "FeatureVector":
        return cls(values={name: 0.0 for name in FEATURE_NAMES}, valid=False)


def _eye_features(landmarks: FaceLandmarks, outer: int, inner: int,
                  lids: Sequence[tuple[int, int]], iris_ids: Sequence[int],
                  flip_axis: bool) -> Optional[EyeFeatures]:
    outer_px = landmarks.pixel(outer)
    inner_px = landmarks.pixel(inner)
    axis = (inner_px - outer_px) if not flip_axis else (outer_px - inner_px)
    width = float(np.linalg.norm(axis))
    if width < _MIN_EYE_WIDTH_PX:
        return None

    unit_along = axis / width
    unit_perp = np.array([-unit_along[1], unit_along[0]])  # +y is downwards

    center = 0.5 * (outer_px + inner_px)
    iris_center = landmarks.pixels(iris_ids).mean(axis=0)
    offset = iris_center - center

    lid_gaps: List[float] = []
    for top, bottom in lids:
        lid_gaps.append(float(np.linalg.norm(landmarks.pixel(top) - landmarks.pixel(bottom))))
    openness = float(np.mean(lid_gaps) / width) if lid_gaps else 0.0

    return EyeFeatures(
        iris_x=float(np.dot(offset, unit_along) / width),
        iris_y=float(np.dot(offset, unit_perp) / width),
        openness=openness,
        width_px=width,
        center_px=center,
        iris_center_px=iris_center,
    )


class FeatureExtractor:
    """Turns raw landmarks plus head pose into a :class:`FeatureVector`."""

    def extract(self, landmarks: Optional[FaceLandmarks],
                head_pose: Optional[HeadPose]) -> FeatureVector:
        if landmarks is None or not landmarks.has_iris:
            return FeatureVector.invalid()

        left = _eye_features(
            landmarks, fl.EYE_LEFT_OUTER, fl.EYE_LEFT_INNER,
            [(fl.EYE_LEFT_TOP, fl.EYE_LEFT_BOTTOM), (fl.EYE_LEFT_TOP2, fl.EYE_LEFT_BOTTOM2)],
            fl.IRIS_LEFT, flip_axis=False,
        )
        right = _eye_features(
            landmarks, fl.EYE_RIGHT_OUTER, fl.EYE_RIGHT_INNER,
            [(fl.EYE_RIGHT_TOP, fl.EYE_RIGHT_BOTTOM), (fl.EYE_RIGHT_TOP2, fl.EYE_RIGHT_BOTTOM2)],
            fl.IRIS_RIGHT, flip_axis=True,
        )
        if left is None or right is None:
            return FeatureVector.invalid()

        pose = head_pose if head_pose is not None else HeadPose.invalid()

        left_eye_px = left.center_px
        right_eye_px = right.center_px
        interocular = float(np.linalg.norm(right_eye_px - left_eye_px))
        face_center = 0.5 * (left_eye_px + right_eye_px)

        values: Dict[str, float] = {
            "iris_l_x": left.iris_x,
            "iris_l_y": left.iris_y,
            "iris_r_x": right.iris_x,
            "iris_r_y": right.iris_y,
            "iris_mean_x": 0.5 * (left.iris_x + right.iris_x),
            "iris_mean_y": 0.5 * (left.iris_y + right.iris_y),
            # Vergence carries weak depth information and helps separate
            # "looking near the centre" from "looking past the screen".
            "iris_vergence": left.iris_x - right.iris_x,
            "yaw": pose.yaw / 45.0,
            "pitch": pose.pitch / 30.0,
            "roll": pose.roll / 30.0,
            "face_x": float(face_center[0] / max(landmarks.frame_width, 1)) - 0.5,
            "face_y": float(face_center[1] / max(landmarks.frame_height, 1)) - 0.5,
            "face_scale": interocular / max(landmarks.frame_width, 1),
            "ear_l": left.openness,
            "ear_r": right.openness,
        }
        return FeatureVector(values=values, valid=True, left=left, right=right, head_pose=pose)
