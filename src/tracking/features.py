"""Eye, iris and face feature extraction.

The features below are the *only* thing the calibration model ever sees, so
what is and is not encoded here sets a hard ceiling on accuracy.

Scale invariance
----------------
Every eye measurement is divided by that eye's own corner-to-corner width, so
moving 10 cm closer to the webcam does not shift the features. Vertical offsets
are normalised by eye *width*, not eye height: eye height collapses during a
blink and would make the vertical feature explode.

Two frames, not one
-------------------
The iris offset is reported **twice**, in two different reference frames, and
the difference between them is the whole story of head tilt.

``iris_*_x`` / ``iris_*_y`` (eye-local)
    Measured along the eye's own axis, outer corner to inner corner, and
    perpendicular to it. That axis rotates with the head, so this pair is
    *roll-invariant*: it is the rotation of the eye **within the head**.

``iris_*_ix`` / ``iris_*_iy`` (image-aligned)
    The same offset resolved on the camera's own horizontal and vertical axes.
    The camera does not tilt, so this pair is *roll-covariant*: it tracks where
    the eye points in the world.

Roll invariance sounds like the desirable property, and for a long time this
module offered only that. It is precisely the wrong one. Screen position is a
world-frame quantity, so the model needs the world-frame offset; an eye-local
offset deliberately discards the tilt that relates the two. A model fed only
eye-local features cannot tell a 20-degree head tilt from no tilt at all, and
measurably does not: in the simulator it goes from 36 px of error upright to
235 px tilted, while the image-aligned pair holds accuracy roughly flat.

Both are kept. The eye-local pair is the more stable description of the eye
itself and still carries the vergence and openness signals; the image-aligned
pair supplies the orientation the eye-local pair throws away. Together with
``eye_tilt`` -- the rotation between the two frames, read straight off the
eye-corner line -- the regression has everything it needs to resolve a tilted
head, and none of it depends on a pose solver getting Euler angles right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import face_tracker as fl
from .face_tracker import FaceLandmarks
from .head_pose import HeadPose

FEATURE_NAMES: tuple[str, ...] = (
    # Eye-local (roll-invariant): eye rotation within the head.
    "iris_l_x", "iris_l_y",
    "iris_r_x", "iris_r_y",
    "iris_mean_x", "iris_mean_y",
    # Image-aligned (roll-covariant): the same offset in the camera's frame.
    "iris_l_ix", "iris_l_iy",
    "iris_r_ix", "iris_r_iy",
    "iris_mean_ix", "iris_mean_iy",
    "iris_vergence",
    "yaw", "pitch", "roll",
    "eye_tilt",
    "face_x", "face_y", "face_scale",
    "head_x", "head_y", "head_z",
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
    "iris_l_ix": 0.18, "iris_l_iy": 0.14,
    "iris_r_ix": 0.18, "iris_r_iy": 0.14,
    "iris_mean_ix": 0.18, "iris_mean_iy": 0.14,
    "iris_vergence": 0.10,
    "yaw": 0.45, "pitch": 0.45, "roll": 0.45,
    "eye_tilt": 0.45,
    "face_x": 0.12, "face_y": 0.10, "face_scale": 0.04,
    "head_x": 0.8, "head_y": 0.6, "head_z": 2.0,
    "ear_l": 0.12, "ear_r": 0.12,
}


def nominal_scales(names) -> np.ndarray:
    """Fixed input scales for an ordered list of feature names."""
    return np.array([NOMINAL_SCALES.get(name, 1.0) for name in names], dtype=np.float64)


@dataclass
class EyeFeatures:
    """Per-eye measurements, scale-normalised by the eye's own width.

    ``iris_x``/``iris_y`` are eye-local (roll-invariant); ``iris_ix``/``iris_iy``
    are the same offset on the camera's axes (roll-covariant). ``tilt_deg`` is
    the angle of the eye axis in the image, which is head roll measured without
    going anywhere near a pose solver.
    """

    iris_x: float
    iris_y: float
    iris_ix: float
    iris_iy: float
    tilt_deg: float
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

    # ``flip_axis`` already orients both eyes the same way -- outer corner
    # towards inner corner for the image-left eye, inner towards outer for the
    # image-right -- so ``unit_along`` points to the image-right on both, and
    # its angle from horizontal is the head roll this eye sees. Image ``y``
    # grows downwards, so a positive angle is a clockwise tilt, matching
    # HeadPose.roll.
    tilt_deg = float(np.degrees(np.arctan2(unit_along[1], unit_along[0])))

    return EyeFeatures(
        iris_x=float(np.dot(offset, unit_along) / width),
        iris_y=float(np.dot(offset, unit_perp) / width),
        iris_ix=float(offset[0] / width),
        iris_iy=float(offset[1] / width),
        tilt_deg=tilt_deg,
        openness=openness,
        width_px=width,
        center_px=center,
        iris_center_px=iris_center,
    )


def _mean_angle_deg(*angles: float) -> float:
    """Circular mean of angles in degrees, safe across the +/-180 wrap."""
    radians = np.radians(angles)
    return float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())))


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
        # The apparent interocular distance is the only scale reference in the
        # image, and it appears in a denominator below, so it is floored at the
        # same width a single eye must reach to be usable at all.
        interocular_px = max(interocular, _MIN_EYE_WIDTH_PX)
        face_center = 0.5 * (left_eye_px + right_eye_px)

        # Head roll straight off the eye-corner lines. Unlike solvePnP this
        # cannot flip sign, drift, or depend on which MediaPipe backend is in
        # use -- it is the very rotation the image-aligned features carry.
        # The two eyes are combined as unit vectors rather than as angles, so
        # the average stays correct if one of them crosses +/-180 degrees.
        eye_tilt = _mean_angle_deg(left.tilt_deg, right.tilt_deg)

        values: Dict[str, float] = {
            "iris_l_x": left.iris_x,
            "iris_l_y": left.iris_y,
            "iris_r_x": right.iris_x,
            "iris_r_y": right.iris_y,
            "iris_mean_x": 0.5 * (left.iris_x + right.iris_x),
            "iris_mean_y": 0.5 * (left.iris_y + right.iris_y),
            "iris_l_ix": left.iris_ix,
            "iris_l_iy": left.iris_iy,
            "iris_r_ix": right.iris_ix,
            "iris_r_iy": right.iris_iy,
            "iris_mean_ix": 0.5 * (left.iris_ix + right.iris_ix),
            "iris_mean_iy": 0.5 * (left.iris_iy + right.iris_iy),
            # Vergence carries weak depth information and helps separate
            # "looking near the centre" from "looking past the screen".
            "iris_vergence": left.iris_x - right.iris_x,
            "yaw": pose.yaw / 45.0,
            "pitch": pose.pitch / 30.0,
            "roll": pose.roll / 30.0,
            "eye_tilt": eye_tilt / 30.0,
            "face_x": float(face_center[0] / max(landmarks.frame_width, 1)) - 0.5,
            "face_y": float(face_center[1] / max(landmarks.frame_height, 1)) - 0.5,
            "face_scale": interocular / max(landmarks.frame_width, 1),
            # Metric head position, in interocular-distance units. Dividing the
            # apparent offset by the apparent size cancels the perspective
            # division, so these are proportional to real millimetres of head
            # displacement rather than to pixels -- and ``head_z`` is
            # proportional to distance rather than to its reciprocal. A degree-2
            # polynomial models a straight line in these easily and a 1/z curve
            # in the raw ones badly, which is what a user leaning back exposes.
            "head_x": float(face_center[0] - 0.5 * landmarks.frame_width) / interocular_px,
            "head_y": float(face_center[1] - 0.5 * landmarks.frame_height) / interocular_px,
            "head_z": float(landmarks.frame_width) / interocular_px,
            "ear_l": left.openness,
            "ear_r": right.openness,
        }
        return FeatureVector(values=values, valid=True, left=left, right=right, head_pose=pose)
