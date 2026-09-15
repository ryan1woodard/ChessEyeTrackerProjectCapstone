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

``iris_*_ix`` / ``iris_*_iy`` (image-aligned)
    The displacement of the iris from the eyeball's centre of rotation,
    resolved on the camera's own horizontal and vertical axes. The camera does
    not tilt, so this is the direction the eye is pointing **in the world** --
    which is the thing a screen position is a function of. Hold a gaze on one
    point and tilt your head 20 degrees and it barely moves.

``iris_*_x`` / ``iris_*_y`` (eye-local)
    The same displacement resolved along the eye's own axis, outer corner to
    inner corner. That axis rotates with the head, so this is the rotation of
    the eye **within the head**. Under the same 20-degree tilt it swings by
    four times as much, because the world direction has to be re-expressed in a
    frame that has turned underneath it.

Roll invariance is the property that sounds desirable, and the eye-local pair
is where the eye itself is most naturally described, so for a long time this
module offered only that. It is the wrong choice. Screen position is a
world-frame quantity, and an eye-local offset has discarded the head
orientation that relates the two. Measured end to end, an eye-local model with
no tilt feature costs 111 px of error against 27 px for the image-aligned one,
and even given the tilt to correct with it only reaches 36 px.

Both are kept, because the eye-local pair still carries vergence and openness,
and ``eye_tilt`` -- the rotation between the two frames -- goes alongside them.

Where the offset is measured from
---------------------------------
Not from the eye corners. The reference is the **centre of rotation of the
eyeball**, which no landmark marks because it is inside the head, projected
through the rigid face frame (see :mod:`~src.tracking.face_frame`). Two things
follow. The measurement is the eye's rotation about its actual centre rather
than a displacement from a nearby point on the skin, so it is the gaze
direction rather than a proxy for it. And the reference is fitted from 18
landmarks instead of 2, which is what takes the frame-to-frame noise in it from
0.35 px down to 0.16 px -- the single largest accuracy gain available, because
the corner pair contributed 71% of the variance in the offset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import face_tracker as fl
from .face_frame import FaceFrame, fit_face_frame
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
    # The same direction as a tangent rather than a sine: linear in screen
    # position where the raw offset is not.
    "ray_l_x", "ray_l_y",
    "ray_r_x", "ray_r_y",
    "ray_mean_x", "ray_mean_y",
    "iris_vergence",
    "yaw", "pitch", "roll",
    "eye_tilt",
    "face_x", "face_y", "face_scale",
    "head_x", "head_y", "head_z",
    "ear_l", "ear_r",
)

_MIN_EYE_WIDTH_PX = 6.0

#: Radius of the eyeball as a fraction of the corner-to-corner eye width:
#: about 11 mm against 28.5 mm. It converts an iris displacement back into the
#: angle that produced it.
#:
#: A fixed constant is enough. Getting it wrong stretches the angle scale
#: smoothly and monotonically, which is exactly the kind of distortion the
#: calibration polynomial exists to absorb; what it cannot absorb is the
#: *shape* of the sine-versus-tangent mismatch, which this removes.
EYEBALL_RADIUS_RATIO = 11.0 / 28.5


def _gaze_ray(offset_x: float, offset_y: float) -> tuple[float, float]:
    """Turn an iris displacement into the tangent of the gaze angle.

    The iris rides on a sphere, so its visible displacement from the eyeball's
    centre goes as ``r * sin(theta)``. A screen is a plane, so the position
    looked at goes as ``distance * tan(theta)``. Feeding the raw displacement
    to the model therefore asks a polynomial to approximate ``tan(asin(x))``
    on top of everything else it is fitting -- a function that is nearly linear
    in the middle and turns sharply at the edges, which is precisely where the
    calibration targets are sparsest and the fit worst constrained.

    Inverting the geometry here costs two square roots a frame and hands the
    regression a quantity that is already proportional to screen displacement,
    leaving it only the head geometry to account for.
    """
    radius = EYEBALL_RADIUS_RATIO
    planar = offset_x * offset_x + offset_y * offset_y
    # Beyond the eyeball's radius the geometry has no solution -- the landmarks
    # are wrong rather than the eye being at 90 degrees -- so the angle is
    # held just short of the limit instead of producing an infinity.
    depth = math.sqrt(max(radius * radius - planar, (0.15 * radius) ** 2))
    return offset_x / depth, offset_y / depth

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
    "ray_l_x": 0.50, "ray_l_y": 0.40,
    "ray_r_x": 0.50, "ray_r_y": 0.40,
    "ray_mean_x": 0.50, "ray_mean_y": 0.40,
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
    ray_x: float
    ray_y: float
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


#: How much of the eye's reference frame comes from the rigid face fit rather
#: than from that eye's own two corner landmarks.
#:
#: The two make opposite mistakes. The corner pair is unbiased -- it is the real
#: eye, not a canonical one -- but noisy, contributing 71% of the variance in
#: the iris offset. The rigid fit averages 18 landmarks so it is far steadier,
#: but it fits an average head to a particular face, and that mismatch leaves a
#: small bias that shifts a little with pose.
#:
#: The bias turns out not to matter. It is very nearly rigid -- measured in the
#: eye's own axes it moves by under a pixel across tilts, turns and leans -- so
#: it behaves like a slightly different eyeball centre, which is precisely the
#: kind of per-user constant calibration exists to absorb. Swept end to end
#: over a range of face shapes and head sizes, error falls monotonically as
#: weight moves to the fit and is lowest when it takes all of it.
#:
#: Kept as a weight rather than hard-coded because the corner path remains the
#: fallback whenever the fit is unusable, and being able to dial the two
#: together is what made the sweep possible.
FRAME_BLEND = 1.0


def _eye_features(landmarks: FaceLandmarks, outer: int, inner: int,
                  lids: Sequence[tuple[int, int]], iris_ids: Sequence[int],
                  flip_axis: bool, side: str,
                  frame: Optional[FaceFrame] = None,
                  blend: float = FRAME_BLEND) -> Optional[EyeFeatures]:
    outer_px = landmarks.pixel(outer)
    inner_px = landmarks.pixel(inner)
    axis = (inner_px - outer_px) if not flip_axis else (outer_px - inner_px)
    width = float(np.linalg.norm(axis))
    if width < _MIN_EYE_WIDTH_PX:
        return None
    unit_along = axis / width
    center = 0.5 * (outer_px + inner_px)

    if frame is not None and frame.valid and blend > 0.0:
        # The eyeball's centre of rotation, which no landmark marks, projected
        # through the rigid frame. Measuring the iris displacement from there
        # rather than from the corner midpoint is both steadier and closer to
        # the quantity that actually matters.
        center = (1.0 - blend) * center + blend * frame.eye_centre(side)
        frame_axis = frame.eye_axis(side)
        if flip_axis:
            frame_axis = -frame_axis
        unit_along = frame_axis
        width = frame.eye_width_px

    unit_perp = np.array([-unit_along[1], unit_along[0]])  # +y is downwards

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

    offset_ix, offset_iy = float(offset[0] / width), float(offset[1] / width)
    ray_x, ray_y = _gaze_ray(offset_ix, offset_iy)

    return EyeFeatures(
        iris_x=float(np.dot(offset, unit_along) / width),
        iris_y=float(np.dot(offset, unit_perp) / width),
        iris_ix=offset_ix,
        iris_iy=offset_iy,
        ray_x=ray_x,
        ray_y=ray_y,
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

    def __init__(self, frame_blend: float = FRAME_BLEND) -> None:
        self.frame_blend = float(frame_blend)

    def extract(self, landmarks: Optional[FaceLandmarks],
                head_pose: Optional[HeadPose],
                frame: Optional[FaceFrame] = None) -> FeatureVector:
        if landmarks is None or not landmarks.has_iris:
            return FeatureVector.invalid()

        frame = frame if frame is not None else fit_face_frame(landmarks)
        left = _eye_features(
            landmarks, fl.EYE_LEFT_OUTER, fl.EYE_LEFT_INNER,
            [(fl.EYE_LEFT_TOP, fl.EYE_LEFT_BOTTOM), (fl.EYE_LEFT_TOP2, fl.EYE_LEFT_BOTTOM2)],
            fl.IRIS_LEFT, flip_axis=False, side="left",
            frame=frame, blend=self.frame_blend,
        )
        right = _eye_features(
            landmarks, fl.EYE_RIGHT_OUTER, fl.EYE_RIGHT_INNER,
            [(fl.EYE_RIGHT_TOP, fl.EYE_RIGHT_BOTTOM), (fl.EYE_RIGHT_TOP2, fl.EYE_RIGHT_BOTTOM2)],
            fl.IRIS_RIGHT, flip_axis=True, side="right",
            frame=frame, blend=self.frame_blend,
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
            "ray_l_x": left.ray_x,
            "ray_l_y": left.ray_y,
            "ray_r_x": right.ray_x,
            "ray_r_y": right.ray_y,
            "ray_mean_x": 0.5 * (left.ray_x + right.ray_x),
            "ray_mean_y": 0.5 * (left.ray_y + right.ray_y),
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
