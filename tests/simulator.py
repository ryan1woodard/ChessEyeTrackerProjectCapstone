"""A geometric eye-and-head simulator for testing the gaze pipeline.

The earlier synthetic features mapped screen position straight to iris offsets,
which made head movement irrelevant by construction -- so they could not
reproduce the failure where accuracy decays as the user shifts in their chair.

This simulator models the actual geometry instead:

* the screen is a plane in millimetres, with the webcam above its centre;
* the eye sits at a 3-D position in front of it;
* looking at a screen point requires a specific gaze *direction*;
* the iris offset the camera sees is that direction expressed in **head**
  coordinates, so head rotation and eye rotation trade off against each other;
* moving the head changes which direction points at a given screen position.

That last property is the important one. It means the same iris offset maps to
a different screen point once the head moves, which is exactly why a model
calibrated at one head position degrades as the user shifts -- and it lets that
degradation be measured rather than guessed at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from src.tracking.features import FeatureVector
from src.tracking.head_pose import HeadPose

#: Millimetres per screen pixel for a typical 24" 1920x1080 monitor.
MM_PER_PX = 0.277

#: Iris offset (in eye-width units) produced by one radian of eye rotation.
#: Chosen so a comfortable +/-25 degree gaze spans roughly +/-0.13, matching
#: the magnitudes the real feature extractor produces.
IRIS_GAIN = 0.30

#: Interocular distance in millimetres, used for the apparent-size feature.
INTEROCULAR_MM = 63.0


@dataclass
class HeadState:
    """Where the eye is and which way the head points.

    Position is in millimetres relative to the centre of the screen: ``x`` to
    the right, ``y`` up, ``z`` *towards the viewer* (so a seated user has a
    positive ``z`` of roughly 600 mm).
    """

    x: float = 0.0
    y: float = -60.0
    z: float = 600.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    def shifted(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                dyaw: float = 0.0, dpitch: float = 0.0,
                droll: float = 0.0) -> "HeadState":
        return HeadState(self.x + dx, self.y + dy, self.z + dz,
                         self.yaw_deg + dyaw, self.pitch_deg + dpitch,
                         self.roll_deg + droll)


@dataclass
class EyeSimulator:
    """Turns (screen target, head state) into a plausible feature vector."""

    screen_width: int = 1920
    screen_height: int = 1080
    mm_per_px: float = MM_PER_PX
    noise: float = 0.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    # ------------------------------------------------------------- geometry
    def screen_to_mm(self, x: float, y: float) -> Tuple[float, float]:
        """Screen pixels to millimetres, origin at the centre of the display."""
        return ((x - self.screen_width / 2) * self.mm_per_px,
                -(y - self.screen_height / 2) * self.mm_per_px)

    def gaze_angles(self, target_px: Tuple[float, float],
                    head: HeadState) -> Tuple[float, float]:
        """Yaw and pitch (radians) the eye must adopt to hit a screen point."""
        tx, ty = self.screen_to_mm(*target_px)
        dx, dy, dz = tx - head.x, ty - head.y, -head.z   # towards the screen
        horizontal = math.atan2(dx, -dz)
        vertical = math.atan2(dy, math.hypot(dx, dz))
        return horizontal, vertical

    # ------------------------------------------------------------- features
    def features(self, target_px: Tuple[float, float],
                 head: Optional[HeadState] = None) -> FeatureVector:
        head = head or HeadState()
        gaze_yaw, gaze_pitch = self.gaze_angles(target_px, head)

        # The camera sees eye rotation relative to the HEAD, so turning the
        # head to face the target reduces the iris offset needed.
        eye_yaw = gaze_yaw - math.radians(head.yaw_deg)
        eye_pitch = gaze_pitch - math.radians(head.pitch_deg)

        iris_x = IRIS_GAIN * math.sin(eye_yaw)
        iris_y = -IRIS_GAIN * math.sin(eye_pitch)   # image y grows downwards

        def jitter(scale: float = 1.0) -> float:
            return self.rng.normal(0.0, self.noise * scale) if self.noise else 0.0

        # Apparent position and size of the face in the camera image.
        face_x = head.x / max(head.z, 1.0) * 0.9
        face_y = -(head.y + 60.0) / max(head.z, 1.0) * 0.9
        face_scale = INTEROCULAR_MM / max(head.z, 1.0) * 1.8

        left_x = iris_x + jitter()
        right_x = iris_x + jitter()
        left_y = iris_y + jitter()
        right_y = iris_y + jitter()

        values = {
            "iris_l_x": left_x, "iris_l_y": left_y,
            "iris_r_x": right_x, "iris_r_y": right_y,
            "iris_mean_x": 0.5 * (left_x + right_x),
            "iris_mean_y": 0.5 * (left_y + right_y),
            "iris_vergence": left_x - right_x,
            "yaw": head.yaw_deg / 45.0 + jitter(0.4),
            "pitch": head.pitch_deg / 30.0 + jitter(0.4),
            "roll": head.roll_deg / 30.0 + jitter(0.2),
            "face_x": face_x + jitter(0.2),
            "face_y": face_y + jitter(0.2),
            "face_scale": face_scale + jitter(0.05),
            "ear_l": 0.30, "ear_r": 0.30,
        }
        return FeatureVector(
            values=values, valid=True,
            head_pose=HeadPose(pitch=head.pitch_deg, yaw=head.yaw_deg,
                               roll=head.roll_deg),
        )

    def blink_features(self, target_px: Tuple[float, float],
                       head: Optional[HeadState] = None,
                       severity: float = 1.0) -> FeatureVector:
        """Features during an eye closure.

        Iris landmarks become meaningless when the lid covers the pupil, and
        MediaPipe reports them somewhere near the lid rather than admitting
        failure -- so a blink injects a large, wrong reading rather than a
        missing one.
        """
        features = self.features(target_px, head)
        features.values["ear_l"] = 0.05
        features.values["ear_r"] = 0.05
        features.values["iris_l_y"] += 0.10 * severity
        features.values["iris_r_y"] += 0.10 * severity
        features.values["iris_l_x"] += self.rng.normal(0, 0.05) * severity
        features.values["iris_r_x"] += self.rng.normal(0, 0.05) * severity
        return features
