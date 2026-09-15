"""A 3-D geometric eye-and-head simulator for testing the gaze pipeline.

Earlier revisions of this file synthesised *feature values* directly. That made
whole classes of failure impossible to reproduce, because the feature
extractor itself was never exercised: head roll, for instance, only ever
appeared as a ``roll`` number, so a mapping that ignored tilt looked perfect.

This simulator instead renders a face. It carries:

* a canonical head in millimetres (eye corners, lids, eyeballs, nose, chin,
  mouth), rigid and rotated about a **neck pivot** rather than about the eyes,
  so tilting the head also translates the eyes the way a real neck does;
* a pinhole camera above the screen, matching the intrinsics
  :class:`~src.tracking.head_pose.HeadPoseEstimator` assumes;
* eyeballs that physically aim at the screen target, so vergence, foreshortening
  and the eye/head rotation trade-off all emerge from the geometry.

It then emits :class:`~src.tracking.face_tracker.FaceLandmarks`, which the real
:class:`~src.tracking.features.FeatureExtractor` and head-pose estimator consume
unchanged. Every number the model sees therefore travels the same path it does
with a live webcam.

Conventions
-----------
World and camera frame: origin at the camera, ``x`` right, ``y`` **down**,
``z`` towards the user. The screen is the plane ``z = 0``, centred below the
camera. Head-local axes coincide with world axes when the user faces the
camera squarely, so ``roll`` is positive for a clockwise tilt in the image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from src.tracking import face_tracker as fl
from src.tracking.face_tracker import FaceLandmarks
from src.tracking.face_frame import fit_face_frame
from src.tracking.features import FeatureExtractor, FeatureVector
from src.tracking.head_pose import HeadPoseEstimator

#: Millimetres per screen pixel for a typical 24" 1920x1080 monitor.
MM_PER_PX = 0.277

#: How far the webcam sits above the centre of the screen, in millimetres.
CAMERA_ABOVE_SCREEN_CENTRE_MM = 165.0

#: Radius of the eyeball in millimetres. The iris centre rides on this sphere,
#: so this is what converts an eye rotation into a visible iris displacement.
EYEBALL_RADIUS_MM = 11.0

#: Distance from the eye-corner plane back to the centre of rotation.
EYEBALL_DEPTH_MM = 11.5

#: Neck pivot in head-local millimetres, measured from the midpoint of the eyes.
#: Head rotation happens about this point, which is why a tilt shifts the eyes
#: sideways instead of spinning them in place.
NECK_PIVOT_MM = np.array([0.0, 115.0, 45.0], dtype=np.float64)

#: Canonical head geometry, head-local millimetres from the eye midpoint.
#:
#: Anthropometrically plausible rather than exact -- what the experiments here
#: need is landmarks in roughly the right places, spread over the face, so that
#: a fit which averages many of them behaves the way it will on a real face.
_FACE_MM: Dict[int, Tuple[float, float, float]] = {
    fl.NOSE_TIP: (0.0, 32.7, -26.0),
    fl.CHIN: (0.0, 96.3, -13.5),
    fl.MOUTH_LEFT: (-28.9, 61.6, -1.9),
    fl.MOUTH_RIGHT: (28.9, 61.6, -1.9),
    fl.FOREHEAD: (0.0, -55.0, 6.0),
    # nose bridge, between the eyes down to the tip
    168: (0.0, -2.0, -6.0),
    6: (0.0, 8.0, -14.0),
    197: (0.0, 2.0, -10.0),
    195: (0.0, 16.0, -20.0),
    4: (0.0, 28.0, -25.0),
    # sides of the nose
    98: (-17.0, 38.0, -12.0),
    327: (17.0, 38.0, -12.0),
    # temples and the upper face oval
    127: (-72.0, -8.0, 38.0),
    356: (72.0, -8.0, 38.0),
    234: (-73.0, 18.0, 34.0),
    454: (73.0, 18.0, 34.0),
    93: (-70.0, 38.0, 30.0),
    323: (70.0, 38.0, 30.0),
    # mid forehead
    151: (0.0, -40.0, -2.0),
}

#: Eye corners in head-local millimetres. The outer corner wraps further round
#: the side of the head, so it sits further back than the inner one.
_EYE_CORNERS_MM = {
    "left": {"outer": np.array([-45.0, 0.0, 8.0]),
             "inner": np.array([-15.0, 0.0, -4.0])},
    "right": {"outer": np.array([45.0, 0.0, 8.0]),
              "inner": np.array([15.0, 0.0, -4.0])},
}

#: Eyeball centres, derived from the corner midpoints.
_EYE_CENTRE_MM = {
    "left": np.array([-30.0, 0.0, EYEBALL_DEPTH_MM]),
    "right": np.array([30.0, 0.0, EYEBALL_DEPTH_MM]),
}

#: Where the lid margins of a relaxed open eye sit, in millimetres from the
#: eye axis. Their sum is the palpebral aperture, about 10 mm on an adult,
#: against a 30 mm eye width -- which is what makes a fully open eye read as an
#: openness of roughly 0.33.
UPPER_APERTURE_MM = 6.4
LOWER_APERTURE_MM = 3.6

#: Where the lids meet when the eye closes: a little below the eye axis, not on
#: it. Both margins travel to this line, so a closure really closes, and the
#: upper lid covers about four fifths of the distance because that is what it
#: does on a real eye.
CLOSURE_OFFSET_MM = 1.1


def _eye_ring_offsets(ring: Sequence[int]) -> Dict[int, Tuple[float, float]]:
    """Map each eyelid landmark to ``(t, lid)`` along the eye.

    ``t`` runs 0 at the outer corner to 1 at the inner corner; ``lid`` is +1 on
    the lower lid and -1 on the upper. MediaPipe's ring is ordered outer corner,
    round the lower lid to the inner corner, then back along the upper lid, so
    the two halves are walked separately. Deriving the lid landmarks from the
    ring rather than listing them keeps the vertically-paired points -- the ones
    the openness measure subtracts -- genuinely aligned.
    """
    lower, upper = ring[:9], ring[8:] + (ring[0],)
    offsets: Dict[int, Tuple[float, float]] = {}
    for index, landmark in enumerate(lower):
        offsets[landmark] = (index / (len(lower) - 1), 1.0)
    for index, landmark in enumerate(upper):
        t = 1.0 - index / (len(upper) - 1)
        if landmark not in offsets:          # corners belong to both halves
            offsets[landmark] = (t, -1.0)
    return offsets


_EYE_RINGS = {"left": fl.EYE_LEFT_RING, "right": fl.EYE_RIGHT_RING}
_RING_OFFSETS = {side: _eye_ring_offsets(ring) for side, ring in _EYE_RINGS.items()}

_IRIS_IDS = {"left": fl.IRIS_LEFT, "right": fl.IRIS_RIGHT}

#: Radius of the visible iris in millimetres, used to place the four rim
#: landmarks MediaPipe reports around the pupil.
IRIS_RADIUS_MM = 5.8


def _lid_profile(t: float) -> float:
    """Fraction of the full aperture reached at position ``t`` along the eye.

    Zero at both corners, where the lids meet, and widest just outside the
    middle. The exponent flattens the curve so the lid is close to fully open
    across the centre of the eye rather than peaking at a single point.
    """
    return float(math.sin(math.pi * min(max(t, 0.0), 1.0)) ** 0.62)


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Head-local to world rotation for ``R_z(roll) @ R_y(yaw) @ R_x(pitch)``.

    Signs follow :class:`~src.tracking.head_pose.HeadPose` so that the angles
    fed in here are the angles a perfect estimator would read back out: a
    positive ``roll`` tilts the head clockwise in the image, a positive ``yaw``
    turns the face towards the right of the image, and a positive ``pitch``
    lifts the chin.
    """
    pitch, yaw, roll = map(math.radians, (-pitch_deg, -yaw_deg, roll_deg))
    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cz, sz = math.cos(roll), math.sin(roll)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


@dataclass
class HeadState:
    """Where the head is and which way it points.

    ``x``/``y``/``z`` place the midpoint of the eyes in millimetres relative to
    the camera: ``x`` to the right, ``y`` **down**, ``z`` away from the camera
    towards the user, so a seated user sits at roughly ``z = 600``.

    ``y`` defaults to just below the camera, which is where a user's eyes
    actually sit when the webcam is perched on top of the monitor.
    """

    x: float = 0.0
    y: float = 60.0
    z: float = 600.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    #: Fraction of the normal lid aperture; 1.0 is a relaxed open eye.
    openness: float = 1.0

    def shifted(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                dyaw: float = 0.0, dpitch: float = 0.0, droll: float = 0.0,
                openness: Optional[float] = None) -> "HeadState":
        return HeadState(
            self.x + dx, self.y + dy, self.z + dz,
            self.yaw_deg + dyaw, self.pitch_deg + dpitch, self.roll_deg + droll,
            self.openness if openness is None else openness,
        )

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @property
    def rotation(self) -> np.ndarray:
        return rotation_matrix(self.yaw_deg, self.pitch_deg, self.roll_deg)


@dataclass
class EyeSimulator:
    """Renders synthetic landmarks for (screen target, head state) pairs."""

    screen_width: int = 1920
    screen_height: int = 1080
    frame_width: int = 1280
    frame_height: int = 720
    mm_per_px: float = MM_PER_PX
    camera_offset_mm: float = CAMERA_ABOVE_SCREEN_CENTRE_MM
    #: Standard deviation of the per-landmark detection noise, in pixels. Real
    #: MediaPipe iris landmarks wobble by a few tenths of a pixel per frame.
    noise_px: float = 0.0
    #: Per-subject deviation from the canonical head, in millimetres. Nobody's
    #: face matches the average one, and anything that fits a canonical model
    #: to landmarks has to cope with that; without this the rigid frame would
    #: be scored against a head it already knows exactly, which flatters it.
    shape_mm: float = 3.0
    #: Overall size of the head relative to the canonical one.
    head_scale: float = 1.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    extractor: FeatureExtractor = field(default_factory=FeatureExtractor)
    pose_estimator: HeadPoseEstimator = field(default_factory=HeadPoseEstimator)

    def __post_init__(self) -> None:
        # One draw per simulator, not per frame: a face is a fixed shape, and
        # a shape that jittered frame to frame would be noise rather than the
        # systematic model mismatch this is meant to represent.
        shape_rng = np.random.default_rng(self.rng.integers(1 << 32))

        def vary(point) -> np.ndarray:
            offset = shape_rng.normal(0.0, self.shape_mm, 3) if self.shape_mm else 0.0
            return np.asarray(point, dtype=np.float64) * self.head_scale + offset

        self.face_mm = {index: vary(point) for index, point in _FACE_MM.items()}
        self.eye_corners_mm = {
            side: {name: vary(point) for name, point in corners.items()}
            for side, corners in _EYE_CORNERS_MM.items()
        }
        # The eyeball centre follows the corners rather than being drawn
        # independently, because it is anatomically tied to them.
        self.eye_centre_mm = {
            side: 0.5 * (self.eye_corners_mm[side]["outer"]
                         + self.eye_corners_mm[side]["inner"])
            + np.array([0.0, 0.0, EYEBALL_DEPTH_MM])
            for side in ("left", "right")
        }

    # ------------------------------------------------------------- geometry
    @property
    def focal_px(self) -> float:
        """Matches :meth:`HeadPoseEstimator.camera_matrix`, which assumes f = width."""
        return float(self.frame_width)

    def screen_to_world(self, x: float, y: float) -> np.ndarray:
        """A screen pixel as a 3-D point in camera millimetres."""
        return np.array([
            (x - self.screen_width / 2.0) * self.mm_per_px,
            (y - self.screen_height / 2.0) * self.mm_per_px + self.camera_offset_mm,
            0.0,
        ], dtype=np.float64)

    def project(self, point_mm: np.ndarray) -> np.ndarray:
        """Pinhole projection of a camera-frame point to pixel coordinates."""
        depth = max(float(point_mm[2]), 1.0)
        return np.array([
            self.focal_px * point_mm[0] / depth + self.frame_width / 2.0,
            self.focal_px * point_mm[1] / depth + self.frame_height / 2.0,
        ], dtype=np.float64)

    def _to_world(self, head: HeadState, local_mm: np.ndarray) -> np.ndarray:
        """Rigid head transform, rotating about the neck rather than the eyes."""
        return head.position + NECK_PIVOT_MM + head.rotation @ (local_mm - NECK_PIVOT_MM)

    def eye_centre_world(self, head: HeadState, side: str) -> np.ndarray:
        return self._to_world(head, self.eye_centre_mm[side])

    def gaze_direction(self, target_px: Tuple[float, float], head: HeadState,
                       side: str) -> np.ndarray:
        """Unit vector from one eyeball centre to the screen target."""
        delta = self.screen_to_world(*target_px) - self.eye_centre_world(head, side)
        norm = float(np.linalg.norm(delta))
        return delta / norm if norm > 1e-9 else np.array([0.0, 0.0, -1.0])

    # ------------------------------------------------------------ rendering
    def landmarks(self, target_px: Tuple[float, float],
                  head: Optional[HeadState] = None,
                  blink_bias_mm: float = 0.0) -> FaceLandmarks:
        """Render the face as MediaPipe-style landmarks.

        ``blink_bias_mm`` drags the iris landmarks towards the upper lid, which
        is what a real detector does while the lid covers the pupil: it reports
        confident, wrong iris points rather than admitting failure.
        """
        head = head or HeadState()
        rotation = head.rotation
        points = np.zeros((fl.REQUIRED_LANDMARKS, 3), dtype=np.float64)

        def place(index: int, world_mm: np.ndarray) -> None:
            pixel = self.project(world_mm)
            if self.noise_px:
                pixel = pixel + self.rng.normal(0.0, self.noise_px, 2)
            points[index, 0] = pixel[0] / self.frame_width
            points[index, 1] = pixel[1] / self.frame_height
            points[index, 2] = world_mm[2] / 1000.0

        for index, local in self.face_mm.items():
            place(index, self._to_world(head, local))

        openness = max(head.openness, 0.0)
        for side in ("left", "right"):
            corners = self.eye_corners_mm[side]
            outer, inner = corners["outer"], corners["inner"]
            # Lid margins for this openness. Both travel towards the closure
            # line, so the upper lid moves about four times as far as the lower
            # one and a full closure leaves no gap at all.
            upper = CLOSURE_OFFSET_MM + openness * (-UPPER_APERTURE_MM - CLOSURE_OFFSET_MM)
            lower = CLOSURE_OFFSET_MM + openness * (LOWER_APERTURE_MM - CLOSURE_OFFSET_MM)
            for landmark, (along, lid) in _RING_OFFSETS[side].items():
                # Slide between the corners, then lift off the eye axis to the
                # lid margin, tapering to nothing at both corners.
                base = outer + (inner - outer) * along
                margin = lower if lid > 0 else upper
                offset = np.array([0.0, margin * _lid_profile(along), 0.0])
                # The lids wrap the eyeball, so the middle of the ring stands
                # proud of the line between the corners.
                bulge = np.array([0.0, 0.0, -3.0 * _lid_profile(along)])
                place(landmark, self._to_world(head, base + offset + bulge))

            # The iris rides on the eyeball surface, aimed at the target.
            direction = self.gaze_direction(target_px, head, side)
            centre_world = self.eye_centre_world(head, side)
            iris_world = centre_world + EYEBALL_RADIUS_MM * direction
            if blink_bias_mm:
                iris_world = iris_world + rotation @ np.array([0.0, -blink_bias_mm, 0.0])

            # A frame perpendicular to the gaze, for the four rim landmarks.
            up = rotation @ np.array([0.0, -1.0, 0.0])
            right = np.cross(up, direction)
            right /= max(float(np.linalg.norm(right)), 1e-9)
            up = np.cross(direction, right)

            ids = _IRIS_IDS[side]
            place(ids[0], iris_world)
            for slot, offset_vector in enumerate((right, up, -right, -up), start=1):
                place(ids[slot], iris_world + IRIS_RADIUS_MM * offset_vector)

        return FaceLandmarks(points, self.frame_width, self.frame_height, has_iris=True)

    # ------------------------------------------------------------- features
    def spoiled_features(self, target_px: Tuple[float, float],
                         head: Optional[HeadState] = None,
                         spike_px: float = 6.0) -> FeatureVector:
        """One frame in which the iris landmarks are badly misplaced.

        Detectors do this: an eyelash, a reflection off glasses or a frame of
        motion blur, and the iris is reported several pixels from where it is
        for exactly one frame. It is not a blink -- the eye is open and every
        other landmark is fine -- so nothing about the frame looks wrong except
        the answer it produces.
        """
        marks = self.landmarks(target_px, head)
        direction = self.rng.normal(0.0, 1.0, 2)
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        shift = direction * spike_px
        points = marks.points.copy()
        for index in fl.IRIS_LEFT + fl.IRIS_RIGHT:
            points[index, 0] += shift[0] / self.frame_width
            points[index, 1] += shift[1] / self.frame_height
        spoiled = FaceLandmarks(points, marks.frame_width, marks.frame_height,
                                has_iris=True)
        frame = fit_face_frame(spoiled)
        return self.extractor.extract(spoiled, self.pose_estimator.estimate(spoiled, frame),
                                      frame)

    def features(self, target_px: Tuple[float, float],
                 head: Optional[HeadState] = None,
                 blink_bias_mm: float = 0.0) -> FeatureVector:
        """Render a frame and run it through the real extraction chain."""
        marks = self.landmarks(target_px, head, blink_bias_mm)
        frame = fit_face_frame(marks)
        pose = self.pose_estimator.estimate(marks, frame)
        return self.extractor.extract(marks, pose, frame)

    def blink_features(self, target_px: Tuple[float, float],
                       head: Optional[HeadState] = None,
                       severity: float = 1.0) -> FeatureVector:
        """Features during an eye closure.

        The lids close and the iris landmarks are dragged up under the lid, so a
        blink injects a large, confident, *wrong* reading rather than a missing
        one. That is the behaviour the pipeline's blink hold exists to survive.
        """
        head = (head or HeadState())
        closed = head.shifted(openness=max(0.0, 1.0 - 0.93 * severity))
        return self.features(target_px, closed, blink_bias_mm=4.0 * severity)

    def true_pose(self, head: Optional[HeadState] = None) -> Tuple[float, float, float]:
        """The ground-truth ``(pitch, yaw, roll)`` a perfect estimator would report."""
        head = head or HeadState()
        return head.pitch_deg, head.yaw_deg, head.roll_deg


# ------------------------------------------------------------------ sequences
def head_sequence(rng: np.random.Generator, scale: float = 1.0,
                  tilt: float = 1.0, base: Optional[HeadState] = None) -> HeadState:
    """One plausible posture drawn from how people actually sit.

    ``scale`` controls translation and yaw/pitch drift; ``tilt`` controls roll
    independently, because head tilt is the axis this project historically got
    wrong and it is useful to vary it on its own.
    """
    base = base or HeadState()
    return base.shifted(
        dx=rng.normal(0.0, 28.0 * scale),
        dy=rng.normal(0.0, 20.0 * scale),
        dz=rng.normal(0.0, 35.0 * scale),
        dyaw=rng.normal(0.0, 4.5 * scale),
        dpitch=rng.normal(0.0, 3.5 * scale),
        droll=rng.normal(0.0, 6.0 * tilt),
    )


def probe_targets(screen: Sequence[int] = (1920, 1080), grid: int = 4,
                  margin: float = 0.12) -> list[Tuple[float, float]]:
    """An evaluation grid deliberately offset from the calibration pattern."""
    width, height = screen
    steps = [margin + (1.0 - 2 * margin) * i / (grid - 1) for i in range(grid)]
    return [(sx * width, sy * height) for sy in steps for sx in steps]
