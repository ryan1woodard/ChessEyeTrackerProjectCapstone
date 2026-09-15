"""The rigid face frame: one least-squares fit that anchors everything else.

Why this exists
---------------
Every eye measurement is a displacement from some reference point, divided by
some scale. Both used to come from two landmarks -- the eye corners -- and that
is the accuracy ceiling of the whole tracker. Measured in the simulator with
0.35 px of per-landmark noise, the corner midpoint wobbles by 0.35 px against
the iris centre's 0.22 px, so **71% of the variance in the iris offset comes
from the reference, not from the iris**. Everything downstream inherits it: the
polynomial amplifies it, the smoothing filter can only trade it against lag,
and no amount of model work recovers information that was never measured.

Averaging more landmarks fixes it, and the honest way to average them is to fit
the thing they actually are: one rigid object seen from one viewpoint. This
module fits a **scaled orthographic** (weak perspective) camera to a canonical
3-D head,

    x = M @ X + t,    M is 2x3, t is 2x1

which is exactly the projection of a rigid object when its depth variation is
small next to its distance -- true of a face at arm's length. It is linear in
the eight unknowns, so it is one least-squares solve over 18 landmarks rather
than an iterative pose search, with nothing to diverge and no previous-frame
state to be poisoned by a bad frame.

What comes out of it
--------------------
``M`` is a scaled rotation, so a single fit yields all of:

* a stable reference point anywhere on the head, by projecting the canonical
  position of it -- including the eyeball centres, which have no landmark;
* the eye axis and eye width, without touching the corner landmarks;
* head orientation, recovered from ``M`` by the nearest scaled-orthonormal
  matrix, which is steadier than ``solvePnP`` and cannot run away;
* apparent scale, which is proportional to 1/distance.

Reference noise drops from 0.35 px to 0.16 px and the eye-width estimate from
0.48 px to 0.06 px, which together cut the end-to-end error from 40 px to
28 px.

The canonical head is an average, not the user's. That mismatch shows up as a
fixed error in the fitted frame, which calibration absorbs along with every
other per-user constant; what matters is that the frame moves *with the head*
correctly, and a rigid fit does that whoever is sitting in front of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import face_tracker as fl
from .face_tracker import FaceLandmarks

#: Canonical head in millimetres: ``x`` right, ``y`` down, ``z`` towards the
#: back of the head, origin at the midpoint of the eyes. Average adult
#: proportions; the per-user difference is absorbed by calibration.
CANONICAL_FACE_MM: Dict[int, Tuple[float, float, float]] = {
    # eye corners
    fl.EYE_LEFT_OUTER: (-44.0, 0.0, 9.0),
    fl.EYE_LEFT_INNER: (-15.5, 0.5, -4.0),
    fl.EYE_RIGHT_OUTER: (44.0, 0.0, 9.0),
    fl.EYE_RIGHT_INNER: (15.5, 0.5, -4.0),
    # nose bridge, between the eyes down to the tip
    168: (0.0, -3.0, -7.0),
    6: (0.0, 7.0, -15.0),
    197: (0.0, 1.0, -11.0),
    195: (0.0, 15.0, -21.0),
    4: (0.0, 27.0, -26.0),
    fl.NOSE_TIP: (0.0, 32.0, -27.0),
    # sides of the nose
    98: (-16.5, 37.0, -13.0),
    327: (16.5, 37.0, -13.0),
    # temples and the upper face oval
    127: (-71.0, -7.0, 39.0),
    356: (71.0, -7.0, 39.0),
    234: (-72.0, 19.0, 35.0),
    454: (72.0, 19.0, 35.0),
    # forehead
    fl.FOREHEAD: (0.0, -56.0, 7.0),
    151: (0.0, -41.0, -1.0),
}

#: Centres of rotation of the two eyeballs, in the same canonical frame. No
#: landmark marks these -- they are inside the head -- which is exactly why
#: projecting them through the fitted frame is worth doing: it is the reference
#: an iris displacement should be measured from.
CANONICAL_EYE_CENTRE_MM: Dict[str, Tuple[float, float, float]] = {
    "left": (-30.0, 0.0, 11.5),
    "right": (30.0, 0.0, 11.5),
}

#: Canonical corner-to-corner eye width in millimetres, used to turn the
#: fitted scale into a pixel width without touching the corner landmarks.
CANONICAL_EYE_WIDTH_MM = 28.5

#: Fits looser than this are not trusted, as a fraction of the apparent
#: interocular distance.
#:
#: Deliberately far above anything a real face produces. Measured in the
#: simulator, the residual is set almost entirely by how much the subject
#: differs from the canonical head -- 0.02 for an exactly average face, 0.07 at
#: a realistic 3 mm per-landmark deviation, 0.20 at a very unusual 9 mm -- and
#: barely moves with head pose at all, 0.06 whether the head is square on or
#: turned 50 degrees. A tight threshold therefore does not reject bad poses; it
#: rejects unusual *people*, and hands exactly the users who most need a stable
#: reference the worst one available. This is a check for landmarks that are
#: not a face at all.
MAX_RESIDUAL_FRACTION = 0.35

_RIGID_IDS: Tuple[int, ...] = tuple(
    index for index in fl.RIGID_FACE_LANDMARKS if index in CANONICAL_FACE_MM
)
_RIGID_MODEL = np.array([CANONICAL_FACE_MM[i] for i in _RIGID_IDS], dtype=np.float64)

#: Minimum landmarks for the fit. Eight points give 16 equations against 8
#: unknowns, which is enough to be over-determined but not enough to average
#: much noise away; in practice all 18 are present or none are.
MIN_RIGID_POINTS = 8


@dataclass(frozen=True)
class FaceFrame:
    """A fitted scaled-orthographic view of the canonical head.

    ``matrix`` and ``offset`` map canonical millimetres to image pixels. The
    frame is rigid: anything with a canonical position can be projected through
    it, whether or not a landmark marks it.
    """

    matrix: np.ndarray            # (2, 3)
    offset: np.ndarray            # (2,)
    #: Root-mean-square fit residual in pixels.
    residual_px: float
    #: Residual as a fraction of the apparent interocular distance, so the
    #: quality measure does not depend on how close the user is sitting.
    residual_fraction: float
    valid: bool = True

    @classmethod
    def invalid(cls) -> "FaceFrame":
        return cls(np.zeros((2, 3)), np.zeros(2), 0.0, 0.0, valid=False)

    # ------------------------------------------------------------- geometry
    def project(self, point_mm: Sequence[float]) -> np.ndarray:
        """Where a canonical head point lands in the image, in pixels."""
        return self.matrix @ np.asarray(point_mm, dtype=np.float64) + self.offset

    def direction(self, vector_mm: Sequence[float]) -> np.ndarray:
        """Where a canonical head *direction* points in the image."""
        return self.matrix @ np.asarray(vector_mm, dtype=np.float64)

    @property
    def scale(self) -> float:
        """Pixels per canonical millimetre, averaged over the two image axes.

        Taken from the singular values rather than from a row norm, because
        under head rotation the two rows shorten by different amounts and their
        average is the isotropic scale that survives it.
        """
        return float(np.linalg.svd(self.matrix, compute_uv=False)[:2].mean())

    @property
    def eye_width_px(self) -> float:
        """Apparent corner-to-corner eye width implied by the fitted scale."""
        return self.scale * CANONICAL_EYE_WIDTH_MM

    def eye_centre(self, side: str) -> np.ndarray:
        """Projected centre of rotation of one eyeball, in pixels."""
        return self.project(CANONICAL_EYE_CENTRE_MM[side])

    def eye_axis(self, side: str) -> np.ndarray:
        """Unit vector along one eye, outer corner towards inner, in the image."""
        outer = CANONICAL_FACE_MM[fl.EYE_LEFT_OUTER if side == "left"
                                  else fl.EYE_RIGHT_OUTER]
        inner = CANONICAL_FACE_MM[fl.EYE_LEFT_INNER if side == "left"
                                  else fl.EYE_RIGHT_INNER]
        axis = self.direction(np.asarray(inner) - np.asarray(outer))
        norm = float(np.linalg.norm(axis))
        return axis / norm if norm > 1e-9 else np.array([1.0, 0.0])

    # ---------------------------------------------------------- orientation
    def rotation(self) -> np.ndarray:
        """The head rotation implied by the fit, as a 3x3 matrix.

        ``matrix`` is a scale times the first two rows of a rotation. Those
        rows come back from least squares neither orthogonal nor of equal
        length, so the nearest scaled-orthonormal pair is taken via an SVD --
        the standard orthogonal Procrustes step -- and the third row is their
        cross product.
        """
        u, _s, vt = np.linalg.svd(self.matrix, full_matrices=False)
        rows = u @ vt                      # (2, 3), orthonormal rows
        third = np.cross(rows[0], rows[1])
        return np.vstack([rows, third])

    def euler_degrees(self) -> Tuple[float, float, float]:
        """``(pitch, yaw, roll)`` in the conventions of :class:`HeadPose`.

        Roll is taken straight from the fitted image axis rather than from the
        decomposition, because it is directly observable in the image and does
        not depend on resolving the rotation's out-of-plane part.
        """
        rotation = self.rotation()
        # Scaled-orthographic projection cannot tell a rotation from its
        # mirror about the image plane, so the sign of the out-of-plane
        # component is a choice; take the solution facing the camera.
        if rotation[2, 2] < 0:
            rotation = rotation * np.array([[1.0], [1.0], [-1.0]])
        # Signs checked against the known rotation: a head turned towards the
        # image-right puts -sin(yaw) in [0, 2], and a lifted chin +sin(pitch)
        # in [1, 2].
        yaw = math.degrees(math.asin(float(np.clip(-rotation[0, 2], -1.0, 1.0))))
        pitch = math.degrees(math.asin(float(np.clip(rotation[1, 2], -1.0, 1.0))))
        axis = self.direction((1.0, 0.0, 0.0))
        roll = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
        return pitch, yaw, roll


def fit_face_frame(landmarks: FaceLandmarks) -> FaceFrame:
    """Fit the rigid frame to one frame's landmarks.

    Solves ``[X 1] @ P = x`` in least squares, where ``X`` are the canonical
    3-D positions of the rigid landmarks and ``x`` their observed pixels.
    """
    if landmarks.points.shape[0] <= max(_RIGID_IDS):
        return FaceFrame.invalid()
    observed = landmarks.pixels(_RIGID_IDS)
    if observed.shape[0] < MIN_RIGID_POINTS or not np.all(np.isfinite(observed)):
        return FaceFrame.invalid()

    design = np.column_stack([_RIGID_MODEL, np.ones(len(_RIGID_IDS))])
    solution, *_ = np.linalg.lstsq(design, observed, rcond=None)
    matrix = solution[:3].T                      # (2, 3)
    offset = solution[3]

    residuals = observed - design @ solution
    residual_px = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    scale = float(np.linalg.svd(matrix, compute_uv=False)[:2].mean())
    if not np.isfinite(scale) or scale <= 1e-6:
        return FaceFrame.invalid()
    interocular_px = scale * abs(CANONICAL_EYE_CENTRE_MM["right"][0]
                                 - CANONICAL_EYE_CENTRE_MM["left"][0])
    fraction = residual_px / max(interocular_px, 1e-6)

    return FaceFrame(matrix=matrix, offset=offset, residual_px=residual_px,
                     residual_fraction=fraction,
                     valid=fraction <= MAX_RESIDUAL_FRACTION)


def canonical_eye_width_px(frame: Optional[FaceFrame], fallback: float) -> float:
    """Eye width from the rigid frame where it is usable, else the measured one."""
    if frame is None or not frame.valid:
        return fallback
    width = frame.eye_width_px
    return width if width > 1e-6 else fallback
