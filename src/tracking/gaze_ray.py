"""A geometric gaze ray: where the eye actually points, in three dimensions.

Why this exists
---------------
The regression used to be handed flat image measurements -- the iris offset in
camera axes, plus head yaw, pitch and roll as numbers -- and asked to work out
what they meant together. For head *roll* that is nearly free, because roll is
an in-plane rotation and the image-aligned offset already absorbs it. For head
*turn* it is not: relating an iris offset to a screen position under yaw is a
three-dimensional rotation, and a degree-2 polynomial in fifteen variables can
only approximate one over the range it was shown. Calibration is not shown much
of that range, because nobody turns their head 30 degrees while staring at a
calibration dot.

Measured in the simulator, that is exactly where the old mapping failed: about
16 px of error facing the camera, and 48-57 px at 30 degrees of yaw.

So the rotation is done properly instead, following the standard model-based
formulation (Sugano's normalization, and the eyeball-model gaze literature):

1. a full perspective pose puts the eyeball's centre of rotation at a real
   point in space, in millimetres;
2. the iris landmark is back-projected into a ray and intersected with the
   eyeball sphere, giving the iris centre in space;
3. the line between them is the **optical axis** -- the direction the eye
   points, in camera coordinates, with head pose already accounted for by
   construction rather than by approximation;
4. rotating it into the head's own frame gives the eye-in-head direction, which
   is what the eye muscles actually control and is independent of where the
   head is.

What is left for calibration is the part that genuinely is per-person: the
angle between the optical axis and the visual axis (*kappa* -- the fovea is not
on the optical axis, so everyone's eye points a few degrees away from where
they are looking), the true eyeball radius, and where the screen is relative to
the camera. Those are a handful of numbers rather than a shape, which is a much
easier thing to fit from thirteen points.

This module produces features, not a final answer. The ray is geometrically
correct but rests on an assumed focal length and an average eyeball, so the
regression downstream still corrects it -- it simply starts from something that
already has the head rotation right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .face_frame import (CANONICAL_EYE_CENTRE_MM, HeadPlacement, camera_matrix)

#: Radius of the eyeball in millimetres, the sphere the iris centre rides on.
EYEBALL_RADIUS_MM = 11.0

#: Nominal screen geometry, used only to turn a ray into a plane intersection
#: so the regression gets a quantity shaped like a screen position.
#:
#: Every one of these is wrong for any particular desk, and none of them needs
#: to be right. The ray carries the head rotation, which is the hard part; a
#: wrong screen plane adds a smooth, nearly affine error that thirteen
#: calibration points remove easily. What matters is that they are *constant*,
#: so the same head pose always produces the same answer.
NOMINAL_MM_PER_PX = 0.277
NOMINAL_CAMERA_ABOVE_CENTRE_MM = 150.0

#: How far a ray may land off the nominal screen before it is treated as
#: meaningless rather than merely wrong, in nominal screen widths.
MAX_PLANE_EXCURSION = 3.0


@dataclass(frozen=True)
class EyeRay(object):
    """One eye's gaze, in three frames at once."""

    #: Eyeball centre of rotation, camera millimetres.
    origin: np.ndarray
    #: Optical axis in camera coordinates, unit length.
    direction: np.ndarray
    #: The same direction in the head's own frame: eye-in-head, pose-free.
    head_direction: np.ndarray
    #: Where the ray meets the nominal screen plane, in nominal screen widths
    #: from its centre. ``None`` when the ray does not meet it usefully.
    plane_hit: Optional[np.ndarray]
    valid: bool = True

    @classmethod
    def invalid(cls) -> "EyeRay":
        zero = np.zeros(3)
        return cls(zero, np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, -1.0]),
                   None, valid=False)

    @property
    def head_yaw(self) -> float:
        """Eye rotation within the head, left-right, in radians."""
        return math.atan2(float(self.head_direction[0]), -float(self.head_direction[2]))

    @property
    def head_pitch(self) -> float:
        """Eye rotation within the head, up-down, in radians."""
        return math.atan2(float(self.head_direction[1]),
                          math.hypot(float(self.head_direction[0]),
                                     float(self.head_direction[2])))


def back_project(pixel: Sequence[float], camera: np.ndarray) -> np.ndarray:
    """A unit ray from the camera centre through an image point."""
    x = (float(pixel[0]) - camera[0, 2]) / camera[0, 0]
    y = (float(pixel[1]) - camera[1, 2]) / camera[1, 1]
    ray = np.array([x, y, 1.0], dtype=np.float64)
    return ray / float(np.linalg.norm(ray))


def intersect_sphere(ray: np.ndarray, centre: np.ndarray,
                     radius: float) -> Optional[np.ndarray]:
    """Nearest intersection of a ray from the origin with a sphere.

    Returns ``None`` when the ray misses. A miss is not exotic: the iris
    landmark and the eyeball centre come from different estimates, so a
    slightly wrong eyeball centre puts the ray just outside the sphere. The
    caller falls back rather than pretending.
    """
    # The ray starts at the camera centre, so its origin term drops out.
    along = float(ray @ centre)
    if along <= 0.0:
        return None
    gap = float(centre @ centre) - along * along
    inside = radius * radius - gap
    if inside < 0.0:
        return None
    return ray * (along - math.sqrt(inside))


def nearest_point_on_ray(ray: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Where a ray passes closest to a point; used when the sphere is missed.

    Grazing the eyeball instead of hitting it means the estimate of its centre
    was slightly off, not that the eye is somewhere else, so the closest
    approach is the right answer to fall back to: it is the same point the
    intersection converges to as the miss goes to zero.
    """
    return ray * max(float(ray @ centre), 1.0)


def eye_ray(pixel: Sequence[float], side: str, placement: HeadPlacement,
            camera: np.ndarray,
            eyeball_radius_mm: float = EYEBALL_RADIUS_MM) -> EyeRay:
    """The gaze ray for one eye, from its iris landmark and the head pose."""
    if not placement.valid:
        return EyeRay.invalid()

    origin = placement.to_camera(CANONICAL_EYE_CENTRE_MM[side])
    if not np.all(np.isfinite(origin)) or origin[2] <= 1.0:
        return EyeRay.invalid()

    ray = back_project(pixel, camera)
    surface = intersect_sphere(ray, origin, eyeball_radius_mm)
    if surface is None:
        surface = nearest_point_on_ray(ray, origin)

    direction = surface - origin
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return EyeRay.invalid()
    direction = direction / norm

    head_direction = placement.rotation.T @ direction
    return EyeRay(origin=origin, direction=direction,
                  head_direction=head_direction,
                  plane_hit=_plane_hit(origin, direction), valid=True)


def _plane_hit(origin: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
    """Where a ray meets the nominal screen plane, in screen widths.

    The plane is ``z = 0`` -- the camera sits in it, which is true enough of a
    webcam clipped to the top of a monitor. A ray heading away from it, or one
    landing several screens away, carries nothing and is reported as a miss.
    """
    if direction[2] >= -1e-3:
        return None
    distance = -float(origin[2]) / float(direction[2])
    hit = origin + distance * direction
    nominal_width_mm = 1920.0 * NOMINAL_MM_PER_PX
    x = float(hit[0]) / nominal_width_mm
    y = (float(hit[1]) - NOMINAL_CAMERA_ABOVE_CENTRE_MM) / nominal_width_mm
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if abs(x) > MAX_PLANE_EXCURSION or abs(y) > MAX_PLANE_EXCURSION:
        return None
    return np.array([x, y], dtype=np.float64)


#: Angle used to measure how the screen hit responds to a change in kappa.
#: Large enough that the finite difference is not dominated by rounding, small
#: enough that the response is still linear over it.
KAPPA_PROBE_RAD = 0.05


def rotate_in_head(direction: np.ndarray, placement: HeadPlacement,
                   about_x: float, about_y: float) -> np.ndarray:
    """Turn a camera-frame direction by small angles in the **head's** frame.

    Kappa lives in the head's frame, not the camera's: the fovea sits at a
    fixed place on the back of the eyeball, so the angle between where the eye
    points and where the person is looking turns with the head. Applying it in
    camera coordinates instead is the mistake that makes a gaze estimate drift
    as the head turns.
    """
    local = placement.rotation.T @ direction
    # Small-angle rotations about the head's x and y axes, composed directly.
    turned = np.array([
        local[0] + about_y * local[2],
        local[1] - about_x * local[2],
        local[2] - about_y * local[0] + about_x * local[1],
    ], dtype=np.float64)
    norm = float(np.linalg.norm(turned))
    if norm < 1e-9:
        return direction
    return placement.rotation @ (turned / norm)


def kappa_response(ray: EyeRay, placement: HeadPlacement,
                   probe: float = KAPPA_PROBE_RAD
                   ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """How far the screen hit moves per radian of kappa, on each head axis.

    The point of returning this rather than a corrected ray is that kappa is
    unknown until the user has been calibrated, and this makes it *linear*: the
    true screen position is the uncorrected hit plus kappa_x times the first
    vector plus kappa_y times the second. Handing those two vectors to the
    existing ridge fit lets it solve for a per-user kappa as part of the same
    least-squares it already does, with no separate optimiser and no iteration.

    Returns ``None`` when any of the probes misses the screen plane.
    """
    if not (ray.valid and placement.valid) or ray.plane_hit is None:
        return None
    responses = []
    for about_x, about_y in ((probe, 0.0), (0.0, probe)):
        turned = rotate_in_head(ray.direction, placement, about_x, about_y)
        hit = _plane_hit(ray.origin, turned)
        if hit is None:
            return None
        responses.append((hit - ray.plane_hit) / probe)
    return responses[0], responses[1]


#: Displacement used to measure how the screen hit responds to moving the
#: eyeball centre, in millimetres.
CENTRE_PROBE_MM = 1.0


def centre_response(pixel: Sequence[float], side: str, placement: HeadPlacement,
                    camera: np.ndarray, base: np.ndarray,
                    eyeball_radius_mm: float = EYEBALL_RADIUS_MM,
                    probe: float = CENTRE_PROBE_MM
                    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """How far the screen hit moves per millimetre of eyeball-centre error.

    This is the term that matters most, and the reason is the lever arm. The
    direction of gaze is the line from the eyeball's centre to the iris, and
    those are only about 11 mm apart -- so an error of 2 mm in where the centre
    is taken to be is an angular error of roughly ten degrees. Worse, the error
    is fixed in the *head's* frame, so it swings into the lateral direction as
    the head turns and shows up as a gaze estimate that slides sideways the
    further round you look.

    Measured in the simulator with an otherwise perfect head pose, a 2 mm
    mismatch in the assumed centre was worth 253 px of drift at 30 degrees of
    yaw -- far more than the pose error it was being blamed on.

    Nobody's eyeball sits where the average one does, so the offset has to come
    from the user. Returning the response to each axis makes it another set of
    linear coefficients for the ridge fit to solve, alongside kappa.
    """
    if not placement.valid:
        return None
    responses = []
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = probe
        moved = placement.to_camera(
            np.asarray(CANONICAL_EYE_CENTRE_MM[side]) + step)
        ray = back_project(pixel, camera)
        surface = intersect_sphere(ray, moved, eyeball_radius_mm)
        if surface is None:
            surface = nearest_point_on_ray(ray, moved)
        direction = surface - moved
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        hit = _plane_hit(moved, direction / norm)
        if hit is None:
            return None
        responses.append((hit - base) / probe)
    return responses[0], responses[1], responses[2]


def both_eyes(left_pixel: Sequence[float], right_pixel: Sequence[float],
              placement: HeadPlacement, frame_width: int, frame_height: int,
              eyeball_radius_mm: float = EYEBALL_RADIUS_MM
              ) -> Tuple[EyeRay, EyeRay]:
    """Gaze rays for both eyes from one frame."""
    camera = camera_matrix(frame_width, frame_height)
    return (eye_ray(left_pixel, "left", placement, camera, eyeball_radius_mm),
            eye_ray(right_pixel, "right", placement, camera, eyeball_radius_mm))
