"""Face, eye and iris landmark detection via MediaPipe.

Two backends are supported:

``face_mesh`` (default)
    ``mediapipe.solutions.face_mesh`` with ``refine_landmarks=True``. The model
    ships inside the ``mediapipe`` wheel, so the application works with no
    network access at all -- which matters given the privacy requirements.

``face_landmarker``
    The newer MediaPipe Tasks API. It additionally provides a facial transform
    matrix (a more stable head pose than solvePnP), but it needs the
    ``face_landmarker.task`` model file placed in ``assets/``. If the file is
    absent the tracker transparently falls back to ``face_mesh``.

Landmark index convention
-------------------------
MediaPipe indices are given in *image* space. ``EYE_LEFT_*`` therefore refers
to the eye on the left of the image, which is the user's **right** eye when
they face the camera. The naming is used consistently throughout the codebase;
because the gaze model is calibrated per user, the labelling never affects
accuracy -- only readability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..utils.model_download import (DEFAULT_MODEL_PATH,
                                    ensure_face_landmarker_model)

logger = logging.getLogger(__name__)

# --- image-left eye ---------------------------------------------------------
EYE_LEFT_OUTER = 33
EYE_LEFT_INNER = 133
EYE_LEFT_TOP = 159
EYE_LEFT_BOTTOM = 145
EYE_LEFT_TOP2 = 158
EYE_LEFT_BOTTOM2 = 153
IRIS_LEFT = (468, 469, 470, 471, 472)

#: The full eyelid contour, outer corner round the lower lid to the inner
#: corner and back along the upper lid. Averaging over the ring is what makes
#: the eye's scale and axis stable enough to normalise iris offsets against;
#: two corner points alone are far too noisy.
EYE_LEFT_RING = (33, 7, 163, 144, 145, 153, 154, 155, 133,
                 173, 157, 158, 159, 160, 161, 246)

# --- image-right eye --------------------------------------------------------
EYE_RIGHT_OUTER = 263
EYE_RIGHT_INNER = 362
EYE_RIGHT_TOP = 386
EYE_RIGHT_BOTTOM = 374
EYE_RIGHT_TOP2 = 385
EYE_RIGHT_BOTTOM2 = 380
IRIS_RIGHT = (473, 474, 475, 476, 477)
EYE_RIGHT_RING = (263, 249, 390, 373, 374, 380, 381, 382, 362,
                  398, 384, 385, 386, 387, 388, 466)

# --- pose reference points --------------------------------------------------
NOSE_TIP = 1
CHIN = 152
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
FOREHEAD = 10

#: Landmarks that do not move with expression, used to fit the rigid face
#: frame (see :mod:`~src.tracking.face_frame`).
#:
#: Membership is chosen for rigidity, not for how clearly the point shows up.
#: Eyebrows are excluded even though they track well, because they move when
#: someone frowns or concentrates -- which is most of a chess game. The mouth
#: and jawline are excluded for the same reason. What is left is the bony
#: structure: the eye corners, the bridge and sides of the nose, the temples
#: and the upper face oval.
RIGID_FACE_LANDMARKS = (
    # eye corners
    33, 133, 263, 362,
    # nose bridge, top to tip
    168, 6, 197, 195, 4, 1,
    # sides of the nose
    98, 327,
    # temples and the upper face oval
    127, 356, 234, 454,
    # forehead
    10, 151,
)

REQUIRED_LANDMARKS = 478  # 468 mesh points + 2 x 5 iris points


@dataclass
class FaceLandmarks:
    """Landmarks for a single detected face.

    Attributes
    ----------
    points:
        ``(N, 3)`` array of normalised coordinates in ``[0, 1]`` for x/y; z is
        MediaPipe's relative depth (smaller is closer to the camera).
    frame_width, frame_height:
        Size of the frame the landmarks were computed from, so callers can
        convert to pixels.
    has_iris:
        Whether iris refinement landmarks are present.
    transform_matrix:
        Optional 4x4 facial transformation matrix (Tasks backend only).
    """

    points: np.ndarray
    frame_width: int
    frame_height: int
    has_iris: bool
    transform_matrix: Optional[np.ndarray] = None

    def pixel(self, index: int) -> np.ndarray:
        """Return landmark ``index`` in pixel coordinates as ``[x, y]``."""
        point = self.points[index]
        return np.array([point[0] * self.frame_width, point[1] * self.frame_height],
                        dtype=np.float64)

    def pixels(self, indices) -> np.ndarray:
        """Return several landmarks in pixel coordinates as an ``(n, 2)`` array."""
        pts = self.points[list(indices), :2].astype(np.float64)
        pts[:, 0] *= self.frame_width
        pts[:, 1] *= self.frame_height
        return pts


class FaceTrackerError(RuntimeError):
    """Raised when no usable MediaPipe face landmark backend can be started."""


def legacy_solutions_available() -> bool:
    """Whether this MediaPipe build still ships the legacy ``solutions`` API."""
    try:
        import mediapipe as mp
        return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
    except Exception:  # pragma: no cover - broken install
        return False


class FaceTracker:
    """Thin, backend-agnostic wrapper around MediaPipe face landmark models."""

    def __init__(self, backend: str = "auto",
                 model_path: Optional[Path] = None,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 allow_model_download: bool = True) -> None:
        self.requested_backend = backend
        self.backend = backend
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._allow_model_download = allow_model_download
        self._impl = None
        self._closed = False
        self._last_timestamp_ms = -1
        self._open()

    # ------------------------------------------------------------------ setup
    def _open(self) -> None:
        """Start whichever backend this MediaPipe build actually supports.

        ``auto`` prefers the legacy ``solutions`` API when it exists, purely
        because it needs no model download. On MediaPipe builds where that API
        has been removed it falls back to Tasks.
        """
        requested = self.requested_backend
        errors: list[str] = []

        if requested in ("auto", "face_mesh"):
            if legacy_solutions_available():
                self._open_face_mesh()
                return
            message = ("this MediaPipe build has no 'solutions' module "
                       "(it was removed in newer releases)")
            if requested == "face_mesh":
                logger.warning("face_mesh backend unavailable: %s", message)
            errors.append(f"face_mesh: {message}")

        try:
            self._open_face_landmarker()
            return
        except Exception as exc:
            errors.append(f"face_landmarker: {exc}")
            logger.warning("FaceLandmarker backend unavailable: %s", exc)

        if requested == "face_landmarker":
            raise FaceTrackerError(errors[-1])

        raise FaceTrackerError(
            "No usable MediaPipe face tracking backend could be started.\n\n"
            + "\n".join(f"  - {line}" for line in errors)
            + "\n\nFix: install the MediaPipe version pinned in requirements.txt\n"
              "    pip install -r requirements.txt --upgrade\n"
              "or connect to the internet once so the model can be downloaded."
        )

    def _open_face_mesh(self) -> None:
        import mediapipe as mp

        self._impl = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,   # required: this is what adds the iris points
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        self.backend = "face_mesh"
        logger.info("FaceTracker using MediaPipe FaceMesh (refine_landmarks=True)")

    def _open_face_landmarker(self) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model = ensure_face_landmarker_model(self.model_path,
                                             allow_download=self._allow_model_download)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        self._impl = mp_vision.FaceLandmarker.create_from_options(options)
        self.backend = "face_landmarker"
        logger.info("FaceTracker using MediaPipe Tasks FaceLandmarker (%s)", model.name)

    # ------------------------------------------------------------- processing
    def process(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> Optional[FaceLandmarks]:
        """Detect a face in a BGR frame. Returns ``None`` when no face is found."""
        if self._impl is None or self._closed:
            return None
        import cv2

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.backend == "face_landmarker":
            return self._process_tasks(rgb, width, height, timestamp_ms)
        return self._process_face_mesh(rgb, width, height)

    def _process_face_mesh(self, rgb: np.ndarray, width: int,
                           height: int) -> Optional[FaceLandmarks]:
        rgb.flags.writeable = False
        result = self._impl.process(rgb)
        if not result.multi_face_landmarks:
            return None
        landmarks = result.multi_face_landmarks[0].landmark
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
        return FaceLandmarks(points, width, height, has_iris=len(points) >= REQUIRED_LANDMARKS)

    def _process_tasks(self, rgb: np.ndarray, width: int, height: int,
                       timestamp_ms: int) -> Optional[FaceLandmarks]:
        import mediapipe as mp

        # Tasks VIDEO mode rejects a timestamp that is not strictly greater
        # than the previous one, which happens whenever two frames land in the
        # same millisecond.
        stamp = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = stamp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._impl.detect_for_video(image, stamp)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
        matrix = None
        if getattr(result, "facial_transformation_matrixes", None):
            matrix = np.array(result.facial_transformation_matrixes[0], dtype=np.float64)
        return FaceLandmarks(points, width, height,
                             has_iris=len(points) >= REQUIRED_LANDMARKS,
                             transform_matrix=matrix)

    # ------------------------------------------------------------------ close
    def close(self) -> None:
        if self._impl is not None and not self._closed:
            try:
                self._impl.close()
            except Exception as exc:  # pragma: no cover
                logger.debug("Error closing face tracker: %s", exc)
        self._closed = True
        self._impl = None
