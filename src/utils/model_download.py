"""One-time model download for the MediaPipe Tasks backend.

Why this exists
---------------
MediaPipe changed its packaging. Up to roughly 0.10.14 the ``mediapipe.solutions``
API shipped its models *inside the wheel*, so face tracking worked completely
offline. From about 0.10.30 the legacy ``solutions`` module was removed
entirely, and the replacement Tasks API expects you to supply the model file
yourself.

So on a newer MediaPipe the application needs ``face_landmarker.task`` on disk.
It is roughly 3.8 MB and is fetched once, from Google's model host.

This is the **only** network access anywhere in the application, it happens at
most once, it downloads a model file and nothing else, and it uploads nothing.
If you would rather stay strictly offline, either install the pinned MediaPipe
version from ``requirements.txt`` (which needs no download at all) or place the
file in ``assets/`` yourself and it will be used as-is.
"""

from __future__ import annotations

import logging
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .config import PROJECT_ROOT

logger = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "assets" / "face_landmarker.task"
MIN_MODEL_BYTES = 500_000  # a truncated or error-page download is far smaller


class ModelDownloadError(RuntimeError):
    """Raised when the landmark model is missing and cannot be fetched."""


def model_is_present(path: Path | None = None) -> bool:
    path = path or DEFAULT_MODEL_PATH
    return path.exists() and path.stat().st_size >= MIN_MODEL_BYTES


def ensure_face_landmarker_model(path: Path | None = None,
                                 allow_download: bool = True,
                                 timeout: float = 60.0) -> Path:
    """Return a usable model path, downloading it once if required."""
    path = path or DEFAULT_MODEL_PATH
    if model_is_present(path):
        return path

    if path.exists():
        logger.warning("Model at %s looks truncated (%d bytes); re-fetching",
                       path, path.stat().st_size)
        path.unlink()

    if not allow_download:
        raise ModelDownloadError(
            f"The face landmark model is missing and downloading is disabled.\n\n"
            f"Place face_landmarker.task in {path.parent} manually, or install the "
            f"MediaPipe version pinned in requirements.txt, which needs no model file."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading face landmark model (~3.8 MB) to %s", path)
    try:
        # Download to a temporary file first so an interrupted transfer can
        # never leave a half-written model that fails confusingly later.
        with urllib.request.urlopen(MODEL_URL, timeout=timeout) as response:
            with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent),
                                             suffix=".part") as handle:
                temporary = Path(handle.name)
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    handle.write(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ModelDownloadError(
            f"Could not download the face landmark model: {exc}\n\n"
            f"Either connect to the internet briefly and restart, download\n"
            f"{MODEL_URL}\nmanually into {path.parent}, or install the MediaPipe "
            f"version pinned in requirements.txt, which bundles its own models."
        ) from exc

    if temporary.stat().st_size < MIN_MODEL_BYTES:
        temporary.unlink(missing_ok=True)
        raise ModelDownloadError(
            "The downloaded model file was too small to be valid. "
            "Check your connection or any proxy that may be intercepting it."
        )

    temporary.replace(path)
    logger.info("Model downloaded (%.1f MB)", path.stat().st_size / 1e6)
    return path
