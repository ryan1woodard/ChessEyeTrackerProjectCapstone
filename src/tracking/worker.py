"""The tracking worker thread.

All camera I/O and MediaPipe inference happen here. The GUI thread only ever
receives finished :class:`GazeSample` objects through a queued signal, so it
stays responsive no matter how slow inference is.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal

from ..utils.config import Config
from .calibration import CalibrationProfile
from .camera import CameraError, CameraManager
from .face_tracker import FaceTrackerError
from .pipeline import GazeSample, TrackingPipeline
from .smoother import SMOOTHING_PRESETS

logger = logging.getLogger(__name__)


class TrackerWorker(QThread):
    """Runs the capture/inference loop until asked to stop."""

    sample_ready = Signal(object)      # GazeSample
    preview_ready = Signal(object)     # np.ndarray (BGR), only when enabled
    error_occurred = Signal(str)
    status_changed = Signal(str)

    def __init__(self, config: Config, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._mutex = QMutex()
        self._running = False
        self._pending_profile: Optional[CalibrationProfile] = None
        self._profile_dirty = False
        self._clear_profile = False
        self._preview_enabled = False
        self._pending_offset: Optional[tuple] = None
        self._preview_interval = 1.0 / 12.0
        self._last_preview = 0.0
        self.pipeline: Optional[TrackingPipeline] = None

    # ------------------------------------------------------------- controls
    def set_profile(self, profile: Optional[CalibrationProfile]) -> None:
        """Thread-safely queue a calibration change; applied on the next frame."""
        with QMutexLocker(self._mutex):
            self._pending_profile = profile
            self._clear_profile = profile is None
            self._profile_dirty = True

    def set_offset(self, dx: float, dy: float) -> None:
        """Queue a bias correction, applied on the next frame."""
        with QMutexLocker(self._mutex):
            self._pending_offset = (float(dx), float(dy))

    def _take_offset(self):
        with QMutexLocker(self._mutex):
            offset, self._pending_offset = self._pending_offset, None
            return offset

    def set_preview_enabled(self, enabled: bool) -> None:
        with QMutexLocker(self._mutex):
            self._preview_enabled = bool(enabled)

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False

    def _should_run(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._running

    def _take_pending_profile(self):
        with QMutexLocker(self._mutex):
            if not self._profile_dirty:
                return False, None
            self._profile_dirty = False
            return True, self._pending_profile

    def _preview_wanted(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._preview_enabled

    # ------------------------------------------------------------------ loop
    def run(self) -> None:  # noqa: D102 - QThread entry point
        with QMutexLocker(self._mutex):
            self._running = True

        camera = CameraManager(
            device_index=int(self._config.get("camera.device_index", 0)),
            width=int(self._config.get("camera.width", 1280)),
            height=int(self._config.get("camera.height", 720)),
            fps=int(self._config.get("camera.fps", 30)),
        )
        pipeline: Optional[TrackingPipeline] = None
        try:
            camera.open()
            self.status_changed.emit(
                f"Camera {camera.device_index} open at "
                f"{camera.frame_size[0]}x{camera.frame_size[1]}"
            )
            try:
                pipeline = self._build_pipeline()
            except FaceTrackerError as exc:
                # Setup problem, not a transient fault: report it verbatim so the
                # user sees the remediation steps rather than a stack trace.
                logger.error("Face tracking backend unavailable: %s", exc)
                self.error_occurred.emit(str(exc))
                return
            self.pipeline = pipeline

            target_period = 1.0 / max(float(self._config.get("camera.fps", 30)), 1.0)
            while self._should_run():
                loop_start = time.monotonic()

                dirty, profile = self._take_pending_profile()
                if dirty:
                    try:
                        pipeline.apply_profile(profile)
                    except Exception as exc:  # pragma: no cover - corrupt profile
                        logger.exception("Failed to apply calibration profile")
                        self.error_occurred.emit(f"Calibration could not be loaded: {exc}")

                offset = self._take_offset()
                if offset is not None:
                    pipeline.offset = offset
                    logger.info("Gaze offset set to (%.0f, %.0f)", *offset)

                try:
                    frame = camera.read()
                except CameraError as exc:
                    logger.error("Camera failure: %s", exc)
                    self.error_occurred.emit(str(exc))
                    break
                if frame is None:
                    self.msleep(5)
                    continue

                try:
                    sample = pipeline.process_frame(frame)
                except Exception as exc:  # pragma: no cover - inference failure
                    logger.exception("Frame processing failed")
                    self.error_occurred.emit(f"Tracking error: {exc}")
                    self.msleep(50)
                    continue

                self.sample_ready.emit(sample)
                self._maybe_emit_preview(frame, sample)

                elapsed = time.monotonic() - loop_start
                remaining = target_period - elapsed
                if remaining > 0.001:
                    self.msleep(int(remaining * 1000))
        except CameraError as exc:
            logger.error("Camera could not be started: %s", exc)
            self.error_occurred.emit(str(exc))
        except Exception as exc:  # pragma: no cover - unexpected
            logger.exception("Tracker worker crashed")
            self.error_occurred.emit(f"Unexpected tracking error: {exc}")
        finally:
            if pipeline is not None:
                pipeline.close()
            self.pipeline = None
            camera.release()
            with QMutexLocker(self._mutex):
                self._running = False
            self.status_changed.emit("Tracking stopped")
            logger.info("Tracker worker finished")

    # ----------------------------------------------------------------- setup
    def _build_pipeline(self) -> TrackingPipeline:
        preset = str(self._config.get("tracking.smoothing_preset", "medium"))
        pipeline = TrackingPipeline(
            backend=str(self._config.get("tracking.backend", "auto")),
            smoothing_preset=preset,
            max_yaw=float(self._config.get("tracking.max_head_yaw_deg", 40.0)),
            max_pitch=float(self._config.get("tracking.max_head_pitch_deg", 30.0)),
            blink_threshold=float(self._config.get("tracking.blink_ear_threshold", 0.16)),
            blink_recovery_seconds=float(
                self._config.get("tracking.blink_recovery_ms", 120)) / 1000.0,
        )
        if preset not in SMOOTHING_PRESETS:
            pipeline.smoother.set_parameters(
                float(self._config.get("tracking.one_euro_min_cutoff", 1.0)),
                float(self._config.get("tracking.one_euro_beta", 0.007)),
                float(self._config.get("tracking.one_euro_d_cutoff", 1.0)),
            )
        return pipeline

    def _maybe_emit_preview(self, frame: np.ndarray, sample: GazeSample) -> None:
        if not self._preview_wanted():
            return
        now = time.monotonic()
        if now - self._last_preview < self._preview_interval:
            return
        self._last_preview = now
        # A copy is essential: the worker reuses its own frame buffers.
        self.preview_ready.emit(frame.copy())
