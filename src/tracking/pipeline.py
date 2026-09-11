"""The frame-processing pipeline, free of any Qt dependency.

    frame -> landmarks -> head pose -> features -> gaze model -> smoothing

Keeping this class GUI-free means the whole chain can be exercised in tests
with synthetic landmarks and no webcam.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np

from ..utils.geometry import clamp
from .blink import BlinkDetector
from .calibration import CalibrationProfile
from .face_tracker import FaceTracker
from .features import FeatureExtractor, FeatureVector
from .gaze_estimator import GazeEstimator, GazeResult
from .head_pose import HeadPose, HeadPoseEstimator
from .smoother import (FEATURE_SMOOTHING_PRESETS, FeatureSmoother, GazeSmoother)

logger = logging.getLogger(__name__)


@dataclass
class GazeSample:
    """Everything one processed frame tells us. Immutable once emitted."""

    timestamp: float
    face_detected: bool = False
    features_valid: bool = False
    calibrated: bool = False
    raw_x: float = 0.0
    raw_y: float = 0.0
    x: float = 0.0
    y: float = 0.0
    confidence: float = 0.0
    valid: bool = False
    eyes_closed: bool = False
    eye_openness: float = 0.0
    head_pose: HeadPose = field(default_factory=HeadPose.invalid)
    fps: float = 0.0
    #: True when this estimate is the previous one held through a blink.
    holding: bool = False
    features: Optional[FeatureVector] = None
    landmarks: Optional[object] = None  # FaceLandmarks, for the debug preview

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)


class ConfidenceScorer:
    """Combines several independent signals into a single 0-1 confidence.

    Each factor is a multiplier, so any one bad signal (closed eyes, extreme
    head angle, unstable estimates) is enough to suppress the result. This is
    intentional: it is far better to report low confidence than to report a
    confident number that is wrong.
    """

    def __init__(self, max_yaw: float = 40.0, max_pitch: float = 30.0,
                 blink_threshold: float = 0.16, history: int = 8,
                 max_roll: float = 35.0) -> None:
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.max_roll = max_roll
        #: Openness at or below which the eyes count as shut, and the openness
        #: of a relaxed open eye. Both are kept current by the pipeline from the
        #: blink detector's learned values.
        self.blink_threshold = blink_threshold
        self.open_baseline: Optional[float] = None
        self._recent: Deque[Tuple[float, float]] = deque(maxlen=history)
        self.calibration_factor = 1.0

    def set_calibration_quality(self, mean_error_px: float, diagonal_px: float) -> None:
        """Scale confidence by how good the calibration was to begin with."""
        if diagonal_px <= 0:
            self.calibration_factor = 1.0
            return
        normalised = mean_error_px / diagonal_px
        # ~2% of the diagonal is excellent, ~10% is unusable.
        self.calibration_factor = float(clamp(1.0 - (normalised - 0.02) / 0.08, 0.2, 1.0))

    def reset(self) -> None:
        self._recent.clear()

    def score(self, features: FeatureVector, gaze: GazeResult,
              screen_diagonal: float) -> float:
        if not features.valid or not gaze.valid:
            self._recent.clear()
            return 0.0

        factors = [self.calibration_factor]

        pose = features.head_pose
        if pose is not None and pose.valid:
            yaw_penalty = clamp((abs(pose.yaw) - 15.0) / max(self.max_yaw - 15.0, 1.0), 0.0, 1.0)
            pitch_penalty = clamp((abs(pose.pitch) - 10.0) / max(self.max_pitch - 10.0, 1.0),
                                  0.0, 1.0)
            # Tilt counts alongside the other two. A head rolled further than
            # calibration ever saw is extrapolation just like a head turned
            # away, and reporting it confidently is how a wrong answer reaches
            # the event detector and becomes a recorded "look" at a square.
            roll_penalty = clamp((abs(pose.roll) - 12.0) / max(self.max_roll - 12.0, 1.0),
                                 0.0, 1.0)
            factors.append(1.0 - 0.85 * max(yaw_penalty, pitch_penalty, roll_penalty))
        else:
            factors.append(0.7)

        # Scored between this user's own shut and open levels, not against a
        # fixed number. Two things go wrong with a fixed one: a user whose
        # relaxed eyes read near it tracks perfectly well and is then reported
        # at zero confidence, so ``minimum_confidence`` discards every one of
        # their samples; and the scale has no upper anchor, so how close a
        # fully open eye comes to 1.0 depends on the user's eye shape rather
        # than on anything about the estimate.
        openness = features.eye_openness
        threshold = max(self.blink_threshold, 1e-6)
        wide_open = self.open_baseline if self.open_baseline else threshold * 2.0
        span = max(wide_open - threshold, 1e-6)
        factors.append(clamp((openness - threshold) / span, 0.0, 1.0))

        # Both eyes should report similar horizontal iris offsets. A large
        # disagreement usually means one iris landmark set is unreliable.
        disagreement = abs(features.get("iris_l_ix") - features.get("iris_r_ix"))
        factors.append(clamp(1.0 - disagreement / 0.35, 0.15, 1.0))

        self._recent.append((gaze.x, gaze.y))
        if len(self._recent) >= 4 and screen_diagonal > 0:
            points = np.array(self._recent, dtype=np.float64)
            spread = float(np.linalg.norm(points.std(axis=0)))
            factors.append(clamp(1.0 - spread / (0.12 * screen_diagonal), 0.2, 1.0))

        score = float(np.prod(factors))
        return float(clamp(score, 0.0, 1.0))


class TrackingPipeline:
    """Owns the per-frame processing chain and the active calibration."""

    def __init__(
        self,
        backend: str = "auto",
        smoothing_preset: str = "medium",
        max_yaw: float = 40.0,
        max_pitch: float = 30.0,
        blink_threshold: float = 0.16,
        blink_recovery_seconds: float = 0.12,
        max_roll: float = 35.0,
        reacquire_seconds: float = 0.5,
    ) -> None:
        #: The MediaPipe backend is opened on the first frame rather than here.
        #: Constructing it costs a second or more and needs a working model, so
        #: deferring it keeps the pipeline importable and testable on machines
        #: with no camera and lets the caller surface a load failure where the
        #: user can see it rather than inside a constructor.
        self.backend = backend
        self._face_tracker: Optional[FaceTracker] = None
        self.head_pose_estimator = HeadPoseEstimator()
        self.feature_extractor = FeatureExtractor()
        self.smoother = GazeSmoother.from_preset(smoothing_preset)
        feature_params = FEATURE_SMOOTHING_PRESETS.get(
            smoothing_preset, FEATURE_SMOOTHING_PRESETS["medium"])
        self.feature_smoother = FeatureSmoother(feature_params["min_cutoff"],
                                                feature_params["beta"])
        self.confidence = ConfidenceScorer(max_yaw, max_pitch, blink_threshold,
                                           max_roll=max_roll)
        self.blink_threshold = blink_threshold
        self.blink = BlinkDetector(blink_threshold)
        #: Extra hold after the eyes reopen; the first frames after a blink
        #: still have the lid partly across the iris.
        self.blink_recovery_seconds = blink_recovery_seconds
        #: How long a gap in tracking may last before the filters are treated
        #: as stale. Shorter than this and their state is still about the same
        #: glance, so keeping it avoids a visible jump; longer and it describes
        #: a moment that has passed, and blending it into the first new frame
        #: drags the estimate back to wherever the user was looking before.
        self.reacquire_seconds = reacquire_seconds
        self.estimator: Optional[GazeEstimator] = None
        #: Manual bias correction applied after the model (see "recentre").
        self.offset: Tuple[float, float] = (0.0, 0.0)
        self._last_valid: Optional[Tuple[float, float, float]] = None
        self._hold_until: float = 0.0
        self._last_seen: Optional[float] = None
        self.screen_size: Tuple[int, int] = (1920, 1080)
        self.screen_origin: Tuple[int, int] = (0, 0)
        self._frame_times: Deque[float] = deque(maxlen=30)
        self._start = time.monotonic()

    @property
    def face_tracker(self) -> FaceTracker:
        """The landmark backend, opened on first use."""
        if self._face_tracker is None:
            self._face_tracker = FaceTracker(backend=self.backend)
        return self._face_tracker

    def open(self) -> None:
        """Open the landmark backend now.

        Callers that want a backend failure reported up front -- rather than on
        the first frame, several layers deep -- call this while they are still
        somewhere a :class:`FaceTrackerError` can be shown to the user.
        """
        _ = self.face_tracker

    # ------------------------------------------------------------ calibration
    def apply_profile(self, profile: Optional[CalibrationProfile]) -> None:
        """Install (or clear) the active calibration profile."""
        if profile is None:
            self.estimator = None
            self.confidence.calibration_factor = 1.0
            logger.info("Calibration cleared")
            return
        estimator = profile.build_estimator()
        self.estimator = estimator
        self.screen_size = (profile.screen_width, profile.screen_height)
        self.screen_origin = tuple(profile.screen_origin)
        if profile.report is not None:
            # The settled figure, not the single-frame one: confidence is
            # reported for smoothed output, and the filters remove most of the
            # per-frame noise that makes the other number larger.
            report = profile.report
            self.confidence.set_calibration_quality(
                report.settled_error_px or report.mean_error_px,
                float(np.hypot(*self.screen_size)),
            )
        self._reset_filters()
        self.blink.reset()
        self.offset = (0.0, 0.0)
        logger.info("Calibration profile %s applied", profile.profile_id)

    def set_estimator(self, estimator: Optional[GazeEstimator]) -> None:
        self.estimator = estimator
        self._reset_filters()

    @property
    def is_calibrated(self) -> bool:
        return self.estimator is not None and self.estimator.is_ready

    # --------------------------------------------------------------- process
    def process_frame(self, frame_bgr: np.ndarray,
                      timestamp: Optional[float] = None) -> GazeSample:
        """Run one frame through the whole chain."""
        now = timestamp if timestamp is not None else time.monotonic()
        self._frame_times.append(now)
        fps = self._current_fps()

        landmarks = self.face_tracker.process(
            frame_bgr, timestamp_ms=int((now - self._start) * 1000)
        )
        if landmarks is None:
            self.confidence.reset()
            return GazeSample(timestamp=now, fps=fps, calibrated=self.is_calibrated)

        pose = self.head_pose_estimator.estimate(landmarks)
        features = self.feature_extractor.extract(landmarks, pose)
        return self.process_features(features, now, fps=fps, landmarks=landmarks)

    def process_features(self, features: FeatureVector, now: float,
                         fps: float = 0.0,
                         landmarks: Optional[object] = None) -> GazeSample:
        """Everything after landmark detection, for one frame.

        Split out from :meth:`process_frame` so the blink hold, the filters and
        the recovery logic can be driven directly from synthetic features, with
        no webcam and no MediaPipe in the way.
        """
        pose = features.head_pose if features.head_pose is not None else HeadPose.invalid()
        eyes_closed = features.valid and self.blink.update(features.eye_openness)
        sample = GazeSample(
            timestamp=now,
            face_detected=True,
            features_valid=features.valid,
            calibrated=self.is_calibrated,
            eye_openness=features.eye_openness,
            eyes_closed=eyes_closed,
            head_pose=pose,
            fps=fps,
            features=features,
            landmarks=landmarks,
        )
        if not features.valid or self.estimator is None or not self.estimator.is_ready:
            return sample

        # --- recover from a gap in tracking ---------------------------
        # The filters hold the last value they saw, with no notion of how long
        # ago that was. After the face is lost for a second or two -- a turn
        # away, a hand across the face, someone walking past -- that value
        # describes a moment that has gone, and blending it into the first
        # frame back drags the estimate towards where the user *used* to be
        # looking and holds it there while the filter catches up. Past the
        # re-acquisition window the state is discarded so tracking restarts
        # clean rather than sliding into place from a stale position.
        if self._last_seen is not None and now - self._last_seen > self.reacquire_seconds:
            self._reset_filters()
        self._last_seen = now

        # --- blink handling -------------------------------------------
        # While the lid covers the pupil, MediaPipe still reports iris
        # landmarks -- they are simply wrong. Feeding them to the model throws
        # the estimate, and worse, poisons the smoothing filter so the error
        # persists for a second or more after the eye reopens. So during a
        # closure (and briefly after it) the last good estimate is held and
        # nothing is fed to the filters at all.
        if sample.eyes_closed:
            self._hold_until = now + self.blink_recovery_seconds
        if now < self._hold_until and self._last_valid is not None:
            held_x, held_y, held_conf = self._last_valid
            sample.x, sample.y = held_x, held_y
            sample.raw_x, sample.raw_y = held_x, held_y
            sample.confidence = held_conf * 0.6
            sample.valid = True
            sample.holding = True
            return sample
        if sample.eyes_closed:
            return sample

        smoothed_values = self.feature_smoother.smooth(features.values, now)
        smoothed = FeatureVector(values=smoothed_values, valid=True,
                                 left=features.left, right=features.right,
                                 head_pose=features.head_pose)

        gaze = self.estimator.estimate(smoothed)
        if not gaze.valid:
            return sample

        diagonal = float(np.hypot(*self.screen_size))
        self.confidence.blink_threshold = self.blink.threshold()
        self.confidence.open_baseline = self.blink.baseline
        confidence = self.confidence.score(smoothed, gaze, diagonal)
        offset_x, offset_y = self.offset
        smooth_x, smooth_y = self.smoother.smooth(gaze.x + offset_x,
                                                 gaze.y + offset_y, now)

        sample.raw_x, sample.raw_y = gaze.x + offset_x, gaze.y + offset_y
        sample.x, sample.y = smooth_x, smooth_y
        sample.confidence = confidence
        sample.valid = True
        self._last_valid = (smooth_x, smooth_y, confidence)
        return sample

    def _reset_filters(self) -> None:
        """Drop all per-frame filter state, keeping the calibration."""
        self.smoother.reset()
        self.feature_smoother.reset()
        self.confidence.reset()
        self._last_valid = None
        self._hold_until = 0.0

    def _current_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        if self._face_tracker is not None:
            self._face_tracker.close()
            self._face_tracker = None
