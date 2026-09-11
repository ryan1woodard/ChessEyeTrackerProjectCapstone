"""Calibration: target patterns, sample collection and profile persistence.

Calibration is the single biggest determinant of accuracy in a webcam gaze
tracker, so this module is conservative about what it accepts:

* samples taken while the face is missing, an eye is closed, or the head is
  turned far away are discarded outright;
* the remaining samples for each target are filtered with a median-absolute-
  deviation test, which removes the frames where the user glanced away or
  blinked mid-collection without discarding the whole target;
* a target that ends up with too few surviving samples is dropped, and the fit
  refuses to proceed if fewer than five targets survive.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.config import PROJECT_ROOT
from .features import FeatureVector
from .gaze_estimator import FitReport, RidgeGazeEstimator

logger = logging.getLogger(__name__)

CALIBRATION_DIR = PROJECT_ROOT / "data" / "calibrations"

MIN_SAMPLES_PER_POINT = 4
MIN_POINTS_FOR_FIT = 5
MAD_THRESHOLD = 3.5
GROSS_OUTLIER_Z = 12.0


# ------------------------------------------------------------------ patterns
def calibration_pattern(name: str = "13point", margin: float = 0.08) -> List[Tuple[float, float]]:
    """Return normalised ``(x, y)`` target positions in ``[0, 1]``.

    A serpentine ordering is used for the outer grid so the user's eyes travel
    the shortest possible path between consecutive targets.
    """
    low, mid, high = margin, 0.5, 1.0 - margin
    if name == "5point":
        return [(low, low), (high, low), (mid, mid), (low, high), (high, high)]

    grid = [
        (low, low), (mid, low), (high, low),
        (high, mid), (mid, mid), (low, mid),
        (low, high), (mid, high), (high, high),
    ]
    if name == "9point":
        return grid

    inner_low, inner_high = 0.5 - (0.5 - margin) * 0.5, 0.5 + (0.5 - margin) * 0.5
    extras = [
        (inner_low, inner_low), (inner_high, inner_low),
        (inner_high, inner_high), (inner_low, inner_high),
    ]
    return grid + extras


def pattern_to_pixels(pattern: Sequence[Tuple[float, float]], width: int, height: int,
                      origin: Tuple[int, int] = (0, 0)) -> List[Tuple[float, float]]:
    """Convert normalised targets into absolute virtual-desktop pixels."""
    ox, oy = origin
    return [(ox + nx * width, oy + ny * height) for nx, ny in pattern]


# ------------------------------------------------------------------ profiles
@dataclass
class CalibrationProfile:
    """A saved calibration: the fitted model plus the context it is valid for."""

    profile_id: str
    name: str
    created_at: str
    monitor_index: int
    screen_width: int
    screen_height: int
    screen_origin: Tuple[int, int]
    feature_names: List[str]
    estimator_data: Dict
    report: Optional[FitReport] = None
    camera_index: int = 0
    pattern: str = "13point"

    def to_dict(self) -> Dict:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "created_at": self.created_at,
            "monitor_index": self.monitor_index,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "screen_origin": list(self.screen_origin),
            "feature_names": self.feature_names,
            "estimator": self.estimator_data,
            "report": self.report.to_dict() if self.report else None,
            "camera_index": self.camera_index,
            "pattern": self.pattern,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CalibrationProfile":
        return cls(
            profile_id=data["profile_id"],
            name=data.get("name", data["profile_id"]),
            created_at=data.get("created_at", ""),
            monitor_index=int(data.get("monitor_index", 1)),
            screen_width=int(data.get("screen_width", 1920)),
            screen_height=int(data.get("screen_height", 1080)),
            screen_origin=tuple(data.get("screen_origin", (0, 0))),
            feature_names=list(data.get("feature_names", [])),
            estimator_data=data["estimator"],
            report=FitReport.from_dict(data["report"]) if data.get("report") else None,
            camera_index=int(data.get("camera_index", 0)),
            pattern=data.get("pattern", "13point"),
        )

    def build_estimator(self) -> RidgeGazeEstimator:
        return RidgeGazeEstimator.from_dict(self.estimator_data)

    def matches_screen(self, width: int, height: int) -> bool:
        return self.screen_width == width and self.screen_height == height

    def summary(self) -> str:
        if self.report is None:
            return self.name
        return (f"{self.name} - {self.report.mean_error_px:.0f} px "
                f"({self.report.quality()})")


class CalibrationStore:
    """Loads and saves calibration profiles as JSON files on disk."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.directory = directory or CALIBRATION_DIR
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, profile_id: str) -> Path:
        return self.directory / f"{profile_id}.json"

    def save(self, profile: CalibrationProfile) -> Path:
        path = self.path_for(profile.profile_id)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(profile.to_dict(), handle, indent=2)
        logger.info("Saved calibration profile %s", path.name)
        return path

    def load(self, profile_id: str) -> Optional[CalibrationProfile]:
        path = self.path_for(profile_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return CalibrationProfile.from_dict(json.load(handle))
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.error("Could not load calibration %s: %s", path.name, exc)
            return None

    def list_profiles(self) -> List[CalibrationProfile]:
        profiles = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    profiles.append(CalibrationProfile.from_dict(json.load(handle)))
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping malformed calibration %s: %s", path.name, exc)
        profiles.sort(key=lambda p: p.created_at, reverse=True)
        return profiles

    def delete(self, profile_id: str) -> bool:
        path = self.path_for(profile_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def latest(self) -> Optional[CalibrationProfile]:
        profiles = self.list_profiles()
        return profiles[0] if profiles else None


# ---------------------------------------------------------------- collection
@dataclass
class CalibrationSample:
    point_id: int
    target: Tuple[float, float]
    features: np.ndarray


def gaze_feature_columns(feature_names: Sequence[str]) -> List[int]:
    """Indices of the features that describe the EYE rather than the head.

    Outlier rejection must look only at these. Calibration deliberately samples
    a range of head positions, so head features vary by design; testing them
    for "outliers" would throw away exactly the data that teaches the model to
    compensate for head movement.
    """
    return [index for index, name in enumerate(feature_names)
            if name.startswith("iris") or name.startswith("ear")]


def robust_filter(samples: np.ndarray, threshold: float = MAD_THRESHOLD,
                  gross_threshold: float = GROSS_OUTLIER_Z,
                  columns: Optional[Sequence[int]] = None) -> np.ndarray:
    """Return a boolean mask of inlier samples, using two-pass sigma clipping.

    A single MAD test is tempting but behaves badly here: with only ~20 samples
    per calibration target the MAD itself is a noisy scale estimate, and when it
    happens to come out small the test rejects perfectly good frames. So:

    1. A *very* loose MAD pass removes gross outliers -- the frames where the
       user blinked or glanced away, which sit orders of magnitude out.
    2. The scale is then re-estimated as an ordinary standard deviation on the
       survivors, which is far better behaved at small n, and the real
       ``threshold`` test is applied against that.

    A sample is rejected if *any* feature fails, since one bad landmark
    contaminates the whole vector.
    """
    count = samples.shape[0]
    if count < 4:
        return np.ones(count, dtype=bool)
    if columns is not None:
        columns = list(columns)
        samples = samples[:, columns] if columns else samples

    median = np.median(samples, axis=0)
    deviation = np.abs(samples - median)
    mad = np.median(deviation, axis=0)
    # 0.6745 converts a MAD into a standard-deviation equivalent for normal data.
    safe_mad = np.where(mad > 1e-9, mad, np.inf)
    gross = np.any(0.6745 * deviation / safe_mad > gross_threshold, axis=1)
    survivors = ~gross
    if survivors.sum() < 4:
        return survivors

    mean = samples[survivors].mean(axis=0)
    std = samples[survivors].std(axis=0)
    safe_std = np.where(std > 1e-9, std, np.inf)
    fine = np.any(np.abs(samples - mean) / safe_std > threshold, axis=1)
    return survivors & ~fine


class CalibrationSession:
    """Accumulates per-target samples and fits a :class:`RidgeGazeEstimator`."""

    def __init__(
        self,
        feature_names: Sequence[str],
        screen_size: Tuple[int, int],
        screen_origin: Tuple[int, int] = (0, 0),
        degree: int = 2,
        alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
        max_yaw: float = 40.0,
        max_pitch: float = 30.0,
        min_openness: float = 0.16,
    ) -> None:
        self.feature_names = list(feature_names)
        self.screen_size = screen_size
        self.screen_origin = screen_origin
        self.degree = degree
        self.alphas = list(alphas)
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.min_openness = min_openness
        self.samples: List[CalibrationSample] = []
        self.rejected = 0

    # -------------------------------------------------------------- sampling
    def accepts(self, features: FeatureVector) -> bool:
        """Frame-level validity gate applied before a sample is stored."""
        if not features.valid:
            return False
        if features.head_pose is not None and features.head_pose.valid:
            if features.head_pose.is_extreme(self.max_yaw, self.max_pitch):
                return False
        if features.eye_openness < self.min_openness:
            return False
        array = features.to_array(self.feature_names)
        return bool(np.all(np.isfinite(array)))

    def add(self, point_id: int, target: Tuple[float, float],
            features: FeatureVector) -> bool:
        """Store one sample. Returns ``False`` if the frame was rejected."""
        if not self.accepts(features):
            self.rejected += 1
            return False
        self.samples.append(
            CalibrationSample(point_id, target, features.to_array(self.feature_names))
        )
        return True

    def count_for_point(self, point_id: int) -> int:
        return sum(1 for s in self.samples if s.point_id == point_id)

    def clear_point(self, point_id: int) -> None:
        self.samples = [s for s in self.samples if s.point_id != point_id]

    # ------------------------------------------------------------------- fit
    def build_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply outlier filtering and return ``(features, targets, groups)``."""
        by_point: Dict[int, List[CalibrationSample]] = {}
        for sample in self.samples:
            by_point.setdefault(sample.point_id, []).append(sample)

        feature_rows: List[np.ndarray] = []
        target_rows: List[Tuple[float, float]] = []
        group_rows: List[int] = []

        for point_id, point_samples in sorted(by_point.items()):
            matrix = np.array([s.features for s in point_samples], dtype=np.float64)
            mask = robust_filter(matrix, columns=gaze_feature_columns(self.feature_names))
            kept = matrix[mask]
            if kept.shape[0] < MIN_SAMPLES_PER_POINT:
                logger.warning("Dropping calibration point %d: only %d usable samples",
                               point_id, kept.shape[0])
                continue
            target = point_samples[0].target
            for row in kept:
                feature_rows.append(row)
                target_rows.append(target)
                group_rows.append(point_id)

        if not feature_rows:
            raise ValueError("No usable calibration samples were collected")
        return (np.array(feature_rows, dtype=np.float64),
                np.array(target_rows, dtype=np.float64),
                np.array(group_rows, dtype=np.int32))

    def fit(self) -> RidgeGazeEstimator:
        features, targets, groups = self.build_matrices()
        distinct = len(np.unique(groups))
        if distinct < MIN_POINTS_FOR_FIT:
            raise ValueError(
                f"Only {distinct} calibration points produced usable data; "
                f"at least {MIN_POINTS_FOR_FIT} are required. Try again with "
                f"better lighting and less head movement."
            )
        return RidgeGazeEstimator.fit(
            features, targets, groups,
            feature_names=self.feature_names,
            screen_size=self.screen_size,
            screen_origin=self.screen_origin,
            degree=self.degree,
            alphas=self.alphas,
        )

    def to_profile(self, estimator: RidgeGazeEstimator, monitor_index: int,
                   camera_index: int, pattern: str,
                   name: Optional[str] = None) -> CalibrationProfile:
        now = datetime.now()
        profile_id = f"cal_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        return CalibrationProfile(
            profile_id=profile_id,
            name=name or f"Calibration {now.strftime('%Y-%m-%d %H:%M')}",
            created_at=now.isoformat(timespec="seconds"),
            monitor_index=monitor_index,
            screen_width=self.screen_size[0],
            screen_height=self.screen_size[1],
            screen_origin=self.screen_origin,
            feature_names=self.feature_names,
            estimator_data=estimator.to_dict(),
            report=estimator.report,
            camera_index=camera_index,
            pattern=pattern,
        )
