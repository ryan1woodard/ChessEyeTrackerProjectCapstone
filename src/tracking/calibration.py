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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.config import PROJECT_ROOT
from .blink import BlinkDetector
from .features import FeatureVector
from .gaze_estimator import FitReport, RidgeGazeEstimator

logger = logging.getLogger(__name__)

CALIBRATION_DIR = PROJECT_ROOT / "data" / "calibrations"

MIN_SAMPLES_PER_POINT = 4
MIN_POINTS_FOR_FIT = 5
MAD_THRESHOLD = 3.5
GROSS_OUTLIER_Z = 12.0

@dataclass(frozen=True)
class Posture:
    """A posture asked of the user at one calibration target.

    ``roll_deg`` is a *target head tilt* in the same convention as
    :class:`~src.tracking.head_pose.HeadPose`: positive tips the image-right
    side of the face downwards, which is what happens when the user tips their
    head towards their own left shoulder.

    Holding the target as a number rather than as a sentence is what lets the
    calibration screen show a live gauge of the user's actual tilt against it.
    A sentence can be misread, and in testing it was: read at a glance, "tilt
    left" is ambiguous about whose left, and there was nothing to correct
    against. A needle the user moves into a zone cannot be misread, because
    tilting the wrong way visibly moves it the wrong way.
    """

    label: str
    #: Target head tilt in degrees, or 0.0 for the postures that are not tilts.
    roll_deg: float = 0.0
    #: What to do other than tilt: -1 nearer the screen, +1 further back.
    lean: float = 0.0
    #: Sideways shift: -1 to the user's left, +1 to their right.
    shift: float = 0.0

    #: How far "lean" moves the user, as a fraction of their seated distance.
    #: About 7 cm at a 60 cm working distance -- enough for the model to learn
    #: from, small enough to be comfortable to hold for a few seconds.
    LEAN_FRACTION = 0.12

    #: How far "shift" moves the user sideways, in interocular widths. One
    #: interocular width is about 63 mm.
    SHIFT_UNITS = 0.75

    @property
    def is_tilt(self) -> bool:
        return abs(self.roll_deg) > 1.0

    @property
    def is_neutral(self) -> bool:
        return not (self.is_tilt or self.lean or self.shift)

    def matches(self, roll_deg: float, tolerance_deg: float) -> bool:
        """Whether a measured head tilt counts as holding this posture."""
        if not self.is_tilt:
            return True
        return abs(roll_deg - self.roll_deg) <= tolerance_deg

    @property
    def destination(self) -> str:
        """A word for the far end of the travel gauge."""
        if self.lean:
            return "further back" if self.lean > 0 else "closer in"
        if self.shift:
            return "your right" if self.shift > 0 else "your left"
        return "here"

    def target_distance(self, baseline_z: float) -> float:
        """The ``head_z`` this posture is asking for, given the user's normal one."""
        return baseline_z * (1.0 + self.LEAN_FRACTION * self.lean)

    def target_offset(self, baseline_x: float) -> float:
        """The ``head_x`` this posture is asking for.

        ``shift`` is signed in the user's own terms -- negative is their left --
        and ``head_x`` is measured in the camera's, where the user's left is to
        the image right. Hence the sign flip, in one place rather than at every
        call site.
        """
        return baseline_x - self.SHIFT_UNITS * self.shift


#: Posture asked of the user at each calibration target, in the order the
#: targets are visited.
#:
#: Hoping the user drifts naturally is not enough. The fitted model is only
#: meaningful over the range of head positions calibration actually observed,
#: and outside it the estimate is clamped to the edge of that range -- so a
#: posture never sampled is a posture never compensated for. Someone who sits
#: rigidly still gives the model no way to tell a tilted head from an upright
#: one, and in the simulator that costs 238 px of error at a 20-degree tilt
#: against 22 px upright.
#:
#: Tilt appears the most often because it is the axis with the largest effect
#: and the one users are least likely to vary on their own. Cues alternate
#: direction so the samples straddle upright rather than sitting to one side.
#: The angles are small: 12 degrees is a glance at a clock on the wall, and 20
#: is the most anyone is asked for.
POSTURES: List[Posture] = [
    Posture("Sit as you normally would"),
    Posture("Tilt your head to the left", roll_deg=12.0),
    Posture("Tilt your head to the right", roll_deg=-12.0),
    Posture("Lean a little closer to the screen", lean=-1.0),
    Posture("Sit back a little", lean=1.0),
    Posture("Tilt further to the left", roll_deg=20.0),
    Posture("Sit as you normally would"),
    Posture("Tilt further to the right", roll_deg=-20.0),
    Posture("Shift a little to your left", shift=-1.0),
    Posture("Shift a little to your right", shift=1.0),
    Posture("Tilt your head to the left", roll_deg=12.0),
    Posture("Tilt your head to the right", roll_deg=-12.0),
    Posture("Sit as you normally would"),
]

#: Features whose range across the calibration samples is worth checking, with
#: the span (in raw feature units) below which coverage counts as too narrow.
#: ``eye_tilt`` and ``roll`` are in units of 30 degrees, so 0.3 is 9 degrees of
#: total spread -- a low bar that a user who tilted at all will clear.
COVERAGE_TARGETS: Dict[str, float] = {
    "eye_tilt": 0.30,
    "roll": 0.30,
    "head_x": 0.25,
    "head_z": 0.40,
}


def posture(point_index: int) -> Posture:
    """The posture to ask for at a given calibration target."""
    return POSTURES[point_index % len(POSTURES)]


def posture_cue(point_index: int) -> str:
    """The wording of the posture asked for at a given calibration target."""
    return posture(point_index).label


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
    #: Axes the calibration barely varied along, phrased for the user. Saved
    #: with the profile so the warning can be shown again later, rather than
    #: only in the dialog that appeared once when the fit finished.
    coverage_warnings: List[str] = field(default_factory=list)

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
            "coverage_warnings": list(self.coverage_warnings),
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
            coverage_warnings=list(data.get("coverage_warnings", [])),
        )

    def build_estimator(self) -> RidgeGazeEstimator:
        return RidgeGazeEstimator.from_dict(self.estimator_data)

    def matches_screen(self, width: int, height: int) -> bool:
        return self.screen_width == width and self.screen_height == height

    def summary(self) -> str:
        if self.report is None:
            return self.name
        error = self.report.settled_error_px or self.report.mean_error_px
        return f"{self.name} - {error:.0f} px ({self.report.quality()})"


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
    eye_prefixes = ("iris", "ray", "ear")
    return [index for index, name in enumerate(feature_names)
            if name.startswith(eye_prefixes)]


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
        margin_fraction: float = 0.15,
    ) -> None:
        self.feature_names = list(feature_names)
        self.screen_size = screen_size
        self.screen_origin = screen_origin
        self.degree = degree
        self.alphas = list(alphas)
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.min_openness = min_openness
        self.margin_fraction = margin_fraction
        #: Shared with the live pipeline so calibration and tracking agree on
        #: what "eyes closed" means for this user. A fixed threshold rejected
        #: every frame from anyone whose relaxed eyes read below it, which made
        #: calibration impossible rather than merely inaccurate.
        self.blink = BlinkDetector(min_openness)
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
        if self.blink.update(features.eye_openness):
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

    def coverage(self) -> Dict[str, float]:
        """Observed span of each coverage-relevant feature, as a fraction of
        the span :data:`COVERAGE_TARGETS` asks for.

        A value below 1.0 means calibration never saw enough of that axis for
        the model to compensate along it. Reported rather than enforced: a
        narrow calibration is still usable, it is just fragile in a way the
        user deserves to be told about.
        """
        if not self.samples:
            return {name: 0.0 for name in COVERAGE_TARGETS}
        matrix = np.array([s.features for s in self.samples], dtype=np.float64)
        spans = matrix.max(axis=0) - matrix.min(axis=0)
        result: Dict[str, float] = {}
        for name, wanted in COVERAGE_TARGETS.items():
            if name not in self.feature_names:
                continue
            observed = float(spans[self.feature_names.index(name)])
            result[name] = observed / wanted if wanted > 0 else 1.0
        return result

    def coverage_warnings(self) -> List[str]:
        """Human-readable notes about axes calibration barely varied along."""
        labels = {"eye_tilt": "head tilt", "roll": "head tilt",
                  "head_x": "side-to-side movement",
                  "head_z": "distance from the screen"}
        seen, warnings = set(), []
        for name, ratio in sorted(self.coverage().items()):
            label = labels.get(name, name)
            if ratio >= 1.0 or label in seen:
                continue
            seen.add(label)
            warnings.append(
                f"Very little {label} during calibration; tracking may drift "
                f"when you {'tilt your head' if label == 'head tilt' else 'move'}."
            )
        return warnings

    def fit(self) -> RidgeGazeEstimator:
        features, targets, groups = self.build_matrices()
        for warning in self.coverage_warnings():
            logger.warning("Calibration coverage: %s", warning)
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
            margin_fraction=self.margin_fraction,
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
            coverage_warnings=self.coverage_warnings(),
        )
