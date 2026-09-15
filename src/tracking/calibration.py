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

#: How much the head must move **at each individual dot**, as a span in raw
#: feature units. ``eye_tilt`` and ``roll`` are in units of 30 degrees, so 0.80
#: is 24 degrees of tilt swept while looking at one dot.
#:
#: Per dot, not pooled over the whole calibration, and that distinction is the
#: whole value of the check. Pooled, a calibration that holds a different fixed
#: posture at every dot looks excellent -- plenty of tilt overall -- while being
#: one of the worst kinds there is, because head position then tells the model
#: which dot was being looked at. Measured in the simulator:
#:
#:     scenario                 true error   per-dot span   pooled span
#:     head roams at every dot      25 px          1.22          1.28
#:     posture held, still drifts   23 px          0.88          2.21
#:     head roams lazily            39 px          0.77          0.82
#:     posture held rigidly         40 px          0.23          1.55
#:     head barely moves            68 px          0.38          0.42
#:
#: The per-dot column orders with the true error; the pooled one does not, and
#: rates the 40 px case above the 25 px one.
#:
#: Tilt alone cannot separate the 23 px and 39 px cases -- their per-dot spans
#: are 0.88 and 0.77, near enough that a threshold between them would flip on
#: noise. Distance and side-to-side separate them cleanly (2.33 against 1.35),
#: so the tilt target is set low enough to pass both and the other two carry
#: the decision. Any one axis failing is enough to warn.
COVERAGE_TARGETS: Dict[str, float] = {
    "eye_tilt": 0.70,
    "roll": 0.70,
    "head_x": 1.60,
    "head_z": 1.60,
}

#: Head features whose correlation with the target position is worth checking.
_CONFOUND_FEATURES = ("eye_tilt", "roll", "yaw", "pitch", "head_x", "head_y", "head_z")

#: Correlation above which head pose and screen position are treated as
#: confounded.
#:
#: If the head is always tilted the same way when looking at the top-left dot,
#: the fit cannot tell "tilted" from "looking top-left", and will happily
#: explain screen position using tilt -- which then falls apart the moment the
#: user tilts while looking somewhere else. Measured in the simulator: a
#: calibration where the head roams freely at every point scores 0.06, one
#: where a posture is held rigidly per point scores 0.33 and costs nearly
#: double the error.
#:
#: The threshold is set well above that 0.33 because the two cases overlap:
#: holding a posture but still drifting naturally also scores around 0.28 and
#: is perfectly good. This flags calibrations that are unambiguously
#: confounded, not every one that leans that way.
MAX_TARGET_CORRELATION = 0.45


#: The two ways of collecting a calibration.
#:
#: ``postures``
#:     One prompted posture per dot, held while the samples are taken. Quick,
#:     and accurate when it is followed -- but it depends on the user actually
#:     following it, and it is fragile in a specific way: if the head is always
#:     in the same place for a given dot, head position and screen position
#:     become interchangeable to the fit. Measured in the simulator, held
#:     rigidly it costs 42 px against 23 px when the head keeps drifting.
#:
#: ``explore``
#:     The head roams freely at every dot while the eyes stay on it. Every
#:     screen position then sees the whole range of head positions, so the two
#:     cannot be confused by construction -- the correlation between them falls
#:     from 0.33 to 0.06. It takes longer and asks the user to keep moving
#:     rather than to hit a particular pose, which is a much easier thing to
#:     do right.
METHODS: Tuple[str, ...] = ("explore", "postures")
DEFAULT_METHOD = "explore"

#: Head tilt, in degrees either side of upright, that the explore method tries
#: to see at every dot. Comfortable to reach without stretching.
EXPLORE_ROLL_RANGE = 18.0

#: How many bins that range is divided into for the coverage ring.
EXPLORE_ROLL_BINS = 12

#: Fraction of the bins that counts as having covered the range. Not all of
#: them: the extremes need a deliberate stretch, and the point of the ring is
#: to keep the user moving, not to hold them at a dot until they hit every bin.
EXPLORE_COVERAGE_TARGET = 0.75


class PoseCoverage:
    """Which head tilts have been seen while looking at the current dot.

    The explore method asks the user to keep moving, which is a vague
    instruction with no way of telling whether you have done enough of it.
    Binning the tilts actually observed turns it into a concrete goal, and one
    that can be drawn as a ring round the dot -- so it is answered without ever
    looking away from the dot to read anything.
    """

    def __init__(self, bins: int = EXPLORE_ROLL_BINS,
                 roll_range: float = EXPLORE_ROLL_RANGE) -> None:
        self.bins = max(int(bins), 2)
        self.roll_range = float(roll_range)
        self.seen = [False] * self.bins

    def bin_for(self, roll_deg: float) -> Optional[int]:
        """Which bin a tilt falls in, or ``None`` if it is outside the range."""
        if not np.isfinite(roll_deg) or abs(roll_deg) > self.roll_range:
            return None
        position = (roll_deg + self.roll_range) / (2 * self.roll_range)
        return min(int(position * self.bins), self.bins - 1)

    def observe(self, roll_deg: float) -> None:
        index = self.bin_for(roll_deg)
        if index is not None:
            self.seen[index] = True

    @property
    def fraction(self) -> float:
        return sum(self.seen) / self.bins

    @property
    def complete(self) -> bool:
        return self.fraction >= EXPLORE_COVERAGE_TARGET

    def reset(self) -> None:
        self.seen = [False] * self.bins


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


@dataclass
class CalibrationDiagnosis:
    """Why a calibration came out the way it did.

    Exists because "POOR" on its own is not actionable. Every field here is
    something the user can do something about: collect more samples, move their
    head more, move it less predictably, or fix the lighting in one corner of
    the screen.
    """

    points_kept: int
    points_attempted: int
    samples_kept: int
    samples_rejected: int
    #: Observed span of each coverage-relevant feature, as a fraction of the
    #: span wanted. Below 1.0 means the model never saw enough of that axis.
    coverage: Dict[str, float] = field(default_factory=dict)
    #: Strongest correlation between a head feature and target position.
    target_correlation: float = 0.0
    target_correlation_detail: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def samples_per_point(self) -> float:
        return self.samples_kept / self.points_kept if self.points_kept else 0.0


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
        """How much the head moved at each dot, against what is wanted.

        For each feature, the *median across dots* of that dot's own span,
        divided by the target in :data:`COVERAGE_TARGETS`. Below 1.0 means the
        head was too still while each dot was on screen, and the model has no
        way to tell head movement from eye movement.

        The median rather than the mean, so one dot where the user fidgeted
        cannot cover for twelve where they did not.

        Reported rather than enforced: a narrow calibration is still usable, it
        is just fragile in a way the user deserves to be told about.
        """
        if not self.samples:
            return {name: 0.0 for name in COVERAGE_TARGETS
                    if name in self.feature_names}

        by_point: Dict[int, List[np.ndarray]] = {}
        for sample in self.samples:
            by_point.setdefault(sample.point_id, []).append(sample.features)

        result: Dict[str, float] = {}
        for name, wanted in COVERAGE_TARGETS.items():
            if name not in self.feature_names or wanted <= 0:
                continue
            index = self.feature_names.index(name)
            spans = [float(np.ptp([row[index] for row in rows]))
                     for rows in by_point.values() if len(rows) > 1]
            result[name] = float(np.median(spans)) / wanted if spans else 0.0
        return result

    def target_correlation(self) -> Tuple[float, str]:
        """Strongest correlation between a head feature and target position.

        A calibration where the head is in a characteristic place for each dot
        teaches the model to read the dot off the head, which works beautifully
        on the calibration data and collapses in use.
        """
        if len(self.samples) < 8:
            return 0.0, ""
        matrix = np.array([s.features for s in self.samples], dtype=np.float64)
        targets = np.array([s.target for s in self.samples], dtype=np.float64)
        worst, detail = 0.0, ""
        for name in _CONFOUND_FEATURES:
            if name not in self.feature_names:
                continue
            column = matrix[:, self.feature_names.index(name)]
            if column.std() < 1e-9:
                continue
            for axis, label in ((0, "horizontally"), (1, "vertically")):
                if targets[:, axis].std() < 1e-9:
                    continue
                value = abs(float(np.corrcoef(column, targets[:, axis])[0, 1]))
                if value > worst:
                    worst, detail = value, f"{name} tracked where the dot was {label}"
        return worst, detail

    def diagnose(self) -> CalibrationDiagnosis:
        """Everything known about why this calibration is as good as it is."""
        attempted = len({s.point_id for s in self.samples})
        try:
            features, _targets, groups = self.build_matrices()
            kept_points = int(len(np.unique(groups)))
            kept_samples = int(features.shape[0])
        except ValueError:
            kept_points, kept_samples = 0, 0

        correlation, detail = self.target_correlation()
        diagnosis = CalibrationDiagnosis(
            points_kept=kept_points,
            points_attempted=attempted,
            samples_kept=kept_samples,
            samples_rejected=self.rejected + max(0, len(self.samples) - kept_samples),
            coverage=self.coverage(),
            target_correlation=correlation,
            target_correlation_detail=detail,
        )

        labels = {"eye_tilt": "head tilt", "roll": "head tilt",
                  "head_x": "side-to-side movement",
                  "head_z": "movement towards and away from the screen"}
        seen = set()
        for name, ratio in sorted(diagnosis.coverage.items()):
            label = labels.get(name, name)
            if ratio >= 1.0 or label in seen:
                continue
            seen.add(label)
            # Graded, because 96% of the wanted range and 28% of it are not
            # the same problem and should not read as the same sentence.
            severity = ("barely moved" if ratio < 0.6 else
                        "did not move much" if ratio < 0.85 else
                        "moved a little less than is ideal")
            diagnosis.warnings.append(
                f"Your head {severity} while each dot was on screen: {label} "
                f"covered {ratio * 100:.0f}% of the range wanted. The tracker "
                f"cannot tell head movement from eye movement unless it sees "
                f"both at the same dot, so it will drift when you move."
            )
        if kept_points < attempted:
            diagnosis.warnings.append(
                f"{attempted - kept_points} of {attempted} points had too few usable "
                f"samples and were dropped. Check your lighting, and that your face "
                f"stays in frame when you look at the edges of the screen."
            )
        if kept_points and diagnosis.samples_per_point < MIN_SAMPLES_PER_POINT * 2:
            diagnosis.warnings.append(
                f"Only {diagnosis.samples_per_point:.0f} samples survived per point. "
                f"More time on each dot, or better lighting, would help."
            )
        if correlation > MAX_TARGET_CORRELATION:
            diagnosis.warnings.append(
                f"Your head position gave away which dot you were looking at "
                f"({detail}). Move your head more freely at every dot, so that "
                f"head position and screen position are not tied together."
            )
        return diagnosis

    def coverage_warnings(self) -> List[str]:
        """Human-readable notes about what went wrong, worst first."""
        return self.diagnose().warnings

    def fit(self) -> RidgeGazeEstimator:
        features, targets, groups = self.build_matrices()
        for warning in self.coverage_warnings():
            logger.warning("Calibration coverage: %s", warning)
        distinct = len(np.unique(groups))
        if distinct < MIN_POINTS_FOR_FIT:
            raise ValueError(
                f"Only {distinct} calibration points produced usable data; "
                f"at least {MIN_POINTS_FOR_FIT} are required. Better lighting, "
                f"and keeping your face in frame when you look at the edges of "
                f"the screen, are what usually fix this."
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
            coverage_warnings=self.diagnose().warnings,
        )
