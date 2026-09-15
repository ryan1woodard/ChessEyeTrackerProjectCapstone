"""Mapping from facial features to screen coordinates.

Design notes
------------
The mapping is a *per-user, per-monitor* regression fitted during calibration.
There is no universal mapping: eye shape, glasses, webcam placement and seating
distance all change the relationship between iris position and screen position.

The model is deliberately small:

    features -> standardise -> polynomial expansion (deg 2) -> ridge -> (x, y)

A 13-point calibration provides only 13 genuinely independent observations, so
an unregularised fit would interpolate the calibration points perfectly and
generalise terribly. Two things prevent that:

* Ridge regularisation, with the strength ``alpha`` chosen automatically.
* **Leave-one-point-out** cross-validation for that choice. Because all samples
  from one calibration target are held out together, the resulting error is an
  estimate of accuracy at *unseen* screen locations, which is what the user
  actually cares about. Training error would look impressively small and mean
  nothing.

The held-out frames are scored individually rather than averaged first. The
average measures only the model's bias at that target and cancels the
frame-to-frame noise regularisation exists to control, which lets an
under-regularised model score well and then wobble badly in use. Both numbers
are reported: ``mean_error_px`` for a single frame, ``settled_error_px`` for a
fixation the smoothing filter has settled on.

``GazeEstimator`` is an abstract interface so the webcam estimator can later be
swapped for a hardware eye tracker without touching the rest of the pipeline.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.geometry import clamp
from .features import NOMINAL_SCALES, FeatureVector, nominal_scales

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GazeResult:
    """An estimated gaze position on the virtual desktop, in pixels."""

    x: float
    y: float
    confidence: float
    valid: bool
    #: True when the unclamped prediction fell outside the screen. Reported
    #: separately so "looking away" stays distinguishable from "clamped".
    out_of_bounds: bool = False

    @classmethod
    def invalid(cls, confidence: float = 0.0) -> "GazeResult":
        return cls(0.0, 0.0, confidence, False)


class GazeEstimator(ABC):
    """Interface for anything that turns features into a screen position."""

    @abstractmethod
    def estimate(self, features: FeatureVector) -> GazeResult:
        """Return the estimated gaze position for one frame."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the estimator has enough information to produce output."""


# --------------------------------------------------------------------- maths
def polynomial_expand(x: np.ndarray, degree: int) -> np.ndarray:
    """Expand ``(n_samples, n_features)`` into polynomial terms with a bias column.

    For ``degree=2`` the output is ``[1, x_i, x_i * x_j for i <= j]``.
    """
    if x.ndim != 2:
        raise ValueError("polynomial_expand expects a 2-D array")
    n_samples, n_features = x.shape
    columns: List[np.ndarray] = [np.ones(n_samples, dtype=np.float64)]
    for d in range(1, max(degree, 1) + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            term = np.ones(n_samples, dtype=np.float64)
            for index in combo:
                term = term * x[:, index]
            columns.append(term)
    return np.column_stack(columns)


def ridge_fit(design: np.ndarray, targets: np.ndarray, alpha: float,
              weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Solve ridge regression, leaving the bias (column 0) unpenalised.

    ``weights`` gives a per-sample weight; see :func:`robust_ridge_fit`.
    """
    n_terms = design.shape[1]
    penalty = np.eye(n_terms, dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    if weights is None:
        gram = design.T @ design + penalty
        rhs = design.T @ targets
    else:
        weighted = design * weights[:, None]
        gram = weighted.T @ design + penalty
        rhs = weighted.T @ targets
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:  # pragma: no cover - singular design
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


#: Residual, in units of the robust spread, beyond which a calibration sample
#: stops counting in full. The classic Huber value.
HUBER_K = 1.345

#: Reweighting passes. The weights settle in two or three; more is wasted work.
ROBUST_PASSES = 3


def robust_ridge_fit(design: np.ndarray, targets: np.ndarray, alpha: float,
                     passes: int = ROBUST_PASSES) -> np.ndarray:
    """Ridge regression that a handful of bad frames cannot drag off course.

    Least squares weights a sample by the square of its error, so one frame
    where the user glanced away during collection pulls the fit further than
    fifty good ones hold it. The outlier filter in
    :mod:`~src.tracking.calibration` catches the frames that look wrong in
    *feature* space -- a blink, a lost iris -- but it cannot see the ones that
    look perfectly normal and are simply aimed somewhere else, because nothing
    about those features is unusual. Only the fit residual reveals them.

    So the fit is repeated, each pass weighting samples by how well the
    previous one predicted them: in full inside a Huber radius of the robust
    spread of residuals, and falling off as 1/residual beyond it. A glance away
    ends up contributing a fraction of a sample instead of dominating its
    target.
    """
    weights = None
    coefficients = ridge_fit(design, targets, alpha)
    for _ in range(max(passes, 1)):
        residuals = np.hypot(*(design @ coefficients - targets).T)
        # Median absolute deviation, scaled to a standard deviation for normal
        # data. Using the plain standard deviation here would be self-defeating:
        # the outliers this exists to find would inflate it and then look normal.
        median = float(np.median(residuals))
        spread = float(np.median(np.abs(residuals - median))) / 0.6745
        if spread < 1e-9:
            break
        scaled = residuals / (HUBER_K * spread)
        weights = np.where(scaled <= 1.0, 1.0, 1.0 / np.maximum(scaled, 1e-9))
        coefficients = ridge_fit(design, targets, alpha, weights)
    return coefficients


@dataclass(frozen=True)
class _AlphaScore:
    """One regularisation strength and how it scored in cross-validation."""

    alpha: float
    #: Mean error over individual held-out frames; what ``alpha`` is chosen on.
    frame_error: float
    standard_error: float
    #: Per-target centroid errors; what a settled fixation is worth.
    point_errors: List[float] = field(default_factory=list)


@dataclass
class FitReport:
    """Accuracy summary produced when a calibration model is fitted."""

    mean_error_px: float
    median_error_px: float
    max_error_px: float
    mean_error_normalised: float
    per_point_error_px: List[float] = field(default_factory=list)
    #: Cross-validated error of the *average* of the held-out frames at each
    #: target: the accuracy of a steady fixation once smoothing has settled.
    #: ``mean_error_px`` is the harsher single-frame figure.
    settled_error_px: float = 0.0
    #: Error at head positions the model was never fitted on.
    #:
    #: The other two figures hold out screen *targets*, which says nothing
    #: about head pose: a model that has quietly learned to read the target off
    #: the head still predicts a held-out target correctly, because the poses
    #: it saw at that target are in the training data for every other one. This
    #: holds out a *range of head poses* instead, and is the figure that tracks
    #: what the user experiences when they move. Zero when there was too little
    #: pose variation to split on -- which the coverage warnings cover instead.
    pose_error_px: float = 0.0
    train_mean_error_px: float = 0.0
    alpha: float = 0.0
    n_samples: int = 0
    n_points: int = 0

    def quality(self, good_px: float = 0.0, fair_px: float = 0.0) -> str:
        """Rate the calibration, scaled to the size of the screen.

        Absolute pixel thresholds are misleading across different monitors:
        100 px of error is respectable on a 4K display and poor on a small
        laptop. These default to fractions of the screen diagonal, which is
        roughly proportional to the visual angle that actually matters.
        A well-set-up webcam achieves about 1-3 degrees, which is around
        4-8% of a typical diagonal.
        """
        good = good_px if good_px > 0 else self.diagonal_px * 0.055
        fair = fair_px if fair_px > 0 else self.diagonal_px * 0.095
        # Rated on the settled error, because that is what someone holding
        # their gaze on a square experiences; the single-frame figure is
        # reported alongside it but is dominated by landmark noise the
        # smoothing filter removes.
        # Rated on whichever honest figure is worse. A model can be steady at
        # the pose it was calibrated in and fall apart a head-turn away, and
        # rating it on the first number alone would call that GOOD.
        score = self.settled_error_px or self.mean_error_px
        if self.pose_error_px > 0:
            score = max(score, self.pose_error_px)
        if score <= good:
            return "GOOD"
        if score <= fair:
            return "FAIR"
        return "POOR"

    @property
    def diagonal_px(self) -> float:
        """Screen diagonal implied by the recorded normalised error."""
        if self.mean_error_normalised > 1e-9:
            return self.mean_error_px / self.mean_error_normalised
        return 2202.0  # 1920x1080 fallback

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_error_px": self.mean_error_px,
            "median_error_px": self.median_error_px,
            "max_error_px": self.max_error_px,
            "mean_error_normalised": self.mean_error_normalised,
            "per_point_error_px": list(self.per_point_error_px),
            "settled_error_px": self.settled_error_px,
            "pose_error_px": self.pose_error_px,
            "train_mean_error_px": self.train_mean_error_px,
            "alpha": self.alpha,
            "n_samples": self.n_samples,
            "n_points": self.n_points,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FitReport":
        return cls(
            mean_error_px=float(data.get("mean_error_px", 0.0)),
            median_error_px=float(data.get("median_error_px", 0.0)),
            max_error_px=float(data.get("max_error_px", 0.0)),
            mean_error_normalised=float(data.get("mean_error_normalised", 0.0)),
            per_point_error_px=list(data.get("per_point_error_px", [])),
            settled_error_px=float(data.get("settled_error_px", 0.0)),
            pose_error_px=float(data.get("pose_error_px", 0.0)),
            train_mean_error_px=float(data.get("train_mean_error_px", 0.0)),
            alpha=float(data.get("alpha", 0.0)),
            n_samples=int(data.get("n_samples", 0)),
            n_points=int(data.get("n_points", 0)),
        )


class RidgeGazeEstimator(GazeEstimator):
    """Polynomial ridge mapping from features to screen pixels."""

    def __init__(
        self,
        feature_names: Sequence[str],
        degree: int = 2,
        mean: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        screen_size: Tuple[int, int] = (1920, 1080),
        screen_origin: Tuple[int, int] = (0, 0),
        report: Optional[FitReport] = None,
        margin_fraction: float = 0.15,
        input_low: Optional[np.ndarray] = None,
        input_high: Optional[np.ndarray] = None,
    ) -> None:
        self.feature_names = list(feature_names)
        self.degree = int(degree)
        self.mean = mean
        self.scale = scale
        self.weights = weights  # shape (n_terms, 2)
        self.screen_size = screen_size
        self.screen_origin = screen_origin
        self.report = report
        self.margin_fraction = margin_fraction
        #: Range of each feature seen during calibration, widened by a margin.
        self.input_low = input_low
        self.input_high = input_high

    # ------------------------------------------------------------- inference
    @property
    def is_ready(self) -> bool:
        return self.weights is not None and self.mean is not None and self.scale is not None

    def _design(self, raw: np.ndarray) -> np.ndarray:
        raw = self.clamp_inputs(raw)
        standardised = (raw - self.mean) / self.scale
        return polynomial_expand(standardised, self.degree)

    def clamp_inputs(self, raw: np.ndarray) -> np.ndarray:
        """Hold features inside the range calibration actually covered.

        A polynomial is only meaningful where it was fitted. If the user moves
        their head well beyond anything seen during calibration, the quadratic
        terms grow without limit and the estimate runs away -- which is felt as
        error that grows the further you drift and never comes back.

        Saturating at the edge of the calibrated range converts that runaway
        into a bounded, roughly constant bias: still wrong, but stable and
        recoverable, and the confidence score reflects it.
        """
        if self.input_low is None or self.input_high is None:
            return raw
        return np.clip(raw, self.input_low, self.input_high)

    def predict_array(self, raw: np.ndarray) -> np.ndarray:
        """Predict screen coordinates for ``(n_samples, n_features)`` raw input."""
        if not self.is_ready:
            raise RuntimeError("Estimator is not calibrated")
        return self._design(np.atleast_2d(raw)) @ self.weights

    def estimate(self, features: FeatureVector) -> GazeResult:
        if not self.is_ready or not features.valid:
            return GazeResult.invalid()
        raw = features.to_array(self.feature_names).reshape(1, -1)
        if not np.all(np.isfinite(raw)):
            return GazeResult.invalid()
        prediction = self.predict_array(raw)[0]
        x, y = float(prediction[0]), float(prediction[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return GazeResult.invalid()

        # A polynomial model extrapolates without limit, so one odd frame can
        # produce a coordinate in the millions. Clamp to the screen plus a
        # margin: beyond that margin the estimate carries no information
        # anyway, and an unbounded value would wreck the smoothing filter for
        # several seconds afterwards.
        clamped_x, clamped_y, outside = self.clamp_to_screen(x, y)
        return GazeResult(x=clamped_x, y=clamped_y, confidence=1.0, valid=True,
                          out_of_bounds=outside)

    def clamp_to_screen(self, x: float, y: float) -> tuple[float, float, bool]:
        """Constrain a prediction to the screen plus ``margin_fraction``."""
        ox, oy = self.screen_origin
        width, height = self.screen_size
        margin_x = width * self.margin_fraction
        margin_y = height * self.margin_fraction
        low_x, high_x = ox - margin_x, ox + width + margin_x
        low_y, high_y = oy - margin_y, oy + height + margin_y
        outside = not (low_x <= x <= high_x and low_y <= y <= high_y)
        return clamp(x, low_x, high_x), clamp(y, low_y, high_y), outside

    def is_on_screen(self, x: float, y: float, margin: float = 0.0) -> bool:
        ox, oy = self.screen_origin
        width, height = self.screen_size
        return (ox - margin) <= x <= (ox + width + margin) and \
               (oy - margin) <= y <= (oy + height + margin)

    # ------------------------------------------------------------------- fit
    @staticmethod
    def input_range(raw_features: np.ndarray,
                    feature_names: Sequence[str],
                    widen: float = 0.25,
                    floor_fraction: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
        """The feature box :meth:`clamp_inputs` saturates against.

        Derived from the calibration data, then widened, then **floored at a
        fraction of each feature's nominal range**. The floor is what makes
        this safe: a user who never tilted their head during calibration
        produces an ``eye_tilt`` span of almost nothing, and a box drawn from
        that span alone can collapse to a point, pinning the feature and
        throwing away whatever compensation it was there to provide.

        Both constants are small on purpose. Sweeping them in the simulator,
        every value of ``widen`` above 0.25 made a calibration that never
        covered tilt *worse* -- letting the polynomial extrapolate turns a
        bounded bias into a runaway -- and every value of ``floor_fraction``
        above 0.25 did the same. Neither constant changes the result at all
        once calibration actually covers the range, which is what the posture
        prompts exist to bring about. Generous extrapolation is not a
        substitute for data.
        """
        low = raw_features.min(axis=0)
        high = raw_features.max(axis=0)
        span = np.maximum(high - low, 1e-6)
        low, high = low - span * widen, high + span * widen

        floor = nominal_scales(feature_names) * floor_fraction
        centre = 0.5 * (low + high)
        half = np.maximum(0.5 * (high - low), floor)
        return centre - half, centre + half

    #: Head-pose features to split on for the pose holdout, best first.
    _POSE_SPLIT_FEATURES = ("yaw", "eye_tilt", "roll", "pitch", "head_x", "head_z")

    #: How many pose groups to split into. Four leaves each fold with three
    #: quarters of the poses to learn from, which is enough to fit, while the
    #: held-out quarter is genuinely outside what it saw.
    POSE_FOLDS = 4

    #: Below this spread, there were not enough distinct head poses to split
    #: on and the holdout is not reported at all. Measured in the standardised
    #: feature units the design matrix uses.
    MIN_POSE_SPREAD = 0.25

    @classmethod
    def _pose_holdout_error(cls, design: np.ndarray, targets: np.ndarray,
                            feature_names: Sequence[str], raw: np.ndarray,
                            alpha: float) -> float:
        """Error at head poses the model never saw, or 0.0 if unmeasurable.

        Samples are sorted by whichever pose feature varied most and split into
        contiguous blocks, so a held-out block is a *range* of head positions
        rather than a scatter of them -- holding out every fourth frame would
        leave near-identical poses in the training set and measure nothing.
        """
        names = [n for n in cls._POSE_SPLIT_FEATURES if n in feature_names]
        if not names or raw.shape[0] < cls.POSE_FOLDS * 4:
            return 0.0

        spreads = {n: float(np.std(raw[:, list(feature_names).index(n)]))
                   / max(NOMINAL_SCALES.get(n, 1.0), 1e-9) for n in names}
        axis, spread = max(spreads.items(), key=lambda item: item[1])
        if spread < cls.MIN_POSE_SPREAD:
            return 0.0

        order = np.argsort(raw[:, list(feature_names).index(axis)])
        folds = np.array_split(order, cls.POSE_FOLDS)
        errors: List[float] = []
        for fold in folds:
            held = np.zeros(raw.shape[0], dtype=bool)
            held[fold] = True
            if held.all() or not held.any():
                continue
            weights = robust_ridge_fit(design[~held], targets[~held], alpha)
            predicted = design[held] @ weights
            errors.extend(np.hypot(*(predicted - targets[held]).T).tolist())
        return float(np.mean(errors)) if errors else 0.0

    @classmethod
    def fit(
        cls,
        raw_features: np.ndarray,
        targets: np.ndarray,
        groups: np.ndarray,
        feature_names: Sequence[str],
        screen_size: Tuple[int, int],
        screen_origin: Tuple[int, int] = (0, 0),
        degree: int = 2,
        alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
        margin_fraction: float = 0.15,
    ) -> "RidgeGazeEstimator":
        """Fit the mapping and cross-validate the regularisation strength.

        Parameters
        ----------
        raw_features:
            ``(n_samples, n_features)`` array of un-standardised features.
        targets:
            ``(n_samples, 2)`` array of true screen coordinates in pixels.
        groups:
            ``(n_samples,)`` calibration-point id used for leave-one-out CV.
        """
        raw_features = np.asarray(raw_features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        groups = np.asarray(groups)
        if raw_features.ndim != 2 or targets.shape[0] != raw_features.shape[0]:
            raise ValueError("raw_features and targets have mismatched shapes")

        unique_groups = np.unique(groups)
        if len(unique_groups) < 5:
            raise ValueError(
                f"At least 5 distinct calibration points are required, got {len(unique_groups)}"
            )

        # Centre on the calibration mean (this removes the user's personal
        # offset) but scale by FIXED nominal ranges rather than the observed
        # standard deviation. See features.NOMINAL_SCALES for why: scaling by
        # observed variance amplifies whichever features happened to stay still
        # during calibration, which is exactly the wrong thing to do.
        mean = raw_features.mean(axis=0)
        scale = nominal_scales(feature_names)

        standardised = (raw_features - mean) / scale
        design = polynomial_expand(standardised, degree)

        # --- choose the regularisation strength -----------------------
        # Leave-one-point-out: every sample from a target is held out together,
        # so the error estimates accuracy at screen positions the model has
        # never seen.
        per_alpha: List[_AlphaScore] = []
        for alpha in alphas:
            frame_errors: List[float] = []
            point_errors: List[float] = []
            for group in unique_groups:
                held_out = groups == group
                if held_out.all():
                    continue
                weights = robust_ridge_fit(design[~held_out], targets[~held_out], alpha)
                predicted = design[held_out] @ weights
                actual = targets[held_out]
                # Two different questions, both worth answering.
                #
                # The per-FRAME error is what a single webcam frame is worth,
                # and it is the one ``alpha`` is chosen against. Scoring the
                # average of the held-out frames instead -- as this did
                # originally -- measures only the model's bias at that target
                # and cancels exactly the frame-to-frame noise that
                # regularisation exists to control. Judged that way an
                # under-regularised model looks excellent, because its centroid
                # sits on the dot, while in use it amplifies landmark jitter
                # into tens of pixels of wobble.
                frame_errors.extend(np.hypot(*(predicted - actual).T).tolist())
                # The per-POINT error is the centroid error once the smoothing
                # filter has settled, i.e. what a steady fixation is worth. It
                # is reported, not optimised.
                point_errors.append(float(np.hypot(
                    *(predicted.mean(axis=0) - actual.mean(axis=0)))))
            if not frame_errors:
                continue
            mean_error = float(np.mean(frame_errors))
            standard_error = float(np.std(frame_errors) / np.sqrt(len(frame_errors)))
            per_alpha.append(_AlphaScore(alpha, mean_error, standard_error, point_errors))
            logger.debug("alpha=%.4g  LOPO frame error=%.1f px (+/- %.1f), "
                         "settled error=%.1f px",
                         alpha, mean_error, standard_error, float(np.mean(point_errors)))

        if not per_alpha:  # pragma: no cover - defensive
            raise ValueError("Cross-validation produced no usable folds")

        # The "one standard error" rule: rather than the alpha with the very
        # best score, take the STRONGEST regularisation whose error is still
        # within one standard error of the best. With only 13 points the
        # minimum is noisy, and the simpler model extrapolates far better
        # outside the calibrated region -- which is where a gaze tracker spends
        # most of its time.
        best = min(per_alpha, key=lambda score: score.frame_error)
        threshold = best.frame_error + best.standard_error
        for score in sorted(per_alpha, key=lambda s: -s.alpha):
            if score.frame_error <= threshold:
                best = score
                break

        pose_error = cls._pose_holdout_error(design, targets, feature_names,
                                             raw_features, best.alpha)

        input_low, input_high = cls.input_range(raw_features, feature_names)

        weights = robust_ridge_fit(design, targets, best.alpha)
        train_predictions = design @ weights
        train_error = float(np.mean(np.hypot(*(train_predictions - targets).T)))

        diagonal = float(np.hypot(*screen_size))
        point_errors = best.point_errors
        report = FitReport(
            mean_error_px=best.frame_error,
            median_error_px=float(np.median(point_errors)) if point_errors else 0.0,
            max_error_px=float(np.max(point_errors)) if point_errors else 0.0,
            mean_error_normalised=best.frame_error / diagonal if diagonal else 0.0,
            per_point_error_px=[float(e) for e in point_errors],
            settled_error_px=float(np.mean(point_errors)) if point_errors else 0.0,
            pose_error_px=pose_error,
            train_mean_error_px=train_error,
            alpha=float(best.alpha),
            n_samples=int(raw_features.shape[0]),
            n_points=int(len(unique_groups)),
        )
        logger.info(
            "Calibration fitted: %d points / %d samples, alpha=%.4g, "
            "LOPO frame error=%.1f px, settled %.1f px, unseen head poses %.1f px "
            "(train %.1f px)",
            report.n_points, report.n_samples, report.alpha,
            report.mean_error_px, report.settled_error_px, report.pose_error_px,
            report.train_mean_error_px,
        )
        return cls(feature_names, degree, mean, scale, weights,
                   screen_size, screen_origin, report,
                   margin_fraction=margin_fraction,
                   input_low=input_low, input_high=input_high)

    # --------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError("Cannot serialise an unfitted estimator")
        return {
            "type": "ridge_polynomial",
            "feature_names": self.feature_names,
            "degree": self.degree,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "screen_size": list(self.screen_size),
            "screen_origin": list(self.screen_origin),
            "margin_fraction": self.margin_fraction,
            "report": self.report.to_dict() if self.report else None,
            "input_low": self.input_low.tolist() if self.input_low is not None else None,
            "input_high": self.input_high.tolist() if self.input_high is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RidgeGazeEstimator":
        report = FitReport.from_dict(data["report"]) if data.get("report") else None
        return cls(
            feature_names=data["feature_names"],
            degree=int(data.get("degree", 2)),
            mean=np.array(data["mean"], dtype=np.float64),
            scale=np.array(data["scale"], dtype=np.float64),
            weights=np.array(data["weights"], dtype=np.float64),
            screen_size=tuple(data.get("screen_size", (1920, 1080))),
            screen_origin=tuple(data.get("screen_origin", (0, 0))),
            margin_fraction=float(data.get("margin_fraction", 0.15)),
            report=report,
            input_low=np.array(data["input_low"], dtype=np.float64)
            if data.get("input_low") is not None else None,
            input_high=np.array(data["input_high"], dtype=np.float64)
            if data.get("input_high") is not None else None,
        )
