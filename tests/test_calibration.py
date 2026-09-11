"""Calibration, gaze-model and smoothing tests.

All of these run on synthetic data -- no webcam, no MediaPipe, no Qt.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tracking.calibration import (CalibrationSession, calibration_pattern,
                                      pattern_to_pixels, robust_filter)
from src.tracking.features import FeatureVector
from src.tracking.gaze_estimator import (FitReport, RidgeGazeEstimator,
                                         polynomial_expand, ridge_fit)
from src.tracking.head_pose import HeadPose
from src.tracking.smoother import GazeSmoother, OneEuroFilter

SCREEN = (1920, 1080)
FEATURES = ["iris_l_x", "iris_l_y", "iris_r_x", "iris_r_y", "yaw", "pitch"]


def synthetic_features(x: float, y: float, noise: float = 0.0,
                       rng: np.random.Generator | None = None) -> FeatureVector:
    """Invent a plausible feature vector for a person looking at ``(x, y)``.

    The relationship is mildly non-linear so a purely linear model would not
    fit it perfectly, which is closer to reality than a straight line.
    """
    rng = rng or np.random.default_rng(0)
    nx = x / SCREEN[0] - 0.5
    ny = y / SCREEN[1] - 0.5
    jitter = (lambda: rng.normal(0.0, noise)) if noise else (lambda: 0.0)
    values = {
        "iris_l_x": 0.20 * nx + 0.02 * nx * nx + jitter(),
        "iris_l_y": 0.16 * ny - 0.01 * ny * ny + jitter(),
        "iris_r_x": 0.19 * nx + jitter(),
        "iris_r_y": 0.17 * ny + jitter(),
        "iris_mean_x": 0.195 * nx,
        "iris_mean_y": 0.165 * ny,
        "iris_vergence": 0.01 * nx,
        "yaw": 0.30 * nx + jitter(),
        "pitch": 0.25 * ny + jitter(),
        "roll": 0.0,
        "face_x": 0.01 * nx,
        "face_y": 0.01 * ny,
        "face_scale": 0.18,
        "ear_l": 0.30,
        "ear_r": 0.30,
    }
    return FeatureVector(values=values, valid=True,
                         head_pose=HeadPose(pitch=8 * ny, yaw=12 * nx, roll=0.0))


class TestPolynomialExpansion:
    def test_degree_one_is_bias_plus_features(self):
        design = polynomial_expand(np.array([[2.0, 3.0]]), degree=1)
        np.testing.assert_allclose(design, [[1.0, 2.0, 3.0]])

    def test_degree_two_includes_interactions(self):
        design = polynomial_expand(np.array([[2.0, 3.0]]), degree=2)
        # 1, x, y, x*x, x*y, y*y
        np.testing.assert_allclose(design, [[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]])

    def test_rejects_one_dimensional_input(self):
        with pytest.raises(ValueError):
            polynomial_expand(np.array([1.0, 2.0]), degree=2)


class TestRidge:
    def test_recovers_a_linear_relationship(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(200, 2))
        design = polynomial_expand(x, degree=1)
        truth = np.array([[5.0, -2.0], [3.0, 1.0], [-1.0, 4.0]])
        weights = ridge_fit(design, design @ truth, alpha=1e-6)
        np.testing.assert_allclose(weights, truth, atol=1e-3)

    def test_regularisation_shrinks_weights_but_not_the_bias(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(50, 2))
        design = polynomial_expand(x, degree=1)
        targets = design @ np.array([[10.0], [4.0], [4.0]])
        weak = ridge_fit(design, targets, alpha=0.001)
        strong = ridge_fit(design, targets, alpha=500.0)
        assert abs(strong[1, 0]) < abs(weak[1, 0])
        assert strong[0, 0] == pytest.approx(10.0, rel=0.05)


class TestCalibrationFit:
    def _session(self, noise: float = 0.004, samples: int = 12) -> CalibrationSession:
        rng = np.random.default_rng(7)
        targets = pattern_to_pixels(calibration_pattern("13point"), *SCREEN)
        session = CalibrationSession(FEATURES, SCREEN, alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        for point_id, target in enumerate(targets):
            for _ in range(samples):
                session.add(point_id, target,
                            synthetic_features(target[0], target[1], noise, rng))
        return session

    def test_fit_is_accurate_on_clean_synthetic_data(self):
        estimator = self._session().fit()
        assert estimator.is_ready
        # Cross-validated, so this is error at *unseen* points.
        assert estimator.report.mean_error_px < 90.0
        assert estimator.report.n_points == 13

    def test_prediction_lands_near_a_held_out_target(self):
        estimator = self._session().fit()
        result = estimator.estimate(synthetic_features(1200, 700))
        assert result.valid
        assert np.hypot(result.x - 1200, result.y - 700) < 120.0

    def test_serialisation_round_trip_is_exact(self):
        estimator = self._session().fit()
        restored = RidgeGazeEstimator.from_dict(estimator.to_dict())
        probe = synthetic_features(900, 400)
        original = estimator.estimate(probe)
        again = restored.estimate(probe)
        assert (again.x, again.y) == pytest.approx((original.x, original.y))
        assert restored.report.mean_error_px == pytest.approx(
            estimator.report.mean_error_px)

    def test_too_few_points_is_refused(self):
        session = CalibrationSession(FEATURES, SCREEN)
        for point_id in range(3):
            for _ in range(10):
                session.add(point_id, (100.0 * point_id, 100.0),
                            synthetic_features(100.0 * point_id, 100.0, 0.003))
        with pytest.raises(ValueError, match="calibration points"):
            session.fit()

    def test_invalid_frames_are_rejected(self):
        session = CalibrationSession(FEATURES, SCREEN, min_openness=0.16)
        closed = synthetic_features(500, 500)
        closed.values["ear_l"] = closed.values["ear_r"] = 0.02
        assert session.add(0, (500, 500), closed) is False

        turned = synthetic_features(500, 500)
        turned.head_pose = HeadPose(pitch=0.0, yaw=70.0, roll=0.0)
        assert session.add(0, (500, 500), turned) is False

        assert session.add(0, (500, 500), FeatureVector.invalid()) is False
        assert session.rejected == 3
        assert session.samples == []

    def test_uncalibrated_estimator_reports_not_ready(self):
        estimator = RidgeGazeEstimator(FEATURES)
        assert not estimator.is_ready
        assert estimator.estimate(synthetic_features(100, 100)).valid is False


class TestPatterns:
    @pytest.mark.parametrize("name,count", [("5point", 5), ("9point", 9), ("13point", 13)])
    def test_expected_point_counts(self, name, count):
        assert len(calibration_pattern(name)) == count

    def test_all_points_are_inside_the_screen(self):
        for nx, ny in calibration_pattern("13point"):
            assert 0.0 < nx < 1.0 and 0.0 < ny < 1.0

    def test_pixel_conversion_respects_the_monitor_origin(self):
        pixels = pattern_to_pixels([(0.0, 0.0), (1.0, 1.0)], 1920, 1080, (1920, 0))
        assert pixels[0] == (1920.0, 0.0)
        assert pixels[1] == (3840.0, 1080.0)


class TestRobustFilter:
    def test_flags_a_single_outlier(self):
        data = np.tile(np.array([1.0, 2.0]), (12, 1))
        data += np.random.default_rng(3).normal(0, 0.01, data.shape)
        data[5] = [50.0, -30.0]
        mask = robust_filter(data)
        assert not mask[5]
        assert mask.sum() == 11

    def test_small_samples_are_left_alone(self):
        data = np.array([[1.0], [99.0], [1.0]])
        assert robust_filter(data).all()


class TestSmoothing:
    def test_a_steady_signal_converges_to_its_value(self):
        filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        value = 0.0
        for i in range(120):
            value = filt.filter(500.0, i / 30.0)
        assert value == pytest.approx(500.0, abs=1.0)

    def test_noise_is_reduced(self):
        rng = np.random.default_rng(4)
        smoother = GazeSmoother.from_preset("medium")
        raw, smoothed = [], []
        for i in range(200):
            noisy = 800.0 + rng.normal(0, 25)
            raw.append(noisy)
            smoothed.append(smoother.smooth(noisy, noisy, i / 30.0)[0])
        assert np.std(smoothed[50:]) < np.std(raw[50:]) * 0.5

    def test_it_still_follows_a_large_jump(self):
        smoother = GazeSmoother.from_preset("medium")
        for i in range(60):
            smoother.smooth(100.0, 100.0, i / 30.0)
        for i in range(60, 80):
            x, _ = smoother.smooth(1500.0, 1500.0, i / 30.0)
        assert x > 1400.0

    def test_reset_clears_history(self):
        smoother = GazeSmoother.from_preset("high")
        for i in range(30):
            smoother.smooth(100.0, 100.0, i / 30.0)
        smoother.reset()
        assert smoother.smooth(900.0, 900.0, 2.0) == (900.0, 900.0)


class TestFitReport:
    def test_quality_thresholds_scale_with_the_screen(self):
        """Thresholds are fractions of the diagonal, not fixed pixel counts."""
        # mean_error_normalised implies the diagonal, so these describe a
        # 1920x1080 screen (diagonal ~2202 px).
        assert FitReport(50, 40, 90, 50 / 2202).quality() == "GOOD"
        assert FitReport(160, 150, 260, 160 / 2202).quality() == "FAIR"
        assert FitReport(400, 380, 600, 400 / 2202).quality() == "POOR"

    def test_the_same_error_rates_better_on_a_larger_screen(self):
        small = FitReport(160, 150, 240, 160 / 1500)   # small laptop
        large = FitReport(160, 150, 240, 160 / 4000)   # large 4K display
        assert small.quality() == "POOR"
        assert large.quality() == "GOOD"

    def test_dict_round_trip(self):
        report = FitReport(70.5, 61.0, 183.0, 0.03, [1.0, 2.0], 45.0, 40.0, 1.0, 260, 13)
        restored = FitReport.from_dict(report.to_dict())
        assert restored.to_dict() == report.to_dict()

    def test_quality_is_rated_on_the_settled_error(self):
        """The single-frame figure is dominated by noise the filters remove."""
        report = FitReport(mean_error_px=260.0, median_error_px=40.0,
                           max_error_px=90.0, mean_error_normalised=260 / 2202,
                           settled_error_px=50.0)
        assert report.quality() == "GOOD"

    def test_quality_falls_back_when_no_settled_error_was_recorded(self):
        """Profiles saved before the settled figure existed still rate sensibly."""
        assert FitReport(400, 380, 600, 400 / 2202).quality() == "POOR"


class TestMarginFraction:
    """``calibration.margin_fraction`` has to survive being saved."""

    def _fit(self, margin):
        import numpy as np
        from src.tracking.gaze_estimator import RidgeGazeEstimator
        rng = np.random.default_rng(0)
        raw = rng.normal(0, 0.1, (60, 2))
        targets = rng.normal(900, 300, (60, 2))
        groups = np.repeat(np.arange(6), 10)
        return RidgeGazeEstimator.fit(raw, targets, groups, ["iris_l_ix", "iris_l_iy"],
                                      (1920, 1080), margin_fraction=margin)

    def test_the_configured_margin_reaches_the_fitted_model(self):
        assert self._fit(0.4).margin_fraction == pytest.approx(0.4)

    def test_it_survives_serialisation(self):
        """It used to reset to the default on load, so the setting did nothing."""
        from src.tracking.gaze_estimator import RidgeGazeEstimator
        estimator = self._fit(0.4)
        restored = RidgeGazeEstimator.from_dict(estimator.to_dict())
        assert restored.margin_fraction == pytest.approx(0.4)

    def test_the_margin_controls_where_predictions_are_clamped(self):
        wide, narrow = self._fit(0.5), self._fit(0.05)
        assert wide.clamp_to_screen(10_000.0, 540.0)[0] > \
            narrow.clamp_to_screen(10_000.0, 540.0)[0]
