"""Coordinate-space and model-robustness tests.

These lock down the two faults that made calibration read POOR and pushed every
estimate off screen:

1. Calibration targets were computed in **physical** pixels (from ``mss``) but
   drawn by Qt in **logical** pixels. At 150% display scaling the model learned
   to output coordinates 1.5x too large, so predictions left the screen.
2. Features were scaled by their standard deviation *within the calibration
   samples*. Features that barely varied while the user sat still were
   amplified enormously, so a small posture change threw the estimate far off.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tracking.calibration import (CalibrationSession, calibration_pattern,
                                      pattern_to_pixels)
from src.tracking.features import NOMINAL_SCALES, nominal_scales
from src.utils.geometry import Rect
from src.utils.screens import ScreenInfo, find_screen

from tests.test_calibration import FEATURES, SCREEN, synthetic_features

ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]


class TestScreenScaling:
    def test_unscaled_screen_is_identical_in_both_spaces(self):
        screen = ScreenInfo(1, "Main", Rect(0, 0, 1920, 1080), 1.0)
        assert screen.physical_rect == screen.rect
        assert not screen.is_scaled

    def test_150_percent_scaling_converts_correctly(self):
        """A 1920x1080 panel at 150% reports 1280x720 to Qt."""
        screen = ScreenInfo(1, "Main", Rect(0, 0, 1280, 720), 1.5)
        assert screen.is_scaled
        assert screen.physical_rect == Rect(0, 0, 1920, 1080)

    def test_physical_and_logical_round_trip(self):
        screen = ScreenInfo(1, "Main", Rect(0, 0, 1280, 720), 1.5)
        board = Rect(200, 100, 600, 600)
        assert screen.to_logical(screen.to_physical(board)) == board

    def test_a_detected_board_converts_back_to_logical(self):
        """Board detection runs on physical pixels; gaze lives in logical."""
        screen = ScreenInfo(1, "Main", Rect(0, 0, 1280, 720), 1.5)
        detected_physical = Rect(450, 150, 900, 900)
        logical = screen.to_logical(detected_physical)
        assert logical == Rect(300, 100, 600, 600)

    def test_secondary_monitor_offsets_are_preserved(self):
        screen = ScreenInfo(2, "Second", Rect(1280, 0, 1280, 720), 1.5)
        assert screen.physical_rect.x == pytest.approx(1920)

    def test_find_screen_falls_back_to_the_first(self):
        screens = [ScreenInfo(1, "A", Rect(0, 0, 800, 600), 1.0)]
        assert find_screen(screens, 1).name == "A"
        assert find_screen(screens, 99).name == "A"
        assert find_screen([], 1) is None


class TestScalingBugRegression:
    """The DPI mismatch, expressed as the numbers the model actually saw."""

    def _fit_on(self, target_width: int, target_height: int):
        """Calibrate with targets spanning a given coordinate range."""
        rng = np.random.default_rng(31)
        targets = pattern_to_pixels(calibration_pattern("13point"),
                                    target_width, target_height)
        session = CalibrationSession(FEATURES, (target_width, target_height),
                                     alphas=ALPHAS)
        for point_id, target in enumerate(targets):
            # The user's eyes always span the SAME physical screen, so the
            # features depend on the fraction across the display, not on the
            # units the targets happen to be expressed in.
            fx = target[0] / target_width * SCREEN[0]
            fy = target[1] / target_height * SCREEN[1]
            for _ in range(20):
                session.add(point_id, target, synthetic_features(fx, fy, 0.004, rng))
        return session.fit()

    def test_mismatched_units_push_estimates_off_screen(self):
        """Reproduces the original bug: trained on 1920, displayed at 1280."""
        estimator = self._fit_on(1920, 1080)   # physical-pixel targets
        # ...but the visible window was only 1280x720 logical.
        centre = estimator.estimate(synthetic_features(*[s / 2 for s in SCREEN]))
        assert centre.valid
        # The model predicts in the 1920-wide space it was trained on, so a
        # look at the far right lands well beyond a 1280-wide window.
        right = estimator.estimate(synthetic_features(SCREEN[0] * 0.92, SCREEN[1] / 2))
        assert right.x > 1280, "this is the failure the DPI fix prevents"

    def test_consistent_units_keep_estimates_on_screen(self):
        """With one coordinate space throughout, predictions stay in bounds."""
        estimator = self._fit_on(1280, 720)
        for fraction_x, fraction_y in [(0.08, 0.08), (0.5, 0.5), (0.92, 0.92),
                                       (0.08, 0.92), (0.92, 0.08)]:
            result = estimator.estimate(
                synthetic_features(SCREEN[0] * fraction_x, SCREEN[1] * fraction_y))
            assert -100 <= result.x <= 1380, f"x={result.x} escaped the screen"
            assert -100 <= result.y <= 820, f"y={result.y} escaped the screen"


class TestFeatureScaling:
    def test_every_feature_has_a_nominal_scale(self):
        from src.tracking.features import FEATURE_NAMES
        for name in FEATURE_NAMES:
            assert name in NOMINAL_SCALES, f"{name} has no nominal scale"
            assert NOMINAL_SCALES[name] > 0

    def test_scales_are_independent_of_the_data(self):
        first = nominal_scales(FEATURES)
        second = nominal_scales(FEATURES)
        np.testing.assert_array_equal(first, second)

    def test_a_posture_change_no_longer_throws_the_estimate(self):
        """The regression that made the dot fly off screen when leaning in."""
        rng = np.random.default_rng(32)
        targets = pattern_to_pixels(calibration_pattern("13point"), *SCREEN)
        session = CalibrationSession(FEATURES, SCREEN, alphas=ALPHAS)
        for point_id, target in enumerate(targets):
            for _ in range(20):
                features = synthetic_features(target[0], target[1], 0.004, rng)
                # A user sitting still: these barely vary during calibration.
                features.values["face_scale"] = 0.18 + rng.normal(0, 0.0008)
                features.values["face_x"] = rng.normal(0, 0.001)
                features.values["face_y"] = rng.normal(0, 0.001)
                session.add(point_id, target, features)
        estimator = session.fit()

        baseline = estimator.estimate(synthetic_features(960, 540))
        leaned_in = synthetic_features(960, 540)
        leaned_in.values["face_scale"] = 0.20   # moved noticeably closer
        shifted = estimator.estimate(leaned_in)

        drift = np.hypot(shifted.x - baseline.x, shifted.y - baseline.y)
        # Before the fix this drift was over 700 px.
        assert drift < 120.0, f"posture change moved the estimate by {drift:.0f} px"


class TestClamping:
    @pytest.fixture()
    def estimator(self):
        rng = np.random.default_rng(33)
        targets = pattern_to_pixels(calibration_pattern("13point"), *SCREEN)
        session = CalibrationSession(FEATURES, SCREEN, alphas=ALPHAS)
        for point_id, target in enumerate(targets):
            for _ in range(20):
                session.add(point_id, target,
                            synthetic_features(target[0], target[1], 0.004, rng))
        return session.fit()

    def test_absurd_features_cannot_produce_absurd_coordinates(self, estimator):
        wild = synthetic_features(960, 540)
        for name in ("iris_l_x", "iris_r_x", "yaw"):
            wild.values[name] = 50.0     # nonsense a bad frame could produce
        result = estimator.estimate(wild)
        assert result.valid
        assert -1000 < result.x < 3000, "prediction escaped all bounds"
        assert result.out_of_bounds is True

    def test_normal_estimates_are_not_flagged(self, estimator):
        result = estimator.estimate(synthetic_features(960, 540))
        assert result.out_of_bounds is False

    def test_clamp_respects_the_margin(self):
        from src.tracking.gaze_estimator import RidgeGazeEstimator

        estimator = RidgeGazeEstimator(FEATURES, screen_size=(1000, 1000),
                                       margin_fraction=0.1)
        x, y, outside = estimator.clamp_to_screen(5000, -5000)
        assert (x, y) == (1100.0, -100.0)
        assert outside is True

    def test_non_finite_predictions_are_rejected(self, estimator):
        broken = synthetic_features(960, 540)
        broken.values["iris_l_x"] = float("nan")
        assert estimator.estimate(broken).valid is False
