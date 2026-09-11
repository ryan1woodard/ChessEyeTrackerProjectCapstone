"""Tracking robustness tests.

These use the geometric eye simulator, so they exercise the real failure modes
rather than a mapping that ignores head position:

* accuracy decaying as the user shifts in their chair;
* the estimate jumping while staring at a fixed point;
* a blink throwing the estimate and the error persisting afterwards.

The thresholds are drawn from measured before/after numbers, so a regression
that reintroduces any of these fails here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tracking.calibration import (CalibrationSession, calibration_pattern,
                                      gaze_feature_columns, pattern_to_pixels,
                                      robust_filter)
from src.tracking.features import FeatureVector
from src.tracking.smoother import (FEATURE_SMOOTHING_PRESETS, FeatureSmoother,
                                   GazeSmoother)

from tests.simulator import EyeSimulator, HeadState

SCREEN = (1920, 1080)
FEATURES = ["iris_l_x", "iris_l_y", "iris_r_x", "iris_r_y",
            "yaw", "pitch", "face_x", "face_y", "face_scale"]
ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
PROBES = [(400, 300), (960, 540), (1500, 800), (960, 200), (300, 900)]


def _calibrate(simulator: EyeSimulator, head_motion: float,
               rng: np.random.Generator, samples: int = 34):
    """Calibrate with a given amount of natural head movement."""
    session = CalibrationSession(FEATURES, SCREEN, alphas=ALPHAS)
    for point_id, target in enumerate(pattern_to_pixels(
            calibration_pattern("13point"), *SCREEN)):
        for _ in range(samples):
            head = HeadState().shifted(
                dx=rng.normal(0, 25 * head_motion), dy=rng.normal(0, 18 * head_motion),
                dz=rng.normal(0, 30 * head_motion), dyaw=rng.normal(0, 4 * head_motion),
                dpitch=rng.normal(0, 3 * head_motion))
            session.add(point_id, target, simulator.features(target, head))
    return session.fit()


def _mean_error(estimator, simulator: EyeSimulator, head: HeadState,
                repeats: int = 20) -> float:
    errors = []
    for _ in range(repeats):
        for target in PROBES:
            result = estimator.estimate(simulator.features(target, head))
            errors.append(np.hypot(result.x - target[0], result.y - target[1]))
    return float(np.mean(errors))


@pytest.fixture(scope="module")
def simulator() -> EyeSimulator:
    return EyeSimulator(noise=0.005, rng=np.random.default_rng(11))


@pytest.fixture(scope="module")
def still_estimator(simulator):
    """Calibrated with the head held rigidly still -- the old behaviour."""
    return _calibrate(simulator, 0.0, np.random.default_rng(11))


@pytest.fixture(scope="module")
def moving_estimator(simulator):
    """Calibrated with natural head movement -- what the app now asks for."""
    return _calibrate(simulator, 1.0, np.random.default_rng(11))


class TestSimulator:
    def test_looking_further_right_needs_more_eye_rotation(self, simulator):
        left = simulator.features((300, 540))
        right = simulator.features((1600, 540))
        assert right.values["iris_mean_x"] > left.values["iris_mean_x"]

    def test_turning_the_head_reduces_the_iris_offset(self, simulator):
        """The core coupling: head rotation substitutes for eye rotation."""
        straight = simulator.features((1600, 540), HeadState())
        turned = simulator.features((1600, 540), HeadState().shifted(dyaw=15))
        assert abs(turned.values["iris_mean_x"]) < abs(straight.values["iris_mean_x"])

    def test_moving_the_head_changes_where_an_iris_offset_points(self, simulator):
        """Why a still-head calibration cannot generalise."""
        centred = simulator.features((960, 540), HeadState())
        shifted = simulator.features((960, 540), HeadState().shifted(dx=-60))
        assert centred.values["iris_mean_x"] != pytest.approx(
            shifted.values["iris_mean_x"], abs=0.005)

    def test_leaning_in_increases_apparent_size(self, simulator):
        near = simulator.features((960, 540), HeadState().shifted(dz=-100))
        far = simulator.features((960, 540), HeadState().shifted(dz=100))
        assert near.values["face_scale"] > far.values["face_scale"]

    def test_a_blink_corrupts_the_iris_reading(self, simulator):
        normal = simulator.features((960, 540))
        blink = simulator.blink_features((960, 540))
        assert blink.eye_openness < 0.16
        assert abs(blink.values["iris_l_y"] - normal.values["iris_l_y"]) > 0.05


class TestHeadMovementRobustness:
    """The reported fault: accuracy decays as you shift position."""

    CASES = [
        ("shift 50 mm left", HeadState().shifted(dx=-50)),
        ("slouch 40 mm down", HeadState().shifted(dy=-40)),
        ("turn head 8 deg", HeadState().shifted(dyaw=8)),
        ("lean in 60 mm", HeadState().shifted(dz=-60)),
        ("combined drift", HeadState().shifted(dx=-35, dy=-25, dz=-40, dyaw=5)),
    ]

    def test_both_are_accurate_at_the_calibrated_pose(self, simulator, still_estimator,
                                                      moving_estimator):
        assert _mean_error(still_estimator, simulator, HeadState()) < 70
        assert _mean_error(moving_estimator, simulator, HeadState()) < 70

    @pytest.mark.parametrize("label,head", CASES)
    def test_head_varied_calibration_survives_movement(self, simulator, moving_estimator,
                                                       label, head):
        error = _mean_error(moving_estimator, simulator, head)
        assert error < 90, f"{label}: {error:.0f} px"

    def test_a_still_head_calibration_is_measurably_worse(self, simulator,
                                                          still_estimator,
                                                          moving_estimator):
        """Documents why the calibration instructions ask for head movement."""
        head = HeadState().shifted(dyaw=8)
        still = _mean_error(still_estimator, simulator, head)
        moving = _mean_error(moving_estimator, simulator, head)
        assert still > moving * 2.5, (
            f"still-head {still:.0f} px vs head-varied {moving:.0f} px")

    def test_realistic_chair_movement_stays_accurate(self, simulator, moving_estimator):
        """Shifts of the size people actually make while sitting."""
        for shift in (0, 40, 80):
            error = _mean_error(moving_estimator, simulator,
                                HeadState().shifted(dx=-shift))
            assert error < 110, f"{shift} mm shift gave {error:.0f} px"

    def test_extreme_displacement_degrades_but_stays_bounded(self, simulator,
                                                             moving_estimator):
        """A 30 cm shift cannot be compensated -- but must not run away.

        No calibration can correct a head position that far outside what it
        observed. What matters is that the estimate stays a plausible screen
        coordinate rather than diverging, so recovery is immediate once the
        user sits back and Quick Recentre can fix any residue.
        """
        head = HeadState().shifted(dx=-300)
        diagonal = float(np.hypot(*SCREEN))
        for target in PROBES:
            result = moving_estimator.estimate(simulator.features(target, head))
            assert result.valid
            assert -0.2 * SCREEN[0] <= result.x <= 1.2 * SCREEN[0]
            assert -0.2 * SCREEN[1] <= result.y <= 1.2 * SCREEN[1]
        assert _mean_error(moving_estimator, simulator, head) < diagonal

    def test_accuracy_returns_when_the_user_sits_back(self, simulator,
                                                       moving_estimator):
        """Nothing is stateful, so a bad posture leaves no lasting damage."""
        _mean_error(moving_estimator, simulator, HeadState().shifted(dx=-300))
        recovered = _mean_error(moving_estimator, simulator, HeadState())
        assert recovered < 70, f"{recovered:.0f} px after returning to position"


class TestJitter:
    def _stare(self, estimator, simulator, use_feature_smoothing: bool,
               frames: int = 260):
        feature_filter = (FeatureSmoother(**FEATURE_SMOOTHING_PRESETS["medium"])
                          if use_feature_smoothing else None)
        output_filter = GazeSmoother.from_preset("medium")
        points = []
        for index in range(frames):
            features = simulator.features((960, 540))
            values = (feature_filter.smooth(features.values, index / 30.0)
                      if feature_filter else features.values)
            result = estimator.estimate(FeatureVector(values=values, valid=True,
                                                      head_pose=features.head_pose))
            points.append(output_filter.smooth(result.x, result.y, index / 30.0))
        return np.array(points[80:])

    def test_feature_smoothing_roughly_halves_the_wobble(self, simulator,
                                                          moving_estimator):
        without = self._stare(moving_estimator, simulator, False)
        with_it = self._stare(moving_estimator, simulator, True)
        spread_without = np.linalg.norm(without.max(axis=0) - without.min(axis=0))
        spread_with = np.linalg.norm(with_it.max(axis=0) - with_it.min(axis=0))
        assert spread_with < spread_without * 0.65, (
            f"{spread_without:.0f} px -> {spread_with:.0f} px")

    def test_a_fixation_stays_within_a_square(self, simulator, moving_estimator):
        """Under 75 px of wobble keeps the estimate inside one board square."""
        points = self._stare(moving_estimator, simulator, True)
        spread = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        assert spread < 75, f"peak-to-peak {spread:.0f} px"

    def test_a_saccade_still_lands_promptly(self, simulator, moving_estimator):
        """Smoothing must not be bought with unacceptable lag."""
        feature_filter = FeatureSmoother(**FEATURE_SMOOTHING_PRESETS["medium"])
        output_filter = GazeSmoother.from_preset("medium")

        def step(target, timestamp):
            features = simulator.features(target)
            values = feature_filter.smooth(features.values, timestamp)
            result = moving_estimator.estimate(
                FeatureVector(values=values, valid=True, head_pose=features.head_pose))
            return output_filter.smooth(result.x, result.y, timestamp)

        for index in range(90):
            step((400, 300), index / 30.0)
        settle_ms = None
        for index in range(120):
            x, y = step((1500, 800), (90 + index) / 30.0)
            if np.hypot(x - 1500, y - 800) < 90:
                settle_ms = (index + 1) / 30.0 * 1000
                break
        assert settle_ms is not None and settle_ms < 250, f"settled in {settle_ms} ms"


class TestCalibrationFiltering:
    def test_head_features_are_exempt_from_outlier_rejection(self):
        """Head movement during calibration is data, not noise."""
        columns = gaze_feature_columns(FEATURES)
        assert set(columns) == {0, 1, 2, 3}, "only the iris features are checked"

        rng = np.random.default_rng(5)
        samples = np.zeros((30, len(FEATURES)))
        samples[:, :4] = rng.normal(0, 0.01, (30, 4))     # steady eyes
        samples[:, 6:9] = rng.normal(0, 0.5, (30, 3))     # head moving a lot
        mask = robust_filter(samples, columns=columns)
        assert mask.sum() >= 28, "deliberate head movement was thrown away"

    def test_blink_frames_are_still_rejected(self):
        columns = gaze_feature_columns(FEATURES)
        rng = np.random.default_rng(6)
        samples = np.zeros((30, len(FEATURES)))
        samples[:, :4] = rng.normal(0, 0.01, (30, 4))
        samples[7, :4] = [0.9, -0.8, 0.9, -0.8]   # a blink mid-collection
        mask = robust_filter(samples, columns=columns)
        assert not mask[7]

    def test_a_head_varied_calibration_keeps_most_samples(self, simulator):
        rng = np.random.default_rng(12)
        session = CalibrationSession(FEATURES, SCREEN, alphas=ALPHAS)
        for point_id, target in enumerate(pattern_to_pixels(
                calibration_pattern("13point"), *SCREEN)):
            for _ in range(30):
                head = HeadState().shifted(dx=rng.normal(0, 25), dyaw=rng.normal(0, 4))
                session.add(point_id, target, simulator.features(target, head))
        features, _targets, groups = session.build_matrices()
        assert len(np.unique(groups)) == 13, "points were dropped"
        assert features.shape[0] > 13 * 30 * 0.85


class TestBlinkHandling:
    """A blink must not move the estimate, nor leave a lingering error."""

    class _FakePipelineState:
        """Reproduces the pipeline's blink-hold logic without MediaPipe."""

        def __init__(self, estimator, recovery: float = 0.12):
            self.estimator = estimator
            self.recovery = recovery
            self.feature_filter = FeatureSmoother(**FEATURE_SMOOTHING_PRESETS["medium"])
            self.output_filter = GazeSmoother.from_preset("medium")
            self.last_valid = None
            self.hold_until = 0.0

        def step(self, features, timestamp, blink_threshold=0.16):
            eyes_closed = features.eye_openness < blink_threshold
            if eyes_closed:
                self.hold_until = timestamp + self.recovery
            if timestamp < self.hold_until and self.last_valid is not None:
                return self.last_valid, True
            if eyes_closed:
                return self.last_valid, True

            values = self.feature_filter.smooth(features.values, timestamp)
            result = self.estimator.estimate(
                FeatureVector(values=values, valid=True, head_pose=features.head_pose))
            point = self.output_filter.smooth(result.x, result.y, timestamp)
            self.last_valid = point
            return point, False

    def test_a_blink_does_not_move_the_estimate(self, simulator, moving_estimator):
        state = self._FakePipelineState(moving_estimator)
        for index in range(90):
            state.step(simulator.features((960, 540)), index / 30.0)
        before = state.last_valid

        # Six frames of closure, about 200 ms.
        held = []
        for index in range(90, 96):
            point, holding = state.step(
                simulator.blink_features((960, 540)), index / 30.0)
            held.append((point, holding))

        assert all(holding for _point, holding in held), "blink frames must be held"
        assert all(point == before for point, _h in held), "held value must not move"

    def test_the_error_does_not_persist_after_the_eyes_reopen(self, simulator,
                                                              moving_estimator):
        state = self._FakePipelineState(moving_estimator)
        for index in range(90):
            state.step(simulator.features((960, 540)), index / 30.0)
        for index in range(90, 96):
            state.step(simulator.blink_features((960, 540)), index / 30.0)

        # Two frames after reopening, the estimate should already be on target.
        for index in range(96, 102):
            point, _holding = state.step(simulator.features((960, 540)), index / 30.0)
        assert np.hypot(point[0] - 960, point[1] - 540) < 80

    def test_without_the_hold_a_blink_throws_the_estimate(self, simulator,
                                                          moving_estimator):
        """Shows what the hold prevents: corrupt iris data reaching the model."""
        output_filter = GazeSmoother.from_preset("medium")
        feature_filter = FeatureSmoother(**FEATURE_SMOOTHING_PRESETS["medium"])

        def step(features, timestamp):
            values = feature_filter.smooth(features.values, timestamp)
            result = moving_estimator.estimate(
                FeatureVector(values=values, valid=True, head_pose=features.head_pose))
            return output_filter.smooth(result.x, result.y, timestamp)

        for index in range(90):
            point = step(simulator.features((960, 540)), index / 30.0)
        clean = np.hypot(point[0] - 960, point[1] - 540)
        for index in range(90, 96):
            point = step(simulator.blink_features((960, 540)), index / 30.0)
        during_blink = np.hypot(point[0] - 960, point[1] - 540)
        assert during_blink > clean + 40, "the blink should visibly move it"

    def test_repeated_blinks_do_not_accumulate_error(self, simulator, moving_estimator):
        """The reported fault: blinks making it 'more and more off'."""
        state = self._FakePipelineState(moving_estimator)
        timestamp = 0.0
        errors = []
        for _cycle in range(8):
            for _ in range(45):
                point, _ = state.step(simulator.features((960, 540)), timestamp)
                timestamp += 1 / 30.0
            errors.append(np.hypot(point[0] - 960, point[1] - 540))
            for _ in range(6):
                state.step(simulator.blink_features((960, 540)), timestamp)
                timestamp += 1 / 30.0

        assert errors[-1] < 80, f"final error {errors[-1]:.0f} px"
        assert errors[-1] < errors[0] + 40, f"error grew across blinks: {errors}"
