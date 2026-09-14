"""Tracking robustness tests.

These drive the real feature extractor, head-pose estimator and pipeline with
landmarks rendered by the geometric simulator, so they exercise the failures
users actually report rather than a mapping that ignores the head:

* accuracy collapsing as soon as the head is **tilted**;
* accuracy decaying as the user shifts or leans in their chair;
* the estimate jumping while staring at a fixed point;
* a blink throwing the estimate, and the error persisting afterwards;
* tracking coming back wrong after the face is lost for a second.

Thresholds come from measured before/after numbers, with enough headroom that
they do not fail on noise, so a regression that reintroduces any of these is
caught here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tracking.blink import BlinkDetector
from src.tracking.calibration import (COVERAGE_TARGETS, CalibrationSession,
                                      calibration_pattern,
                                      gaze_feature_columns, pattern_to_pixels,
                                      posture_cue, robust_filter)
from src.tracking import face_tracker as fl
from src.tracking.face_frame import fit_face_frame
from src.tracking.face_tracker import FaceLandmarks
from src.tracking.gaze_estimator import (polynomial_expand, ridge_fit,
                                         robust_ridge_fit)
from src.tracking.head_pose import HeadPoseEstimator
from src.tracking.pipeline import TrackingPipeline
from src.tracking.smoother import (FEATURE_SMOOTHING_PRESETS, SMOOTHING_PRESETS,
                                   FeatureSmoother, GazeSmoother)

from tests.simulator import (EyeSimulator, HeadState, probe_targets,
                             rotation_matrix)

SCREEN = (1920, 1080)
FPS = 30.0

#: The shipped feature set: iris offsets on the camera's axes, the tilt that
#: relates them to the head, and metric head position.
FEATURES = ["iris_l_ix", "iris_l_iy", "iris_r_ix", "iris_r_iy",
            "ray_l_x", "ray_l_y", "ray_r_x", "ray_r_y",
            "eye_tilt", "yaw", "pitch", "head_x", "head_y", "head_z"]

#: The feature set this project shipped before head tilt was handled: iris
#: offsets in the eye's own frame, which are roll-invariant by construction.
LEGACY_FEATURES = ["iris_l_x", "iris_l_y", "iris_r_x", "iris_r_y",
                   "yaw", "pitch", "face_x", "face_y", "face_scale"]

ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
PROBES = probe_targets(SCREEN, grid=3)


def _calibrate(simulator: EyeSimulator, features, rng: np.random.Generator,
               motion: float = 1.0, tilt: float = 1.0, samples: int = 26):
    """Calibrate with a given amount of head movement and head tilt."""
    session = CalibrationSession(features, SCREEN, alphas=ALPHAS)
    for point_id, target in enumerate(pattern_to_pixels(
            calibration_pattern("13point"), *SCREEN)):
        for _ in range(samples):
            head = HeadState().shifted(
                dx=rng.normal(0, 28 * motion), dy=rng.normal(0, 20 * motion),
                dz=rng.normal(0, 35 * motion), dyaw=rng.normal(0, 4.5 * motion),
                dpitch=rng.normal(0, 3.5 * motion),
                droll=rng.normal(0, 8.0 * tilt))
            session.add(point_id, target, simulator.features(target, head))
    return session.fit()


def _mean_error(estimator, simulator: EyeSimulator, head: HeadState,
                repeats: int = 3) -> float:
    """Single-frame error, averaged over a grid of probe targets."""
    errors = []
    for _ in range(repeats):
        for target in PROBES:
            result = estimator.estimate(simulator.features(target, head))
            errors.append(np.hypot(result.x - target[0], result.y - target[1]))
    return float(np.mean(errors))


@pytest.fixture(scope="module")
def simulator() -> EyeSimulator:
    """A subject whose face is not the canonical one, as everybody's is not."""
    return EyeSimulator(noise_px=0.35, shape_mm=3.0, rng=np.random.default_rng(11))


@pytest.fixture(scope="module")
def average_face() -> EyeSimulator:
    """A subject who happens to match the canonical head exactly.

    Only for asserting absolute angles. Anything about accuracy should use
    ``simulator``, because a tracker that only works on the average face is not
    a working tracker.
    """
    return EyeSimulator(noise_px=0.35, shape_mm=0.0, rng=np.random.default_rng(11))


@pytest.fixture(scope="module")
def estimator(simulator):
    """The shipped configuration: tilt-aware features, varied calibration."""
    return _calibrate(simulator, FEATURES, np.random.default_rng(11))


@pytest.fixture(scope="module")
def legacy_estimator(simulator):
    """The old configuration, kept to document what was actually wrong."""
    return _calibrate(simulator, LEGACY_FEATURES, np.random.default_rng(11))


@pytest.fixture(scope="module")
def still_estimator(simulator):
    """Calibrated with the head held rigidly still."""
    return _calibrate(simulator, FEATURES, np.random.default_rng(11),
                      motion=0.0, tilt=0.0)


class TestSimulatorGeometry:
    """The simulator has to be right before anything measured with it means much."""

    def test_looking_further_right_needs_more_eye_rotation(self, simulator):
        left = simulator.features((300, 540))
        right = simulator.features((1600, 540))
        assert right.values["iris_mean_ix"] > left.values["iris_mean_ix"]

    def test_turning_the_head_reduces_the_iris_offset(self, simulator):
        """The core coupling: head rotation substitutes for eye rotation."""
        straight = simulator.features((1600, 540), HeadState())
        turned = simulator.features((1600, 540), HeadState(yaw_deg=15))
        assert abs(turned.values["iris_mean_ix"]) < abs(straight.values["iris_mean_ix"])

    def test_moving_the_head_changes_where_an_iris_offset_points(self, simulator):
        centred = simulator.features((960, 540), HeadState())
        shifted = simulator.features((960, 540), HeadState(x=-60))
        assert centred.values["iris_mean_ix"] != pytest.approx(
            shifted.values["iris_mean_ix"], abs=0.005)

    def test_leaning_in_increases_apparent_size(self, simulator):
        near = simulator.features((960, 540), HeadState(z=500))
        far = simulator.features((960, 540), HeadState(z=700))
        assert near.values["face_scale"] > far.values["face_scale"]
        assert near.values["head_z"] < far.values["head_z"]

    def test_head_position_features_are_metric(self, simulator):
        """``head_x`` counts interocular widths, so it is linear in millimetres."""
        base = simulator.features((960, 540), HeadState())
        shifted = simulator.features((960, 540), HeadState(x=-60))
        # 60 mm against a ~60 mm interocular distance is about one unit.
        assert base.values["head_x"] - shifted.values["head_x"] == pytest.approx(1.0, abs=0.15)

    def test_a_blink_corrupts_the_iris_reading(self, simulator):
        normal = simulator.features((960, 540))
        blink = simulator.blink_features((960, 540))
        assert blink.eye_openness < 0.5 * normal.eye_openness
        assert abs(blink.values["iris_l_iy"] - normal.values["iris_l_iy"]) > 0.03


class TestHeadTilt:
    """The reported fault: tracking falls apart once the head is tilted."""

    TILTS = [-25, -15, -8, 0, 8, 15, 25]

    def test_the_image_aligned_pair_is_the_one_that_survives_tilt(self, simulator):
        """Why the old feature set could not be fixed by more calibration.

        Holding a gaze on one screen point and tilting the head does not change
        where the eye points in the world, so the image-aligned offset barely
        moves. The eye-local offset is that same direction re-expressed in a
        frame that has rotated underneath it, so it swings -- and a model given
        only the eye-local pair is being asked to predict an unchanged screen
        position from an input that changed.
        """
        upright = simulator.features((960, 540), HeadState())
        tilted = simulator.features((960, 540), HeadState(roll_deg=20))
        image_change = abs(tilted.values["iris_mean_ix"]
                           - upright.values["iris_mean_ix"])
        local_change = abs(tilted.values["iris_mean_x"]
                           - upright.values["iris_mean_x"])
        assert local_change > 2 * image_change, (
            f"image-aligned moved {image_change:.4f}, eye-local {local_change:.4f}")

    def test_eye_tilt_measures_roll_accurately(self, simulator):
        for roll in self.TILTS:
            features = simulator.features((960, 540), HeadState(roll_deg=roll))
            assert features.values["eye_tilt"] * 30.0 == pytest.approx(roll, abs=2.0)

    @pytest.mark.parametrize("roll", TILTS)
    def test_accuracy_survives_head_tilt(self, simulator, estimator, roll):
        error = _mean_error(estimator, simulator, HeadState(roll_deg=roll))
        assert error < 90, f"tilt {roll} deg gave {error:.0f} px"

    def test_tilt_is_no_worse_than_sitting_upright(self, simulator, estimator):
        upright = _mean_error(estimator, simulator, HeadState())
        tilted = _mean_error(estimator, simulator, HeadState(roll_deg=18))
        assert tilted < upright * 1.8, f"upright {upright:.0f} px, tilted {tilted:.0f} px"

    def test_the_old_feature_set_was_measurably_broken_by_tilt(
            self, simulator, estimator, legacy_estimator):
        """Documents the bug this feature set exists to fix."""
        head = HeadState(roll_deg=20)
        legacy = _mean_error(legacy_estimator, simulator, head)
        fixed = _mean_error(estimator, simulator, head)
        assert legacy > fixed * 2.5, (
            f"eye-local {legacy:.0f} px vs image-aligned {fixed:.0f} px")

    def test_tilt_combined_with_other_movement(self, simulator, estimator):
        for label, head in (
            ("tilt and turn", HeadState(roll_deg=16, yaw_deg=10)),
            ("tilt and lean", HeadState(roll_deg=-16, z=520, y=90)),
            ("tilt and shift", HeadState(roll_deg=14, x=-45)),
        ):
            error = _mean_error(estimator, simulator, head)
            assert error < 110, f"{label}: {error:.0f} px"

    def test_a_tilt_calibration_never_saw_still_degrades_gracefully(
            self, simulator, still_estimator):
        """A rigid calibration cannot compensate, but must not run away."""
        for roll in (-25, 25):
            error = _mean_error(still_estimator, simulator, HeadState(roll_deg=roll))
            assert error < float(np.hypot(*SCREEN))


class TestHeadMovementRobustness:
    """Accuracy as the user shifts in their chair."""

    CASES = [
        ("shift 50 mm left", HeadState(x=-50)),
        ("slouch 40 mm down", HeadState(y=100)),
        ("turn head 8 deg", HeadState(yaw_deg=8)),
        ("lean in 60 mm", HeadState(z=540)),
        ("sit back 90 mm", HeadState(z=690)),
        ("combined drift", HeadState(x=-35, y=85, z=560, yaw_deg=5)),
    ]

    def test_accurate_at_the_calibrated_pose(self, simulator, estimator):
        assert _mean_error(estimator, simulator, HeadState()) < 60

    @pytest.mark.parametrize("label,head", CASES)
    def test_movement_is_survived(self, simulator, estimator, label, head):
        error = _mean_error(estimator, simulator, head)
        assert error < 100, f"{label}: {error:.0f} px"

    def test_a_still_calibration_is_measurably_worse(self, simulator, estimator,
                                                     still_estimator):
        """Documents why calibration asks for posture changes."""
        head = HeadState(z=700, x=-40)
        still = _mean_error(still_estimator, simulator, head)
        varied = _mean_error(estimator, simulator, head)
        assert still > varied * 1.8, (
            f"still-head {still:.0f} px vs varied {varied:.0f} px")

    def test_extreme_displacement_degrades_but_stays_bounded(self, simulator, estimator):
        """A 30 cm shift cannot be compensated -- but must not diverge.

        What matters is that the estimate stays a plausible screen coordinate,
        so recovery is immediate once the user sits back.
        """
        head = HeadState(x=-300)
        for target in PROBES:
            result = estimator.estimate(simulator.features(target, head))
            assert result.valid
            assert -0.2 * SCREEN[0] <= result.x <= 1.2 * SCREEN[0]
            assert -0.2 * SCREEN[1] <= result.y <= 1.2 * SCREEN[1]

    def test_accuracy_returns_when_the_user_sits_back(self, simulator, estimator):
        """The estimator is stateless, so a bad posture leaves no damage."""
        _mean_error(estimator, simulator, HeadState(x=-300, roll_deg=30))
        recovered = _mean_error(estimator, simulator, HeadState())
        assert recovered < 60, f"{recovered:.0f} px after returning to position"


class TestHeadPose:
    """Both landmark backends must mean the same thing by an angle."""

    @pytest.mark.parametrize("yaw,pitch,roll", [
        (0, 0, 0), (15, 0, 0), (0, 10, 0), (0, 0, 15), (10, 8, -12), (-12, -6, 20),
    ])
    def test_the_two_fallback_backends_agree(self, average_face, yaw, pitch, roll):
        """A profile fitted on one backend has to stay valid on the other.

        Both are fallbacks now -- the rigid frame is preferred -- but they are
        what runs when it is unusable, and they used to disagree. The Tasks
        matrix decomposes to all three angles negated; negating only two of
        them left roll reading backwards on that backend, so head tilt was
        compensated in the wrong direction depending on which MediaPipe
        happened to be installed.
        """
        head = HeadState(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)
        marks = average_face.landmarks((960, 540), head)
        from_pnp = HeadPoseEstimator(use_face_frame=False).estimate(marks)

        # MediaPipe Tasks reports a y-up, z-towards-viewer face-to-camera matrix.
        flip = np.diag([1.0, -1.0, -1.0])
        matrix = np.eye(4)
        matrix[:3, :3] = flip @ rotation_matrix(yaw, pitch, roll) @ flip
        from_tasks = HeadPoseEstimator(use_face_frame=False).estimate(
            FaceLandmarks(marks.points, marks.frame_width, marks.frame_height,
                          True, matrix))

        assert from_tasks.yaw == pytest.approx(from_pnp.yaw, abs=2.0)
        assert from_tasks.pitch == pytest.approx(from_pnp.pitch, abs=2.5)
        assert from_tasks.roll == pytest.approx(from_pnp.roll, abs=0.5)

    @pytest.mark.parametrize("roll", [-25, -10, 0, 10, 25])
    def test_roll_is_accurate(self, average_face, roll):
        marks = average_face.landmarks((960, 540), HeadState(roll_deg=roll))
        assert HeadPoseEstimator().estimate(marks).roll == pytest.approx(roll, abs=2.0)

    @pytest.mark.parametrize("roll", [-25, -10, 0, 10, 25])
    def test_roll_stays_accurate_on_an_unusual_face(self, simulator, roll):
        """The canonical head is an average; nobody's face is the average."""
        marks = simulator.landmarks((960, 540), HeadState(roll_deg=roll))
        assert HeadPoseEstimator().estimate(marks).roll == pytest.approx(roll, abs=5.0)

    def test_the_frame_is_far_steadier_than_solvepnp(self, simulator):
        """Six landmarks cannot average anything; eighteen can.

        ``solvePnP`` on six points is not merely noisier, it is unusable on the
        pitch axis -- several degrees of wobble frame to frame while the head
        is perfectly still.
        """
        def spread(use_frame):
            angles = []
            estimator = HeadPoseEstimator(use_face_frame=use_frame)
            for _ in range(60):
                pose = estimator.estimate(
                    simulator.landmarks((960, 540), HeadState()))
                angles.append([pose.yaw, pose.pitch, pose.roll])
            return np.std(angles, axis=0)

        frame, pnp = spread(True), spread(False)
        assert np.all(frame < pnp), f"frame {frame.round(2)} vs solvePnP {pnp.round(2)}"
        assert frame.max() < 0.5, f"frame wobble {frame.round(2)} deg"

    def test_solvepnp_is_the_one_that_unusual_faces_break(self):
        """Why the rigid frame is preferred rather than merely available.

        Fitting six landmarks to a canonical six leaves nothing over to absorb
        the difference between that canonical face and the person in front of
        the camera, so the mismatch goes straight into the angles. Eighteen
        landmarks and eight parameters leave room for it.
        """
        poses = [(0, 0, 0), (18, 0, 0), (-18, 0, 0), (0, 12, 0), (0, -12, 0)]

        def tracking_error(use_frame: bool, shape_mm: float) -> float:
            subjects = []
            for seed in range(4):
                sim = EyeSimulator(noise_px=0.35, shape_mm=shape_mm,
                                   rng=np.random.default_rng(seed))
                estimator = HeadPoseEstimator(use_face_frame=use_frame)
                estimated, true = [], []
                for yaw, pitch, roll in poses:
                    pose = estimator.estimate(sim.landmarks((960, 540), HeadState(
                        yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)))
                    estimated.append([pose.yaw, pose.pitch])
                    true.append([yaw, pitch])
                estimated, true = np.array(estimated), np.array(true)
                # Each subject's constant bias is absorbed by calibration; what
                # matters is whether the estimate MOVES with the head.
                subjects.append(np.abs((estimated - estimated.mean(0))
                                       - (true - true.mean(0))).mean())
            return float(np.mean(subjects))

        average = tracking_error(use_frame=False, shape_mm=0.0)
        unusual = tracking_error(use_frame=False, shape_mm=6.0)
        assert unusual > 2 * average, (
            f"solvePnP: {average:.2f} deg on an average face, {unusual:.2f} on an "
            f"unusual one")

        # The rigid frame tracks a little less faithfully in absolute terms --
        # weak perspective compresses out-of-plane angles -- but it does so by
        # the same amount whoever is sitting there, which is what a calibrated
        # feature needs.
        frame_average = tracking_error(use_frame=True, shape_mm=0.0)
        frame_unusual = tracking_error(use_frame=True, shape_mm=6.0)
        assert frame_unusual < 1.3 * frame_average, (
            f"frame: {frame_average:.2f} deg vs {frame_unusual:.2f} deg")


def _pipeline(estimator, preset: str = "medium") -> TrackingPipeline:
    pipeline = TrackingPipeline(smoothing_preset=preset)
    pipeline.set_estimator(estimator)
    pipeline.screen_size = SCREEN
    return pipeline


def _stare(pipeline, simulator, target, head=None, frames=120, start=0.0):
    points = []
    for index in range(frames):
        sample = pipeline.process_features(
            simulator.features(target, head), start + index / FPS)
        points.append((sample.x, sample.y) if sample.valid else None)
    return points


class TestJitterAndLatency:
    def test_a_fixation_stays_within_a_square(self, simulator, estimator):
        """Under 75 px of wobble keeps the estimate inside one board square."""
        points = np.array([p for p in _stare(_pipeline(estimator), simulator,
                                             (960, 540))[60:] if p])
        spread = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        assert spread < 75, f"peak-to-peak {spread:.0f} px"

    def test_smoothing_is_what_buys_that(self, simulator, estimator):
        def spread(preset):
            points = np.array([p for p in _stare(_pipeline(estimator, preset), simulator,
                                                 (960, 540), frames=140)[70:] if p])
            return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        assert spread("medium") < spread("off") * 0.6

    def test_a_saccade_still_lands_promptly(self, simulator, estimator):
        """Smoothing must not be bought with unacceptable lag."""
        pipeline = _pipeline(estimator)
        _stare(pipeline, simulator, (400, 300), frames=90)
        settle_ms = None
        for index in range(150):
            sample = pipeline.process_features(
                simulator.features((1500, 800)), (90 + index) / FPS)
            if sample.valid and np.hypot(sample.x - 1500, sample.y - 800) < 90:
                settle_ms = (index + 1) / FPS * 1000
                break
        assert settle_ms is not None and settle_ms < 250, f"settled in {settle_ms} ms"

    def test_a_settled_fixation_is_accurate(self, simulator, estimator):
        points = [p for p in _stare(_pipeline(estimator), simulator, (960, 540))[90:] if p]
        error = float(np.mean([np.hypot(x - 960, y - 540) for x, y in points]))
        assert error < 45, f"settled error {error:.0f} px"

    def test_a_settled_fixation_is_accurate_with_the_head_tilted(self, simulator,
                                                                 estimator):
        head = HeadState(roll_deg=18)
        points = [p for p in _stare(_pipeline(estimator), simulator,
                                    (960, 540), head)[90:] if p]
        error = float(np.mean([np.hypot(x - 960, y - 540) for x, y in points]))
        assert error < 70, f"settled error while tilted {error:.0f} px"

    def test_every_preset_can_follow_a_saccade(self, estimator, simulator):
        """A low cutoff with no speed term never opens up and lags for a second."""
        for preset in SMOOTHING_PRESETS:
            if preset == "off":
                continue
            assert SMOOTHING_PRESETS[preset]["beta"] > 0, preset
            assert FEATURE_SMOOTHING_PRESETS[preset]["beta"] > 0, preset


class TestBlinkHandling:
    """A blink must not move the estimate, nor leave a lingering error."""

    def _blink(self, pipeline, simulator, target, start_frame, frames=6, head=None):
        held = []
        for index in range(frames):
            sample = pipeline.process_features(
                simulator.blink_features(target, head),
                (start_frame + index) / FPS)
            held.append(sample)
        return held

    def test_a_blink_does_not_move_the_estimate(self, simulator, estimator):
        pipeline = _pipeline(estimator)
        _stare(pipeline, simulator, (960, 540), frames=90)
        before = pipeline._last_valid[:2]

        samples = self._blink(pipeline, simulator, (960, 540), 90)
        assert all(s.holding for s in samples), "blink frames must be held"
        assert all((s.x, s.y) == before for s in samples), "held value must not move"

    def test_the_error_does_not_persist_after_the_eyes_reopen(self, simulator, estimator):
        pipeline = _pipeline(estimator)
        _stare(pipeline, simulator, (960, 540), frames=90)
        self._blink(pipeline, simulator, (960, 540), 90)
        points = _stare(pipeline, simulator, (960, 540), frames=8, start=96 / FPS)
        x, y = points[-1]
        assert np.hypot(x - 960, y - 540) < 80

    def test_without_the_hold_a_blink_throws_the_estimate(self, simulator, estimator):
        """Shows what the hold prevents: corrupt iris data reaching the model."""
        feature_filter = FeatureSmoother(**FEATURE_SMOOTHING_PRESETS["medium"])
        output_filter = GazeSmoother.from_preset("medium")

        def step(features, timestamp):
            from src.tracking.features import FeatureVector
            values = feature_filter.smooth(features.values, timestamp)
            result = estimator.estimate(FeatureVector(values=values, valid=True,
                                                      head_pose=features.head_pose))
            return output_filter.smooth(result.x, result.y, timestamp)

        for index in range(90):
            point = step(simulator.features((960, 540)), index / FPS)
        clean = np.hypot(point[0] - 960, point[1] - 540)
        for index in range(90, 96):
            point = step(simulator.blink_features((960, 540)), index / FPS)
        during = np.hypot(point[0] - 960, point[1] - 540)
        assert during > clean + 30, "the blink should visibly move it"

    def test_repeated_blinks_do_not_accumulate_error(self, simulator, estimator):
        """The reported fault: blinks making it 'more and more off'."""
        pipeline = _pipeline(estimator)
        frame, errors = 0, []
        for _cycle in range(8):
            points = _stare(pipeline, simulator, (960, 540), frames=45,
                            start=frame / FPS)
            frame += 45
            x, y = points[-1]
            errors.append(np.hypot(x - 960, y - 540))
            self._blink(pipeline, simulator, (960, 540), frame)
            frame += 6

        assert errors[-1] < 60, f"final error {errors[-1]:.0f} px"
        assert errors[-1] < errors[0] + 40, f"error grew across blinks: {errors}"

    def test_blinking_while_tilted_is_also_handled(self, simulator, estimator):
        head = HeadState(roll_deg=15)
        pipeline = _pipeline(estimator)
        _stare(pipeline, simulator, (960, 540), head, frames=90)
        samples = self._blink(pipeline, simulator, (960, 540), 90, head=head)
        assert all(s.holding for s in samples)


class TestAdaptiveBlinkThreshold:
    """One fixed openness threshold cannot fit every pair of eyes."""

    @pytest.mark.parametrize("openness", [1.0, 0.7, 0.5, 0.42, 0.35])
    def test_narrow_eyes_are_still_tracked(self, simulator, estimator, openness):
        head = HeadState(openness=openness)
        pipeline = _pipeline(estimator)
        samples = [pipeline.process_features(simulator.features((960, 540), head),
                                             index / FPS)
                   for index in range(120)]
        tracked = sum(1 for s in samples if s.valid and not s.holding)
        assert tracked > 100, f"openness x{openness} tracked only {tracked}/120 frames"

    @pytest.mark.parametrize("openness", [1.0, 0.7, 0.5, 0.42, 0.35])
    def test_blinks_are_still_detected_for_those_eyes(self, simulator, estimator,
                                                      openness):
        head = HeadState(openness=openness)
        pipeline = _pipeline(estimator)
        for index in range(120):
            pipeline.process_features(simulator.features((960, 540), head), index / FPS)
        closed = [pipeline.process_features(
            simulator.blink_features((960, 540), head), (120 + index) / FPS).eyes_closed
            for index in range(6)]
        assert all(closed), f"openness x{openness} missed the blink"

    def test_a_sustained_closure_stays_closed(self, simulator, estimator):
        """The learned baseline must not drift down to meet shut eyes."""
        pipeline = _pipeline(estimator)
        for index in range(200):
            sample = pipeline.process_features(
                simulator.blink_features((960, 540)), index / FPS)
        assert sample.eyes_closed

    def test_the_baseline_bootstraps_from_every_frame(self):
        """Learning only from frames already judged open deadlocks.

        A user whose open eyes read below the starting threshold would be
        called closed on frame one, never contribute an open sample, and never
        earn a baseline that lets them be seen as open.
        """
        detector = BlinkDetector(absolute_threshold=0.16)
        for _ in range(40):
            detector.update(0.11)
        assert detector.baseline == pytest.approx(0.11, abs=0.01)
        assert not detector.update(0.11), "narrow but open eyes read as closed"
        assert detector.update(0.02), "a real closure was missed"


class TestConfidence:
    """Confidence gates everything downstream, so it has to mean the same
    thing for every user and fall off for poses the model cannot handle."""

    @pytest.mark.parametrize("openness", [1.0, 0.6, 0.45, 0.38])
    def test_eye_shape_does_not_change_the_confidence(self, simulator, estimator,
                                                      openness):
        """Otherwise ``minimum_confidence`` discards narrow-eyed users entirely.

        They track fine; scored against a fixed openness threshold they were
        reported at zero confidence anyway, so every sample was thrown away
        downstream and the adaptive blink detection bought them nothing.
        """
        pipeline = _pipeline(estimator)
        head = HeadState(openness=openness)
        for index in range(120):
            sample = pipeline.process_features(
                simulator.features((960, 540), head), index / FPS)
        assert sample.confidence > 0.6, (
            f"openness x{openness} scored {sample.confidence:.2f}")

    def test_confidence_falls_off_for_extreme_poses(self, simulator, estimator):
        def score(head):
            pipeline = _pipeline(estimator)
            for index in range(120):
                sample = pipeline.process_features(
                    simulator.features((960, 540), head), index / FPS)
            return sample.confidence

        upright = score(HeadState())
        assert upright > 0.6
        assert score(HeadState(roll_deg=20)) < upright
        assert score(HeadState(roll_deg=40)) < 0.3, "a 40 deg tilt read as confident"
        assert score(HeadState(yaw_deg=35)) < 0.3


class TestTrackLossRecovery:
    """Losing the face for a moment must not poison what comes next."""

    def test_tracking_resumes_immediately_after_a_long_gap(self, simulator, estimator):
        pipeline = _pipeline(estimator)
        _stare(pipeline, simulator, (400, 300), frames=90)

        # Two seconds in which no frame is processed at all.
        resume = 90 / FPS + 2.0
        settled = None
        for index in range(30):
            sample = pipeline.process_features(
                simulator.features((1500, 800)), resume + index / FPS)
            if sample.valid and np.hypot(sample.x - 1500, sample.y - 800) < 90:
                settled = index + 1
                break
        assert settled is not None and settled <= 2, (
            f"took {settled} frames to re-lock after the gap")

    def test_stale_filters_are_what_made_that_slow(self, simulator, estimator):
        """Keeping the filter state across the gap drags the estimate back."""
        pipeline = _pipeline(estimator)
        pipeline.reacquire_seconds = 1e9   # never treat the gap as a gap
        _stare(pipeline, simulator, (400, 300), frames=90)
        resume = 90 / FPS + 2.0
        settled = None
        for index in range(40):
            sample = pipeline.process_features(
                simulator.features((1500, 800)), resume + index / FPS)
            if sample.valid and np.hypot(sample.x - 1500, sample.y - 800) < 90:
                settled = index + 1
                break
        assert settled is None or settled > 2

    def test_a_short_gap_keeps_its_state(self, simulator, estimator):
        """A dropped frame or two is not a track loss; resetting would jump."""
        pipeline = _pipeline(estimator)
        _stare(pipeline, simulator, (960, 540), frames=90)
        before = pipeline._last_valid
        pipeline.process_features(simulator.features((960, 540)), 90 / FPS + 0.1)
        assert pipeline._last_valid is not None
        assert np.hypot(pipeline._last_valid[0] - before[0],
                        pipeline._last_valid[1] - before[1]) < 40


class TestFaceFrame:
    """The rigid fit is what the rest of the accuracy rests on."""

    def test_the_reference_is_far_steadier_than_two_corners(self, simulator):
        """71% of the variance in an iris offset used to come from here."""
        corners, fitted = [], []
        for _ in range(200):
            marks = simulator.landmarks((960, 540), HeadState())
            corners.append(0.5 * (marks.pixel(fl.EYE_LEFT_OUTER)
                                  + marks.pixel(fl.EYE_LEFT_INNER)))
            fitted.append(fit_face_frame(marks).eye_centre("left"))
        corner_noise = float(np.linalg.norm(np.std(corners, axis=0)))
        frame_noise = float(np.linalg.norm(np.std(fitted, axis=0)))
        assert frame_noise < 0.6 * corner_noise, (
            f"corners {corner_noise:.3f} px, frame {frame_noise:.3f} px")

    def test_the_scale_is_steadier_still(self, simulator):
        widths, fitted = [], []
        for _ in range(200):
            marks = simulator.landmarks((960, 540), HeadState())
            widths.append(float(np.linalg.norm(marks.pixel(fl.EYE_LEFT_INNER)
                                               - marks.pixel(fl.EYE_LEFT_OUTER))))
            fitted.append(fit_face_frame(marks).eye_width_px)
        assert np.std(fitted) < 0.25 * np.std(widths)

    @pytest.mark.parametrize("shape_mm,limit", [(0.0, 2.0), (3.0, 4.0), (6.0, 6.0)])
    def test_the_bias_is_nearly_rigid(self, shape_mm, limit):
        """A canonical head is not the user's, and that is fine.

        What matters is that the mismatch behaves like a fixed offset on the
        head -- which calibration absorbs, as it does every other per-user
        constant -- rather than shifting as the head turns, which it could not.

        It is not perfectly rigid, and the limits say how far it moves: about
        1.3 px across poses for a subject who matches the canonical head, and
        growing with the mismatch. The residue is a smooth function of pose,
        and pose is itself a model input, so the fit can account for what is
        left.
        """
        poses = (HeadState(), HeadState(roll_deg=20), HeadState(roll_deg=-20),
                 HeadState(yaw_deg=15), HeadState(z=520), HeadState(x=-60))
        for seed in range(4):
            clean = EyeSimulator(noise_px=0.0, shape_mm=shape_mm,
                                 rng=np.random.default_rng(seed))
            biases = []
            for head in poses:
                frame = fit_face_frame(clean.landmarks((960, 540), head))
                offset = (frame.eye_centre("left")
                          - clean.project(clean.eye_centre_world(head, "left")))
                axis = frame.eye_axis("left")
                # Resolved in the eye's own axes, a rigid bias reads the same
                # in every pose however the head is turned.
                biases.append([float(offset @ axis),
                               float(offset @ np.array([-axis[1], axis[0]]))])
            spread = np.ptp(np.array(biases), axis=0)
            assert np.all(spread < limit), (
                f"subject {seed}: bias moved by {spread.round(2)} px across poses")

    def test_an_unusual_face_is_still_accepted(self):
        """The gate is for landmarks that are not a face, not for faces that
        are not the average."""
        for shape in (0.0, 3.0, 6.0, 9.0):
            sim = EyeSimulator(noise_px=0.35, shape_mm=shape,
                               rng=np.random.default_rng(2))
            frame = fit_face_frame(sim.landmarks((960, 540), HeadState()))
            assert frame.valid, f"a face {shape} mm from canonical was rejected"

    def test_nonsense_landmarks_are_rejected(self, simulator):
        marks = simulator.landmarks((960, 540), HeadState())
        scrambled = marks.points.copy()
        rng = np.random.default_rng(0)
        for index in fl.RIGID_FACE_LANDMARKS:
            scrambled[index, :2] = rng.random(2)
        assert not fit_face_frame(FaceLandmarks(
            scrambled, marks.frame_width, marks.frame_height, True)).valid

    def test_the_frame_survives_every_pose_it_will_meet(self, simulator):
        for head in (HeadState(), HeadState(yaw_deg=35), HeadState(pitch_deg=25),
                     HeadState(roll_deg=30), HeadState(z=750), HeadState(z=450)):
            assert fit_face_frame(simulator.landmarks((960, 540), head)).valid


class TestGazeRay:
    """Turning the iris displacement back into an angle before fitting."""

    def test_the_ray_is_linear_in_screen_position(self):
        """A sphere gives sin(theta); a screen wants tan(theta)."""
        clean = EyeSimulator(noise_px=0.0, shape_mm=0.0)
        xs = np.array([200, 500, 960, 1400, 1700], dtype=float)
        offsets = np.array([clean.features((x, 540)).values["iris_mean_ix"] for x in xs])
        rays = np.array([clean.features((x, 540)).values["ray_mean_x"] for x in xs])

        def straightness(values):
            fit = np.polyfit(values, xs, 1)
            return float(np.abs(xs - np.polyval(fit, values)).max())

        assert straightness(rays) < 0.2 * straightness(offsets)
        assert straightness(rays) < 2.0, "the ray should be almost exactly linear"

    def test_the_ray_stays_finite_when_the_geometry_cannot_hold(self):
        """Landmarks can put the iris further out than an eyeball allows."""
        from src.tracking.features import _gaze_ray
        for offset in (0.0, 0.2, 0.5, 5.0, -5.0):
            x, y = _gaze_ray(offset, offset)
            assert np.isfinite(x) and np.isfinite(y)

    def test_both_frames_are_kept_because_the_radius_is_a_guess(self, simulator):
        """The eyeball radius is a constant, and real eyes vary around it.

        Rays alone degrade badly when the assumed radius is too small, because
        the correction then bends the wrong way. Keeping the raw offsets
        alongside gives the fit something unaffected to lean on.
        """
        names = set(FEATURES)
        assert {"ray_l_x", "ray_l_y"} <= names
        assert {"iris_l_ix", "iris_l_iy"} <= names


class TestRobustFit:
    """A glance away during calibration looks perfectly normal in features."""

    @staticmethod
    def _contaminated(rng, n=240, bad=20):
        design = polynomial_expand(rng.normal(0, 1, (n, 2)), 2)
        truth = design @ rng.normal(0, 1, (design.shape[1], 2))
        observed = truth + rng.normal(0, 5, (n, 2))
        observed[:bad] += 800.0            # frames aimed somewhere else
        return design, observed, truth

    def test_outliers_barely_move_the_robust_fit(self):
        rng = np.random.default_rng(0)
        design, observed, truth = self._contaminated(rng)
        plain = np.hypot(*(design[20:] @ ridge_fit(design, observed, 1.0)
                           - truth[20:]).T).mean()
        robust = np.hypot(*(design[20:] @ robust_ridge_fit(design, observed, 1.0)
                            - truth[20:]).T).mean()
        assert robust < 0.3 * plain, f"plain {plain:.1f}, robust {robust:.1f}"

    def test_it_costs_little_on_clean_data(self):
        """Downweighting is not free, but it is cheap.

        On this deliberately harsh synthetic case -- few samples, heavy noise --
        it costs about 12% and never more than 40%. On a real calibration,
        where there are hundreds of samples per fit, the end-to-end difference
        measured 21.9 px against 21.7 px.
        """
        ratios = []
        for seed in range(12):
            rng = np.random.default_rng(seed)
            design, observed, truth = self._contaminated(rng, bad=0)
            plain = np.hypot(*(design @ ridge_fit(design, observed, 1.0) - truth).T).mean()
            robust = np.hypot(*(design @ robust_ridge_fit(design, observed, 1.0)
                                - truth).T).mean()
            ratios.append(robust / plain)
        assert np.mean(ratios) < 1.2, f"mean cost {np.mean(ratios):.3f}"
        assert np.max(ratios) < 1.45, f"worst cost {np.max(ratios):.3f}"

    def test_a_calibration_with_glances_still_works(self, simulator):
        """The end-to-end case: the user looks away for a few frames a point."""
        rng = np.random.default_rng(7)
        session = CalibrationSession(FEATURES, SCREEN, alphas=ALPHAS)
        targets = pattern_to_pixels(calibration_pattern("13point"), *SCREEN)
        for point_id, target in enumerate(targets):
            for _ in range(26):
                head = HeadState().shifted(dx=rng.normal(0, 28), dz=rng.normal(0, 35),
                                           droll=rng.normal(0, 8))
                looked_at = target
                if rng.random() < 0.08:
                    looked_at = (rng.uniform(0, SCREEN[0]), rng.uniform(0, SCREEN[1]))
                session.add(point_id, target, simulator.features(looked_at, head))
        estimator = session.fit()
        assert _mean_error(estimator, simulator, HeadState()) < 60


class TestSpikeRejection:
    """Frames where the iris landmarks are simply wrong.

    Not a blink -- the eye is open and every other landmark is fine -- so
    nothing upstream flags them. At 30 fps a reflection off glasses or a frame
    of motion blur lasts one or two frames, so both cases are tested.
    """

    SPIKES = (120, 121, 180, 240, 241, 300)

    def _run(self, estimator, simulator, prefilter: bool):
        pipeline = _pipeline(estimator)
        if not prefilter:
            pipeline.feature_prefilter.window = 1
        worst, settled = 0.0, []
        for index in range(380):
            features = (simulator.spoiled_features((960, 540)) if index in self.SPIKES
                        else simulator.features((960, 540)))
            sample = pipeline.process_features(features, index / FPS)
            if sample.valid and index > 60:
                error = np.hypot(sample.x - 960, sample.y - 540)
                worst = max(worst, error)
                if index > 320:
                    settled.append(error)
        return worst, float(np.mean(settled))

    def test_a_spike_no_longer_throws_the_estimate(self, simulator, estimator):
        without, _ = self._run(estimator, simulator, prefilter=False)
        with_it, _ = self._run(estimator, simulator, prefilter=True)
        assert without > 200, f"the spike should be visible without the median: {without:.0f}"
        assert with_it < 60, f"worst excursion {with_it:.0f} px"

    def test_the_window_is_long_enough_for_consecutive_bad_frames(self,
                                                                  simulator, estimator):
        """Why the default window is five rather than three.

        A median of ``n`` absorbs a burst of ``(n - 1) / 2`` bad frames and
        then fails abruptly. Two consecutive bad frames -- what a reflection
        off glasses or a frame of motion blur lasts at 30 fps -- get past a
        three-tap median whenever the two happen to err the same way, so its
        typical excursion looks survivable and its bad case does not.

        Measured on the distribution rather than on one draw, because which
        case you get is a coin toss.
        """
        pipeline = _pipeline(estimator)
        assert pipeline.feature_prefilter.window >= 5

        def excursions(window, burst, trials=12):
            peaks = []
            for _ in range(trials):
                pipe = _pipeline(estimator)
                pipe.feature_prefilter.window = window
                spikes = set(range(90, 90 + burst))
                peak = 0.0
                for index in range(170):
                    features = (simulator.spoiled_features((960, 540))
                                if index in spikes else simulator.features((960, 540)))
                    sample = pipe.process_features(features, index / FPS)
                    if sample.valid and index > 60:
                        peak = max(peak, np.hypot(sample.x - 960, sample.y - 540))
                peaks.append(peak)
            return np.array(peaks)

        pair_at_three = excursions(3, burst=2)
        pair_at_five = excursions(5, burst=2)
        assert pair_at_three.max() > 150, (
            f"a pair should sometimes defeat a three-tap median: "
            f"worst {pair_at_three.max():.0f} px")
        assert pair_at_five.max() < 60, (
            f"a five-tap median should absorb every pair: "
            f"worst {pair_at_five.max():.0f} px")

    def test_the_median_costs_almost_nothing_when_settled(self, simulator, estimator):
        _, without = self._run(estimator, simulator, prefilter=False)
        _, with_it = self._run(estimator, simulator, prefilter=True)
        assert with_it < without + 3.0, f"{without:.1f} px -> {with_it:.1f} px"

    def test_one_euro_alone_amplifies_a_spike(self):
        """Why a median is needed at all.

        One Euro widens its cutoff in proportion to the signal's speed, and a
        one-frame spike is the fastest thing it ever sees, so it opens up and
        follows it -- and stays open afterwards while its speed estimate decays.
        """
        from src.tracking.smoother import OneEuroFilter
        filt = OneEuroFilter(**FEATURE_SMOOTHING_PRESETS["medium"])
        for index in range(30):
            steady = filt.filter(0.10, index / FPS)
        spiked = filt.filter(0.95, 30 / FPS)
        assert spiked - steady > 0.3, "the spike passed through barely filtered"


class TestCalibrationFiltering:
    def test_head_features_are_exempt_from_outlier_rejection(self):
        """Head movement during calibration is data, not noise."""
        columns = gaze_feature_columns(FEATURES)
        assert set(columns) == {0, 1, 2, 3, 4, 5, 6, 7}, \
            "only the eye features are checked"

        rng = np.random.default_rng(5)
        samples = np.zeros((30, len(FEATURES)))
        samples[:, :8] = rng.normal(0, 0.01, (30, 8))     # steady eyes
        samples[:, 9:13] = rng.normal(0, 0.5, (30, 4))    # head moving a lot
        mask = robust_filter(samples, columns=columns)
        assert mask.sum() >= 28, "deliberate head movement was thrown away"

    def test_blink_frames_are_still_rejected(self):
        columns = gaze_feature_columns(FEATURES)
        rng = np.random.default_rng(6)
        samples = np.zeros((30, len(FEATURES)))
        samples[:, :8] = rng.normal(0, 0.01, (30, 8))
        samples[7, :8] = [0.9, -0.8, 0.9, -0.8, 2.4, -2.1, 2.4, -2.1]  # a blink
        mask = robust_filter(samples, columns=columns)
        assert not mask[7]

    def test_a_varied_calibration_keeps_most_samples(self, simulator):
        rng = np.random.default_rng(12)
        session = CalibrationSession(FEATURES, SCREEN, alphas=ALPHAS)
        for point_id, target in enumerate(pattern_to_pixels(
                calibration_pattern("13point"), *SCREEN)):
            for _ in range(24):
                head = HeadState().shifted(dx=rng.normal(0, 25), dyaw=rng.normal(0, 4),
                                           droll=rng.normal(0, 8))
                session.add(point_id, target, simulator.features(target, head))
        features, _targets, groups = session.build_matrices()
        assert len(np.unique(groups)) == 13, "points were dropped"
        assert features.shape[0] > 13 * 24 * 0.85

    def test_narrow_eyes_can_calibrate_at_all(self, simulator):
        """A fixed openness gate rejected every frame from some users."""
        session = CalibrationSession(FEATURES, SCREEN, alphas=ALPHAS)
        head = HeadState(openness=0.40)
        accepted = sum(session.add(0, (960.0, 540.0),
                                   simulator.features((960, 540), head))
                       for _ in range(60))
        assert accepted > 40, f"only {accepted}/60 frames accepted"


class TestCalibrationCoverage:
    def test_posture_cues_cover_both_tilt_directions(self):
        cues = [posture_cue(i).lower() for i in range(13)]
        assert sum("left" in c for c in cues) >= 3
        assert sum("right" in c for c in cues) >= 3
        assert any("closer" in c or "back" in c for c in cues)

    def test_a_varied_calibration_reports_full_coverage(self, simulator):
        rng = np.random.default_rng(4)
        session = CalibrationSession(FEATURES, SCREEN, alphas=ALPHAS)
        for point_id, target in enumerate(pattern_to_pixels(
                calibration_pattern("13point"), *SCREEN)):
            for _ in range(20):
                head = HeadState().shifted(dx=rng.normal(0, 28), dz=rng.normal(0, 35),
                                           droll=rng.normal(0, 8))
                session.add(point_id, target, simulator.features(target, head))
        coverage = session.coverage()
        assert set(coverage) <= set(COVERAGE_TARGETS)
        assert all(v >= 1.0 for v in coverage.values()), coverage
        assert session.coverage_warnings() == []

    def test_a_rigid_calibration_is_flagged(self, simulator):
        session = CalibrationSession(FEATURES, SCREEN, alphas=ALPHAS)
        for point_id, target in enumerate(pattern_to_pixels(
                calibration_pattern("13point"), *SCREEN)):
            for _ in range(20):
                session.add(point_id, target, simulator.features(target))
        warnings = session.coverage_warnings()
        assert warnings, "a perfectly still calibration should be flagged"
        assert any("tilt" in w for w in warnings)


class TestFitReport:
    def test_frame_and_settled_errors_are_both_reported(self, estimator):
        report = estimator.report
        assert report.settled_error_px > 0
        # Averaging held-out frames cancels the noise, so the settled figure is
        # always the smaller of the two; reporting only it flatters the model.
        assert report.settled_error_px < report.mean_error_px
        assert len(report.per_point_error_px) == report.n_points

    def test_the_report_survives_a_round_trip(self, estimator):
        from src.tracking.gaze_estimator import FitReport
        restored = FitReport.from_dict(estimator.report.to_dict())
        assert restored.settled_error_px == pytest.approx(estimator.report.settled_error_px)
        assert restored.quality() == estimator.report.quality()

    def test_the_input_box_stays_open_on_axes_calibration_barely_saw(self):
        """A feature that never varied must not collapse to a single point."""
        from src.tracking.gaze_estimator import RidgeGazeEstimator
        names = ["iris_l_ix", "eye_tilt"]
        raw = np.column_stack([np.linspace(-0.1, 0.1, 20), np.full(20, 0.3)])
        low, high = RidgeGazeEstimator.input_range(raw, names)
        assert high[1] - low[1] > 0.1, "a constant feature was pinned"
        assert low[1] < 0.3 < high[1]
