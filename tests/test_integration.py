"""End-to-end pipeline test.

This walks the whole chain the application depends on, with synthetic input
and no hardware:

    features -> calibration fit -> gaze estimate -> smoothing -> classification
             -> state machine -> event aggregation -> database -> export

It is the logical equivalent of the manual acceptance test in the README: look
at squares, look at the clock, look away, stop, read the session back.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest

from src.events.detector import EventDetector, compute_statistics
from src.events.models import AttentionState, Session
from src.events.state_machine import AttentionStateMachine, TrackerObservation
from src.screen.classifier import GazeRegionClassifier
from src.screen.regions import KIND_CLOCK, Region, RegionManager
from src.storage.database import Database
from src.storage.export import export_events_csv, export_session_json
from src.tracking.calibration import (CalibrationSession, CalibrationStore,
                                      calibration_pattern, pattern_to_pixels)
from src.tracking.smoother import GazeSmoother
from src.utils.geometry import Rect, square_rect

from tests.test_calibration import FEATURES, SCREEN, synthetic_features

BOARD = Rect(400, 100, 800, 800)
FRAME = 1.0 / 30.0


@pytest.fixture(scope="module")
def estimator():
    """Calibrate once from synthetic samples, as the real app does at startup."""
    rng = np.random.default_rng(21)
    targets = pattern_to_pixels(calibration_pattern("13point"), *SCREEN)
    session = CalibrationSession(FEATURES, SCREEN)
    for point_id, target in enumerate(targets):
        for _ in range(15):
            session.add(point_id, target,
                        synthetic_features(target[0], target[1], 0.004, rng))
    return session.fit()


@pytest.fixture()
def classifier():
    regions = RegionManager()
    regions.set_board(BOARD)
    regions.add(Region("White Clock", 1250, 700, 220, 70, priority=50, kind=KIND_CLOCK))
    return GazeRegionClassifier(regions, Rect(0, 0, *SCREEN), off_screen_margin=60)


class TestGazeToSquare:
    def test_looking_at_a_square_resolves_to_that_square(self, estimator, classifier):
        """The headline capability: look at e4, get 'e4'."""
        hits = 0
        squares = ["a8", "e4", "d5", "h1", "c6", "f3"]
        for square in squares:
            centre = square_rect(square, BOARD.x, BOARD.y, BOARD.width, BOARD.height).center
            result = estimator.estimate(synthetic_features(*centre))
            assert result.valid
            classification = classifier.classify(result.x, result.y)
            if classification.square == square:
                hits += 1
        # With ~60 px of modelled error against 100 px squares, most but not
        # every square is expected to land exactly.
        assert hits >= len(squares) - 1

    def test_estimates_stay_near_the_true_position(self, estimator):
        rng = np.random.default_rng(22)
        errors = []
        for _ in range(50):
            x = rng.uniform(100, SCREEN[0] - 100)
            y = rng.uniform(100, SCREEN[1] - 100)
            result = estimator.estimate(synthetic_features(x, y, 0.003, rng))
            errors.append(np.hypot(result.x - x, result.y - y))
        assert np.mean(errors) < 90.0

    def test_smoothing_reduces_jitter_without_losing_the_target(self, estimator):
        rng = np.random.default_rng(23)
        smoother = GazeSmoother.from_preset("medium")
        target = square_rect("e4", BOARD.x, BOARD.y, BOARD.width, BOARD.height).center
        raw_points, smooth_points = [], []
        for i in range(90):
            result = estimator.estimate(synthetic_features(*target, 0.006, rng))
            raw_points.append((result.x, result.y))
            smooth_points.append(smoother.smooth(result.x, result.y, i * FRAME))

        raw = np.array(raw_points[30:])
        smooth = np.array(smooth_points[30:])
        assert smooth.std(axis=0).mean() < raw.std(axis=0).mean()
        assert np.hypot(*(smooth.mean(axis=0) - np.array(target))) < 90.0


class TestFullSession:
    def test_a_recorded_session_round_trips_through_the_database(
            self, estimator, classifier, tmp_path):
        rng = np.random.default_rng(24)
        machine = AttentionStateMachine(away_threshold_ms=500, region_dwell_ms=200,
                                        square_dwell_ms=250, minimum_confidence=0.4)
        detector = EventDetector(machine, min_event_ms=120)
        smoother = GazeSmoother.from_preset("low")

        database = Database(tmp_path / "integration.db")
        database.connect()
        session = Session("2026-09-03_13-42-10", datetime(2026, 9, 3, 13, 42, 10), 0.0)
        database.create_session(session)

        def run(seconds: float, point, start: float, *, on_screen=True):
            """Simulate looking at a screen point (or away) for a duration."""
            t = start
            while t < start + seconds:
                if on_screen:
                    result = estimator.estimate(synthetic_features(*point, 0.004, rng))
                    x, y = smoother.smooth(result.x, result.y, t)
                    classification = classifier.classify(x, y)
                    confidence = 0.85
                else:
                    x, y = -500.0, -500.0
                    classification = classifier.classify(x, y)
                    confidence = 0.6
                observation = TrackerObservation(t, True, False, confidence, True,
                                                 True, classification)
                event = detector.process(observation, x, y)
                if event is not None:
                    database.insert_event(session.db_id, event, 0.0)
                t += FRAME
            return t

        e4 = square_rect("e4", BOARD.x, BOARD.y, BOARD.width, BOARD.height).center
        t = run(3.0, e4, 0.0)
        t = run(2.0, (1360, 735), t)          # the clock
        t = run(2.5, None, t, on_screen=False)  # look away
        t = run(2.0, e4, t)
        final = detector.flush(t)
        if final is not None:
            database.insert_event(session.db_id, final, 0.0)

        session.end_monotonic = t
        session.end_wall_clock = datetime(2026, 9, 3, 13, 42, 20)
        database.finish_session(session)

        # --- read it all back -----------------------------------------
        events = database.get_events(session.db_id)
        stats = compute_statistics(events, duration=t)
        states = {event.state for event in events}

        assert AttentionState.LOOKING_AT_BOARD in states
        assert AttentionState.LOOKING_AT_CLOCK in states
        assert AttentionState.LOOKING_AWAY in states

        assert stats.away_event_count == 1
        assert 1.5 < stats.away_longest_seconds < 2.6
        assert stats.bucket_seconds["board"] > 3.5
        assert stats.bucket_seconds["clock"] > 1.0
        assert stats.square_seconds, "board time should be attributed to squares"
        assert sum(stats.bucket_seconds.values()) == pytest.approx(t, abs=0.6)

        # --- and export it --------------------------------------------
        csv_path = export_events_csv(tmp_path / "events.csv", events)
        assert csv_path.read_text(encoding="utf-8").count("\n") == len(events) + 1

        json_path = export_session_json(tmp_path / "session.json",
                                        session.to_dict(), events)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["statistics"]["away_event_count"] == 1
        assert len(payload["events"]) == len(events)

        database.close()


class TestCalibrationPersistence:
    def test_a_saved_profile_produces_identical_estimates(self, estimator, tmp_path):
        store = CalibrationStore(tmp_path)
        session = CalibrationSession(FEATURES, SCREEN)
        profile = session.to_profile(estimator, monitor_index=1, camera_index=0,
                                     pattern="13point")
        store.save(profile)

        restored = store.load(profile.profile_id)
        assert restored is not None
        assert restored.matches_screen(*SCREEN)

        probe = synthetic_features(1000, 600)
        original = estimator.estimate(probe)
        again = restored.build_estimator().estimate(probe)
        assert (again.x, again.y) == pytest.approx((original.x, original.y))

    def test_latest_returns_the_newest_profile(self, estimator, tmp_path):
        store = CalibrationStore(tmp_path)
        session = CalibrationSession(FEATURES, SCREEN)
        for created in ("2026-01-01T10:00:00", "2026-06-01T10:00:00"):
            profile = session.to_profile(estimator, 1, 0, "13point")
            profile.created_at = created
            profile.name = created
            store.save(profile)
        assert store.latest().name == "2026-06-01T10:00:00"
        assert len(store.list_profiles()) == 2

    def test_a_corrupt_profile_is_skipped_not_fatal(self, tmp_path):
        (tmp_path / "broken.json").write_text("{oops", encoding="utf-8")
        store = CalibrationStore(tmp_path)
        assert store.list_profiles() == []
        assert store.load("broken") is None
