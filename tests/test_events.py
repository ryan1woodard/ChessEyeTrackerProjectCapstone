"""State machine and event aggregation tests.

The scenarios below are written as timed sequences of synthetic observations,
which is how the real system sees the world: a stream of frames at ~30 Hz.
"""

from __future__ import annotations

import pytest

from src.events.detector import EventDetector, compute_statistics
from src.events.models import AttentionState, GazeEvent, format_duration
from src.events.state_machine import AttentionStateMachine, TrackerObservation
from src.screen.classifier import GazeClassification
from src.screen.regions import KIND_BOARD, KIND_CLOCK

FRAME = 1.0 / 30.0

BOARD_E4 = GazeClassification("Chessboard", "e4", KIND_BOARD, True, True)
BOARD_D5 = GazeClassification("Chessboard", "d5", KIND_BOARD, True, True)
CLOCK = GazeClassification("White Clock", None, KIND_CLOCK, True, False)
OFFSCREEN = GazeClassification("Off screen", None, "away", False, False)


def observation(t: float, classification=BOARD_E4, *, face=True, eyes_closed=False,
                confidence=0.9, valid=True, calibrated=True) -> TrackerObservation:
    return TrackerObservation(t, face, eyes_closed, confidence, valid, calibrated,
                              classification)


def feed(target, start: float, seconds: float, **kwargs):
    """Push observations at 30 Hz for a duration; returns the end timestamp."""
    events = []
    t = start
    end = start + seconds
    while t < end:
        result = target.process(observation(t, **kwargs)) \
            if isinstance(target, EventDetector) else target.update(observation(t, **kwargs))
        if result is not None:
            events.append(result)
        t += FRAME
    return t, events


@pytest.fixture()
def machine() -> AttentionStateMachine:
    return AttentionStateMachine(away_threshold_ms=500, region_dwell_ms=200,
                                 square_dwell_ms=250, eyes_closed_ms=1200,
                                 minimum_confidence=0.45)


class TestStateMachine:
    def test_starts_unknown(self, machine):
        assert machine.state is AttentionState.UNKNOWN

    def test_commits_after_the_dwell_time(self, machine):
        t, _ = feed(machine, 0.0, 0.15)
        assert machine.state is AttentionState.UNKNOWN, "200 ms dwell not reached yet"
        feed(machine, t, 0.2)
        assert machine.state is AttentionState.LOOKING_AT_BOARD

    def test_a_single_bad_frame_does_not_cause_a_look_away(self, machine):
        t, _ = feed(machine, 0.0, 1.0)
        assert machine.state is AttentionState.LOOKING_AT_BOARD
        machine.update(observation(t, OFFSCREEN))
        machine.update(observation(t + FRAME, BOARD_E4))
        assert machine.state is AttentionState.LOOKING_AT_BOARD

    def test_sustained_absence_becomes_a_look_away(self, machine):
        t, _ = feed(machine, 0.0, 1.0)
        feed(machine, t, 0.9, classification=OFFSCREEN)
        assert machine.state is AttentionState.LOOKING_AWAY

    def test_a_blink_never_becomes_a_look_away(self, machine):
        """A 200 ms eye closure must leave the committed state untouched."""
        t, _ = feed(machine, 0.0, 1.0)
        t, _ = feed(machine, t, 0.2, eyes_closed=True)
        assert machine.state is AttentionState.LOOKING_AT_BOARD
        feed(machine, t, 0.3)
        assert machine.state is AttentionState.LOOKING_AT_BOARD

    def test_a_long_closure_does_commit(self, machine):
        t, _ = feed(machine, 0.0, 1.0)
        feed(machine, t, 1.5, eyes_closed=True)
        assert machine.state is AttentionState.BLINKING

    def test_missing_face_becomes_face_not_detected(self, machine):
        t, _ = feed(machine, 0.0, 1.0)
        feed(machine, t, 0.8, face=False)
        assert machine.state is AttentionState.FACE_NOT_DETECTED
        assert machine.state.is_away

    def test_low_confidence_is_reported_separately(self, machine):
        t, _ = feed(machine, 0.0, 1.0)
        feed(machine, t, 0.8, confidence=0.1)
        assert machine.state is AttentionState.LOW_CONFIDENCE

    def test_uncalibrated_stays_unknown(self, machine):
        feed(machine, 0.0, 1.0, calibrated=False)
        assert machine.state is AttentionState.UNKNOWN

    def test_square_change_needs_the_square_dwell(self, machine):
        t, _ = feed(machine, 0.0, 1.0)
        t, _ = feed(machine, t, 0.15, classification=BOARD_D5)
        assert machine.snapshot.square == "e4", "250 ms square dwell not reached"
        feed(machine, t, 0.2, classification=BOARD_D5)
        assert machine.snapshot.square == "d5"

    def test_board_to_clock_uses_the_region_dwell(self, machine):
        t, _ = feed(machine, 0.0, 1.0)
        feed(machine, t, 0.3, classification=CLOCK)
        assert machine.state is AttentionState.LOOKING_AT_CLOCK

    def test_reset(self, machine):
        feed(machine, 0.0, 1.0)
        machine.reset()
        assert machine.state is AttentionState.UNKNOWN


class TestEventAggregation:
    @pytest.fixture()
    def detector(self, machine) -> EventDetector:
        return EventDetector(machine, min_event_ms=120)

    def test_a_long_fixation_is_one_event_not_forty(self, detector):
        t, events = feed(detector, 0.0, 1.4)
        detector.flush(t)
        board_events = [e for e in events if e.state is AttentionState.LOOKING_AT_BOARD]
        assert len(board_events) <= 1
        assert detector.current_event is None or detector.current_event.sample_count > 30

    def test_event_duration_and_averages(self, detector):
        t = 0.0
        while t < 1.5:
            detector.process(observation(t), 700.0, 400.0)
            t += FRAME
        while t < 3.0:
            detector.process(observation(t, CLOCK), 1300.0, 720.0)
            t += FRAME
        final = detector.flush(t)

        assert final is not None
        assert final.state is AttentionState.LOOKING_AT_CLOCK
        assert final.avg_gaze_x == pytest.approx(1300.0, abs=1.0)
        assert final.duration > 1.0

    def test_look_away_event_duration_is_recorded(self, detector):
        t, _ = feed(detector, 0.0, 1.5)
        t, _ = feed(detector, t, 2.2, classification=OFFSCREEN)
        _, events = feed(detector, t, 1.0)
        detector.flush(t + 1.0)
        away = [e for e in events if e.state is AttentionState.LOOKING_AWAY]
        assert len(away) == 1
        # The event starts when the state commits, i.e. after the 500 ms threshold.
        assert 1.4 < away[0].duration < 2.0

    def test_events_below_the_minimum_are_dropped(self, machine):
        detector = EventDetector(machine, min_event_ms=5000)
        t, events = feed(detector, 0.0, 2.0)
        t, more = feed(detector, t, 2.0, classification=CLOCK)
        assert events + more == []

    def test_reset_clears_everything(self, detector):
        feed(detector, 0.0, 1.0)
        detector.reset()
        assert detector.current_event is None
        assert detector.state is AttentionState.UNKNOWN


class TestStatistics:
    def _events(self):
        return [
            GazeEvent(0.0, 10.0, AttentionState.LOOKING_AT_BOARD, "Chessboard", "e4",
                      confidence=0.9, sample_count=300),
            GazeEvent(10.0, 12.0, AttentionState.LOOKING_AT_BOARD, "Chessboard", "d4",
                      confidence=0.8, sample_count=60),
            GazeEvent(12.0, 14.0, AttentionState.LOOKING_AT_CLOCK, "White Clock",
                      confidence=0.7, sample_count=60),
            GazeEvent(14.0, 17.0, AttentionState.LOOKING_AWAY, confidence=0.2,
                      sample_count=90),
            GazeEvent(17.0, 18.0, AttentionState.FACE_NOT_DETECTED, confidence=0.0,
                      sample_count=30),
            GazeEvent(18.0, 20.0, AttentionState.LOOKING_AT_BOARD, "Chessboard", "e4",
                      confidence=0.85, sample_count=60),
        ]

    def test_buckets_sum_to_the_duration(self):
        stats = compute_statistics(self._events())
        assert stats.duration == pytest.approx(20.0)
        assert sum(stats.bucket_seconds.values()) == pytest.approx(20.0)
        assert stats.bucket_seconds["board"] == pytest.approx(14.0)
        assert stats.bucket_percent("board") == pytest.approx(70.0)

    def test_away_metrics_include_missing_face(self):
        stats = compute_statistics(self._events())
        assert stats.away_event_count == 2
        assert stats.away_total_seconds == pytest.approx(4.0)
        assert stats.away_longest_seconds == pytest.approx(3.0)
        assert stats.away_average_seconds == pytest.approx(2.0)

    def test_square_time_accumulates_across_visits(self):
        stats = compute_statistics(self._events())
        assert stats.square_seconds["e4"] == pytest.approx(12.0)
        assert stats.top_squares(1) == [("e4", 12.0)]

    def test_confidence_is_duration_weighted(self):
        stats = compute_statistics(self._events())
        assert 0.6 < stats.mean_confidence < 0.8

    def test_empty_input(self):
        stats = compute_statistics([], duration=5.0)
        assert stats.duration == 5.0
        assert stats.event_count == 0
        assert stats.bucket_percent("board") == 0.0


class TestFormatting:
    @pytest.mark.parametrize("seconds,expected", [
        (0, "0:00"), (9.4, "0:09"), (71, "1:11"), (1271, "21:11"), (3671, "1:01:11"),
        (-5, "0:00"),
    ])
    def test_duration_formatting(self, seconds, expected):
        assert format_duration(seconds) == expected
