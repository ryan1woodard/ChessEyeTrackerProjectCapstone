"""Event aggregation.

A webcam produces ~30 samples a second; writing one database row per sample
would generate tens of thousands of near-identical rows per game and tell you
nothing. Instead consecutive frames that share a committed state are collapsed
into a single :class:`GazeEvent` with a duration, so "looked at e4 for 1.4 s"
is one row rather than forty.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

from .models import (AttentionState, GazeEvent, SessionStatistics, SUMMARY_BUCKETS)
from .state_machine import AttentionStateMachine, StateSnapshot, TrackerObservation

logger = logging.getLogger(__name__)


class EventDetector:
    """Drives the state machine and turns committed states into events."""

    def __init__(self, state_machine: AttentionStateMachine,
                 min_event_ms: float = 120.0) -> None:
        self.state_machine = state_machine
        self.min_event_seconds = min_event_ms / 1000.0
        self._current: Optional[GazeEvent] = None
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_conf = 0.0

    # ---------------------------------------------------------------- state
    @property
    def current_event(self) -> Optional[GazeEvent]:
        return self._current

    @property
    def state(self) -> AttentionState:
        return self.state_machine.state

    def reset(self) -> None:
        self.state_machine.reset()
        self._current = None
        self._sum_x = self._sum_y = self._sum_conf = 0.0

    # --------------------------------------------------------------- update
    def process(self, obs: TrackerObservation, gaze_x: float = 0.0,
                gaze_y: float = 0.0) -> Optional[GazeEvent]:
        """Feed one observation. Returns a completed event if one just ended."""
        committed = self.state_machine.update(obs)

        if committed is not None:
            finished = self._close_current(obs.timestamp)
            self._open(committed, obs.timestamp)
            self._accumulate(gaze_x, gaze_y, obs.confidence)
            return finished

        if self._current is None:
            self._open(self.state_machine.snapshot, obs.timestamp)
        self._current.end_time = obs.timestamp
        self._accumulate(gaze_x, gaze_y, obs.confidence)
        return None

    def flush(self, now: float) -> Optional[GazeEvent]:
        """Close the in-progress event, e.g. when tracking stops."""
        finished = self._close_current(now)
        self._current = None
        return finished

    # -------------------------------------------------------------- helpers
    def _open(self, snapshot: StateSnapshot, timestamp: float) -> None:
        self._current = GazeEvent(
            start_time=timestamp,
            end_time=timestamp,
            state=snapshot.state,
            region=snapshot.region,
            square=snapshot.square,
        )
        self._sum_x = self._sum_y = self._sum_conf = 0.0

    def _accumulate(self, x: float, y: float, confidence: float) -> None:
        if self._current is None:
            return
        self._sum_x += x
        self._sum_y += y
        self._sum_conf += confidence
        self._current.sample_count += 1

    def _close_current(self, timestamp: float) -> Optional[GazeEvent]:
        event = self._current
        self._current = None
        if event is None:
            return None
        event.end_time = max(event.end_time, timestamp)
        count = max(event.sample_count, 1)
        event.avg_gaze_x = self._sum_x / count
        event.avg_gaze_y = self._sum_y / count
        event.confidence = self._sum_conf / count
        if event.duration < self.min_event_seconds:
            logger.debug("Discarding %.0f ms event (%s)", event.duration * 1000, event.state)
            return None
        return event


def compute_statistics(events: Iterable[GazeEvent],
                       duration: Optional[float] = None) -> SessionStatistics:
    """Summarise a list of events into :class:`SessionStatistics`."""
    events = list(events)
    stats = SessionStatistics()
    stats.bucket_seconds = {bucket: 0.0 for bucket in SUMMARY_BUCKETS}
    stats.event_count = len(events)
    if not events:
        stats.duration = duration or 0.0
        return stats

    away_durations: List[float] = []
    square_seconds: Dict[str, float] = {}
    weighted_confidence = 0.0
    total = 0.0

    for event in events:
        seconds = event.duration
        total += seconds
        stats.bucket_seconds[event.bucket] = stats.bucket_seconds.get(event.bucket, 0.0) + seconds
        weighted_confidence += event.confidence * seconds
        if event.state.is_away:
            away_durations.append(seconds)
        if event.square:
            square_seconds[event.square] = square_seconds.get(event.square, 0.0) + seconds

    stats.duration = duration if duration is not None else total
    stats.square_seconds = square_seconds
    stats.away_event_count = len(away_durations)
    stats.away_total_seconds = sum(away_durations)
    stats.away_longest_seconds = max(away_durations) if away_durations else 0.0
    stats.away_average_seconds = (
        stats.away_total_seconds / len(away_durations) if away_durations else 0.0
    )
    stats.mean_confidence = weighted_confidence / total if total > 0 else 0.0
    return stats
