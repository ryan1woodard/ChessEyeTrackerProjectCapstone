"""The attention state machine.

Every transition is *debounced*: a candidate state has to persist for a minimum
duration before it is committed. This is what stops a single noisy frame from
manufacturing a look-away event, and it is why blinks do not register as the
user leaving the screen -- a blink simply is not long enough to clear the
threshold, so the previous state survives it untouched.

Different transitions get different thresholds:

===========================  ===========================================
Transition                   Threshold
===========================  ===========================================
to an away-like state        ``away_threshold_ms`` (default 500 ms)
eyes closed                  ``eyes_closed_ms`` (default 1200 ms)
different region             ``region_dwell_ms`` (default 200 ms)
different square, same state ``square_dwell_ms`` (default 250 ms)
===========================  ===========================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..screen.classifier import GazeClassification
from ..screen.regions import KIND_BOARD, KIND_CLOCK
from .models import AttentionState

StateKey = Tuple[AttentionState, Optional[str], Optional[str]]


@dataclass(frozen=True)
class StateSnapshot:
    """The committed state at a point in time."""

    state: AttentionState
    region: Optional[str] = None
    square: Optional[str] = None

    @property
    def key(self) -> StateKey:
        return (self.state, self.region, self.square)


@dataclass
class TrackerObservation:
    """The subset of a :class:`GazeSample` the state machine needs.

    Kept separate from ``GazeSample`` so the state machine can be tested with
    plain synthetic values and no camera or Qt involvement.
    """

    timestamp: float
    face_detected: bool
    eyes_closed: bool
    confidence: float
    valid: bool
    calibrated: bool
    classification: Optional[GazeClassification] = None


class AttentionStateMachine:
    """Converts a noisy per-frame signal into stable, committed states."""

    def __init__(
        self,
        away_threshold_ms: float = 500.0,
        region_dwell_ms: float = 200.0,
        square_dwell_ms: float = 250.0,
        eyes_closed_ms: float = 1200.0,
        minimum_confidence: float = 0.45,
    ) -> None:
        self.away_threshold = away_threshold_ms / 1000.0
        self.region_dwell = region_dwell_ms / 1000.0
        self.square_dwell = square_dwell_ms / 1000.0
        self.eyes_closed = eyes_closed_ms / 1000.0
        self.minimum_confidence = minimum_confidence

        self._committed = StateSnapshot(AttentionState.UNKNOWN)
        self._candidate: Optional[StateSnapshot] = None
        self._candidate_since: float = 0.0
        self._committed_since: Optional[float] = None

    # ------------------------------------------------------------- accessors
    @property
    def state(self) -> AttentionState:
        return self._committed.state

    @property
    def snapshot(self) -> StateSnapshot:
        return self._committed

    def time_in_state(self, now: float) -> float:
        if self._committed_since is None:
            return 0.0
        return max(0.0, now - self._committed_since)

    def reset(self) -> None:
        self._committed = StateSnapshot(AttentionState.UNKNOWN)
        self._candidate = None
        self._candidate_since = 0.0
        self._committed_since = None

    # ---------------------------------------------------------- raw mapping
    def raw_snapshot(self, obs: TrackerObservation) -> StateSnapshot:
        """The instantaneous, un-debounced interpretation of one frame."""
        if not obs.face_detected:
            return StateSnapshot(AttentionState.FACE_NOT_DETECTED)
        if obs.eyes_closed:
            return StateSnapshot(AttentionState.BLINKING)
        if not obs.calibrated:
            return StateSnapshot(AttentionState.UNKNOWN)
        if not obs.valid or obs.confidence < self.minimum_confidence:
            return StateSnapshot(AttentionState.LOW_CONFIDENCE)

        classification = obs.classification
        if classification is None:
            return StateSnapshot(AttentionState.LOOKING_AT_SCREEN)
        if not classification.on_screen:
            return StateSnapshot(AttentionState.LOOKING_AWAY)
        if classification.kind == KIND_BOARD:
            return StateSnapshot(AttentionState.LOOKING_AT_BOARD,
                                 classification.region_name, classification.square)
        if classification.kind == KIND_CLOCK:
            return StateSnapshot(AttentionState.LOOKING_AT_CLOCK, classification.region_name)
        if classification.region_name in ("Unknown", None):
            return StateSnapshot(AttentionState.LOOKING_AT_SCREEN)
        return StateSnapshot(AttentionState.LOOKING_AT_OTHER, classification.region_name)

    # ------------------------------------------------------------ thresholds
    def threshold_for(self, current: StateSnapshot, candidate: StateSnapshot) -> float:
        """Seconds ``candidate`` must persist before replacing ``current``."""
        if candidate.state is AttentionState.BLINKING:
            return self.eyes_closed
        if candidate.state != current.state:
            if candidate.state.is_away or candidate.state is AttentionState.LOW_CONFIDENCE:
                return self.away_threshold
            return self.region_dwell
        if candidate.region != current.region:
            return self.region_dwell
        return self.square_dwell

    # ---------------------------------------------------------------- update
    def update(self, obs: TrackerObservation) -> Optional[StateSnapshot]:
        """Feed one frame in. Returns the new snapshot if the state changed."""
        raw = self.raw_snapshot(obs)
        now = obs.timestamp

        if self._committed_since is None:
            self._committed_since = now

        if raw.key == self._committed.key:
            self._candidate = None
            return None

        if self._candidate is None or self._candidate.key != raw.key:
            self._candidate = raw
            self._candidate_since = now
            return None

        if (now - self._candidate_since) >= self.threshold_for(self._committed, raw):
            self._committed = raw
            self._committed_since = now
            self._candidate = None
            return self._committed
        return None
