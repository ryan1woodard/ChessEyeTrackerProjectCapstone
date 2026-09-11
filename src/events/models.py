"""Data models for attention states, aggregated events and sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class AttentionState(str, Enum):
    """What the user is doing, as far as the tracker can tell."""

    UNKNOWN = "UNKNOWN"
    LOOKING_AT_SCREEN = "LOOKING_AT_SCREEN"
    LOOKING_AT_BOARD = "LOOKING_AT_BOARD"
    LOOKING_AT_CLOCK = "LOOKING_AT_CLOCK"
    LOOKING_AT_OTHER = "LOOKING_AT_OTHER"
    LOOKING_AWAY = "LOOKING_AWAY"
    BLINKING = "BLINKING"
    FACE_NOT_DETECTED = "FACE_NOT_DETECTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"

    @property
    def is_away(self) -> bool:
        """Whether this state counts as attention having left the screen."""
        return self in (AttentionState.LOOKING_AWAY, AttentionState.FACE_NOT_DETECTED)

    @property
    def is_engaged(self) -> bool:
        return self in (
            AttentionState.LOOKING_AT_SCREEN,
            AttentionState.LOOKING_AT_BOARD,
            AttentionState.LOOKING_AT_CLOCK,
            AttentionState.LOOKING_AT_OTHER,
        )

    def label(self) -> str:
        return self.value.replace("_", " ").title()


#: Coarse buckets used for the session summary percentages.
SUMMARY_BUCKETS = ("board", "clock", "other", "away", "unknown")


def bucket_for(state: AttentionState) -> str:
    if state is AttentionState.LOOKING_AT_BOARD:
        return "board"
    if state is AttentionState.LOOKING_AT_CLOCK:
        return "clock"
    if state in (AttentionState.LOOKING_AT_OTHER, AttentionState.LOOKING_AT_SCREEN):
        return "other"
    if state.is_away:
        return "away"
    return "unknown"


@dataclass
class GazeEvent:
    """A run of consecutive frames sharing the same state, region and square."""

    start_time: float
    end_time: float
    state: AttentionState
    region: Optional[str] = None
    square: Optional[str] = None
    avg_gaze_x: float = 0.0
    avg_gaze_y: float = 0.0
    confidence: float = 0.0
    sample_count: int = 0
    event_id: Optional[int] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def bucket(self) -> str:
        return bucket_for(self.state)

    def to_dict(self) -> Dict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "state": self.state.value,
            "region": self.region,
            "square": self.square,
            "avg_gaze_x": self.avg_gaze_x,
            "avg_gaze_y": self.avg_gaze_y,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
        }


@dataclass
class GazeSampleRecord:
    """A downsampled raw gaze position, retained only to build heatmaps."""

    timestamp: float
    x: float
    y: float
    confidence: float
    square: Optional[str] = None


@dataclass
class SessionStatistics:
    """Aggregate numbers for one tracking session."""

    duration: float = 0.0
    bucket_seconds: Dict[str, float] = field(default_factory=dict)
    away_event_count: int = 0
    away_total_seconds: float = 0.0
    away_longest_seconds: float = 0.0
    away_average_seconds: float = 0.0
    square_seconds: Dict[str, float] = field(default_factory=dict)
    mean_confidence: float = 0.0
    event_count: int = 0

    def bucket_percent(self, bucket: str) -> float:
        if self.duration <= 0:
            return 0.0
        return 100.0 * self.bucket_seconds.get(bucket, 0.0) / self.duration

    def top_squares(self, limit: int = 8) -> List[tuple[str, float]]:
        return sorted(self.square_seconds.items(), key=lambda kv: kv[1], reverse=True)[:limit]

    def to_dict(self) -> Dict:
        return {
            "duration": self.duration,
            "bucket_seconds": dict(self.bucket_seconds),
            "bucket_percent": {b: self.bucket_percent(b) for b in SUMMARY_BUCKETS},
            "away_event_count": self.away_event_count,
            "away_total_seconds": self.away_total_seconds,
            "away_longest_seconds": self.away_longest_seconds,
            "away_average_seconds": self.away_average_seconds,
            "square_seconds": dict(self.square_seconds),
            "mean_confidence": self.mean_confidence,
            "event_count": self.event_count,
        }


@dataclass
class Session:
    """A tracking session: a start, an end, and everything recorded between."""

    session_id: str
    start_wall_clock: datetime
    start_monotonic: float
    end_wall_clock: Optional[datetime] = None
    end_monotonic: Optional[float] = None
    calibration_id: Optional[str] = None
    monitor_index: int = 1
    db_id: Optional[int] = None
    notes: str = ""

    @property
    def duration(self) -> float:
        if self.end_monotonic is None:
            return 0.0
        return max(0.0, self.end_monotonic - self.start_monotonic)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "start_time": self.start_wall_clock.isoformat(timespec="seconds"),
            "end_time": self.end_wall_clock.isoformat(timespec="seconds")
            if self.end_wall_clock else None,
            "duration": self.duration,
            "calibration_id": self.calibration_id,
            "monitor_index": self.monitor_index,
            "notes": self.notes,
        }


def format_duration(seconds: float) -> str:
    """Format seconds as ``M:SS`` or ``H:MM:SS``."""
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
