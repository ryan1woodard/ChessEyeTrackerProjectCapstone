"""Session export to CSV and JSON."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..events.detector import compute_statistics
from ..events.models import GazeEvent, GazeSampleRecord

logger = logging.getLogger(__name__)

CSV_FIELDS: Sequence[str] = (
    "timestamp", "duration", "state", "region", "square",
    "gaze_x", "gaze_y", "confidence", "sample_count",
)


def export_events_csv(path: Path, events: Iterable[GazeEvent]) -> Path:
    """Write one row per aggregated event.

    ``timestamp`` is the event's offset in seconds from the start of the
    session, matching how events are stored.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for event in events:
            writer.writerow({
                "timestamp": round(event.start_time, 3),
                "duration": round(event.duration, 3),
                "state": event.state.value,
                "region": event.region or "",
                "square": event.square or "",
                "gaze_x": round(event.avg_gaze_x, 1),
                "gaze_y": round(event.avg_gaze_y, 1),
                "confidence": round(event.confidence, 3),
                "sample_count": event.sample_count,
            })
    logger.info("Exported events to %s", path)
    return path


def export_samples_csv(path: Path, samples: Iterable[GazeSampleRecord]) -> Path:
    """Write raw (downsampled) gaze coordinates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "gaze_x", "gaze_y", "confidence", "square"])
        for sample in samples:
            writer.writerow([round(sample.timestamp, 3), round(sample.x, 1),
                             round(sample.y, 1), round(sample.confidence, 3),
                             sample.square or ""])
    return path


def export_session_json(path: Path, session_meta: Dict, events: List[GazeEvent],
                        samples: Optional[List[GazeSampleRecord]] = None,
                        include_samples: bool = False) -> Path:
    """Write session metadata, statistics and events as a single JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = compute_statistics(events, duration=session_meta.get("duration"))
    payload = {
        "format": "eye_tracker_session",
        "version": 1,
        "session": session_meta,
        "statistics": stats.to_dict(),
        "events": [event.to_dict() for event in events],
    }
    if include_samples and samples:
        payload["gaze_samples"] = [
            {"t": round(s.timestamp, 3), "x": round(s.x, 1), "y": round(s.y, 1),
             "confidence": round(s.confidence, 3), "square": s.square}
            for s in samples
        ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Exported session JSON to %s", path)
    return path
