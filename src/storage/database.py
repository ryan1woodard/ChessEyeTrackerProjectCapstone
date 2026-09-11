"""SQLite persistence.

Only *derived* data is stored: aggregated events, downsampled gaze coordinates
and calibration metadata. No webcam frames and no screenshots ever reach the
database.

Event times are stored as offsets in seconds from the start of their session,
which makes timelines trivial to draw and keeps the data independent of the
wall clock.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..events.models import (AttentionState, GazeEvent, GazeSampleRecord, Session)
from ..utils.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "sessions" / "eye_tracker.db"
SCHEMA_VERSION = 1

_MIGRATIONS: Dict[int, Sequence[str]] = {
    1: (
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    TEXT    NOT NULL UNIQUE,
            start_time    TEXT    NOT NULL,
            end_time      TEXT,
            duration      REAL    NOT NULL DEFAULT 0,
            calibration_id TEXT,
            monitor_index INTEGER NOT NULL DEFAULT 1,
            notes         TEXT    NOT NULL DEFAULT ''
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            start_offset REAL    NOT NULL,
            end_offset   REAL    NOT NULL,
            duration     REAL    NOT NULL,
            state        TEXT    NOT NULL,
            region       TEXT,
            square       TEXT,
            avg_gaze_x   REAL    NOT NULL DEFAULT 0,
            avg_gaze_y   REAL    NOT NULL DEFAULT 0,
            confidence   REAL    NOT NULL DEFAULT 0,
            sample_count INTEGER NOT NULL DEFAULT 0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS gaze_samples (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            t          REAL    NOT NULL,
            x          REAL    NOT NULL,
            y          REAL    NOT NULL,
            confidence REAL    NOT NULL DEFAULT 0,
            square     TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS calibrations (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id    TEXT    NOT NULL UNIQUE,
            created_at    TEXT    NOT NULL,
            monitor_index INTEGER NOT NULL DEFAULT 1,
            screen_width  INTEGER NOT NULL,
            screen_height INTEGER NOT NULL,
            quality       TEXT    NOT NULL DEFAULT '',
            mean_error    REAL    NOT NULL DEFAULT 0,
            median_error  REAL    NOT NULL DEFAULT 0,
            max_error     REAL    NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_samples_session ON gaze_samples(session_id)",
    ),
}


class Database:
    """A small, thread-safe wrapper over the session database."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_DB_PATH
        self._lock = threading.RLock()
        self._connection: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------- lifecycle
    def connect(self) -> None:
        if self._connection is not None:
            return
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._migrate()
        logger.info("Database ready at %s", self.path)

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.commit()
                self._connection.close()
                self._connection = None

    def __enter__(self) -> "Database":
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self.connect()
        assert self._connection is not None
        return self._connection

    def _migrate(self) -> None:
        conn = self._connection
        assert conn is not None
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        current = int(row["version"]) if row else 0
        for version in sorted(_MIGRATIONS):
            if version <= current:
                continue
            logger.info("Applying database migration %d", version)
            for statement in _MIGRATIONS[version]:
                conn.execute(statement)
            current = version
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (current,))
        conn.commit()

    # -------------------------------------------------------------- sessions
    def create_session(self, session: Session) -> int:
        with self._lock:
            cursor = self.connection.execute(
                "INSERT INTO sessions (session_id, start_time, calibration_id, monitor_index)"
                " VALUES (?, ?, ?, ?)",
                (session.session_id, session.start_wall_clock.isoformat(timespec="seconds"),
                 session.calibration_id, session.monitor_index),
            )
            self.connection.commit()
            session.db_id = int(cursor.lastrowid)
            return session.db_id

    def finish_session(self, session: Session) -> None:
        if session.db_id is None:
            return
        end = session.end_wall_clock or datetime.now()
        with self._lock:
            self.connection.execute(
                "UPDATE sessions SET end_time = ?, duration = ? WHERE id = ?",
                (end.isoformat(timespec="seconds"), session.duration, session.db_id),
            )
            self.connection.commit()

    def list_sessions(self, limit: int = 100) -> List[sqlite3.Row]:
        with self._lock:
            return list(self.connection.execute(
                "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall())

    def get_session(self, db_id: int) -> Optional[sqlite3.Row]:
        with self._lock:
            return self.connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (db_id,)).fetchone()

    def delete_session(self, db_id: int) -> None:
        with self._lock:
            self.connection.execute("DELETE FROM events WHERE session_id = ?", (db_id,))
            self.connection.execute("DELETE FROM gaze_samples WHERE session_id = ?", (db_id,))
            self.connection.execute("DELETE FROM sessions WHERE id = ?", (db_id,))
            self.connection.commit()

    # ---------------------------------------------------------------- events
    def insert_event(self, session_db_id: int, event: GazeEvent,
                     session_start: float) -> int:
        with self._lock:
            cursor = self.connection.execute(
                "INSERT INTO events (session_id, start_offset, end_offset, duration, state,"
                " region, square, avg_gaze_x, avg_gaze_y, confidence, sample_count)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_db_id, event.start_time - session_start,
                 event.end_time - session_start, event.duration, event.state.value,
                 event.region, event.square, event.avg_gaze_x, event.avg_gaze_y,
                 event.confidence, event.sample_count),
            )
            self.connection.commit()
            event.event_id = int(cursor.lastrowid)
            return event.event_id

    def get_events(self, session_db_id: int) -> List[GazeEvent]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM events WHERE session_id = ? ORDER BY start_offset",
                (session_db_id,),
            ).fetchall()
        events: List[GazeEvent] = []
        for row in rows:
            try:
                state = AttentionState(row["state"])
            except ValueError:  # pragma: no cover - forward compatibility
                state = AttentionState.UNKNOWN
            events.append(GazeEvent(
                start_time=float(row["start_offset"]),
                end_time=float(row["end_offset"]),
                state=state,
                region=row["region"],
                square=row["square"],
                avg_gaze_x=float(row["avg_gaze_x"]),
                avg_gaze_y=float(row["avg_gaze_y"]),
                confidence=float(row["confidence"]),
                sample_count=int(row["sample_count"]),
                event_id=int(row["id"]),
            ))
        return events

    # --------------------------------------------------------------- samples
    def insert_samples(self, session_db_id: int,
                       samples: Iterable[GazeSampleRecord],
                       session_start: float) -> int:
        rows = [(session_db_id, s.timestamp - session_start, s.x, s.y, s.confidence, s.square)
                for s in samples]
        if not rows:
            return 0
        with self._lock:
            self.connection.executemany(
                "INSERT INTO gaze_samples (session_id, t, x, y, confidence, square)"
                " VALUES (?, ?, ?, ?, ?, ?)", rows,
            )
            self.connection.commit()
        return len(rows)

    def get_samples(self, session_db_id: int) -> List[GazeSampleRecord]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT t, x, y, confidence, square FROM gaze_samples"
                " WHERE session_id = ? ORDER BY t", (session_db_id,),
            ).fetchall()
        return [GazeSampleRecord(float(r["t"]), float(r["x"]), float(r["y"]),
                                 float(r["confidence"]), r["square"]) for r in rows]

    # ---------------------------------------------------------- calibrations
    def record_calibration(self, profile) -> None:
        """Store calibration metadata (the model itself lives in a JSON file)."""
        report = profile.report
        with self._lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO calibrations (profile_id, created_at, monitor_index,"
                " screen_width, screen_height, quality, mean_error, median_error, max_error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (profile.profile_id, profile.created_at, profile.monitor_index,
                 profile.screen_width, profile.screen_height,
                 report.quality() if report else "",
                 report.mean_error_px if report else 0.0,
                 report.median_error_px if report else 0.0,
                 report.max_error_px if report else 0.0),
            )
            self.connection.commit()
