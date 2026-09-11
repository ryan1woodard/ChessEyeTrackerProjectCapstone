"""Storage, export and configuration tests."""

from __future__ import annotations

import csv
import json
from datetime import datetime

import pytest

from src.events.models import AttentionState, GazeEvent, GazeSampleRecord, Session
from src.storage.database import Database, SCHEMA_VERSION
from src.storage.export import (export_events_csv, export_samples_csv,
                                export_session_json)
from src.utils.config import Config, deep_merge


@pytest.fixture()
def db(tmp_path) -> Database:
    database = Database(tmp_path / "test.db")
    database.connect()
    yield database
    database.close()


@pytest.fixture()
def session() -> Session:
    return Session(session_id="2026-09-03_13-42-10",
                   start_wall_clock=datetime(2026, 9, 3, 13, 42, 10),
                   start_monotonic=1000.0, monitor_index=1,
                   calibration_id="cal_test")


def sample_events():
    return [
        GazeEvent(1000.0, 1002.5, AttentionState.LOOKING_AT_BOARD, "Chessboard", "e4",
                  740.0, 420.0, 0.87, 75),
        GazeEvent(1002.5, 1004.0, AttentionState.LOOKING_AT_CLOCK, "White Clock", None,
                  1300.0, 720.0, 0.71, 45),
        GazeEvent(1004.0, 1006.2, AttentionState.LOOKING_AWAY, None, None,
                  0.0, 0.0, 0.1, 66),
    ]


class TestDatabase:
    def test_schema_version_is_recorded(self, db):
        row = db.connection.execute("SELECT version FROM schema_version").fetchone()
        assert row["version"] == SCHEMA_VERSION

    def test_migration_is_idempotent(self, tmp_path):
        path = tmp_path / "twice.db"
        for _ in range(2):
            database = Database(path)
            database.connect()
            database.close()
        database = Database(path)
        database.connect()
        rows = database.connection.execute("SELECT version FROM schema_version").fetchall()
        assert len(rows) == 1
        database.close()

    def test_session_lifecycle(self, db, session):
        db_id = db.create_session(session)
        assert db_id > 0 and session.db_id == db_id

        session.end_monotonic = session.start_monotonic + 1271.0
        session.end_wall_clock = datetime(2026, 9, 3, 14, 3, 21)
        db.finish_session(session)

        row = db.get_session(db_id)
        assert row["session_id"] == "2026-09-03_13-42-10"
        assert row["duration"] == pytest.approx(1271.0)
        assert row["end_time"].startswith("2026-09-03T14:03:21")

    def test_events_are_stored_as_offsets(self, db, session):
        db.create_session(session)
        for event in sample_events():
            db.insert_event(session.db_id, event, session.start_monotonic)

        events = db.get_events(session.db_id)
        assert len(events) == 3
        assert events[0].start_time == pytest.approx(0.0)
        assert events[0].square == "e4"
        assert events[0].state is AttentionState.LOOKING_AT_BOARD
        assert events[2].state is AttentionState.LOOKING_AWAY

    def test_samples_round_trip(self, db, session):
        db.create_session(session)
        samples = [GazeSampleRecord(1000.0 + i * 0.1, 700 + i, 400 + i, 0.8, "e4")
                   for i in range(25)]
        assert db.insert_samples(session.db_id, samples, session.start_monotonic) == 25

        stored = db.get_samples(session.db_id)
        assert len(stored) == 25
        assert stored[0].timestamp == pytest.approx(0.0)
        assert stored[-1].x == pytest.approx(724.0)

    def test_inserting_no_samples_is_a_no_op(self, db, session):
        db.create_session(session)
        assert db.insert_samples(session.db_id, [], session.start_monotonic) == 0

    def test_delete_removes_children(self, db, session):
        db.create_session(session)
        db.insert_event(session.db_id, sample_events()[0], session.start_monotonic)
        db.insert_samples(session.db_id,
                          [GazeSampleRecord(1000.0, 1, 2, 0.5)], session.start_monotonic)
        db.delete_session(session.db_id)

        assert db.get_session(session.db_id) is None
        assert db.get_events(session.db_id) == []
        assert db.get_samples(session.db_id) == []

    def test_sessions_are_listed_newest_first(self, db):
        for hour in (10, 11, 12):
            db.create_session(Session(f"s{hour}", datetime(2026, 9, 3, hour, 0, 0), 0.0))
        assert [row["session_id"] for row in db.list_sessions()] == ["s12", "s11", "s10"]

    def test_unknown_state_strings_do_not_crash_reads(self, db, session):
        db.create_session(session)
        db.connection.execute(
            "INSERT INTO events (session_id, start_offset, end_offset, duration, state)"
            " VALUES (?, 0, 1, 1, 'SOMETHING_NEW')", (session.db_id,))
        db.connection.commit()
        assert db.get_events(session.db_id)[0].state is AttentionState.UNKNOWN


class TestExport:
    def test_csv_contains_one_row_per_event(self, tmp_path):
        path = export_events_csv(tmp_path / "events.csv", sample_events())
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3
        assert rows[0]["square"] == "e4"
        assert rows[0]["state"] == "LOOKING_AT_BOARD"
        assert float(rows[0]["duration"]) == pytest.approx(2.5)
        assert rows[2]["square"] == ""

    def test_samples_csv(self, tmp_path):
        samples = [GazeSampleRecord(0.1 * i, i, i, 0.5, None) for i in range(5)]
        path = export_samples_csv(tmp_path / "samples.csv", samples)
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 6

    def test_json_includes_statistics_and_events(self, tmp_path):
        meta = {"session_id": "test", "duration": 6.2}
        path = export_session_json(tmp_path / "s.json", meta, sample_events())
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["format"] == "eye_tracker_session"
        assert payload["session"]["session_id"] == "test"
        assert len(payload["events"]) == 3
        assert payload["statistics"]["away_event_count"] == 1
        assert "gaze_samples" not in payload

    def test_json_can_include_samples(self, tmp_path):
        samples = [GazeSampleRecord(0.0, 1.0, 2.0, 0.9, "e4")]
        path = export_session_json(tmp_path / "s2.json", {"duration": 1.0},
                                   sample_events(), samples, include_samples=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["gaze_samples"][0]["square"] == "e4"


class TestConfig:
    def test_deep_merge_preserves_untouched_keys(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        merged = deep_merge(base, {"a": {"c": 99}})
        assert merged == {"a": {"b": 1, "c": 99}, "d": 3}
        assert base["a"]["c"] == 2, "the original must not be mutated"

    def test_defaults_load_and_dotted_access(self):
        config = Config.load()
        assert config.get("camera.width") == 1280
        assert config.get("tracking.minimum_confidence") == pytest.approx(0.45)
        assert config.get("nothing.here", "fallback") == "fallback"

    def test_set_creates_intermediate_dicts(self):
        config = Config({})
        config.set("a.b.c", 42)
        assert config.get("a.b.c") == 42

    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "user.json"
        config = Config.load(user_path=path)
        config.set("overlay.dot_radius", 25)
        config.save()

        reloaded = Config.load(user_path=path)
        assert reloaded.get("overlay.dot_radius") == 25
        assert reloaded.get("camera.width") == 1280, "defaults still apply"

    def test_a_corrupt_user_file_falls_back_to_defaults(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{{{", encoding="utf-8")
        assert Config.load(user_path=path).get("camera.width") == 1280

    def test_reset_to_defaults(self, tmp_path):
        config = Config.load(user_path=tmp_path / "u.json")
        config.set("camera.width", 640)
        config.reset_to_defaults()
        assert config.get("camera.width") == 1280

    def test_every_calibration_feature_name_is_real(self):
        from src.tracking.features import FEATURE_NAMES
        config = Config.load()
        for name in config.get("calibration.model_features"):
            assert name in FEATURE_NAMES, f"{name} is not a known feature"
