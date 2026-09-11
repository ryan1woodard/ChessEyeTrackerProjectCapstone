"""Backend-selection and model-download tests.

These cover the MediaPipe packaging split: builds up to ~0.10.14 expose
``mediapipe.solutions`` with bundled models, while newer builds removed it and
require a downloaded Tasks model. The application has to cope with both, and
fail comprehensibly when it can do neither.
"""

from __future__ import annotations

import pytest

from src.tracking import face_tracker as ft
from src.utils import model_download as md


class TestBackendDetection:
    def test_legacy_probe_reports_a_boolean(self):
        assert isinstance(ft.legacy_solutions_available(), bool)

    def test_probe_is_false_when_solutions_is_absent(self, monkeypatch):
        class FakeMediaPipe:
            pass
        monkeypatch.setitem(__import__("sys").modules, "mediapipe", FakeMediaPipe())
        assert ft.legacy_solutions_available() is False

    def test_auto_falls_back_to_tasks_when_legacy_is_missing(self, monkeypatch):
        """The exact failure reported on MediaPipe 0.10.30+."""
        monkeypatch.setattr(ft, "legacy_solutions_available", lambda: False)
        attempted = []

        def fake_landmarker(self):
            attempted.append("tasks")
            self.backend = "face_landmarker"
            self._impl = object()

        monkeypatch.setattr(ft.FaceTracker, "_open_face_landmarker", fake_landmarker)
        tracker = ft.FaceTracker(backend="auto")
        assert attempted == ["tasks"]
        assert tracker.backend == "face_landmarker"

    def test_auto_prefers_legacy_when_present(self, monkeypatch):
        """Legacy needs no download, so it wins when it is available."""
        monkeypatch.setattr(ft, "legacy_solutions_available", lambda: True)
        monkeypatch.setattr(ft.FaceTracker, "_open_face_mesh",
                            lambda self: setattr(self, "backend", "face_mesh"))
        monkeypatch.setattr(ft.FaceTracker, "_open_face_landmarker",
                            lambda self: pytest.fail("should not reach Tasks"))
        assert ft.FaceTracker(backend="auto").backend == "face_mesh"

    def test_both_unavailable_raises_an_actionable_error(self, monkeypatch):
        monkeypatch.setattr(ft, "legacy_solutions_available", lambda: False)

        def explode(self):
            raise md.ModelDownloadError("no network")

        monkeypatch.setattr(ft.FaceTracker, "_open_face_landmarker", explode)
        with pytest.raises(ft.FaceTrackerError) as info:
            ft.FaceTracker(backend="auto")

        message = str(info.value)
        # The message must name both causes and tell the user what to do.
        assert "solutions" in message
        assert "no network" in message
        assert "requirements.txt" in message

    def test_explicitly_requesting_a_broken_backend_reports_only_that(self, monkeypatch):
        def explode(self):
            raise md.ModelDownloadError("model missing")

        monkeypatch.setattr(ft.FaceTracker, "_open_face_landmarker", explode)
        with pytest.raises(ft.FaceTrackerError, match="model missing"):
            ft.FaceTracker(backend="face_landmarker")


class TestModelDownload:
    def test_absent_file_is_not_present(self, tmp_path):
        assert md.model_is_present(tmp_path / "nothing.task") is False

    def test_a_truncated_file_is_rejected(self, tmp_path):
        path = tmp_path / "face_landmarker.task"
        path.write_bytes(b"nope")
        assert md.model_is_present(path) is False

    def test_a_full_sized_file_is_accepted(self, tmp_path):
        path = tmp_path / "face_landmarker.task"
        path.write_bytes(b"\0" * (md.MIN_MODEL_BYTES + 1))
        assert md.model_is_present(path) is True
        # An existing model must be returned untouched, with no download.
        assert md.ensure_face_landmarker_model(path, allow_download=False) == path

    def test_download_disabled_gives_manual_instructions(self, tmp_path):
        with pytest.raises(md.ModelDownloadError, match="manually"):
            md.ensure_face_landmarker_model(tmp_path / "m.task", allow_download=False)

    def test_network_failure_is_reported_with_the_url(self, tmp_path, monkeypatch):
        import urllib.error

        def fail(*args, **kwargs):
            raise urllib.error.URLError("offline")

        monkeypatch.setattr(md.urllib.request, "urlopen", fail)
        with pytest.raises(md.ModelDownloadError) as info:
            md.ensure_face_landmarker_model(tmp_path / "m.task")
        assert md.MODEL_URL in str(info.value)

    def test_a_short_download_is_treated_as_a_failure(self, tmp_path, monkeypatch):
        """A captive portal or proxy returning an HTML page must not be saved."""
        class FakeResponse:
            def __init__(self):
                self._chunks = [b"<html>error</html>"]

            def read(self, _size):
                return self._chunks.pop(0) if self._chunks else b""

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(md.urllib.request, "urlopen",
                            lambda *a, **k: FakeResponse())
        target = tmp_path / "m.task"
        with pytest.raises(md.ModelDownloadError, match="too small"):
            md.ensure_face_landmarker_model(target)
        assert not target.exists(), "a bad download must not be left behind"
        assert list(tmp_path.glob("*.part")) == [], "no partial files left behind"
