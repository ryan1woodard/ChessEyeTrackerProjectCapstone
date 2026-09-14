"""GUI smoke tests.

These construct the real windows against an offscreen Qt platform. They will
not catch visual problems, but they do catch the errors that static analysis
cannot: bad signal/slot connections, misspelled Qt enums, and widgets wired to
methods that do not exist.

Skipped automatically when PySide6 is unavailable.
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.events.models import AttentionState, GazeEvent, GazeSampleRecord, Session  # noqa: E402
from src.storage.database import Database  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.geometry import Rect  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def populated_db(tmp_path):
    database = Database(tmp_path / "gui.db")
    database.connect()
    session = Session("2026-09-03_13-42-10", datetime(2026, 9, 3, 13, 42, 10), 0.0)
    database.create_session(session)
    events = [
        GazeEvent(0.0, 8.0, AttentionState.LOOKING_AT_BOARD, "Chessboard", "e4",
                  740, 420, 0.9, 240),
        GazeEvent(8.0, 10.0, AttentionState.LOOKING_AT_CLOCK, "White Clock", None,
                  1300, 720, 0.7, 60),
        GazeEvent(10.0, 13.0, AttentionState.LOOKING_AWAY, None, None, 0, 0, 0.1, 90),
    ]
    for event in events:
        database.insert_event(session.db_id, event, 0.0)
    database.insert_samples(
        session.db_id,
        [GazeSampleRecord(i * 0.1, 700 + i, 400 + (i % 50), 0.8, "e4") for i in range(120)],
        0.0,
    )
    session.end_monotonic = 13.0
    session.end_wall_clock = datetime(2026, 9, 3, 13, 42, 23)
    database.finish_session(session)
    yield database
    database.close()


class TestOverlay:
    def test_it_is_click_through_and_positioned(self, qapp):
        from PySide6.QtCore import Qt
        from src.visualization.gaze_overlay import GazeOverlay

        overlay = GazeOverlay(Rect(0, 0, 1920, 1080))
        assert overlay.testAttribute(Qt.WA_TransparentForMouseEvents)
        assert overlay.testAttribute(Qt.WA_TranslucentBackground)
        assert overlay.windowFlags() & Qt.WindowTransparentForInput
        assert overlay.geometry().width() == 1920

    def test_painting_with_every_layer_enabled(self, qapp):
        from PySide6.QtGui import QPixmap
        from src.visualization.gaze_overlay import GazeOverlay

        overlay = GazeOverlay(Rect(0, 0, 800, 600))
        overlay.set_layers(True, True, True, True)
        overlay.set_board(Rect(100, 100, 400, 400))
        overlay.set_regions([("White Clock", Rect(520, 300, 200, 60))])
        overlay.set_debug_lines(["FPS 30.0", "Region Chessboard"])
        overlay.set_gaze(300, 250, 0.87, True)
        overlay.set_square_rect(Rect(200, 200, 50, 50))
        overlay.render(QPixmap(800, 600))  # exercises paintEvent

    def test_painting_an_invalid_estimate(self, qapp):
        from PySide6.QtGui import QPixmap
        from src.visualization.gaze_overlay import GazeOverlay

        overlay = GazeOverlay(Rect(0, 0, 400, 300))
        overlay.set_gaze(100, 100, 0.0, False)
        overlay.render(QPixmap(400, 300))

    def test_moving_to_a_second_monitor(self, qapp):
        from src.visualization.gaze_overlay import GazeOverlay

        overlay = GazeOverlay(Rect(0, 0, 1920, 1080))
        overlay.set_screen_rect(Rect(1920, 0, 1280, 1024))
        assert overlay.geometry().x() == 1920


class TestCalibrationWindow:
    def test_targets_cover_the_monitor(self, qapp, tmp_path):
        from src.gui.calibration_window import CalibrationWindow

        config = Config.load(user_path=tmp_path / "c.json")
        window = CalibrationWindow(config, Rect(0, 0, 1920, 1080), 1, 0)
        assert len(window._targets) == 13
        xs = [t[0] for t in window._targets]
        ys = [t[1] for t in window._targets]
        assert min(xs) < 200 and max(xs) > 1700
        assert min(ys) < 120 and max(ys) > 960

    def test_targets_are_offset_for_a_secondary_monitor(self, qapp, tmp_path):
        from src.gui.calibration_window import CalibrationWindow

        config = Config.load(user_path=tmp_path / "c.json")
        window = CalibrationWindow(config, Rect(1920, 0, 1280, 1024), 2, 0)
        assert all(t[0] >= 1920 for t in window._targets)

    def test_painting_each_phase(self, qapp, tmp_path):
        from PySide6.QtGui import QPixmap
        from src.gui.calibration_window import (CalibrationWindow, PHASE_COLLECT,
                                                PHASE_INTRO, PHASE_SETTLE)

        config = Config.load(user_path=tmp_path / "c.json")
        window = CalibrationWindow(config, Rect(0, 0, 800, 600), 1, 0)
        window.resize(800, 600)
        for phase in (PHASE_INTRO, PHASE_SETTLE, PHASE_COLLECT):
            window._phase = phase
            window.render(QPixmap(800, 600))

    def test_a_posture_is_requested_at_every_target(self, qapp, tmp_path):
        """Each target asks for a posture, and tilt is asked for repeatedly.

        The model can only correct for head positions calibration observed, so
        the cues are what put head tilt into the training data at all.
        """
        from src.gui.calibration_window import CalibrationWindow
        from src.tracking.calibration import posture

        config = Config.load(user_path=tmp_path / "c.json")
        window = CalibrationWindow(config, Rect(0, 0, 1920, 1080), 1, 0)
        assert window._posture_hint
        postures = [posture(i) for i in range(len(window._targets))]
        assert all(p.label for p in postures)
        assert sum(p.is_tilt for p in postures) >= 4
        assert any(p.lean for p in postures) and any(p.shift for p in postures)
        # Tilts straddle upright rather than all going one way.
        tilts = [p.roll_deg for p in postures if p.is_tilt]
        assert min(tilts) < 0 < max(tilts)

    def test_the_pose_phase_comes_before_the_dot(self, qapp, tmp_path):
        """The instruction has to be readable before anything is recorded.

        Reading a cue and acting on it cannot be done while fixating a dot, so
        the posture gets a phase of its own with no target on screen.
        """
        from PySide6.QtGui import QPixmap
        from src.gui.calibration_window import CalibrationWindow, PHASE_POSE

        config = Config.load(user_path=tmp_path / "c.json")
        window = CalibrationWindow(config, Rect(0, 0, 1000, 800), 1, 0)
        window.resize(1000, 800)
        window._begin_point(1)
        assert window._phase == PHASE_POSE
        for index in range(len(window._targets)):
            window._index = index
            window.render(QPixmap(1000, 800))

    def test_the_pose_phase_waits_for_the_posture(self, qapp, tmp_path):
        from src.gui.calibration_window import (CalibrationWindow, PHASE_POSE,
                                                PHASE_SETTLE)

        config = Config.load(user_path=tmp_path / "c.json")
        window = CalibrationWindow(config, Rect(0, 0, 1000, 800), 1, 0)
        window._begin_point(1)                      # a tilt posture
        window._face_present = True
        window._head_roll = 0.0                     # not tilted yet

        for _ in range(int(window._pose_min_ms / 25) + 8):
            window._tick()
        assert window._phase == PHASE_POSE, "advanced before the posture was held"

        window._head_roll = window.posture.roll_deg
        for _ in range(int(window._pose_hold_ms / 25) + 2):
            window._tick()
        assert window._phase == PHASE_SETTLE

    def test_the_pose_phase_gives_up_rather_than_stranding_anyone(self, qapp, tmp_path):
        """A tilt some webcams cannot see must not block the whole calibration."""
        from src.gui.calibration_window import CalibrationWindow, PHASE_SETTLE

        config = Config.load(user_path=tmp_path / "c.json")
        window = CalibrationWindow(config, Rect(0, 0, 1000, 800), 1, 0)
        window._begin_point(1)
        window._face_present = True
        window._head_roll = 0.0
        for _ in range(int(window._pose_ms / 25) + 4):
            window._tick()
        assert window._phase == PHASE_SETTLE

    def test_tilting_the_wrong_way_does_not_count(self, qapp, tmp_path):
        from src.gui.calibration_window import CalibrationWindow

        config = Config.load(user_path=tmp_path / "c.json")
        window = CalibrationWindow(config, Rect(0, 0, 1000, 800), 1, 0)
        window._index = 1
        window._face_present = True
        wanted = window.posture.roll_deg
        window._head_roll = wanted
        assert window.posture_held
        window._head_roll = -wanted
        assert not window.posture_held

    def test_lean_and_shift_are_measured_not_assumed(self, qapp, tmp_path):
        """The screen must not say "that's it" about something it never measured.

        Leaning and shifting are checked against the user's own resting
        position, so before one has been observed the answer is "unknown"
        rather than "fine".
        """
        from src.gui.calibration_window import CalibrationWindow

        config = Config.load(user_path=tmp_path / "c.json")
        window = CalibrationWindow(config, Rect(0, 0, 1000, 800), 1, 0)
        window._index = 3                               # lean closer
        window._face_present = True
        window._head_x, window._head_z = 0.0, 10.0
        assert window.posture_progress() is None, "claimed progress with no baseline"
        assert not window.posture_held

        window._rest_samples.extend([(0.0, 10.0)] * 20)
        assert window.posture_progress() == pytest.approx(0.0, abs=0.01)
        window._head_z = window.posture.target_distance(10.0)
        assert window.posture_progress() == pytest.approx(1.0, abs=0.01)
        assert window.posture_held

    def test_leaning_the_wrong_way_reads_as_negative_progress(self, qapp, tmp_path):
        from src.gui.calibration_window import CalibrationWindow

        config = Config.load(user_path=tmp_path / "c.json")
        window = CalibrationWindow(config, Rect(0, 0, 1000, 800), 1, 0)
        window._index = 3                               # lean closer
        window._face_present = True
        window._rest_samples.extend([(0.0, 10.0)] * 20)
        window._head_x, window._head_z = 0.0, 11.0      # further away instead
        assert window.posture_progress() < 0
        assert not window.posture_held

    def test_a_neutral_posture_does_not_pollute_the_baseline(self, qapp, tmp_path):
        """Only postures that ask for nothing define what "normal" is.

        Otherwise a lean would redefine the resting position and then be
        measured against it, and always read as already held.
        """
        from src.gui.calibration_window import CalibrationWindow
        from src.tracking.features import FeatureVector
        from src.tracking.pipeline import GazeSample

        config = Config.load(user_path=tmp_path / "c.json")
        window = CalibrationWindow(config, Rect(0, 0, 1000, 800), 1, 0)

        def feed(index, head_z):
            window._index = index
            features = FeatureVector(values={"head_x": 0.0, "head_z": head_z},
                                     valid=True)
            window.on_sample(GazeSample(timestamp=0.0, face_detected=True,
                                        features=features))

        for _ in range(20):
            feed(0, 10.0)                                # neutral
        for _ in range(40):
            feed(3, 8.8)                                 # leaning in
        assert window.resting_position[1] == pytest.approx(10.0, abs=0.01)

    def test_the_posture_reminder_stays_next_to_the_dot(self, qapp, tmp_path):
        """It is for peripheral vision, so it must not sit across the screen."""
        from PySide6.QtGui import QPixmap
        from src.gui.calibration_window import (CalibrationWindow, PHASE_COLLECT,
                                                PHASE_SETTLE)

        config = Config.load(user_path=tmp_path / "c.json")
        window = CalibrationWindow(config, Rect(0, 0, 800, 600), 1, 0)
        window.resize(800, 600)
        window._face_present = True
        window._head_roll = 10.0
        window._rest_samples.extend([(0.0, 10.0)] * 20)
        window._head_x, window._head_z = 0.0, 10.0
        for phase in (PHASE_SETTLE, PHASE_COLLECT):
            for index in range(len(window._targets)):
                window._phase, window._index = phase, index
                window.render(QPixmap(800, 600))

    def test_the_session_uses_the_configured_feature_set(self, qapp, tmp_path):
        """A profile fitted on features the live pipeline does not produce is
        silently useless, so the two have to come from the same config key."""
        from src.gui.calibration_window import CalibrationWindow
        from src.tracking.features import FEATURE_NAMES

        config = Config.load(user_path=tmp_path / "c.json")
        window = CalibrationWindow(config, Rect(0, 0, 1920, 1080), 1, 0)
        names = window._session.feature_names
        assert names == list(config.get("calibration.model_features"))
        assert set(names) <= set(FEATURE_NAMES), "config names an unknown feature"


class TestBoardSelector:
    def test_square_lock_produces_a_square(self, qapp):
        from PySide6.QtCore import QPoint
        from src.gui.board_selector import BoardSelector

        selector = BoardSelector(Rect(0, 0, 1920, 1080))
        selector._origin = QPoint(100, 100)
        selector._current = QPoint(500, 420)
        selector._square_lock = True
        rect = selector._selection()
        assert rect.width() == rect.height() == 320

    def test_dragging_up_and_left_still_works(self, qapp):
        from PySide6.QtCore import QPoint
        from src.gui.board_selector import BoardSelector

        selector = BoardSelector(Rect(0, 0, 1920, 1080))
        selector._origin = QPoint(500, 500)
        selector._current = QPoint(100, 180)
        selector._square_lock = True
        rect = selector._selection()
        assert rect.width() == rect.height() == 320
        # The rectangle stays anchored at the point the drag started.
        assert rect.x() + rect.width() == 500
        assert rect.y() + rect.height() == 500


class TestSessionWindow:
    def test_it_renders_a_recorded_session(self, qapp, populated_db):
        from src.gui.session_window import SessionWindow

        window = SessionWindow(populated_db)
        assert window._list.count() == 1
        assert len(window._events) == 3
        assert window._stats.away_event_count == 1
        assert "SESSION SUMMARY" in window._summary.toPlainText()
        assert "e4" in window._summary.toPlainText()

    def test_board_heatmap_grid_is_oriented_correctly(self, qapp, populated_db):
        from src.gui.session_window import SessionWindow

        window = SessionWindow(populated_db)
        grid = window._square_seconds_grid()
        # e4 -> file index 4, rank 4 -> row index 8 - 4 = 4
        assert grid[4, 4] == pytest.approx(8.0)
        assert grid.sum() == pytest.approx(8.0)

    def test_an_empty_database_does_not_crash(self, qapp, tmp_path):
        from src.gui.session_window import SessionWindow

        database = Database(tmp_path / "empty.db")
        database.connect()
        window = SessionWindow(database)
        assert "No sessions" in window._summary.toPlainText()
        database.close()


class TestSettingsWindow:
    def test_values_round_trip_through_the_config(self, qapp, tmp_path):
        from src.gui.settings_window import SettingsWindow
        from src.tracking.camera import CameraInfo
        from src.utils.screens import ScreenInfo

        config = Config.load(user_path=tmp_path / "s.json")
        dialog = SettingsWindow(config, [CameraInfo(0, "Cam", 1280, 720)],
                                [ScreenInfo(1, "Main", Rect(0, 0, 1920, 1080), 1.0)])
        dialog.dot_radius_spin.setValue(31)
        dialog.away_spin.setValue(750)
        dialog._accept()

        assert config.get("overlay.dot_radius") == 31
        assert config.get("events.away_threshold_ms") == 750
        assert dialog.restart_required is False

    def test_changing_the_camera_flags_a_restart(self, qapp, tmp_path):
        from src.gui.settings_window import SettingsWindow
        from src.tracking.camera import CameraInfo
        from src.utils.screens import ScreenInfo

        config = Config.load(user_path=tmp_path / "s2.json")
        dialog = SettingsWindow(config, [CameraInfo(0, "A", 1280, 720)],
                                [ScreenInfo(1, "Main", Rect(0, 0, 1920, 1080), 1.0)])
        dialog.fps_spin.setValue(15)
        dialog._accept()
        assert dialog.restart_required is True


class TestTray:
    def test_actions_reflect_the_tracking_state(self, qapp):
        from src.gui.tray import SystemTray, make_icon

        assert not make_icon(True).isNull()
        tray = SystemTray()
        tray.set_tracking(True)
        assert tray._start_action.isEnabled() is False
        assert tray._stop_action.isEnabled() is True
        tray.set_tracking(False)
        assert tray._start_action.isEnabled() is True


class TestMainWindow:
    """Constructs the real main window with the hardware probes stubbed out."""

    @pytest.fixture()
    def window(self, qapp, tmp_path, monkeypatch):
        import src.gui.main_window as module
        from src.tracking.camera import CameraInfo
        from src.utils.screens import ScreenInfo

        monkeypatch.setattr(module, "enumerate_cameras",
                            lambda: [CameraInfo(0, "Test Cam", 1280, 720)])
        monkeypatch.setattr(module, "list_screens",
                            lambda: [ScreenInfo(1, "Main", Rect(0, 0, 1920, 1080), 1.0)])
        monkeypatch.setattr(module, "REGIONS_PATH", tmp_path / "regions.json")

        config = Config.load(user_path=tmp_path / "cfg.json")
        database = Database(tmp_path / "main.db")
        database.connect()
        window = module.MainWindow(config, database, debug=True)
        yield window
        window._shutting_down = True
        database.close()

    def test_it_builds_with_devices_listed(self, window):
        assert window.camera_combo.count() == 1
        assert window.monitor_combo.count() == 1
        assert window.start_button.isEnabled()
        assert not window.stop_button.isEnabled()

    def test_a_sample_is_classified_and_shown(self, window):
        import time
        from src.tracking.head_pose import HeadPose
        from src.tracking.pipeline import GazeSample

        window._apply_board(Rect(400, 100, 800, 800))
        sample = GazeSample(timestamp=time.monotonic(), face_detected=True,
                            features_valid=True, calibrated=True, x=450.0, y=150.0,
                            confidence=0.9, valid=True, head_pose=HeadPose(1, 2, 3),
                            fps=30.0)
        window._on_sample(sample)
        window._refresh_status()

        assert window._last_square == "a8"
        assert window._last_region == "Chessboard"
        assert window._value_labels["square"].text() == "a8"
        assert len(window._debug_lines(sample)) > 5

    def test_setting_the_board_also_suggests_clock_regions(self, window):
        window._apply_board(Rect(400, 100, 800, 800))
        names = {region.name for region in window._regions.regions}
        assert "Chessboard" in names
        assert "White Clock" in names and "Black Clock" in names

    def test_visualization_toggles_persist_to_the_config(self, window):
        window.dot_check.setChecked(False)
        window.board_check.setChecked(True)
        assert window._config.get("overlay.show_gaze_dot") is False
        assert window._config.get("overlay.show_board_rect") is True

    def test_no_overlay_flag_suppresses_the_dot(self, qapp, tmp_path, monkeypatch):
        import src.gui.main_window as module
        from src.tracking.camera import CameraInfo
        from src.utils.screens import ScreenInfo

        monkeypatch.setattr(module, "enumerate_cameras",
                            lambda: [CameraInfo(0, "Cam", 640, 480)])
        monkeypatch.setattr(module, "list_screens",
                            lambda: [ScreenInfo(1, "Main", Rect(0, 0, 1920, 1080), 1.0)])
        monkeypatch.setattr(module, "REGIONS_PATH", tmp_path / "r.json")

        database = Database(":memory:")
        database.connect()
        window = module.MainWindow(Config.load(user_path=tmp_path / "c2.json"),
                                   database, no_overlay=True)
        window.dot_check.setChecked(True)
        window._sync_overlay_visibility()
        assert window._overlay._show_dot is False
        window._shutting_down = True
        database.close()

    def test_calibration_is_refused_without_a_running_tracker(self, window, monkeypatch):
        from PySide6.QtWidgets import QMessageBox
        shown = []
        monkeypatch.setattr(QMessageBox, "information",
                            lambda *args, **kwargs: shown.append(args))
        window.run_calibration()
        assert shown, "the user must be told to start tracking first"


class TestHeatmapOverlay:
    """The live heatmap, from toggle through to painted pixels."""

    def test_overlay_paints_both_heatmap_layers(self, qapp):
        from PySide6.QtGui import QPixmap
        from src.visualization.gaze_overlay import GazeOverlay
        from src.visualization.heatmap import MODE_BOTH

        overlay = GazeOverlay(Rect(0, 0, 800, 600))
        overlay.set_layers(True, False, False, False, show_heatmap=True)
        overlay.set_heatmap_mode(MODE_BOTH)
        overlay.set_board(Rect(100, 100, 400, 400))
        overlay.set_board_heatmap({"e4": 12.0, "d5": 6.0, "a1": 0.2})
        overlay.set_heatmap_image(QPixmap(96, 54))
        overlay.set_gaze(300, 250, 0.9, True)
        overlay.render(QPixmap(800, 600))

    def test_heatmap_layer_counts_as_visible(self, qapp):
        from src.visualization.gaze_overlay import GazeOverlay

        overlay = GazeOverlay(Rect(0, 0, 800, 600))
        overlay.set_layers(False, False, False, False, show_heatmap=True)
        assert overlay.any_layer_visible is True
        overlay.set_layers(False, False, False, False, show_heatmap=False)
        assert overlay.any_layer_visible is False

    def test_board_heatmap_with_no_board_is_safe(self, qapp):
        from PySide6.QtGui import QPixmap
        from src.visualization.gaze_overlay import GazeOverlay
        from src.visualization.heatmap import MODE_BOARD

        overlay = GazeOverlay(Rect(0, 0, 800, 600))
        overlay.set_layers(False, False, False, False, show_heatmap=True)
        overlay.set_heatmap_mode(MODE_BOARD)
        overlay.set_board_heatmap({"e4": 5.0})   # board rect never set
        overlay.render(QPixmap(800, 600))

    def test_invalid_square_names_are_skipped(self, qapp):
        from PySide6.QtGui import QPixmap
        from src.visualization.gaze_overlay import GazeOverlay

        overlay = GazeOverlay(Rect(0, 0, 800, 600))
        overlay.set_layers(False, False, False, False, show_heatmap=True)
        overlay.set_board(Rect(0, 0, 400, 400))
        overlay.set_board_heatmap({"zz": 3.0, "e4": 4.0})
        overlay.render(QPixmap(800, 600))


class TestHeatmapInMainWindow:
    @pytest.fixture()
    def window(self, qapp, tmp_path, monkeypatch):
        import src.gui.main_window as module
        from src.tracking.camera import CameraInfo
        from src.utils.screens import ScreenInfo

        monkeypatch.setattr(module, "enumerate_cameras",
                            lambda: [CameraInfo(0, "Cam", 1280, 720)])
        monkeypatch.setattr(module, "list_screens",
                            lambda: [ScreenInfo(1, "Main", Rect(0, 0, 1920, 1080), 1.0)])
        monkeypatch.setattr(module, "REGIONS_PATH", tmp_path / "regions.json")

        database = Database(tmp_path / "hm.db")
        database.connect()
        window = module.MainWindow(Config.load(user_path=tmp_path / "hm.json"), database)
        yield window
        window._shutting_down = True
        database.close()

    def _look_at(self, window, x, y, square, count=20, start=100.0, step=0.05):
        from src.tracking.head_pose import HeadPose
        from src.tracking.pipeline import GazeSample

        window._apply_board(Rect(400, 100, 800, 800))
        for i in range(count):
            sample = GazeSample(timestamp=start + i * step, face_detected=True,
                                features_valid=True, calibrated=True, x=x, y=y,
                                confidence=0.9, valid=True,
                                head_pose=HeadPose(0, 0, 0), fps=30.0)
            window._on_sample(sample)

    def test_looking_at_a_square_accumulates_there(self, window):
        from src.utils.geometry import square_rect

        centre = square_rect("e4", 400, 100, 800, 800).center
        self._look_at(window, centre[0], centre[1], "e4")

        assert window._last_square == "e4"
        squares = window._heatmap.square_seconds
        assert squares.get("e4", 0) > 0.5
        assert max(squares, key=squares.get) == "e4"

    def test_screen_peak_matches_where_the_user_looked(self, window):
        self._look_at(window, 700.0, 400.0, None)
        x, y = window._heatmap.peak_position()
        assert x == pytest.approx(700, abs=40)
        assert y == pytest.approx(400, abs=40)

    def test_a_stall_cannot_dump_time_onto_one_square(self, window):
        """A long gap between frames must be clamped, not credited in full."""
        from src.utils.geometry import square_rect

        centre = square_rect("e4", 400, 100, 800, 800).center
        self._look_at(window, centre[0], centre[1], "e4", count=2, step=30.0)
        assert window._heatmap.square_seconds.get("e4", 0) <= 0.25

    def test_toggle_persists_and_reaches_the_overlay(self, window):
        window.heatmap_check.setChecked(True)
        assert window._config.get("overlay.show_heatmap") is True
        assert window._overlay._show_heatmap is True
        window.heatmap_check.setChecked(False)
        assert window._overlay._show_heatmap is False

    def test_mode_selection_reaches_the_overlay(self, window):
        index = window.heatmap_mode_combo.findData("board")
        window.heatmap_mode_combo.setCurrentIndex(index)
        assert window._overlay._heatmap_mode == "board"

    def test_clear_empties_the_map(self, window):
        self._look_at(window, 700.0, 400.0, None)
        assert not window._heatmap.is_empty
        window.clear_heatmap()
        assert window._heatmap.is_empty
        assert window._heatmap.square_seconds == {}

    def test_rendering_produces_a_scaled_pixmap(self, window):
        window.heatmap_check.setChecked(True)
        self._look_at(window, 700.0, 400.0, None)
        window._last_heatmap_render = 0.0
        window._refresh_heatmap_overlay()

        pixmap = window._overlay._heatmap_pixmap
        assert pixmap is not None and not pixmap.isNull()
        assert pixmap.width() == int(window._screen_rect.width)
        assert window._overlay._board_cells is not None

    def test_rendering_is_throttled(self, window):
        import time
        window.heatmap_check.setChecked(True)
        self._look_at(window, 700.0, 400.0, None)
        window._last_heatmap_render = time.monotonic()
        window._overlay.set_heatmap_image(None)
        window._refresh_heatmap_overlay()
        assert window._overlay._heatmap_pixmap is None, "should have been skipped"

    def test_changing_monitor_resets_the_map(self, window, monkeypatch):
        import src.gui.main_window as module
        from src.utils.screens import ScreenInfo

        self._look_at(window, 700.0, 400.0, None)
        monkeypatch.setattr(module, "list_screens",
                            lambda: [ScreenInfo(1, "Main", Rect(0, 0, 1280, 720), 1.0)])
        window._monitors = module.list_screens()
        window._on_monitor_changed()
        assert window._heatmap.is_empty
        assert window._heatmap.rect.width == 1280
