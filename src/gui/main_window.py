"""The main application window.

This is the only component that knows about all the others. It owns the two
worker threads, the overlay, the database and the event pipeline, and it is
responsible for shutting all of them down cleanly.

Threading contract
------------------
* ``TrackerWorker`` and ``ScreenWorker`` run on their own threads and only
  communicate through queued signals.
* Everything in this file runs on the GUI thread. Per-sample work here is
  deliberately cheap (a few rectangle tests); anything expensive belongs in a
  worker.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import List, Optional

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction, QCloseEvent, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from ..events.detector import EventDetector, compute_statistics
from ..events.models import (GazeEvent, GazeSampleRecord, Session, SUMMARY_BUCKETS,
                             format_duration)
from ..events.state_machine import AttentionStateMachine, TrackerObservation
from ..screen.chessboard import ChessboardRegion
from ..screen.classifier import GazeRegionClassifier
from ..screen.regions import RegionManager, suggested_clock_regions
from ..screen.worker import ScreenWorker
from ..storage.database import Database
from ..tracking.calibration import CalibrationProfile, CalibrationStore
from ..tracking.camera import CameraInfo, enumerate_cameras
from ..tracking.pipeline import GazeSample
from ..tracking.worker import TrackerWorker
from ..utils.config import Config, PROJECT_ROOT
from ..utils.geometry import Rect, square_rect
from ..utils.screens import ScreenInfo, find_screen, list_screens, qt_screen_for
from ..visualization.debug_overlay import bgr_to_qimage, draw_debug_frame
from ..visualization.gaze_overlay import GazeOverlay, primary_screen_rect
from ..visualization.heatmap import GazeHeatmap, MODE_BOTH, rgba_to_qimage
from .board_selector import BoardSelector
from .calibration_window import CalibrationWindow
from .recentre_window import RecentreWindow
from .session_window import SessionWindow
from .settings_window import SettingsWindow
from .tray import SystemTray, make_icon

logger = logging.getLogger(__name__)

REGIONS_PATH = PROJECT_ROOT / "data" / "regions.json"

DARK_STYLE = """
QMainWindow, QDialog, QWidget { background: #15181d; color: #e6eaf0;
    font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px; }
QGroupBox { border: 1px solid #2c323b; border-radius: 8px; margin-top: 14px;
    padding: 12px 10px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px;
    color: #9aa4b2; font-size: 11px; letter-spacing: 1px; }
QPushButton { background: #232830; border: 1px solid #333a45; border-radius: 6px;
    padding: 8px 14px; }
QPushButton:hover { background: #2b313b; }
QPushButton:pressed { background: #1e232a; }
QPushButton:disabled { color: #5a6472; background: #1b1f25; }
QPushButton#primary { background: #2f7d5b; border-color: #379268; font-weight: 600; }
QPushButton#primary:hover { background: #369068; }
QPushButton#danger { background: #8a3a30; border-color: #a24438; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QListWidget {
    background: #1b1f26; border: 1px solid #2f3641; border-radius: 6px; padding: 5px; }
QTextEdit, QListWidget { font-family: Consolas, 'Cascadia Mono', monospace; }
QLabel#value { font-family: Consolas, monospace; font-size: 15px; color: #ffffff; }
QLabel#caption { color: #8b95a3; font-size: 11px; letter-spacing: 0.5px; }
QLabel#status { font-weight: 600; }
QCheckBox { padding: 3px 0; }
QTabBar::tab { background: #1b1f26; padding: 7px 14px; border: 1px solid #2f3641;
    border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; }
QTabBar::tab:selected { background: #262c35; }
QSplitter::handle { background: #232830; }
"""


class MainWindow(QMainWindow):
    """The application's primary window."""

    def __init__(self, config: Config, database: Database,
                 debug: bool = False, no_overlay: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Eye Tracker")
        self.setWindowIcon(make_icon(False))
        self.setStyleSheet(DARK_STYLE)
        self.setMinimumSize(940, 620)

        self._config = config
        self._db = database
        self._debug = debug
        self._force_no_overlay = no_overlay

        # --- hardware inventories -------------------------------------
        self._cameras: List[CameraInfo] = enumerate_cameras()
        self._monitors: List[ScreenInfo] = list_screens()

        # --- pipeline state -------------------------------------------
        self._calibration_store = CalibrationStore()
        self._profile: Optional[CalibrationProfile] = None
        self._regions = RegionManager()
        self._regions.load(REGIONS_PATH)
        self._screen_rect = self._resolve_screen_rect()
        self._classifier = GazeRegionClassifier(
            self._regions, self._screen_rect,
            orientation=str(config.get("screen.board_orientation", "white_bottom")),
            off_screen_margin=float(config.get("events.off_screen_margin_px", 60)),
        )
        self._detector = self._build_detector()

        self._tracker: Optional[TrackerWorker] = None
        self._screen_worker: Optional[ScreenWorker] = None
        self._session: Optional[Session] = None
        self._session_events: List[GazeEvent] = []
        self._sample_buffer: List[GazeSampleRecord] = []
        self._last_sample: Optional[GazeSample] = None
        self._last_square: Optional[str] = None
        self._last_region: str = "-"
        self._last_sample_log = 0.0
        self._board_confidence = 0.0
        self._shutting_down = False
        #: Accumulated manual bias correction from Quick Recentre.
        self._gaze_offset: tuple = (0.0, 0.0)

        # --- live heatmap ---------------------------------------------
        self._heatmap = GazeHeatmap(
            self._screen_rect,
            sigma_px=float(config.get("overlay.heatmap_sigma_px", 55.0)),
            half_life_seconds=float(config.get("overlay.heatmap_half_life_s", 0.0)),
        )
        self._last_heatmap_render = 0.0
        self._last_heatmap_sample_time: Optional[float] = None

        # --- UI --------------------------------------------------------
        self._build_ui()
        self._overlay = GazeOverlay(self._screen_rect)
        self._apply_overlay_settings()

        self._tray = SystemTray(self)
        self._connect_tray()
        self._tray.show()

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(120)
        self._ui_timer.timeout.connect(self._refresh_status)
        self._ui_timer.start()

        self._load_latest_calibration()
        self._restore_board_from_config()
        self._update_buttons()

    # ================================================================== UI
    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)
        root.addWidget(self._build_left_panel(), 0)
        root.addWidget(self._build_right_panel(), 1)
        self.setCentralWidget(central)
        self._build_menu()
        self.statusBar().showMessage("Ready")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for text, slot in (("Se&ttings...", self.open_settings),
                           ("View &Sessions...", self.open_sessions)):
            action = QAction(text, self)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        file_menu.addSeparator()
        quit_action = QAction("E&xit", self)
        quit_action.triggered.connect(self.quit_application)
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("&Help")
        for text, slot in (("Webcam &positioning", self.show_positioning_help),
                           ("&About", self.show_about)):
            action = QAction(text, self)
            action.triggered.connect(slot)
            help_menu.addAction(action)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(330)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # --- devices ---------------------------------------------------
        devices = QGroupBox("DEVICES")
        device_layout = QVBoxLayout(devices)
        self.camera_combo = QComboBox()
        for camera in self._cameras:
            self.camera_combo.addItem(camera.label(), camera.index)
        if not self._cameras:
            self.camera_combo.addItem("No webcam detected", 0)
            self.camera_combo.setEnabled(False)
        self._select_data(self.camera_combo, int(self._config.get("camera.device_index", 0)))
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)

        self.monitor_combo = QComboBox()
        for monitor in self._monitors:
            self.monitor_combo.addItem(monitor.label(), monitor.index)
        if self.monitor_combo.count() == 0:
            self.monitor_combo.addItem("Primary monitor", 1)
        self._select_data(self.monitor_combo, int(self._config.get("screen.monitor_index", 1)))
        self.monitor_combo.currentIndexChanged.connect(self._on_monitor_changed)

        device_layout.addWidget(self._caption("Camera"))
        device_layout.addWidget(self.camera_combo)
        device_layout.addWidget(self._caption("Monitor"))
        device_layout.addWidget(self.monitor_combo)
        layout.addWidget(devices)

        # --- controls --------------------------------------------------
        controls = QGroupBox("CONTROLS")
        control_layout = QVBoxLayout(controls)
        self.start_button = QPushButton("Start Tracking")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start_tracking)
        self.stop_button = QPushButton("Stop Tracking")
        self.stop_button.setObjectName("danger")
        self.stop_button.clicked.connect(self.stop_tracking)
        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.clicked.connect(self.run_calibration)
        self.detect_button = QPushButton("Detect Chessboard")
        self.detect_button.clicked.connect(self.detect_board)
        self.recentre_button = QPushButton("Quick Recentre")
        self.recentre_button.setToolTip(
            "Corrects drift after you shift position, without a full calibration")
        self.recentre_button.clicked.connect(self.run_recentre)
        self.select_button = QPushButton("Select Chessboard Manually")
        self.select_button.clicked.connect(self.select_board_manually)
        for button in (self.start_button, self.stop_button, self.calibrate_button,
                       self.recentre_button, self.detect_button, self.select_button):
            control_layout.addWidget(button)
        layout.addWidget(controls)

        # --- visualization --------------------------------------------
        visual = QGroupBox("VISUALIZATION")
        visual_layout = QVBoxLayout(visual)
        self.dot_check = QCheckBox("Gaze Dot")
        self.dot_check.setChecked(bool(self._config.get("overlay.show_gaze_dot", True)))
        self.debug_check = QCheckBox("Debug Overlay")
        self.debug_check.setChecked(bool(self._config.get("overlay.show_debug_overlay", False)))
        self.board_check = QCheckBox("Board Region")
        self.board_check.setChecked(bool(self._config.get("overlay.show_board_rect", False)))
        self.regions_check = QCheckBox("Region Boundaries")
        self.regions_check.setChecked(bool(self._config.get("overlay.show_region_rects", False)))
        self.heatmap_check = QCheckBox("Live Heatmap")
        self.heatmap_check.setChecked(bool(self._config.get("overlay.show_heatmap", False)))
        self.preview_check = QCheckBox("Camera Preview")
        self.preview_check.setChecked(self._debug)
        self.landmarks_check = QCheckBox("Face / Eye Landmarks")
        self.landmarks_check.setChecked(self._debug)
        for check in (self.dot_check, self.debug_check, self.board_check,
                      self.regions_check, self.heatmap_check, self.preview_check,
                      self.landmarks_check):
            visual_layout.addWidget(check)
            check.toggled.connect(self._on_visualization_changed)

        heatmap_row = QHBoxLayout()
        self.heatmap_mode_combo = QComboBox()
        for mode, label in (("both", "Screen + Board"), ("screen", "Screen only"),
                            ("board", "Board only")):
            self.heatmap_mode_combo.addItem(label, mode)
        self._select_data(self.heatmap_mode_combo,
                          str(self._config.get("overlay.heatmap_mode", MODE_BOTH)))
        self.heatmap_mode_combo.currentIndexChanged.connect(self._on_visualization_changed)
        clear_button = QPushButton("Clear")
        clear_button.setToolTip("Reset the accumulated heatmap")
        clear_button.clicked.connect(self.clear_heatmap)
        heatmap_row.addWidget(self.heatmap_mode_combo, 1)
        heatmap_row.addWidget(clear_button)
        visual_layout.addLayout(heatmap_row)
        layout.addWidget(visual)

        layout.addStretch(1)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        status = QGroupBox("LIVE")
        grid = QGridLayout(status)
        self._value_labels = {}
        fields = [("status", "Status"), ("gaze", "Gaze X / Y"), ("confidence", "Confidence"),
                  ("region", "Region"), ("square", "Square"), ("state", "Attention state"),
                  ("fps", "FPS"), ("calibration", "Calibration"), ("board", "Board")]
        for index, (key, caption) in enumerate(fields):
            row, column = divmod(index, 3)
            holder = QVBoxLayout()
            holder.setSpacing(2)
            holder.addWidget(self._caption(caption))
            value = QLabel("-")
            value.setObjectName("value")
            holder.addWidget(value)
            self._value_labels[key] = value
            container = QWidget()
            container.setLayout(holder)
            grid.addWidget(container, row, column)
        layout.addWidget(status)

        self.preview_label = QLabel("Camera preview off")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(240)
        self.preview_label.setFrameShape(QFrame.StyledPanel)
        self.preview_label.setStyleSheet(
            "background:#101318; border:1px solid #2c323b; border-radius:8px; color:#6d7684;"
        )
        layout.addWidget(self.preview_label, 1)

        session = QGroupBox("CURRENT SESSION")
        session_layout = QVBoxLayout(session)
        self.session_label = QLabel("No active session")
        self.session_label.setStyleSheet("font-family: Consolas, monospace;")
        session_layout.addWidget(self.session_label)

        buttons = QHBoxLayout()
        view_button = QPushButton("View Sessions")
        view_button.clicked.connect(self.open_sessions)
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self.open_settings)
        buttons.addWidget(view_button)
        buttons.addWidget(settings_button)
        buttons.addStretch(1)
        session_layout.addLayout(buttons)
        layout.addWidget(session)
        return panel

    @staticmethod
    def _caption(text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("caption")
        return label

    @staticmethod
    def _select_data(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _connect_tray(self) -> None:
        self._tray.show_window_requested.connect(self.show_normal_window)
        self._tray.start_requested.connect(self.start_tracking)
        self._tray.stop_requested.connect(self.stop_tracking)
        self._tray.calibrate_requested.connect(self.run_calibration)
        self._tray.sessions_requested.connect(self.open_sessions)
        self._tray.settings_requested.connect(self.open_settings)
        self._tray.exit_requested.connect(self.quit_application)
        self._tray.dot_toggled.connect(self.dot_check.setChecked)
        self._tray.heatmap_toggled.connect(self.heatmap_check.setChecked)
        self._tray.set_dot_checked(self.dot_check.isChecked())
        self._tray.set_heatmap_checked(self.heatmap_check.isChecked())

    # =========================================================== geometry
    def _current_screen(self) -> Optional[ScreenInfo]:
        index = int(self._config.get("screen.monitor_index", 1))
        return find_screen(self._monitors, index)

    def _resolve_screen_rect(self) -> Rect:
        """The active monitor in LOGICAL pixels.

        Everything downstream -- calibration targets, gaze estimates, regions,
        squares and the overlay -- uses this one coordinate space. Screen
        capture converts to physical pixels at the point of grabbing.
        """
        screen = self._current_screen()
        return screen.rect if screen is not None else primary_screen_rect()

    def _build_detector(self) -> EventDetector:
        machine = AttentionStateMachine(
            away_threshold_ms=float(self._config.get("events.away_threshold_ms", 500)),
            region_dwell_ms=float(self._config.get("events.region_dwell_ms", 200)),
            square_dwell_ms=float(self._config.get("events.square_dwell_ms", 250)),
            eyes_closed_ms=float(self._config.get("tracking.eyes_closed_ms", 1200)),
            minimum_confidence=float(self._config.get("tracking.minimum_confidence", 0.45)),
        )
        return EventDetector(machine,
                             min_event_ms=float(self._config.get("events.min_event_ms", 120)))

    # ======================================================== calibration
    def _load_latest_calibration(self) -> None:
        profile = self._calibration_store.latest()
        if profile is None:
            self._set_value("calibration", "None")
            return
        if not profile.matches_screen(int(self._screen_rect.width),
                                      int(self._screen_rect.height)):
            logger.warning("Latest calibration was made for a different resolution")
        self._profile = profile
        if self._tracker is not None:
            self._tracker.set_profile(profile)
        self._set_value("calibration", profile.summary())

    @Slot()
    def run_calibration(self) -> None:
        if self._tracker is None or not self._tracker.isRunning():
            QMessageBox.information(
                self, "Start tracking first",
                "Calibration needs a live camera feed.\n\nPress Start Tracking, then "
                "run Calibrate again."
            )
            return

        self._overlay.hide()
        monitor_index = int(self.monitor_combo.currentData() or 1)
        window = CalibrationWindow(self._config, self._screen_rect, monitor_index,
                                   int(self.camera_combo.currentData() or 0),
                                   qt_screen=qt_screen_for(monitor_index))
        self._tracker.sample_ready.connect(window.on_sample)

        def cleanup() -> None:
            try:
                self._tracker.sample_ready.disconnect(window.on_sample)
            except (RuntimeError, TypeError):  # pragma: no cover - already gone
                pass
            self._sync_overlay_visibility()

        window.completed.connect(lambda profile: (cleanup(),
                                                  self._on_calibration_complete(profile)))
        window.failed.connect(lambda message: (cleanup(),
                                               self._on_calibration_failed(message)))
        window.cancelled.connect(cleanup)
        self._calibration_window = window  # keep a reference alive
        window.start()

    def _on_calibration_complete(self, profile: CalibrationProfile) -> None:
        report = profile.report
        # 0 means "use the screen-relative defaults" (see FitReport.quality).
        good = float(self._config.get("calibration.quality_good_px", 0.0))
        fair = float(self._config.get("calibration.quality_fair_px", 0.0))
        quality = report.quality(good, fair) if report else "UNKNOWN"

        if report:
            lines = [
                f"Holding your gaze steady:   {report.settled_error_px:.0f} px",
                f"Any single frame:           {report.mean_error_px:.0f} px",
            ]
            if report.pose_error_px > 0:
                lines.append(
                    f"At head positions unseen:   {report.pose_error_px:.0f} px")
            lines.append(
                f"Worst calibration point:    {report.max_error_px:.0f} px")
            message = "\n".join(lines) + (
                f"\n\nQuality: {quality}\n\n"
                "The first figure is what you will notice: the error once\n"
                "smoothing has settled on a square you are looking at. The\n"
                "second is one unsmoothed frame, which is noisier by nature.\n"
            )
            if report.pose_error_px > 0:
                message += (
                    "The third is the one that matters when you move: it holds\n"
                    "back a whole range of head positions and measures how the\n"
                    "tracker does at positions it never saw.\n"
                )
            message += (
                "\nThese come from cross-validation, so they estimate accuracy\n"
                "in situations the model was not fitted on. Webcam gaze\n"
                "tracking is an estimate, not a measurement.\n\n"
                f"Collected {report.n_samples} samples over {report.n_points} points."
            )
            if profile.coverage_warnings:
                message += "\n\nWhat went wrong:\n\n"
                message += "\n\n".join(f"- {note}" for note in profile.coverage_warnings)
        else:
            message = "Calibration finished."

        if quality == "POOR":
            box = QMessageBox(self)
            box.setWindowTitle("Calibration quality is low")
            advice = ("\n\nAccuracy at this level may not resolve individual squares "
                      "reliably.")
            if not profile.coverage_warnings:
                # Nothing specific was detectable, so say what usually helps
                # rather than leaving the user with a verdict and no next step.
                advice += (
                    "\n\nNothing specific was detectable in the collection, so the "
                    "usual causes, in order:\n\n"
                    "- Light on your face from the front. A window or lamp behind you "
                    "is the single most common cause.\n"
                    "- Webcam at the top centre of the screen, not off to one side.\n"
                    "- 50-70 cm away, with your whole face in frame even when you look "
                    "at the corners of the screen.\n"
                    "- Glasses catching a reflection; tilting them slightly can help.\n"
                    "- Keep your head moving at every dot, all three ways the "
                    "instructions ask for.")
            box.setText(message + advice)
            again = box.addButton("Calibrate Again", QMessageBox.AcceptRole)
            box.addButton("Use Anyway", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is again:
                QTimer.singleShot(200, self.run_calibration)
                return
        else:
            QMessageBox.information(self, "Calibration complete", message)

        self._calibration_store.save(profile)
        try:
            self._db.record_calibration(profile)
        except Exception:  # pragma: no cover - non-fatal
            logger.exception("Could not record calibration metadata")
        self._profile = profile
        self._gaze_offset = (0.0, 0.0)
        if self._tracker is not None:
            self._tracker.set_profile(profile)
            self._tracker.set_offset(0.0, 0.0)
        self._set_value("calibration", profile.summary())
        self._detector.reset()
        self._classifier.reset()

    def _on_calibration_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Calibration failed", message)

    @Slot()
    def run_recentre(self) -> None:
        """Measure and apply a constant gaze bias from one centre target."""
        if self._tracker is None or not self._tracker.isRunning():
            QMessageBox.information(self, "Start tracking first",
                                    "Quick Recentre needs a live camera feed.")
            return
        if self._profile is None:
            QMessageBox.information(self, "Calibrate first",
                                    "Quick Recentre adjusts an existing calibration.")
            return

        self._overlay.hide()
        monitor_index = int(self.monitor_combo.currentData() or 1)
        window = RecentreWindow(self._screen_rect,
                                qt_screen=qt_screen_for(monitor_index))
        self._tracker.sample_ready.connect(window.on_sample)

        def cleanup() -> None:
            try:
                self._tracker.sample_ready.disconnect(window.on_sample)
            except (RuntimeError, TypeError):  # pragma: no cover
                pass
            self._sync_overlay_visibility()

        window.completed.connect(lambda dx, dy: (cleanup(), self._apply_offset(dx, dy)))
        window.failed.connect(lambda message: (
            cleanup(), QMessageBox.warning(self, "Recentre failed", message)))
        window.cancelled.connect(cleanup)
        self._recentre_window = window   # keep a reference alive
        window.start()

    def _apply_offset(self, dx: float, dy: float) -> None:
        self._gaze_offset = (self._gaze_offset[0] + dx, self._gaze_offset[1] + dy)
        if self._tracker is not None:
            self._tracker.set_offset(*self._gaze_offset)
        self._detector.reset()
        self._classifier.reset()
        self.statusBar().showMessage(
            f"Recentred by {dx:+.0f}, {dy:+.0f} px", 5000)

    # ============================================================ tracking
    @Slot()
    def start_tracking(self) -> None:
        if self._tracker is not None and self._tracker.isRunning():
            return
        if not self._cameras:
            QMessageBox.warning(
                self, "No webcam detected",
                "No webcam detected.\n\nPlease connect a webcam and restart the application."
            )
            return

        self._detector = self._build_detector()
        self._classifier.reset()
        self._session_events.clear()
        self._sample_buffer.clear()
        if bool(self._config.get("overlay.heatmap_reset_per_session", True)):
            self._heatmap.clear()
            self._last_heatmap_sample_time = None

        self._tracker = TrackerWorker(self._config)
        self._tracker.sample_ready.connect(self._on_sample)
        self._tracker.preview_ready.connect(self._on_preview)
        self._tracker.error_occurred.connect(self._on_tracker_error)
        self._tracker.status_changed.connect(self.statusBar().showMessage)
        self._tracker.finished.connect(self._on_tracker_finished)
        if self._profile is not None:
            self._tracker.set_profile(self._profile)
        if self._gaze_offset != (0.0, 0.0):
            self._tracker.set_offset(*self._gaze_offset)
        self._tracker.set_preview_enabled(self.preview_check.isChecked())
        self._tracker.start()

        self._start_screen_worker()
        self._start_session()
        self._sync_overlay_visibility()
        self._tray.set_tracking(True)
        self._update_buttons()
        self.statusBar().showMessage("Tracking started")

    @Slot()
    def stop_tracking(self) -> None:
        if self._tracker is not None:
            self._tracker.stop()
            if not self._tracker.wait(4000):  # pragma: no cover - stuck driver
                logger.warning("Tracker thread did not stop in time; terminating")
                self._tracker.terminate()
                self._tracker.wait(1000)
            self._tracker = None
        self._stop_screen_worker()
        self._end_session()
        self._overlay.clear_gaze()
        self._overlay.hide()
        self._tray.set_tracking(False)
        self._update_buttons()
        self.preview_label.setText("Camera preview off")
        self.statusBar().showMessage("Tracking stopped")

    def _on_tracker_finished(self) -> None:
        self._tray.set_tracking(False)
        self._update_buttons()

    def _on_tracker_error(self, message: str) -> None:
        logger.error("Tracker error: %s", message)
        self.statusBar().showMessage(message, 8000)
        if not self._shutting_down:
            QMessageBox.warning(self, "Tracking problem", message)

    def _update_buttons(self) -> None:
        running = self._tracker is not None and self._tracker.isRunning()
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.calibrate_button.setEnabled(running)
        self.recentre_button.setEnabled(running and self._profile is not None)
        self.camera_combo.setEnabled(not running and bool(self._cameras))

    # ======================================================= screen worker
    def _start_screen_worker(self) -> None:
        if not bool(self._config.get("screen.auto_detect_board", True)):
            return
        screen = self._current_screen()
        if screen is None:
            return
        self._screen_worker = ScreenWorker(
            screen,
            interval=1.0 / max(float(self._config.get("screen.analysis_fps", 2.0)), 0.2),
            redetect_seconds=float(self._config.get("screen.board_redetect_seconds", 20.0)),
        )
        self._screen_worker.board_detected.connect(self._on_board_detected)
        self._screen_worker.board_lost.connect(self._on_board_lost)
        self._screen_worker.error_occurred.connect(
            lambda message: self.statusBar().showMessage(message, 5000)
        )
        self._screen_worker.start()

    def _stop_screen_worker(self) -> None:
        if self._screen_worker is not None:
            self._screen_worker.stop()
            if not self._screen_worker.wait(3000):  # pragma: no cover
                self._screen_worker.terminate()
                self._screen_worker.wait(500)
            self._screen_worker = None

    @Slot()
    def detect_board(self) -> None:
        if self._screen_worker is None:
            self._start_screen_worker()
            if self._screen_worker is None:
                QMessageBox.information(
                    self, "Automatic detection disabled",
                    "Enable 'Detect the board automatically' in Settings, or use "
                    "manual selection."
                )
                return
        self._screen_worker.request_detection()
        self.statusBar().showMessage("Looking for the chessboard...", 3000)

    def _on_board_detected(self, region: ChessboardRegion) -> None:
        current = self._regions.board
        if current is not None and current.rect.iou(region.rect) > 0.97:
            return
        self._board_confidence = region.confidence
        self._apply_board(region.rect, persist=True)
        self.statusBar().showMessage(
            f"Chessboard detected ({region.confidence * 100:.0f}% confidence)", 4000
        )

    def _on_board_lost(self) -> None:
        if self._regions.board is None:
            self._set_value("board", "Not found")

    @Slot()
    def select_board_manually(self) -> None:
        self._overlay.hide()
        selector = BoardSelector(self._screen_rect,
                                 self._regions.board.rect if self._regions.board else None)
        selector.selected.connect(self._on_board_selected)
        selector.cancelled.connect(self._sync_overlay_visibility)
        self._board_selector = selector  # keep alive
        selector.start()

    def _on_board_selected(self, rect: Rect) -> None:
        self._board_confidence = 1.0
        self._apply_board(rect, persist=True)
        self._sync_overlay_visibility()
        self.statusBar().showMessage("Chessboard set manually", 4000)

    def _apply_board(self, rect: Rect, persist: bool = False) -> None:
        self._regions.set_board(rect)
        if not any(region.kind == "clock" for region in self._regions.regions):
            for region in suggested_clock_regions(rect, self._screen_rect):
                self._regions.add(region)
        self._classifier.reset()
        self._overlay.set_board(rect)
        self._overlay.set_regions([(r.name, r.rect) for r in self._regions.regions
                                   if r.name != RegionManager.BOARD_NAME])
        if persist:
            self._regions.save(REGIONS_PATH)
            self._config.set("screen.board_rect", list(rect.as_tuple()))
            self._config.save()

    def _restore_board_from_config(self) -> None:
        stored = self._config.get("screen.board_rect")
        if self._regions.board is not None:
            self._apply_board(self._regions.board.rect)
        elif isinstance(stored, list) and len(stored) == 4:
            self._apply_board(Rect(*[float(v) for v in stored]))

    # ============================================================ sessions
    def _start_session(self) -> None:
        now = datetime.now()
        session = Session(
            session_id=now.strftime("%Y-%m-%d_%H-%M-%S"),
            start_wall_clock=now,
            start_monotonic=time.monotonic(),
            calibration_id=self._profile.profile_id if self._profile else None,
            monitor_index=int(self.monitor_combo.currentData() or 1),
        )
        try:
            self._db.create_session(session)
        except Exception as exc:  # pragma: no cover - disk failure
            logger.exception("Could not create session record")
            QMessageBox.warning(self, "Database error",
                                f"Session logging is disabled: {exc}")
            return
        self._session = session
        logger.info("Session %s started", session.session_id)

    def _end_session(self) -> None:
        session = self._session
        if session is None:
            return
        now = time.monotonic()
        final_event = self._detector.flush(now)
        if final_event is not None:
            self._store_event(final_event)
        self._flush_samples()

        session.end_monotonic = now
        session.end_wall_clock = datetime.now()
        try:
            self._db.finish_session(session)
        except Exception:  # pragma: no cover
            logger.exception("Could not finalise session record")
        logger.info("Session %s ended after %.1f s", session.session_id, session.duration)
        self._session = None

    def _store_event(self, event: GazeEvent) -> None:
        self._session_events.append(event)
        if self._session is None or self._session.db_id is None:
            return
        try:
            self._db.insert_event(self._session.db_id, event, self._session.start_monotonic)
        except Exception:  # pragma: no cover
            logger.exception("Could not store event")

    def _flush_samples(self) -> None:
        if not self._sample_buffer or self._session is None or self._session.db_id is None:
            self._sample_buffer.clear()
            return
        try:
            self._db.insert_samples(self._session.db_id, self._sample_buffer,
                                    self._session.start_monotonic)
        except Exception:  # pragma: no cover
            logger.exception("Could not store gaze samples")
        self._sample_buffer.clear()

    # ============================================================== samples
    @Slot(object)
    def _on_sample(self, sample: GazeSample) -> None:
        self._last_sample = sample

        classification = None
        if sample.valid:
            classification = self._classifier.classify(sample.x, sample.y)
            self._last_region = classification.region_name
            self._last_square = classification.square
        elif sample.face_detected:
            self._last_region = "-"
            self._last_square = None

        observation = TrackerObservation(
            timestamp=sample.timestamp,
            face_detected=sample.face_detected,
            eyes_closed=sample.eyes_closed,
            confidence=sample.confidence,
            valid=sample.valid,
            calibrated=sample.calibrated,
            classification=classification,
        )
        finished = self._detector.process(observation, sample.x, sample.y)
        if finished is not None:
            self._store_event(finished)

        if sample.valid:
            self._accumulate_heatmap(sample)
            self._overlay.set_gaze(sample.x, sample.y, sample.confidence, True)
            board = self._regions.board
            if self._last_square and board is not None and self.dot_check.isChecked():
                self._overlay.set_square_rect(square_rect(
                    self._last_square, board.x, board.y, board.width, board.height,
                    self._classifier.orientation))
            else:
                self._overlay.set_square_rect(None)
            self._maybe_log_sample(sample)
        elif sample.face_detected:
            self._overlay.set_gaze(sample.x, sample.y, 0.0, False)
        else:
            self._overlay.clear_gaze()

    def _accumulate_heatmap(self, sample: GazeSample) -> None:
        """Feed one estimate into the live heatmap.

        Board squares accumulate *time* rather than sample count, so the map
        does not change meaning when the camera frame rate varies.
        """
        previous = self._last_heatmap_sample_time
        self._last_heatmap_sample_time = sample.timestamp
        # Weight by confidence: an uncertain estimate should not stain the map
        # as strongly as a confident one.
        self._heatmap.add(sample.x, sample.y, weight=max(sample.confidence, 0.05),
                          timestamp=sample.timestamp)
        if previous is None:
            return
        # Clamp the gap so a pause or a stall cannot dump a huge lump of time
        # onto whichever square happened to be current.
        delta = min(max(sample.timestamp - previous, 0.0), 0.25)
        self._accumulate_square_time(delta)

    def _accumulate_square_time(self, delta: float) -> None:
        if self._last_square and delta > 0:
            self._heatmap.add_square_time(self._last_square, delta)

    @Slot()
    def clear_heatmap(self) -> None:
        self._heatmap.clear()
        self._last_heatmap_sample_time = None
        self._overlay.set_heatmap_image(None)
        self._overlay.set_board_heatmap({}, self._classifier.orientation)
        self.statusBar().showMessage("Heatmap cleared", 2500)

    def _refresh_heatmap_overlay(self) -> None:
        """Re-render the heatmap at a low rate; painting stays a cheap blit.

        Regenerating the image on every frame would be wasteful -- the map
        barely changes in 33 ms -- so it is rebuilt a few times a second and
        cached as a pixmap.
        """
        if not self.heatmap_check.isChecked() or not self._heatmap.dirty:
            return
        now = time.monotonic()
        interval = float(self._config.get("overlay.heatmap_refresh_s", 0.4))
        if now - self._last_heatmap_render < interval:
            return
        self._last_heatmap_render = now

        opacity = float(self._config.get("overlay.heatmap_opacity", 0.55))
        image = rgba_to_qimage(self._heatmap.to_rgba(opacity=opacity))
        # Scale once here, smoothly, so the coarse grid reads as a soft cloud.
        scaled = QPixmap.fromImage(image).scaled(
            max(int(self._screen_rect.width), 1), max(int(self._screen_rect.height), 1),
            Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self._overlay.set_heatmap_image(scaled)
        self._overlay.set_board_heatmap(self._heatmap.square_seconds,
                                        self._classifier.orientation)
        self._heatmap.mark_clean()

    def _maybe_log_sample(self, sample: GazeSample) -> None:
        if not bool(self._config.get("privacy.store_gaze_samples", True)):
            return
        rate = float(self._config.get("tracking.sample_log_hz", 10.0))
        if rate <= 0:
            return
        if sample.timestamp - self._last_sample_log < 1.0 / rate:
            return
        self._last_sample_log = sample.timestamp
        self._sample_buffer.append(GazeSampleRecord(
            sample.timestamp, sample.x, sample.y, sample.confidence, self._last_square
        ))
        if len(self._sample_buffer) >= 200:
            self._flush_samples()

    @Slot(object)
    def _on_preview(self, frame) -> None:
        if not self.preview_check.isChecked():
            return
        sample = self._last_sample
        annotated = draw_debug_frame(
            frame, sample,
            landmarks=sample.landmarks if (sample and self.landmarks_check.isChecked())
            else None,
            show_face=self.landmarks_check.isChecked(),
            show_eyes=self.landmarks_check.isChecked(),
            show_pose=self.landmarks_check.isChecked(),
            mirror=bool(self._config.get("camera.mirror_preview", True)),
        )
        pixmap = QPixmap.fromImage(bgr_to_qimage(annotated))
        self.preview_label.setPixmap(pixmap.scaled(
            self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ============================================================== status
    def _refresh_status(self) -> None:
        sample = self._last_sample
        running = self._tracker is not None and self._tracker.isRunning()

        if not running:
            self._set_value("status", "Idle")
        elif sample is None:
            self._set_value("status", "Starting...")
        elif not sample.face_detected:
            self._set_value("status", "Face not detected")
        elif not sample.calibrated:
            self._set_value("status", "Not calibrated")
        elif sample.confidence < float(self._config.get("tracking.minimum_confidence", 0.45)):
            self._set_value("status", "Low confidence")
        else:
            self._set_value("status", "Tracking")

        if sample is not None:
            self._set_value("gaze", f"{sample.x:.0f}, {sample.y:.0f}"
                            if sample.valid else "-")
            self._set_value("confidence", f"{sample.confidence * 100:.0f}%")
            self._set_value("fps", f"{sample.fps:.1f}")
        self._set_value("region", self._last_region or "-")
        self._set_value("square", self._last_square or "-")
        self._set_value("state", self._detector.state.label())

        board = self._regions.board
        self._set_value("board", f"{int(board.width)} px" if board else "Not set")

        if self.debug_check.isChecked() and sample is not None:
            self._overlay.set_debug_lines(self._debug_lines(sample))
        self._refresh_heatmap_overlay()
        self._refresh_session_summary()

    def _debug_lines(self, sample: GazeSample) -> List[str]:
        pose = sample.head_pose
        board = self._regions.board
        return [
            f"FPS         {sample.fps:5.1f}",
            f"Face        {'yes' if sample.face_detected else 'no'}",
            f"Gaze        {sample.x:7.0f}, {sample.y:.0f}",
            f"Raw         {sample.raw_x:7.0f}, {sample.raw_y:.0f}",
            f"Confidence  {sample.confidence * 100:5.1f}%",
            f"Head Y/P/R  {pose.yaw:+5.1f} {pose.pitch:+5.1f} {pose.roll:+5.1f}",
            f"Openness    {sample.eye_openness:5.3f}",
            f"Region      {self._last_region}",
            f"Square      {self._last_square or '-'}",
            f"State       {self._detector.state.value}",
            f"Board conf  {self._board_confidence * 100:5.1f}%"
            if board else "Board       not set",
        ]

    def _refresh_session_summary(self) -> None:
        if self._session is None:
            self.session_label.setText("No active session")
            return
        elapsed = time.monotonic() - self._session.start_monotonic
        stats = compute_statistics(self._session_events, duration=elapsed)
        parts = [f"{self._session.session_id}   {format_duration(elapsed)}"]
        for bucket in SUMMARY_BUCKETS:
            parts.append(f"  {bucket.capitalize():<8}{stats.bucket_percent(bucket):5.1f}%")
        parts.append(f"  Look-aways: {stats.away_event_count}"
                     f"   Events: {stats.event_count}")
        self.session_label.setText("\n".join(parts))

    def _set_value(self, key: str, text: str) -> None:
        label = self._value_labels.get(key)
        if label is not None and label.text() != text:
            label.setText(text)

    # ========================================================== overlay/UI
    def _apply_overlay_settings(self) -> None:
        self._overlay.configure(
            int(self._config.get("overlay.dot_radius", 12)),
            float(self._config.get("overlay.dot_opacity", 0.75)),
            str(self._config.get("overlay.dot_color", "#ff2d2d")),
        )
        self._overlay.set_screen_rect(self._screen_rect)

    def _on_visualization_changed(self) -> None:
        self._config.set("overlay.show_gaze_dot", self.dot_check.isChecked())
        self._config.set("overlay.show_debug_overlay", self.debug_check.isChecked())
        self._config.set("overlay.show_board_rect", self.board_check.isChecked())
        self._config.set("overlay.show_region_rects", self.regions_check.isChecked())
        self._config.set("overlay.show_heatmap", self.heatmap_check.isChecked())
        self._config.set("overlay.heatmap_mode", self.heatmap_mode_combo.currentData())
        self._config.save()

        self._overlay.set_heatmap_mode(str(self.heatmap_mode_combo.currentData()))
        self._overlay.set_layers(
            self.dot_check.isChecked() and not self._force_no_overlay,
            self.debug_check.isChecked(),
            self.board_check.isChecked(),
            self.regions_check.isChecked(),
            self.heatmap_check.isChecked(),
        )
        self._tray.set_dot_checked(self.dot_check.isChecked())
        self._tray.set_heatmap_checked(self.heatmap_check.isChecked())
        if self._tracker is not None:
            self._tracker.set_preview_enabled(self.preview_check.isChecked())
        if not self.preview_check.isChecked():
            self.preview_label.clear()
            self.preview_label.setText("Camera preview off")
        self._sync_overlay_visibility()

    def _sync_overlay_visibility(self) -> None:
        running = self._tracker is not None and self._tracker.isRunning()
        self._overlay.set_heatmap_mode(str(self.heatmap_mode_combo.currentData()))
        self._overlay.set_layers(
            self.dot_check.isChecked() and not self._force_no_overlay,
            self.debug_check.isChecked(),
            self.board_check.isChecked(),
            self.regions_check.isChecked(),
            self.heatmap_check.isChecked(),
        )
        if running and self._overlay.any_layer_visible:
            self._overlay.show()
            self._overlay.raise_()
        else:
            self._overlay.hide()

    def _on_camera_changed(self) -> None:
        self._config.set("camera.device_index", int(self.camera_combo.currentData() or 0))
        self._config.save()

    def _on_monitor_changed(self) -> None:
        index = int(self.monitor_combo.currentData() or 1)
        self._config.set("screen.monitor_index", index)
        self._config.save()
        self._screen_rect = self._resolve_screen_rect()
        self._classifier.set_screen(self._screen_rect)
        self._heatmap.set_rect(self._screen_rect)
        self._apply_overlay_settings()
        screen = self._current_screen()
        if self._screen_worker is not None and screen is not None:
            self._screen_worker.set_screen(screen)

    # ============================================================= dialogs
    @Slot()
    def open_settings(self) -> None:
        dialog = SettingsWindow(self._config, self._cameras, self._monitors, self)
        if dialog.exec() != SettingsWindow.Accepted:
            return
        self._select_data(self.camera_combo, int(self._config.get("camera.device_index", 0)))
        self._select_data(self.monitor_combo, int(self._config.get("screen.monitor_index", 1)))
        self._screen_rect = self._resolve_screen_rect()
        self._classifier.set_screen(self._screen_rect)
        self._classifier.set_orientation(
            str(self._config.get("screen.board_orientation", "white_bottom")))
        self._apply_overlay_settings()
        self._detector = self._build_detector()
        if dialog.restart_required and self._tracker is not None:
            self.statusBar().showMessage("Restarting tracking with new camera settings...")
            self.stop_tracking()
            QTimer.singleShot(400, self.start_tracking)

    @Slot()
    def open_sessions(self) -> None:
        window = SessionWindow(self._db, self,
                               self._session.db_id if self._session else None)
        window.exec()

    @Slot()
    def show_positioning_help(self) -> None:
        QMessageBox.information(
            self, "Webcam positioning",
            "Recommended setup\n\n"
            "        Webcam\n"
            "           o\n"
            "           |\n"
            "       +---+---+\n"
            "       |       |\n"
            "       | Face  |\n"
            "       |       |\n"
            "       +-------+\n\n"
            "  - Put the webcam at the top centre of the monitor you calibrate on.\n"
            "  - Sit roughly 50-70 cm away and keep your face centred in frame.\n"
            "  - Use even, front-facing light; avoid a bright window behind you.\n"
            "  - Tilt glasses slightly if reflections cover your eyes.\n"
            "  - Recalibrate if you move your chair, your monitor or the camera."
        )

    @Slot()
    def show_about(self) -> None:
        QMessageBox.information(
            self, "About Eye Tracker",
            "Eye Tracker - webcam gaze analysis for chess study\n\n"
            "All processing is local. No webcam video or screenshots are saved and "
            "nothing is uploaded.\n\n"
            "The application observes only. It never interacts with chess.com: no "
            "clicking, no automation, no reading of game state.\n\n"
            "Webcam gaze estimation is approximate. Always read the reported "
            "confidence and calibration error before drawing conclusions."
        )

    # ============================================================ lifecycle
    @Slot()
    def show_normal_window(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    @Slot()
    def quit_application(self) -> None:
        self._shutting_down = True
        self.stop_tracking()
        self._overlay.close()
        self._tray.hide()
        QApplication.quit()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._shutting_down:
            event.accept()
            return
        if bool(self._config.get("ui.minimize_to_tray", True)) and self._tray.isVisible():
            event.ignore()
            self.hide()
            self._tray.showMessage("Eye Tracker",
                                   "Still running in the tray. Right-click the icon to exit.",
                                   make_icon(self._tracker is not None), 3000)
            return
        answer = QMessageBox.question(self, "Exit Eye Tracker", "Stop tracking and exit?")
        if answer == QMessageBox.Yes:
            self.quit_application()
            event.accept()
        else:
            event.ignore()
