"""Settings dialog.

Every value here is persisted to the user config file; nothing is hard-coded in
the tracking code. Changes that require a restart of the worker are reported
back to the main window through the ``restart_required`` flag.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QGroupBox, QLabel,
                               QSpinBox, QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from ..utils.screens import ScreenInfo
from ..tracking.calibration import DEFAULT_METHOD, METHODS
from ..tracking.camera import CameraInfo
from ..tracking.smoother import SMOOTHING_PRESETS
from ..utils.config import Config

PRIVACY_TEXT = """\
Everything this application does happens on your computer.

  - Webcam frames are analysed in memory and discarded immediately.
  - No webcam video is ever written to disk.
  - Screen captures are used only to locate the chessboard, in memory.
  - No screenshots or recordings are saved.
  - Nothing is uploaded. There is no cloud service, no account and no network
    connection used by the tracking engine.
  - Only derived numbers are stored: gaze coordinates, aggregated events,
    session statistics and calibration model parameters.

The application observes your screen and webcam. It never interacts with
chess.com: it does not click, move pieces, read game state, inject scripts or
automate anything.

Local data lives in:
  data/sessions/eye_tracker.db   session events and statistics
  data/calibrations/*.json       calibration profiles
  logs/eye_tracker.log           diagnostic logs (no images)

Deleting those files removes all stored data.
"""


class SettingsWindow(QDialog):
    """Modal settings dialog."""

    def __init__(self, config: Config, cameras: List[CameraInfo],
                 monitors: List[ScreenInfo], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)
        self._config = config
        self._cameras = cameras
        self._monitors = monitors
        self.restart_required = False

        tabs = QTabWidget(self)
        tabs.addTab(self._build_camera_tab(), "Camera")
        tabs.addTab(self._build_tracking_tab(), "Tracking")
        tabs.addTab(self._build_board_tab(), "Board")
        tabs.addTab(self._build_overlay_tab(), "Overlay")
        tabs.addTab(self._build_privacy_tab(), "Privacy")

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._restore)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------ tabs
    def _build_camera_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.camera_combo = QComboBox()
        for camera in self._cameras:
            self.camera_combo.addItem(camera.label(), camera.index)
        if not self._cameras:
            self.camera_combo.addItem("No camera detected", 0)
        self._select_data(self.camera_combo, int(self._config.get("camera.device_index", 0)))

        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 4096)
        self.width_spin.setSingleStep(160)
        self.width_spin.setValue(int(self._config.get("camera.width", 1280)))

        self.height_spin = QSpinBox()
        self.height_spin.setRange(240, 2160)
        self.height_spin.setSingleStep(120)
        self.height_spin.setValue(int(self._config.get("camera.height", 720)))

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(5, 60)
        self.fps_spin.setValue(int(self._config.get("camera.fps", 30)))

        self.mirror_check = QCheckBox("Mirror the preview image")
        self.mirror_check.setChecked(bool(self._config.get("camera.mirror_preview", True)))

        form.addRow("Camera:", self.camera_combo)
        form.addRow("Capture width:", self.width_spin)
        form.addRow("Capture height:", self.height_spin)
        form.addRow("Target FPS:", self.fps_spin)
        form.addRow("", self.mirror_check)
        form.addRow(QLabel("Changing camera settings restarts tracking."))
        return page

    def _build_tracking_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.calibration_method_combo = QComboBox()
        for method, hint in (
            ("explore", "Move your head at each dot - recommended"),
            ("postures", "Hold a posture at each dot - quicker"),
        ):
            if method in METHODS:
                self.calibration_method_combo.addItem(hint, method)
        self._select_data(self.calibration_method_combo,
                          str(self._config.get("calibration.method", DEFAULT_METHOD)))
        self.calibration_method_combo.setToolTip(
            "How calibration collects its samples. Moving your head at every dot "
            "is what lets the tracker tell head movement from eye movement; "
            "holding a posture is faster but only works if you stay relaxed.")

        self.smoothing_combo = QComboBox()
        for preset, hint in (("off", "Off - raw, very jittery"),
                             ("low", "Low - fastest response"),
                             ("medium", "Medium - recommended"),
                             ("high", "High - steadiest, slight lag")):
            if preset in SMOOTHING_PRESETS:
                self.smoothing_combo.addItem(hint, preset)
        self._select_data(self.smoothing_combo,
                          str(self._config.get("tracking.smoothing_preset", "medium")))

        self.blink_recovery_spin = QSpinBox()
        self.blink_recovery_spin.setRange(0, 600)
        self.blink_recovery_spin.setSingleStep(20)
        self.blink_recovery_spin.setSuffix(" ms")
        self.blink_recovery_spin.setToolTip(
            "How long the last good estimate is held after the eyes reopen")
        self.blink_recovery_spin.setValue(
            int(self._config.get("tracking.blink_recovery_ms", 120)))

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 0.95)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setValue(float(self._config.get("tracking.minimum_confidence", 0.45)))

        self.away_spin = QSpinBox()
        self.away_spin.setRange(100, 5000)
        self.away_spin.setSingleStep(50)
        self.away_spin.setSuffix(" ms")
        self.away_spin.setValue(int(self._config.get("events.away_threshold_ms", 500)))

        self.region_dwell_spin = QSpinBox()
        self.region_dwell_spin.setRange(50, 2000)
        self.region_dwell_spin.setSuffix(" ms")
        self.region_dwell_spin.setValue(int(self._config.get("events.region_dwell_ms", 200)))

        self.square_dwell_spin = QSpinBox()
        self.square_dwell_spin.setRange(50, 2000)
        self.square_dwell_spin.setSuffix(" ms")
        self.square_dwell_spin.setValue(int(self._config.get("events.square_dwell_ms", 250)))

        self.eyes_closed_spin = QSpinBox()
        self.eyes_closed_spin.setRange(200, 5000)
        self.eyes_closed_spin.setSuffix(" ms")
        self.eyes_closed_spin.setValue(int(self._config.get("tracking.eyes_closed_ms", 1200)))

        form.addRow("Calibration method:", self.calibration_method_combo)
        form.addRow("Smoothing:", self.smoothing_combo)
        form.addRow("Minimum confidence:", self.confidence_spin)
        form.addRow("Look-away threshold:", self.away_spin)
        form.addRow("Region dwell:", self.region_dwell_spin)
        form.addRow("Square dwell:", self.square_dwell_spin)
        form.addRow("Eyes closed counts as away after:", self.eyes_closed_spin)
        form.addRow("Hold estimate after a blink:", self.blink_recovery_spin)

        note = QLabel(
            "Smoothing filters both the tracker's inputs and its output.\n"
            "Higher settings are steadier but respond a little slower."
        )
        note.setWordWrap(True)
        form.addRow(note)
        return page

    def _build_board_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.monitor_combo = QComboBox()
        for monitor in self._monitors:
            self.monitor_combo.addItem(monitor.label(), monitor.index)
        if self.monitor_combo.count() == 0:
            self.monitor_combo.addItem("Primary monitor", 1)
        self._select_data(self.monitor_combo, int(self._config.get("screen.monitor_index", 1)))

        self.orientation_combo = QComboBox()
        self.orientation_combo.addItem("White at bottom", "white_bottom")
        self.orientation_combo.addItem("Black at bottom", "black_bottom")
        self._select_data(self.orientation_combo,
                          str(self._config.get("screen.board_orientation", "white_bottom")))

        self.auto_detect_check = QCheckBox("Detect the board automatically")
        self.auto_detect_check.setChecked(bool(self._config.get("screen.auto_detect_board", True)))

        self.redetect_spin = QDoubleSpinBox()
        self.redetect_spin.setRange(2.0, 300.0)
        self.redetect_spin.setSuffix(" s")
        self.redetect_spin.setValue(float(self._config.get("screen.board_redetect_seconds", 20.0)))

        note = QLabel(
            "Board orientation cannot be read from the game, so set it to match\n"
            "the side you are playing. Automatic detection is a convenience;\n"
            "manual selection is always exact."
        )
        note.setWordWrap(True)

        form.addRow("Monitor:", self.monitor_combo)
        form.addRow("Board orientation:", self.orientation_combo)
        form.addRow("", self.auto_detect_check)
        form.addRow("Re-detect every:", self.redetect_spin)
        form.addRow(note)
        return page

    def _build_overlay_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        dot_group = QGroupBox("Gaze dot")
        dot_form = QFormLayout(dot_group)

        self.dot_radius_spin = QSpinBox()
        self.dot_radius_spin.setRange(3, 60)
        self.dot_radius_spin.setSuffix(" px")
        self.dot_radius_spin.setValue(int(self._config.get("overlay.dot_radius", 12)))

        self.dot_opacity_spin = QDoubleSpinBox()
        self.dot_opacity_spin.setRange(0.05, 1.0)
        self.dot_opacity_spin.setSingleStep(0.05)
        self.dot_opacity_spin.setValue(float(self._config.get("overlay.dot_opacity", 0.75)))

        self.dot_color_combo = QComboBox()
        for name, value in (("Red", "#ff2d2d"), ("Green", "#3ddc84"),
                            ("Blue", "#4da6ff"), ("Yellow", "#ffd23d")):
            self.dot_color_combo.addItem(name, value)
        self._select_data(self.dot_color_combo, str(self._config.get("overlay.dot_color",
                                                                     "#ff2d2d")))

        dot_form.addRow("Radius:", self.dot_radius_spin)
        dot_form.addRow("Opacity:", self.dot_opacity_spin)
        dot_form.addRow("Colour:", self.dot_color_combo)

        ui_group = QGroupBox("Window behaviour")
        ui_form = QFormLayout(ui_group)
        self.tray_check = QCheckBox("Minimise to the system tray instead of exiting")
        self.tray_check.setChecked(bool(self._config.get("ui.minimize_to_tray", True)))
        ui_form.addRow(self.tray_check)

        layout.addWidget(dot_group)
        layout.addWidget(ui_group)
        layout.addStretch(1)
        return page

    def _build_privacy_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(PRIVACY_TEXT)
        layout.addWidget(text)

        self.store_samples_check = QCheckBox(
            "Store gaze coordinates for heatmaps (numbers only, no images)"
        )
        self.store_samples_check.setChecked(
            bool(self._config.get("privacy.store_gaze_samples", True))
        )
        layout.addWidget(self.store_samples_check)
        return page

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _select_data(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    # ---------------------------------------------------------------- accept
    def _accept(self) -> None:
        cfg = self._config
        old_camera = (cfg.get("camera.device_index"), cfg.get("camera.width"),
                      cfg.get("camera.height"), cfg.get("camera.fps"))

        cfg.set("camera.device_index", int(self.camera_combo.currentData() or 0))
        cfg.set("camera.width", self.width_spin.value())
        cfg.set("camera.height", self.height_spin.value())
        cfg.set("camera.fps", self.fps_spin.value())
        cfg.set("camera.mirror_preview", self.mirror_check.isChecked())

        cfg.set("tracking.smoothing_preset", self.smoothing_combo.currentData())
        cfg.set("calibration.method", self.calibration_method_combo.currentData())
        cfg.set("tracking.minimum_confidence", self.confidence_spin.value())
        cfg.set("tracking.eyes_closed_ms", self.eyes_closed_spin.value())
        cfg.set("tracking.blink_recovery_ms", self.blink_recovery_spin.value())

        cfg.set("events.away_threshold_ms", self.away_spin.value())
        cfg.set("events.region_dwell_ms", self.region_dwell_spin.value())
        cfg.set("events.square_dwell_ms", self.square_dwell_spin.value())

        cfg.set("screen.monitor_index", int(self.monitor_combo.currentData() or 1))
        cfg.set("screen.board_orientation", self.orientation_combo.currentData())
        cfg.set("screen.auto_detect_board", self.auto_detect_check.isChecked())
        cfg.set("screen.board_redetect_seconds", self.redetect_spin.value())

        cfg.set("overlay.dot_radius", self.dot_radius_spin.value())
        cfg.set("overlay.dot_opacity", self.dot_opacity_spin.value())
        cfg.set("overlay.dot_color", self.dot_color_combo.currentData())

        cfg.set("ui.minimize_to_tray", self.tray_check.isChecked())
        cfg.set("privacy.store_gaze_samples", self.store_samples_check.isChecked())

        new_camera = (cfg.get("camera.device_index"), cfg.get("camera.width"),
                      cfg.get("camera.height"), cfg.get("camera.fps"))
        self.restart_required = old_camera != new_camera
        cfg.save()
        self.accept()

    def _restore(self) -> None:
        self._config.reset_to_defaults()
        self._config.save()
        self.restart_required = True
        self.accept()
