"""System tray integration.

The tray icon is drawn at runtime rather than loaded from a file, so the
application has no external image dependency and packages cleanly.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QWidget


def make_icon(tracking: bool = False, size: int = 64) -> QIcon:
    """Draw a simple eye glyph; the pupil turns red while tracking."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    painter.setBrush(QColor("#e8edf4"))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(QRectF(size * 0.06, size * 0.24, size * 0.88, size * 0.52))

    painter.setBrush(QColor("#ff3b30") if tracking else QColor("#39414d"))
    painter.drawEllipse(QRectF(size * 0.34, size * 0.32, size * 0.32, size * 0.36))

    painter.setBrush(QColor("#12151a"))
    painter.drawEllipse(QRectF(size * 0.43, size * 0.41, size * 0.14, size * 0.18))
    painter.end()
    return QIcon(pixmap)


class SystemTray(QSystemTrayIcon):
    """Tray icon exposing the main actions while the window is hidden."""

    show_window_requested = Signal()
    start_requested = Signal()
    stop_requested = Signal()
    calibrate_requested = Signal()
    sessions_requested = Signal()
    settings_requested = Signal()
    exit_requested = Signal()
    dot_toggled = Signal(bool)
    heatmap_toggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(make_icon(False), parent)
        self.setToolTip("Eye Tracker - idle")

        menu = QMenu()
        self._status_action = QAction("Idle", menu)
        self._status_action.setEnabled(False)
        menu.addAction(self._status_action)
        menu.addSeparator()

        self._add(menu, "Show Window", self.show_window_requested)
        self._start_action = self._add(menu, "Start Tracking", self.start_requested)
        self._stop_action = self._add(menu, "Stop Tracking", self.stop_requested)
        self._stop_action.setEnabled(False)

        self._dot_action = QAction("Show Gaze Dot", menu)
        self._dot_action.setCheckable(True)
        self._dot_action.toggled.connect(self.dot_toggled.emit)
        menu.addAction(self._dot_action)

        self._heatmap_action = QAction("Show Heatmap", menu)
        self._heatmap_action.setCheckable(True)
        self._heatmap_action.toggled.connect(self.heatmap_toggled.emit)
        menu.addAction(self._heatmap_action)

        menu.addSeparator()
        self._add(menu, "Calibrate", self.calibrate_requested)
        self._add(menu, "View Sessions", self.sessions_requested)
        self._add(menu, "Settings", self.settings_requested)
        menu.addSeparator()
        self._add(menu, "Exit", self.exit_requested)

        self._menu = menu
        self.setContextMenu(menu)
        self.activated.connect(self._on_activated)

    @staticmethod
    def _add(menu: QMenu, text: str, signal) -> QAction:
        action = QAction(text, menu)
        action.triggered.connect(signal.emit)
        menu.addAction(action)
        return action

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.show_window_requested.emit()

    # -------------------------------------------------------------- updates
    def set_tracking(self, tracking: bool) -> None:
        self.setIcon(make_icon(tracking))
        self._status_action.setText("Tracking" if tracking else "Idle")
        self._start_action.setEnabled(not tracking)
        self._stop_action.setEnabled(tracking)
        self.setToolTip(f"Eye Tracker - {'tracking' if tracking else 'idle'}")

    def set_dot_checked(self, checked: bool) -> None:
        self._dot_action.blockSignals(True)
        self._dot_action.setChecked(checked)
        self._dot_action.blockSignals(False)

    def set_heatmap_checked(self, checked: bool) -> None:
        self._heatmap_action.blockSignals(True)
        self._heatmap_action.setChecked(checked)
        self._heatmap_action.blockSignals(False)
