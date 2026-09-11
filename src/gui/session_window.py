"""Session results dashboard.

Shows the statistics, the state timeline and two heatmaps for a recorded
session, and exports it to CSV or JSON.

The heatmaps are reconstructed from stored gaze coordinates, never from saved
screenshots -- the application does not keep any.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QSplitter,
                               QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from ..events.detector import compute_statistics
from ..events.models import (GazeEvent, GazeSampleRecord, SessionStatistics,
                             SUMMARY_BUCKETS, format_duration)
from ..storage.database import Database
from ..storage.export import (export_events_csv, export_samples_csv, export_session_json)
from ..utils.geometry import FILES

logger = logging.getLogger(__name__)

_BUCKET_COLORS = {
    "board": "#4caf7d",
    "clock": "#4da6ff",
    "other": "#b48ee8",
    "away": "#ff7043",
    "unknown": "#6b7280",
}


class _Canvas(FigureCanvasQTAgg):
    """A matplotlib canvas styled to match the dark UI."""

    def __init__(self, width: float = 6.0, height: float = 3.2) -> None:
        self.figure = Figure(figsize=(width, height), facecolor="#1a1d23")
        super().__init__(self.figure)

    def reset(self):
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        axes.set_facecolor("#1a1d23")
        for spine in axes.spines.values():
            spine.set_color("#39404b")
        axes.tick_params(colors="#aab2bf", labelsize=8)
        axes.xaxis.label.set_color("#aab2bf")
        axes.yaxis.label.set_color("#aab2bf")
        axes.title.set_color("#e6eaf0")
        return axes


class SessionWindow(QDialog):
    """Browse recorded sessions and inspect one in detail."""

    def __init__(self, database: Database, parent: Optional[QWidget] = None,
                 select_session_id: Optional[int] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sessions")
        self.resize(1020, 680)
        self._db = database
        self._events: List[GazeEvent] = []
        self._samples: List[GazeSampleRecord] = []
        self._stats = SessionStatistics()
        self._current_row = None

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_session_selected)

        self._summary = QTextEdit()
        self._summary.setReadOnly(True)

        self._timeline = _Canvas(7.0, 2.6)
        self._squares = _Canvas(7.0, 3.0)
        self._board_heatmap = _Canvas(5.0, 5.0)
        self._screen_heatmap = _Canvas(7.0, 4.0)

        tabs = QTabWidget()
        tabs.addTab(self._summary, "Summary")
        tabs.addTab(self._wrap(self._timeline), "Timeline")
        tabs.addTab(self._wrap(self._squares), "Most viewed squares")
        tabs.addTab(self._wrap(self._board_heatmap), "Board heatmap")
        tabs.addTab(self._wrap(self._screen_heatmap), "Screen heatmap")

        export_csv = QPushButton("Export CSV")
        export_csv.clicked.connect(self._export_csv)
        export_json = QPushButton("Export JSON")
        export_json.clicked.connect(self._export_json)
        delete_button = QPushButton("Delete session")
        delete_button.clicked.connect(self._delete_session)

        buttons = QHBoxLayout()
        buttons.addWidget(export_csv)
        buttons.addWidget(export_json)
        buttons.addStretch(1)
        buttons.addWidget(delete_button)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(tabs)
        right_layout.addLayout(buttons)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Recorded sessions"))
        left_layout.addWidget(self._list)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 3)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        self.reload(select_session_id)

    @staticmethod
    def _wrap(canvas: _Canvas) -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        return holder

    # ----------------------------------------------------------------- data
    def reload(self, select_session_id: Optional[int] = None) -> None:
        self._list.clear()
        rows = self._db.list_sessions()
        for row in rows:
            label = f"{row['start_time'].replace('T', '  ')}   " \
                    f"{format_duration(row['duration'] or 0)}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, int(row["id"]))
            self._list.addItem(item)
            if select_session_id is not None and int(row["id"]) == select_session_id:
                self._list.setCurrentItem(item)
        if self._list.currentItem() is None and self._list.count():
            self._list.setCurrentRow(0)
        if not rows:
            self._summary.setPlainText("No sessions recorded yet.")

    def _on_session_selected(self, current: Optional[QListWidgetItem], _previous) -> None:
        if current is None:
            return
        db_id = int(current.data(Qt.UserRole))
        row = self._db.get_session(db_id)
        if row is None:
            return
        self._current_row = row
        self._events = self._db.get_events(db_id)
        self._samples = self._db.get_samples(db_id)
        self._stats = compute_statistics(self._events, duration=row["duration"])
        self._render()

    # -------------------------------------------------------------- render
    def _render(self) -> None:
        self._render_summary()
        self._render_timeline()
        self._render_squares()
        self._render_board_heatmap()
        self._render_screen_heatmap()

    def _render_summary(self) -> None:
        row = self._current_row
        stats = self._stats
        lines = [
            "SESSION SUMMARY",
            "",
            f"Started    {row['start_time'].replace('T', ' ')}",
            f"Ended      {(row['end_time'] or '-').replace('T', ' ')}",
            f"Duration   {format_duration(stats.duration)}",
            f"Monitor    {row['monitor_index']}",
            f"Calibration {row['calibration_id'] or '-'}",
            "",
            "ATTENTION",
        ]
        for bucket in SUMMARY_BUCKETS:
            seconds = stats.bucket_seconds.get(bucket, 0.0)
            lines.append(f"  {bucket.capitalize():<10} {stats.bucket_percent(bucket):5.1f}%"
                         f"   {format_duration(seconds)}")
        lines += [
            "",
            "LOOK-AWAY EVENTS",
            f"  Total      {stats.away_event_count}",
            f"  Longest    {stats.away_longest_seconds:.2f} s",
            f"  Average    {stats.away_average_seconds:.2f} s",
            f"  Total time {format_duration(stats.away_total_seconds)}",
            "",
            "QUALITY",
            f"  Mean confidence  {stats.mean_confidence * 100:.1f}%",
            f"  Events recorded  {stats.event_count}",
            f"  Gaze samples     {len(self._samples)}",
            "",
            "MOST VIEWED SQUARES",
        ]
        top = stats.top_squares(10)
        if top:
            for square, seconds in top:
                lines.append(f"  {square:<4} {seconds:6.1f} s")
        else:
            lines.append("  (no board time recorded)")
        self._summary.setPlainText("\n".join(lines))

    def _render_timeline(self) -> None:
        axes = self._timeline.reset()
        axes.set_title("Attention timeline")
        if not self._events:
            axes.text(0.5, 0.5, "No events", ha="center", va="center", color="#aab2bf")
            self._timeline.draw_idle()
            return

        lanes = list(SUMMARY_BUCKETS)
        for event in self._events:
            lane = lanes.index(event.bucket)
            axes.barh(lane, event.duration, left=event.start_time, height=0.62,
                      color=_BUCKET_COLORS.get(event.bucket, "#6b7280"),
                      edgecolor="none")
        axes.set_yticks(range(len(lanes)))
        axes.set_yticklabels([lane.capitalize() for lane in lanes])
        axes.set_xlabel("Seconds since session start")
        axes.set_xlim(0, max(self._stats.duration, 1.0))
        axes.invert_yaxis()
        self._timeline.figure.tight_layout()
        self._timeline.draw_idle()

    def _render_squares(self) -> None:
        axes = self._squares.reset()
        axes.set_title("Most viewed squares")
        top = self._stats.top_squares(12)
        if not top:
            axes.text(0.5, 0.5, "No board time recorded", ha="center", va="center",
                      color="#aab2bf")
            self._squares.draw_idle()
            return
        names = [t[0] for t in top][::-1]
        values = [t[1] for t in top][::-1]
        axes.barh(names, values, color="#4caf7d")
        axes.set_xlabel("Seconds")
        self._squares.figure.tight_layout()
        self._squares.draw_idle()

    def _square_seconds_grid(self) -> np.ndarray:
        """Board time as an 8x8 grid indexed ``[rank_index][file_index]``."""
        grid = np.zeros((8, 8), dtype=np.float64)
        for square, seconds in self._stats.square_seconds.items():
            if len(square) != 2 or square[0] not in FILES or not square[1].isdigit():
                continue
            file_index = FILES.index(square[0])
            rank = int(square[1])
            if 1 <= rank <= 8:
                grid[8 - rank, file_index] += seconds
        return grid

    def _render_board_heatmap(self) -> None:
        axes = self._board_heatmap.reset()
        axes.set_title("Chessboard heatmap (seconds)")
        grid = self._square_seconds_grid()
        image = axes.imshow(grid, cmap="magma", interpolation="nearest")
        axes.set_xticks(range(8), list(FILES))
        axes.set_yticks(range(8), [str(r) for r in range(8, 0, -1)])
        if grid.max() > 0:
            for row in range(8):
                for col in range(8):
                    if grid[row, col] <= 0:
                        continue
                    axes.text(col, row, f"{grid[row, col]:.0f}", ha="center", va="center",
                              fontsize=7,
                              color="#101216" if grid[row, col] > grid.max() * 0.55
                              else "#e6eaf0")
        bar = self._board_heatmap.figure.colorbar(image, ax=axes, fraction=0.046)
        bar.ax.tick_params(colors="#aab2bf", labelsize=7)
        self._board_heatmap.figure.tight_layout()
        self._board_heatmap.draw_idle()

    def _render_screen_heatmap(self) -> None:
        axes = self._screen_heatmap.reset()
        axes.set_title("Screen gaze heatmap")
        points = [(s.x, s.y) for s in self._samples if s.confidence > 0.2]
        if len(points) < 10:
            axes.text(0.5, 0.5, "Not enough gaze samples", ha="center", va="center",
                      color="#aab2bf")
            self._screen_heatmap.draw_idle()
            return
        array = np.array(points)
        axes.hist2d(array[:, 0], array[:, 1], bins=(64, 40), cmap="magma")
        axes.set_xlabel("Screen X (px)")
        axes.set_ylabel("Screen Y (px)")
        axes.invert_yaxis()
        self._screen_heatmap.figure.tight_layout()
        self._screen_heatmap.draw_idle()

    # -------------------------------------------------------------- actions
    def _session_meta(self) -> Dict:
        row = self._current_row
        return {
            "session_id": row["session_id"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "duration": row["duration"],
            "calibration_id": row["calibration_id"],
            "monitor_index": row["monitor_index"],
        }

    def _export_csv(self) -> None:
        if self._current_row is None:
            return
        default = f"{self._current_row['session_id']}_events.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export events as CSV", default,
                                              "CSV files (*.csv)")
        if not path:
            return
        try:
            export_events_csv(Path(path), self._events)
            if self._samples:
                export_samples_csv(Path(path).with_name(Path(path).stem + "_samples.csv"),
                                   self._samples)
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Export complete", f"Saved to {path}")

    def _export_json(self) -> None:
        if self._current_row is None:
            return
        default = f"{self._current_row['session_id']}.json"
        path, _ = QFileDialog.getSaveFileName(self, "Export session as JSON", default,
                                              "JSON files (*.json)")
        if not path:
            return
        try:
            export_session_json(Path(path), self._session_meta(), self._events,
                                self._samples, include_samples=True)
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Export complete", f"Saved to {path}")

    def _delete_session(self) -> None:
        if self._current_row is None:
            return
        answer = QMessageBox.question(
            self, "Delete session",
            f"Delete session {self._current_row['session_id']} and all of its events?",
        )
        if answer != QMessageBox.Yes:
            return
        self._db.delete_session(int(self._current_row["id"]))
        self._current_row = None
        self.reload()
