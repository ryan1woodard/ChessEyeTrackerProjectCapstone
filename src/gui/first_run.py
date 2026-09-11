"""First-run guidance.

Rather than a modal wizard that duplicates the main window's controls, this is
a short checklist that stays out of the way. Each step maps directly onto a
button in the main window, so the user learns the interface while setting up.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QLabel,
                               QVBoxLayout, QWidget)

STEPS = """\
<h2>Welcome to Eye Tracker</h2>
<p>Five steps and you are ready to play.</p>
<ol>
  <li><b>Pick your webcam and monitor</b> in the panel on the left.</li>
  <li><b>Press Start Tracking.</b> The camera opens and the status turns to
      <i>Not calibrated</i>.</li>
  <li><b>Press Calibrate</b> and look at each dot until its ring fills.
      Keep your head reasonably still.</li>
  <li><b>Tick "Gaze Dot"</b> and look around the screen. The red dot should
      follow you within roughly a square's width. If it does not, recalibrate
      with better lighting.</li>
  <li><b>Open chess.com</b>, then press <b>Detect Chessboard</b>, or
      <b>Select Chessboard Manually</b> if detection misses.</li>
</ol>
<p>Then just play. Minimise the window &mdash; tracking continues in the tray.
When you finish, press Stop Tracking and open <b>View Sessions</b>.</p>
<p style="color:#8b95a3">Everything stays on this computer. No video, no
screenshots, no uploads. The application only watches; it never interacts with
chess.com.</p>
"""


class FirstRunDialog(QDialog):
    """Shown once, on the first launch."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Getting started")
        self.setMinimumWidth(560)

        label = QLabel(STEPS)
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)

        self.dont_show = QCheckBox("Do not show this again")
        self.dont_show.setChecked(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(label)
        layout.addWidget(self.dont_show)
        layout.addWidget(buttons)

    @property
    def suppress_future(self) -> bool:
        return self.dont_show.isChecked()
