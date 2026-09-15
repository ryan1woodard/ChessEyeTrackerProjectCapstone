"""The calibration screen.

A full-screen window shows one target at a time. Each target runs through three
phases:

``pose``
    No dot yet. The posture being asked for fills the middle of the screen,
    with a live gauge of the user's head tilt against the one wanted. This
    phase exists because reading and acting on an instruction cannot be done
    while fixating a dot: the previous version showed the cue as a caption at
    the moment the dot appeared, and by the time it had been read the recording
    had started. Nothing is asked of the user's gaze here, so there is time to
    read, adopt the posture, and see that it was adopted.

``settle``
    The target appears and pulses. Nothing is recorded yet -- the user needs a
    moment to find it and fixate. The posture prompt shrinks to a compact gauge
    beside the dot, close enough to be read peripherally without looking away.

``collect``
    Samples are recorded until either the sample quota or the dwell time is
    reached. A ring around the target fills up so the user can see progress and
    knows to keep looking.

Samples arrive on the ``sample_ready`` signal from the tracking worker, i.e.
already fully processed on the worker thread; this window only stores feature
vectors and never touches the camera itself.
"""

from __future__ import annotations

import logging
from collections import deque
from statistics import median
from typing import Deque, List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..tracking.calibration import (DEFAULT_METHOD, METHODS, CalibrationSession,
                                    PoseCoverage, Posture, calibration_pattern,
                                    pattern_to_pixels, posture)
from ..tracking.pipeline import GazeSample
from ..utils.config import Config
from ..utils.geometry import Rect

logger = logging.getLogger(__name__)

PHASE_INTRO = "intro"
PHASE_POSE = "pose"
PHASE_SETTLE = "settle"
PHASE_COLLECT = "collect"
PHASE_DONE = "done"

#: Colours, kept together so the phases read consistently.
_INK = QColor(240, 244, 250)
_MUTED = QColor(165, 176, 192)
_ACCENT = QColor(120, 190, 255)
_GOOD = QColor(90, 220, 150)
_WARN = QColor(255, 175, 80)
_TARGET = QColor(255, 60, 60)


class CalibrationWindow(QWidget):
    """Runs the calibration procedure over one monitor."""

    completed = Signal(object)   # CalibrationProfile
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, config: Config, screen_rect: Rect, monitor_index: int,
                 camera_index: int, parent: Optional[QWidget] = None,
                 qt_screen=None) -> None:
        super().__init__(parent)
        self._qt_screen = qt_screen
        self._config = config
        self._screen_rect = screen_rect
        self._monitor_index = monitor_index
        self._camera_index = camera_index

        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Calibration")
        self.setCursor(Qt.BlankCursor)
        self.setGeometry(int(screen_rect.x), int(screen_rect.y),
                         int(screen_rect.width), int(screen_rect.height))

        self._pattern_name = str(config.get("calibration.pattern", "13point"))
        pattern = calibration_pattern(self._pattern_name)
        self._targets: List[Tuple[float, float]] = pattern_to_pixels(
            pattern, int(screen_rect.width), int(screen_rect.height),
            (int(screen_rect.x), int(screen_rect.y))
        )

        self._session = CalibrationSession(
            feature_names=list(config.get("calibration.model_features", [])),
            screen_size=(int(screen_rect.width), int(screen_rect.height)),
            screen_origin=(int(screen_rect.x), int(screen_rect.y)),
            degree=int(config.get("calibration.poly_degree", 2)),
            alphas=list(config.get("calibration.ridge_alphas", [0.1, 1.0, 10.0])),
            max_yaw=float(config.get("tracking.max_head_yaw_deg", 40.0)),
            max_pitch=float(config.get("tracking.max_head_pitch_deg", 30.0)),
            min_openness=float(config.get("tracking.blink_ear_threshold", 0.16)),
            margin_fraction=float(config.get("calibration.margin_fraction", 0.15)),
        )

        self._settle_ms = int(config.get("calibration.settle_ms", 700))
        self._dwell_ms = int(config.get("calibration.dwell_ms", 900))
        self._quota = int(config.get("calibration.samples_per_point", 20))
        self._radius = int(config.get("calibration.target_radius_px", 18))
        self._posture_hint = bool(config.get("calibration.posture_guidance", True))
        #: Longest the pose phase will wait for the posture to be adopted.
        self._pose_ms = int(config.get("calibration.posture_ms", 3000))
        #: Shortest it will stay, so a posture already held is still readable.
        self._pose_min_ms = int(config.get("calibration.posture_min_ms", 1200))
        #: How long the tilt must sit within tolerance before moving on.
        self._pose_hold_ms = int(config.get("calibration.posture_hold_ms", 500))
        self._pose_tolerance = float(config.get("calibration.posture_tolerance_deg", 6.0))
        method = str(config.get("calibration.method", DEFAULT_METHOD))
        self._method = method if method in METHODS else DEFAULT_METHOD
        #: How long each dot gets in the explore method.
        self._explore_ms = int(config.get("calibration.explore_ms", 5000))
        #: Never leave a dot before this, even if coverage completes at once.
        self._explore_min_ms = int(config.get("calibration.explore_min_ms", 2500))
        self._coverage = PoseCoverage()

        self._index = 0
        self._phase = PHASE_INTRO
        self._phase_elapsed = 0
        self._animation = 0.0
        self._collected = 0
        self._face_present = False
        self._message = ""
        #: Most recent measured head tilt, for the live gauge. ``None`` until a
        #: face has been seen, which is itself worth showing.
        self._head_roll: Optional[float] = None
        self._head_yaw: Optional[float] = None
        self._head_pitch: Optional[float] = None
        #: Lateral offset and distance, in the metric units the feature
        #: extractor produces, and the user's own resting values for them.
        #: Without a baseline "lean in" has no measurable target; with one it
        #: can be shown on a gauge like the tilt.
        self._head_x: Optional[float] = None
        self._head_z: Optional[float] = None
        self._rest_samples: Deque[Tuple[float, float]] = deque(maxlen=90)
        self._held_ms = 0

        self._timer = QTimer(self)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._tick)

    def _rebuild_targets(self) -> None:
        """Recompute targets from the widget's real on-screen geometry.

        The window is what the user actually looks at, so its size is the
        authority on where the dots are. Deriving the targets from anything
        else risks training the model on coordinates that do not match where
        the dots appeared -- which produces a model whose predictions are
        uniformly scaled wrong and land off screen.
        """
        width, height = self.width(), self.height()
        origin = (self.x(), self.y())
        if width < 100 or height < 100:
            return
        pattern = calibration_pattern(self._pattern_name)
        self._targets = pattern_to_pixels(pattern, width, height, origin)
        self._session.screen_size = (width, height)
        self._session.screen_origin = origin
        logger.info("Calibration targets built for %dx%d at %s", width, height, origin)

    # ---------------------------------------------------------------- start
    def start(self) -> None:
        # Bind to the chosen monitor before going full screen, otherwise a
        # multi-monitor setup can open the calibration on the wrong display and
        # every recorded target coordinate would be meaningless.
        if self._qt_screen is not None:
            handle = self.windowHandle()
            if handle is None:
                self.create()
                handle = self.windowHandle()
            if handle is not None:
                handle.setScreen(self._qt_screen)
                geometry = self._qt_screen.geometry()
                self.setGeometry(geometry)
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.OtherFocusReason)
        self._rebuild_targets()
        self._phase = PHASE_INTRO
        self._phase_elapsed = 0
        self._timer.start()

    # ------------------------------------------------------------- samples
    @property
    def posture(self) -> Posture:
        """The posture being asked for at the current target."""
        return posture(self._index)

    def on_sample(self, sample: GazeSample) -> None:
        """Slot for ``TrackerWorker.sample_ready`` (queued to the GUI thread)."""
        self._face_present = sample.face_detected
        usable_pose = sample.face_detected and sample.head_pose.valid
        self._head_roll = sample.head_pose.roll if usable_pose else None
        self._head_yaw = sample.head_pose.yaw if usable_pose else None
        self._head_pitch = sample.head_pose.pitch if usable_pose else None
        features = sample.features
        if features is not None and features.valid:
            self._head_x = features.get("head_x")
            self._head_z = features.get("head_z")
            # Only postures that ask for nothing feed the resting baseline, so
            # a lean cannot redefine what "normal" means and then be measured
            # against it.
            if self.posture.is_neutral:
                self._rest_samples.append((self._head_x, self._head_z))
        else:
            self._head_x = self._head_z = None
        if self._phase == PHASE_COLLECT and self.exploring and self._head_roll is not None:
            self._coverage.observe(self._head_roll, self._head_yaw or 0.0,
                                   self._head_pitch or 0.0)
        if self._phase != PHASE_COLLECT or sample.features is None:
            return
        if not self.exploring and self._collected >= self._quota:
            return
        target = self._targets[self._index]
        if self._session.add(self._index, target, sample.features):
            self._collected += 1

    # ----------------------------------------------------------------- loop
    def _tick(self) -> None:
        self._phase_elapsed += self._timer.interval()
        self._animation += 0.08

        if self._phase == PHASE_INTRO:
            if self._phase_elapsed >= 2800:
                self._begin_point(0)
        elif self._phase == PHASE_POSE:
            self._tick_pose()
        elif self._phase == PHASE_SETTLE:
            if self._phase_elapsed >= self._settle_ms:
                self._phase = PHASE_COLLECT
                self._phase_elapsed = 0
                self._collected = 0
        elif self._phase == PHASE_COLLECT:
            if self._collect_finished():
                self._finish_point()
        self.update()

    def _collect_finished(self) -> bool:
        """Whether this dot has had enough.

        The two methods stop on different things. Holding a posture, what
        matters is the sample count -- every frame is near enough the same
        pose, so more of them adds little. Exploring, what matters is the
        *range* covered, so the dot stays until the head has been round it or
        the time runs out.
        """
        if not self.exploring:
            return (self._collected >= self._quota
                    or self._phase_elapsed >= self._dwell_ms)
        if self._phase_elapsed < self._explore_min_ms:
            return False
        return self._coverage.complete or self._phase_elapsed >= self._explore_ms

    def _tick_pose(self) -> None:
        """Wait for the posture, but never wait forever.

        Advancing on a timer alone would start recording whether or not the
        user had moved; refusing to advance until they did would strand anyone
        whose webcam angle makes the target tilt unreachable. So the posture is
        held for a moment to move on early, and the timeout moves on regardless
        -- with a calibration that is merely narrower rather than stuck.
        """
        if self.posture_held:
            self._held_ms += self._timer.interval()
        else:
            self._held_ms = 0

        settled = (self._phase_elapsed >= self._pose_min_ms
                   and self._held_ms >= self._pose_hold_ms)
        if settled or self._phase_elapsed >= self._pose_ms:
            self._phase = PHASE_SETTLE
            self._phase_elapsed = 0

    @property
    def resting_position(self) -> Optional[Tuple[float, float]]:
        """The user's normal ``(head_x, head_z)``, once enough has been seen."""
        if len(self._rest_samples) < 10:
            return None
        values = list(self._rest_samples)
        return (median(v[0] for v in values), median(v[1] for v in values))

    def posture_progress(self) -> Optional[float]:
        """How far towards the posture the user has moved, 0 to 1 and beyond.

        ``None`` when it cannot be measured -- no face, or no resting baseline
        yet for a lean or a shift. Returning ``None`` rather than a number
        matters: the screen must not tell someone they have it right when
        nothing was measured.
        """
        wanted = self.posture
        if not self._face_present or wanted.is_neutral:
            return None
        if wanted.is_tilt:
            if self._head_roll is None:
                return None
            return self._head_roll / wanted.roll_deg if wanted.roll_deg else None

        rest = self.resting_position
        if rest is None or self._head_x is None or self._head_z is None:
            return None
        rest_x, rest_z = rest
        if wanted.lean:
            target = wanted.target_distance(rest_z)
            span = target - rest_z
            return (self._head_z - rest_z) / span if abs(span) > 1e-6 else None
        target = wanted.target_offset(rest_x)
        span = target - rest_x
        return (self._head_x - rest_x) / span if abs(span) > 1e-6 else None

    @property
    def posture_held(self) -> bool:
        """Whether the head is currently in the posture being asked for."""
        if not self._face_present:
            return False
        if self.posture.is_neutral:
            return True
        if self.posture.is_tilt:
            if self._head_roll is None:
                return False
            return self.posture.matches(self._head_roll, self._pose_tolerance)
        progress = self.posture_progress()
        # Generous, because leaning and shifting are asked for in words rather
        # than to a number: most of the way there is the whole point.
        return progress is not None and progress >= 0.6

    @property
    def exploring(self) -> bool:
        """Whether this run asks the head to roam rather than hold a posture."""
        return self._method == "explore"

    def _begin_point(self, index: int) -> None:
        self._index = index
        # Explore has nothing to pose for: the instruction is the same at every
        # dot, so a phase spent reading it would be a phase spent not moving.
        wants_pose = self._posture_hint and not self.exploring
        self._phase = PHASE_POSE if wants_pose else PHASE_SETTLE
        self._phase_elapsed = 0
        self._held_ms = 0
        self._collected = 0
        self._coverage.reset()

    def _finish_point(self) -> None:
        usable = self._session.count_for_point(self._index)
        logger.debug("Calibration point %d collected %d samples", self._index + 1, usable)
        if self._index + 1 < len(self._targets):
            self._begin_point(self._index + 1)
        else:
            self._complete()

    # -------------------------------------------------------------- finish
    def _complete(self) -> None:
        self._phase = PHASE_DONE
        self._timer.stop()
        try:
            estimator = self._session.fit()
        except ValueError as exc:
            logger.error("Calibration failed: %s", exc)
            self.close()
            self.failed.emit(str(exc))
            return
        profile = self._session.to_profile(
            estimator, self._monitor_index, self._camera_index, self._pattern_name
        )
        self.close()
        self.completed.emit(profile)

    def _cancel(self) -> None:
        self._timer.stop()
        self.close()
        self.cancelled.emit()

    # ---------------------------------------------------------------- input
    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self._cancel()
        elif event.key() == Qt.Key_Space and self._phase == PHASE_INTRO:
            self._begin_point(0)
        elif event.key() == Qt.Key_M and self._phase == PHASE_INTRO:
            # Offered here rather than buried in Settings: this is the moment
            # the choice is about to matter, and the screen explaining the two
            # is already in front of the user.
            index = METHODS.index(self._method)
            self._method = METHODS[(index + 1) % len(METHODS)]
            logger.info("Calibration method set to %s", self._method)
            self.update()
        elif event.key() == Qt.Key_Space and self._phase == PHASE_POSE:
            # For the user who has the posture and does not want to wait.
            self._phase = PHASE_SETTLE
            self._phase_elapsed = 0
        elif event.key() == Qt.Key_R and self._phase in (PHASE_POSE, PHASE_SETTLE,
                                                         PHASE_COLLECT):
            # Redo the current point.
            self._session.clear_point(self._index)
            self._begin_point(self._index)
        else:
            super().keyPressEvent(event)

    # ---------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(16, 18, 22))

        if self._phase == PHASE_INTRO:
            self._paint_intro(painter)
        elif self._phase == PHASE_POSE:
            self._paint_pose(painter)
        elif self._phase in (PHASE_SETTLE, PHASE_COLLECT):
            self._paint_target(painter)
            self._paint_status(painter)
            if self.exploring:
                self._paint_explore_prompt(painter)
            else:
                self._paint_posture_reminder(painter)
        painter.end()

    def _paint_intro(self, painter: QPainter) -> None:
        painter.setPen(QPen(_INK))
        painter.setFont(QFont("Segoe UI", 30, QFont.DemiBold))
        painter.drawText(QRectF(0, self.height() * 0.10, self.width(), 50),
                         Qt.AlignCenter, "Calibration")

        painter.setFont(QFont("Segoe UI", 15))
        painter.setPen(QPen(_MUTED))
        # Down to the key hints, not a fraction of the height: the explore
        # text is long enough that a fixed fraction silently clipped its last
        # line, which was the one saying how long the whole thing takes.
        top = self.height() * 0.20
        painter.drawText(QRectF(0, top, self.width(), self.height() - 140 - top),
                         Qt.AlignHCenter | Qt.AlignTop, self._intro_text())

        painter.setPen(QPen(_ACCENT))
        painter.setFont(QFont("Segoe UI", 14, QFont.DemiBold))
        painter.drawText(QRectF(0, self.height() - 120, self.width(), 26), Qt.AlignCenter,
                         f"M switches method  -  now: {self._method_name()}")
        painter.setPen(QPen(_MUTED))
        painter.setFont(QFont("Segoe UI", 14))
        painter.drawText(QRectF(0, self.height() - 88, self.width(), 26), Qt.AlignCenter,
                         "Space to begin  -  R to redo a point  -  Esc to cancel")

    def _method_name(self) -> str:
        return ("Move your head (recommended)" if self.exploring
                else "Hold a posture (quicker)")

    def _intro_text(self) -> str:
        if self.exploring:
            return (
                "Look straight at each dot and keep looking at it.\n\n"
                "While you look, keep your head moving, three ways:\n\n"
                "   TILT it side to side, ear towards shoulder\n"
                "   TURN it left and right, as if glancing at someone beside you\n"
                "   NOD it up and down\n\n"
                "Lean in and back too. Your eyes stay on the dot throughout.\n\n"
                "Three rings around each dot fill in as you go, one per\n"
                "movement, innermost first. All three must fill before the dot\n"
                "moves on, so if one is stuck, that is the movement to do more\n"
                "of. You never need to look away to check them.\n\n"
                "This is what teaches the tracker to tell head movement apart\n"
                "from eye movement. If your head sits in the same place at\n"
                "every dot, the two look identical to it, and tracking falls\n"
                "apart the first time you move during a game.\n\n"
                f"{len(self._targets)} points, about {self._estimated_seconds():.0f} seconds."
            )
        return (
            "Each point has two parts.\n\n"
            "First you are asked to sit a certain way -- tilt your head a\n"
            "little to one side, lean in, sit back. Nothing is recorded yet,\n"
            "and a gauge shows when you have it.\n\n"
            "Then a red dot appears. Look straight at it and keep looking\n"
            "until its ring fills. Hold the posture, but stay relaxed and let\n"
            "your head drift gently -- holding rigidly still is what makes a\n"
            "calibration fragile.\n\n"
            "Sit at your normal playing distance.\n\n"
            f"{len(self._targets)} points, about {self._estimated_seconds():.0f} seconds."
        )

    def _estimated_seconds(self) -> float:
        if self.exploring:
            per_point = self._settle_ms + (self._explore_min_ms + self._explore_ms) / 2
        else:
            per_point = self._pose_min_ms + self._settle_ms + self._dwell_ms
        return len(self._targets) * per_point / 1000.0

    # ------------------------------------------------------------ pose phase
    def _paint_pose(self, painter: QPainter) -> None:
        """The full-screen posture prompt, with the dot deliberately absent.

        Laid out as fractions of the window rather than fixed offsets, so the
        heading cannot collide with the counter above it on a short screen.
        """
        wanted = self.posture
        height, width = self.height(), self.width()
        centre = QPointF(width / 2.0, height * 0.52)

        painter.setPen(QPen(_MUTED))
        painter.setFont(QFont("Segoe UI", 14))
        painter.drawText(QRectF(0, height * 0.07, width, 30),
                         Qt.AlignCenter, f"Point {self._index + 1} of {len(self._targets)}")

        painter.setPen(QPen(_INK))
        painter.setFont(QFont("Segoe UI", 32, QFont.DemiBold))
        painter.drawText(QRectF(0, height * 0.14, width, 60), Qt.AlignCenter, wanted.label)

        gauge = min(height * 0.16, 150.0)
        if wanted.is_tilt:
            self._paint_tilt_gauge(painter, centre, radius=gauge, thickness=gauge * 0.06)
        elif wanted.is_neutral:
            self._paint_tilt_gauge(painter, centre, radius=gauge, thickness=gauge * 0.06)
        else:
            self._paint_travel_gauge(painter, centre, wanted, gauge)

        self._paint_pose_readout(painter, QRectF(0, height * 0.74, width, 30))

        painter.setFont(QFont("Segoe UI", 17, QFont.DemiBold))
        if not self._face_present:
            painter.setPen(QPen(_WARN))
            message = "Face not detected - check your lighting and camera"
        elif self.posture_held:
            painter.setPen(QPen(_GOOD))
            message = "That's it - hold it"
        elif wanted.is_tilt:
            painter.setPen(QPen(_ACCENT))
            message = "Tilt until the white line sits inside the blue band"
        else:
            painter.setPen(QPen(_ACCENT))
            message = "Move until the dot reaches the blue zone"
        painter.drawText(QRectF(0, height * 0.80, width, 34), Qt.AlignCenter, message)

        self._paint_pose_countdown(painter, QRectF(0, height * 0.88, width, 28))

    def _paint_pose_readout(self, painter: QPainter, box: QRectF) -> None:
        """How far the user has moved, in the terms the posture was asked in."""
        wanted = self.posture
        painter.setPen(QPen(_MUTED))
        painter.setFont(QFont("Segoe UI", 15))
        if wanted.is_neutral:
            text = "Sit however is comfortable"
        elif wanted.is_tilt:
            if self._head_roll is None:
                text = "Waiting for your face"
            else:
                # Signed, and phrased as what is left to do. An unsigned
                # "4 of 12 degrees" reads as a third of the way there even when
                # the head is tilted the wrong way entirely, which is exactly
                # the reading that leaves someone stuck.
                remaining = wanted.roll_deg - self._head_roll
                if abs(remaining) <= self._pose_tolerance:
                    text = f"Holding {abs(self._head_roll):.0f}\u00b0 - that is the one"
                else:
                    way = "left" if remaining > 0 else "right"
                    text = f"{abs(remaining):.0f}\u00b0 further to your {way}"
        else:
            progress = self.posture_progress()
            text = ("Waiting for your face" if progress is None
                    else f"{min(max(progress, 0.0), 1.5) * 100:.0f}% of the way")
        painter.drawText(box, Qt.AlignCenter, text)

    def _paint_pose_countdown(self, painter: QPainter, box: QRectF) -> None:
        """How long until the dot arrives, so it is never a surprise."""
        remaining = max(0, self._pose_ms - self._phase_elapsed)
        if self.posture_held and self._phase_elapsed >= self._pose_min_ms:
            remaining = min(remaining, max(0, self._pose_hold_ms - self._held_ms))
        painter.setPen(QPen(_MUTED))
        painter.setFont(QFont("Segoe UI", 14))
        painter.drawText(box, Qt.AlignCenter,
                         f"The dot appears in {remaining / 1000.0:.1f} s"
                         "     -     Space to skip ahead")

    def _paint_tilt_gauge(self, painter: QPainter, centre: QPointF,
                          radius: float, thickness: float) -> None:
        """Where the head should be, and where it is, inside one head outline.

        A matching task is shown rather than a written direction because a
        picture of a head has to be mentally mirrored -- whose left? -- while
        matching does not: tilt the wrong way and the bright line visibly
        leaves the band, so the instruction corrects itself.

        One circle, not two. Drawing a whole head for each of the target and
        the measurement put two rings of nearly the same size on top of each
        other, which read as a single thick ring with clutter inside it. The
        circle is the head; what rotates within it is the line of the eyes.
        """
        import math

        wanted = self.posture.roll_deg

        painter.setPen(QPen(QColor(70, 78, 92), max(thickness * 0.8, 2.0)))
        painter.setBrush(QColor(28, 32, 40))
        painter.drawEllipse(centre, radius, radius)

        def axis(angle_deg: float, length: float):
            radians = math.radians(angle_deg)
            cos, sin = math.cos(radians), math.sin(radians)
            return (QPointF(centre.x() - cos * length, centre.y() - sin * length),
                    QPointF(centre.x() + cos * length, centre.y() + sin * length),
                    (sin, cos))

        # The target, as a wide translucent band the live line can sit inside.
        left, right, _ = axis(wanted, radius * 0.72)
        painter.setPen(QPen(QColor(_ACCENT.red(), _ACCENT.green(), _ACCENT.blue(), 95),
                            max(thickness * 3.4, 14.0), Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(left, right)

        if self._head_roll is None:
            return

        colour = _GOOD if self.posture_held else _INK
        left, right, (sin, cos) = axis(self._head_roll, radius * 0.6)
        painter.setPen(QPen(colour, max(thickness, 3.0), Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(left, right)
        painter.setPen(Qt.NoPen)
        painter.setBrush(colour)
        for point in (left, right):
            painter.drawEllipse(point, radius * 0.11, radius * 0.11)
        # A nose, so the picture cannot be read upside down.
        painter.setPen(QPen(colour, max(thickness * 0.8, 2.0), Qt.SolidLine, Qt.RoundCap))
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(centre, QPointF(centre.x() + sin * radius * 0.5,
                                         centre.y() - cos * radius * 0.5))

    def _paint_travel_gauge(self, painter: QPainter, centre: QPointF,
                            wanted: Posture, size: float) -> None:
        """A track with a target zone, for leaning and shifting.

        These have no angle to match, but they are still measured -- against
        the user's own resting position -- so they get a gauge rather than a
        bare arrow. An arrow alone cannot say how far, and "closer to the
        screen" has no direction on screen that an arrow could honestly point.
        """
        vertical = bool(wanted.lean)
        # Longer across than down: there is room sideways, and the vertical
        # track has the end labels stacked above and below it.
        length = size * (1.7 if vertical else 2.6)
        track = max(size * 0.20, 12.0)
        progress = self.posture_progress()

        def along(fraction: float) -> QPointF:
            offset = (fraction - 0.5) * length
            if vertical:
                return QPointF(centre.x(), centre.y() + offset)
            return QPointF(centre.x() + offset, centre.y())

        painter.setPen(QPen(QColor(56, 63, 76), track, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(along(0.0), along(1.0))

        # The target zone: reaching it is what "held" means.
        painter.setPen(QPen(QColor(_ACCENT.red(), _ACCENT.green(), _ACCENT.blue(), 150),
                            track, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(along(0.78), along(1.0))

        painter.setFont(QFont("Segoe UI", 13))

        def label(at: QPointF, text: str, colour: QColor) -> None:
            painter.setPen(QPen(colour))
            if vertical:
                # Wide enough for the longest of them; a tight box silently
                # clips rather than shrinking, and a half-word reads as a typo.
                box = QRectF(at.x() - track - 210, at.y() - 12, 200, 24)
                flags = Qt.AlignRight | Qt.AlignVCenter
            else:
                box = QRectF(at.x() - 100, at.y() + track * 0.8, 200, 24)
                flags = Qt.AlignCenter
            painter.drawText(box, flags, text)

        label(along(0.0), "where you were", _MUTED)
        label(along(0.95), wanted.destination, _ACCENT)

        if progress is None:
            return
        marker = along(min(max(progress * 0.78 + 0.11, 0.04), 0.96))
        painter.setPen(Qt.NoPen)
        painter.setBrush(_GOOD if self.posture_held else _INK)
        painter.drawEllipse(marker, track * 0.62, track * 0.62)

    def _paint_explore_prompt(self, painter: QPainter) -> None:
        """One line of instruction, the same at every dot so it need not be reread.

        Placed along the bottom rather than beside the dot because unlike the
        posture reminder it does not change, and a sentence that never changes
        stops being read after the first dot -- which is the point. The ring
        round the dot is what carries the moment-to-moment feedback.
        """
        painter.setPen(QPen(_ACCENT if self._face_present else _WARN))
        painter.setFont(QFont("Segoe UI", 15, QFont.DemiBold))
        message = ("Face not detected - check your lighting and camera"
                   if not self._face_present else
                   "Eyes on the dot - tilt your head side to side, turn it "
                   "left and right, nod it up and down")
        painter.drawText(QRectF(0, self.height() - 108, self.width(), 30),
                         Qt.AlignCenter, message)

    def _paint_coverage_ring(self, painter: QPainter, centre: QPointF,
                             radius: float) -> None:
        """The head poses seen so far at this dot, as rings round the dot.

        Drawn concentric with the target on purpose. The user has to keep
        looking at the dot, so anything they need to check has to be readable
        without moving their eyes -- a ring around the thing they are already
        staring at is the only place that is true of.

        One ring per axis -- tilt, turn, nod, innermost outwards -- because a
        single ring could be filled by rocking the head side to side without
        ever turning it, and a calibration that never saw the head turn cannot
        correct for a turned head.
        """
        bins = self._coverage.bins
        gap = 360.0 / bins * 0.22
        span = 360.0 / bins - gap
        for depth, axis in enumerate(("tilt", "turn", "nod")):
            ring = radius * (1.0 + 0.28 * depth)
            box = QRectF(centre.x() - ring, centre.y() - ring, ring * 2, ring * 2)
            for index, seen in enumerate(self._coverage.seen[axis]):
                start = 180.0 - (index + 0.5) * (360.0 / bins) - span / 2
                painter.setPen(QPen(_GOOD if seen else QColor(64, 72, 86),
                                    5 if seen else 3, Qt.SolidLine, Qt.RoundCap))
                painter.drawArc(box, int(start * 16), int(span * 16))

    def _paint_posture_reminder(self, painter: QPainter) -> None:
        """A compact reminder beside the dot, once the dot is what matters.

        Small and close to the target on purpose. It is there to be caught in
        peripheral vision while the user keeps looking at the dot, not to be
        read -- anything larger, or further away, invites a glance, and a
        glance is a wasted sample.
        """
        if not self._posture_hint or self.posture.is_neutral:
            return
        target = self._targets[self._index]
        anchor = QPointF(target[0] - self._screen_rect.x,
                         target[1] - self._screen_rect.y)
        # Placed on whichever side of the dot has room, so the gauge never
        # covers the thing being looked at.
        offset_x = 100.0 if anchor.x() < self.width() / 2 else -100.0
        offset_y = 100.0 if anchor.y() < self.height() / 2 else -100.0
        centre = QPointF(anchor.x() + offset_x, anchor.y() + offset_y)

        if self.posture.is_tilt:
            self._paint_tilt_gauge(painter, centre, radius=44.0, thickness=4.0)
        else:
            painter.setPen(QPen(_ACCENT))
            painter.setFont(QFont("Segoe UI", 12, QFont.DemiBold))
            painter.drawText(QRectF(centre.x() - 150, centre.y() - 14, 300, 28),
                             Qt.AlignCenter, self.posture.label)

    def _paint_target(self, painter: QPainter) -> None:
        import math

        target = self._targets[self._index]
        center = QPointF(target[0] - self._screen_rect.x, target[1] - self._screen_rect.y)

        pulse = 1.0 + 0.18 * math.sin(self._animation * 2.2)
        outer = self._radius * 2.6 * (pulse if self._phase == PHASE_SETTLE else 1.0)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 70, 70, 40))
        painter.drawEllipse(center, outer, outer)

        if self._phase == PHASE_COLLECT:
            if self.exploring:
                # Coverage, not sample count: the dot is finished when the head
                # has been round it, and a plain progress bar would say nothing
                # about whether the moving is happening.
                self._paint_coverage_ring(painter, center, outer * 1.35)
            else:
                progress = min(1.0, self._collected / max(self._quota, 1))
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(80, 220, 140), 4, Qt.SolidLine, Qt.RoundCap))
                span = int(-progress * 360 * 16)
                painter.drawArc(int(center.x() - outer), int(center.y() - outer),
                                int(outer * 2), int(outer * 2), 90 * 16, span)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 60, 60))
        painter.drawEllipse(center, self._radius, self._radius)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(center, max(2.0, self._radius * 0.22),
                            max(2.0, self._radius * 0.22))

    def _paint_status(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 14))
        painter.setPen(QPen(QColor(200, 208, 220)))
        text = f"Calibration {self._index + 1} of {len(self._targets)}"
        painter.drawText(self.rect().adjusted(0, 0, 0, -60), Qt.AlignHCenter | Qt.AlignBottom,
                         text)

        if not self._face_present:
            painter.setPen(QPen(QColor(255, 170, 70)))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(self.rect().adjusted(0, 0, 0, -30),
                             Qt.AlignHCenter | Qt.AlignBottom,
                             "Face not detected - check your lighting and camera")
