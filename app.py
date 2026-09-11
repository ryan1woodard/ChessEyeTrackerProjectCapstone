#!/usr/bin/env python
"""Eye Tracker - webcam gaze analysis for chess study.

Entry point. Parses command-line options, sets up logging and configuration,
opens the database and starts the Qt event loop.

Usage::

    python app.py
    python app.py --debug
    python app.py --calibrate
    python app.py --no-overlay
    python app.py --camera 1
    python app.py --monitor 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import Config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eye_tracker",
        description="Estimate where you are looking on screen using a normal webcam.",
    )
    parser.add_argument("--debug", action="store_true",
                        help="enable debug logging, the camera preview and the debug overlay")
    parser.add_argument("--calibrate", action="store_true",
                        help="start tracking and open calibration immediately")
    parser.add_argument("--no-overlay", action="store_true",
                        help="never show the gaze dot overlay in this run")
    parser.add_argument("--camera", type=int, default=None, metavar="INDEX",
                        help="override the webcam device index")
    parser.add_argument("--monitor", type=int, default=None, metavar="INDEX",
                        help="override the monitor to track (1 is primary)")
    parser.add_argument("--config", type=Path, default=None,
                        help="path to an alternative user config file")
    parser.add_argument("--database", type=Path, default=None,
                        help="path to an alternative session database")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_arguments(argv)
    log_file = setup_logging(debug=args.debug)
    logger = logging.getLogger("eye_tracker")
    logger.info("Starting Eye Tracker (debug=%s), logging to %s", args.debug, log_file)

    config = Config.load(user_path=args.config)
    if args.camera is not None:
        config.set("camera.device_index", args.camera)
    if args.monitor is not None:
        config.set("screen.monitor_index", args.monitor)

    # Qt and the heavy CV imports happen after argument parsing so that
    # --help stays instant.
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication, QMessageBox

    # Windows display scaling (125%, 150%, 175%) is the single most common
    # cause of a calibration that looks fine but predicts off-screen. Qt must
    # pass the fractional factor through rather than rounding it, so that the
    # logical geometry the calibration window reports matches what is drawn.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    from src.gui.first_run import FirstRunDialog
    from src.gui.main_window import MainWindow
    from src.storage.database import Database

    app = QApplication(sys.argv)
    app.setApplicationName("Eye Tracker")
    app.setOrganizationName("EyeTracker")
    app.setQuitOnLastWindowClosed(False)  # the tray keeps the app alive

    database = Database(args.database)
    try:
        database.connect()
    except Exception as exc:  # pragma: no cover - disk failure
        logger.exception("Database could not be opened")
        QMessageBox.critical(None, "Database error",
                             f"The session database could not be opened:\n\n{exc}")
        return 1

    window = MainWindow(config, database, debug=args.debug, no_overlay=args.no_overlay)
    if not bool(config.get("ui.start_minimized", False)):
        window.show()

    if not bool(config.get("ui.first_run_completed", False)):
        dialog = FirstRunDialog(window)
        dialog.exec()
        if dialog.suppress_future:
            config.set("ui.first_run_completed", True)
            config.save()

    if args.calibrate:
        QTimer.singleShot(400, window.start_tracking)
        QTimer.singleShot(2500, window.run_calibration)

    try:
        code = app.exec()
    finally:
        database.close()
        logger.info("Eye Tracker exited with code %s", locals().get("code", 0))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
