#!/usr/bin/env python
"""Diagnose an installation without launching the GUI.

Run this first whenever something does not work::

    python check_setup.py

It reports the Python version, every dependency, which MediaPipe face-tracking
backend is available, and which cameras and monitors were found.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

OK, BAD, WARN = "[ ok ]", "[FAIL]", "[warn]"


def check_python() -> bool:
    version = sys.version_info
    good = version >= (3, 11)
    print(f"{OK if good else BAD} Python {platform.python_version()} on {platform.system()}")
    if not good:
        print("       Python 3.11 or newer is required.")
    return good


def check_packages() -> bool:
    packages = [("numpy", "numpy"), ("cv2", "opencv-contrib-python"),
                ("PySide6", "PySide6"), ("mss", "mss"),
                ("matplotlib", "matplotlib"), ("mediapipe", "mediapipe")]
    all_good = True
    for module_name, package_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "?")
            print(f"{OK} {package_name:<24} {version}")
        except ImportError as exc:
            print(f"{BAD} {package_name:<24} not installed ({exc})")
            all_good = False
    return all_good


def check_face_backend() -> bool:
    try:
        from src.tracking.face_tracker import FaceTracker, legacy_solutions_available
        from src.utils.model_download import DEFAULT_MODEL_PATH, model_is_present
    except ImportError as exc:
        print(f"{BAD} Could not import the tracking package: {exc}")
        return False

    legacy = legacy_solutions_available()
    print(f"{OK if legacy else WARN} mediapipe.solutions (offline models): "
          f"{'available' if legacy else 'removed in this MediaPipe version'}")
    print(f"{OK if model_is_present() else WARN} Tasks model file: "
          f"{DEFAULT_MODEL_PATH if model_is_present() else 'not downloaded yet'}")

    try:
        tracker = FaceTracker(backend="auto")
    except Exception as exc:
        print(f"{BAD} No face-tracking backend could start:\n\n{exc}\n")
        return False

    print(f"{OK} Face tracking backend active: {tracker.backend}")
    try:
        import numpy as np
        tracker.process(np.zeros((480, 640, 3), dtype=np.uint8))
        print(f"{OK} Processed a test frame successfully")
    except Exception as exc:
        print(f"{BAD} Processing a frame failed: {exc}")
        return False
    finally:
        tracker.close()
    return True


def check_devices() -> None:
    try:
        from src.tracking.camera import enumerate_cameras
        cameras = enumerate_cameras()
        if cameras:
            for camera in cameras:
                print(f"{OK} Camera {camera.index}: {camera.width}x{camera.height}")
        else:
            print(f"{BAD} No webcam detected (check it is connected and not in use)")
    except Exception as exc:
        print(f"{BAD} Camera enumeration failed: {exc}")

    try:
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtCore import Qt

        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        application = QGuiApplication.instance() or QGuiApplication([])

        from src.utils.screens import list_screens
        screens = list_screens()
        for screen in screens:
            print(f"{OK} {screen.label()}")
            if screen.is_scaled:
                physical = screen.physical_rect
                print(f"       display scaling is on: Qt sees "
                      f"{int(screen.rect.width)}x{int(screen.rect.height)}, "
                      f"the panel is {int(physical.width)}x{int(physical.height)}. "
                      f"This is handled, but recalibrate if you change it.")
        if not screens:
            print(f"{WARN} No monitors detected (screen analysis will be unavailable)")
        del application
    except Exception as exc:
        print(f"{WARN} Monitor enumeration failed: {exc}")


def main() -> int:
    print("=" * 62)
    print(" Eye Tracker setup check")
    print("=" * 62)
    results = [check_python()]
    print()
    results.append(check_packages())
    print()
    if results[-1]:
        results.append(check_face_backend())
        print()
        check_devices()
    print()
    if all(results):
        print("Everything looks good. Run:  python app.py --debug")
        return 0
    print("Fix the [FAIL] items above, then run this again.")
    print("Most problems are solved by:  pip install -r requirements.txt --upgrade")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
