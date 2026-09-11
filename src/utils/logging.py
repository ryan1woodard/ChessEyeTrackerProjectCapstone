"""Application logging.

Logs go to ``logs/eye_tracker.log`` (rotating) and to stderr. Webcam frames and
screen captures are never logged -- only derived numeric state.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from .config import PROJECT_ROOT

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"


def setup_logging(debug: bool = False, log_dir: Path | None = None) -> Path:
    """Configure root logging and return the log file path."""
    log_dir = log_dir or (PROJECT_ROOT / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "eye_tracker.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    root.addHandler(file_handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(_FORMAT))
    stream.setLevel(logging.DEBUG if debug else logging.WARNING)
    root.addHandler(stream)

    # MediaPipe and matplotlib are extremely chatty at DEBUG level.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    return log_file
