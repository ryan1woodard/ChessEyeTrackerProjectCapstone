"""Application configuration.

Configuration is layered: a packaged ``config/default_config.json`` provides
every key, and a user file (``%APPDATA%/EyeTracker/config.json`` on Windows,
``~/.config/eye_tracker/config.json`` elsewhere) overrides individual values.
Missing keys in the user file always fall back to the defaults, so upgrading
the application never breaks an existing installation.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.json"


def user_data_dir() -> Path:
    """Return the per-user writable directory for config, database and logs."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "EyeTracker"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "EyeTracker"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "eye_tracker"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """Dotted-path accessor over the merged configuration dictionary."""

    def __init__(self, data: Dict[str, Any], path: Path | None = None,
                 defaults: Dict[str, Any] | None = None) -> None:
        self._data = copy.deepcopy(data)
        self._path = path
        # A deep copy is essential: without it ``set()`` would mutate the
        # defaults and ``reset_to_defaults()`` would become a no-op.
        self._defaults = copy.deepcopy(defaults) if defaults is not None \
            else copy.deepcopy(data)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, defaults_path: Path | None = None, user_path: Path | None = None) -> "Config":
        defaults_path = defaults_path or DEFAULT_CONFIG_PATH
        with open(defaults_path, "r", encoding="utf-8") as handle:
            defaults = json.load(handle)

        if user_path is None:
            user_path = user_data_dir() / "config.json"

        data = defaults
        if user_path.exists():
            try:
                with open(user_path, "r", encoding="utf-8") as handle:
                    data = deep_merge(defaults, json.load(handle))
            except (OSError, json.JSONDecodeError) as exc:
                logger.error("Could not read user config %s (%s); using defaults", user_path, exc)
        return cls(data, user_path, defaults)

    # ------------------------------------------------------------- accessors
    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):  # pragma: no cover - defensive
                raise KeyError(f"{dotted_key} traverses a non-dict value")
        node[parts[-1]] = value

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def reset_to_defaults(self) -> None:
        self._data = copy.deepcopy(self._defaults)

    # ------------------------------------------------------------------ save
    def save(self, path: Path | None = None) -> None:
        target = path or self._path
        if target is None:
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(self._data, handle, indent=2)
            tmp.replace(target)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            logger.error("Failed to save config to %s: %s", target, exc)
