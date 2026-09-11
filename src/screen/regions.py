"""Named screen regions and priority-based hit testing.

Regions are generic rectangles, not chess-specific, so a future version can
track attention on any application by supplying a different region set.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..utils.geometry import Rect

logger = logging.getLogger(__name__)

KIND_BOARD = "board"
KIND_CLOCK = "clock"
KIND_OTHER = "other"


@dataclass
class Region:
    """A named rectangle on the virtual desktop.

    Higher ``priority`` wins when regions overlap, which lets a small clock
    sit on top of a large browser region without ambiguity.
    """

    name: str
    x: float
    y: float
    width: float
    height: float
    priority: int = 0
    kind: str = KIND_OTHER
    enabled: bool = True

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.width, self.height)

    def contains(self, x: float, y: float) -> bool:
        return self.enabled and self.rect.contains(x, y)

    @classmethod
    def from_rect(cls, name: str, rect: Rect, priority: int = 0,
                  kind: str = KIND_OTHER) -> "Region":
        return cls(name, rect.x, rect.y, rect.width, rect.height, priority, kind)

    def to_dict(self) -> Dict:
        return {"name": self.name, "x": self.x, "y": self.y, "width": self.width,
                "height": self.height, "priority": self.priority, "kind": self.kind,
                "enabled": self.enabled}

    @classmethod
    def from_dict(cls, data: Dict) -> "Region":
        return cls(
            name=data["name"], x=float(data["x"]), y=float(data["y"]),
            width=float(data["width"]), height=float(data["height"]),
            priority=int(data.get("priority", 0)), kind=data.get("kind", KIND_OTHER),
            enabled=bool(data.get("enabled", True)),
        )


class RegionManager:
    """Holds the region set and answers "what is at this point?"."""

    BOARD_NAME = "Chessboard"

    def __init__(self, regions: Optional[Iterable[Region]] = None) -> None:
        self._regions: List[Region] = list(regions or [])

    # -------------------------------------------------------------- mutation
    def add(self, region: Region) -> None:
        self.remove(region.name)
        self._regions.append(region)

    def remove(self, name: str) -> bool:
        before = len(self._regions)
        self._regions = [r for r in self._regions if r.name != name]
        return len(self._regions) != before

    def clear(self) -> None:
        self._regions.clear()

    def get(self, name: str) -> Optional[Region]:
        for region in self._regions:
            if region.name == name:
                return region
        return None

    def set_board(self, rect: Rect) -> None:
        """Install or replace the board region, which always has top priority."""
        self.add(Region.from_rect(self.BOARD_NAME, rect, priority=100, kind=KIND_BOARD))

    @property
    def board(self) -> Optional[Region]:
        return self.get(self.BOARD_NAME)

    @property
    def regions(self) -> List[Region]:
        return list(self._regions)

    # ------------------------------------------------------------- hit tests
    def classify(self, x: float, y: float) -> Optional[Region]:
        """Return the highest-priority enabled region containing the point.

        Ties are broken by the smaller area, so a nested region wins over the
        container it sits inside.
        """
        matches = [r for r in self._regions if r.contains(x, y)]
        if not matches:
            return None
        matches.sort(key=lambda r: (-r.priority, r.rect.area))
        return matches[0]

    # ---------------------------------------------------------- persistence
    def to_list(self) -> List[Dict]:
        return [r.to_dict() for r in self._regions]

    def load_list(self, data: Iterable[Dict]) -> None:
        self._regions = [Region.from_dict(item) for item in data]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_list(), handle, indent=2)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with open(path, "r", encoding="utf-8") as handle:
                self.load_list(json.load(handle))
            return True
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.error("Could not load regions from %s: %s", path, exc)
            return False


def suggested_clock_regions(board: Rect, screen: Rect) -> List[Region]:
    """Heuristic clock rectangles for a typical chess.com layout.

    On the standard desktop layout the clocks sit in the side panel to the
    right of the board, one above and one below the middle. This is only a
    starting point -- the user can move or delete them in Settings.
    """
    panel_left = board.right + board.width * 0.02
    panel_width = min(board.width * 0.45, max(screen.right - panel_left, 0.0))
    if panel_width < 40:
        return []
    clock_height = board.height * 0.10
    top_clock = Rect(panel_left, board.top + board.height * 0.06, panel_width, clock_height)
    bottom_clock = Rect(panel_left, board.bottom - board.height * 0.16,
                        panel_width, clock_height)
    return [
        Region.from_rect("Black Clock", top_clock, priority=50, kind=KIND_CLOCK),
        Region.from_rect("White Clock", bottom_clock, priority=50, kind=KIND_CLOCK),
    ]
