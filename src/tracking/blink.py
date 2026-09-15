"""Eye-closure detection, calibrated to the individual user.

Lives in its own module because both the live pipeline and the calibration
session need it, and calibration cannot import the pipeline: the pipeline
imports calibration.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np


class BlinkDetector:
    """Decides when the eyes are closed, relative to *this* user's eyes.

    A fixed openness threshold cannot work. Openness here is the lid gap
    divided by the eye width, and that ratio varies by more than a factor of
    two between people -- narrow or hooded eyes, glasses that clip the lower
    lid, or simply a camera looking up from a laptop. One fixed number is
    therefore either too high for some users, who are treated as permanently
    blinking and get no tracking at all, or too low for others, whose blinks
    sail through and throw the estimate.

    So the open-eye level is learned: a high quantile of recent openness is the
    user's own baseline, and a closure is a fall to a fraction of it.

    The baseline is taken over **every** frame, blinks included, rather than
    over frames already judged open. Judging first is circular and deadlocks
    on exactly the users this exists for: someone whose open eyes sit below the
    starting threshold is called closed on frame one, never contributes an open
    sample, and so never earns a baseline that would let them be seen as open.
    Including every frame is safe because blinks are a small minority of them,
    so a 0.7 quantile is an open-eye value regardless.

    What that leaves is a user who holds their eyes shut long enough to fill
    the window, which would drag the baseline down to the closed level. The
    absolute floor is what catches that case, and it is the only job it has.
    """

    #: The learned baseline is never allowed to push the threshold below this
    #: fraction of the configured absolute threshold.
    FLOOR_FRACTION = 0.5

    def __init__(self, absolute_threshold: float = 0.16,
                 relative_threshold: float = 0.55,
                 history: int = 150, min_samples: int = 12) -> None:
        self.absolute_threshold = absolute_threshold
        self.relative_threshold = relative_threshold
        self.min_samples = min_samples
        self._samples: Deque[float] = deque(maxlen=history)
        self._baseline: Optional[float] = None

    @property
    def baseline(self) -> Optional[float]:
        """The learned open-eye openness, or ``None`` before enough frames."""
        return self._baseline

    def threshold(self) -> float:
        """The openness below which this user's eyes count as closed."""
        if self._baseline is None:
            return self.absolute_threshold
        return max(self.absolute_threshold * self.FLOOR_FRACTION,
                   self._baseline * self.relative_threshold)

    def update(self, openness: float) -> bool:
        """Record one frame's openness and report whether the eyes are closed."""
        self._samples.append(float(openness))
        if len(self._samples) >= self.min_samples:
            self._baseline = float(np.quantile(self._samples, 0.7))
        return openness < self.threshold()

    def reset(self) -> None:
        self._samples.clear()
        self._baseline = None
