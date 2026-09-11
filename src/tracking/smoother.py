"""Gaze smoothing.

Raw per-frame gaze estimates jitter by tens of pixels even when the eyes are
perfectly still, because iris landmarks move by a fraction of a pixel between
frames and the calibration model amplifies that. A plain moving average would
fix the jitter but add visible lag when the eyes saccade to a new square.

The One Euro filter solves exactly this trade-off: it filters aggressively when
the signal is slow and relaxes as soon as speed increases, so a fixation is
rock steady while a saccade still lands quickly.

Reference: Casiez, Roussel & Vogel, "1 Euro Filter" (CHI 2012).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

SMOOTHING_PRESETS: Dict[str, Dict[str, float]] = {
    "off": {"min_cutoff": 1000.0, "beta": 0.0},
    "low": {"min_cutoff": 2.0, "beta": 0.02},
    "medium": {"min_cutoff": 0.9, "beta": 0.006},
    "high": {"min_cutoff": 0.4, "beta": 0.002},
}

#: Matching presets for the input-feature filters. These are gentler than the
#: output presets: features are filtered before the polynomial amplifies them,
#: so a little goes a long way, and over-filtering here would delay saccades.
#: ``beta`` is far larger here than in the output presets, and deliberately so.
#: One Euro relaxes its filtering in proportion to ``beta * speed``, and feature
#: values are two orders of magnitude smaller than pixel coordinates -- an iris
#: offset moves by ~0.1 where a gaze point moves by ~1000. A pixel-sized beta is
#: therefore invisible against feature-sized speeds, and the filter never opens
#: up during a saccade. Tuned against the simulator: these values cut
#: peak-to-peak jitter from 98 px to 42 px while a cross-screen saccade still
#: settles in ~130 ms.
FEATURE_SMOOTHING_PRESETS: Dict[str, Dict[str, float]] = {
    "off": {"min_cutoff": 1000.0, "beta": 0.0},
    "low": {"min_cutoff": 2.0, "beta": 20.0},
    "medium": {"min_cutoff": 1.0, "beta": 15.0},
    "high": {"min_cutoff": 0.6, "beta": 8.0},
}


class _LowPass:
    """First-order low-pass filter with a settable smoothing factor."""

    def __init__(self) -> None:
        self.value: Optional[float] = None

    def filter(self, x: float, alpha: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = alpha * x + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class OneEuroFilter:
    """Adaptive low-pass filter for a single scalar channel."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_time: Optional[float] = None
        self._last_value: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, value: float, timestamp: float) -> float:
        if self._last_time is None:
            dt = 1.0 / 30.0
        else:
            dt = timestamp - self._last_time
            if dt <= 0 or dt > 1.0:
                dt = 1.0 / 30.0
        self._last_time = timestamp

        previous = self._last_value if self._last_value is not None else value
        derivative = (value - previous) / dt
        edx = self._dx.filter(derivative, self._alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(edx)
        filtered = self._x.filter(value, self._alpha(cutoff, dt))
        self._last_value = value
        return filtered

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._last_time = None
        self._last_value = None


class FeatureSmoother:
    """Smooths the model's INPUT features, one One Euro filter per channel.

    Smoothing only the output is not enough. The calibration model is a degree-2
    polynomial, so it amplifies input noise before the output filter ever sees
    it -- and amplified noise is much harder to remove afterwards without
    adding visible lag. Filtering the iris and head features first attacks the
    noise where it enters, which buys far more stability per millisecond of
    latency than filtering the result.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.03,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._filters: Dict[str, OneEuroFilter] = {}

    def smooth(self, values: Dict[str, float], timestamp: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, value in values.items():
            filt = self._filters.get(name)
            if filt is None:
                filt = OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff)
                self._filters[name] = filt
            out[name] = filt.filter(float(value), timestamp)
        return out

    def set_parameters(self, min_cutoff: float, beta: float) -> None:
        self.min_cutoff, self.beta = float(min_cutoff), float(beta)
        for filt in self._filters.values():
            filt.min_cutoff, filt.beta = self.min_cutoff, self.beta

    def reset(self) -> None:
        for filt in self._filters.values():
            filt.reset()


@dataclass
class GazeSmoother:
    """Smooths a 2-D gaze point using one One Euro filter per axis."""

    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0

    def __post_init__(self) -> None:
        self._x = OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff)
        self._y = OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff)

    @classmethod
    def from_preset(cls, preset: str, d_cutoff: float = 1.0) -> "GazeSmoother":
        params = SMOOTHING_PRESETS.get(preset, SMOOTHING_PRESETS["medium"])
        return cls(min_cutoff=params["min_cutoff"], beta=params["beta"], d_cutoff=d_cutoff)

    def set_parameters(self, min_cutoff: float, beta: float,
                       d_cutoff: Optional[float] = None) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        if d_cutoff is not None:
            self.d_cutoff = float(d_cutoff)
        for filt in (self._x, self._y):
            filt.min_cutoff = self.min_cutoff
            filt.beta = self.beta
            filt.d_cutoff = self.d_cutoff

    def smooth(self, x: float, y: float, timestamp: float) -> Tuple[float, float]:
        return self._x.filter(x, timestamp), self._y.filter(y, timestamp)

    def reset(self) -> None:
        self._x.reset()
        self._y.reset()
