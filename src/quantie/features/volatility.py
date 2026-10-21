"""
Realized volatility estimator — EWMA on log-price returns.

Updated on every Binance 100ms mid-price tick.
Output is per-second vol (√10 × per-tick vol).
"""
from __future__ import annotations

import numpy as np


class RealizedVolEstimator:
    """EWMA realized volatility from 100ms Binance mid-price ticks.

    halflife_seconds controls the exponential decay:
        alpha = 1 - 0.5^(0.1 / halflife_seconds)

    Call update(log_price) on every depth tick (≈100ms).
    Returns per-second vol (sqrt(10) × per-tick vol).
    """

    def __init__(self, halflife_seconds: float = 60.0) -> None:
        # alpha calibrated for 100ms ticks
        self.alpha: float = 1.0 - 0.5 ** (0.1 / halflife_seconds)
        self._var: float | None = None
        self._last_log_price: float | None = None
        self.last_vol_per_sec: float = 0.0

    def update(self, log_price: float) -> float:
        """Update with new log-price observation; return current per-second vol."""
        if self._last_log_price is None:
            self._last_log_price = log_price
            self._var = (1e-4) ** 2          # sensible cold-start (~1 bp tick)
            self.last_vol_per_sec = float(np.sqrt(self._var * 10))
            return self.last_vol_per_sec

        ret = log_price - self._last_log_price
        self._last_log_price = log_price

        if self._var is None:
            self._var = ret ** 2
        else:
            self._var = (1.0 - self.alpha) * self._var + self.alpha * ret ** 2

        # multiply by 10 to convert per-100ms variance → per-second variance
        self.last_vol_per_sec = float(np.sqrt(self._var * 10))
        return self.last_vol_per_sec

    @property
    def vol(self) -> float:
        """Current per-second vol estimate (0 before first update)."""
        return self.last_vol_per_sec

    def reset(self) -> None:
        """Clear state — call on window rollover if desired."""
        self._var = None
        self._last_log_price = None
        self.last_vol_per_sec = 0.0
