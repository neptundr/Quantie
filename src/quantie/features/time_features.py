"""
Time-based and strike-distance features.

Used by the GBM baseline and the ML residual model.
"""
from __future__ import annotations

import datetime
from typing import Optional

import numpy as np

WINDOW_SECONDS: float = 300.0
_EPS = 1e-9


def compute_tau_features(tau_sec: float) -> dict[str, float]:
    """Return tau_seconds and tau_normalized (tau / 300).

    tau_normalized ∈ [0, 1]; 1.0 = just opened, 0.0 = resolved.
    """
    tau_sec = max(0.0, tau_sec)
    return {
        "tau_seconds": tau_sec,
        "tau_normalized": tau_sec / WINDOW_SECONDS,
    }


def compute_distance_from_strike(
    S_t: float,
    S_0: float,
    sigma_per_sec: float,
    tau_sec: float,
) -> Optional[float]:
    """(S_t - S_0) / (sigma * sqrt(tau)) — signed distance in sigma units.

    Positive = above strike (current price favours UP).
    Returns None when tau ≈ 0 or sigma ≈ 0 (undefined).
    """
    if tau_sec < _EPS or sigma_per_sec < _EPS:
        return None
    denom = sigma_per_sec * np.sqrt(tau_sec)
    if denom < _EPS:
        return None
    return float(np.log(S_t / S_0) / denom)


def hour_of_day(now: Optional[datetime.datetime] = None) -> int:
    """UTC hour [0, 23]."""
    t = now or datetime.datetime.now(datetime.timezone.utc)
    return t.hour


def day_of_week(now: Optional[datetime.datetime] = None) -> int:
    """ISO weekday [0=Mon, 6=Sun]."""
    t = now or datetime.datetime.now(datetime.timezone.utc)
    return t.weekday()
