"""
Window clock utilities for 5-minute Polymarket BTC up/down markets.

Key facts:
- Windows start at every Unix timestamp divisible by 300.
- Slug format: btc-updown-5m-{window_start_unix_ts}
- Resolution = first Chainlink update at or after window_start + 300s.
"""
import time
from typing import Optional

WINDOW_SECONDS = 300


def current_window_start_ts(now: Optional[float] = None) -> int:
    """Return the Unix timestamp (seconds) of the current window's start.

    The current window started at the largest multiple of 300 ≤ now.
    """
    t = now if now is not None else time.time()
    return int(t) // WINDOW_SECONDS * WINDOW_SECONDS


def next_window_start_ts(now: Optional[float] = None) -> int:
    """Return the Unix timestamp of the next window's start."""
    return current_window_start_ts(now) + WINDOW_SECONDS


def time_to_resolution(now: Optional[float] = None) -> float:
    """Seconds remaining until the current window resolves (tau).

    Returns 0.0 if past resolution point (guards against clock jitter).
    """
    t = now if now is not None else time.time()
    window_start = current_window_start_ts(t)
    resolution_ts = window_start + WINDOW_SECONDS
    return max(0.0, resolution_ts - t)


def make_slug(window_start_ts: int) -> str:
    """Return the deterministic Polymarket market slug for a window.

    Example: make_slug(1714000000) -> 'btc-updown-5m-1714000000'
    """
    return f"btc-updown-5m-{window_start_ts}"


def elapsed_in_window(now: Optional[float] = None) -> float:
    """Seconds elapsed since the current window started."""
    t = now if now is not None else time.time()
    return t - current_window_start_ts(t)
