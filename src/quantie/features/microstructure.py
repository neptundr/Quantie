"""
Microstructure feature functions — all pure, no I/O.

Feature groups:
  - core order-flow / depth / volatility features
  - ML residual feature set
  - manipulation-setup features
  - in-progress swing detection
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Calibration constants (update from 99th-percentile of 1-week data run)
# ---------------------------------------------------------------------------
LARGE_PRINT_THRESHOLD: float = 1.0    # BTC qty — single-trade size threshold
VELOCITY_NORMALIZER: float = 0.001    # 0.1% 5s log-return normalises to 1.0

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Order flow imbalance
# ---------------------------------------------------------------------------

def compute_ofi(
    trades: Sequence[tuple[int, float, bool]],   # (ts_ms, qty, is_buy)
    window_ms: int,
    now_ms: int,
) -> float:
    """Net signed volume (buy - sell) over the trailing window.

    Positive = buy pressure; negative = sell pressure.
    """
    cutoff = now_ms - window_ms
    total = 0.0
    for ts_ms, qty, is_buy in trades:
        if ts_ms >= cutoff:
            total += qty if is_buy else -qty
    return total


# ---------------------------------------------------------------------------
# Depth imbalance (Binance or Polymarket)
# ---------------------------------------------------------------------------

def compute_depth_imbalance(
    bids: Sequence[tuple[float, float]],   # (price, qty)
    asks: Sequence[tuple[float, float]],
    levels: int = 5,
) -> float:
    """(bid_qty - ask_qty) / (bid_qty + ask_qty) at top `levels` levels.

    Range [-1, 1]. Positive = bid-heavy book.
    """
    bid_qty = sum(q for _, q in bids[:levels])
    ask_qty = sum(q for _, q in asks[:levels])
    denom = bid_qty + ask_qty
    if denom < _EPS:
        return 0.0
    return (bid_qty - ask_qty) / denom


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def compute_momentum(
    log_prices: Sequence[tuple[int, float]],   # (ts_ms, log_price), newest last
    window_ms: int,
    now_ms: int,
) -> float | None:
    """Log-return over trailing window. None if insufficient history."""
    if not log_prices:
        return None
    cutoff = now_ms - window_ms
    anchor: float | None = None
    for ts_ms, lp in log_prices:
        if ts_ms >= cutoff:
            anchor = lp
            break
    if anchor is None:
        return None
    return log_prices[-1][1] - anchor


# ---------------------------------------------------------------------------
# Chainlink-Binance basis
# ---------------------------------------------------------------------------

def compute_basis(binance_mid: float, chainlink_price: float) -> float:
    """binance_mid - chainlink_price (price units, not %)."""
    return binance_mid - chainlink_price


def update_basis_ewma(
    current_basis: float,
    prior_ewma: float | None,
    alpha: float = 0.05,
) -> float:
    """Simple EWMA for the basis series."""
    if prior_ewma is None:
        return current_basis
    return (1.0 - alpha) * prior_ewma + alpha * current_basis


# ---------------------------------------------------------------------------
# Polymarket microstructure
# ---------------------------------------------------------------------------

def compute_poly_mid_velocity(
    poly_mids: Sequence[tuple[int, float]],   # (ts_ms, mid), newest last
    window_ms: int,
    now_ms: int,
) -> float | None:
    """Rate of change of poly mid-price (units/sec). None if insufficient history."""
    if len(poly_mids) < 2:
        return None
    cutoff = now_ms - window_ms
    anchor_ts: int | None = None
    anchor_mid: float | None = None
    for ts_ms, mid in poly_mids:
        if ts_ms >= cutoff:
            anchor_ts, anchor_mid = ts_ms, mid
            break
    if anchor_ts is None:
        return None
    latest_ts, latest_mid = poly_mids[-1]
    dt_sec = (latest_ts - anchor_ts) / 1000.0
    if dt_sec < _EPS:
        return None
    return (latest_mid - anchor_mid) / dt_sec


# ---------------------------------------------------------------------------
# A9.3  Manipulation setup score
# ---------------------------------------------------------------------------

def manipulation_setup_score(
    abs_distance_to_strike_in_sigma: float,
    polymarket_extremity: float,
    realized_vol_60s_zscore: float,
    binance_volume_zscore_60s: float,
    polymarket_one_sided_imbalance: float,
) -> float:
    """Heuristic P(manipulation setup) — range [0, 1].

    v1 heuristic; replace with calibrated logistic in v2.
    Higher score = more likely a setup is forming.
    """
    z = (
        -1.0 * abs_distance_to_strike_in_sigma
        + 2.0 * polymarket_extremity
        - 0.5 * realized_vol_60s_zscore
        - 0.5 * binance_volume_zscore_60s
        + 0.3 * polymarket_one_sided_imbalance
    )
    # Clamp to avoid OverflowError: math.exp raises for |arg| > ~709
    z = max(-500.0, min(500.0, z))
    return float(1.0 / (1.0 + math.exp(-z)))


# ---------------------------------------------------------------------------
# A9.4  In-progress swing score
# ---------------------------------------------------------------------------

def inprogress_swing_score(
    buy_volume_5s: float,
    sell_volume_5s: float,
    log_return_5s: float,
    trade_count_zscore_1s: float,
    largest_trade_5s: float,
) -> tuple[float, str]:
    """Detect an in-progress price swing.

    Returns (score in [0,1], direction "UP" | "DOWN" | "NONE" | "AMBIGUOUS").
    """
    trade_spike = trade_count_zscore_1s > 2.5
    big_print = largest_trade_5s > LARGE_PRINT_THRESHOLD

    if not trade_spike and not big_print:
        return 0.0, "NONE"

    buy_dom = buy_volume_5s > 3.0 * (sell_volume_5s + _EPS) and log_return_5s > 0
    sell_dom = sell_volume_5s > 3.0 * (buy_volume_5s + _EPS) and log_return_5s < 0

    if buy_dom:
        return min(1.0, abs(log_return_5s) / VELOCITY_NORMALIZER), "UP"
    if sell_dom:
        return min(1.0, abs(log_return_5s) / VELOCITY_NORMALIZER), "DOWN"
    return 0.3, "AMBIGUOUS"


# ---------------------------------------------------------------------------
# Rolling z-score helper
# ---------------------------------------------------------------------------

def rolling_zscore(values: Sequence[float], current: float) -> float:
    """Z-score of `current` vs history. Returns 0.0 if std ≈ 0."""
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=float)
    sigma = float(arr.std())
    if sigma < _EPS:
        return 0.0
    return float((current - arr.mean()) / sigma)
