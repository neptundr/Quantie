"""Unit tests for feature functions — synthetic inputs, no network."""
from __future__ import annotations

import math
import pytest

from quantie.features.volatility import RealizedVolEstimator
from quantie.features.microstructure import (
    LARGE_PRINT_THRESHOLD,
    compute_basis,
    compute_depth_imbalance,
    compute_momentum,
    compute_ofi,
    compute_poly_mid_velocity,
    inprogress_swing_score,
    manipulation_setup_score,
    rolling_zscore,
)
from quantie.features.time_features import (
    compute_distance_from_strike,
    compute_tau_features,
)


# ---------------------------------------------------------------------------
# RealizedVolEstimator
# ---------------------------------------------------------------------------

def test_vol_estimator_returns_positive():
    est = RealizedVolEstimator(halflife_seconds=60.0)
    for i in range(10):
        est.update(math.log(60000 + i * 10))
    assert est.vol > 0.0


def test_vol_estimator_cold_start_nonzero():
    est = RealizedVolEstimator()
    v = est.update(math.log(60000.0))
    assert v > 0.0


def test_vol_estimator_higher_vol_for_volatile_series():
    stable = RealizedVolEstimator(halflife_seconds=30.0)
    volatile = RealizedVolEstimator(halflife_seconds=30.0)
    for i in range(20):
        stable.update(math.log(60000.0 + i * 0.1))
        volatile.update(math.log(60000.0 + i * 50.0))
    assert volatile.vol > stable.vol


def test_vol_estimator_reset():
    est = RealizedVolEstimator()
    for i in range(5):
        est.update(math.log(60000.0 + i))
    est.reset()
    assert est.vol == 0.0


def test_vol_estimator_short_halflife_reacts_faster():
    fast = RealizedVolEstimator(halflife_seconds=5.0)
    slow = RealizedVolEstimator(halflife_seconds=300.0)
    base = math.log(60000.0)
    for _ in range(50):
        fast.update(base)
        slow.update(base)
    spike = math.log(61000.0)
    v_fast = fast.update(spike)
    v_slow = slow.update(spike)
    assert v_fast > v_slow


# ---------------------------------------------------------------------------
# compute_ofi
# ---------------------------------------------------------------------------

def test_ofi_buy_heavy():
    trades = [(1000, 1.0, True), (2000, 0.5, True), (3000, 0.2, False)]
    assert compute_ofi(trades, 5000, 5000) == pytest.approx(1.3)


def test_ofi_sell_heavy():
    trades = [(1000, 2.0, False), (2000, 0.5, False)]
    assert compute_ofi(trades, 5000, 5000) == pytest.approx(-2.5)


def test_ofi_empty():
    assert compute_ofi([], 5000, 5000) == 0.0


def test_ofi_excludes_outside_window():
    trades = [(1000, 5.0, True), (9000, 1.0, True)]
    # cutoff = 10000 - 2000 = 8000; only ts=9000 qualifies
    assert compute_ofi(trades, 2000, 10000) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_depth_imbalance
# ---------------------------------------------------------------------------

def test_depth_imbalance_bid_heavy():
    bids = [(60000, 5.0), (59999, 3.0)]
    asks = [(60001, 1.0), (60002, 0.5)]
    assert compute_depth_imbalance(bids, asks) > 0.0


def test_depth_imbalance_ask_heavy():
    bids = [(60000, 0.5)]
    asks = [(60001, 5.0), (60002, 3.0)]
    assert compute_depth_imbalance(bids, asks) < 0.0


def test_depth_imbalance_balanced():
    bids = [(60000, 2.0)]
    asks = [(60001, 2.0)]
    assert compute_depth_imbalance(bids, asks) == pytest.approx(0.0)


def test_depth_imbalance_empty():
    assert compute_depth_imbalance([], []) == 0.0


# ---------------------------------------------------------------------------
# compute_momentum
# ---------------------------------------------------------------------------

def test_momentum_positive():
    prices = [(0, math.log(60000)), (5000, math.log(60100))]
    result = compute_momentum(prices, 10000, 10000)
    assert result is not None and result > 0.0


def test_momentum_negative():
    prices = [(0, math.log(60100)), (5000, math.log(59900))]
    result = compute_momentum(prices, 10000, 10000)
    assert result is not None and result < 0.0


def test_momentum_empty_returns_none():
    assert compute_momentum([], 5000, 5000) is None


def test_momentum_no_data_in_window_returns_none():
    prices = [(0, math.log(60000))]
    # cutoff = 9000; ts=0 is outside window
    assert compute_momentum(prices, 1000, 10000) is None


# ---------------------------------------------------------------------------
# compute_basis
# ---------------------------------------------------------------------------

def test_basis_positive_when_binance_above():
    assert compute_basis(60100.0, 60000.0) == pytest.approx(100.0)


def test_basis_negative_when_binance_below():
    assert compute_basis(59900.0, 60000.0) == pytest.approx(-100.0)


# ---------------------------------------------------------------------------
# compute_poly_mid_velocity
# ---------------------------------------------------------------------------

def test_poly_mid_velocity_rising():
    mids = [(0, 0.48), (5000, 0.52)]
    result = compute_poly_mid_velocity(mids, window_ms=10000, now_ms=10000)
    assert result is not None and result > 0.0


def test_poly_mid_velocity_none_for_single_point():
    assert compute_poly_mid_velocity([(0, 0.5)], window_ms=5000, now_ms=5000) is None


# ---------------------------------------------------------------------------
# manipulation_setup_score
# ---------------------------------------------------------------------------

def test_manipulation_score_in_range():
    score = manipulation_setup_score(0.5, 0.3, -1.0, -1.0, 0.2)
    assert 0.0 < score < 1.0


def test_manipulation_score_higher_near_strike_extreme():
    high = manipulation_setup_score(0.0, 0.49, -2.0, -2.0, 0.8)
    low = manipulation_setup_score(3.0, 0.0, 2.0, 2.0, 0.0)
    assert high > low


# ---------------------------------------------------------------------------
# inprogress_swing_score
# ---------------------------------------------------------------------------

def test_inprogress_no_signal():
    score, direction = inprogress_swing_score(0.5, 0.5, 0.0001, 0.5, 0.1)
    assert direction == "NONE"
    assert score == 0.0


def test_inprogress_up_signal():
    score, direction = inprogress_swing_score(
        10.0, 0.1, 0.005, 3.0, LARGE_PRINT_THRESHOLD + 0.1
    )
    assert direction == "UP"
    assert score > 0.0


def test_inprogress_down_signal():
    score, direction = inprogress_swing_score(
        0.1, 10.0, -0.005, 3.0, LARGE_PRINT_THRESHOLD + 0.1
    )
    assert direction == "DOWN"
    assert score > 0.0


# ---------------------------------------------------------------------------
# compute_tau_features
# ---------------------------------------------------------------------------

def test_tau_features_mid_window():
    f = compute_tau_features(150.0)
    assert f["tau_seconds"] == pytest.approx(150.0)
    assert f["tau_normalized"] == pytest.approx(0.5)


def test_tau_features_clamps_negative():
    f = compute_tau_features(-10.0)
    assert f["tau_seconds"] == 0.0
    assert f["tau_normalized"] == 0.0


# ---------------------------------------------------------------------------
# compute_distance_from_strike
# ---------------------------------------------------------------------------

def test_distance_above_strike_positive():
    d = compute_distance_from_strike(60100, 60000, 0.001, 120)
    assert d is not None and d > 0.0


def test_distance_below_strike_negative():
    d = compute_distance_from_strike(59900, 60000, 0.001, 120)
    assert d is not None and d < 0.0


def test_distance_none_at_zero_tau():
    assert compute_distance_from_strike(60000, 60000, 0.001, 0.0) is None


def test_distance_none_at_zero_sigma():
    assert compute_distance_from_strike(60000, 60000, 0.0, 120.0) is None


# ---------------------------------------------------------------------------
# rolling_zscore
# ---------------------------------------------------------------------------

def test_rolling_zscore_within_range():
    z = rolling_zscore([1.0, 2.0, 3.0, 4.0, 5.0], 5.0)
    assert abs(z) < 3.0


def test_rolling_zscore_empty_returns_zero():
    assert rolling_zscore([], 5.0) == 0.0


def test_rolling_zscore_uniform_returns_zero():
    assert rolling_zscore([3.0, 3.0, 3.0], 3.0) == 0.0
