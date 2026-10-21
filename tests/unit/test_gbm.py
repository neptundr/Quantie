"""Unit tests for GBM closed-form model."""
from __future__ import annotations

import pytest
from quantie.models.gbm import gbm_p_up


def test_at_expiry_above_strike_returns_one():
    assert gbm_p_up(S0=60000, S_t=60100, sigma_per_sec=0.001, tau_sec=0) == 1.0


def test_at_expiry_below_strike_returns_zero():
    assert gbm_p_up(S0=60000, S_t=59900, sigma_per_sec=0.001, tau_sec=0) == 0.0


def test_at_expiry_equal_to_strike_returns_one():
    # Equality resolves UP (Polymarket rule)
    assert gbm_p_up(S0=60000, S_t=60000, sigma_per_sec=0.001, tau_sec=0) == 1.0


def test_at_the_money_zero_drift_approx_half():
    p = gbm_p_up(S0=60000, S_t=60000, sigma_per_sec=0.001, tau_sec=150)
    assert p == pytest.approx(0.5, abs=0.01)


def test_above_strike_gives_p_above_half():
    p = gbm_p_up(S0=60000, S_t=60500, sigma_per_sec=0.001, tau_sec=60)
    assert p > 0.5


def test_below_strike_gives_p_below_half():
    p = gbm_p_up(S0=60000, S_t=59500, sigma_per_sec=0.001, tau_sec=60)
    assert p < 0.5


def test_output_bounded_in_unit_interval():
    for s_t in [55000, 59999, 60000, 60001, 65000]:
        p = gbm_p_up(S0=60000, S_t=s_t, sigma_per_sec=0.001, tau_sec=120)
        assert 0.0 <= p <= 1.0


def test_zero_vol_above_strike_returns_one():
    assert gbm_p_up(S0=60000, S_t=60100, sigma_per_sec=0.0, tau_sec=120) == 1.0


def test_zero_vol_below_strike_returns_zero():
    assert gbm_p_up(S0=60000, S_t=59900, sigma_per_sec=0.0, tau_sec=120) == 0.0


def test_longer_tau_increases_uncertainty():
    # Short tau → more certain → higher P when above strike
    p_short = gbm_p_up(S0=60000, S_t=60100, sigma_per_sec=0.001, tau_sec=10)
    p_long = gbm_p_up(S0=60000, S_t=60100, sigma_per_sec=0.001, tau_sec=290)
    assert p_short > p_long


def test_positive_drift_increases_p_up():
    p_no = gbm_p_up(S0=60000, S_t=60000, sigma_per_sec=0.001, tau_sec=150, drift_per_sec=0.0)
    p_dr = gbm_p_up(S0=60000, S_t=60000, sigma_per_sec=0.001, tau_sec=150, drift_per_sec=0.01)
    assert p_dr > p_no
