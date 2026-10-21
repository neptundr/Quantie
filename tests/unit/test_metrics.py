"""Unit tests for Prometheus metrics helpers."""
from __future__ import annotations

import pytest

from quantie.metrics import (
    _REGIME_VALUES,
    model_p_up,
    orders_cancelled_total,
    orders_filled_total,
    orders_placed_total,
    pnl_realized_usd,
    position_size_usd,
    record_error,
    record_order_cancelled,
    record_order_filled,
    record_order_placed,
    regime_gauge,
    set_model_p_up,
    set_pnl,
    set_position,
    set_regime,
)


# ---------------------------------------------------------------------------
# Counters increment
# ---------------------------------------------------------------------------

def test_record_order_placed_increments():
    before = orders_placed_total._value.get()
    record_order_placed()
    assert orders_placed_total._value.get() == before + 1


def test_record_order_filled_increments():
    before = orders_filled_total._value.get()
    record_order_filled(filled_usd=25.0)
    assert orders_filled_total._value.get() == before + 1


def test_record_order_cancelled_increments():
    before = orders_cancelled_total._value.get()
    record_order_cancelled()
    assert orders_cancelled_total._value.get() == before + 1


def test_record_error_increments():
    from quantie.metrics import errors_total
    before = errors_total.labels(error_type="test_err")._value.get()
    record_error("test_err")
    assert errors_total.labels(error_type="test_err")._value.get() == before + 1


# ---------------------------------------------------------------------------
# Gauges set correctly
# ---------------------------------------------------------------------------

def test_set_model_p_up():
    set_model_p_up(0.73)
    assert model_p_up._value.get() == pytest.approx(0.73)


def test_set_pnl():
    set_pnl(-12.50)
    assert pnl_realized_usd._value.get() == pytest.approx(-12.50)


def test_set_position():
    set_position(45.0)
    assert position_size_usd._value.get() == pytest.approx(45.0)


def test_set_regime_early():
    set_regime("early")
    assert regime_gauge._value.get() == pytest.approx(0)


def test_set_regime_mid():
    set_regime("mid")
    assert regime_gauge._value.get() == pytest.approx(1)


def test_set_regime_late():
    set_regime("late")
    assert regime_gauge._value.get() == pytest.approx(2)


def test_set_regime_terminal():
    set_regime("terminal")
    assert regime_gauge._value.get() == pytest.approx(3)


def test_set_regime_unknown_sets_minus_one():
    set_regime("unknown_regime")
    assert regime_gauge._value.get() == pytest.approx(-1)


def test_set_regime_case_insensitive():
    set_regime("MID")
    assert regime_gauge._value.get() == pytest.approx(1)


# ---------------------------------------------------------------------------
# Regime values mapping completeness
# ---------------------------------------------------------------------------

def test_regime_values_complete():
    for name in ("early", "mid", "late", "terminal"):
        assert name in _REGIME_VALUES
