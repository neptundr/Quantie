"""Unit tests for policy layer — regime, edge, sizing, decision."""
from __future__ import annotations

import pytest

from quantie.policy.regime import (
    Regime,
    classify_regime,
    is_entry_allowed,
    is_passive_only,
)
from quantie.policy.edge import compute_edge, SIGNAL_BUY_UP, SIGNAL_BUY_DOWN, SIGNAL_HOLD
from quantie.policy.sizing import kelly_size
from quantie.policy.decision import Action, make_decision, should_hold_to_resolution


# ---------------------------------------------------------------------------
# classify_regime
# ---------------------------------------------------------------------------

def test_regime_early():
    assert classify_regime(300.0) == Regime.EARLY


def test_regime_early_boundary():
    assert classify_regime(240.1) == Regime.EARLY


def test_regime_mid():
    assert classify_regime(150.0) == Regime.MID


def test_regime_mid_at_boundary():
    assert classify_regime(240.0) == Regime.MID


def test_regime_late():
    assert classify_regime(45.0) == Regime.LATE


def test_regime_late_at_boundary():
    assert classify_regime(60.0) == Regime.LATE


def test_regime_terminal():
    assert classify_regime(20.0) == Regime.TERMINAL


def test_regime_terminal_at_boundary():
    assert classify_regime(30.0) == Regime.TERMINAL


def test_is_entry_allowed_early():
    assert is_entry_allowed(Regime.EARLY) is True


def test_is_entry_allowed_mid():
    assert is_entry_allowed(Regime.MID) is True


def test_is_entry_allowed_late():
    assert is_entry_allowed(Regime.LATE) is True


def test_is_entry_not_allowed_terminal():
    assert is_entry_allowed(Regime.TERMINAL) is False


def test_is_passive_only_early():
    assert is_passive_only(Regime.EARLY) is True


def test_is_passive_only_mid_false():
    assert is_passive_only(Regime.MID) is False


# ---------------------------------------------------------------------------
# compute_edge
# ---------------------------------------------------------------------------

def test_edge_buy_up_signal():
    # p_up=0.80, ask=0.60 → edge_buy_up=0.20, threshold≈0.042
    result = compute_edge(
        p_up=0.80, poly_up_ask=0.60, poly_up_bid=0.58,
        ci_width=0.10, regime=Regime.MID,
    )
    assert result.signal == SIGNAL_BUY_UP
    assert result.edge_buy_up == pytest.approx(0.20)


def test_edge_buy_down_signal():
    # p_up=0.20, bid=0.42 → edge_buy_down=0.22
    result = compute_edge(
        p_up=0.20, poly_up_ask=0.55, poly_up_bid=0.42,
        ci_width=0.10, regime=Regime.MID,
    )
    assert result.signal == SIGNAL_BUY_DOWN
    assert result.edge_buy_down == pytest.approx(0.22)


def test_edge_hold_no_signal():
    result = compute_edge(
        p_up=0.50, poly_up_ask=0.50, poly_up_bid=0.49,
        ci_width=0.05, regime=Regime.MID,
    )
    assert result.signal == SIGNAL_HOLD


def test_edge_late_extra_threshold():
    mid = compute_edge(
        p_up=0.75, poly_up_ask=0.70, poly_up_bid=0.68,
        ci_width=0.05, regime=Regime.MID,
    )
    late = compute_edge(
        p_up=0.75, poly_up_ask=0.70, poly_up_bid=0.68,
        ci_width=0.05, regime=Regime.LATE,
    )
    assert late.threshold > mid.threshold
    assert late.threshold == pytest.approx(mid.threshold + 0.01)


def test_edge_late_wide_ci_forces_hold():
    # ci_width >= 0.08 → HOLD in LATE regardless of edge
    result = compute_edge(
        p_up=0.90, poly_up_ask=0.50, poly_up_bid=0.48,
        ci_width=0.08, regime=Regime.LATE,
    )
    assert result.signal == SIGNAL_HOLD


def test_edge_poly_mid():
    result = compute_edge(
        p_up=0.55, poly_up_ask=0.52, poly_up_bid=0.48,
        ci_width=0.05, regime=Regime.MID,
    )
    assert result.poly_mid == pytest.approx(0.50)


def test_edge_ci_width_stored():
    result = compute_edge(
        p_up=0.55, poly_up_ask=0.52, poly_up_bid=0.48,
        ci_width=0.12, regime=Regime.MID,
    )
    assert result.ci_width == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# kelly_size
# ---------------------------------------------------------------------------

def test_kelly_positive_edge():
    size = kelly_size(p_win=0.70, price=0.50, bankroll=1000.0)
    assert size > 0.0


def test_kelly_capped_at_15pct():
    size = kelly_size(p_win=0.99, price=0.50, bankroll=1000.0)
    assert size <= 150.0


def test_kelly_zero_when_below_min():
    size = kelly_size(p_win=0.55, price=0.50, bankroll=10.0)
    assert size == 0.0


def test_kelly_zero_when_no_edge():
    size = kelly_size(p_win=0.50, price=0.50, bankroll=1000.0)
    assert size == 0.0


def test_kelly_zero_invalid_price():
    assert kelly_size(p_win=0.70, price=0.0, bankroll=1000.0) == 0.0
    assert kelly_size(p_win=0.70, price=1.0, bankroll=1000.0) == 0.0


def test_kelly_zero_empty_bankroll():
    assert kelly_size(p_win=0.70, price=0.50, bankroll=0.0) == 0.0


def test_kelly_rounded_to_cents():
    size = kelly_size(p_win=0.65, price=0.50, bankroll=1000.0)
    assert round(size, 2) == size


def test_kelly_larger_edge_larger_size():
    # p_win=0.55 → fraction=0.025 → $25 (below cap); p_win=0.65 → $50 (capped)
    s1 = kelly_size(p_win=0.55, price=0.50, bankroll=1000.0)
    s2 = kelly_size(p_win=0.65, price=0.50, bankroll=1000.0)
    assert s2 > s1


# ---------------------------------------------------------------------------
# make_decision
# ---------------------------------------------------------------------------

def test_decision_terminal_exits():
    d = make_decision(
        p_up=0.80, poly_up_ask=0.60, poly_up_bid=0.58,
        ci_width=0.05, tau_sec=20.0, bankroll=1000.0,
    )
    assert d.action == Action.EXIT
    assert d.size_usd == 0.0


def test_decision_mid_take_up():
    d = make_decision(
        p_up=0.80, poly_up_ask=0.60, poly_up_bid=0.58,
        ci_width=0.05, tau_sec=150.0, bankroll=1000.0,
    )
    assert d.action == Action.TAKE_UP
    assert d.size_usd > 0.0
    assert d.limit_price is None


def test_decision_mid_take_down():
    d = make_decision(
        p_up=0.20, poly_up_ask=0.55, poly_up_bid=0.40,
        ci_width=0.05, tau_sec=150.0, bankroll=1000.0,
    )
    assert d.action == Action.TAKE_DOWN
    assert d.size_usd > 0.0


def test_decision_mid_hold_no_edge():
    d = make_decision(
        p_up=0.50, poly_up_ask=0.50, poly_up_bid=0.49,
        ci_width=0.05, tau_sec=150.0, bankroll=1000.0,
    )
    assert d.action == Action.HOLD
    assert d.size_usd == 0.0


def test_decision_early_make_up():
    d = make_decision(
        p_up=0.80, poly_up_ask=0.60, poly_up_bid=0.58,
        ci_width=0.05, tau_sec=280.0, bankroll=1000.0,
        fair_value=0.75,
    )
    assert d.action == Action.MAKE_UP
    assert d.limit_price is not None
    assert d.limit_price == pytest.approx(0.75 - 0.015, abs=1e-4)


def test_decision_early_make_down():
    d = make_decision(
        p_up=0.20, poly_up_ask=0.55, poly_up_bid=0.40,
        ci_width=0.05, tau_sec=280.0, bankroll=1000.0,
        fair_value=0.25,
    )
    assert d.action == Action.MAKE_DOWN
    assert d.limit_price is not None


def test_decision_late_hold_wide_ci():
    d = make_decision(
        p_up=0.90, poly_up_ask=0.50, poly_up_bid=0.48,
        ci_width=0.08, tau_sec=45.0, bankroll=1000.0,
    )
    assert d.action == Action.HOLD


def test_decision_has_regime():
    d = make_decision(
        p_up=0.80, poly_up_ask=0.60, poly_up_bid=0.58,
        ci_width=0.05, tau_sec=150.0, bankroll=1000.0,
    )
    assert d.regime == Regime.MID


def test_decision_reason_nonempty():
    d = make_decision(
        p_up=0.80, poly_up_ask=0.60, poly_up_bid=0.58,
        ci_width=0.05, tau_sec=150.0, bankroll=1000.0,
    )
    assert len(d.reason) > 0


# ---------------------------------------------------------------------------
# should_hold_to_resolution
# ---------------------------------------------------------------------------

def test_hold_to_resolution_above_threshold():
    assert should_hold_to_resolution(0.75) is True


def test_hold_to_resolution_at_threshold():
    assert should_hold_to_resolution(0.70) is True


def test_hold_to_resolution_below_threshold():
    assert should_hold_to_resolution(0.65) is False
