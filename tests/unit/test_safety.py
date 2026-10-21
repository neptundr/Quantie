"""Unit tests for pre-trade safety checks."""
from __future__ import annotations

import pytest

from quantie.policy.decision import Action
from quantie.safety.checks import SafetySnapshot, pre_trade_check

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_snapshot(**overrides) -> SafetySnapshot:
    """Return a passing SafetySnapshot; override fields to trigger failures."""
    defaults = dict(
        kill_switch=False,
        realized_pnl_usd=0.0,
        starting_bankroll=1000.0,
        chainlink_age_ms=500.0,
        binance_age_ms=500.0,
        window_exposure_usd=0.0,
        max_position_usd=50.0,
        tau_sec=120.0,
    )
    defaults.update(overrides)
    return SafetySnapshot(**defaults)


# ---------------------------------------------------------------------------
# EXIT / HOLD bypass checks
# ---------------------------------------------------------------------------

def test_exit_always_allowed():
    snap = _ok_snapshot(kill_switch=True)
    ok, reason = pre_trade_check(Action.EXIT, snap)
    assert ok is True
    assert reason == ""


def test_hold_always_allowed():
    snap = _ok_snapshot(kill_switch=True)
    ok, reason = pre_trade_check(Action.HOLD, snap)
    assert ok is True


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

def test_kill_switch_blocks_take_up():
    snap = _ok_snapshot(kill_switch=True)
    ok, reason = pre_trade_check(Action.TAKE_UP, snap)
    assert ok is False
    assert "kill switch" in reason


def test_kill_switch_blocks_make_up():
    snap = _ok_snapshot(kill_switch=True)
    ok, _ = pre_trade_check(Action.MAKE_UP, snap)
    assert ok is False


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def test_drawdown_blocks_at_limit():
    # 28% of 1000 = 280; pnl = -280 → blocked
    snap = _ok_snapshot(realized_pnl_usd=-280.0)
    ok, reason = pre_trade_check(Action.TAKE_UP, snap)
    assert ok is False
    assert "drawdown" in reason


def test_drawdown_allows_just_above_limit():
    snap = _ok_snapshot(realized_pnl_usd=-279.99)
    ok, _ = pre_trade_check(Action.TAKE_UP, snap)
    assert ok is True


def test_drawdown_positive_pnl_allowed():
    snap = _ok_snapshot(realized_pnl_usd=50.0)
    ok, _ = pre_trade_check(Action.TAKE_UP, snap)
    assert ok is True


# ---------------------------------------------------------------------------
# Stale Chainlink
# ---------------------------------------------------------------------------

def test_stale_chainlink_blocks():
    snap = _ok_snapshot(chainlink_age_ms=2501.0)
    ok, reason = pre_trade_check(Action.TAKE_UP, snap)
    assert ok is False
    assert "Chainlink" in reason


def test_chainlink_at_limit_allowed():
    snap = _ok_snapshot(chainlink_age_ms=2500.0)
    ok, _ = pre_trade_check(Action.TAKE_UP, snap)
    assert ok is True


# ---------------------------------------------------------------------------
# Stale Binance
# ---------------------------------------------------------------------------

def test_stale_binance_blocks():
    snap = _ok_snapshot(binance_age_ms=1501.0)
    ok, reason = pre_trade_check(Action.TAKE_UP, snap)
    assert ok is False
    assert "Binance" in reason


def test_binance_at_limit_allowed():
    snap = _ok_snapshot(binance_age_ms=1500.0)
    ok, _ = pre_trade_check(Action.TAKE_UP, snap)
    assert ok is True


# ---------------------------------------------------------------------------
# Position cap
# ---------------------------------------------------------------------------

def test_position_cap_blocks_at_limit():
    snap = _ok_snapshot(window_exposure_usd=50.0, max_position_usd=50.0)
    ok, reason = pre_trade_check(Action.TAKE_UP, snap)
    assert ok is False
    assert "cap" in reason


def test_position_cap_allows_below_limit():
    snap = _ok_snapshot(window_exposure_usd=49.99, max_position_usd=50.0)
    ok, _ = pre_trade_check(Action.TAKE_UP, snap)
    assert ok is True


# ---------------------------------------------------------------------------
# Terminal zone
# ---------------------------------------------------------------------------

def test_terminal_zone_blocks_at_boundary():
    snap = _ok_snapshot(tau_sec=30.0)
    ok, reason = pre_trade_check(Action.TAKE_UP, snap)
    assert ok is False
    assert "terminal" in reason


def test_terminal_zone_blocks_make_action():
    snap = _ok_snapshot(tau_sec=25.0)
    ok, _ = pre_trade_check(Action.MAKE_UP, snap)
    assert ok is False


def test_terminal_zone_allows_above_boundary():
    snap = _ok_snapshot(tau_sec=30.1)
    ok, _ = pre_trade_check(Action.TAKE_UP, snap)
    assert ok is True


# ---------------------------------------------------------------------------
# All-clear path
# ---------------------------------------------------------------------------

def test_all_checks_pass_returns_true():
    snap = _ok_snapshot()
    ok, reason = pre_trade_check(Action.TAKE_UP, snap)
    assert ok is True
    assert reason == ""


def test_take_down_also_checked():
    snap = _ok_snapshot(kill_switch=True)
    ok, _ = pre_trade_check(Action.TAKE_DOWN, snap)
    assert ok is False


# ---------------------------------------------------------------------------
# Dynamic position cap
# ---------------------------------------------------------------------------

def test_dynamic_cap_blocks_when_exposure_reaches_balance_based_cap():
    """max_position_usd driven by dynamic equity (20% of window balance)."""
    # window balance=$500, cap=20%=$100, exposure=$100 → blocked
    snap = _ok_snapshot(
        window_exposure_usd=100.0,
        max_position_usd=100.0,
        starting_bankroll=500.0,
    )
    ok, reason = pre_trade_check(Action.TAKE_UP, snap)
    assert ok is False
    assert "cap" in reason


def test_dynamic_cap_allows_below_balance_based_cap():
    snap = _ok_snapshot(
        window_exposure_usd=99.99,
        max_position_usd=100.0,
        starting_bankroll=500.0,
    )
    ok, _ = pre_trade_check(Action.TAKE_UP, snap)
    assert ok is True
