"""Unit tests for P&L tracker."""
from __future__ import annotations

import pytest

from quantie.pnl import PnLTracker


def _tracker() -> PnLTracker:
    return PnLTracker(starting_bankroll=1000.0)


# ---------------------------------------------------------------------------
# record_fill
# ---------------------------------------------------------------------------

def test_fill_increments_orders_filled():
    t = _tracker()
    t.record_fill("o1", "UP", fill_price=0.60, filled_usd=20.0)
    assert t.orders_filled == 1


def test_fill_zero_price_ignored():
    t = _tracker()
    t.record_fill("o1", "UP", fill_price=0.0, filled_usd=20.0)
    assert t.orders_filled == 0


def test_open_exposure_after_fill():
    t = _tracker()
    t.record_fill("o1", "UP", fill_price=0.60, filled_usd=20.0)
    assert t.total_open_exposure() == pytest.approx(20.0)


def test_open_fills_returns_unresolved():
    t = _tracker()
    t.record_fill("o1", "UP", fill_price=0.60, filled_usd=20.0)
    assert len(t.open_fills()) == 1


# ---------------------------------------------------------------------------
# record_placed / record_cancelled
# ---------------------------------------------------------------------------

def test_record_placed_increments():
    t = _tracker()
    t.record_placed()
    t.record_placed()
    assert t.orders_placed == 2


def test_record_cancelled_increments():
    t = _tracker()
    t.record_cancelled()
    assert t.orders_cancelled == 1


# ---------------------------------------------------------------------------
# record_resolution — win
# ---------------------------------------------------------------------------

def test_resolution_win_positive_pnl():
    t = _tracker()
    # shares = 20/0.60 = 33.33; payout = 33.33; pnl = +13.33
    t.record_fill("o1", "UP", fill_price=0.60, filled_usd=20.0)
    delta = t.record_resolution("UP", won=True)
    assert delta == pytest.approx(20.0 / 0.60 - 20.0, rel=1e-4)
    assert t.realized_pnl == pytest.approx(delta)


def test_resolution_loss_negative_pnl():
    t = _tracker()
    t.record_fill("o1", "UP", fill_price=0.60, filled_usd=20.0)
    delta = t.record_resolution("UP", won=False)
    assert delta == pytest.approx(-20.0)
    assert t.realized_pnl == pytest.approx(-20.0)


def test_resolution_clears_open_fills():
    t = _tracker()
    t.record_fill("o1", "UP", fill_price=0.60, filled_usd=20.0)
    t.record_resolution("UP", won=True)
    assert t.open_fills() == []
    assert t.total_open_exposure() == pytest.approx(0.0)


def test_resolution_only_affects_matching_asset():
    t = _tracker()
    t.record_fill("o1", "UP", fill_price=0.60, filled_usd=20.0)
    t.record_fill("o2", "DOWN", fill_price=0.40, filled_usd=10.0)
    delta = t.record_resolution("UP", won=True)
    assert delta > 0
    assert len(t.open_fills()) == 1
    assert t.open_fills()[0].asset_id == "DOWN"


def test_resolution_idempotent():
    t = _tracker()
    t.record_fill("o1", "UP", fill_price=0.60, filled_usd=20.0)
    t.record_resolution("UP", won=True)
    delta2 = t.record_resolution("UP", won=True)
    assert delta2 == pytest.approx(0.0)


def test_multiple_fills_same_asset_resolved_together():
    t = _tracker()
    t.record_fill("o1", "UP", fill_price=0.60, filled_usd=20.0)
    t.record_fill("o2", "UP", fill_price=0.65, filled_usd=15.0)
    delta = t.record_resolution("UP", won=False)
    assert delta == pytest.approx(-35.0)


def test_pnl_accumulates_across_windows():
    t = _tracker()
    t.record_fill("o1", "UP", fill_price=0.60, filled_usd=20.0)
    t.record_resolution("UP", won=True)
    win1 = t.realized_pnl

    t.record_fill("o2", "DOWN", fill_price=0.40, filled_usd=10.0)
    t.record_resolution("DOWN", won=False)
    assert t.realized_pnl == pytest.approx(win1 - 10.0)


# ---------------------------------------------------------------------------
# window_summary
# ---------------------------------------------------------------------------

def test_window_summary_has_required_keys():
    t = _tracker()
    t.record_placed()
    t.record_fill("o1", "UP", fill_price=0.60, filled_usd=20.0)
    t.record_cancelled()
    s = t.window_summary()
    for key in ("realized_pnl_usd", "orders_placed", "orders_filled",
                "orders_cancelled", "open_exposure_usd"):
        assert key in s


def test_window_summary_counts_correct():
    t = _tracker()
    t.record_placed()
    t.record_fill("o1", "UP", fill_price=0.60, filled_usd=20.0)
    s = t.window_summary()
    assert s["orders_placed"] == 1
    assert s["orders_filled"] == 1
    assert s["open_exposure_usd"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# starting_bankroll
# ---------------------------------------------------------------------------

def test_starting_bankroll_preserved():
    t = PnLTracker(starting_bankroll=500.0)
    assert t.starting_bankroll == pytest.approx(500.0)


def test_initial_pnl_zero():
    t = _tracker()
    assert t.realized_pnl == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# record_early_exit
# ---------------------------------------------------------------------------

class TestEarlyExit:
    def test_profit_on_favourable_exit(self) -> None:
        t = _tracker()
        t.record_fill("o1", "UP", fill_price=0.50, filled_usd=50.0)
        delta = t.record_early_exit("o1", proceeds_usd=60.0)
        assert delta == pytest.approx(10.0)
        assert t.realized_pnl == pytest.approx(10.0)

    def test_loss_on_unfavourable_exit(self) -> None:
        t = _tracker()
        t.record_fill("o1", "UP", fill_price=0.60, filled_usd=60.0)
        delta = t.record_early_exit("o1", proceeds_usd=40.0)
        assert delta == pytest.approx(-20.0)
        assert t.realized_pnl == pytest.approx(-20.0)

    def test_fill_marked_resolved(self) -> None:
        t = _tracker()
        t.record_fill("o1", "UP", fill_price=0.50, filled_usd=50.0)
        t.record_early_exit("o1", proceeds_usd=55.0)
        assert t.open_fills() == []

    def test_double_exit_ignored(self) -> None:
        t = _tracker()
        t.record_fill("o1", "UP", fill_price=0.50, filled_usd=50.0)
        t.record_early_exit("o1", proceeds_usd=55.0)
        delta2 = t.record_early_exit("o1", proceeds_usd=55.0)
        assert delta2 == pytest.approx(0.0)
        assert t.realized_pnl == pytest.approx(5.0)

    def test_unknown_order_id_returns_zero(self) -> None:
        t = _tracker()
        delta = t.record_early_exit("nonexistent", proceeds_usd=50.0)
        assert delta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mark_pending_resolution / pending_fills
# ---------------------------------------------------------------------------

class TestPendingResolution:
    def test_mark_moves_fills_out_of_open(self) -> None:
        """After window ends, old fills must not appear in open_fills()."""
        t = _tracker()
        t.record_fill("o1", "UP", fill_price=0.60, filled_usd=12.0)
        assert len(t.open_fills()) == 1

        marked = t.mark_pending_resolution()

        assert marked == 1
        assert t.open_fills() == []           # new window sees no inherited exposure
        assert len(t.pending_fills()) == 1    # fill is tracked, not lost

    def test_resolution_still_works_on_pending_fill(self) -> None:
        """Gamma resolution must settle pending fills and record PnL correctly."""
        t = _tracker()
        t.record_fill("o1", "UP", fill_price=0.60, filled_usd=12.0)
        t.mark_pending_resolution()

        # shares = 12.0 / 0.60 = 20; payout = 20 * 1.0 = 20; pnl = 20 - 12 = 8
        delta = t.record_resolution("UP", won=True)

        assert delta == pytest.approx(8.0)
        assert t.realized_pnl == pytest.approx(8.0)
        assert t.pending_fills() == []

    def test_resolution_loss_on_pending_fill(self) -> None:
        t = _tracker()
        t.record_fill("o1", "UP", fill_price=0.60, filled_usd=12.0)
        t.mark_pending_resolution()

        # payout = 0; pnl = 0 - 12 = -12
        delta = t.record_resolution("UP", won=False)

        assert delta == pytest.approx(-12.0)
        assert t.realized_pnl == pytest.approx(-12.0)

    def test_early_exit_blocked_once_pending(self) -> None:
        """Cannot early-exit a position after it has been marked pending resolution."""
        t = _tracker()
        t.record_fill("o1", "UP", fill_price=0.60, filled_usd=12.0)
        t.mark_pending_resolution()

        delta = t.record_early_exit("o1", proceeds_usd=15.0)

        assert delta == pytest.approx(0.0)   # silently ignored
        assert t.realized_pnl == pytest.approx(0.0)

    def test_already_resolved_fill_not_marked_pending(self) -> None:
        t = _tracker()
        t.record_fill("o1", "UP", fill_price=0.60, filled_usd=12.0)
        t.record_early_exit("o1", proceeds_usd=14.0)  # already resolved
        marked = t.mark_pending_resolution()
        assert marked == 0

    def test_new_window_starts_with_clean_exposure(self) -> None:
        """Total open exposure must be zero for the new window after mark."""
        t = _tracker()
        t.record_fill("o1", "UP", fill_price=0.60, filled_usd=12.0)
        t.mark_pending_resolution()
        assert t.total_open_exposure() == pytest.approx(0.0)
