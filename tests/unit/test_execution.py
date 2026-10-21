"""Unit tests for execution layer — PaperBroker and OrderExecutor."""
from __future__ import annotations

import pytest

from quantie.execution.broker import STATUS_CANCELLED, STATUS_FILLED, STATUS_OPEN, SIDE_SELL
from quantie.execution.paper_broker import PaperBroker
from quantie.execution.order_executor import OrderExecutor
from quantie.policy.decision import Action, PolicyDecision
from quantie.policy.edge import compute_edge
from quantie.policy.regime import Regime
from quantie.safety.checks import SafetySnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_safety(**overrides) -> SafetySnapshot:
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


def _take_up_decision(size: float = 20.0) -> PolicyDecision:
    regime = Regime.MID
    edge = compute_edge(0.80, 0.60, 0.58, 0.05, regime)
    return PolicyDecision(
        action=Action.TAKE_UP,
        size_usd=size,
        limit_price=None,
        edge=edge,
        regime=regime,
        reason="test",
    )


def _make_up_decision(limit: float = 0.58, size: float = 15.0) -> PolicyDecision:
    regime = Regime.EARLY
    edge = compute_edge(0.80, 0.60, 0.58, 0.05, regime)
    return PolicyDecision(
        action=Action.MAKE_UP,
        size_usd=size,
        limit_price=limit,
        edge=edge,
        regime=regime,
        reason="test passive",
    )


def _hold_decision() -> PolicyDecision:
    regime = Regime.MID
    edge = compute_edge(0.50, 0.50, 0.49, 0.05, regime)
    return PolicyDecision(
        action=Action.HOLD,
        size_usd=0.0,
        limit_price=None,
        edge=edge,
        regime=regime,
        reason="no edge",
    )


# ---------------------------------------------------------------------------
# PaperBroker — FOK
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fok_fills_when_price_ge_ask():
    broker = PaperBroker()
    result = await broker.place_fok_buy("UP", price=0.65, size_usd=20.0, current_ask=0.60)
    assert result.status == STATUS_FILLED
    assert result.filled_usd == 20.0
    assert result.is_paper is True


@pytest.mark.asyncio
async def test_fok_cancels_when_price_lt_ask():
    broker = PaperBroker()
    result = await broker.place_fok_buy("UP", price=0.55, size_usd=20.0, current_ask=0.60)
    assert result.status == STATUS_CANCELLED
    assert result.filled_usd == 0.0


@pytest.mark.asyncio
async def test_fok_applies_slippage():
    broker = PaperBroker()
    ask = 0.60
    result = await broker.place_fok_buy("UP", price=0.65, size_usd=20.0, current_ask=ask)
    assert result.price > ask


@pytest.mark.asyncio
async def test_fok_fills_at_price_when_no_ask_given():
    broker = PaperBroker()
    result = await broker.place_fok_buy("UP", price=0.60, size_usd=10.0)
    assert result.status == STATUS_FILLED


# ---------------------------------------------------------------------------
# PaperBroker — GTC
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gtc_returns_open():
    broker = PaperBroker()
    result = await broker.place_gtc_buy("UP", price=0.55, size_usd=15.0)
    assert result.status == STATUS_OPEN
    assert result.filled_usd == 0.0


@pytest.mark.asyncio
async def test_gtc_rests_in_open_orders():
    broker = PaperBroker()
    result = await broker.place_gtc_buy("UP", price=0.55, size_usd=15.0)
    assert result.order_id in broker.open_orders


@pytest.mark.asyncio
async def test_gtc_tick_fills_when_ask_at_or_below_limit():
    broker = PaperBroker()
    result = await broker.place_gtc_buy("UP", price=0.55, size_usd=15.0)
    # ask=0.54 <= limit=0.55 → fills
    fills = broker.tick_book("UP", best_ask=0.54)
    assert len(fills) == 1
    assert fills[0].status == STATUS_FILLED
    assert fills[0].order_id == result.order_id
    assert result.order_id not in broker.open_orders


@pytest.mark.asyncio
async def test_gtc_tick_no_fill_when_ask_above_limit():
    broker = PaperBroker()
    await broker.place_gtc_buy("UP", price=0.55, size_usd=15.0)
    # ask=0.56 > limit=0.55 → no fill
    fills = broker.tick_book("UP", best_ask=0.56)
    assert fills == []


@pytest.mark.asyncio
async def test_gtc_tick_ignores_other_asset():
    broker = PaperBroker()
    await broker.place_gtc_buy("UP", price=0.55, size_usd=15.0)
    fills = broker.tick_book("DOWN", best_ask=0.01)
    assert fills == []


# ---------------------------------------------------------------------------
# PaperBroker — cancel
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_order_removes_from_open():
    broker = PaperBroker()
    r = await broker.place_gtc_buy("UP", price=0.55, size_usd=15.0)
    cancelled = await broker.cancel_order(r.order_id)
    assert cancelled is True
    assert r.order_id not in broker.open_orders


@pytest.mark.asyncio
async def test_cancel_missing_order_returns_false():
    broker = PaperBroker()
    assert await broker.cancel_order("nonexistent") is False


@pytest.mark.asyncio
async def test_cancel_all_clears_open():
    broker = PaperBroker()
    await broker.place_gtc_buy("UP", price=0.55, size_usd=10.0)
    await broker.place_gtc_buy("UP", price=0.54, size_usd=10.0)
    count = await broker.cancel_all()
    assert count == 2
    assert len(broker.open_orders) == 0


# ---------------------------------------------------------------------------
# PaperBroker — get_order
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_order_returns_open():
    broker = PaperBroker()
    r = await broker.place_gtc_buy("UP", price=0.55, size_usd=15.0)
    fetched = await broker.get_order(r.order_id)
    assert fetched is not None
    assert fetched.status == STATUS_OPEN


@pytest.mark.asyncio
async def test_get_order_unknown_returns_none():
    broker = PaperBroker()
    assert await broker.get_order("ghost") is None


# ---------------------------------------------------------------------------
# OrderExecutor
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_executor_take_up_filled():
    broker = PaperBroker()
    executor = OrderExecutor(broker)
    result = await executor.execute(_take_up_decision(size=20.0), _ok_safety(),
                                    up_ask=0.60, up_bid=0.58)
    assert result is not None
    assert result.status == STATUS_FILLED
    assert executor.window_exposure_usd == 20.0


@pytest.mark.asyncio
async def test_executor_make_up_rests():
    broker = PaperBroker()
    executor = OrderExecutor(broker)
    result = await executor.execute(_make_up_decision(limit=0.58, size=15.0), _ok_safety(),
                                    up_ask=0.60, up_bid=0.58)
    assert result is not None
    assert result.status == STATUS_OPEN
    assert executor.window_exposure_usd == 0.0  # GTC not filled yet


@pytest.mark.asyncio
async def test_executor_hold_returns_none():
    broker = PaperBroker()
    executor = OrderExecutor(broker)
    result = await executor.execute(_hold_decision(), _ok_safety(), up_ask=0.50, up_bid=0.49)
    assert result is None
    assert executor.window_exposure_usd == 0.0


@pytest.mark.asyncio
async def test_executor_blocked_by_kill_switch():
    broker = PaperBroker()
    executor = OrderExecutor(broker)
    result = await executor.execute(_take_up_decision(), _ok_safety(kill_switch=True),
                                    up_ask=0.60, up_bid=0.58)
    assert result is None


@pytest.mark.asyncio
async def test_executor_reset_window_clears_exposure():
    broker = PaperBroker()
    executor = OrderExecutor(broker)
    await executor.execute(_take_up_decision(size=20.0), _ok_safety(),
                           up_ask=0.60, up_bid=0.58)
    assert executor.window_exposure_usd == 20.0
    executor.reset_window()
    assert executor.window_exposure_usd == 0.0


@pytest.mark.asyncio
async def test_executor_cancel_all():
    broker = PaperBroker()
    executor = OrderExecutor(broker)
    await executor.execute(_make_up_decision(), _ok_safety(), up_ask=0.60, up_bid=0.58)
    assert len(broker.open_orders) == 1
    count = await executor.cancel_all_open()
    assert count == 1
    assert len(broker.open_orders) == 0


@pytest.mark.asyncio
async def test_executor_unique_order_ids():
    broker = PaperBroker()
    executor = OrderExecutor(broker)
    r1 = await executor.execute(_take_up_decision(), _ok_safety(), up_ask=0.60, up_bid=0.58)
    r2 = await executor.execute(_take_up_decision(), _ok_safety(), up_ask=0.60, up_bid=0.58)
    assert r1 is not None and r2 is not None
    assert r1.order_id != r2.order_id


# ---------------------------------------------------------------------------
# PaperBroker.place_fok_sell
# ---------------------------------------------------------------------------

class TestFokSell:
    @pytest.mark.asyncio
    async def test_sell_fills_at_bid_minus_slippage(self) -> None:
        broker = PaperBroker()
        result = await broker.place_fok_sell(asset_id="UP", shares=100.0, current_bid=0.60)
        assert result.status == STATUS_FILLED
        assert result.side == SIDE_SELL
        # sell_price = 0.60 * (1 - 2/10000) slightly below bid
        assert result.price < 0.60
        assert result.filled_usd > 0.0

    @pytest.mark.asyncio
    async def test_sell_proceeds_equal_shares_times_sell_price(self) -> None:
        broker = PaperBroker()
        result = await broker.place_fok_sell(asset_id="UP", shares=50.0, current_bid=0.70)
        expected = 50.0 * (0.70 * (1 - 2 / 10_000))
        assert abs(result.filled_usd - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_sell_cancels_on_zero_bid(self) -> None:
        broker = PaperBroker()
        result = await broker.place_fok_sell(asset_id="UP", shares=10.0, current_bid=0.0)
        assert result.status == STATUS_CANCELLED
        assert result.filled_usd == 0.0

    @pytest.mark.asyncio
    async def test_sell_cancels_on_zero_shares(self) -> None:
        broker = PaperBroker()
        result = await broker.place_fok_sell(asset_id="UP", shares=0.0, current_bid=0.60)
        assert result.status == STATUS_CANCELLED

    @pytest.mark.asyncio
    async def test_sell_asset_id_preserved(self) -> None:
        broker = PaperBroker()
        result = await broker.place_fok_sell(asset_id="TOKEN_ABC", shares=10.0, current_bid=0.55)
        assert result.asset_id == "TOKEN_ABC"


# ---------------------------------------------------------------------------
# OrderExecutor.exit_positions
# ---------------------------------------------------------------------------

class TestExitPositions:
    @pytest.mark.asyncio
    async def test_exit_sells_up_fill_at_up_bid(self) -> None:
        from quantie.pnl import PnLTracker
        broker = PaperBroker()
        executor = OrderExecutor(broker)
        tracker = PnLTracker(1000.0)
        tracker.record_fill("oid1", "UP", fill_price=0.55, filled_usd=55.0)
        fills = tracker.open_fills()
        results = await executor.exit_positions(
            open_fills=fills,
            up_asset_id="UP",
            down_asset_id="DOWN",
            up_bid=0.65,
            down_bid=0.35,
        )
        assert len(results) == 1
        assert results[0].status == STATUS_FILLED
        assert results[0].side == SIDE_SELL
        assert results[0].asset_id == "UP"

    @pytest.mark.asyncio
    async def test_exit_sells_down_fill_at_down_bid(self) -> None:
        from quantie.pnl import PnLTracker
        broker = PaperBroker()
        executor = OrderExecutor(broker)
        tracker = PnLTracker(1000.0)
        tracker.record_fill("oid2", "DOWN", fill_price=0.45, filled_usd=45.0)
        fills = tracker.open_fills()
        results = await executor.exit_positions(
            open_fills=fills,
            up_asset_id="UP",
            down_asset_id="DOWN",
            up_bid=0.58,
            down_bid=0.42,
        )
        assert len(results) == 1
        assert results[0].asset_id == "DOWN"

    @pytest.mark.asyncio
    async def test_exit_skips_unknown_asset(self) -> None:
        from quantie.pnl import PnLTracker
        broker = PaperBroker()
        executor = OrderExecutor(broker)
        tracker = PnLTracker(1000.0)
        tracker.record_fill("oid3", "OTHER", fill_price=0.50, filled_usd=50.0)
        fills = tracker.open_fills()
        results = await executor.exit_positions(
            open_fills=fills,
            up_asset_id="UP",
            down_asset_id="DOWN",
            up_bid=0.60,
            down_bid=0.40,
        )
        assert results == []
