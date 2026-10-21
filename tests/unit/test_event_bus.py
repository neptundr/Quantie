"""Unit tests for EventBus publish/consume behaviour."""
import asyncio

import pytest

from quantie.data.event_bus import BinanceTrade, ChainlinkPrice, EventBus


@pytest.mark.asyncio
async def test_publish_and_consume_single_event():
    bus = EventBus()
    event = BinanceTrade(ts_ms=1000, trade_ts_ms=999, price=60000.0, qty=0.1, is_buyer_maker=False)
    await bus.publish(event)
    received = await bus.consume()
    assert received == event


@pytest.mark.asyncio
async def test_queue_size_increments_on_publish():
    bus = EventBus()
    for i in range(5):
        await bus.publish(BinanceTrade(ts_ms=i, trade_ts_ms=i, price=1.0, qty=1.0, is_buyer_maker=True))
    assert bus.qsize() == 5


@pytest.mark.asyncio
async def test_fifo_ordering():
    bus = EventBus()
    prices = [60_000.0, 60_001.0, 59_999.0]
    for p in prices:
        await bus.publish(BinanceTrade(ts_ms=0, trade_ts_ms=0, price=p, qty=1.0, is_buyer_maker=False))
    received = [await bus.consume() for _ in prices]
    assert [e.price for e in received] == prices


@pytest.mark.asyncio
async def test_publish_drops_oldest_when_full():
    bus = EventBus(maxsize=3)
    for i in range(5):
        await bus.publish(BinanceTrade(ts_ms=i, trade_ts_ms=i, price=float(i), qty=1.0, is_buyer_maker=False))
    assert bus.qsize() == 3


@pytest.mark.asyncio
async def test_mixed_event_types_preserve_order():
    bus = EventBus()
    trade = BinanceTrade(ts_ms=1, trade_ts_ms=1, price=1.0, qty=1.0, is_buyer_maker=False)
    price_event = ChainlinkPrice(payload_ts_ms=2, received_ts_ms=3, price=60500.0)
    await bus.publish(trade)
    await bus.publish(price_event)
    assert await bus.consume() == trade
    assert await bus.consume() == price_event
