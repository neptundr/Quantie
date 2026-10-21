"""
Typed event definitions and EventBus for the Quantie data pipeline.

Events flow: WebSocket clients -> EventBus.queue -> state consumers.
All dataclasses are frozen (immutable) so they are safe to pass across
asyncio tasks without defensive copying.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Union


# ---------------------------------------------------------------------------
# Binance events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BinanceTrade:
    """Single aggTrade from btcusdt@aggTrade stream."""
    ts_ms: int        # event time (ms)
    trade_ts_ms: int  # trade execution time (ms)
    price: float
    qty: float
    is_buyer_maker: bool  # True = sell was the aggressor


@dataclass(frozen=True)
class BinanceDepth:
    """L2 snapshot from btcusdt@depth20@100ms stream (top-20 levels)."""
    ts_ms: int
    bids: tuple[tuple[float, float], ...]  # ((price, qty), ...) sorted desc by price
    asks: tuple[tuple[float, float], ...]  # ((price, qty), ...) sorted asc by price


# ---------------------------------------------------------------------------
# Chainlink / RTDS events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainlinkPrice:
    """BTC/USD price from Polymarket RTDS crypto_prices_chainlink topic.

    payload_ts_ms = oracle publication time from the RTDS payload.
    received_ts_ms = our wall-clock receipt time (for stale-feed detection).
    """
    payload_ts_ms: int
    received_ts_ms: int
    price: float


# ---------------------------------------------------------------------------
# Polymarket CLOB market-channel events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolyBookUpdate:
    """Full order book snapshot (event_type='book')."""
    ts_ms: int
    asset_id: str
    market: str    # condition ID (hex)
    bids: tuple[tuple[float, float], ...]  # ((price, qty), ...) sorted desc
    asks: tuple[tuple[float, float], ...]  # ((price, qty), ...) sorted asc
    book_hash: str


@dataclass(frozen=True)
class PolyPriceChange:
    """Best bid/ask update (event_type='price_change')."""
    ts_ms: int
    market: str
    asset_id: str
    price: float
    size: float
    side: str       # "BUY" or "SELL"
    best_bid: float
    best_ask: float


@dataclass(frozen=True)
class PolyTrade:
    """Executed trade (event_type='last_trade_price')."""
    ts_ms: int
    asset_id: str
    market: str
    price: float
    size: float
    side: str  # "BUY" or "SELL"


# ---------------------------------------------------------------------------
# Polymarket CLOB user-channel events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolyOurOrderUpdate:
    """Our own order lifecycle event from the authenticated user channel."""
    ts_ms: int
    order_id: str
    asset_id: str
    market: str
    price: float
    original_size: float
    size_matched: float
    side: str     # "BUY" or "SELL"
    outcome: str  # "YES" / "NO"


# ---------------------------------------------------------------------------
# Union type for the bus
# ---------------------------------------------------------------------------

Event = Union[
    BinanceTrade,
    BinanceDepth,
    ChainlinkPrice,
    PolyBookUpdate,
    PolyPriceChange,
    PolyTrade,
    PolyOurOrderUpdate,
]


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """Single asyncio.Queue shared by all producers and the main consumer loop.

    Producers: await bus.publish(event)
    Consumer:  event = await bus.consume()
    """

    def __init__(self, maxsize: int = 50_000) -> None:
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)

    async def publish(self, event: Event) -> None:
        """Enqueue an event. If the queue is full, drop the oldest to make room."""
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.queue.put_nowait(event)

    async def consume(self) -> Event:
        """Block until an event is available and return it."""
        return await self.queue.get()

    def qsize(self) -> int:
        return self.queue.qsize()
