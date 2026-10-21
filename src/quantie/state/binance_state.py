"""
BinanceState — rolling buffer of trades and latest L2 book snapshot.

Consumed by the feature pipeline (Phase 3) to compute momentum,
order-flow imbalance, realized vol, and depth imbalance.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from quantie.data.event_bus import BinanceDepth, BinanceTrade


class BinanceState:
    def __init__(self, trade_buffer_len: int = 10_000) -> None:
        self.trades: deque[BinanceTrade] = deque(maxlen=trade_buffer_len)
        self.depth: Optional[BinanceDepth] = None

    def on_trade(self, event: BinanceTrade) -> None:
        self.trades.append(event)

    def on_depth(self, event: BinanceDepth) -> None:
        self.depth = event

    @property
    def best_bid(self) -> Optional[float]:
        if self.depth and self.depth.bids:
            return self.depth.bids[0][0]
        return None

    @property
    def best_ask(self) -> Optional[float]:
        if self.depth and self.depth.asks:
            return self.depth.asks[0][0]
        return None

    @property
    def mid_price(self) -> Optional[float]:
        bid = self.best_bid
        ask = self.best_ask
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return None

    @property
    def last_trade_price(self) -> Optional[float]:
        if self.trades:
            return self.trades[-1].price
        return None
