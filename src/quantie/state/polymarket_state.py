"""
PolymarketState — latest UP/DOWN order books, best prices, and our position.

The policy layer reads up_book/down_book for edge calculation and
position for sizing and risk checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from quantie.data.event_bus import PolyBookUpdate, PolyPriceChange, PolyTrade


@dataclass
class BookSide:
    """Best bid/ask and shallow depth for a single token (UP or DOWN)."""
    best_bid: float = 0.0
    best_ask: float = 1.0
    bid_depth_usd: float = 0.0  # notional at top-3 bid levels
    ask_depth_usd: float = 0.0  # notional at top-3 ask levels

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid


@dataclass
class Position:
    """Our current open position in a single token."""
    token_id: str = ""
    side: str = ""           # "UP" or "DOWN"
    qty: float = 0.0         # shares held
    avg_cost: float = 0.0    # average entry price per share
    realised_pnl: float = 0.0

    @property
    def notional_usd(self) -> float:
        return self.qty * self.avg_cost

    def is_flat(self) -> bool:
        return self.qty == 0.0


class PolymarketState:
    """Mutable state updated by the main consumer loop."""

    def __init__(self) -> None:
        self._best: dict[str, BookSide] = {}
        self._books: dict[str, PolyBookUpdate] = {}
        self.up_token_id: Optional[str] = None
        self.down_token_id: Optional[str] = None
        self.position: Position = Position()
        self._recent_trades: list[PolyTrade] = []

    def set_tokens(self, up_token_id: str, down_token_id: str) -> None:
        """Called by WindowManager on each window rollover."""
        self.up_token_id = up_token_id
        self.down_token_id = down_token_id
        self._books.clear()
        self._best.clear()
        self.position = Position()
        self._recent_trades.clear()

    def on_book(self, event: PolyBookUpdate) -> None:
        self._books[event.asset_id] = event
        side = self._best.setdefault(event.asset_id, BookSide())
        if event.bids:
            side.best_bid = event.bids[0][0]
            side.bid_depth_usd = sum(p * q for p, q in event.bids[:3])
        if event.asks:
            side.best_ask = event.asks[0][0]
            side.ask_depth_usd = sum(p * q for p, q in event.asks[:3])

    def on_price_change(self, event: PolyPriceChange) -> None:
        side = self._best.setdefault(event.asset_id, BookSide())
        side.best_bid = event.best_bid
        side.best_ask = event.best_ask

    def on_trade(self, event: PolyTrade) -> None:
        self._recent_trades.append(event)
        if len(self._recent_trades) > 200:
            self._recent_trades = self._recent_trades[-200:]

    @property
    def up_book(self) -> Optional[BookSide]:
        if self.up_token_id:
            return self._best.get(self.up_token_id)
        return None

    @property
    def down_book(self) -> Optional[BookSide]:
        if self.down_token_id:
            return self._best.get(self.down_token_id)
        return None
