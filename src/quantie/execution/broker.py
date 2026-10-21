"""
Broker Protocol and shared types.

All broker implementations (live ClobClient, PaperBroker) satisfy this Protocol.

Order flow:
  FOK (fill-or-kill): immediate fill or cancel — used for TAKE_* actions
  GTC (good-till-cancel): resting limit order — used for MAKE_* actions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# Order statuses
STATUS_OPEN = "OPEN"
STATUS_FILLED = "FILLED"
STATUS_CANCELLED = "CANCELLED"
STATUS_REJECTED = "REJECTED"

# Sides
SIDE_BUY = "BUY"
SIDE_SELL = "SELL"


@dataclass(frozen=True)
class OrderResult:
    """Returned by every broker method that touches an order."""

    order_id: str
    asset_id: str
    side: str            # SIDE_BUY or SIDE_SELL
    price: float         # limit price (shares cost)
    size_usd: float      # notional requested
    filled_usd: float    # notional actually filled (0 for open GTC)
    status: str          # STATUS_* constant
    is_paper: bool       # True when simulated


@dataclass
class OpenOrder:
    """Mutable record of a resting GTC order (used by PaperBroker)."""

    order_id: str
    asset_id: str
    price: float
    size_usd: float
    filled_usd: float = field(default=0.0)
    status: str = field(default=STATUS_OPEN)


@runtime_checkable
class Broker(Protocol):
    """Interface that both ClobClient and PaperBroker implement."""

    async def place_fok_buy(
        self,
        asset_id: str,
        price: float,
        size_usd: float,
    ) -> OrderResult:
        """Place a fill-or-kill buy. Returns immediately with FILLED or CANCELLED."""
        ...

    async def place_gtc_buy(
        self,
        asset_id: str,
        price: float,
        size_usd: float,
    ) -> OrderResult:
        """Place a good-till-cancel limit buy. Returns with status OPEN."""
        ...

    async def place_fok_sell(
        self,
        asset_id: str,
        shares: float,
        current_bid: float,
    ) -> OrderResult:
        """Sell shares at current bid (FOK). Returns FILLED or CANCELLED."""
        ...

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel one open order. Returns True if successfully cancelled."""
        ...

    async def cancel_all(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        ...

    async def get_order(self, order_id: str) -> OrderResult | None:
        """Fetch current state of an order. Returns None if not found."""
        ...
