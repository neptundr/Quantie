"""
Paper trading broker — in-memory fill simulator.

FOK fills:
  - Fill if our limit price >= current ask (book can fill us)
  - Fill at ask price (not our limit price) — conservative sim
  - Cancel immediately if price < ask

GTC fills:
  - Order rests in self._open; call tick_book() on each book update
    to simulate passive fills when best_ask <= our limit price

Slippage model:
  FOK fills at ask * (1 + 2bps) to avoid over-estimating paper PnL.
"""
from __future__ import annotations

import uuid
from typing import Dict

from quantie.execution.broker import (
    SIDE_BUY,
    SIDE_SELL,
    STATUS_CANCELLED,
    STATUS_FILLED,
    STATUS_OPEN,
    OpenOrder,
    OrderResult,
)

_SLIPPAGE_BPS: float = 2.0   # basis points added to ask for FOK fills


class PaperBroker:
    """Simulates order execution without touching real money."""

    def __init__(self) -> None:
        self._open: Dict[str, OpenOrder] = {}

    # ------------------------------------------------------------------
    # Broker interface
    # ------------------------------------------------------------------

    async def place_fok_buy(
        self,
        asset_id: str,
        price: float,
        size_usd: float,
        current_ask: float | None = None,
    ) -> OrderResult:
        """Simulate FOK fill.

        Args:
            asset_id:    Polymarket token ID
            price:       our limit price (max we'll pay)
            size_usd:    notional in USD
            current_ask: live ask from book; if None, fills at price
        """
        order_id = _new_id()
        ask = current_ask if current_ask is not None else price
        fill_price = ask * (1 + _SLIPPAGE_BPS / 10_000)

        if price < ask:
            return OrderResult(
                order_id=order_id,
                asset_id=asset_id,
                side=SIDE_BUY,
                price=price,
                size_usd=size_usd,
                filled_usd=0.0,
                status=STATUS_CANCELLED,
                is_paper=True,
            )

        return OrderResult(
            order_id=order_id,
            asset_id=asset_id,
            side=SIDE_BUY,
            price=fill_price,
            size_usd=size_usd,
            filled_usd=size_usd,
            status=STATUS_FILLED,
            is_paper=True,
        )

    async def place_gtc_buy(
        self,
        asset_id: str,
        price: float,
        size_usd: float,
    ) -> OrderResult:
        """Post a resting GTC limit order. Fills via tick_book()."""
        order_id = _new_id()
        self._open[order_id] = OpenOrder(
            order_id=order_id,
            asset_id=asset_id,
            price=price,
            size_usd=size_usd,
        )
        return OrderResult(
            order_id=order_id,
            asset_id=asset_id,
            side=SIDE_BUY,
            price=price,
            size_usd=size_usd,
            filled_usd=0.0,
            status=STATUS_OPEN,
            is_paper=True,
        )

    async def place_fok_sell(
        self,
        asset_id: str,
        shares: float,
        current_bid: float,
    ) -> OrderResult:
        """Simulate FOK sell at current bid minus slippage."""
        order_id = _new_id()
        if current_bid <= 0.0 or shares <= 0.0:
            return OrderResult(
                order_id=order_id,
                asset_id=asset_id,
                side=SIDE_SELL,
                price=current_bid,
                size_usd=0.0,
                filled_usd=0.0,
                status=STATUS_CANCELLED,
                is_paper=True,
            )
        sell_price = current_bid * (1 - _SLIPPAGE_BPS / 10_000)
        proceeds = shares * sell_price
        return OrderResult(
            order_id=order_id,
            asset_id=asset_id,
            side=SIDE_SELL,
            price=sell_price,
            size_usd=proceeds,
            filled_usd=proceeds,
            status=STATUS_FILLED,
            is_paper=True,
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel one resting order. Returns True if it existed and was open."""
        order = self._open.pop(order_id, None)
        if order is None:
            return False
        order.status = STATUS_CANCELLED
        return True

    async def cancel_all(self) -> int:
        """Cancel all open GTC orders. Returns count cancelled."""
        ids = list(self._open.keys())
        for oid in ids:
            await self.cancel_order(oid)
        return len(ids)

    async def get_order(self, order_id: str) -> OrderResult | None:
        """Return current state of a resting order, or None if not found."""
        order = self._open.get(order_id)
        if order is None:
            return None
        return OrderResult(
            order_id=order.order_id,
            asset_id=order.asset_id,
            side=SIDE_BUY,
            price=order.price,
            size_usd=order.size_usd,
            filled_usd=order.filled_usd,
            status=order.status,
            is_paper=True,
        )

    # ------------------------------------------------------------------
    # Paper-only helpers
    # ------------------------------------------------------------------

    def tick_book(self, asset_id: str, best_ask: float) -> list[OrderResult]:
        """Simulate passive fills when best_ask drops to or below our limit price.

        A resting BUY limit fills when the ask is <= our limit price —
        someone is willing to sell at what we're offering to pay.

        Call on every PolyBookUpdate for the relevant asset.
        Returns list of OrderResults for any orders that filled.
        """
        filled: list[OrderResult] = []
        for oid, order in list(self._open.items()):
            if order.asset_id != asset_id:
                continue
            if best_ask <= order.price:
                order.filled_usd = order.size_usd
                order.status = STATUS_FILLED
                del self._open[oid]
                filled.append(OrderResult(
                    order_id=order.order_id,
                    asset_id=order.asset_id,
                    side=SIDE_BUY,
                    price=order.price,
                    size_usd=order.size_usd,
                    filled_usd=order.size_usd,
                    status=STATUS_FILLED,
                    is_paper=True,
                ))
        return filled

    @property
    def open_orders(self) -> dict[str, OpenOrder]:
        """Snapshot of resting orders (copy)."""
        return dict(self._open)


def _new_id() -> str:
    return f"paper-{uuid.uuid4().hex[:12]}"
