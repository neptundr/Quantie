"""
P&L and fill tracker — wired into engine for SafetySnapshot and metrics.

Binary token P&L model:
  - Pay cost_usd to acquire position at fill_price
  - shares = cost_usd / fill_price
  - On resolution: payout = shares * 1.0 if won else 0.0
  - P&L = payout - cost_usd

Fill lifecycle:
  open → (window ends) → pending_resolution → (Gamma confirms) → resolved

record_resolution() is called after Gamma confirms outcome.
open_fills() excludes pending_resolution fills so the new window
does not see previous-window positions as active exposure.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FillRecord:
    order_id: str
    asset_id: str
    fill_price: float    # cost per share ∈ (0, 1)
    filled_usd: float    # notional paid
    shares: float        # filled_usd / fill_price
    resolved: bool = False
    pending_resolution: bool = False   # window ended; awaiting Gamma confirmation
    won: bool = False
    pnl_usd: float = 0.0


class PnLTracker:
    """Tracks fills and computes realized P&L on resolution."""

    def __init__(self, starting_bankroll: float) -> None:
        self._starting_bankroll = starting_bankroll
        self._fills: dict[str, FillRecord] = {}
        self._realized_pnl: float = 0.0
        self._orders_placed: int = 0
        self._orders_filled: int = 0
        self._orders_cancelled: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def starting_bankroll(self) -> float:
        return self._starting_bankroll

    @property
    def orders_placed(self) -> int:
        return self._orders_placed

    @property
    def orders_filled(self) -> int:
        return self._orders_filled

    @property
    def orders_cancelled(self) -> int:
        return self._orders_cancelled

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record_placed(self) -> None:
        self._orders_placed += 1

    def record_cancelled(self) -> None:
        self._orders_cancelled += 1

    def record_fill(
        self,
        order_id: str,
        asset_id: str,
        fill_price: float,
        filled_usd: float,
        fee_usd: float = 0.0,
    ) -> None:
        """Record a confirmed fill. P&L realized only after record_resolution().

        fee_usd is added to cost so that resolution PnL = payout - (notional + fee).
        Shares are based on notional only (fee doesn't change shares received).
        """
        if fill_price <= 0:
            return
        shares = filled_usd / fill_price
        cost_usd = filled_usd + fee_usd
        self._fills[order_id] = FillRecord(
            order_id=order_id,
            asset_id=asset_id,
            fill_price=fill_price,
            filled_usd=cost_usd,
            shares=shares,
        )
        self._orders_filled += 1

    def mark_pending_resolution(self) -> int:
        """Mark all open fills as pending_resolution at window end.

        Call this in _on_window_completed BEFORE the new window's token IDs
        are set. Pending fills are excluded from open_fills() so the new
        window starts with clean exposure accounting.

        Returns count of fills marked.
        """
        count = 0
        for record in self._fills.values():
            if not record.resolved and not record.pending_resolution:
                record.pending_resolution = True
                count += 1
        return count

    def record_early_exit(
        self,
        order_id: str,
        proceeds_usd: float,
        fee_usd: float = 0.0,
    ) -> float:
        """Record early position exit via sell order.

        Args:
            order_id:     original buy order ID
            proceeds_usd: gross USD received from the sell (before fee)
            fee_usd:      sell-side taker fee deducted from proceeds

        Returns:
            Realized PnL delta (net_proceeds - cost). Negative = loss on exit.
        """
        record = self._fills.get(order_id)
        if record is None or record.resolved or record.pending_resolution:
            return 0.0
        net_proceeds = proceeds_usd - fee_usd
        pnl = net_proceeds - record.filled_usd
        record.pnl_usd = pnl
        record.resolved = True
        record.won = pnl > 0
        self._realized_pnl += pnl
        return pnl

    def record_resolution(self, asset_id: str, won: bool) -> float:
        """Call when a window resolves. Returns P&L delta for this asset.

        Handles both open and pending_resolution fills for this asset.

        Args:
            asset_id: token ID that resolved (UP or DOWN token)
            won:      True if token paid $1/share, False if $0

        Returns:
            Realized P&L delta in USD for this resolution event.
        """
        delta = 0.0
        for record in self._fills.values():
            if record.asset_id != asset_id or record.resolved:
                continue
            payout = record.shares * 1.0 if won else 0.0
            record.pnl_usd = payout - record.filled_usd
            record.resolved = True
            record.won = won
            delta += record.pnl_usd

        self._realized_pnl += delta
        return delta

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def open_fills(self) -> list[FillRecord]:
        """Fills active in the current window (not resolved, not pending resolution)."""
        return [r for r in self._fills.values() if not r.resolved and not r.pending_resolution]

    def pending_fills(self) -> list[FillRecord]:
        """Fills from completed windows awaiting Gamma resolution."""
        return [r for r in self._fills.values() if r.pending_resolution and not r.resolved]

    def total_open_exposure(self) -> float:
        """Sum of filled_usd for current-window open fills (excludes pending)."""
        return sum(r.filled_usd for r in self.open_fills())

    def window_summary(self) -> dict:
        """Dict suitable for structlog at window close."""
        return {
            "realized_pnl_usd": round(self._realized_pnl, 4),
            "orders_placed": self._orders_placed,
            "orders_filled": self._orders_filled,
            "orders_cancelled": self._orders_cancelled,
            "open_exposure_usd": round(self.total_open_exposure(), 4),
        }
