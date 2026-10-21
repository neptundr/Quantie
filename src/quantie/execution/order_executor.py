"""
OrderExecutor — routes PolicyDecision to broker after safety checks.

Responsibilities:
  1. Run pre_trade_check — block if any safety gate fails
  2. Map Action → broker call (FOK for TAKE, GTC for MAKE, cancel_all for EXIT)
  3. Track window_exposure_usd — reset on new window
  4. Log every decision outcome via structlog

Thread safety: single asyncio task assumed (no locking).
"""
from __future__ import annotations

import structlog

from quantie.execution.broker import Broker, OrderResult, STATUS_FILLED
from quantie.policy.decision import Action, PolicyDecision
from quantie.safety.checks import SafetySnapshot, pre_trade_check

log = structlog.get_logger(__name__)


class OrderExecutor:
    """Routes decisions to a Broker implementation, enforcing safety gates."""

    def __init__(self, broker: Broker) -> None:
        self._broker = broker
        self._window_exposure_usd: float = 0.0

    @property
    def window_exposure_usd(self) -> float:
        return self._window_exposure_usd

    def reset_window(self) -> None:
        """Call on every window rollover to clear per-window exposure."""
        self._window_exposure_usd = 0.0

    async def execute(
        self,
        decision: PolicyDecision,
        safety: SafetySnapshot,
        up_ask: float,
        up_bid: float,
        up_asset_id: str = "",
        down_asset_id: str = "",
    ) -> OrderResult | None:
        """Execute a PolicyDecision.

        Args:
            decision:       output of make_decision()
            safety:         pre-trade safety snapshot
            up_ask:         best ask for UP token
            up_bid:         best bid for UP token
            up_asset_id:    Polymarket token ID for UP side
            down_asset_id:  Polymarket token ID for DOWN side

        Returns:
            OrderResult if an order was submitted, None if blocked/HOLD.
        """
        allowed, reason = pre_trade_check(decision.action, safety)
        if not allowed:
            log.info(
                "order.blocked",
                action=decision.action,
                reason=reason,
                regime=str(decision.regime),
            )
            return None

        result = await self._dispatch(decision, up_ask, up_bid, up_asset_id, down_asset_id)

        if result is not None and result.status == STATUS_FILLED:
            self._window_exposure_usd += result.filled_usd

        log.debug(
            "order.submitted",
            action=str(decision.action),
            asset_id=result.asset_id if result else None,
            status=result.status if result else None,
            filled_usd=result.filled_usd if result else 0.0,
            window_exposure=self._window_exposure_usd,
            regime=str(decision.regime),
            reason=decision.reason,
        )
        return result

    async def exit_positions(
        self,
        open_fills: list,
        up_asset_id: str,
        down_asset_id: str,
        up_bid: float,
        down_bid: float,
    ) -> list:
        """FOK-sell all open filled positions at current market bids.

        Args:
            open_fills:    list of FillRecord from PnLTracker.open_fills()
            up_asset_id:   token ID for UP side
            down_asset_id: token ID for DOWN side
            up_bid:        current best bid for UP token
            down_bid:      current best bid for DOWN token (= 1 - up_ask)

        Returns:
            List of OrderResult, one per fill attempted.
        """
        results = []
        for fill in open_fills:
            if fill.asset_id == up_asset_id:
                bid = up_bid
            elif fill.asset_id == down_asset_id:
                bid = down_bid
            else:
                continue
            result = await self._broker.place_fok_sell(
                asset_id=fill.asset_id,
                shares=fill.shares,
                current_bid=bid,
            )
            log.info(
                "order.early_exit",
                asset_id=fill.asset_id,
                shares=fill.shares,
                bid=bid,
                status=result.status,
                proceeds_usd=result.filled_usd,
            )
            results.append(result)
        return results

    async def cancel_all_open(self) -> int:
        """Cancel all resting GTC orders — call at late/terminal transition."""
        count = await self._broker.cancel_all()
        log.info("orders.cancelled_all", count=count)
        return count

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    async def _dispatch(
        self,
        decision: PolicyDecision,
        up_ask: float,
        up_bid: float,
        up_asset_id: str,
        down_asset_id: str,
    ) -> OrderResult | None:
        action = decision.action
        size = decision.size_usd

        # Defensive guard: decision.py should never emit TAKE/MAKE with size=0,
        # but catch it here too — never submit a zero-size order to any broker.
        if action in (Action.TAKE_UP, Action.TAKE_DOWN, Action.MAKE_UP, Action.MAKE_DOWN):
            if size <= 0.0:
                log.warning("order.zero_size_blocked", action=str(action), size_usd=size)
                return None

        if action == Action.TAKE_UP:
            return await self._broker.place_fok_buy(
                asset_id=up_asset_id,
                price=up_ask,
                size_usd=size,
                current_ask=up_ask,
            )

        if action == Action.TAKE_DOWN:
            down_ask = 1.0 - up_bid
            return await self._broker.place_fok_buy(
                asset_id=down_asset_id,
                price=down_ask,
                size_usd=size,
                current_ask=down_ask,
            )

        if action == Action.MAKE_UP:
            if self._has_open_order_for(up_asset_id):
                return None  # already resting — don't stack GTC orders
            limit = decision.limit_price if decision.limit_price is not None else up_bid
            return await self._broker.place_gtc_buy(
                asset_id=up_asset_id,
                price=limit,
                size_usd=size,
            )

        if action == Action.MAKE_DOWN:
            if self._has_open_order_for(down_asset_id):
                return None  # already resting — don't stack GTC orders
            limit = decision.limit_price if decision.limit_price is not None else (1.0 - up_ask)
            return await self._broker.place_gtc_buy(
                asset_id=down_asset_id,
                price=limit,
                size_usd=size,
            )

        if action == Action.EXIT:
            await self._broker.cancel_all()
            return None

        # HOLD / HOLD_TO_RESOLUTION — no order
        return None

    def _has_open_order_for(self, asset_id: str) -> bool:
        """True if the broker already has a resting GTC order for this asset."""
        open_orders = getattr(self._broker, "open_orders", {})
        return any(o.asset_id == asset_id for o in open_orders.values())
