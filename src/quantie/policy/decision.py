"""
Policy decision layer.

Combines regime, edge, and sizing into a single PolicyDecision.

Actions:
  HOLD              — no trade signal or blocked
  TAKE_UP           — market-take UP token (MID/LATE)
  TAKE_DOWN         — market-take DOWN token (MID/LATE)
  MAKE_UP           — passive GTC limit UP token (EARLY only)
  MAKE_DOWN         — passive GTC limit DOWN token (EARLY only)
  EXIT              — exit open position (TERMINAL or forced)
  HOLD_TO_RESOLUTION — keep position open; let market resolve

Resolution hold: hold if p_win > 0.70 at TERMINAL entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from quantie.policy.edge import EdgeConfig, EdgeResult, compute_edge_with_config
from quantie.policy.regime import (
    EARLY_PASSIVE_OFFSET,
    Regime,
    RegimeConfig,
    classify_regime_with_config,
    is_entry_allowed,
    is_passive_only,
)
from quantie.policy.sizing import SizingConfig, kelly_size_with_config

_HOLD_THRESHOLD: float = 0.70   # p_win above which we hold to resolution


class Action(str, Enum):
    HOLD = "HOLD"
    TAKE_UP = "TAKE_UP"
    TAKE_DOWN = "TAKE_DOWN"
    MAKE_UP = "MAKE_UP"
    MAKE_DOWN = "MAKE_DOWN"
    EXIT = "EXIT"
    HOLD_TO_RESOLUTION = "HOLD_TO_RESOLUTION"


@dataclass(frozen=True)
class PolicyDecision:
    action: Action
    size_usd: float            # 0.0 when action is HOLD/EXIT/HOLD_TO_RESOLUTION
    limit_price: float | None  # set for MAKE_* actions, None otherwise
    edge: EdgeResult
    regime: Regime
    reason: str                # human-readable rationale for logging


def make_decision(
    p_up: float,
    poly_up_ask: float,
    poly_up_bid: float,
    ci_width: float,
    tau_sec: float,
    bankroll: float,
    fair_value: float | None = None,
    early_passive_offset: float = EARLY_PASSIVE_OFFSET,
    edge_cfg: EdgeConfig | None = None,
    sizing_cfg: SizingConfig | None = None,
    regime_cfg: RegimeConfig | None = None,
) -> PolicyDecision:
    """Produce a trading decision from model output and market state.

    Args:
        p_up:               model P(UP) ∈ [0.01, 0.99]
        poly_up_ask:        best ask for UP token
        poly_up_bid:        best bid for UP token
        ci_width:           confidence interval width
        tau_sec:            seconds remaining in window
        bankroll:           available bankroll in USD
        fair_value:         model fair-value price for passive orders (EARLY)
        early_passive_offset: GTC offset below fair_value (default 1.5c)

    Returns:
        PolicyDecision with action, size, and rationale.
    """
    edge_cfg = edge_cfg or EdgeConfig()
    sizing_cfg = sizing_cfg or SizingConfig()
    regime_cfg = regime_cfg or RegimeConfig()
    regime = classify_regime_with_config(tau_sec, regime_cfg)
    edge = compute_edge_with_config(
        p_up=p_up,
        poly_up_ask=poly_up_ask,
        poly_up_bid=poly_up_bid,
        ci_width=ci_width,
        regime=regime,
        cfg=edge_cfg,
    )

    if regime == Regime.TERMINAL:
        return PolicyDecision(
            action=Action.EXIT,
            size_usd=0.0,
            limit_price=None,
            edge=edge,
            regime=regime,
            reason="terminal zone — exit only",
        )

    if not is_entry_allowed(regime):
        return PolicyDecision(
            action=Action.HOLD,
            size_usd=0.0,
            limit_price=None,
            edge=edge,
            regime=regime,
            reason=f"entry not allowed in {regime}",
        )

    if edge.signal == "HOLD":
        return PolicyDecision(
            action=Action.HOLD,
            size_usd=0.0,
            limit_price=None,
            edge=edge,
            regime=regime,
            reason="edge below threshold",
        )

    if is_passive_only(regime):
        return _passive_decision(edge, regime, bankroll, p_up, fair_value, early_passive_offset, sizing_cfg)

    return _active_decision(edge, regime, bankroll, p_up, poly_up_ask, poly_up_bid, sizing_cfg)


def should_hold_to_resolution(p_win: float, threshold: float = _HOLD_THRESHOLD) -> bool:
    """True when position is strong enough to hold rather than exit early."""
    return p_win >= threshold


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _passive_decision(
    edge: EdgeResult,
    regime: Regime,
    bankroll: float,
    p_up: float,
    fair_value: float | None,
    offset: float,
    sizing_cfg: SizingConfig,
) -> PolicyDecision:
    if edge.signal == "BUY_UP":
        fv = fair_value if fair_value is not None else p_up
        limit = round(fv - offset, 4)
        size = kelly_size_with_config(p_win=p_up, price=limit, bankroll=bankroll, cfg=sizing_cfg)
        if size <= 0.0:
            return _hold_below_min(edge, regime, "UP", limit, bankroll)
        return PolicyDecision(
            action=Action.MAKE_UP,
            size_usd=size,
            limit_price=limit,
            edge=edge,
            regime=regime,
            reason=f"passive UP GTC at {limit}",
        )
    if edge.signal == "BUY_DOWN":
        fv = fair_value if fair_value is not None else (1.0 - p_up)
        limit = round(fv - offset, 4)
        size = kelly_size_with_config(p_win=1.0 - p_up, price=limit, bankroll=bankroll, cfg=sizing_cfg)
        if size <= 0.0:
            return _hold_below_min(edge, regime, "DOWN", limit, bankroll)
        return PolicyDecision(
            action=Action.MAKE_DOWN,
            size_usd=size,
            limit_price=limit,
            edge=edge,
            regime=regime,
            reason=f"passive DOWN GTC at {limit}",
        )
    return PolicyDecision(
        action=Action.HOLD,
        size_usd=0.0,
        limit_price=None,
        edge=edge,
        regime=regime,
        reason="no signal in early regime",
    )


def _active_decision(
    edge: EdgeResult,
    regime: Regime,
    bankroll: float,
    p_up: float,
    poly_up_ask: float,
    poly_up_bid: float,
    sizing_cfg: SizingConfig,
) -> PolicyDecision:
    if edge.signal == "BUY_UP":
        size = kelly_size_with_config(
            p_win=p_up, price=poly_up_ask, bankroll=bankroll, cfg=sizing_cfg
        )
        if size <= 0.0:
            return _hold_below_min(edge, regime, "UP", poly_up_ask, bankroll)
        return PolicyDecision(
            action=Action.TAKE_UP,
            size_usd=size,
            limit_price=None,
            edge=edge,
            regime=regime,
            reason=f"take UP edge={edge.edge_buy_up:.4f} thr={edge.threshold:.4f}",
        )
    if edge.signal == "BUY_DOWN":
        down_price = 1.0 - poly_up_bid
        size = kelly_size_with_config(
            p_win=1.0 - p_up, price=down_price, bankroll=bankroll, cfg=sizing_cfg
        )
        if size <= 0.0:
            return _hold_below_min(edge, regime, "DOWN", down_price, bankroll)
        return PolicyDecision(
            action=Action.TAKE_DOWN,
            size_usd=size,
            limit_price=None,
            edge=edge,
            regime=regime,
            reason=f"take DOWN edge={edge.edge_buy_down:.4f} thr={edge.threshold:.4f}",
        )
    return PolicyDecision(
        action=Action.HOLD,
        size_usd=0.0,
        limit_price=None,
        edge=edge,
        regime=regime,
        reason="no signal",
    )


def _hold_below_min(
    edge: EdgeResult,
    regime: Regime,
    side: str,
    price: float,
    bankroll: float,
) -> PolicyDecision:
    """Edge present but Kelly sizing returned 0 (below min order or capped).

    Returning HOLD here is critical: previously TAKE_*/MAKE_* could be emitted
    with size_usd=0, which the broker then submitted as a zero-size order.
    """
    return PolicyDecision(
        action=Action.HOLD,
        size_usd=0.0,
        limit_price=None,
        edge=edge,
        regime=regime,
        reason=(
            f"{side} signal but size below min order "
            f"(price={price:.4f} bankroll={bankroll:.2f})"
        ),
    )
