"""
Edge computation — measures mispricing between model P(UP) and Polymarket prices.

edge_buy_up   = p_up - poly_up_ask      (positive → UP token underpriced)
edge_buy_down = poly_up_bid - p_up      (positive → DOWN token underpriced)

Threshold:
  base(0.025) + 0.3 × ci_width + fee_buffer(0.012)
  LATE regime adds +0.01; entry also blocked if ci_width ≥ 0.08.

  ci_multiplier 0.3 prevents threshold explosion: ci_width=0.30 adds only 0.09,
  keeping threshold around 0.127 vs the old 0.495 with multiplier=1.5.
  fee_buffer 0.012 covers ~1.8% taker fee on a typical 0.65 ask.

Signal: BUY_UP | BUY_DOWN | HOLD
"""
from __future__ import annotations

from dataclasses import dataclass

from quantie.policy.regime import Regime

_BASE_THRESHOLD: float = 0.025
_CI_MULTIPLIER: float = 0.3
_FEE_BUFFER: float = 0.012
_LATE_EXTRA: float = 0.01
_LATE_MAX_CI: float = 0.08

SIGNAL_BUY_UP = "BUY_UP"
SIGNAL_BUY_DOWN = "BUY_DOWN"
SIGNAL_HOLD = "HOLD"


@dataclass(frozen=True)
class EdgeResult:
    p_up: float           # model probability of UP outcome
    poly_mid: float       # midpoint of UP token market
    edge_buy_up: float    # p_up - ask; positive → buy UP
    edge_buy_down: float  # bid - p_up; positive → buy DOWN
    threshold: float      # minimum edge required to trade
    signal: str           # SIGNAL_* constant
    ci_width: float       # CI width at time of computation


@dataclass(frozen=True)
class EdgeConfig:
    base_threshold: float = _BASE_THRESHOLD
    ci_multiplier: float = _CI_MULTIPLIER
    fee_buffer: float = _FEE_BUFFER
    late_extra: float = _LATE_EXTRA
    late_max_ci: float = _LATE_MAX_CI


def compute_edge(
    p_up: float,
    poly_up_ask: float,
    poly_up_bid: float,
    ci_width: float,
    regime: Regime,
    base_threshold: float = _BASE_THRESHOLD,
    ci_multiplier: float = _CI_MULTIPLIER,
    fee_buffer: float = _FEE_BUFFER,
) -> EdgeResult:
    return compute_edge_with_config(
        p_up=p_up,
        poly_up_ask=poly_up_ask,
        poly_up_bid=poly_up_bid,
        ci_width=ci_width,
        regime=regime,
        cfg=EdgeConfig(
            base_threshold=base_threshold,
            ci_multiplier=ci_multiplier,
            fee_buffer=fee_buffer,
            late_extra=_LATE_EXTRA,
            late_max_ci=_LATE_MAX_CI,
        ),
    )


def compute_edge_with_config(
    p_up: float,
    poly_up_ask: float,
    poly_up_bid: float,
    ci_width: float,
    regime: Regime,
    cfg: EdgeConfig,
) -> EdgeResult:
    """Compute trading edge and signal for a given model output and book state.

    Args:
        p_up:          model P(UP) ∈ [0.01, 0.99]
        poly_up_ask:   best ask for UP token on Polymarket
        poly_up_bid:   best bid for UP token on Polymarket
        ci_width:      confidence interval width (ci_high - ci_low)
        regime:        current trading regime
        base_threshold: minimum edge before CI/fee adjustment
        ci_multiplier:  scales CI width contribution to threshold
        fee_buffer:    flat fee/slippage allowance

    Returns:
        EdgeResult with signal BUY_UP, BUY_DOWN, or HOLD
    """
    poly_mid = (poly_up_ask + poly_up_bid) / 2.0
    edge_buy_up = p_up - poly_up_ask
    edge_buy_down = poly_up_bid - p_up
    threshold = cfg.base_threshold + cfg.ci_multiplier * ci_width + cfg.fee_buffer

    if regime == Regime.LATE:
        threshold += cfg.late_extra
        if ci_width >= cfg.late_max_ci:
            return EdgeResult(
                p_up=p_up,
                poly_mid=poly_mid,
                edge_buy_up=edge_buy_up,
                edge_buy_down=edge_buy_down,
                threshold=threshold,
                signal=SIGNAL_HOLD,
                ci_width=ci_width,
            )

    if edge_buy_up > threshold:
        signal = SIGNAL_BUY_UP
    elif edge_buy_down > threshold:
        signal = SIGNAL_BUY_DOWN
    else:
        signal = SIGNAL_HOLD

    return EdgeResult(
        p_up=p_up,
        poly_mid=poly_mid,
        edge_buy_up=edge_buy_up,
        edge_buy_down=edge_buy_down,
        threshold=threshold,
        signal=signal,
        ci_width=ci_width,
    )
