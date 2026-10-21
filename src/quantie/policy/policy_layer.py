"""Execution policy layer: rule baseline + contextual bandit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from quantie.policy.bandit import LinUCBBandit
from quantie.policy.decision import Action, PolicyDecision, make_decision
from quantie.policy.edge import EdgeConfig
from quantie.policy.regime import RegimeConfig, classify_regime_with_config
from quantie.policy.sizing import SizingConfig
from quantie.safety.circuit_breakers import CircuitBreaker


@dataclass(frozen=True)
class PolicyContext:
    now_ms: int
    tau_sec: float
    p_up: float
    ci_width: float
    gbm_p_up: float
    ml_residual: float
    up_ask: float
    up_bid: float
    up_mid: float
    poly_spread: float
    poly_extremity: float
    poly_imbalance: float
    ofi_5s: float
    momentum_5s: float
    depth_imbalance: float
    binance_age_ms: float
    chainlink_age_ms: float
    bankroll: float
    is_live: bool
    stale_chainlink_ms: float
    stale_binance_ms: float
    extreme_mkt_lo: float
    extreme_mkt_hi: float
    circuit_breaker: CircuitBreaker


@dataclass(frozen=True)
class BanditFeedback:
    action: Action
    reward: float
    context_vec: np.ndarray


class ExecutionPolicy:
    def __init__(
        self,
        *,
        edge_cfg: EdgeConfig,
        sizing_cfg: SizingConfig,
        regime_cfg: RegimeConfig,
        bandit: LinUCBBandit,
        bandit_state_path: Path,
    ) -> None:
        self._edge_cfg = edge_cfg
        self._sizing_cfg = sizing_cfg
        self._regime_cfg = regime_cfg
        self._bandit = bandit
        self._bandit_state_path = Path(bandit_state_path)
        if self._bandit_state_path.exists():
            self._bandit.load(self._bandit_state_path)

    def choose(self, ctx: PolicyContext) -> tuple[PolicyDecision, np.ndarray]:
        rule = self._rule_decision(ctx)
        if rule.action == Action.EXIT:
            return rule, self.vectorize(ctx)
        force_reason = self._force_hold_reason(ctx)
        if force_reason:
            hold = PolicyDecision(
                action=Action.HOLD,
                size_usd=0.0,
                limit_price=None,
                edge=rule.edge,
                regime=rule.regime,
                reason=force_reason,
            )
            return hold, self.vectorize(ctx)

        x = self.vectorize(ctx)
        if not self._bandit.is_warm():
            return rule, x
        chosen = self._bandit.select(x, is_live=ctx.is_live)
        if chosen == Action.HOLD:
            return PolicyDecision(
                action=Action.HOLD,
                size_usd=0.0,
                limit_price=None,
                edge=rule.edge,
                regime=rule.regime,
                reason="bandit HOLD",
            ), x
        if chosen == rule.action:
            return rule, x
        # keep reasoning + sizing semantics from rule baseline by side
        if chosen in (Action.TAKE_UP, Action.MAKE_UP):
            alt = self._rule_decision(ctx, force_side="UP")
        elif chosen in (Action.TAKE_DOWN, Action.MAKE_DOWN):
            alt = self._rule_decision(ctx, force_side="DOWN")
        else:
            alt = rule
        return alt, x

    def _rule_decision(self, ctx: PolicyContext, force_side: Optional[str] = None) -> PolicyDecision:
        d = make_decision(
            p_up=ctx.p_up,
            poly_up_ask=ctx.up_ask,
            poly_up_bid=ctx.up_bid,
            ci_width=ctx.ci_width,
            tau_sec=ctx.tau_sec,
            bankroll=ctx.bankroll,
            edge_cfg=self._edge_cfg,
            sizing_cfg=self._sizing_cfg,
            regime_cfg=self._regime_cfg,
        )
        if force_side is None:
            return d
        if force_side == "UP" and d.action in (Action.HOLD, Action.TAKE_DOWN, Action.MAKE_DOWN):
            # bias by shifting probability slightly toward UP then recompute
            return make_decision(
                p_up=min(0.99, ctx.p_up + 0.05),
                poly_up_ask=ctx.up_ask,
                poly_up_bid=ctx.up_bid,
                ci_width=ctx.ci_width,
                tau_sec=ctx.tau_sec,
                bankroll=ctx.bankroll,
                edge_cfg=self._edge_cfg,
                sizing_cfg=self._sizing_cfg,
                regime_cfg=self._regime_cfg,
            )
        if force_side == "DOWN" and d.action in (Action.HOLD, Action.TAKE_UP, Action.MAKE_UP):
            return make_decision(
                p_up=max(0.01, ctx.p_up - 0.05),
                poly_up_ask=ctx.up_ask,
                poly_up_bid=ctx.up_bid,
                ci_width=ctx.ci_width,
                tau_sec=ctx.tau_sec,
                bankroll=ctx.bankroll,
                edge_cfg=self._edge_cfg,
                sizing_cfg=self._sizing_cfg,
                regime_cfg=self._regime_cfg,
            )
        return d

    def _force_hold_reason(self, ctx: PolicyContext) -> str:
        """Return a non-empty reason string if a hold must be forced, else ''."""
        if ctx.circuit_breaker.hard_stop_pending:
            return "hard stop: session loss limit hit — waiting for positions to close"
        if ctx.circuit_breaker.is_halted:
            return f"circuit breaker [{ctx.circuit_breaker.halt_kind}]: {ctx.circuit_breaker.halt_reason}"
        if ctx.chainlink_age_ms > ctx.stale_chainlink_ms:
            return f"stale chainlink: {ctx.chainlink_age_ms:.0f}ms (limit {ctx.stale_chainlink_ms:.0f}ms)"
        if ctx.binance_age_ms > ctx.stale_binance_ms:
            return f"stale binance: {ctx.binance_age_ms:.0f}ms (limit {ctx.stale_binance_ms:.0f}ms)"
        if ctx.up_mid < ctx.extreme_mkt_lo or ctx.up_mid > ctx.extreme_mkt_hi:
            return f"extreme market: poly_mid={ctx.up_mid:.3f} (allowed {ctx.extreme_mkt_lo:.2f}–{ctx.extreme_mkt_hi:.2f})"
        regime = classify_regime_with_config(ctx.tau_sec, self._regime_cfg)
        if regime.value == "late" and ctx.ci_width >= 0.08:
            return f"late regime: ci_width={ctx.ci_width:.3f} >= 0.08"
        return ""

    def vectorize(self, ctx: PolicyContext) -> np.ndarray:
        regime = classify_regime_with_config(ctx.tau_sec, self._regime_cfg)
        one_hot = {
            "early": [1.0, 0.0, 0.0, 0.0],
            "mid": [0.0, 1.0, 0.0, 0.0],
            "late": [0.0, 0.0, 1.0, 0.0],
            "terminal": [0.0, 0.0, 0.0, 1.0],
        }[regime.value]
        return np.array(
            [
                ctx.p_up,
                ctx.ci_width,
                ctx.gbm_p_up,
                ctx.ml_residual,
                ctx.up_mid,
                ctx.poly_spread,
                ctx.poly_extremity,
                ctx.poly_imbalance,
                ctx.ofi_5s,
                ctx.momentum_5s,
                ctx.depth_imbalance,
                ctx.binance_age_ms / 1000.0,
                ctx.chainlink_age_ms / 1000.0,
                *one_hot,
            ],
            dtype=float,
        )

    def update(self, feedback: BanditFeedback) -> None:
        self._bandit.update(feedback.action, feedback.context_vec, feedback.reward)

    def persist(self) -> None:
        self._bandit.save(self._bandit_state_path)

