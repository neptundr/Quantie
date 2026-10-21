from __future__ import annotations

from pathlib import Path

from quantie.policy.bandit import LinUCBBandit
from quantie.policy.edge import EdgeConfig
from quantie.policy.policy_layer import ExecutionPolicy, PolicyContext
from quantie.policy.regime import RegimeConfig
from quantie.policy.sizing import SizingConfig
from quantie.safety.circuit_breakers import CircuitBreaker


def _ctx(**overrides):
    base = dict(
        now_ms=0,
        tau_sec=120.0,
        p_up=0.7,
        ci_width=0.04,
        gbm_p_up=0.65,
        ml_residual=0.05,
        up_ask=0.55,
        up_bid=0.53,
        up_mid=0.54,
        poly_spread=0.02,
        poly_extremity=0.04,
        poly_imbalance=0.1,
        ofi_5s=0.2,
        momentum_5s=0.1,
        depth_imbalance=0.05,
        binance_age_ms=100.0,
        chainlink_age_ms=100.0,
        bankroll=1000.0,
        is_live=False,
        stale_chainlink_ms=2500.0,
        stale_binance_ms=1500.0,
        extreme_mkt_lo=0.15,
        extreme_mkt_hi=0.85,
        circuit_breaker=CircuitBreaker(),
    )
    base.update(overrides)
    return PolicyContext(**base)


def test_policy_forces_hold_when_stale(tmp_path) -> None:
    policy = ExecutionPolicy(
        edge_cfg=EdgeConfig(),
        sizing_cfg=SizingConfig(),
        regime_cfg=RegimeConfig(),
        bandit=LinUCBBandit(dim=17),
        bandit_state_path=tmp_path / "s.npz",
    )
    decision, _ = policy.choose(_ctx(chainlink_age_ms=5000.0))
    assert decision.action.value == "HOLD"


def test_policy_terminal_exits(tmp_path) -> None:
    policy = ExecutionPolicy(
        edge_cfg=EdgeConfig(),
        sizing_cfg=SizingConfig(),
        regime_cfg=RegimeConfig(),
        bandit=LinUCBBandit(dim=17),
        bandit_state_path=Path(tmp_path / "s2.npz"),
    )
    decision, _ = policy.choose(_ctx(tau_sec=20.0))
    assert decision.action.value == "EXIT"

