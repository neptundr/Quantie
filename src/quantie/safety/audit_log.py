"""
Audit log — append-only JSON lines for every decision.

Every tick that reaches make_decision() is logged here with the full
feature snapshot and model output. This is the primary source of
training data for the ML residual model.

Written to logs/audit.log via the structlog audit logger (see logging_config.py).
"""
from __future__ import annotations

from quantie.logging_config import get_audit_logger

_audit = get_audit_logger()


def log_decision(
    *,
    ts_ms: int,
    window_start_ts: int,
    tau_sec: float,
    feature_row: dict,
    p_up: float,
    gbm_p_up: float,
    ml_residual: float,
    ci_width: float,
    action: str,
    size_usd: float,
    edge: float,
    regime: str,
    reason: str,
) -> None:
    """Log one decision tick to the audit log.

    All numeric feature fields from FeatureVector.to_lgbm_row() are included
    so each row is a self-contained training sample once outcome is joined.
    """
    numeric_features = {
        k: round(v, 6)
        for k, v in feature_row.items()
        if isinstance(v, (int, float))
    }
    _audit.info(
        "decision",
        ts_ms=ts_ms,
        window_start_ts=window_start_ts,
        tau_sec=round(tau_sec, 2),
        p_up=round(p_up, 4),
        gbm_p_up=round(gbm_p_up, 4),
        ml_residual=round(ml_residual, 6),
        ci_width=round(ci_width, 4),
        action=action,
        size_usd=round(size_usd, 4),
        edge=round(edge, 4),
        regime=regime,
        reason=reason,
        **numeric_features,
    )
