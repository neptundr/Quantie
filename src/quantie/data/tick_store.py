"""
TickStore — buffers per-tick feature rows during a window, flushes to
parquet with outcome on rollover.

Each parquet file at data/ticks/ticks_{window_start_ts}.parquet contains
one row per decision tick for that window, with outcome (1=UP, 0=DOWN)
appended on flush. These files are the training dataset for the ML model.

Schema (columns):
  ts_ms             int    wall-clock time of tick
  window_start_ts   int    Unix ts of window start
  p_up              float  final calibrated P(UP)
  gbm_p_up          float  GBM-only component
  ml_residual       float  ML correction
  ci_width          float  confidence interval width
  action            str    Action enum value
  regime            str    Regime enum value
  outcome           float  1.0=UP won, 0.0=DOWN won, NaN=unresolved
  <feature fields>  float  all numeric FeatureVector fields
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

_TICKS_DIR = Path(__file__).resolve().parents[3] / "data" / "ticks"


class TickStore:
    """Buffer tick rows in memory per window; flush to parquet on rollover."""

    def __init__(self) -> None:
        _TICKS_DIR.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict] = []

    def record(
        self,
        *,
        ts_ms: int,
        window_start_ts: int,
        feature_row: dict,
        p_up: float,
        gbm_p_up: float,
        ml_residual: float,
        ci_width: float,
        action: str,
        regime: str,
    ) -> None:
        """Buffer one tick. Called every decision tick (~1/sec)."""
        row: dict = {
            "ts_ms": ts_ms,
            "window_start_ts": window_start_ts,
            "p_up": p_up,
            "gbm_p_up": gbm_p_up,
            "ml_residual": ml_residual,
            "ci_width": ci_width,
            "action": action,
            "regime": regime,
            "outcome": float("nan"),
        }
        for k, v in feature_row.items():
            if isinstance(v, (int, float)):
                row[k] = v
        self._rows.append(row)

    def flush(self, window_start_ts: int, outcome: Optional[str]) -> None:
        """Write buffered rows for this window to parquet, then discard them.

        outcome: "UP" → 1.0, "DOWN" → 0.0, None → NaN
        """
        rows = [r for r in self._rows if r["window_start_ts"] == window_start_ts]
        self._rows = [r for r in self._rows if r["window_start_ts"] != window_start_ts]

        if not rows:
            return

        outcome_val = 1.0 if outcome == "UP" else (0.0 if outcome == "DOWN" else float("nan"))
        for r in rows:
            r["outcome"] = outcome_val

        path = _TICKS_DIR / f"ticks_{window_start_ts}.parquet"
        try:
            df = pd.DataFrame(rows)
            df.to_parquet(path, index=False)
            log.info(
                "tick_store.flushed",
                window_start_ts=window_start_ts,
                rows=len(rows),
                outcome=outcome,
                path=str(path),
            )
        except Exception as exc:
            log.error("tick_store.flush_failed", window_start_ts=window_start_ts, error=str(exc))
