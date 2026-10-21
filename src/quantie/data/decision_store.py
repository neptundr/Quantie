"""
Daily-rotating CSV log of every live bot decision.

One row per tick evaluated by the policy layer.
File rotates at UTC midnight: data/decisions/decisions_YYYY-MM-DD_HH-MM.csv
The HH-MM suffix is the UTC start time of the bot run.
"""
from __future__ import annotations

import csv
import datetime
from pathlib import Path

_DECISIONS_DIR = Path("data/decisions")

_COLUMNS = [
    "ts_ms",
    "date_utc",
    "window_start_ts",
    "tau_sec",
    "p_up",
    "gbm_p_up",
    "ml_residual",
    "ci_width",
    "action",
    "size_usd",
    "edge",
    "regime",
    "reason",
    "balance_total",
    "balance_cash",
]


class DecisionStore:
    """Append-only CSV writer for policy decisions.

    Usage::

        store = DecisionStore()
        store.record(ts_ms=..., window_start_ts=..., ...)
        store.close()  # call on shutdown
    """

    def __init__(self) -> None:
        _DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        self._start_tag: str = now.strftime("%H-%M")
        self._current_date: str = ""
        self._writer: csv.DictWriter | None = None
        self._fh = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        ts_ms: int,
        window_start_ts: int,
        tau_sec: float,
        p_up: float,
        gbm_p_up: float,
        ml_residual: float,
        ci_width: float,
        action: str,
        size_usd: float,
        edge: float,
        regime: str,
        reason: str,
        balance_total: float,
        balance_cash: float,
    ) -> None:
        """Write one decision row and flush to disk immediately."""
        dt = datetime.datetime.fromtimestamp(
            ts_ms / 1000.0, tz=datetime.timezone.utc
        )
        date_str = dt.strftime("%Y-%m-%d")

        if date_str != self._current_date:
            self._rotate(date_str)

        assert self._writer is not None
        self._writer.writerow(
            {
                "ts_ms": ts_ms,
                "date_utc": dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                "window_start_ts": window_start_ts,
                "tau_sec": round(tau_sec, 2),
                "p_up": round(p_up, 6),
                "gbm_p_up": round(gbm_p_up, 6),
                "ml_residual": round(ml_residual, 6),
                "ci_width": round(ci_width, 6),
                "action": action,
                "size_usd": round(size_usd, 4),
                "edge": round(edge, 6),
                "regime": regime,
                "reason": reason,
                "balance_total": round(balance_total, 4),
                "balance_cash": round(balance_cash, 4),
            }
        )
        self._fh.flush()

    def close(self) -> None:
        """Close any open file handle cleanly."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None
            self._current_date = ""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rotate(self, date_str: str) -> None:
        if self._fh is not None:
            self._fh.close()

        path = _DECISIONS_DIR / f"decisions_{date_str}_{self._start_tag}.csv"
        new_file = not path.exists()

        self._fh = open(path, "a", newline="", encoding="utf-8")  # noqa: SIM115
        self._writer = csv.DictWriter(self._fh, fieldnames=_COLUMNS)

        if new_file:
            self._writer.writeheader()
            self._fh.flush()

        self._current_date = date_str
