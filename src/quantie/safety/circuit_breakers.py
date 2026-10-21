"""
Circuit breakers.

Pause durations differ by severity:
  1. Drawdown   — cumulative PnL falls > 20% of bankroll → pause 1 hour
  2. Error rate — order rejection rate > 10% over last 60s → pause 1 hour
  3. Latency    — Binance p99 latency > 500ms over rolling 5 min → pause 5 min
                  (250ms was too tight; routine WS jitter trips it for 1h)
  4. Session loss  — realized PnL < -20% of session-start balance → hard_stop_pending=True
                     no new entries allowed; engine shuts down after open positions close
  5. Losing streak — 3 consecutive losing windows → pause 30 min (auto-resume)

EXIT and HOLD actions are never blocked by circuit breakers.
"""
from __future__ import annotations

import time
from collections import deque

import structlog

from quantie.execution.broker import STATUS_CANCELLED, STATUS_REJECTED

log = structlog.get_logger(__name__)

_PAUSE_SECONDS: float = 3_600.0          # drawdown + error rate: 1 hour
_LATENCY_PAUSE_SECONDS: float = 300.0    # latency blip: 5 minutes
_STREAK_PAUSE_SECONDS: float = 1_800.0   # losing streak: 30 minutes
_ERROR_WINDOW_SECONDS: float = 60.0
_ERROR_RATE_THRESHOLD: float = 0.10
_ERROR_MIN_SAMPLES: int = 10
_LATENCY_WINDOW_SECONDS: float = 300.0
_LATENCY_RECOVERY_WINDOW_SECONDS: float = 30.0   # shorter window for recovery check
_LATENCY_P99_HALT_MS: float = 500.0      # raised from 250ms — normal WS jitter
_LATENCY_MIN_SAMPLES: int = 30
_LATENCY_RECOVERY_MIN_SAMPLES: int = 5   # fewer samples needed for recovery confirmation
_LOSING_STREAK_LIMIT: int = 3

_REJECTED_STATUSES = frozenset({STATUS_CANCELLED, STATUS_REJECTED})


class CircuitBreaker:
    """Tracks system health and pauses new entries when limits are breached."""

    def __init__(self, max_drawdown_frac: float = 0.20) -> None:
        self._max_drawdown_frac = max_drawdown_frac
        self._paused_until: float = 0.0
        self._halt_reason: str = ""
        self._halt_kind: str = ""
        self._halted_at: float = 0.0
        self._peak_pnl: float = 0.0
        self._order_outcomes: deque[tuple[float, bool]] = deque()
        self._binance_latencies: deque[tuple[float, float]] = deque()
        self._hard_stop_pending: bool = False
        self._consecutive_losses: int = 0

    @property
    def is_halted(self) -> bool:
        return time.time() < self._paused_until

    @property
    def halt_reason(self) -> str:
        return self._halt_reason if self.is_halted else ""

    @property
    def halt_kind(self) -> str:
        """Category of the active halt: 'drawdown', 'error_rate', 'latency', or ''."""
        return self._halt_kind if self.is_halted else ""

    @property
    def halted_at(self) -> float:
        """Unix timestamp when the breaker tripped (0.0 if never tripped)."""
        return self._halted_at

    @property
    def seconds_until_resume(self) -> float:
        """Seconds remaining in the pause window (0.0 if not halted)."""
        return max(0.0, self._paused_until - time.time())

    @property
    def hard_stop_pending(self) -> bool:
        """True when session loss >= 20% of session-start balance.

        No new entries allowed. Engine shuts down after open positions close.
        """
        return self._hard_stop_pending

    def check_session_loss(self, realized_pnl: float, session_start_balance: float) -> None:
        """Trip hard stop if realized loss exceeds 20% of session-start balance."""
        if self._hard_stop_pending:
            return
        loss_limit = self._max_drawdown_frac * session_start_balance
        if realized_pnl <= -loss_limit:
            self._hard_stop_pending = True
            log.warning(
                "circuit_breaker.hard_stop_pending",
                realized_pnl=round(realized_pnl, 2),
                session_start_balance=round(session_start_balance, 2),
                loss_limit=round(loss_limit, 2),
            )

    def record_window_pnl(self, window_pnl: float) -> None:
        """Track consecutive losing windows. Pause 30 min after 3 in a row."""
        if window_pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        if self._consecutive_losses >= _LOSING_STREAK_LIMIT and not self.is_halted:
            self._trip(
                f"{self._consecutive_losses} consecutive losing windows",
                kind="losing_streak",
                pause_seconds=_STREAK_PAUSE_SECONDS,
            )

    def check_drawdown(self, realized_pnl: float, bankroll: float) -> None:
        self._peak_pnl = max(self._peak_pnl, realized_pnl)
        drawdown = self._peak_pnl - realized_pnl
        limit = self._max_drawdown_frac * bankroll
        if drawdown >= limit and not self.is_halted:
            self._trip(
                f"drawdown {drawdown:.2f} USD >= limit {limit:.2f} USD",
                kind="drawdown",
            )

    def record_order_result(self, was_rejected: bool) -> None:
        """Legacy bool-based API. Prefer record_order_status() for new call sites."""
        self._record_rejection(was_rejected)

    def record_order_status(self, status: str) -> None:
        """Record an order outcome by broker status string.

        Only STATUS_CANCELLED and STATUS_REJECTED count as rejections.
        STATUS_OPEN (resting GTC) and STATUS_FILLED are non-rejections.
        """
        self._record_rejection(status in _REJECTED_STATUSES)

    def record_binance_latency(self, latency_ms: float) -> None:
        now = time.time()
        self._binance_latencies.append((now, latency_ms))
        self._prune(self._binance_latencies, now - _LATENCY_WINDOW_SECONDS)
        if len(self._binance_latencies) < _LATENCY_MIN_SAMPLES:
            return
        lats = sorted(v for _, v in self._binance_latencies)
        p99 = lats[max(0, int(len(lats) * 0.99) - 1)]
        if p99 > _LATENCY_P99_HALT_MS and not self.is_halted:
            self._trip(
                f"Binance p99 latency {p99:.0f}ms > {_LATENCY_P99_HALT_MS:.0f}ms",
                kind="latency",
                pause_seconds=_LATENCY_PAUSE_SECONDS,
            )
        elif self.is_halted and self._halt_kind == "latency":
            # Recovery: check only recent 30s — resume as soon as latency is back to normal,
            # not when the 5-min detection window has drained (which caused the long wait).
            recent_cutoff = now - _LATENCY_RECOVERY_WINDOW_SECONDS
            recent_lats = sorted(v for t, v in self._binance_latencies if t >= recent_cutoff)
            if (
                len(recent_lats) >= _LATENCY_RECOVERY_MIN_SAMPLES
                and recent_lats[max(0, int(len(recent_lats) * 0.99) - 1)] <= _LATENCY_P99_HALT_MS
            ):
                self._paused_until = 0.0
                self._halt_reason = ""
                self._halt_kind = ""
                # Flush old bad samples so the detection window doesn't immediately re-trip.
                self._prune(self._binance_latencies, recent_cutoff)
                log.info("circuit_breaker.latency_recovered", p99_ms=round(p99, 1))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_rejection(self, was_rejected: bool) -> None:
        now = time.time()
        self._order_outcomes.append((now, was_rejected))
        self._prune(self._order_outcomes, now - _ERROR_WINDOW_SECONDS)
        if len(self._order_outcomes) < _ERROR_MIN_SAMPLES:
            return
        rejections = sum(1 for _, r in self._order_outcomes if r)
        rate = rejections / len(self._order_outcomes)
        if rate > _ERROR_RATE_THRESHOLD and not self.is_halted:
            self._trip(
                f"order rejection rate {rate:.1%} > {_ERROR_RATE_THRESHOLD:.1%}",
                kind="error_rate",
            )

    def _trip(
        self,
        reason: str,
        kind: str = "",
        pause_seconds: float = _PAUSE_SECONDS,
    ) -> None:
        now = time.time()
        self._paused_until = now + pause_seconds
        self._halt_reason = reason
        self._halt_kind = kind
        self._halted_at = now
        log.warning(
            "circuit_breaker.tripped",
            reason=reason,
            kind=kind,
            paused_for_seconds=pause_seconds,
        )

    @staticmethod
    def _prune(dq: deque, cutoff: float) -> None:
        while dq and dq[0][0] < cutoff:
            dq.popleft()
