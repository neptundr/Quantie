"""
WindowState — tracks the current 5-minute window.

Responsibilities:
- Detect window rollover (new window_start_ts).
- Capture open_price: first Chainlink update with
  payload_ts_ms >= window_start_ts * 1000.
- Expose tau (time to resolution) via clock module.
- Record completed windows to a rolling history deque.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from quantie.clock import make_slug, current_window_start_ts, time_to_resolution
from quantie.data.event_bus import ChainlinkPrice
from quantie.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class CompletedWindow:
    window_start_ts: int
    open_price: float
    close_price: Optional[float]  # last Chainlink price before resolution
    outcome: Optional[str]        # "UP" | "DOWN" | None (not yet resolved)


class WindowState:
    """Mutable state updated by the central event consumer loop.

    On each ChainlinkPrice event:
      1. Check for window rollover — fire callbacks and reset if detected.
      2. Capture open_price on first qualifying oracle update this window.
      3. Update current_chainlink_price and last_chainlink_ts_ms.
    """

    def __init__(self, history_len: int = 500) -> None:
        self.window_start_ts: int = current_window_start_ts()
        self.open_price: Optional[float] = None
        self.current_chainlink_price: Optional[float] = None
        self.last_chainlink_ts_ms: int = 0
        self.history: deque[CompletedWindow] = deque(maxlen=history_len)
        self._rollover_callbacks: list[Callable[[int], None]] = []
        self._completed_callbacks: list[Callable[[CompletedWindow], None]] = []

    @property
    def slug(self) -> str:
        return make_slug(self.window_start_ts)

    @property
    def tau(self) -> float:
        return time_to_resolution()

    @property
    def window_start_ms(self) -> int:
        return self.window_start_ts * 1000

    def on_chainlink(self, event: ChainlinkPrice) -> None:
        """Called by the main consumer loop on every ChainlinkPrice event."""
        new_window_start = current_window_start_ts()

        if new_window_start != self.window_start_ts:
            self._handle_rollover(new_window_start, event)
            return

        if self.open_price is None and event.payload_ts_ms >= self.window_start_ms:
            self.open_price = event.price
            log.info(
                "window_open_price_captured",
                window=self.slug,
                open_price=self.open_price,
                oracle_lag_ms=event.received_ts_ms - event.payload_ts_ms,
            )

        self.current_chainlink_price = event.price
        self.last_chainlink_ts_ms = event.received_ts_ms

    def _handle_rollover(self, new_window_start: int, trigger: ChainlinkPrice) -> None:
        # Determine close price using oracle timestamp (payload_ts_ms), not system time.
        # If the trigger event's oracle timestamp is still within the closing window
        # (payload_ts_ms < window_end_ms) it IS the last oracle price for this window
        # and gives the most accurate close.  If oracle timestamp is already in the new
        # window, fall back to the penultimate price (current_chainlink_price).
        window_end_ms = self.window_start_ms + 300_000
        if trigger.payload_ts_ms < window_end_ms:
            close_price = trigger.price   # oracle still in old window — best close
        else:
            close_price = self.current_chainlink_price  # oracle jumped ahead — use penultimate

        completed = CompletedWindow(
            window_start_ts=self.window_start_ts,
            open_price=self.open_price or trigger.price,
            close_price=close_price,
            outcome=_compute_outcome(self.open_price, close_price),
        )
        self.history.append(completed)
        log.info(
            "window_completed",
            window=make_slug(completed.window_start_ts),
            open=completed.open_price,
            close=completed.close_price,
            outcome=completed.outcome,
            trigger_payload_ms=trigger.payload_ts_ms,
            window_end_ms=window_end_ms,
        )

        for cb in self._completed_callbacks:
            try:
                cb(completed)
            except Exception as exc:
                log.error("completed_callback_error", error=str(exc))

        self.window_start_ts = new_window_start
        self.open_price = None
        self.current_chainlink_price = trigger.price

        for cb in self._rollover_callbacks:
            try:
                cb(new_window_start)
            except Exception as exc:
                log.error("rollover_callback_error", error=str(exc))

    def add_rollover_callback(self, cb: Callable[[int], None]) -> None:
        self._rollover_callbacks.append(cb)

    def add_completed_callback(self, cb: Callable[[CompletedWindow], None]) -> None:
        """Register callback fired with CompletedWindow on each rollover (includes outcome)."""
        self._completed_callbacks.append(cb)


def _compute_outcome(
    open_price: Optional[float], close_price: Optional[float]
) -> Optional[str]:
    if open_price is None or close_price is None:
        return None
    return "UP" if close_price >= open_price else "DOWN"
