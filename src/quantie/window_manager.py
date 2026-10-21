"""
WindowManager — orchestrates the 5-minute window lifecycle.

Responsibilities:
1. Register as a rollover callback on WindowState.
2. On each rollover:
   a. Persist the CompletedWindow to parquet (data/windows/).
   b. Resolve next window's token IDs via Gamma API (thread pool).
   c. Call clob_ws.resubscribe(up_id, down_id).
   d. Call poly_state.set_tokens(up_id, down_id).
3. Pre-fetch the *current* window's tokens at startup.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from quantie.data.gamma_client import GammaAPIError, resolve_window_tokens
from quantie.data.polymarket_clob_ws import CLOBMarketWebSocket
from quantie.logging_config import get_logger
from quantie.state.polymarket_state import PolymarketState
from quantie.state.window_state import CompletedWindow, WindowState

log = get_logger(__name__)

_WINDOWS_DIR = Path(__file__).resolve().parents[2] / "data" / "windows"


class WindowManager:
    """Wires WindowState rollover callbacks to downstream systems.

    Instantiate once, call ``start(window_start_ts)`` to fetch initial tokens,
    then let rollover callbacks drive subsequent windows.
    """

    def __init__(
        self,
        window_state: WindowState,
        clob_ws: CLOBMarketWebSocket,
        poly_state: PolymarketState,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        on_tokens: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self._window_state = window_state
        self._clob_ws = clob_ws
        self._poly_state = poly_state
        self._loop = loop  # injected for sync→async bridging in callbacks
        self._on_tokens = on_tokens

        _WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
        window_state.add_rollover_callback(self._on_rollover)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, window_start_ts: int) -> None:
        """Fetch tokens for the current window and wire up the CLOB feed.

        Call once at bot startup (synchronously, before the event loop
        takes over).
        """
        self._fetch_and_apply(window_start_ts)

    # ------------------------------------------------------------------
    # Rollover callback  (called from WindowState inside the event loop)
    # ------------------------------------------------------------------

    def _on_rollover(self, new_window_start_ts: int) -> None:
        """Fired by WindowState._handle_rollover on each window boundary.

        Runs in the asyncio event loop thread (callback is synchronous per
        WindowState contract). Heavy I/O (Gamma API) runs in a thread pool
        so it doesn't block the loop.
        """
        # 1. Persist the just-completed window (newest history entry).
        if self._window_state.history:
            self._save_completed_window(self._window_state.history[-1])

        # 2. Resolve next window tokens in a thread (retries up to ~5× max 30s).
        loop = self._loop or asyncio.get_event_loop()
        loop.run_in_executor(None, self._fetch_and_apply, new_window_start_ts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_and_apply(self, window_start_ts: int) -> None:
        """Resolve token IDs from Gamma and push to CLOB WS + PolyState."""
        try:
            up_id, down_id = resolve_window_tokens(window_start_ts)
        except (GammaAPIError, Exception) as exc:
            log.error(
                "gamma_resolve_failed",
                window_start_ts=window_start_ts,
                error=str(exc),
            )
            return

        log.info(
            "window_tokens_resolved",
            window_start_ts=window_start_ts,
            up_token=up_id[:12] + "…",
            down_token=down_id[:12] + "…",
        )
        self._clob_ws.resubscribe(up_id, down_id)
        self._poly_state.set_tokens(up_id, down_id)
        if self._on_tokens is not None:
            self._on_tokens(up_id, down_id)

    def _save_completed_window(self, window: CompletedWindow) -> None:
        """Persist one CompletedWindow to a single-row parquet file."""
        path = _WINDOWS_DIR / f"window_{window.window_start_ts}.parquet"
        try:
            df = pd.DataFrame([{
                "window_start_ts": window.window_start_ts,
                "open_price": window.open_price,
                "close_price": window.close_price,
                "outcome": window.outcome,
                "saved_at_ms": int(time.time() * 1000),
            }])
            df.to_parquet(path, index=False)
            log.info(
                "window_saved",
                path=str(path),
                window_start_ts=window.window_start_ts,
                outcome=window.outcome,
            )
        except Exception as exc:
            log.error("window_save_failed", path=str(path), error=str(exc))
