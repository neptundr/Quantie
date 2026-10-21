"""
Binance WebSocket client — subscribes to btcusdt@aggTrade and
btcusdt@depth20@100ms via the combined stream endpoint.

Pushes BinanceTrade and BinanceDepth events to the shared EventBus.
Reconnects with exponential backoff on any disconnection.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import websockets
import websockets.exceptions

from quantie.data.event_bus import BinanceDepth, BinanceTrade, EventBus
from quantie.logging_config import get_logger

log = get_logger(__name__)

_URL = (
    "wss://stream.binance.com:9443/stream"
    "?streams=btcusdt@aggTrade/btcusdt@depth20@100ms"
)
_RECONNECT_DELAY_MIN = 1.0
_RECONNECT_DELAY_MAX = 60.0
_HEARTBEAT_TIMEOUT_MS = 5_000


class BinanceWebSocket:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._running = False
        self.last_msg_ts: float = 0.0
        self.msgs_received: int = 0

    async def run(self) -> None:
        """Run forever, reconnecting with exponential backoff."""
        self._running = True
        delay = _RECONNECT_DELAY_MIN
        while self._running:
            try:
                async with websockets.connect(
                    _URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    log.info("binance_ws_connected")
                    delay = _RECONNECT_DELAY_MIN
                    await self._run_session(ws)
            except (
                websockets.exceptions.WebSocketException,
                OSError,
                asyncio.TimeoutError,
            ) as exc:
                log.warning("binance_ws_disconnected", error=str(exc), reconnect_in=delay)
            except Exception as exc:
                log.error("binance_ws_unexpected_error", error=str(exc))

            if not self._running:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, _RECONNECT_DELAY_MAX)

    async def _run_session(self, ws: websockets.WebSocketClientProtocol) -> None:
        async for raw in ws:
            if not self._running:
                break
            self.last_msg_ts = time.time()
            self.msgs_received += 1
            try:
                msg = json.loads(raw)
                stream: str = msg.get("stream", "")
                data: dict = msg.get("data", {})
                if "aggTrade" in stream:
                    event = _parse_agg_trade(data)
                    if event:
                        await self._bus.publish(event)
                elif "depth20" in stream:
                    event = _parse_depth(data)
                    if event:
                        await self._bus.publish(event)
            except (KeyError, ValueError, TypeError) as exc:
                log.warning("binance_ws_parse_error", error=str(exc))

    def stop(self) -> None:
        self._running = False

    def is_stale(self, threshold_ms: float = _HEARTBEAT_TIMEOUT_MS) -> bool:
        """Return True if no message received within threshold_ms."""
        if self.last_msg_ts == 0.0:
            return False
        return (time.time() - self.last_msg_ts) * 1000 > threshold_ms


# ---------------------------------------------------------------------------
# Parsers (module-level so they can be unit-tested without networking)
# ---------------------------------------------------------------------------

def _parse_agg_trade(data: dict) -> Optional[BinanceTrade]:
    try:
        return BinanceTrade(
            ts_ms=int(data["E"]),
            trade_ts_ms=int(data["T"]),
            price=float(data["p"]),
            qty=float(data["q"]),
            is_buyer_maker=bool(data["m"]),
        )
    except (KeyError, ValueError):
        return None


def _parse_depth(data: dict) -> Optional[BinanceDepth]:
    try:
        ts_ms = int(time.time() * 1000)
        bids = tuple(
            (float(p), float(q)) for p, q in data["bids"] if float(q) > 0
        )
        asks = tuple(
            (float(p), float(q)) for p, q in data["asks"] if float(q) > 0
        )
        return BinanceDepth(ts_ms=ts_ms, bids=bids, asks=asks)
    except (KeyError, ValueError):
        return None
