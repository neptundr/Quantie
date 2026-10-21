"""
Polymarket RTDS WebSocket client — subscribes to crypto_prices_chainlink
(btc/usd) and pushes ChainlinkPrice events to the EventBus.

Protocol (docs.polymarket.com/market-data/websocket/rtds):
- Endpoint:  wss://ws-live-data.polymarket.com
- Must send the string "PING" every 5 seconds to keep alive.
- Subscribe:  {"action":"subscribe","subscriptions":[{...}]}
- Envelope:   {"topic":..., "type":..., "timestamp":ms, "payload":{...}}
- Chainlink payload: {"symbol":"btc/usd", "timestamp":ms, "value":float}
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import websockets
import websockets.exceptions

from quantie.data.event_bus import ChainlinkPrice, EventBus
from quantie.logging_config import get_logger

log = get_logger(__name__)

_URL = "wss://ws-live-data.polymarket.com"
_PING_INTERVAL = 5.0        # seconds — mandatory per RTDS docs
_RECONNECT_DELAY_MIN = 1.0
_RECONNECT_DELAY_MAX = 60.0
_HEARTBEAT_TIMEOUT_MS = 2_500  # stale_chainlink_ms from v1.yaml

_SUBSCRIBE_MSG = json.dumps({
    "action": "subscribe",
    "subscriptions": [
        {
            "topic": "crypto_prices_chainlink",
            "type": "*",
            "filters": '{"symbol":"btc/usd"}',
        }
    ],
})


class RTDSWebSocket:
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
                    ping_interval=None,  # we drive PING manually (server expects "PING" every 5s)
                    close_timeout=5,
                ) as ws:
                    log.info("rtds_ws_connected")
                    delay = _RECONNECT_DELAY_MIN
                    await ws.send(_SUBSCRIBE_MSG)
                    await self._run_session(ws)
            except (
                websockets.exceptions.WebSocketException,
                OSError,
                asyncio.TimeoutError,
            ) as exc:
                log.warning("rtds_ws_disconnected", error=str(exc), reconnect_in=delay)
            except Exception as exc:
                log.error("rtds_ws_unexpected_error", error=str(exc))

            if not self._running:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, _RECONNECT_DELAY_MAX)

    async def _run_session(self, ws: websockets.WebSocketClientProtocol) -> None:
        recv_task = asyncio.create_task(self._recv_loop(ws))
        ping_task = asyncio.create_task(self._ping_loop(ws))
        done, pending = await asyncio.wait(
            {recv_task, ping_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        for task in done:
            exc = task.exception()
            if exc:
                raise exc

    async def _recv_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        async for raw in ws:
            if not self._running:
                break
            received_ts_ms = int(time.time() * 1000)
            self.last_msg_ts = time.time()
            self.msgs_received += 1
            # Server sends plain-text "PONG" in response to our "PING" — skip it.
            if isinstance(raw, str) and not raw.startswith("{"):
                continue
            try:
                msg = json.loads(raw)
                event = _parse_chainlink(msg, received_ts_ms)
                if event:
                    await self._bus.publish(event)
            except (KeyError, ValueError, TypeError) as exc:
                log.warning("rtds_ws_parse_error", error=str(exc))

    async def _ping_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Send mandatory PING string every 5 s to maintain the connection."""
        while self._running:
            await asyncio.sleep(_PING_INTERVAL)
            await ws.send("PING")

    def stop(self) -> None:
        self._running = False

    def is_stale(self, threshold_ms: float = _HEARTBEAT_TIMEOUT_MS) -> bool:
        if self.last_msg_ts == 0.0:
            return False
        return (time.time() - self.last_msg_ts) * 1000 > threshold_ms


# ---------------------------------------------------------------------------
# Parser (module-level for unit testing without networking)
# ---------------------------------------------------------------------------

def _parse_chainlink(msg: dict, received_ts_ms: int) -> Optional[ChainlinkPrice]:
    if msg.get("topic") != "crypto_prices_chainlink":
        return None
    payload = msg.get("payload", {})
    try:
        return ChainlinkPrice(
            payload_ts_ms=int(payload["timestamp"]),
            received_ts_ms=received_ts_ms,
            price=float(payload["value"]),
        )
    except (KeyError, ValueError):
        return None
