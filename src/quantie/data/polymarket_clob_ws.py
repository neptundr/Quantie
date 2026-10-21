"""
Polymarket CLOB WebSocket client — market channel.

Market channel (no auth):
  URL: wss://ws-subscriptions-clob.polymarket.com/ws/market
  Subscribe: {"assets_ids":[...], "type":"market", "custom_feature_enabled":true}
  Events: book, price_change, last_trade_price, tick_size_change

User channel (Phase 6, requires L2 API creds):
  URL: wss://ws-subscriptions-clob.polymarket.com/ws/user
  Subscribe: {"auth":{"apiKey":...,"secret":...,"passphrase":...},
              "markets":[condition_id], "type":"user"}
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional, Union

import websockets
import websockets.exceptions

from quantie.data.event_bus import (
    EventBus,
    PolyBookUpdate,
    PolyPriceChange,
    PolyTrade,
)
from quantie.logging_config import get_logger

log = get_logger(__name__)

_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
_RECONNECT_DELAY_MIN = 1.0
_RECONNECT_DELAY_MAX = 60.0
_HEARTBEAT_TIMEOUT_MS = 5_000


class CLOBMarketWebSocket:
    """Subscribes to UP+DOWN token order books for the current window."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._running = False
        self._asset_ids: list[str] = []
        self._resubscribe_event = asyncio.Event()
        self.last_msg_ts: float = 0.0
        self.msgs_received: int = 0

    def resubscribe(self, up_token_id: str, down_token_id: str) -> None:
        """Called by WindowManager on each window rollover."""
        self._asset_ids = [up_token_id, down_token_id]
        self._resubscribe_event.set()

    async def run(self) -> None:
        """Run forever, reconnecting with exponential backoff."""
        self._running = True
        delay = _RECONNECT_DELAY_MIN
        while self._running:
            if not self._asset_ids:
                await asyncio.sleep(0.1)
                continue
            try:
                async with websockets.connect(
                    _MARKET_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    log.info("clob_market_ws_connected", asset_ids=self._asset_ids)
                    delay = _RECONNECT_DELAY_MIN
                    await self._send_subscribe(ws)
                    self._resubscribe_event.clear()
                    await self._run_session(ws)
            except (
                websockets.exceptions.WebSocketException,
                OSError,
                asyncio.TimeoutError,
            ) as exc:
                log.warning("clob_market_ws_disconnected", error=str(exc), reconnect_in=delay)
            except Exception as exc:
                log.error("clob_market_ws_unexpected_error", error=str(exc))

            if not self._running:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, _RECONNECT_DELAY_MAX)

    async def _run_session(self, ws: websockets.WebSocketClientProtocol) -> None:
        recv_task = asyncio.create_task(self._recv_loop(ws))
        resub_task = asyncio.create_task(self._resub_watcher(ws))
        done, pending = await asyncio.wait(
            {recv_task, resub_task},
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
            self.last_msg_ts = time.time()
            self.msgs_received += 1
            try:
                events = _parse_market_message(json.loads(raw))
                for event in events:
                    await self._bus.publish(event)
            except (KeyError, ValueError, TypeError) as exc:
                log.warning("clob_market_ws_parse_error", error=str(exc))

    async def _resub_watcher(self, ws: websockets.WebSocketClientProtocol) -> None:
        """On window rollover, send updated subscription without reconnecting."""
        await self._resubscribe_event.wait()
        log.info("clob_market_ws_resubscribing", asset_ids=self._asset_ids)
        await self._send_subscribe(ws)
        self._resubscribe_event.clear()
        # Don't return — let recv_loop continue until next rollover or disconnect

    async def _send_subscribe(self, ws: websockets.WebSocketClientProtocol) -> None:
        await ws.send(json.dumps({
            "assets_ids": self._asset_ids,
            "type": "market",
            "custom_feature_enabled": True,
        }))

    def stop(self) -> None:
        self._running = False

    def is_stale(self, threshold_ms: float = _HEARTBEAT_TIMEOUT_MS) -> bool:
        if self.last_msg_ts == 0.0:
            return False
        return (time.time() - self.last_msg_ts) * 1000 > threshold_ms


# ---------------------------------------------------------------------------
# Parsers (module-level for unit testing without networking)
# ---------------------------------------------------------------------------

_ParsedEvent = Union[PolyBookUpdate, PolyPriceChange, PolyTrade]


def _parse_market_message(msg: dict | list) -> list[_ParsedEvent]:
    """Parse one raw market-channel message into zero or more typed events."""
    if isinstance(msg, list):
        results: list[_ParsedEvent] = []
        for item in msg:
            results.extend(_parse_market_message(item))
        return results

    event_type = msg.get("event_type", "")
    if event_type == "book":
        ev = _parse_book(msg)
        return [ev] if ev else []
    if event_type == "price_change":
        return _parse_price_change(msg)
    if event_type == "last_trade_price":
        ev = _parse_trade(msg)
        return [ev] if ev else []
    return []


def _parse_book(msg: dict) -> Optional[PolyBookUpdate]:
    try:
        bids = tuple(
            (float(b["price"]), float(b["size"])) for b in msg.get("bids", [])
        )
        asks = tuple(
            (float(a["price"]), float(a["size"])) for a in msg.get("asks", [])
        )
        return PolyBookUpdate(
            ts_ms=int(msg["timestamp"]),
            asset_id=str(msg["asset_id"]),
            market=str(msg["market"]),
            bids=bids,
            asks=asks,
            book_hash=str(msg.get("hash", "")),
        )
    except (KeyError, ValueError):
        return None


def _parse_price_change(msg: dict) -> list[PolyPriceChange]:
    results = []
    try:
        ts_ms = int(msg["timestamp"])
        market = str(msg["market"])
        for change in msg.get("price_changes", []):
            try:
                results.append(PolyPriceChange(
                    ts_ms=ts_ms,
                    market=market,
                    asset_id=str(change["asset_id"]),
                    price=float(change["price"]),
                    size=float(change["size"]),
                    side=str(change["side"]),
                    best_bid=float(change.get("best_bid", 0)),
                    best_ask=float(change.get("best_ask", 1)),
                ))
            except (KeyError, ValueError):
                pass
    except (KeyError, ValueError):
        pass
    return results


def _parse_trade(msg: dict) -> Optional[PolyTrade]:
    try:
        return PolyTrade(
            ts_ms=int(msg["timestamp"]),
            asset_id=str(msg["asset_id"]),
            market=str(msg["market"]),
            price=float(msg["price"]),
            size=float(msg["size"]),
            side=str(msg["side"]),
        )
    except (KeyError, ValueError):
        return None
