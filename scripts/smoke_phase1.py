"""
Phase 1 smoke test — connect all three WebSocket feeds for N seconds,
log every event to parquet, then print an arrival-rate summary.

Usage:
    python scripts/smoke_phase1.py            # 30-minute run (default)
    python scripts/smoke_phase1.py --duration 60  # quick 60-second sanity check

Output file: data/ticks/phase1_smoke_<YYYYMMDD_HHMMSS>.parquet
  columns: ts_ms (int64), event_type (str), price (float64),
           qty (float64), asset_id (str)

Pass criteria:
  - Binance aggTrade arrival rate >= 1/s
  - Chainlink update rate >= 1 per 10s
  - Chainlink gaps > 5s < 1% of total updates
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantie.data.binance_ws import BinanceWebSocket
from quantie.data.event_bus import (
    BinanceDepth,
    BinanceTrade,
    ChainlinkPrice,
    EventBus,
    PolyBookUpdate,
    PolyPriceChange,
    PolyTrade,
)
from quantie.data.polymarket_clob_ws import CLOBMarketWebSocket
from quantie.data.polymarket_rtds_ws import RTDSWebSocket
from quantie.logging_config import get_logger, setup_logging
from quantie.state.binance_state import BinanceState
from quantie.state.polymarket_state import PolymarketState
from quantie.state.window_state import WindowState

setup_logging("INFO")
log = get_logger("smoke_phase1")

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ticks"
DATA_DIR.mkdir(parents=True, exist_ok=True)


async def run_smoke(duration_seconds: int) -> None:
    bus = EventBus()
    window_state = WindowState()
    binance_state = BinanceState()
    poly_state = PolymarketState()

    binance_ws = BinanceWebSocket(bus)
    rtds_ws = RTDSWebSocket(bus)
    clob_ws = CLOBMarketWebSocket(bus)

    records: list[dict] = []
    stop_event = asyncio.Event()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    async def consumer_loop() -> None:
        while not stop_event.is_set():
            try:
                event = await asyncio.wait_for(bus.consume(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            record: dict = {
                "ts_ms": int(time.time() * 1000),
                "event_type": type(event).__name__,
                "price": None,
                "qty": None,
                "asset_id": None,
            }

            if isinstance(event, BinanceTrade):
                binance_state.on_trade(event)
                record.update(price=event.price, qty=event.qty)
            elif isinstance(event, BinanceDepth):
                binance_state.on_depth(event)
                record["price"] = event.bids[0][0] if event.bids else None
            elif isinstance(event, ChainlinkPrice):
                window_state.on_chainlink(event)
                record["price"] = event.price
            elif isinstance(event, PolyBookUpdate):
                poly_state.on_book(event)
                record["asset_id"] = event.asset_id
                record["price"] = event.asks[0][0] if event.asks else None
            elif isinstance(event, PolyPriceChange):
                poly_state.on_price_change(event)
                record.update(asset_id=event.asset_id, price=event.price)
            elif isinstance(event, PolyTrade):
                poly_state.on_trade(event)
                record.update(asset_id=event.asset_id, price=event.price, qty=event.size)

            records.append(record)

    async def timer() -> None:
        await asyncio.sleep(duration_seconds)
        stop_event.set()

    log.info("smoke_starting", duration_seconds=duration_seconds)
    tasks = [
        asyncio.create_task(binance_ws.run()),
        asyncio.create_task(rtds_ws.run()),
        asyncio.create_task(clob_ws.run()),
        asyncio.create_task(consumer_loop()),
        asyncio.create_task(timer()),
    ]
    await stop_event.wait()
    for task in tasks:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    _save_and_report(records, binance_ws, rtds_ws, duration_seconds)


def _save_and_report(
    records: list[dict],
    binance_ws: BinanceWebSocket,
    rtds_ws: RTDSWebSocket,
    duration_seconds: int,
) -> None:
    if not records:
        print("No events recorded — check network connectivity.")
        return

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / f"phase1_smoke_{ts_str}.parquet"
    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} events to {out_path}")

    by_type = df.groupby("event_type").size()
    print(f"\n=== Phase 1 Smoke Test Summary — {duration_seconds}s ===")
    print("Event counts:")
    for etype, count in by_type.items():
        print(f"  {etype:<30} {count:>7}  ({count / duration_seconds:.2f}/s)")

    binance_trades = by_type.get("BinanceTrade", 0)
    chainlink_prices = by_type.get("ChainlinkPrice", 0)

    print("\nChecks:")
    _check("Binance aggTrade >= 1/s", binance_trades / duration_seconds >= 1.0)
    _check("Chainlink >= 1 per 10s", chainlink_prices / duration_seconds >= 0.1)
    _check("Binance WS received > 0 msgs", binance_ws.msgs_received > 0)
    _check("RTDS WS received > 0 msgs", rtds_ws.msgs_received > 0)

    chainlink_df = df[df["event_type"] == "ChainlinkPrice"].sort_values("ts_ms")
    if len(chainlink_df) > 1:
        gaps_s = chainlink_df["ts_ms"].diff().dropna() / 1000
        gap_pct = (gaps_s > 5.0).sum() / len(gaps_s) * 100
        _check(
            f"Chainlink gaps >5s < 1% of updates (actual: {gap_pct:.1f}%)",
            gap_pct < 1.0,
        )
    else:
        print("  [SKIP] Too few Chainlink events for gap analysis")


def _check(label: str, passed: bool) -> None:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=1800,
                        help="Run duration in seconds (default: 1800)")
    args = parser.parse_args()
    asyncio.run(run_smoke(args.duration))


if __name__ == "__main__":
    main()
