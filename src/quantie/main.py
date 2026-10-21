"""
Quantie entry point — `quantie` CLI command (pyproject.toml scripts).

Usage:
    quantie                          # paper mode, config/paper.yaml
    quantie --live                   # live mode, config/v1.yaml
    quantie --config config/v1.yaml  # explicit config
    quantie --metrics-port 8001      # custom Prometheus port

Environment variables required for live mode (see .env.example):
    POLY_PRIVATE_KEY, POLY_API_KEY, POLY_API_SECRET,
    POLY_API_PASSPHRASE, POLY_FUNDER, POLY_CHAIN_ID
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path

import structlog
from dotenv import load_dotenv

log = structlog.get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quantie",
        description="Polymarket BTC 5-minute prediction market bot",
    )
    p.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Enable live trading (default: paper mode)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: config/paper.yaml or config/v1.yaml)",
    )
    p.add_argument(
        "--metrics-port",
        type=int,
        default=8000,
        help="Prometheus metrics HTTP port (default: 8000)",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory containing trained model files",
    )
    p.add_argument(
        "--bankroll",
        type=float,
        default=None,
        help="Starting bankroll in USD — overrides config value (e.g. --bankroll 50)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print every decision to stdout",
    )
    p.add_argument(
        "--conn-test",
        action="store_true",
        default=False,
        help="Test all connections and credentials then exit — no trades placed",
    )
    p.add_argument(
        "--wait-window",
        action="store_true",
        default=False,
        help="Wait for the next clean 5-minute window boundary before trading",
    )
    return p


def _load_config(path: Path):
    from quantie.config import AppConfig

    return AppConfig.from_yaml(path)


async def _conn_test() -> None:
    """Test all external connections and credentials. Exit 0 on success, 1 on any failure."""
    from quantie.logging_config import setup_logging
    setup_logging()

    ok = True
    sep = "─" * 52

    def _pass(name: str, detail: str = "") -> None:
        suffix = f"  {detail}" if detail else ""
        print(f"  ✓  {name}{suffix}", flush=True)

    def _fail(name: str, error: str) -> None:
        nonlocal ok
        ok = False
        print(f"  ✗  {name}  →  {error}", flush=True)

    print(f"\n{sep}\n  QUANTIE CONNECTION TEST\n{sep}", flush=True)

    # 1. CLOB credentials + API
    print("\n[1/4] Polymarket CLOB credentials", flush=True)
    try:
        from quantie.execution.clob_client import ClobClient
        client = ClobClient.from_env()
        client._c.get_open_orders()
        _pass("CLOB credentials", f"api_key={client._c.creds.api_key[:8]}...")
    except Exception as exc:
        _fail("CLOB credentials", str(exc))



    # 2. Gamma REST API
    print("\n[2/4] Gamma REST API", flush=True)
    try:
        from quantie.data.gamma_client import get_market_by_slug
        from quantie.clock import make_slug, current_window_start_ts
        slug = make_slug(current_window_start_ts())
        market = get_market_by_slug(slug)
        _pass("Gamma API", f"slug={slug}  active={market.get('active', '?')}")
    except Exception as exc:
        _fail("Gamma API", str(exc))

    # 3. Binance WebSocket (connect + receive one message)
    print("\n[3/4] Binance WebSocket", flush=True)
    try:
        import websockets
        async with websockets.connect(
            "wss://stream.binance.com:9443/stream?streams=btcusdt@aggTrade",
            open_timeout=10,
        ) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)
        _pass("Binance WebSocket")
    except Exception as exc:
        _fail("Binance WebSocket", str(exc))

    # 4. Polymarket RTDS WebSocket (connect + subscribe + receive one message)
    print("\n[4/4] Polymarket RTDS WebSocket", flush=True)
    try:
        import json as _json
        import websockets
        _rtds_sub = _json.dumps({
            "action": "subscribe",
            "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*", "filters": '{"symbol":"btc/usd"}'}],
        })
        async with websockets.connect(
            "wss://ws-live-data.polymarket.com",
            ping_interval=None,
            open_timeout=10,
        ) as ws:
            await ws.send(_rtds_sub)
            # drain until we get a non-PONG message (first real price event)
            for _ in range(10):
                msg = await asyncio.wait_for(ws.recv(), timeout=10)
                if msg != "PONG":
                    _pass("Polymarket RTDS WebSocket", f"got={msg[:40]!r}")
                    break
            else:
                _fail("Polymarket RTDS WebSocket", "no price event received in 10 messages")
    except Exception as exc:
        _fail("Polymarket RTDS WebSocket", str(exc) or type(exc).__name__)

    print(f"\n{sep}", flush=True)
    if ok:
        print("  ALL CHECKS PASSED — ready for live trading\n", flush=True)
        sys.exit(0)
    else:
        print("  SOME CHECKS FAILED — fix errors above before going live\n", flush=True)
        sys.exit(1)


async def _run(args: argparse.Namespace) -> None:
    from quantie.clock import current_window_start_ts
    from quantie.data.binance_ws import BinanceWebSocket
    from quantie.data.event_bus import EventBus
    from quantie.data.polymarket_clob_ws import CLOBMarketWebSocket
    from quantie.data.polymarket_rtds_ws import RTDSWebSocket
    from quantie.engine import EngineConfig, TradingEngine
    from quantie.logging_config import setup_logging
    from quantie.metrics import record_error, start_metrics_server
    from quantie.window_manager import WindowManager

    setup_logging()

    if args.wait_window:
        from quantie.clock import next_window_start_ts
        wait_sec = next_window_start_ts() - time.time()
        if wait_sec > 0:
            import math
            print(
                f"\n  Waiting {math.ceil(wait_sec)}s for next window boundary...\n",
                flush=True,
            )
            await asyncio.sleep(wait_sec)

    paper = not args.live
    config_path = args.config
    if config_path is None:
        config_path = Path("config/paper.yaml") if paper else Path("config/v1.yaml")

    cfg = _load_config(config_path)
    log.info("quantie.start", paper=paper, config=str(config_path))
    log.info("quantie.config.resolved", config=json.dumps(cfg.model_dump()))

    # Prometheus
    try:
        start_metrics_server(port=args.metrics_port)
        log.info("metrics.started", port=args.metrics_port)
    except OSError as exc:
        log.warning("metrics.start.failed", error=str(exc))

    bus = EventBus(maxsize=50_000)
    bankroll = args.bankroll if args.bankroll is not None else float(cfg.capital.bankroll_usd)
    # Actual per-window cap is computed dynamically by the engine (20% of equity).
    # This value is stored in EngineConfig for reference only.
    max_pos = float(cfg.capital.max_position_per_window_usd)

    engine_cfg = EngineConfig(
        bankroll=bankroll,
        max_position_per_window_usd=max_pos,
        stale_chainlink_ms=float(cfg.safety.stale_chainlink_ms),
        stale_binance_ms=float(cfg.safety.stale_binance_ms),
        paper=paper,
        model_dir=args.model_dir,
        verbose=args.verbose,
        app_config=cfg,
    )
    engine = TradingEngine(event_bus=bus, config=engine_cfg)

    from quantie.notifications.telegram_notifier import build_notifier
    notifier = build_notifier()
    if notifier:
        await notifier.start()
        engine.set_notifier(notifier)
        notifier.set_status_callback(engine.get_status_snapshot)

    clob_ws = CLOBMarketWebSocket(bus)
    window_mgr = WindowManager(
        engine.window_state,
        clob_ws,
        engine.poly_state,
        on_tokens=engine.set_token_ids,
    )
    window_mgr.start(current_window_start_ts())
    binance_ws = BinanceWebSocket(bus)
    rtds_ws = RTDSWebSocket(bus)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _on_signal() -> None:
        log.info("quantie.shutdown_signal")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    async def _watchdog() -> None:
        await stop_event.wait()
        engine.stop()
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

    try:
        await asyncio.gather(
            binance_ws.run(),
            rtds_ws.run(),
            clob_ws.run(),
            engine.run(),
            _watchdog(),
        )
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        log.error("quantie.fatal", error=str(exc))
        record_error("fatal")
        sys.exit(1)
    finally:
        if notifier:
            await notifier.stop()
        log.info("quantie.exit")


def main() -> None:
    load_dotenv()
    args = _build_parser().parse_args()

    if args.conn_test:
        asyncio.run(_conn_test())
        return

    if args.live:
        print(
            "\n⚠️  LIVE MODE: real orders will be placed.\n"
            "Ensure POLY_* env vars are set and wallet is funded.\n"
            "Press Ctrl+C within 5s to abort...",
            flush=True,
        )
        time.sleep(5)

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
