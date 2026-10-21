from __future__ import annotations

from pathlib import Path

from quantie.config import AppConfig
from quantie.data.event_bus import EventBus
from quantie.engine import EngineConfig, TradingEngine


def test_app_config_loads_nested_yaml() -> None:
    cfg = AppConfig.from_yaml(Path("config/paper.yaml"))
    assert cfg.capital.bankroll_usd == 25
    assert cfg.safety.stale_chainlink_ms == 2500
    assert cfg.regime_thresholds.early_to_mid_seconds == 240
    assert cfg.execution.cancel_all_on_window_close_seconds == 35


def test_engine_uses_nested_config_values() -> None:
    cfg = AppConfig.from_yaml(Path("config/v1.yaml"))
    engine = TradingEngine(
        event_bus=EventBus(),
        config=EngineConfig(
            bankroll=cfg.capital.bankroll_usd,
            max_position_per_window_usd=cfg.capital.max_position_per_window_usd,
            stale_chainlink_ms=cfg.safety.stale_chainlink_ms,
            stale_binance_ms=cfg.safety.stale_binance_ms,
            app_config=cfg,
        ),
    )
    assert engine._cancel_gtc_before_terminal == cfg.execution.cancel_all_on_window_close_seconds
    assert engine._cfg.stale_chainlink_ms == cfg.safety.stale_chainlink_ms
    assert engine._cfg.stale_binance_ms == cfg.safety.stale_binance_ms

