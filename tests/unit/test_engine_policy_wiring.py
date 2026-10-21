from __future__ import annotations

from pathlib import Path

from quantie.config import AppConfig
from quantie.data.event_bus import EventBus
from quantie.engine import EngineConfig, TradingEngine


def test_engine_has_execution_policy_and_forecaster() -> None:
    cfg = AppConfig.from_yaml(Path("config/paper.yaml"))
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
    assert engine._policy is not None
    assert hasattr(engine._model, "predict")

