"""Typed application configuration loaded from nested YAML."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class CapitalConfig(BaseModel):
    bankroll_usd: float = Field(gt=0)
    max_position_per_window_usd: float = Field(gt=0)
    max_concurrent_positions_usd: float = Field(gt=0)
    max_daily_drawdown_usd: float = Field(gt=0)
    max_total_drawdown_pct: float = Field(gt=0, lt=1)


class SizingConfig(BaseModel):
    kelly_fraction: float = Field(gt=0, le=1)
    max_position_frac_of_bankroll: float = Field(gt=0, le=1)
    min_order_size_usd: float = Field(gt=0)


class PolicyConfig(BaseModel):
    edge_threshold_base: float = Field(gt=0)
    edge_threshold_ci_multiplier: float = Field(gt=0)
    fee_buffer: float = Field(ge=0)
    terminal_zone_seconds: float = Field(gt=0)


class RegimeThresholdsConfig(BaseModel):
    early_to_mid_seconds: float = Field(gt=0)
    mid_to_late_seconds: float = Field(gt=0)
    late_to_terminal_seconds: float = Field(gt=0)


class SafetyConfig(BaseModel):
    stale_chainlink_ms: float = Field(gt=0)
    stale_binance_ms: float = Field(gt=0)
    max_errors_per_minute: int = Field(gt=0)
    drawdown_pause_hours: float = Field(gt=0)


class ExecutionConfig(BaseModel):
    order_type_for_takes: str
    order_type_for_passive: str
    rate_limit_ops_per_sec: float = Field(gt=0)
    reconcile_interval_sec: float = Field(gt=0)
    cancel_all_on_window_close_seconds: float = Field(gt=0)
    taker_fee_rate: float = Field(default=0.018, ge=0, le=0.05)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capital: CapitalConfig
    sizing: SizingConfig
    policy: PolicyConfig
    regime_thresholds: RegimeThresholdsConfig
    safety: SafetyConfig
    execution: ExecutionConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

