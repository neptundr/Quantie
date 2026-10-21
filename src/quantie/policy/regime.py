"""
Regime classification — maps tau to a trading regime.

Thresholds from config/v1.yaml:
  early      tau > 240s        — passive entries only (GTC at fair_value - 1.5c)
  mid        60 < tau ≤ 240s   — primary trading, take when edge > threshold
  late       30 < tau ≤ 60s    — take-only, threshold +1c, CI < 0.08
  terminal   tau ≤ 30s         — no new entries, exit only

A9.2 sub-regimes (late-active 20-30s, late-watch 10-20s) are tracked
for logging but do NOT change v1 policy decisions — shadow mode only.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Regime(str, Enum):
    EARLY = "early"
    MID = "mid"
    LATE = "late"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class RegimeConfig:
    early_to_mid_seconds: float = 240.0
    mid_to_late_seconds: float = 60.0
    late_to_terminal_seconds: float = 30.0

# Late regime adjustments
LATE_EXTRA_THRESHOLD: float = 0.01     # raise edge threshold by +1c
LATE_MAX_CI_WIDTH: float = 0.08        # entry only if CI < this

# Passive entry price offset in early regime (fair_value - 1.5c)
EARLY_PASSIVE_OFFSET: float = 0.015


def classify_regime(tau_sec: float) -> Regime:
    """Return the trading regime for a given tau (seconds remaining)."""
    return classify_regime_with_config(tau_sec, RegimeConfig())


def classify_regime_with_config(tau_sec: float, cfg: RegimeConfig) -> Regime:
    """Return regime using runtime-configurable thresholds."""
    if tau_sec > cfg.early_to_mid_seconds:
        return Regime.EARLY
    if tau_sec > cfg.mid_to_late_seconds:
        return Regime.MID
    if tau_sec > cfg.late_to_terminal_seconds:
        return Regime.LATE
    return Regime.TERMINAL


def is_entry_allowed(regime: Regime) -> bool:
    """True if new entries are permitted in this regime (v1 rules)."""
    return regime in (Regime.EARLY, Regime.MID, Regime.LATE)


def is_passive_only(regime: Regime) -> bool:
    """True if only passive (GTC) entries are allowed — early regime."""
    return regime == Regime.EARLY
