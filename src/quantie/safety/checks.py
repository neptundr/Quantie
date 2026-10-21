"""
Pre-trade safety checks.

All checks run before any order is placed. A single failed check blocks the trade.

Checks (in order):
  1. kill_switch       — manual halt flag
  2. drawdown          — realized loss exceeds max drawdown
  3. stale_chainlink   — Chainlink price too old
  4. stale_binance     — Binance price too old
  5. position_cap      — already at max exposure for this window
  6. terminal_zone     — tau ≤ terminal threshold (no new entries)
"""
from __future__ import annotations

from dataclasses import dataclass

from quantie.policy.decision import Action

# Staleness limits (ms) — match config/v1.yaml
_STALE_CHAINLINK_MS: float = 2_500.0
_STALE_BINANCE_MS: float = 1_500.0

# Drawdown limit: 20 % of starting bankroll
_MAX_DRAWDOWN_FRAC: float = 0.28

# Terminal zone: no new entries at or below this tau
_TERMINAL_TAU_SEC: float = 30.0


@dataclass(frozen=True)
class SafetySnapshot:
    """Snapshot of system state used for pre-trade safety evaluation."""

    kill_switch: bool               # True → all trading halted immediately
    realized_pnl_usd: float         # cumulative realized P&L this session (negative = loss)
    starting_bankroll: float        # bankroll at session start (denominator for drawdown)
    chainlink_age_ms: float         # ms since last Chainlink price received
    binance_age_ms: float           # ms since last Binance trade received
    window_exposure_usd: float      # total USD committed to open positions this window
    max_position_usd: float         # per-window exposure cap (from config)
    tau_sec: float                  # seconds remaining in window
    circuit_breaker_halted: bool = False  # True → circuit breaker tripped


def pre_trade_check(
    action: Action,
    safety: SafetySnapshot,
) -> tuple[bool, str]:
    """Return (allowed, reason) for a proposed action.

    Returns (True, "") when trade is permitted.
    Returns (False, "<reason>") when any check fails.

    HOLD and EXIT actions skip most checks (EXIT always allowed).
    """
    # EXIT always permitted — never block an exit
    if action == Action.EXIT:
        return True, ""

    # HOLD never needs a safety check
    if action == Action.HOLD:
        return True, ""

    # 1. Kill switch
    if safety.kill_switch:
        return False, "kill switch active"

    # 2. Circuit breaker
    if safety.circuit_breaker_halted:
        return False, "circuit breaker halted"

    # 3. Drawdown
    max_loss = _MAX_DRAWDOWN_FRAC * safety.starting_bankroll
    if safety.realized_pnl_usd <= -max_loss:
        return False, (
            f"drawdown limit hit: pnl={safety.realized_pnl_usd:.2f} "
            f"limit={-max_loss:.2f}"
        )

    # 4. Stale Chainlink oracle
    if safety.chainlink_age_ms > _STALE_CHAINLINK_MS:
        return False, (
            f"stale Chainlink: age={safety.chainlink_age_ms:.0f}ms "
            f"limit={_STALE_CHAINLINK_MS:.0f}ms"
        )

    # 4. Stale Binance feed
    if safety.binance_age_ms > _STALE_BINANCE_MS:
        return False, (
            f"stale Binance: age={safety.binance_age_ms:.0f}ms "
            f"limit={_STALE_BINANCE_MS:.0f}ms"
        )

    # 5. Position cap
    if safety.window_exposure_usd >= safety.max_position_usd:
        return False, (
            f"position cap reached: exposure={safety.window_exposure_usd:.2f} "
            f"cap={safety.max_position_usd:.2f}"
        )

    # 6. Terminal zone — MAKE_* entries also blocked (no new entries at all)
    if safety.tau_sec <= _TERMINAL_TAU_SEC:
        return False, f"terminal zone: tau={safety.tau_sec:.1f}s"

    return True, ""
