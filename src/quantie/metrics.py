"""
Prometheus metrics.

Exported metrics:
  quantie_pnl_realized_usd       Gauge   — cumulative realized P&L
  quantie_position_size_usd      Gauge   — current open position notional
  quantie_model_p_up             Gauge   — latest model P(UP)
  quantie_orders_placed_total    Counter — orders submitted to broker
  quantie_orders_filled_total    Counter — orders confirmed filled
  quantie_orders_cancelled_total Counter — FOK misses + explicit cancels
  quantie_errors_total           Counter — errors by type label
  quantie_regime                 Gauge   — 0=EARLY,1=MID,2=LATE,3=TERMINAL

Usage:
    from quantie.metrics import start_metrics_server, record_order_placed
    start_metrics_server(port=8000)
    record_order_placed()
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, start_http_server

# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

pnl_realized_usd = Gauge(
    "quantie_pnl_realized_usd",
    "Cumulative realized P&L in USD for this session",
)

position_size_usd = Gauge(
    "quantie_position_size_usd",
    "Current open position notional in USD",
)

model_p_up = Gauge(
    "quantie_model_p_up",
    "Latest model P(UP) prediction",
)

orders_placed_total = Counter(
    "quantie_orders_placed_total",
    "Total orders submitted to broker (FOK + GTC)",
)

orders_filled_total = Counter(
    "quantie_orders_filled_total",
    "Total orders confirmed filled",
)

orders_cancelled_total = Counter(
    "quantie_orders_cancelled_total",
    "Total orders cancelled (FOK misses + explicit cancels)",
)

errors_total = Counter(
    "quantie_errors_total",
    "Total errors by type",
    ["error_type"],
)

regime_gauge = Gauge(
    "quantie_regime",
    "Current trading regime (0=EARLY, 1=MID, 2=LATE, 3=TERMINAL)",
)

_REGIME_VALUES: dict[str, int] = {
    "early": 0,
    "mid": 1,
    "late": 2,
    "terminal": 3,
}

# ------------------------------------------------------------------
# Helpers called from engine / executor
# ------------------------------------------------------------------

def start_metrics_server(port: int = 8000) -> None:
    """Start Prometheus HTTP server on given port."""
    start_http_server(port)


def record_order_placed() -> None:
    orders_placed_total.inc()


def record_order_filled(filled_usd: float) -> None:
    orders_filled_total.inc()


def record_order_cancelled() -> None:
    orders_cancelled_total.inc()


def record_error(error_type: str) -> None:
    errors_total.labels(error_type=error_type).inc()


def set_model_p_up(p: float) -> None:
    model_p_up.set(p)


def set_pnl(pnl_usd: float) -> None:
    pnl_realized_usd.set(pnl_usd)


def set_position(notional_usd: float) -> None:
    position_size_usd.set(notional_usd)


def set_regime(regime_name: str) -> None:
    value = _REGIME_VALUES.get(regime_name.lower(), -1)
    regime_gauge.set(value)
