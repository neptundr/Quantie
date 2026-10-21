"""
Kelly position sizing.

Formula (fractional Kelly):
    f = kelly_fraction × (p_win - (1 - p_win) / b)
    where b = (1 - price) / price  (binary-market odds)

Caps:
  - max position: max_position_frac × bankroll (default 5 %)
  - min order: min_order_usd (default $5); returns 0.0 if below

p_win:  model P(outcome we are betting on) — pass p_up for BUY_UP, (1-p_up) for BUY_DOWN
price:  token ask price (cost per share, ∈ (0, 1))
"""
from __future__ import annotations

from dataclasses import dataclass

_KELLY_FRACTION: float = 0.25
_MAX_POSITION_FRAC: float = 0.15   # 15 % of bankroll per window (~$9 at $60)
_MIN_ORDER_USD: float = 2.0


@dataclass(frozen=True)
class SizingConfig:
    kelly_fraction: float = _KELLY_FRACTION
    max_position_frac: float = _MAX_POSITION_FRAC
    min_order_usd: float = _MIN_ORDER_USD


def kelly_size(
    p_win: float,
    price: float,
    bankroll: float,
    kelly_fraction: float = _KELLY_FRACTION,
    max_position_frac: float = _MAX_POSITION_FRAC,
    min_order_usd: float = _MIN_ORDER_USD,
) -> float:
    return kelly_size_with_config(
        p_win=p_win,
        price=price,
        bankroll=bankroll,
        cfg=SizingConfig(
            kelly_fraction=kelly_fraction,
            max_position_frac=max_position_frac,
            min_order_usd=min_order_usd,
        ),
    )


def kelly_size_with_config(
    p_win: float,
    price: float,
    bankroll: float,
    cfg: SizingConfig,
) -> float:
    """Return position size in USD for a single-side binary market bet.

    Returns 0.0 when:
    - price is outside (0, 1)
    - Kelly formula yields non-positive fraction
    - Sized amount is below min_order_usd

    Args:
        p_win:             P(win) for the token being bought
        price:             token ask price ∈ (0, 1)
        bankroll:          current available bankroll in USD
        kelly_fraction:    fractional Kelly multiplier (default 0.25)
        max_position_frac: bankroll fraction cap (default 0.05 → 5 %)
        min_order_usd:     floor; below this returns 0.0

    Returns:
        Position size in USD, rounded to 2 decimal places.
    """
    if price <= 0.0 or price >= 1.0 or bankroll <= 0.0:
        return 0.0

    # Binary market: win (1-price) per dollar risked, lose price per dollar
    b = (1.0 - price) / price
    edge_fraction = p_win - (1.0 - p_win) / b  # Kelly criterion
    if edge_fraction <= 0.0:
        return 0.0

    fraction = cfg.kelly_fraction * edge_fraction
    raw_usd = fraction * bankroll
    capped_usd = min(raw_usd, cfg.max_position_frac * bankroll)

    if capped_usd < cfg.min_order_usd:
        return 0.0

    return round(capped_usd, 2)
