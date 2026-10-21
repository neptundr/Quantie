"""
GBM closed-form baseline — P(S_T >= S_0 | S_t, tau).

Closed-form GBM probability. Uses Chainlink price as S_t (not Binance).
Binance is used only for vol estimation (see features/volatility.py).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def gbm_p_up(
    S0: float,
    S_t: float,
    sigma_per_sec: float,
    tau_sec: float,
    drift_per_sec: float = 0.0,
) -> float:
    """P(S_T >= S_0 | S_t, tau) under GBM.

    Args:
        S0:            strike price (window open price, Chainlink)
        S_t:           current Chainlink price
        sigma_per_sec: realized vol per second (from RealizedVolEstimator)
        tau_sec:       seconds remaining until resolution
        drift_per_sec: optional drift (default 0 — risk-neutral)

    Returns:
        Probability in [0, 1] that the window resolves UP.

    At expiry (tau <= 0): returns 1.0 if S_t >= S0, else 0.0.
    Equality counts as UP (Polymarket resolution rule).
    """
    if tau_sec <= 0.0:
        return 1.0 if S_t >= S0 else 0.0

    if sigma_per_sec <= 0.0:
        # zero vol — price won't move
        return 1.0 if S_t >= S0 else 0.0

    log_ratio = np.log(S0 / S_t)
    drift_adj = (drift_per_sec - 0.5 * sigma_per_sec ** 2) * tau_sec
    z = (log_ratio - drift_adj) / (sigma_per_sec * np.sqrt(tau_sec))
    return float(1.0 - norm.cdf(z))
