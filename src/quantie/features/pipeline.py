"""
FeaturePipeline — stateful rolling feature computation.

Call on_binance_trade / on_binance_depth / on_chainlink / on_poly_trade
from the main consumer loop, then call compute() on each tick (≈200ms).

FeatureVector is a frozen dataclass — all fields Optional[float] except the
time/window fields which are always available once window_state has data.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from quantie.data.event_bus import BinanceDepth, BinanceTrade, ChainlinkPrice, PolyTrade
from quantie.features.microstructure import (
    compute_basis,
    compute_depth_imbalance,
    compute_momentum,
    compute_ofi,
    compute_poly_mid_velocity,
    inprogress_swing_score,
    manipulation_setup_score,
    rolling_zscore,
    update_basis_ewma,
)
from quantie.features.time_features import (
    compute_distance_from_strike,
    compute_tau_features,
    day_of_week,
    hour_of_day,
)
from quantie.features.volatility import RealizedVolEstimator
from quantie.state.binance_state import BinanceState
from quantie.state.polymarket_state import PolymarketState
from quantie.state.window_state import WindowState

import numpy as np

# Rolling buffer sizes
_TRADE_BUF = 10_000
_MID_BUF = 5_000
_POLY_MID_BUF = 500
_VOL_HISTORY_BUF = 1_800    # ~30 min at 1 update/sec
_VOLUME_HISTORY_BUF = 1_800
_EPS = 1e-9


@dataclass(frozen=True)
class FeatureVector:
    """Full feature snapshot at one tick. None = insufficient history."""

    # ── Time / window ──────────────────────────────────────────────────────
    tau_seconds: float
    tau_normalized: float
    hour_of_day: int
    day_of_week: int

    # ── Volatility ─────────────────────────────────────────────────────────
    realized_vol_short: Optional[float]         # halflife 15s
    realized_vol_mid: Optional[float]           # halflife 60s
    realized_vol_long: Optional[float]          # halflife 300s
    realized_vol_ratio_short_long: Optional[float]
    realized_vol_60s_zscore: Optional[float]

    # ── Momentum ───────────────────────────────────────────────────────────
    momentum_1s: Optional[float]
    momentum_5s: Optional[float]
    momentum_15s: Optional[float]
    momentum_30s: Optional[float]

    # ── Order flow / depth ─────────────────────────────────────────────────
    ofi_binance_1s: Optional[float]
    ofi_binance_5s: Optional[float]
    ofi_binance_30s: Optional[float]
    depth_imbalance_binance: Optional[float]

    # ── Chainlink-Binance basis ────────────────────────────────────────────
    chainlink_binance_basis: Optional[float]
    chainlink_binance_basis_ewma: Optional[float]

    # ── Polymarket microstructure ──────────────────────────────────────────
    poly_mid: Optional[float]
    poly_spread: Optional[float]
    poly_mid_velocity: Optional[float]
    poly_implied_prob: Optional[float]
    polymarket_extremity: Optional[float]        # |poly_mid - 0.5|
    polymarket_volume_in_window: Optional[float]
    polymarket_one_sided_imbalance: Optional[float]

    # ── Strike distance ────────────────────────────────────────────────────
    distance_from_strike_in_sigma: Optional[float]
    abs_distance_to_strike_in_sigma: Optional[float]

    # ── A9.3 Manipulation setup ────────────────────────────────────────────
    binance_volume_zscore_60s: Optional[float]
    manipulation_setup_score: Optional[float]

    # ── A9.4 In-progress swing ─────────────────────────────────────────────
    binance_aggressive_buy_volume_5s: Optional[float]
    binance_aggressive_sell_volume_5s: Optional[float]
    binance_trade_count_zscore_1s: Optional[float]
    binance_largest_trade_5s: Optional[float]
    btc_log_return_1s: Optional[float]
    btc_log_return_5s: Optional[float]
    inprogress_swing_score: Optional[float]
    inprogress_swing_direction: Optional[str]

    def to_lgbm_row(self) -> dict:
        """Flat dict for LightGBM inference (numeric fields only)."""
        return {
            k: v for k, v in self.__dict__.items()
            if isinstance(v, (int, float, type(None)))
        }


class FeaturePipeline:
    """Stateful feature engine. Update via on_* callbacks; call compute() per tick."""

    def __init__(
        self,
        window_state: WindowState,
        binance_state: BinanceState,
        poly_state: PolymarketState,
    ) -> None:
        self._ws = window_state
        self._bs = binance_state
        self._ps = poly_state

        # Vol estimators
        self._vol_short = RealizedVolEstimator(halflife_seconds=15.0)
        self._vol_mid = RealizedVolEstimator(halflife_seconds=60.0)
        self._vol_long = RealizedVolEstimator(halflife_seconds=300.0)

        # (ts_ms, log_price, qty, is_buy)
        self._trades: deque[tuple[int, float, float, bool]] = deque(maxlen=_TRADE_BUF)
        # (ts_ms, log_mid)
        self._mids: deque[tuple[int, float]] = deque(maxlen=_MID_BUF)
        # (ts_ms, poly_mid)
        self._poly_mids: deque[tuple[int, float]] = deque(maxlen=_POLY_MID_BUF)
        # vol samples for zscore
        self._vol_history: deque[float] = deque(maxlen=_VOL_HISTORY_BUF)
        # per-second trade counts for zscore
        self._trade_counts_1s: deque[float] = deque(maxlen=60)
        self._last_tc_flush_s: int = 0
        self._tc_this_sec: int = 0
        # volume history for zscore
        self._volume_history: deque[float] = deque(maxlen=_VOLUME_HISTORY_BUF)

        # Basis EWMA state
        self._basis_ewma: Optional[float] = None

        # Poly window volume (reset on rollover)
        self._poly_buy_vol_up: float = 0.0
        self._poly_buy_vol_down: float = 0.0
        self._cached_window_ts: int = 0

    # ------------------------------------------------------------------
    # Update callbacks
    # ------------------------------------------------------------------

    def on_binance_trade(self, event: BinanceTrade) -> None:
        lp = _safe_log(event.price)
        if lp is None:
            return
        is_buy = not event.is_buyer_maker
        self._trades.append((event.ts_ms, lp, event.qty, is_buy))
        # flush per-second count bucket
        now_s = event.ts_ms // 1000
        if now_s != self._last_tc_flush_s:
            self._trade_counts_1s.append(float(self._tc_this_sec))
            self._tc_this_sec = 0
            self._last_tc_flush_s = now_s
        self._tc_this_sec += 1

    def on_binance_depth(self, event: BinanceDepth) -> None:
        if not event.bids or not event.asks:
            return
        mid = (event.bids[0][0] + event.asks[0][0]) / 2.0
        lp = _safe_log(mid)
        if lp is None:
            return
        self._mids.append((event.ts_ms, lp))
        vs = self._vol_short.update(lp)
        self._vol_mid.update(lp)
        self._vol_long.update(lp)
        self._vol_history.append(vs)

    def on_chainlink(self, event: ChainlinkPrice) -> None:
        if self._bs.mid_price is not None:
            basis = compute_basis(self._bs.mid_price, event.price)
            self._basis_ewma = update_basis_ewma(basis, self._basis_ewma)

    def on_poly_trade(self, event: PolyTrade) -> None:
        # Reset on window rollover
        if self._ws.window_start_ts != self._cached_window_ts:
            self._poly_buy_vol_up = 0.0
            self._poly_buy_vol_down = 0.0
            self._cached_window_ts = self._ws.window_start_ts

        if event.side == "BUY":
            if event.asset_id == self._ps.up_token_id:
                self._poly_buy_vol_up += event.size
            elif event.asset_id == self._ps.down_token_id:
                self._poly_buy_vol_down += event.size

        up_book = self._ps.up_book
        if up_book is not None:
            self._poly_mids.append((event.ts_ms, up_book.mid))

    @property
    def sigma_per_sec(self) -> float:
        """Per-second vol from 60s EWMA estimator (primary sigma for GBM/model)."""
        return self._vol_mid.vol

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute(self, now_ms: Optional[int] = None) -> FeatureVector:
        now_ms = now_ms or int(time.time() * 1000)
        tau = self._ws.tau

        tau_feats = compute_tau_features(tau)

        # vol
        vs = self._vol_short.vol if self._vol_short.vol > 0 else None
        vm = self._vol_mid.vol if self._vol_mid.vol > 0 else None
        vl = self._vol_long.vol if self._vol_long.vol > 0 else None
        vol_ratio = (vs / vl) if (vs and vl and vl > _EPS) else None
        vol_zscore = (
            rolling_zscore(list(self._vol_history), vs)
            if (vs and len(self._vol_history) >= 2) else None
        )

        # momentum
        mids = list(self._mids)
        mom_1s = compute_momentum(mids, 1_000, now_ms)
        mom_5s = compute_momentum(mids, 5_000, now_ms)
        mom_15s = compute_momentum(mids, 15_000, now_ms)
        mom_30s = compute_momentum(mids, 30_000, now_ms)

        # OFI — extract (ts, qty, is_buy) triples
        tseq = [(t, q, b) for t, _, q, b in self._trades]
        ofi_1s = compute_ofi(tseq, 1_000, now_ms)
        ofi_5s = compute_ofi(tseq, 5_000, now_ms)
        ofi_30s = compute_ofi(tseq, 30_000, now_ms)

        # depth imbalance
        depth_imb: Optional[float] = None
        if self._bs.depth is not None:
            depth_imb = compute_depth_imbalance(
                self._bs.depth.bids, self._bs.depth.asks
            )

        # basis
        raw_basis: Optional[float] = None
        if (self._bs.mid_price is not None
                and self._ws.current_chainlink_price is not None):
            raw_basis = compute_basis(
                self._bs.mid_price, self._ws.current_chainlink_price
            )

        # poly microstructure
        up_book = self._ps.up_book
        poly_mid: Optional[float] = None
        poly_spread: Optional[float] = None
        poly_extremity: Optional[float] = None
        if up_book is not None:
            poly_mid = up_book.mid
            poly_spread = up_book.spread
            poly_extremity = abs(poly_mid - 0.5)

        poly_vel = compute_poly_mid_velocity(
            list(self._poly_mids), window_ms=10_000, now_ms=now_ms
        )

        vol_total = self._poly_buy_vol_up + self._poly_buy_vol_down
        poly_imb: Optional[float] = (
            (self._poly_buy_vol_up - self._poly_buy_vol_down) / vol_total
            if vol_total > _EPS else None
        )

        # strike distance
        dist: Optional[float] = None
        if (self._ws.open_price is not None
                and self._ws.current_chainlink_price is not None
                and vs is not None):
            dist = compute_distance_from_strike(
                self._ws.current_chainlink_price,
                self._ws.open_price, vs, tau,
            )
        abs_dist = abs(dist) if dist is not None else None

        # volume zscore (A9.3)
        recent_vol = sum(q for t, _, q, _ in self._trades if t >= now_ms - 5_000)
        vol_zscore_60s: Optional[float] = (
            rolling_zscore(list(self._volume_history), recent_vol)
            if len(self._volume_history) >= 2 else None
        )
        self._volume_history.append(recent_vol)

        # manipulation setup score
        manip: Optional[float] = None
        if None not in (abs_dist, poly_extremity, vol_zscore, vol_zscore_60s, poly_imb):
            manip = manipulation_setup_score(
                abs_dist, poly_extremity, vol_zscore,  # type: ignore[arg-type]
                vol_zscore_60s, poly_imb,              # type: ignore[arg-type]
            )

        # A9.4 inprogress features
        buy_5s = sum(q for t, _, q, b in self._trades if b and t >= now_ms - 5_000)
        sell_5s = sum(q for t, _, q, b in self._trades if not b and t >= now_ms - 5_000)
        largest_5s = max(
            (q for t, _, q, _ in self._trades if t >= now_ms - 5_000), default=0.0
        )
        tc_zscore: Optional[float] = (
            rolling_zscore(list(self._trade_counts_1s), float(self._tc_this_sec))
            if len(self._trade_counts_1s) >= 2 else None
        )
        ip_score: Optional[float] = None
        ip_dir: Optional[str] = None
        if mom_5s is not None and tc_zscore is not None:
            ip_score, ip_dir = inprogress_swing_score(
                buy_5s, sell_5s, mom_5s, tc_zscore, largest_5s
            )

        return FeatureVector(
            tau_seconds=tau_feats["tau_seconds"],
            tau_normalized=tau_feats["tau_normalized"],
            hour_of_day=hour_of_day(),
            day_of_week=day_of_week(),
            realized_vol_short=vs,
            realized_vol_mid=vm,
            realized_vol_long=vl,
            realized_vol_ratio_short_long=vol_ratio,
            realized_vol_60s_zscore=vol_zscore,
            momentum_1s=mom_1s,
            momentum_5s=mom_5s,
            momentum_15s=mom_15s,
            momentum_30s=mom_30s,
            ofi_binance_1s=ofi_1s,
            ofi_binance_5s=ofi_5s,
            ofi_binance_30s=ofi_30s,
            depth_imbalance_binance=depth_imb,
            chainlink_binance_basis=raw_basis,
            chainlink_binance_basis_ewma=self._basis_ewma,
            poly_mid=poly_mid,
            poly_spread=poly_spread,
            poly_mid_velocity=poly_vel,
            poly_implied_prob=poly_mid,
            polymarket_extremity=poly_extremity,
            polymarket_volume_in_window=vol_total,
            polymarket_one_sided_imbalance=poly_imb,
            distance_from_strike_in_sigma=dist,
            abs_distance_to_strike_in_sigma=abs_dist,
            binance_volume_zscore_60s=vol_zscore_60s,
            manipulation_setup_score=manip,
            binance_aggressive_buy_volume_5s=buy_5s,
            binance_aggressive_sell_volume_5s=sell_5s,
            binance_trade_count_zscore_1s=tc_zscore,
            binance_largest_trade_5s=largest_5s,
            btc_log_return_1s=mom_1s,
            btc_log_return_5s=mom_5s,
            inprogress_swing_score=ip_score,
            inprogress_swing_direction=ip_dir,
        )


def _safe_log(x: float) -> Optional[float]:
    return float(np.log(x)) if x > 0 else None
