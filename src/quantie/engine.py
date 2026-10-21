"""
TradingEngine — async orchestrator tying all layers together.

Data flow per tick:
  EventBus → FeaturePipeline → EnsembleModel → make_decision
    → OrderExecutor (pre_trade_check → Broker)

Window lifecycle:
  On rollover: cancel_all_open → reset_window → resubscribe CLOB WS
               → update poly token IDs → retrain model if enough data

Paper mode:
  Pass paper=True (default) to use PaperBroker.
  Live mode requires paper=False and all POLY_* env vars set.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

from quantie.data.event_bus import (
    BinanceDepth,
    BinanceTrade,
    ChainlinkPrice,
    EventBus,
    PolyBookUpdate,
    PolyPriceChange,
    PolyTrade,
)
from quantie.data.decision_store import DecisionStore
from quantie.data.tick_store import TickStore
from quantie.execution.broker import Broker, STATUS_FILLED
from quantie.execution.order_executor import OrderExecutor
from quantie.execution.paper_broker import PaperBroker
from quantie.features.pipeline import FeaturePipeline
from quantie.models.forecaster import Forecaster
from quantie.models.ensemble import EnsembleModel
from quantie.policy.bandit import LinUCBBandit
from quantie.policy.decision import Action, PolicyDecision, should_hold_to_resolution
from quantie.policy.edge import EdgeConfig
from quantie.policy.policy_layer import BanditFeedback, ExecutionPolicy, PolicyContext
from quantie.policy.regime import Regime, RegimeConfig, classify_regime_with_config
from quantie.policy.sizing import SizingConfig
from quantie.pnl import PnLTracker
from quantie.safety.audit_log import log_decision
from quantie.safety.checks import SafetySnapshot, pre_trade_check
from quantie.safety.circuit_breakers import CircuitBreaker
from quantie.state.binance_state import BinanceState
from quantie.state.polymarket_state import PolymarketState
from quantie.state.window_state import CompletedWindow, WindowState
from quantie.config import AppConfig

log = structlog.get_logger(__name__)

_TICK_INTERVAL: float = 1.0
_FILE_KILL_SWITCH = Path("/tmp/bot_kill")
_BANDIT_STATE_FILE = Path("models/v1/bandit_state.npz")
_SLOW_FORECAST_MS = 20.0

_DEFAULT_WINDOW_RISK_FRACTION: float = 0.20
_EXTREME_MKT_LO: float = 0.10
_EXTREME_MKT_HI: float = 0.90
_HISTORY_MAX: int = 3
_R_WIDTH: int = 42


@dataclass
class EngineConfig:
    bankroll: float = 1000.0
    max_position_per_window_usd: float = 50.0
    stale_chainlink_ms: float = 2_500.0
    stale_binance_ms: float = 1_500.0
    paper: bool = True
    model_dir: Optional[Path] = None
    verbose: bool = False
    app_config: Optional[AppConfig] = None
    taker_fee_rate: float = 0.018


class TradingEngine:
    """Main async engine. Create one instance per process."""

    def __init__(
        self,
        event_bus: EventBus,
        config: EngineConfig | None = None,
        broker: Broker | None = None,
    ) -> None:
        self._bus = event_bus
        self._cfg = config or EngineConfig()
        self._app_cfg = self._cfg.app_config
        regime_cfg = (
            RegimeConfig(
                early_to_mid_seconds=float(self._app_cfg.regime_thresholds.early_to_mid_seconds),
                mid_to_late_seconds=float(self._app_cfg.regime_thresholds.mid_to_late_seconds),
                late_to_terminal_seconds=float(self._app_cfg.regime_thresholds.late_to_terminal_seconds),
            )
            if self._app_cfg
            else RegimeConfig()
        )
        edge_cfg = (
            EdgeConfig(
                base_threshold=float(self._app_cfg.policy.edge_threshold_base),
                ci_multiplier=float(self._app_cfg.policy.edge_threshold_ci_multiplier),
                fee_buffer=float(self._app_cfg.policy.fee_buffer),
            )
            if self._app_cfg
            else EdgeConfig()
        )
        sizing_cfg = (
            SizingConfig(
                kelly_fraction=float(self._app_cfg.sizing.kelly_fraction),
                max_position_frac=float(self._app_cfg.sizing.max_position_frac_of_bankroll),
                min_order_usd=float(self._app_cfg.sizing.min_order_size_usd),
            )
            if self._app_cfg
            else SizingConfig()
        )
        self._regime_cfg = regime_cfg
        self._edge_cfg = edge_cfg
        self._sizing_cfg = sizing_cfg
        self._cancel_gtc_before_terminal = (
            float(self._app_cfg.execution.cancel_all_on_window_close_seconds)
            if self._app_cfg
            else 35.0
        )
        self._window_risk_fraction = (
            float(self._app_cfg.capital.max_position_per_window_usd)
            / max(float(self._app_cfg.capital.bankroll_usd), 1.0)
            if self._app_cfg
            else _DEFAULT_WINDOW_RISK_FRACTION
        )
        self._extreme_market_lo = _EXTREME_MKT_LO
        self._extreme_market_hi = _EXTREME_MKT_HI

        # State layers
        self._binance = BinanceState()
        self._poly = PolymarketState()
        self._window = WindowState()

        # ML pipeline
        self._features = FeaturePipeline(self._window, self._binance, self._poly)
        self._model: Forecaster = EnsembleModel()
        if self._cfg.model_dir and Path(self._cfg.model_dir).exists():
            cast_model = self._model
            if isinstance(cast_model, EnsembleModel):
                cast_model.load(self._cfg.model_dir)

        # Execution
        if broker is None:
            broker = PaperBroker() if self._cfg.paper else _build_live_broker()
        self._executor = OrderExecutor(broker)
        self._policy = ExecutionPolicy(
            edge_cfg=self._edge_cfg,
            sizing_cfg=self._sizing_cfg,
            regime_cfg=self._regime_cfg,
            bandit=LinUCBBandit(dim=17),
            bandit_state_path=_BANDIT_STATE_FILE,
        )

        # Phase 7: safety + data collection
        drawdown_frac = (
            float(self._app_cfg.capital.max_total_drawdown_pct)
            if self._app_cfg
            else 0.40
        )
        self._circuit_breaker = CircuitBreaker(max_drawdown_frac=drawdown_frac)
        self._decision_store = DecisionStore()
        self._tick_store = TickStore()
        self._pnl = PnLTracker(self._cfg.bankroll)
        self._taker_fee_rate: float = (
            float(self._app_cfg.execution.taker_fee_rate)
            if self._app_cfg
            else self._cfg.taker_fee_rate
        )
        self._window.add_completed_callback(self._on_window_completed)

        # Staleness tracking
        self._last_chainlink_ms: float = 0.0
        self._last_binance_ms: float = 0.0

        # Current window token IDs (set by set_token_ids)
        self._up_asset_id: str = ""
        self._down_asset_id: str = ""

        # Dynamic per-window risk baseline (refreshed on each window roll)
        self._window_start_balance_usd: float = self._cfg.bankroll
        self._window_risk_cap_usd: float = self._cfg.bankroll * self._window_risk_fraction

        # Session-level hard stop reference: set once at startup, never updated
        self._session_start_balance_usd: float = self._cfg.bankroll

        # Per-window PnL: snapshot of realized_pnl at window start, used for window-specific delta
        self._window_pnl_start: float = 0.0

        # Bug #7: track fills already attempted for early exit (avoid retrying on every tick)
        self._exit_attempted_order_ids: set[str] = set()

        # Anti-oscillation: after early exit, block re-entry for that direction for 60s.
        # key = asset_id, value = monotonic time.time() when cooldown expires.
        self._early_exit_cooldown_until: dict[str, float] = {}
        self._bandit_pending_by_order: dict[str, tuple[Action, object, str]] = {}
        self._history: deque[tuple[str, str, str]] = deque(maxlen=_HISTORY_MAX)

        self._running = False
        self._notifier = None  # set by set_notifier() before run()
        self._last_cb_kind: str = ""  # track last reported circuit breaker kind

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_notifier(self, notifier) -> None:
        self._notifier = notifier

    def get_status_snapshot(self) -> dict:
        """Called by Telegram /status handler to get a point-in-time state snapshot."""
        open_fills = self._pnl.open_fills()
        open_exposure = sum(f.filled_usd for f in open_fills)
        balance_cash = self._cfg.bankroll + self._pnl.realized_pnl - open_exposure
        balance_total = self._estimate_total_equity()
        cb = self._circuit_breaker
        if cb.hard_stop_pending:
            status = "hard stop pending"
        elif cb.is_halted:
            status = f"halted [{cb.halt_kind}] — resume in {cb.seconds_until_resume:.0f}s"
        elif not self._running:
            status = "stopped"
        else:
            status = "running"
        pending = self._pnl.pending_fills()
        return {
            "mode": "PAPER" if self._cfg.paper else "LIVE",
            "bankroll": self._cfg.bankroll,
            "balance_total": balance_total,
            "balance_cash": balance_cash,
            "realized_pnl": self._pnl.realized_pnl,
            "window_pnl": self._pnl.realized_pnl - self._window_pnl_start,
            "pending_resolution_count": len(pending),
            "pending_resolution_usd": sum(f.filled_usd for f in pending),
            "status": status,
        }

    async def run(self) -> None:
        """Start engine. Runs until stop() is called or task is cancelled."""
        self._running = True
        _mode = "PAPER" if self._cfg.paper else "LIVE"
        log.info("engine.start", paper=self._cfg.paper, bankroll=self._cfg.bankroll)
        if self._cfg.verbose:
            _sep = "═" * 60
            print(
                f"\n{_sep}\n"
                f"  QUANTIE BOT  —  {_mode} MODE\n"
                f"  Bankroll   : ${self._cfg.bankroll:.2f}\n"
                f"  Model      : {self._cfg.model_dir or 'GBM-only (untrained)'}\n"
                f"  Max/window : ${self._cfg.bankroll * self._window_risk_fraction:.2f} "
                f"({self._window_risk_fraction*100:.1f}% equity)\n"
                f"  Starting up — waiting for first window...\n"
                f"{_sep}\n",
                flush=True,
            )
        if self._notifier:
            await self._notifier.notify_startup(
                mode=_mode,
                bankroll=self._cfg.bankroll,
                model_dir=str(self._cfg.model_dir or "GBM-only"),
            )
        try:
            await asyncio.gather(
                self._consume_events(),
                self._decision_ticker(),
                self._hourly_pnl_task(),
            )
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            await self._executor.cancel_all_open()
            self._policy.persist()
            self._decision_store.close()
            log.info("engine.stopped")
            if self._notifier:
                await self._notifier.notify_shutdown(
                    mode=_mode,
                    realized_pnl=self._pnl.realized_pnl,
                )
            if self._cfg.verbose:
                print(
                    f"\n{'═' * 60}\n"
                    f"  QUANTIE STOPPED\n"
                    f"  Final PnL  : {self._pnl.realized_pnl:+.2f}\n"
                    f"{'═' * 60}\n",
                    flush=True,
                )

    def stop(self) -> None:
        self._running = False

    async def _hourly_pnl_task(self) -> None:
        _mode = "PAPER" if self._cfg.paper else "LIVE"
        while self._running:
            await asyncio.sleep(3600)
            if self._notifier and self._running:
                await self._notifier.notify_pnl(
                    realized_pnl=self._pnl.realized_pnl,
                    mode=_mode,
                    bankroll=self._cfg.bankroll,
                )

    @property
    def window_state(self) -> WindowState:
        return self._window

    @property
    def poly_state(self) -> PolymarketState:
        return self._poly

    def set_token_ids(self, up_asset_id: str, down_asset_id: str) -> None:
        """Called by window_manager after resolving tokens for new window."""
        self._policy.persist()
        self._up_asset_id = up_asset_id
        self._down_asset_id = down_asset_id
        self._refresh_window_risk_baseline()
        self._executor.reset_window()
        self._exit_attempted_order_ids = set()  # reset per-window exit tracking
        self._early_exit_cooldown_until = {}    # reset direction cooldowns on new window
        self._window_pnl_start = self._pnl.realized_pnl  # snapshot for per-window delta
        log.info(
            "engine.new_window",
            up=up_asset_id,
            down=down_asset_id,
            window_start_balance_usd=round(self._window_start_balance_usd, 2),
            window_risk_cap_usd=round(self._window_risk_cap_usd, 2),
        )
    # ------------------------------------------------------------------
    # Window completion callback
    # ------------------------------------------------------------------

    def _on_window_completed(self, window: CompletedWindow) -> None:
        """Flush tick parquet and schedule Gamma-verified PnL resolution.

        Asset IDs are captured now because set_token_ids() will overwrite them
        once the rollover callbacks fire (immediately after this callback).
        PnL is NOT recorded here — _apply_resolution does it after Gamma confirms.

        open_fills() from the old window are immediately marked pending_resolution
        so the new window starts with zero inherited exposure.
        """
        self._tick_store.flush(window.window_start_ts, window.outcome)

        # Move old-window positions out of open_fills() before new window starts.
        marked = self._pnl.mark_pending_resolution()
        if marked:
            log.info("window.positions_pending_resolution", count=marked)

        prev_up = self._up_asset_id
        prev_down = self._down_asset_id

        if prev_up and prev_down:
            asyncio.ensure_future(
                self._resolve_window_pnl(prev_up, prev_down, window)
            )
        elif window.outcome is not None:
            # Startup edge case: tokens not yet known, use self-computed outcome.
            asyncio.ensure_future(
                self._apply_resolution(prev_up, prev_down, window.outcome, window)
            )

    async def _resolve_window_pnl(
        self,
        up_asset_id: str,
        down_asset_id: str,
        window: CompletedWindow,
    ) -> None:
        """Poll Gamma API for the verified outcome, then record PnL.

        Falls back to self-computed window.outcome after 120 s if Gamma
        doesn't confirm resolution in time.
        """
        from quantie.data.gamma_client import GammaAPIError, get_market_by_slug
        from quantie.clock import make_slug

        slug = make_slug(window.window_start_ts)
        deadline = time.time() + 120.0

        while time.time() < deadline:
            await asyncio.sleep(10)
            try:
                market = await asyncio.get_event_loop().run_in_executor(
                    None, get_market_by_slug, slug
                )
                gamma_outcome = _extract_gamma_outcome(market)
                if gamma_outcome is not None:
                    if gamma_outcome != window.outcome:
                        self._warn(
                            "resolution.mismatch",
                            slug=slug,
                            bot_outcome=window.outcome,
                            gamma_outcome=gamma_outcome,
                        )
                    window.outcome = gamma_outcome
                    await self._apply_resolution(up_asset_id, down_asset_id, gamma_outcome, window)
                    return
            except (GammaAPIError, Exception) as exc:
                log.debug("gamma.resolution_poll.failed", slug=slug, error=str(exc))

        # 120 s timeout — fall back to self-computed outcome
        fallback = window.outcome
        self._warn("gamma.resolution_timeout", slug=slug, fallback=fallback)
        if fallback is not None:
            await self._apply_resolution(up_asset_id, down_asset_id, fallback, window)
        else:
            self._err("gamma.resolution_unknown", slug=slug)

    async def _apply_resolution(
        self,
        up_asset_id: str,
        down_asset_id: str,
        outcome: str,
        window: CompletedWindow,
    ) -> None:
        """Record PnL for a resolved window and emit verbose output."""
        import datetime as _dt
        won_up = outcome == "UP"

        # Capture pending fills per asset before resolution so we can emit per-fill history.
        _pending_up = [f for f in self._pnl.pending_fills() if f.asset_id == up_asset_id]
        _pending_down = [f for f in self._pnl.pending_fills() if f.asset_id == down_asset_id]

        delta_up = self._pnl.record_resolution(up_asset_id, won=won_up)
        delta_down = self._pnl.record_resolution(down_asset_id, won=not won_up)
        if not self._circuit_breaker.is_halted:
            up_orders = [
                (oid, arm, vec)
                for oid, (arm, vec, asset) in self._bandit_pending_by_order.items()
                if asset == up_asset_id
            ]
            down_orders = [
                (oid, arm, vec)
                for oid, (arm, vec, asset) in self._bandit_pending_by_order.items()
                if asset == down_asset_id
            ]
            up_reward = (delta_up / len(up_orders)) if up_orders else 0.0
            down_reward = (delta_down / len(down_orders)) if down_orders else 0.0
            for oid, arm, vec in up_orders:
                self._policy.update(BanditFeedback(action=arm, reward=float(up_reward), context_vec=vec))
                self._bandit_pending_by_order.pop(oid, None)
            for oid, arm, vec in down_orders:
                self._policy.update(BanditFeedback(action=arm, reward=float(down_reward), context_vec=vec))
                self._bandit_pending_by_order.pop(oid, None)
        window_pnl = delta_up + delta_down
        self._circuit_breaker.record_window_pnl(window_pnl)
        log.info(
            "window.pnl_resolved",
            window_start_ts=window.window_start_ts,
            outcome=outcome,
            pnl_delta_usd=round(window_pnl, 4),
            cumulative_pnl_usd=round(self._pnl.realized_pnl, 4),
        )
        _ts_h = _dt.datetime.now(_dt.timezone.utc).strftime("%H:%M:%S")
        _outcome_str = "⬆️  UP won" if won_up else "⬇️  DOWN won"

        # Per-fill RESOLVED entries (positions held to resolution, not early-exited).
        for _f in _pending_up + _pending_down:
            _f_won = won_up if _f.asset_id == up_asset_id else not won_up
            _f_dir = "UP  ▲" if _f.asset_id == up_asset_id else "DOWN ▼"
            _payout_usd = _f.shares if _f_won else 0.0
            self._add_history(
                f"  ✦ RESOLVED {_f_dir}  [{_ts_h} UTC]",
                f"  ${_f.filled_usd:.2f} → ${_payout_usd:.2f}  PnL {_f.pnl_usd:+.2f}",
                f"  @ {_f.fill_price:.4f} → {'1.0000' if _f_won else '0.0000'}/sh",
            )

        self._add_history(
            f"  ═ WINDOW  {_outcome_str}  [{_ts_h} UTC]",
            f"  delta {window_pnl:+.2f}  (up {delta_up:+.2f}  dn {delta_down:+.2f})",
            f"  total PnL: {self._pnl.realized_pnl:+.2f}",
        )

        # Hard stop: session loss limit was hit and all positions are now closed → shut down
        if self._circuit_breaker.hard_stop_pending and self._pnl.realized_pnl <= -(
            0.20 * self._session_start_balance_usd
        ):
            log.warning(
                "engine.hard_stop",
                realized_pnl=round(self._pnl.realized_pnl, 2),
                session_start_balance=round(self._session_start_balance_usd, 2),
            )
            if self._notifier:
                await self._notifier.notify_hard_stop(
                    realized_pnl=self._pnl.realized_pnl,
                    session_start=self._session_start_balance_usd,
                )
            if self._cfg.verbose:
                print(
                    f"\n{'═' * 60}\n"
                    f"  HARD STOP — session loss limit reached\n"
                    f"  Realized PnL : {self._pnl.realized_pnl:+.2f}\n"
                    f"  Limit        : -{0.20 * self._session_start_balance_usd:.2f} "
                    f"(20% of ${self._session_start_balance_usd:.2f})\n"
                    f"  Bot shutting down.\n"
                    f"{'═' * 60}\n",
                    flush=True,
                )
            import sys
            sys.exit(1)

    # ------------------------------------------------------------------
    # Event consumption
    # ------------------------------------------------------------------

    async def _consume_events(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._bus.consume(), timeout=_TICK_INTERVAL)
            except asyncio.TimeoutError:
                continue

            now_ms = int(time.time() * 1000)

            if isinstance(event, BinanceTrade):
                self._binance.on_trade(event)
                self._features.on_binance_trade(event)
                self._last_binance_ms = float(now_ms)
                self._circuit_breaker.record_binance_latency(float(now_ms - event.ts_ms))

            elif isinstance(event, BinanceDepth):
                self._binance.on_depth(event)
                self._features.on_binance_depth(event)

            elif isinstance(event, ChainlinkPrice):
                self._window.on_chainlink(event)
                self._features.on_chainlink(event)
                self._last_chainlink_ms = float(now_ms)

            elif isinstance(event, PolyBookUpdate):
                self._poly.on_book(event)
                self._tick_paper_fills(event)

            elif isinstance(event, PolyPriceChange):
                self._poly.on_price_change(event)

            elif isinstance(event, PolyTrade):
                self._poly.on_trade(event)
                self._features.on_poly_trade(event)

    def _tick_paper_fills(self, event: PolyBookUpdate) -> None:
        broker = self._executor._broker
        if not isinstance(broker, PaperBroker):
            return
        best_ask = event.asks[0][0] if event.asks else 1.0
        for fill in broker.tick_book(event.asset_id, best_ask):
            self._executor._window_exposure_usd += fill.filled_usd
            self._pnl.record_fill(
                order_id=fill.order_id,
                asset_id=fill.asset_id,
                fill_price=fill.price,
                filled_usd=fill.filled_usd,
                fee_usd=fill.filled_usd * self._taker_fee_rate,
            )
            log.info("paper.gtc.filled", order_id=fill.order_id, filled_usd=fill.filled_usd)

    # ------------------------------------------------------------------
    # Decision ticker
    # ------------------------------------------------------------------

    async def _decision_ticker(self) -> None:
        while self._running:
            await asyncio.sleep(_TICK_INTERVAL)
            try:
                await self._maybe_decide()
            except Exception as exc:
                self._err("engine.tick.error", error=str(exc))

    async def _maybe_decide(self) -> None:
        now_ms = int(time.time() * 1000)
        tau = self._window.tau

        kill_switch = _FILE_KILL_SWITCH.exists()
        if kill_switch:
            self._warn("engine.kill_switch.file_detected", path=str(_FILE_KILL_SWITCH))

        if tau <= self._cancel_gtc_before_terminal:
            await self._executor.cancel_all_open()

        up_book = self._poly.up_book
        down_book = self._poly.down_book
        if up_book is None or down_book is None:
            return
        if up_book.best_ask <= 0.0 or up_book.best_bid <= 0.0:
            return

        up_ask = up_book.best_ask
        up_bid = up_book.best_bid
        up_mid = (up_ask + up_bid) / 2.0
        _extreme_market_blocked = up_mid > self._extreme_market_hi or up_mid < self._extreme_market_lo

        s_t = self._window.current_chainlink_price
        if s_t is None:
            return

        s_0 = self._window.open_price
        if s_0 is None:
            return

        sigma = self._features.sigma_per_sec
        feature_row = self._features.compute(now_ms)
        row_dict = feature_row.to_lgbm_row() if hasattr(feature_row, "to_lgbm_row") else {}

        _predict_start = time.perf_counter()
        out = self._model.predict(
            feature_row=row_dict,
            S0=s_0,
            S_t=s_t,
            sigma_per_sec=sigma,
            tau_sec=tau,
        )
        _predict_ms = (time.perf_counter() - _predict_start) * 1000.0
        if _predict_ms > _SLOW_FORECAST_MS:
            self._warn("model.predict.slow", elapsed_ms=round(_predict_ms, 2))

        # Early exit: two conditions trigger a sell:
        #
        #   A) Profitable sell: net proceeds after sell fee exceed total round-trip cost.
        #      Breakeven bid = fill_price × (1 + fee) / (1 - fee) ≈ fill_price × 1.037
        #      at 1.8% fee. Must clear this bar to actually profit from exiting early.
        #
        #   B) Clear loser: p_win < 0.40 → position likely pays $0; cut losses now.
        #      Even if bid is below cost, recovering something beats losing everything.
        #
        # NEVER exit only because bid is slightly above fill_price — fees make that a loss.
        # Skip in TERMINAL — market resolves in seconds, spread too wide to sell.
        _PAPER_SLIPPAGE = 2 / 10_000
        _CLEAR_LOSER_THRESHOLD = 0.30
        _fee = self._taker_fee_rate
        regime = classify_regime_with_config(tau, self._regime_cfg)
        if regime != Regime.TERMINAL and self._up_asset_id:
            open_fills = self._pnl.open_fills()
            if open_fills:
                exits_needed = []
                for fill in open_fills:
                    if fill.order_id in self._exit_attempted_order_ids:
                        continue
                    if fill.asset_id == self._up_asset_id:
                        p_win = out.p_up
                        fill_bid = up_bid
                    elif fill.asset_id == self._down_asset_id:
                        p_win = 1.0 - out.p_up
                        _down_book = self._poly.down_book
                        fill_bid = (
                            _down_book.best_bid
                            if (_down_book is not None and _down_book.best_bid > 0.0)
                            else 1.0 - up_ask
                        )
                    else:
                        continue  # unknown asset, skip
                    # Net proceeds per share after sell-side taker fee + slippage
                    net_bid = fill_bid * (1 - _fee) * (1 - _PAPER_SLIPPAGE)
                    # Breakeven: must recover buy-side fee too (fill_price × (1+fee))
                    breakeven = fill.fill_price * (1 + _fee)
                    profitable_sell = net_bid >= breakeven
                    clear_loser = p_win < _CLEAR_LOSER_THRESHOLD  # likely total loss
                    if profitable_sell or clear_loser:
                        exits_needed.append(fill)
                if exits_needed:
                    _reasons = []
                    for _f in exits_needed:
                        _pw = out.p_up if _f.asset_id == self._up_asset_id else 1.0 - out.p_up
                        _b = up_bid if _f.asset_id == self._up_asset_id else 1.0 - up_ask
                        _nb = _b * (1 - _fee) * (1 - _PAPER_SLIPPAGE)
                        _be = _f.fill_price * (1 + _fee)
                        if _nb >= _be:
                            _reasons.append(f"profitable net_bid={_nb:.3f}≥breakeven={_be:.3f}")
                        else:
                            _reasons.append(f"clear_loser p_win={_pw:.3f}")
                    _exit_decision_reason = "; ".join(_reasons)
                    log_decision(
                        ts_ms=now_ms,
                        window_start_ts=self._window.window_start_ts,
                        tau_sec=tau,
                        feature_row=row_dict,
                        p_up=out.p_up,
                        gbm_p_up=out.gbm_p_up,
                        ml_residual=out.ml_residual,
                        ci_width=out.ci_width,
                        action="EARLY_EXIT",
                        size_usd=0.0,
                        edge=0.0,
                        regime=str(regime),
                        reason=_exit_decision_reason,
                    )
                    self._decision_store.record(
                        ts_ms=now_ms,
                        window_start_ts=self._window.window_start_ts,
                        tau_sec=tau,
                        p_up=out.p_up,
                        gbm_p_up=out.gbm_p_up,
                        ml_residual=out.ml_residual,
                        ci_width=out.ci_width,
                        action="EARLY_EXIT",
                        size_usd=0.0,
                        edge=0.0,
                        regime=str(regime),
                        reason=_exit_decision_reason,
                        balance_total=self._cfg.bankroll + self._pnl.realized_pnl,
                        balance_cash=self._cfg.bankroll + self._pnl.realized_pnl
                            - self._pnl.total_open_exposure(),
                    )
                    await self._do_early_exit(exits_needed, up_ask, up_bid, now_ms)
                    return

        chainlink_age = float(now_ms - self._last_chainlink_ms) if self._last_chainlink_ms else 9999.0
        binance_age = float(now_ms - self._last_binance_ms) if self._last_binance_ms else 9999.0

        self._circuit_breaker.check_drawdown(
            self._pnl.realized_pnl, self._window_start_balance_usd
        )
        self._circuit_breaker.check_session_loss(
            self._pnl.realized_pnl, self._session_start_balance_usd
        )
        await self._notify_circuit_breaker_if_new()

        poly_spread = max(0.0, up_ask - up_bid)
        policy_ctx = PolicyContext(
            now_ms=now_ms,
            tau_sec=tau,
            p_up=out.p_up,
            ci_width=out.ci_width,
            gbm_p_up=out.gbm_p_up,
            ml_residual=out.ml_residual,
            up_ask=up_ask,
            up_bid=up_bid,
            up_mid=up_mid,
            poly_spread=poly_spread,
            poly_extremity=abs(up_mid - 0.5),
            poly_imbalance=float(row_dict.get("polymarket_one_sided_imbalance") or 0.0),
            ofi_5s=float(row_dict.get("ofi_binance_5s") or 0.0),
            momentum_5s=float(row_dict.get("momentum_5s") or 0.0),
            depth_imbalance=float(row_dict.get("depth_imbalance_binance") or 0.0),
            binance_age_ms=binance_age,
            chainlink_age_ms=chainlink_age,
            bankroll=self._cfg.bankroll,
            is_live=not self._cfg.paper,
            stale_chainlink_ms=self._cfg.stale_chainlink_ms,
            stale_binance_ms=self._cfg.stale_binance_ms,
            extreme_mkt_lo=self._extreme_market_lo,
            extreme_mkt_hi=self._extreme_market_hi,
            circuit_breaker=self._circuit_breaker,
        )
        decision, bandit_vec = self._policy.choose(policy_ctx)

        safety = SafetySnapshot(
            kill_switch=kill_switch,
            realized_pnl_usd=self._pnl.realized_pnl,
            starting_bankroll=self._window_start_balance_usd,
            chainlink_age_ms=chainlink_age,
            binance_age_ms=binance_age,
            window_exposure_usd=self._executor.window_exposure_usd,
            max_position_usd=self._window_risk_cap_usd,
            tau_sec=tau,
            circuit_breaker_halted=self._circuit_breaker.is_halted,
        )

        # Pick the edge value relevant to the chosen direction so the logged
        # value matches the reason string (positive when we have a signal).
        action_edge = _edge_for_action(decision)

        log_decision(
            ts_ms=now_ms,
            window_start_ts=self._window.window_start_ts,
            tau_sec=tau,
            feature_row=row_dict,
            p_up=out.p_up,
            gbm_p_up=out.gbm_p_up,
            ml_residual=out.ml_residual,
            ci_width=out.ci_width,
            action=str(decision.action),
            size_usd=decision.size_usd,
            edge=action_edge,
            regime=str(decision.regime),
            reason=decision.reason,
        )

        # ---- balance snapshot (shared by verbose print and decision log) ----
        open_fills = self._pnl.open_fills()
        open_exposure = sum(f.filled_usd for f in open_fills)
        down_mid = 1.0 - up_mid
        market_value = 0.0
        for _f in open_fills:
            if _f.asset_id == self._up_asset_id:
                market_value += _f.shares * up_mid
            elif _f.asset_id == self._down_asset_id:
                market_value += _f.shares * down_mid
        balance_cash = self._cfg.bankroll + self._pnl.realized_pnl - open_exposure
        balance_total = balance_cash + market_value

        self._decision_store.record(
            ts_ms=now_ms,
            window_start_ts=self._window.window_start_ts,
            tau_sec=tau,
            p_up=out.p_up,
            gbm_p_up=out.gbm_p_up,
            ml_residual=out.ml_residual,
            ci_width=out.ci_width,
            action=str(decision.action),
            size_usd=decision.size_usd,
            edge=action_edge,
            regime=str(decision.regime),
            reason=decision.reason,
            balance_total=balance_total,
            balance_cash=balance_cash,
        )

        self._tick_store.record(
            ts_ms=now_ms,
            window_start_ts=self._window.window_start_ts,
            feature_row=row_dict,
            p_up=out.p_up,
            gbm_p_up=out.gbm_p_up,
            ml_residual=out.ml_residual,
            ci_width=out.ci_width,
            action=str(decision.action),
            regime=str(decision.regime),
        )

        # Anti-oscillation cooldown: after an early exit, block re-entry for that direction.
        _trade_asset = None
        if decision.action in (Action.TAKE_UP, Action.MAKE_UP):
            _trade_asset = self._up_asset_id
        elif decision.action in (Action.TAKE_DOWN, Action.MAKE_DOWN):
            _trade_asset = self._down_asset_id
        _in_cooldown = (
            _trade_asset is not None
            and time.time() < self._early_exit_cooldown_until.get(_trade_asset, 0.0)
        )

        # In live mode, GTC fills are not tracked (no PolyOurOrderUpdate handler).
        # Block MAKE_* in live to prevent silent exposure accumulation.
        _is_passive = decision.action in (Action.MAKE_UP, Action.MAKE_DOWN)
        _live_passive_blocked = _is_passive and not self._cfg.paper

        # One-fill-per-window: once any fill exists for a direction this window,
        # block further entries. Prevents double-fill bug where sequential FOK/GTC
        # orders both fill before window_exposure_usd is re-checked.
        _is_trade = decision.action in (Action.TAKE_UP, Action.MAKE_UP, Action.TAKE_DOWN, Action.MAKE_DOWN)
        _window_fill_blocked = (
            _is_trade
            and _trade_asset is not None
            and any(f.asset_id == _trade_asset for f in self._pnl.open_fills())
        )

        if _extreme_market_blocked:
            result = None
            log.debug(
                "engine.extreme_market.blocked",
                up_mid=round(up_mid, 4),
                lo=self._extreme_market_lo,
                hi=self._extreme_market_hi,
            )
        elif _in_cooldown:
            result = None
            log.debug("order.cooldown_blocked", asset_id=_trade_asset, action=str(decision.action))
        elif _live_passive_blocked:
            result = None
            log.debug("live.passive_blocked", action=str(decision.action),
                      reason="GTC fills untracked in live mode")
        elif _window_fill_blocked:
            result = None
            log.debug("engine.one_fill_limit.blocked", asset_id=_trade_asset, action=str(decision.action))
        else:
            result = await self._executor.execute(
                decision=decision,
                safety=safety,
                up_ask=up_ask,
                up_bid=up_bid,
                up_asset_id=self._up_asset_id,
                down_asset_id=self._down_asset_id,
            )

        if result is not None:
            self._circuit_breaker.record_order_status(result.status)
            if result.status == STATUS_FILLED and result.filled_usd > 0:
                self._pnl.record_fill(
                    order_id=result.order_id,
                    asset_id=result.asset_id or "",
                    fill_price=result.price,
                    filled_usd=result.filled_usd,
                    fee_usd=result.filled_usd * self._taker_fee_rate,
                )
                if decision.action in (
                    Action.TAKE_UP,
                    Action.TAKE_DOWN,
                    Action.MAKE_UP,
                    Action.MAKE_DOWN,
                ):
                    self._bandit_pending_by_order[result.order_id] = (
                        decision.action,
                        bandit_vec,
                        result.asset_id or "",
                    )
                import datetime as _dt
                _fts = _dt.datetime.fromtimestamp(now_ms / 1000, tz=_dt.timezone.utc).strftime("%H:%M:%S")
                _dir = "BUY UP  ▲" if result.asset_id == self._up_asset_id else "BUY DOWN ▼"
                _shares = result.filled_usd / result.price if result.price > 0 else 0.0
                self._add_history(
                    f"  ★ {_dir}  [{_fts} UTC]",
                    f"  ${result.filled_usd:.2f} @ {result.price:.4f}/sh  ({_shares:.2f} sh)",
                    f"  order: {result.order_id[:20]}",
                )

        if self._cfg.verbose:
            # Re-query fills AFTER execute so current fill is included.
            import datetime as _dt
            _ts = _dt.datetime.fromtimestamp(now_ms / 1000, tz=_dt.timezone.utc).strftime("%H:%M:%S")
            _sep = "─" * 116
            _sepBold = "═" * 116
            _regime_label = str(decision.regime).replace("Regime.", "")
            _action_label = str(decision.action).replace("Action.", "")

            post_fills = self._pnl.open_fills()
            post_exposure = sum(f.filled_usd for f in post_fills)
            post_mval = 0.0
            for _f in post_fills:
                if _f.asset_id == self._up_asset_id:
                    post_mval += _f.shares * up_mid
                elif _f.asset_id == self._down_asset_id:
                    post_mval += _f.shares * down_mid
            post_cash = self._cfg.bankroll + self._pnl.realized_pnl - post_exposure
            post_total = post_cash + post_mval

            # Resting GTC orders (MAKE_* — placed but not yet filled)
            _open_orders: dict = getattr(self._executor._broker, "open_orders", {})
            _resting = list(_open_orders.values())

            _label  = "  Positions : "
            _indent = " " * len(_label)
            _pos_lines = []
            _line_idx = 0

            for _f in post_fills:
                _dir  = "BUY_UP " if _f.asset_id == self._up_asset_id else "BUY_DOWN"
                _mid  = up_mid if _f.asset_id == self._up_asset_id else down_mid
                _mval = _f.shares * _mid
                _upnl = _mval - _f.filled_usd
                _pfx  = _label if _line_idx == 0 else _indent
                _line_idx += 1
                _pos_lines.append(
                    f"{_pfx}FILLED  {_dir}  "
                    f"${_f.filled_usd:.2f} cost  "
                    f"@ {_f.fill_price:.4f}/share  "
                    f"({_f.shares:.2f} shares)  "
                    f"→  ${_mval:.2f} now  (unreal {_upnl:+.2f})"
                )

            for _f in self._pnl.pending_fills():
                _pfx = _label if _line_idx == 0 else _indent
                _line_idx += 1
                _pos_lines.append(
                    f"{_pfx}PENDING {_f.asset_id[:12]}...  "
                    f"${_f.filled_usd:.2f} cost  "
                    f"@ {_f.fill_price:.4f}/share  "
                    f"({_f.shares:.2f} shares)  "
                    f"awaiting Gamma resolution"
                )

            for _o in _resting:
                _dir = "BUY_UP " if _o.asset_id == self._up_asset_id else "BUY_DOWN"
                _pfx = _label if _line_idx == 0 else _indent
                _line_idx += 1
                _pos_lines.append(
                    f"{_pfx}RESTING {_dir}  "
                    f"${_o.size_usd:.2f} limit  "
                    f"@ {_o.price:.4f}/share  "
                    f"(waiting to fill)"
                )

            _pos_str = "\n".join(_pos_lines) if _pos_lines else "  Positions : none"

            _size_str = f" ${decision.size_usd:.2f}" if decision.size_usd > 0 else ""

            # Build a unified status string covering ALL reasons the bot may not trade.
            _cb = self._circuit_breaker
            if _cb.is_halted:
                import datetime as _dt2
                _hit_utc = _dt2.datetime.fromtimestamp(
                    _cb.halted_at, tz=_dt2.timezone.utc
                ).strftime("%H:%M:%S")
                _cb_str = (
                    f"HALTED [{_cb.halt_kind}]  "
                    f"resume in {_cb.seconds_until_resume:.0f}s  "
                    f"(tripped {_hit_utc} UTC)"
                )
            else:
                _allowed, _block_reason = pre_trade_check(decision.action, safety)
                if not _allowed:
                    _cb_str = f"blocked: {_block_reason}"
                elif _extreme_market_blocked:
                    _cb_str = f"extreme market: poly_mid={up_mid:.3f} (allowed 0.10–0.90)"
                elif decision.action in (Action.HOLD, Action.EXIT, Action.HOLD_TO_RESOLUTION):
                    _cb_str = f"holding: {decision.reason}"
                elif _live_passive_blocked:
                    _cb_str = "live mode — GTC passive blocked (no fill tracking)"
                elif _window_fill_blocked:
                    _cb_str = f"one-fill limit: already filled {_trade_asset[:12]}... this window"
                elif _in_cooldown and _trade_asset:
                    _secs_left = self._early_exit_cooldown_until.get(_trade_asset, 0.0) - time.time()
                    _cb_str = f"cooldown after exit — re-entry blocked {_secs_left:.0f}s"
                elif result is None:
                    # Action was a trade but executor returned None (e.g. GTC already resting)
                    _cb_str = "GTC already resting — skipped duplicate order"
                else:
                    _cb_str = "clear"

            _left = [
                _sepBold,
                f"[QUANTIE DA$HBOARD] {'PAPER' if self._cfg.paper else 'LIVE'} MODE",
                _sepBold,
                (
                    f"  Balance   : ${post_total:.2f} total   |   ${post_cash:.2f} cash"
                    f"   |   PnL: {self._pnl.realized_pnl:+.2f}"
                    f"   |   window: {self._pnl.realized_pnl - self._window_pnl_start:+.2f}"
                ),
                (
                    f"  Signal    : p_up {out.p_up:.4f}  gbm {out.gbm_p_up:.4f}"
                    f"  ml_res {out.ml_residual:+.4f}"
                    f"   |   edge {action_edge:.4f}"
                    f"  thr {decision.edge.threshold:.4f}"
                    f"  ci {decision.edge.ci_width:.4f}"
                    f"  →  {_action_label}{_size_str}"
                ),
                f"  Poly      : impl_prob {up_mid:.4f} | btc open ${s_0:.2f} now ${s_t:.2f}",
                _sep,
                *_pos_str.split("\n"),
                _sep,
                (
                    f"  Risk      : window_start ${self._window_start_balance_usd:.2f}"
                    f"  cap ${self._window_risk_cap_usd:.2f}"
                    f"  used ${self._executor.window_exposure_usd:.2f}"
                ),
                f"  Status    : {_cb_str}",
                _sep,
                f"  Network   : binance {binance_age:.0f}ms  chainlink {chainlink_age:.0f}ms",
                _sep,
                f"[{_ts} UTC]   TAU {tau:.0f}s   {_regime_label}",
            ]
            _right = self._render_history_column()
            _max_h = max(len(_left), len(_right))
            _left  += [""] * (_max_h - len(_left))
            _right += [""] * (_max_h - len(_right))
            _out = "\n" * 10
            for _ll, _rl in zip(_left, _right):
                _out += _ll[:116].ljust(116) + " |$| " + _rl + "\n"
            print(_out, end="", flush=True)


    async def _do_early_exit(
        self,
        exits_needed: list,
        up_ask: float,
        up_bid: float,
        now_ms: int,
    ) -> None:
        """Sell specified positions when p_win drops below hold threshold.

        exits_needed: pre-filtered fills — all have known asset_ids, none already attempted.
        """
        # Bug #4: use actual DOWN book bid when available; fall back to complement of UP ask.
        down_book = self._poly.down_book
        down_bid = (
            down_book.best_bid
            if (down_book is not None and down_book.best_bid > 0.0)
            else 1.0 - up_ask
        )

        await self._executor.cancel_all_open()

        # Bug #7: mark all attempted fills before submitting so that even if sell
        # fails we don't retry the same fill on the next tick.
        for fill in exits_needed:
            self._exit_attempted_order_ids.add(fill.order_id)

        sell_results = await self._executor.exit_positions(
            open_fills=exits_needed,
            up_asset_id=self._up_asset_id,
            down_asset_id=self._down_asset_id,
            up_bid=up_bid,
            down_bid=down_bid,
        )
        # sell_results is 1:1 with exits_needed (all asset_ids are known, none skipped)
        for fill, result in zip(exits_needed, sell_results):
            # Bug #5: compare against STATUS_FILLED constant, not hardcoded string
            if result.status == STATUS_FILLED:
                pnl_delta = self._pnl.record_early_exit(
                    fill.order_id,
                    result.filled_usd,
                    fee_usd=result.filled_usd * self._taker_fee_rate,
                )
                pending = self._bandit_pending_by_order.pop(fill.order_id, None)
                if pending is not None and not self._circuit_breaker.is_halted:
                    arm, vec, _ = pending
                    self._policy.update(BanditFeedback(action=arm, reward=float(pnl_delta), context_vec=vec))
                # Bug #8: decrement window exposure after position is sold
                self._executor._window_exposure_usd -= fill.filled_usd
                # Anti-oscillation: block re-entry for this direction for 60s
                self._early_exit_cooldown_until[fill.asset_id] = time.time() + 20.0
                log.info(
                    "engine.early_exit",
                    order_id=fill.order_id,
                    asset_id=fill.asset_id,
                    proceeds_usd=round(result.filled_usd, 4),
                    pnl_delta=round(pnl_delta, 4),
                )
                import datetime as _dt
                _fts = _dt.datetime.fromtimestamp(now_ms / 1000, tz=_dt.timezone.utc).strftime("%H:%M:%S")
                _dir = "UP  ▲" if fill.asset_id == self._up_asset_id else "DOWN ▼"
                _exit_price = result.filled_usd / fill.shares if fill.shares > 0 else 0.0
                self._add_history(
                    f"  ⚡ EXIT {_dir}  [{_fts} UTC]",
                    f"  ${fill.filled_usd:.2f} → ${result.filled_usd:.2f}  PnL {pnl_delta:+.2f}",
                    f"  @ {fill.fill_price:.4f} → {_exit_price:.4f}/sh",
                )
            else:
                self._warn(
                    "engine.early_exit.failed",
                    order_id=fill.order_id,
                    asset_id=fill.asset_id,
                    status=result.status,
                )

    def _add_history(self, line1: str, line2: str, line3: str) -> None:
        self._history.appendleft((line1, line2, line3))

    def _warn(self, event: str, **kw) -> None:
        log.warning(event, **kw)

    def _err(self, event: str, **kw) -> None:
        log.error(event, **kw)

    async def _notify_circuit_breaker_if_new(self) -> None:
        """Send Telegram alert when a new circuit breaker kind trips."""
        if not self._notifier:
            return
        cb = self._circuit_breaker
        # Hard stop is handled separately in _apply_resolution
        if cb.hard_stop_pending and self._last_cb_kind != "session_loss":
            self._last_cb_kind = "session_loss"
            await self._notifier.notify_circuit_breaker(
                kind="session_loss",
                detail=f"Realized PnL {self._pnl.realized_pnl:+.2f} USD — no new entries, waiting for positions to close",
            )
            return
        kind = cb.halt_kind
        if kind and kind != self._last_cb_kind:
            self._last_cb_kind = kind
            await self._notifier.notify_circuit_breaker(
                kind=kind,
                detail=cb.halt_reason,
            )
        elif not kind and not cb.hard_stop_pending:
            self._last_cb_kind = ""

    def _render_history_column(self) -> list[str]:
        rw = _R_WIDTH
        col: list[str] = [
            "═" * rw,
            "[HI$TORY]".ljust(rw),
            "═" * rw,
        ]
        entries = list(self._history)
        for i in range(_HISTORY_MAX):
            if i < len(entries):
                l1, l2, l3 = entries[i]
                col.append(l1[:rw].ljust(rw))
                col.append(l2[:rw].ljust(rw))
                col.append(l3[:rw].ljust(rw))
            else:
                col.append("  —".ljust(rw))
                col.append("".ljust(rw))
                col.append("".ljust(rw))
            if i < _HISTORY_MAX - 1:
                col.append("─" * rw)
        return col

    def _estimate_total_equity(self) -> float:
        """Cash + mark-to-market of open positions.

        Uses current book mids when available; falls back to cost basis per fill
        when the book for that asset hasn't been seen yet.
        """
        open_fills = self._pnl.open_fills()
        open_exposure = sum(f.filled_usd for f in open_fills)

        up_mid: float | None = None
        down_mid: float | None = None
        if self._up_asset_id and self._down_asset_id:
            up_book = self._poly.up_book
            if (
                up_book is not None
                and up_book.best_ask > 0.0
                and up_book.best_bid > 0.0
            ):
                up_mid = (up_book.best_ask + up_book.best_bid) / 2.0
                down_mid = 1.0 - up_mid

        market_value = 0.0
        for f in open_fills:
            if f.asset_id == self._up_asset_id and up_mid is not None:
                market_value += f.shares * up_mid
            elif f.asset_id == self._down_asset_id and down_mid is not None:
                market_value += f.shares * down_mid
            else:
                market_value += f.filled_usd  # fallback: cost basis

        cash = self._cfg.bankroll + self._pnl.realized_pnl - open_exposure
        return cash + market_value

    def _refresh_window_risk_baseline(self) -> None:
        """Snapshot current equity at window start; derive per-window risk cap.

        Capped at configured bankroll — bot never risks more than the starting
        budget even if realized PnL grows beyond it.
        """
        equity = min(self._estimate_total_equity(), self._cfg.bankroll)
        self._window_start_balance_usd = equity
        self._window_risk_cap_usd = equity * self._window_risk_fraction


def _build_live_broker() -> Broker:
    from quantie.execution.clob_client import ClobClient
    return ClobClient.from_env()


def _edge_for_action(decision: PolicyDecision) -> float:
    """Return the edge value relevant to the chosen action.

    UP actions  → edge_buy_up   (positive when UP token underpriced)
    DOWN actions → edge_buy_down (positive when DOWN token underpriced)
    Neutral actions (HOLD/EXIT) → larger of the two so the dashboard shows
    the strongest available signal.
    """
    e = decision.edge
    if decision.action in (Action.TAKE_UP, Action.MAKE_UP):
        return e.edge_buy_up
    if decision.action in (Action.TAKE_DOWN, Action.MAKE_DOWN):
        return e.edge_buy_down
    return max(e.edge_buy_up, e.edge_buy_down)


def _extract_gamma_outcome(market: dict) -> Optional[str]:
    """Return 'UP' or 'DOWN' from a resolved Gamma market dict, or None.

    Gamma may encode the winner as:
      - outcome string  ("UP" / "DOWN")
      - token ID        (clobTokenIds[0] = UP, clobTokenIds[1] = DOWN)
    Both forms are handled with a fallback chain.
    """
    if not market.get("resolved"):
        return None

    token_ids: list = market.get("clobTokenIds", [])
    winner = market.get("winner") or market.get("winnerOutcome") or ""

    if isinstance(winner, str) and winner:
        upper = winner.strip().upper()
        if upper in ("UP", "DOWN"):
            return upper
        # winner encoded as token ID
        if token_ids and winner == token_ids[0]:
            return "UP"
        if len(token_ids) > 1 and winner == token_ids[1]:
            return "DOWN"

    return None
