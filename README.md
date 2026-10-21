# Quantie

An async trading system for Polymarket's 5-minute BTC Up/Down markets.

Every five minutes Polymarket opens a market on whether BTC will close the window above its opening price. Quantie streams live market data, estimates the fair probability of "Up" in real time, and trades when the market price disagrees with that estimate by more than fees and model uncertainty.

## How it works

```
Binance (trades + depth)  ┐
Polymarket RTDS (oracle)  ├─► asyncio event bus ─► state ─► features ─► fair-value model ─► policy ─► execution
Polymarket CLOB (book)    ┘                                                                    │
                                                                                     safety layer (always on)
```

**Data.** Three WebSocket feeds (Binance aggregated trades and order-book depth, the Chainlink price oracle via Polymarket RTDS, and the Polymarket order book) are parsed into typed events and pushed through a single asyncio event bus into state objects for the current window.

**Fair value.** Two layers:
1. A closed-form baseline: the probability that BTC finishes above the window's opening price under geometric Brownian motion, using realised volatility from Binance.
2. A LightGBM model that corrects the baseline using 35 microstructure features: order-flow imbalance, book-depth imbalance, multi-horizon realised volatility, momentum, oracle–exchange basis and Polymarket book state. Monotone constraints keep the correction economically sensible, and isotonic calibration makes the output probabilities honest.

Training uses a time-ordered split by window, so no window appears in both train and validation.

**Policy.** Behaviour depends on time to expiry: passive quoting early in the window, active taking in the middle, high-conviction-only late, and exit-only in the final seconds. An order is placed only when the edge exceeds a threshold that accounts for taker fees and the model's confidence interval. Size comes from quarter-Kelly with hard caps per window and per bankroll. A LinUCB contextual bandit chooses between holding, taking and quoting.

**Execution.** FOK orders for takes and GTC orders for passive entries via the Polymarket CLOB client. A paper broker with a slippage model allows full dry runs.

**Safety.** Stale-feed checks, latency and drawdown circuit breakers, an error-rate breaker, a file-based kill switch and an append-only audit log. Prometheus metrics and optional Telegram notifications are included for monitoring.

## Results

- **Model:** 9% lower Brier score than the Brownian-motion baseline on a time-ordered hold-out.
- **Live:** traded with a $1,000 bankroll and broke even. The main lessons were that execution latency and order-book manipulation near expiry eat most of the modelled edge.

## Repository layout

```
src/quantie/
  data/           WebSocket clients, event bus, tick and decision stores
  state/          per-window, Binance and Polymarket state
  features/       volatility, microstructure and time features
  models/         GBM baseline, LightGBM residual, calibration, ensemble
  policy/         regimes, edge, Kelly sizing, LinUCB bandit, decisions
  execution/      CLOB client, order executor, paper broker
  safety/         pre-trade checks, circuit breakers, audit log
  notifications/  Telegram notifier
  engine.py       main event loop
scripts/          data smoke test, model training/evaluation, bandit training
config/           paper and live configurations
tests/unit/       310 unit tests (no network needed)
docs/RUNBOOK.md   operating guide
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/unit/                              # run the test suite
python scripts/smoke_phase1.py --duration 60    # check live feeds
quantie                                         # paper trading (default)
```

Live trading needs Polymarket credentials in `.env` (see `.env.example`) and `quantie --live --config config/v1.yaml`. Read `docs/RUNBOOK.md` first.

Recorded market data and trained models are not included in this repository.