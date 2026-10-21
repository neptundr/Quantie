# Quantie Operator Runbook

Polymarket BTC 5-minute UP/DOWN trading bot — v1.

---

## Quick start

```bash
# Paper mode (default, safe)
source .venv/bin/activate
quantie

# Live mode (real orders — read Live mode checklist first)
quantie --live --config config/v1.yaml
```

---

## Prerequisites

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# edit .env with your keys
```

---

## Configuration

| File | Purpose |
|------|---------|
| `config/paper.yaml` | Paper trading (small limits, safe defaults) |
| `config/v1.yaml` | Live trading ($1k bankroll, $50/window cap) |

Key config fields:
```yaml
bankroll_usd: 1000
max_position_per_window_usd: 50
kelly_fraction: 0.25
stale_chainlink_ms: 2500
stale_binance_ms: 1500
terminal_zone_seconds: 30
```

---

## Environment variables (live mode)

| Variable | Description |
|----------|-------------|
| `POLY_PRIVATE_KEY` | EVM private key (hex, with or without 0x) |
| `POLY_API_KEY` | Polymarket API key |
| `POLY_API_SECRET` | Polymarket API secret |
| `POLY_API_PASSPHRASE` | Polymarket API passphrase |
| `POLY_FUNDER` | Funder address (checksummed) |
| `POLY_CHAIN_ID` | 137 = Polygon mainnet, 80002 = Amoy testnet |

**Never commit `.env` to git.**

---

## Live mode checklist

Before running `quantie --live` for the first time:

- [ ] Wallet funded with USDC.e (not USDC native) on Polygon
- [ ] `POLY_*` env vars set and verified
- [ ] Smoke test passed: place one $5 GTC, cancel it, verify via Polymarket UI
- [ ] Start with `max_position_per_window_usd: 10` in config
- [ ] Prometheus metrics reachable: `curl localhost:8000/metrics`
- [ ] Paper-trade P&L positive over ≥ 72h before going live

---

## Monitoring

Prometheus scrapes `http://localhost:8000/metrics` (override with `--metrics-port`).

| Metric | Type | Description |
|--------|------|-------------|
| `quantie_pnl_realized_usd` | Gauge | Session realized P&L in USD |
| `quantie_position_size_usd` | Gauge | Current open position notional |
| `quantie_model_p_up` | Gauge | Latest model P(UP) |
| `quantie_orders_placed_total` | Counter | Orders submitted to broker |
| `quantie_orders_filled_total` | Counter | Orders confirmed filled |
| `quantie_orders_cancelled_total` | Counter | FOK misses + explicit cancels |
| `quantie_errors_total{error_type}` | Counter | Errors by type label |
| `quantie_regime` | Gauge | 0=EARLY 1=MID 2=LATE 3=TERMINAL |

**Recommended alerts:**
- `quantie_pnl_realized_usd < -200` → drawdown approaching limit
- `quantie_errors_total` rate > 1/min → check logs immediately
- `quantie_orders_filled_total` unchanged after 30+ windows → model not trading

---

## Logs

```bash
tail -f logs/app.log    # all events (structured JSON)
tail -f logs/audit.log  # orders and fills only (append-only)
```

Key log events:

| Event | Key fields |
|-------|-----------|
| `order.submitted` | `action`, `status`, `filled_usd`, `window_exposure`, `regime` |
| `order.blocked` | `action`, `reason`, `regime` |
| `paper.gtc.filled` | `order_id`, `filled_usd` |
| `engine.new_window` | `up`, `down` (token IDs) |
| `orders.cancelled_all` | `count` |
| `quantie.fatal` | `error` |

---

## Kill switch

**Immediate halt:**
```bash
kill -TERM <pid>
```

SIGTERM cancels all open GTC orders then exits cleanly. SIGINT (Ctrl+C) does the same.

The engine also self-halts when:
- Drawdown ≥ 20% of starting bankroll
- Kill switch flag set in `SafetySnapshot`

---

## Regime behaviour

| Regime | tau | Behaviour |
|--------|-----|-----------|
| EARLY | > 240s | Passive GTC only at `fair_value − 1.5¢` |
| MID | 60–240s | Primary zone; take when edge > threshold |
| LATE | 30–60s | Take-only; threshold +1¢; CI must be < 0.08 |
| TERMINAL | ≤ 30s | No new entries; exit only |

GTC orders are cancelled automatically at tau ≤ 35s.

---

## Scaling ramp (live)

Follow strictly — do not skip steps:

| Step | max_position_per_window_usd | Requirement to advance |
|------|-----------------------------|------------------------|
| 1 | $10 | 48h stable, no unexpected halts |
| 2 | $20 | ≥ 100 windows, fill rate matches paper |
| 3 | $35 | Brier score < 0.245 |
| 4 | $50 | Live vs paper P&L divergence < 30% |

**v1 hard cap: $50/window. No exceptions.**

---

## Common issues

### Bot not trading in MID regime
- Check `quantie_model_p_up` — if near 0.5, model is cold-starting (GBM only, no ML residual)
- Untrained CI width = 0.15 → threshold ≈ 0.26; edge rarely exceeds this early
- Wait ~50 windows for enough data to train LightGBM residual

### Stale oracle block
- `order.blocked reason="stale Chainlink"` in logs
- RTDS WebSocket may have dropped — check reconnect events in `logs/app.log`
- Verify connectivity: `wss://ws-live-data.polymarket.com`

### FOK never fills
- Book too thin or stale ask used for edge calculation
- Monitor `quantie_orders_cancelled_total` rate
- Consider raising `base_threshold` if adverse selection is suspected

### Drawdown halt
- `order.blocked reason="drawdown limit hit"` — session loss ≥ 20% of bankroll
- Do NOT restart immediately; review recent fills and model output
- Reduce `bankroll_usd` in config after root cause identified

### Token ID errors
- `engine.new_window` log shows empty `up`/`down` → Gamma API call failed
- Check `GAMMA_API` reachability: `curl https://gamma-api.polymarket.com`
- WindowManager retries with tenacity; check retry logs

---

## Running tests

```bash
pytest tests/unit/                           # fast, no network (~2s)
pytest --cov=src --cov-report=term-missing   # with coverage report
```

Coverage minimum: **80%** (enforced by `fail_under = 80` in `pyproject.toml`).
