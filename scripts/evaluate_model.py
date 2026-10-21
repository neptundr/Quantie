"""
Evaluate trained model: calibration, paper P&L simulation, feature importance.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model-dir models/v1 --val-windows 300
    python scripts/evaluate_model.py --out-dir reports/eval_v1

Outputs (saved to --out-dir):
    calibration.png         predicted prob vs actual win rate
    pnl_simulation.png      cumulative paper P&L on val set
    feature_importance.png  gain-based feature importance
    brier_by_tau.png        Brier score across window lifecycle
    summary.txt             text report with verdict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantie.models.calibration import IsotonicCalibrator
from quantie.models.ml_residual import FEATURE_NAMES, LGBMResidualModel

# paper trade constants
_EDGE_THRESHOLD   = 0.04
_MAX_POSITION_USD = 50.0
_KELLY_FRACTION   = 0.25
_MIN_TAU          = 30.0


def load_data(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("ticks_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquets in {data_dir}")
    print(f"Loading {len(files)} files...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df["outcome"].isin([0.0, 1.0])].copy()
    print(f"  {len(df):,} rows, {df['window_start_ts'].nunique():,} windows")
    return df


def time_split(df: pd.DataFrame, n_val: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    wts = sorted(df["window_start_ts"].unique())
    val_ts = set(wts[-n_val:])
    return df[~df["window_start_ts"].isin(val_ts)], df[df["window_start_ts"].isin(val_ts)]


def simulate_paper_trades(
    df_val: pd.DataFrame,
    p_cal: np.ndarray,
    cal: IsotonicCalibrator,
) -> pd.DataFrame:
    """Simulate one trade per window on val set. Returns trade log."""
    df_val = df_val.copy()
    df_val["p_cal"] = p_cal

    records = []
    for wts, grp in df_val.groupby("window_start_ts", sort=True):
        outcome = float(grp["outcome"].iloc[0])
        eligible = grp[grp["tau_seconds"] > _MIN_TAU].copy()
        if eligible.empty:
            continue

        eligible["edge_buy_up"]   = eligible["p_cal"] - eligible["poly_mid"].clip(0.01, 0.99)
        eligible["edge_buy_down"] = (1 - eligible["p_cal"]) - (1 - eligible["poly_mid"].clip(0.01, 0.99))
        eligible["best_edge"]     = eligible[["edge_buy_up", "edge_buy_down"]].max(axis=1)

        best = eligible.loc[eligible["best_edge"].idxmax()]
        edge = float(best["best_edge"])

        if edge < _EDGE_THRESHOLD:
            records.append({"window_start_ts": wts, "action": "HOLD",
                             "edge": edge, "size_usd": 0.0,
                             "outcome": outcome, "pnl": 0.0,
                             "p_cal": float(best["p_cal"])})
            continue

        size = min(_KELLY_FRACTION * edge * _MAX_POSITION_USD / 0.5, _MAX_POSITION_USD)
        size = round(size, 2)
        poly_price = float(best["poly_mid"])

        if float(best["edge_buy_up"]) >= float(best["edge_buy_down"]):
            action = "BUY_UP"
            won = (outcome == 1.0)
        else:
            action = "BUY_DOWN"
            won = (outcome == 0.0)

        pnl = size * (1.0 / max(poly_price, 0.01) - 1.0) if won else -size
        records.append({"window_start_ts": wts, "action": action,
                         "edge": edge, "size_usd": size,
                         "outcome": outcome, "pnl": round(pnl, 4),
                         "p_cal": float(best["p_cal"])})

    return pd.DataFrame(records)


def plot_calibration(p_raw: np.ndarray, p_cal: np.ndarray,
                     y: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Calibration: predicted probability vs actual win rate", fontsize=13)

    for ax, probs, label, color in [
        (axes[0], p_raw, "Raw LightGBM", "#e05c5c"),
        (axes[1], p_cal, "Calibrated",   "#5c9be0"),
    ]:
        bins = np.linspace(0, 1, 11)
        bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, len(bins) - 2)
        means, actuals, counts = [], [], []
        for i in range(len(bins) - 1):
            mask = bin_idx == i
            if mask.sum() >= 5:
                means.append(probs[mask].mean())
                actuals.append(y[mask].mean())
                counts.append(mask.sum())

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        ax.scatter(means, actuals, s=[max(c / 50, 10) for c in counts],
                   color=color, alpha=0.8, zorder=3)
        ax.plot(means, actuals, color=color, alpha=0.6, label=label)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Actual win rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(); ax.grid(alpha=0.3)
        brier = float(np.mean((probs - y) ** 2))
        ax.set_title(f"{label}  (Brier={brier:.4f})")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def plot_pnl(trades: pd.DataFrame, out: Path) -> dict:
    active = trades[trades["action"] != "HOLD"].copy()
    n_trades = len(active)
    n_hold   = int((trades["action"] == "HOLD").sum())
    won      = int((active["pnl"] > 0).sum())
    lost     = n_trades - won
    win_rate = won / n_trades if n_trades > 0 else 0.0
    total_pnl = float(active["pnl"].sum())
    avg_win   = float(active[active["pnl"] > 0]["pnl"].mean()) if won > 0 else 0.0
    avg_loss  = float(active[active["pnl"] <= 0]["pnl"].mean()) if lost > 0 else 0.0
    avg_edge  = float(active["edge"].mean()) if n_trades > 0 else 0.0
    sharpe    = float(active["pnl"].mean() / active["pnl"].std()) \
                if n_trades > 1 and active["pnl"].std() > 0 else 0.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Paper P&L Simulation (held-out val set)", fontsize=13)

    # cumulative P&L
    ax = axes[0, 0]
    cum = active["pnl"].cumsum().values
    ax.plot(cum, color="#5c9be0", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(range(len(cum)), cum, 0, where=cum >= 0, alpha=0.2, color="green")
    ax.fill_between(range(len(cum)), cum, 0, where=cum < 0,  alpha=0.2, color="red")
    ax.set_title(f"Cumulative P&L  (total ${total_pnl:+.2f})")
    ax.set_xlabel("Trade #"); ax.set_ylabel("USD"); ax.grid(alpha=0.3)

    # win / loss bar
    ax = axes[0, 1]
    ax.bar(["Won", "Lost"], [won, lost], color=["#4caf50", "#e05c5c"])
    ax.set_title(f"Win/Loss  {won}/{n_trades} trades  ({100*win_rate:.1f}%)")
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate([won, lost]):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=11)

    # P&L distribution
    ax = axes[1, 0]
    ax.hist(active["pnl"], bins=40, color="#5c9be0", edgecolor="white", alpha=0.8)
    ax.axvline(active["pnl"].mean(), color="red", linestyle="--",
               label=f"Mean ${active['pnl'].mean():.2f}")
    ax.set_title("P&L per trade")
    ax.set_xlabel("USD"); ax.legend(); ax.grid(alpha=0.3)

    # edge distribution
    ax = axes[1, 1]
    ax.hist(active["edge"], bins=30, color="#9b5ce0", edgecolor="white", alpha=0.8)
    ax.set_title(f"Edge  (mean={avg_edge:.3f})")
    ax.set_xlabel("Edge"); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")

    return dict(n_windows=len(trades), n_trades=n_trades, n_hold=n_hold,
                win_rate=win_rate, total_pnl=total_pnl, avg_win=avg_win,
                avg_loss=avg_loss, avg_edge=avg_edge, sharpe=sharpe)


def plot_feature_importance(model: LGBMResidualModel, out: Path) -> None:
    imps  = model._model.booster_.feature_importance(importance_type="gain")
    pairs = sorted(zip(FEATURE_NAMES, imps), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in pairs]
    vals  = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(names[::-1], vals[::-1], color="#5c9be0", edgecolor="white")
    ax.set_title("Feature importance (gain)")
    ax.set_xlabel("Gain")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def plot_brier_by_tau(df_val: pd.DataFrame, p_raw: np.ndarray,
                      gbm_p: np.ndarray, out: Path) -> None:
    df_val = df_val.copy()
    df_val["p_pred"] = p_raw
    df_val["gbm_p"]  = gbm_p

    bins   = np.arange(0, 305, 30)
    labels = [f"{b}-{b+30}s" for b in bins[:-1]]
    df_val["tau_bin"] = pd.cut(df_val["tau_seconds"], bins=bins, labels=labels)

    brier_ml  = df_val.groupby("tau_bin", observed=True).apply(
        lambda g: float(np.mean((g["p_pred"] - g["outcome"]) ** 2)))
    brier_gbm = df_val.groupby("tau_bin", observed=True).apply(
        lambda g: float(np.mean((g["gbm_p"]  - g["outcome"]) ** 2)))

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(brier_ml))
    ax.plot(x, brier_gbm.values, "o--", color="#e05c5c", label="GBM baseline", linewidth=1.5)
    ax.plot(x, brier_ml.values,  "o-",  color="#5c9be0", label="LightGBM",     linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(list(brier_ml.index), rotation=45, ha="right")
    ax.set_title("Brier score by time-remaining bucket  (lower = better)")
    ax.set_ylabel("Brier score")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    type=Path, default=Path("data/ticks"))
    parser.add_argument("--model-dir",   type=Path, default=Path("models/v1"))
    parser.add_argument("--out-dir",     type=Path, default=Path("reports/eval_v1"))
    parser.add_argument("--val-windows", type=int,  default=300)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data_dir)
    _, df_val = time_split(df, args.val_windows)
    print(f"Val set: {len(df_val):,} rows, {df_val['window_start_ts'].nunique()} windows")

    model = LGBMResidualModel()
    model.load(args.model_dir / "lgbm_residual.pkl")
    cal = IsotonicCalibrator()
    cal.load(args.model_dir / "calibrator.pkl")

    p_raw = model._model.predict_proba(df_val[FEATURE_NAMES].values.astype(float))[:, 1]
    p_cal = np.array([cal.transform(float(p)) for p in p_raw])
    y     = df_val["outcome"].values.astype(float)
    gbm_p = df_val["gbm_p_up"].values.astype(float)

    print("\nGenerating plots...")
    plot_calibration(p_raw, p_cal, y, args.out_dir / "calibration.png")
    plot_feature_importance(model, args.out_dir / "feature_importance.png")
    plot_brier_by_tau(df_val, p_raw, gbm_p, args.out_dir / "brier_by_tau.png")

    print("\nRunning paper trade simulation...")
    df_val = df_val.copy()
    trades = simulate_paper_trades(df_val, p_cal, cal)
    stats  = plot_pnl(trades, args.out_dir / "pnl_simulation.png")

    brier_gbm = float(np.mean((gbm_p - y) ** 2))
    brier_ml  = float(np.mean((p_raw  - y) ** 2))
    brier_cal = float(np.mean((p_cal  - y) ** 2))

    ml_ok  = brier_cal < brier_gbm
    pnl_ok = stats["total_pnl"] > 0
    wr_ok  = stats["win_rate"] > 0.50

    summary = f"""
=== MODEL EVALUATION — {args.model_dir} ===

BRIER SCORES  (lower = better | random baseline = 0.25)
  GBM baseline:        {brier_gbm:.4f}
  LightGBM raw:        {brier_ml:.4f}   ({100*(brier_gbm-brier_ml)/brier_gbm:+.1f}% vs GBM)
  LightGBM calibrated: {brier_cal:.4f}   ({100*(brier_gbm-brier_cal)/brier_gbm:+.1f}% vs GBM)

PAPER TRADE SIMULATION  ({stats['n_windows']} windows)
  Trades placed:       {stats['n_trades']}  ({100*stats['n_trades']/stats['n_windows']:.0f}% of windows)
  HOLD (below edge):   {stats['n_hold']}
  Win rate:            {100*stats['win_rate']:.1f}%
  Total P&L:           ${stats['total_pnl']:+.2f}
  Avg win:             ${stats['avg_win']:+.2f}
  Avg loss:            ${stats['avg_loss']:+.2f}
  Avg edge traded:     {stats['avg_edge']:.3f}
  Sharpe (per trade):  {stats['sharpe']:.2f}

VERDICT
  {'✓' if ml_ok  else '✗'} ML {'improves' if ml_ok else 'does NOT improve'} on GBM baseline
  {'✓' if pnl_ok else '✗'} Paper P&L {'positive' if pnl_ok else 'NEGATIVE — review before deploying'}
  {'✓' if wr_ok  else '✗'} Win rate {'above' if wr_ok else 'below'} 50%

  {'SAFE TO DEPLOY with --model-dir ' + str(args.model_dir) if (ml_ok and pnl_ok) else 'DO NOT DEPLOY — model not ready'}

Plots saved to: {args.out_dir}/
"""
    print(summary)
    (args.out_dir / "summary.txt").write_text(summary)
    print(f"  Saved {args.out_dir}/summary.txt")


if __name__ == "__main__":
    main()
