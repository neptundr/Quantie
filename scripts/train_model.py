"""
Train LightGBM residual model + isotonic calibrator from collected tick data.

Usage:
    python scripts/train_model.py                        # default paths
    python scripts/train_model.py --data-dir data/ticks --model-dir models/v1
    python scripts/train_model.py --val-windows 200      # explicit val size

Output:
    models/v1/lgbm_residual.pkl   — LightGBM classifier
    models/v1/calibrator.pkl      — isotonic calibrator fitted on val set

Evaluation printed to stdout:
    Brier score (train/val), log loss (val), feature importances top-20
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path so we can import quantie without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantie.models.calibration import IsotonicCalibrator
from quantie.models.ml_residual import FEATURE_NAMES, LGBMResidualModel


def load_data(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("ticks_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No tick parquets found in {data_dir}")
    print(f"Loading {len(files)} parquet files...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  Rows: {len(df):,}  Windows: {df['window_start_ts'].nunique():,}")
    return df


def time_split(df: pd.DataFrame, n_val_windows: int) -> tuple[pd.Index, pd.Index]:
    """Split by window_start_ts — last n_val_windows go to val. No leakage."""
    window_ts = sorted(df["window_start_ts"].unique())
    if n_val_windows >= len(window_ts):
        raise ValueError(
            f"n_val_windows={n_val_windows} >= total windows={len(window_ts)}"
        )
    val_ts = set(window_ts[-n_val_windows:])
    train_idx = df.index[~df["window_start_ts"].isin(val_ts)]
    val_idx = df.index[df["window_start_ts"].isin(val_ts)]
    return train_idx, val_idx


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Quantie LightGBM residual model")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ticks"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/v1"))
    parser.add_argument(
        "--val-windows",
        type=int,
        default=300,
        help="Number of most-recent windows reserved for validation (default: 300)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ load
    df = load_data(args.data_dir)

    # Drop rows with no valid outcome
    before = len(df)
    df = df[df["outcome"].isin([0.0, 1.0])].copy()
    if len(df) < before:
        print(f"  Dropped {before - len(df):,} rows with missing/invalid outcome")

    total_windows = df["window_start_ts"].nunique()
    up_count = int((df.groupby("window_start_ts")["outcome"].first() == 1).sum())
    down_count = int((df.groupby("window_start_ts")["outcome"].first() == 0).sum())
    print(f"\nOutcome balance: UP={up_count}  DOWN={down_count}  total={total_windows} windows")

    # ------------------------------------------------------------------ split
    n_val = min(args.val_windows, total_windows // 5)  # cap at 20%
    train_idx, val_idx = time_split(df, n_val_windows=n_val)
    df_train, df_val = df.loc[train_idx], df.loc[val_idx]
    print(f"\nTrain: {len(df_train):,} rows ({df_train['window_start_ts'].nunique()} windows)")
    print(f"Val:   {len(df_val):,} rows  ({df_val['window_start_ts'].nunique()} windows)")

    # Fill missing feature columns with NaN (LightGBM handles natively)
    missing_feats = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_feats:
        print(f"  Warning: {len(missing_feats)} features not in data, padding NaN: {missing_feats}")
        for f in missing_feats:
            df_train[f] = np.nan
            df_val[f] = np.nan

    X_train = df_train[FEATURE_NAMES].values.astype(float)
    y_train = df_train["outcome"].values.astype(float)
    X_val = df_val[FEATURE_NAMES].values.astype(float)
    y_val = df_val["outcome"].values.astype(float)
    gbm_val = df_val["gbm_p_up"].values.astype(float)

    # ------------------------------------------------------------------ train
    print("\nTraining LightGBM...")
    model = LGBMResidualModel()
    model.train(X_train, y_train)
    print("  Done.")

    # ------------------------------------------------------------------ eval
    p_train = model._model.predict_proba(X_train)[:, 1]
    p_val = model._model.predict_proba(X_val)[:, 1]

    from sklearn.metrics import log_loss

    bs_baseline = brier_score(y_val, gbm_val)
    bs_train = brier_score(y_train, p_train)
    bs_val = brier_score(y_val, p_val)
    ll_val = log_loss(y_val, p_val)

    print(f"\n{'='*55}")
    print(f"Brier score  GBM baseline (val):  {bs_baseline:.4f}")
    print(f"Brier score  LightGBM   (train):  {bs_train:.4f}")
    print(f"Brier score  LightGBM   (val):    {bs_val:.4f}  {'✓ beats GBM' if bs_val < bs_baseline else '✗ worse than GBM baseline'}")
    print(f"Log loss                (val):    {ll_val:.4f}")

    # ------------------------------------------------------------------ calibrate
    print("\nFitting isotonic calibrator on val set...")
    cal = IsotonicCalibrator()
    cal.fit(p_val, y_val)
    p_val_cal = np.array([cal.transform(float(p)) for p in p_val])
    bs_val_cal = brier_score(y_val, p_val_cal)
    print(f"Brier score  calibrated (val):    {bs_val_cal:.4f}  {'✓ better' if bs_val_cal <= bs_val else '~ no change'}")

    # ------------------------------------------------------------------ CI
    # Compute CI half-width from calibration curve residuals.
    # Strategy: bin calibrated predictions into 10 equal-width buckets,
    # compare mean predicted rate vs actual win rate per bucket.
    # ci_half_width = weighted RMSE of calibration error.
    # This measures how wrong our probability estimates are — not noise from
    # binary outcomes. Used at inference to set the edge threshold dynamically.
    _N_BINS = 10
    _MIN_BIN_SAMPLES = 20
    bins = np.linspace(0.0, 1.0, _N_BINS + 1)
    bin_idx = np.clip(np.digitize(p_val_cal, bins) - 1, 0, _N_BINS - 1)
    sq_errors, weights = [], []
    for i in range(_N_BINS):
        mask = bin_idx == i
        if mask.sum() < _MIN_BIN_SAMPLES:
            continue
        pred_rate = float(p_val_cal[mask].mean())
        actual_rate = float(y_val[mask].mean())
        sq_errors.append((pred_rate - actual_rate) ** 2)
        weights.append(int(mask.sum()))
    if sq_errors:
        w = np.array(weights, dtype=float) / sum(weights)
        ci_half_width = float(np.sqrt(np.dot(w, sq_errors)))
        ci_half_width = max(ci_half_width, 0.02)  # floor: never go below 2c
    else:
        ci_half_width = 0.08  # fallback if val set too small to bin
    print(f"CI half-width (calibration RMSE): {ci_half_width:.4f}  → ci_width={2*ci_half_width:.4f} at inference")

    # ------------------------------------------------------------------ feature importance
    print(f"\n{'='*55}")
    print("Top-20 feature importances (gain):")
    importances = model._model.booster_.feature_importance(importance_type="gain")
    pairs = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    max_imp = pairs[0][1] if pairs[0][1] > 0 else 1
    for name, imp in pairs[:20]:
        bar = "█" * max(1, int(40 * imp / max_imp))
        print(f"  {name:45s} {imp:9.1f}  {bar}")

    # ------------------------------------------------------------------ save
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.model_dir / "lgbm_residual.pkl")
    cal.save(args.model_dir / "calibrator.pkl")

    meta = {
        "ci_half_width": round(ci_half_width, 6),
        "brier_cal": round(bs_val_cal, 6),
        "brier_baseline": round(bs_baseline, 6),
        "n_train_windows": int(df_train["window_start_ts"].nunique()),
        "n_val_windows": int(df_val["window_start_ts"].nunique()),
        "trained_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    meta_path = args.model_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n{'='*55}")
    print(f"Saved to {args.model_dir}/")
    print(f"  lgbm_residual.pkl")
    print(f"  calibrator.pkl")
    print(f"  model_meta.json  (ci_half_width={ci_half_width:.4f})")
    print(f"\nRun bot with trained model:")
    print(f"  quantie --model-dir {args.model_dir}")


if __name__ == "__main__":
    main()
