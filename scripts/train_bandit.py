"""Offline initialize LinUCB bandit from decision logs + tick outcomes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantie.policy.bandit import LinUCBBandit
from quantie.policy.decision import Action


_ACTIONS = {
    "Action.HOLD": Action.HOLD,
    "Action.TAKE_UP": Action.TAKE_UP,
    "Action.TAKE_DOWN": Action.TAKE_DOWN,
    "Action.MAKE_UP": Action.MAKE_UP,
    "Action.MAKE_DOWN": Action.MAKE_DOWN,
}
_EPS = 0.05


def load_decisions(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("decisions_*.csv"))
    if not files:
        raise FileNotFoundError(f"no decisions csv in {path}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def load_outcomes(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("ticks_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no ticks parquet in {path}")
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    out = df.groupby("window_start_ts", as_index=False)["outcome"].first()
    out = out[out["outcome"].isin([0.0, 1.0])]
    return out


def context_vector(row: pd.Series) -> np.ndarray:
    tau = float(row.get("tau_sec", 0.0))
    regime = str(row.get("regime", ""))
    regime_oh = [
        1.0 if "EARLY" in regime else 0.0,
        1.0 if "MID" in regime else 0.0,
        1.0 if "LATE" in regime else 0.0,
        1.0 if "TERMINAL" in regime else 0.0,
    ]
    return np.array(
        [
            float(row.get("p_up", 0.5)),
            float(row.get("ci_width", 0.1)),
            float(row.get("gbm_p_up", 0.5)),
            float(row.get("ml_residual", 0.0)),
            float(row.get("edge", 0.0)),
            tau / 300.0,
            *regime_oh,
            float(row.get("size_usd", 0.0)) / 100.0,
            float(row.get("balance_total", 0.0)) / 1000.0,
            float(row.get("balance_cash", 0.0)) / 1000.0,
        ],
        dtype=float,
    )


def reward_from_row(row: pd.Series) -> float:
    action = str(row["action"])
    outcome = float(row["outcome"])
    p_up = float(row.get("p_up", 0.5))
    size = float(row.get("size_usd", 0.0))
    if action in ("Action.HOLD", "HOLD"):
        edge = abs(p_up - 0.5)
        return -0.05 * edge
    if action in ("Action.TAKE_UP", "Action.MAKE_UP"):
        won = 1.0 if outcome == 1.0 else 0.0
    elif action in ("Action.TAKE_DOWN", "Action.MAKE_DOWN"):
        won = 1.0 if outcome == 0.0 else 0.0
    else:
        return 0.0
    pnl_proxy = size * (won - (1.0 - won))
    # simple IPS smoothing for deterministic logger policy
    propensity = max(_EPS / 5.0, 1.0 - _EPS)
    return pnl_proxy / propensity


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LinUCB bandit from logged decisions")
    parser.add_argument("--decisions-dir", type=Path, default=Path("data/decisions"))
    parser.add_argument("--ticks-dir", type=Path, default=Path("data/ticks"))
    parser.add_argument("--out", type=Path, default=Path("models/v1/bandit_state.npz"))
    args = parser.parse_args()

    decisions = load_decisions(args.decisions_dir)
    outcomes = load_outcomes(args.ticks_dir)
    df = decisions.merge(outcomes, on="window_start_ts", how="inner")
    if df.empty:
        raise ValueError("no joined decision/outcome rows")

    sample_vec = context_vector(df.iloc[0])
    bandit = LinUCBBandit(dim=int(sample_vec.shape[0]))

    n_rows = 0
    for _, row in df.iterrows():
        action_key = str(row.get("action", ""))
        if action_key not in _ACTIONS:
            continue
        arm = _ACTIONS[action_key]
        x = context_vector(row)
        r = reward_from_row(row)
        bandit.update(arm, x, float(r))
        n_rows += 1

    if n_rows == 0:
        raise ValueError("no valid action rows for training")
    bandit.save(args.out)
    print(f"trained rows={n_rows} saved={args.out}")


if __name__ == "__main__":
    main()

