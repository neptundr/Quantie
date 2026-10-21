"""
LightGBM residual model — predicts bias correction on top of GBM baseline.

target   = actual_outcome (1=UP, 0=DOWN) - gbm_p_up
features = FEATURE_NAMES (subset of FeatureVector numeric fields)

Final P_up = clip(gbm_p_up + predict_residual(fv), 0.01, 0.99)

Monotone constraints:
  momentum_30s                  → +1 (higher momentum → more likely UP)
  distance_from_strike_in_sigma → +1 (above strike → UP)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

# Feature names — must match FeatureVector field names exactly
FEATURE_NAMES: list[str] = [
    "tau_seconds",
    "tau_normalized",
    "hour_of_day",
    "day_of_week",
    "realized_vol_short",
    "realized_vol_mid",
    "realized_vol_long",
    "realized_vol_ratio_short_long",
    "realized_vol_60s_zscore",
    "momentum_1s",
    "momentum_5s",
    "momentum_15s",
    "momentum_30s",
    "ofi_binance_1s",
    "ofi_binance_5s",
    "ofi_binance_30s",
    "depth_imbalance_binance",
    "chainlink_binance_basis",
    "chainlink_binance_basis_ewma",
    "poly_mid",
    "poly_spread",
    "poly_mid_velocity",
    "polymarket_extremity",
    "polymarket_one_sided_imbalance",
    "distance_from_strike_in_sigma",
    "abs_distance_to_strike_in_sigma",
    "binance_volume_zscore_60s",
    "manipulation_setup_score",
    "binance_aggressive_buy_volume_5s",
    "binance_aggressive_sell_volume_5s",
    "binance_trade_count_zscore_1s",
    "binance_largest_trade_5s",
    "btc_log_return_1s",
    "btc_log_return_5s",
    "inprogress_swing_score",
]

# Monotone constraints indexed to FEATURE_NAMES: +1 increasing, -1 decreasing, 0 free
_MONOTONE = [0] * len(FEATURE_NAMES)
_MONOTONE[FEATURE_NAMES.index("momentum_30s")] = 1
_MONOTONE[FEATURE_NAMES.index("distance_from_strike_in_sigma")] = 1


class LGBMResidualModel:
    """LightGBM binary classifier trained to predict actual_outcome.

    Cold-start (untrained): predict_residual returns 0.0 → GBM-only mode.
    After training on logged tick data: provides microstructure correction.
    """

    def __init__(self) -> None:
        self._model = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit on (features, binary outcome) arrays.

        X: shape (n_samples, len(FEATURE_NAMES)), NaN OK (LightGBM handles them)
        y: shape (n_samples,), values 0.0 or 1.0
        """
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("lightgbm required for training") from exc

        # Ticks within a window share the same outcome label → high correlation.
        # Heavy regularisation prevents memorising window-level patterns.
        self._model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=16,           # shallow trees → less overfit
            min_child_samples=200,   # each leaf needs 200 samples minimum
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=1.0,           # L1 regularisation
            reg_lambda=5.0,          # L2 regularisation
            monotone_constraints=_MONOTONE,
            monotone_constraints_method="advanced",
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X, y, feature_name=FEATURE_NAMES)
        self._is_trained = True

    def predict_proba_up(self, row: dict) -> float:
        """P(UP) from ML model alone. Returns 0.5 if not trained."""
        if not self._is_trained or self._model is None:
            return 0.5
        import warnings
        vals = [v if (v := row.get(f)) is not None else np.nan for f in FEATURE_NAMES]
        x = np.array([vals], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(self._model.predict_proba(x)[0, 1])

    def predict_residual(self, row: dict, gbm_p: float) -> float:
        """ML correction on top of GBM. Returns 0.0 if not trained."""
        if not self._is_trained:
            return 0.0
        return self.predict_proba_up(row) - gbm_p

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "trained": self._is_trained}, path)

    def load(self, path: Path) -> None:
        data = joblib.load(Path(path))
        self._model = data["model"]
        self._is_trained = data["trained"]
