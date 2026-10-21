"""
EnsembleModel — combines GBM baseline + LightGBM residual + isotonic calibration.

v1 cold-start: ML residual = 0.0, calibrator = identity → pure GBM.
After training: P_up = clip(gbm_p_up + ml_residual, 0.01, 0.99), then calibrated.

Output: ModelOutput(p_up, ci_low, ci_high, gbm_p_up, ml_residual)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quantie.models.calibration import IsotonicCalibrator
from quantie.models.gbm import gbm_p_up
from quantie.models.ml_residual import LGBMResidualModel

_CLIP_LO = 0.01
_CLIP_HI = 0.99

# CI half-width constants.
# _CI_UNTRAINED:        cold start — no model loaded yet.
# _CI_TRAINED_FALLBACK: model loaded but model_meta.json absent (legacy pkl only).
# Real value comes from model_meta.json written by train_model.py.
_CI_UNTRAINED = 0.15
_CI_TRAINED_FALLBACK = 0.08


@dataclass(frozen=True)
class ModelOutput:
    p_up: float          # final calibrated P(UP) ∈ [0.01, 0.99]
    ci_low: float        # lower confidence bound
    ci_high: float       # upper confidence bound
    gbm_p_up: float      # raw GBM component (diagnostics)
    ml_residual: float   # ML correction applied (diagnostics)

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low


class EnsembleModel:
    """Two-layer fair value model.

    Usage:
        model = EnsembleModel()
        out = model.predict(fv_row, S0=60000, S_t=60050, sigma=0.0003, tau=120)
        # out.p_up, out.ci_width available immediately (GBM-only until trained)
    """

    def __init__(self) -> None:
        self._ml = LGBMResidualModel()
        self._cal = IsotonicCalibrator()
        self._ci_half_width: float = _CI_UNTRAINED

    @property
    def is_trained(self) -> bool:
        return self._ml.is_trained

    def predict(
        self,
        feature_row: dict,
        S0: float,
        S_t: float,
        sigma_per_sec: float,
        tau_sec: float,
        drift_per_sec: float = 0.0,
    ) -> ModelOutput:
        """Compute P(UP) with confidence interval.

        Args:
            feature_row:   FeatureVector.to_lgbm_row()
            S0:            window open price (Chainlink)
            S_t:           current Chainlink price
            sigma_per_sec: per-second vol from RealizedVolEstimator
            tau_sec:       seconds remaining
            drift_per_sec: optional drift (default 0)
        """
        gbm = gbm_p_up(S0, S_t, sigma_per_sec, tau_sec, drift_per_sec)
        residual = self._ml.predict_residual(feature_row, gbm)
        raw = float(np.clip(gbm + residual, _CLIP_LO, _CLIP_HI))
        calibrated = float(np.clip(self._cal.transform(raw), _CLIP_LO, _CLIP_HI))

        half_ci = self._ci_half_width
        return ModelOutput(
            p_up=calibrated,
            ci_low=max(_CLIP_LO, calibrated - half_ci),
            ci_high=min(_CLIP_HI, calibrated + half_ci),
            gbm_p_up=gbm,
            ml_residual=residual,
        )

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self._ml.save(model_dir / "lgbm_residual.pkl")
        self._cal.save(model_dir / "calibrator.pkl")

    def load(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        ml_path = model_dir / "lgbm_residual.pkl"
        cal_path = model_dir / "calibrator.pkl"
        meta_path = model_dir / "model_meta.json"
        if ml_path.exists():
            self._ml.load(ml_path)
        if cal_path.exists():
            self._cal.load(cal_path)
        # Load real CI half-width computed during training.
        # Falls back to _CI_TRAINED_FALLBACK if file absent (legacy models).
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._ci_half_width = float(meta["ci_half_width"])
        elif self._ml.is_trained:
            self._ci_half_width = _CI_TRAINED_FALLBACK
