"""
Isotonic calibration — ensures model probabilities are well-calibrated.

Goal: "when the model says 0.62, it really means 62%".
Train on a held-out set after LightGBM training; apply on every inference.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


class IsotonicCalibrator:
    """Thin wrapper around sklearn IsotonicRegression.

    Fit on (raw_scores, binary_outcomes) from a held-out validation set.
    Transform on inference-time scores before passing to ensemble.
    """

    def __init__(self) -> None:
        self._cal = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, y_scores: np.ndarray, y_true: np.ndarray) -> None:
        """Fit isotonic regression.

        y_scores: raw model probabilities, shape (n,)
        y_true:   binary outcomes (0/1), shape (n,)
        """
        from sklearn.isotonic import IsotonicRegression

        self._cal = IsotonicRegression(out_of_bounds="clip")
        self._cal.fit(y_scores, y_true)
        self._is_fitted = True

    def transform(self, score: float) -> float:
        """Map raw score → calibrated probability. Returns score if not fitted."""
        if not self._is_fitted or self._cal is None:
            return float(score)
        result = self._cal.transform(np.array([score]))
        return float(np.clip(result[0], 0.0, 1.0))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"cal": self._cal, "fitted": self._is_fitted}, path)

    def load(self, path: Path) -> None:
        data = joblib.load(Path(path))
        self._cal = data["cal"]
        self._is_fitted = data["fitted"]
