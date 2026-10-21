"""Unit tests for LGBMResidualModel — dtype safety and cold-start behaviour."""
from __future__ import annotations

import numpy as np
import pytest

from quantie.models.ml_residual import FEATURE_NAMES, LGBMResidualModel


def _make_trained_model() -> LGBMResidualModel:
    """Train a minimal model on synthetic data."""
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        pytest.skip("lightgbm not installed")

    model = LGBMResidualModel()
    rng = np.random.default_rng(0)
    n = 400
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 2, size=n).astype(float)
    model.train(X, y)
    return model


class TestColdStart:
    def test_predict_proba_returns_half_when_untrained(self) -> None:
        model = LGBMResidualModel()
        row = {f: 0.0 for f in FEATURE_NAMES}
        assert model.predict_proba_up(row) == 0.5

    def test_predict_residual_returns_zero_when_untrained(self) -> None:
        model = LGBMResidualModel()
        row = {f: 0.0 for f in FEATURE_NAMES}
        assert model.predict_residual(row, gbm_p=0.6) == 0.0

    def test_is_trained_false_at_init(self) -> None:
        assert LGBMResidualModel().is_trained is False


class TestDtypeSafety:
    def test_none_values_do_not_raise(self) -> None:
        """None features must not cause LightGBM dtype error."""
        model = _make_trained_model()
        row = {f: None for f in FEATURE_NAMES}
        result = model.predict_proba_up(row)
        assert 0.0 <= result <= 1.0

    def test_mixed_none_and_float_do_not_raise(self) -> None:
        model = _make_trained_model()
        row = {f: (0.5 if i % 2 == 0 else None) for i, f in enumerate(FEATURE_NAMES)}
        result = model.predict_proba_up(row)
        assert 0.0 <= result <= 1.0

    def test_missing_keys_treated_as_nan(self) -> None:
        """Features absent from row dict must not raise."""
        model = _make_trained_model()
        row: dict = {}
        result = model.predict_proba_up(row)
        assert 0.0 <= result <= 1.0

    def test_result_is_valid_probability(self) -> None:
        model = _make_trained_model()
        row = {f: 0.1 for f in FEATURE_NAMES}
        result = model.predict_proba_up(row)
        assert 0.0 <= result <= 1.0
