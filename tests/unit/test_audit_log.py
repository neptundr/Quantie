"""Unit tests for audit_log.log_decision."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quantie.safety.audit_log import log_decision


def _call(**overrides) -> MagicMock:
    mock_logger = MagicMock()
    defaults = dict(
        ts_ms=1_700_000_000_000,
        window_start_ts=1_700_000_000,
        tau_sec=120.5,
        feature_row={"feat_x": 0.42, "feat_y": 0.01, "non_numeric": "skip"},
        p_up=0.60,
        gbm_p_up=0.55,
        ml_residual=0.05,
        ci_width=0.08,
        action="BUY_UP",
        size_usd=25.0,
        edge=0.12,
        regime="TRENDING",
        reason="edge above threshold",
    )
    defaults.update(overrides)
    with patch("quantie.safety.audit_log._audit", mock_logger):
        log_decision(**defaults)
    return mock_logger


class TestLogDecisionCalled:
    def test_calls_audit_logger_once(self) -> None:
        mock = _call()
        mock.info.assert_called_once()

    def test_event_name_is_decision(self) -> None:
        mock = _call()
        args, _ = mock.info.call_args
        assert args[0] == "decision"


class TestCoreFields:
    def test_ts_ms_passed(self) -> None:
        mock = _call(ts_ms=999)
        _, kw = mock.info.call_args
        assert kw["ts_ms"] == 999

    def test_p_up_passed(self) -> None:
        mock = _call(p_up=0.7)
        _, kw = mock.info.call_args
        assert kw["p_up"] == pytest.approx(0.7, abs=1e-4)

    def test_action_passed(self) -> None:
        mock = _call(action="HOLD")
        _, kw = mock.info.call_args
        assert kw["action"] == "HOLD"

    def test_regime_passed(self) -> None:
        mock = _call(regime="VOLATILE")
        _, kw = mock.info.call_args
        assert kw["regime"] == "VOLATILE"

    def test_tau_sec_rounded_to_2dp(self) -> None:
        mock = _call(tau_sec=99.9876)
        _, kw = mock.info.call_args
        assert kw["tau_sec"] == pytest.approx(99.99, abs=0.01)

    def test_p_up_rounded_to_4dp(self) -> None:
        mock = _call(p_up=0.123456789)
        _, kw = mock.info.call_args
        assert kw["p_up"] == pytest.approx(0.1235, abs=1e-4)


class TestFeatureHandling:
    def test_numeric_features_included(self) -> None:
        mock = _call()
        _, kw = mock.info.call_args
        assert "feat_x" in kw
        assert kw["feat_x"] == pytest.approx(0.42, abs=1e-5)

    def test_non_numeric_features_excluded(self) -> None:
        mock = _call()
        _, kw = mock.info.call_args
        assert "non_numeric" not in kw

    def test_empty_feature_row_ok(self) -> None:
        mock = _call(feature_row={})
        mock.info.assert_called_once()

    def test_feature_values_rounded_to_6dp(self) -> None:
        mock = _call(feature_row={"f": 0.123456789})
        _, kw = mock.info.call_args
        assert kw["f"] == pytest.approx(0.123457, abs=1e-6)
