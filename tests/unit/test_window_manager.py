"""Unit tests for WindowManager — no network, no real async loop needed."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from quantie.data.gamma_client import GammaAPIError
from quantie.state.window_state import CompletedWindow, WindowState
from quantie.state.polymarket_state import PolymarketState
from quantie.window_manager import WindowManager

_UP_ID = "71321045679252212594626385532706912750332728571942532289631379312455583992563"
_DOWN_ID = "16678291189211314787145083999015737376658799626183230671114808138726389093913"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(tmp_path):
    """WindowManager with external deps mocked, parquet dir set to tmp_path."""
    window_state = WindowState()
    clob_ws = MagicMock()
    poly_state = MagicMock(spec=PolymarketState)
    with patch("quantie.window_manager._WINDOWS_DIR", tmp_path):
        mgr = WindowManager(window_state, clob_ws, poly_state)
    return mgr, window_state, clob_ws, poly_state


# ---------------------------------------------------------------------------
# start() — initial token fetch
# ---------------------------------------------------------------------------

def test_start_calls_resubscribe_and_set_tokens(tmp_path):
    mgr, _, clob_ws, poly_state = _make_manager(tmp_path)
    with patch("quantie.window_manager.resolve_window_tokens", return_value=(_UP_ID, _DOWN_ID)):
        with patch("quantie.window_manager._WINDOWS_DIR", tmp_path):
            mgr.start(1714000000)
    clob_ws.resubscribe.assert_called_once_with(_UP_ID, _DOWN_ID)
    poly_state.set_tokens.assert_called_once_with(_UP_ID, _DOWN_ID)


def test_start_gamma_error_does_not_raise(tmp_path):
    mgr, _, clob_ws, poly_state = _make_manager(tmp_path)
    with patch(
        "quantie.window_manager.resolve_window_tokens",
        side_effect=GammaAPIError("not found"),
    ):
        mgr.start(1714000000)  # must not raise
    clob_ws.resubscribe.assert_not_called()
    poly_state.set_tokens.assert_not_called()


# ---------------------------------------------------------------------------
# _save_completed_window — parquet persistence
# ---------------------------------------------------------------------------

def test_save_completed_window_writes_parquet(tmp_path):
    mgr, _, _, _ = _make_manager(tmp_path)
    window = CompletedWindow(
        window_start_ts=1714000000,
        open_price=60000.0,
        close_price=60100.0,
        outcome="UP",
    )
    with patch("quantie.window_manager._WINDOWS_DIR", tmp_path):
        mgr._save_completed_window(window)

    path = tmp_path / "window_1714000000.parquet"
    assert path.exists()
    df = pd.read_parquet(path)
    assert df.iloc[0]["window_start_ts"] == 1714000000
    assert df.iloc[0]["open_price"] == pytest.approx(60000.0)
    assert df.iloc[0]["close_price"] == pytest.approx(60100.0)
    assert df.iloc[0]["outcome"] == "UP"
    assert df.iloc[0]["saved_at_ms"] > 0


def test_save_completed_window_none_outcome(tmp_path):
    mgr, _, _, _ = _make_manager(tmp_path)
    window = CompletedWindow(
        window_start_ts=1714000300,
        open_price=60000.0,
        close_price=None,
        outcome=None,
    )
    with patch("quantie.window_manager._WINDOWS_DIR", tmp_path):
        mgr._save_completed_window(window)

    path = tmp_path / "window_1714000300.parquet"
    assert path.exists()
    df = pd.read_parquet(path)
    assert pd.isna(df.iloc[0]["outcome"])


# ---------------------------------------------------------------------------
# _on_rollover — callback wiring
# ---------------------------------------------------------------------------

def test_on_rollover_saves_last_history_entry(tmp_path):
    mgr, window_state, _, _ = _make_manager(tmp_path)
    completed = CompletedWindow(1714000000, 60000.0, 60100.0, "UP")
    window_state.history.append(completed)

    loop = MagicMock()
    mgr._loop = loop
    with patch("quantie.window_manager.resolve_window_tokens", return_value=(_UP_ID, _DOWN_ID)):
        with patch("quantie.window_manager._WINDOWS_DIR", tmp_path):
            mgr._on_rollover(1714000300)

    path = tmp_path / "window_1714000000.parquet"
    assert path.exists()
    loop.run_in_executor.assert_called_once()


def test_on_rollover_empty_history_no_save(tmp_path):
    mgr, window_state, _, _ = _make_manager(tmp_path)
    assert len(window_state.history) == 0

    loop = MagicMock()
    mgr._loop = loop
    with patch("quantie.window_manager.resolve_window_tokens", return_value=(_UP_ID, _DOWN_ID)):
        mgr._on_rollover(1714000300)

    assert list(tmp_path.glob("*.parquet")) == []
    loop.run_in_executor.assert_called_once()
