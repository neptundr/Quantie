"""Unit tests for TickStore — per-window parquet persistence."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from quantie.data.tick_store import TickStore


def _record(store: TickStore, window_start_ts: int = 1000, ts_ms: int = 1001) -> None:
    store.record(
        ts_ms=ts_ms,
        window_start_ts=window_start_ts,
        feature_row={"feat_a": 0.5, "feat_b": 1.0, "label": "skip_me"},
        p_up=0.55,
        gbm_p_up=0.50,
        ml_residual=0.05,
        ci_width=0.10,
        action="BUY_UP",
        regime="NEUTRAL",
    )


class TestRecord:
    def test_buffer_grows(self) -> None:
        store = TickStore()
        assert len(store._rows) == 0
        _record(store)
        assert len(store._rows) == 1

    def test_outcome_defaults_to_nan(self) -> None:
        store = TickStore()
        _record(store)
        assert math.isnan(store._rows[0]["outcome"])

    def test_numeric_features_included(self) -> None:
        store = TickStore()
        _record(store)
        row = store._rows[0]
        assert row["feat_a"] == pytest.approx(0.5)
        assert row["feat_b"] == pytest.approx(1.0)

    def test_non_numeric_features_excluded(self) -> None:
        store = TickStore()
        _record(store)
        assert "label" not in store._rows[0]

    def test_core_fields_present(self) -> None:
        store = TickStore()
        _record(store, window_start_ts=2000, ts_ms=2001)
        row = store._rows[0]
        assert row["ts_ms"] == 2001
        assert row["window_start_ts"] == 2000
        assert row["p_up"] == pytest.approx(0.55)
        assert row["action"] == "BUY_UP"
        assert row["regime"] == "NEUTRAL"


class TestFlush:
    def test_flush_writes_parquet(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("quantie.data.tick_store._TICKS_DIR", tmp_path)
        store = TickStore()
        _record(store, window_start_ts=5000, ts_ms=5001)
        store.flush(5000, "UP")
        assert (tmp_path / "ticks_5000.parquet").exists()

    def test_flush_sets_outcome_up(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("quantie.data.tick_store._TICKS_DIR", tmp_path)
        store = TickStore()
        _record(store, window_start_ts=5000)
        store.flush(5000, "UP")
        df = pd.read_parquet(tmp_path / "ticks_5000.parquet")
        assert df["outcome"].iloc[0] == pytest.approx(1.0)

    def test_flush_sets_outcome_down(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("quantie.data.tick_store._TICKS_DIR", tmp_path)
        store = TickStore()
        _record(store, window_start_ts=5000)
        store.flush(5000, "DOWN")
        df = pd.read_parquet(tmp_path / "ticks_5000.parquet")
        assert df["outcome"].iloc[0] == pytest.approx(0.0)

    def test_flush_sets_outcome_nan_when_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("quantie.data.tick_store._TICKS_DIR", tmp_path)
        store = TickStore()
        _record(store, window_start_ts=5000)
        store.flush(5000, None)
        df = pd.read_parquet(tmp_path / "ticks_5000.parquet")
        assert math.isnan(df["outcome"].iloc[0])

    def test_flush_clears_buffer_for_window(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("quantie.data.tick_store._TICKS_DIR", tmp_path)
        store = TickStore()
        _record(store, window_start_ts=5000)
        _record(store, window_start_ts=6000)
        store.flush(5000, "UP")
        assert not any(r["window_start_ts"] == 5000 for r in store._rows)

    def test_flush_preserves_other_window_rows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("quantie.data.tick_store._TICKS_DIR", tmp_path)
        store = TickStore()
        _record(store, window_start_ts=5000)
        _record(store, window_start_ts=6000)
        store.flush(5000, "UP")
        assert sum(1 for r in store._rows if r["window_start_ts"] == 6000) == 1

    def test_flush_noop_when_no_rows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("quantie.data.tick_store._TICKS_DIR", tmp_path)
        store = TickStore()
        store.flush(9999, "UP")
        assert not list(tmp_path.glob("*.parquet"))

    def test_flush_multiple_rows_same_window(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("quantie.data.tick_store._TICKS_DIR", tmp_path)
        store = TickStore()
        for i in range(3):
            _record(store, window_start_ts=5000, ts_ms=5001 + i)
        store.flush(5000, "DOWN")
        df = pd.read_parquet(tmp_path / "ticks_5000.parquet")
        assert len(df) == 3
        assert df["outcome"].tolist() == pytest.approx([0.0, 0.0, 0.0])
