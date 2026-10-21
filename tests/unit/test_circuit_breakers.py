"""Unit tests for CircuitBreaker."""
from __future__ import annotations

import time

import pytest

from quantie.execution.broker import STATUS_CANCELLED, STATUS_FILLED, STATUS_OPEN, STATUS_REJECTED
from quantie.safety.circuit_breakers import (
    CircuitBreaker,
    _ERROR_MIN_SAMPLES,
    _LATENCY_MIN_SAMPLES,
    _LATENCY_P99_HALT_MS,
    _LATENCY_RECOVERY_MIN_SAMPLES,
    _PAUSE_SECONDS,
)


def _make_cb() -> CircuitBreaker:
    return CircuitBreaker(max_drawdown_frac=0.20)


class TestInitialState:
    def test_not_halted_at_start(self) -> None:
        cb = _make_cb()
        assert cb.is_halted is False

    def test_halt_reason_empty_when_not_halted(self) -> None:
        cb = _make_cb()
        assert cb.halt_reason == ""


class TestDrawdownBreaker:
    def test_no_trip_below_limit(self) -> None:
        cb = _make_cb()
        cb.check_drawdown(realized_pnl=-150.0, bankroll=1000.0)
        assert cb.is_halted is False

    def test_trips_at_limit(self) -> None:
        cb = _make_cb()
        cb.check_drawdown(0.0, 1000.0)
        cb.check_drawdown(-200.0, 1000.0)
        assert cb.is_halted is True

    def test_trips_above_limit(self) -> None:
        cb = _make_cb()
        cb.check_drawdown(-300.0, 1000.0)
        assert cb.is_halted is True

    def test_peak_tracked_correctly(self) -> None:
        cb = _make_cb()
        cb.check_drawdown(100.0, 1000.0)
        cb.check_drawdown(-101.0, 1000.0)  # drawdown = 201 >= 200 limit
        assert cb.is_halted is True

    def test_does_not_re_trip_when_already_halted(self) -> None:
        cb = _make_cb()
        cb.check_drawdown(-300.0, 1000.0)
        first_paused_until = cb._paused_until
        cb.check_drawdown(-500.0, 1000.0)
        assert cb._paused_until == first_paused_until

    def test_halt_reason_contains_drawdown_info(self) -> None:
        cb = _make_cb()
        cb.check_drawdown(-300.0, 1000.0)
        assert "drawdown" in cb.halt_reason

    def test_paused_for_one_hour(self) -> None:
        cb = _make_cb()
        before = time.time()
        cb.check_drawdown(-300.0, 1000.0)
        assert cb._paused_until >= before + _PAUSE_SECONDS - 1


class TestErrorRateBreaker:
    def test_no_trip_below_min_samples(self) -> None:
        cb = _make_cb()
        for _ in range(_ERROR_MIN_SAMPLES - 1):
            cb.record_order_result(was_rejected=True)
        assert cb.is_halted is False

    def test_no_trip_below_threshold(self) -> None:
        cb = _make_cb()
        # 1 rejection out of 10 = 10% — not > threshold
        for i in range(_ERROR_MIN_SAMPLES):
            cb.record_order_result(was_rejected=(i == 0))
        assert cb.is_halted is False

    def test_trips_above_threshold(self) -> None:
        cb = _make_cb()
        n = _ERROR_MIN_SAMPLES
        # 2 rejections out of 10 = 20% > 10%
        for i in range(n):
            cb.record_order_result(was_rejected=(i < 2))
        assert cb.is_halted is True

    def test_old_samples_pruned(self) -> None:
        cb = _make_cb()
        old_time = time.time() - 70
        for _ in range(_ERROR_MIN_SAMPLES):
            cb._order_outcomes.append((old_time, True))
        for _ in range(_ERROR_MIN_SAMPLES):
            cb.record_order_result(was_rejected=False)
        assert cb.is_halted is False


class TestLatencyBreaker:
    def test_no_trip_below_min_samples(self) -> None:
        cb = _make_cb()
        for _ in range(_LATENCY_MIN_SAMPLES - 1):
            cb.record_binance_latency(1000.0)
        assert cb.is_halted is False

    def test_no_trip_below_p99_threshold(self) -> None:
        cb = _make_cb()
        for _ in range(_LATENCY_MIN_SAMPLES):
            cb.record_binance_latency(100.0)
        assert cb.is_halted is False

    def test_trips_when_p99_exceeds_limit(self) -> None:
        cb = _make_cb()
        # p99 index = int(N * 0.99) - 1; need ≥2 high values to reach that index
        for _ in range(_LATENCY_MIN_SAMPLES - 2):
            cb.record_binance_latency(10.0)
        for _ in range(2):
            cb.record_binance_latency(_LATENCY_P99_HALT_MS + 1)
        assert cb.is_halted is True

    def test_halt_reason_contains_latency_info(self) -> None:
        cb = _make_cb()
        for _ in range(_LATENCY_MIN_SAMPLES - 2):
            cb.record_binance_latency(10.0)
        for _ in range(2):
            cb.record_binance_latency(9999.0)
        assert "latency" in cb.halt_reason

    def test_old_latency_samples_pruned(self) -> None:
        cb = _make_cb()
        old_time = time.time() - 400
        for _ in range(_LATENCY_MIN_SAMPLES):
            cb._binance_latencies.append((old_time, 9999.0))
        for _ in range(_LATENCY_MIN_SAMPLES):
            cb.record_binance_latency(10.0)
        assert cb.is_halted is False

    def test_latency_recovery_clears_halt_without_waiting_full_window(self) -> None:
        """Once halted, fresh good samples in the 30s recovery window clear the halt immediately.

        Old behaviour: halt persisted until the 5-min detection window drained (up to 300s).
        New behaviour: 5+ recent samples all below threshold → resume right away.
        """
        cb = _make_cb()
        # Trip: inject enough old high-latency samples to exceed p99 threshold
        now = time.time()
        # 60s ago: within 300s detection window but outside 30s recovery window
        old_time = now - 60
        for _ in range(_LATENCY_MIN_SAMPLES - 2):
            cb._binance_latencies.append((old_time, 10.0))
        for _ in range(2):
            cb._binance_latencies.append((old_time, 9999.0))
        # Force the trip by recording one more sample that triggers the check
        cb.record_binance_latency(9999.0)
        assert cb.is_halted is True
        assert cb.halt_kind == "latency"

        # Now latency recovers: send enough recent good samples to satisfy recovery window
        for _ in range(_LATENCY_RECOVERY_MIN_SAMPLES):
            cb.record_binance_latency(10.0)

        assert cb.is_halted is False


class TestIsHaltedExpiry:
    def test_halted_returns_false_after_pause_window(self) -> None:
        cb = _make_cb()
        cb.check_drawdown(-300.0, 1000.0)
        assert cb.is_halted is True
        cb._paused_until = time.time() - 1
        assert cb.is_halted is False
        assert cb.halt_reason == ""


class TestRecordOrderStatus:
    def test_all_open_no_trip(self) -> None:
        cb = _make_cb()
        for _ in range(_ERROR_MIN_SAMPLES * 2):
            cb.record_order_status(STATUS_OPEN)
        assert cb.is_halted is False

    def test_all_filled_no_trip(self) -> None:
        cb = _make_cb()
        for _ in range(_ERROR_MIN_SAMPLES * 2):
            cb.record_order_status(STATUS_FILLED)
        assert cb.is_halted is False

    def test_all_cancelled_trips(self) -> None:
        cb = _make_cb()
        for _ in range(_ERROR_MIN_SAMPLES):
            cb.record_order_status(STATUS_CANCELLED)
        assert cb.is_halted is True

    def test_all_rejected_trips(self) -> None:
        cb = _make_cb()
        for _ in range(_ERROR_MIN_SAMPLES):
            cb.record_order_status(STATUS_REJECTED)
        assert cb.is_halted is True

    def test_mix_cancelled_and_open_trips_when_rate_exceeded(self) -> None:
        cb = _make_cb()
        # 3 cancelled out of 10 = 30% > 10% threshold
        for i in range(_ERROR_MIN_SAMPLES):
            cb.record_order_status(STATUS_CANCELLED if i < 3 else STATUS_OPEN)
        assert cb.is_halted is True

    def test_below_threshold_does_not_trip(self) -> None:
        cb = _make_cb()
        # 1 cancelled out of 10 = 10% — not > threshold
        for i in range(_ERROR_MIN_SAMPLES):
            cb.record_order_status(STATUS_CANCELLED if i == 0 else STATUS_FILLED)
        assert cb.is_halted is False


class TestObservability:
    def test_halt_kind_empty_when_not_halted(self) -> None:
        cb = _make_cb()
        assert cb.halt_kind == ""

    def test_halted_at_zero_before_trip(self) -> None:
        cb = _make_cb()
        assert cb.halted_at == 0.0

    def test_seconds_until_resume_zero_when_not_halted(self) -> None:
        cb = _make_cb()
        assert cb.seconds_until_resume == 0.0

    def test_halt_kind_drawdown_on_drawdown_trip(self) -> None:
        cb = _make_cb()
        cb.check_drawdown(-300.0, 1000.0)
        assert cb.halt_kind == "drawdown"

    def test_halt_kind_error_rate_on_rejection_trip(self) -> None:
        cb = _make_cb()
        for _ in range(_ERROR_MIN_SAMPLES):
            cb.record_order_status(STATUS_REJECTED)
        assert cb.halt_kind == "error_rate"

    def test_halted_at_set_on_trip(self) -> None:
        cb = _make_cb()
        before = time.time()
        cb.check_drawdown(-300.0, 1000.0)
        assert cb.halted_at >= before

    def test_seconds_until_resume_positive_when_halted(self) -> None:
        cb = _make_cb()
        cb.check_drawdown(-300.0, 1000.0)
        assert cb.seconds_until_resume > 0.0

    def test_halt_kind_clears_after_expiry(self) -> None:
        cb = _make_cb()
        cb.check_drawdown(-300.0, 1000.0)
        cb._paused_until = time.time() - 1
        assert cb.halt_kind == ""

    def test_metadata_when_not_halted(self) -> None:
        cb = _make_cb()
        assert cb.halt_kind == ""
        assert cb.halt_reason == ""
        assert cb.seconds_until_resume == 0.0
