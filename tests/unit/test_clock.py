"""Unit tests for quantie.clock using freezegun."""
import pytest
from freezegun import freeze_time

from quantie.clock import (
    WINDOW_SECONDS,
    current_window_start_ts,
    elapsed_in_window,
    make_slug,
    next_window_start_ts,
    time_to_resolution,
)


# ---------------------------------------------------------------------------
# current_window_start_ts
# ---------------------------------------------------------------------------

@freeze_time("2024-04-25 12:00:00")  # Unix 1714046400 — divisible by 300
def test_window_start_at_exact_boundary():
    ts = current_window_start_ts()
    assert ts == 1714046400
    assert ts % WINDOW_SECONDS == 0


@freeze_time("2024-04-25 12:02:30")  # 150 s into window starting at 12:00:00
def test_window_start_mid_window():
    ts = current_window_start_ts()
    assert ts == 1714046400
    assert ts % WINDOW_SECONDS == 0


@freeze_time("2024-04-25 12:04:59.999")  # 1 ms before next window
def test_window_start_just_before_rollover():
    ts = current_window_start_ts()
    assert ts == 1714046400


@freeze_time("2024-04-25 12:05:00")  # exactly the next window boundary
def test_window_start_at_next_boundary():
    ts = current_window_start_ts()
    assert ts == 1714046700  # 1714046400 + 300


def test_window_start_accepts_explicit_now():
    ts = current_window_start_ts(now=1714046401.0)
    assert ts == 1714046400


# ---------------------------------------------------------------------------
# next_window_start_ts
# ---------------------------------------------------------------------------

@freeze_time("2024-04-25 12:02:30")
def test_next_window_start():
    nxt = next_window_start_ts()
    cur = current_window_start_ts()
    assert nxt == cur + WINDOW_SECONDS


# ---------------------------------------------------------------------------
# time_to_resolution
# ---------------------------------------------------------------------------

@freeze_time("2024-04-25 12:00:00")  # at window start → 300s remain
def test_tau_at_window_start():
    tau = time_to_resolution()
    assert tau == pytest.approx(300.0, abs=0.01)


@freeze_time("2024-04-25 12:02:30")  # 150 s in → 150s remain
def test_tau_mid_window():
    tau = time_to_resolution()
    assert tau == pytest.approx(150.0, abs=0.01)


@freeze_time("2024-04-25 12:04:59")  # 299s in → 1s remain
def test_tau_near_expiry():
    tau = time_to_resolution()
    assert tau == pytest.approx(1.0, abs=0.01)


def test_tau_one_second_into_next_window():
    # At window_start + 301 we are in the NEXT window (started at +300).
    # That window has 299 seconds left.
    tau = time_to_resolution(now=1714046400 + 301.0)
    assert tau == pytest.approx(299.0, abs=0.01)


def test_tau_at_new_window_boundary():
    # Exactly at a window boundary we are at the START of a fresh window.
    # tau should be the full window length.
    tau = time_to_resolution(now=float(1714046400 + 300))
    assert tau == pytest.approx(300.0, abs=0.001)


# ---------------------------------------------------------------------------
# make_slug
# ---------------------------------------------------------------------------

def test_make_slug_format():
    slug = make_slug(1714046400)
    assert slug == "btc-updown-5m-1714046400"


def test_make_slug_uses_window_start():
    ts = 1714046700
    assert make_slug(ts) == f"btc-updown-5m-{ts}"


# ---------------------------------------------------------------------------
# elapsed_in_window
# ---------------------------------------------------------------------------

@freeze_time("2024-04-25 12:00:00")
def test_elapsed_at_window_start():
    assert elapsed_in_window() == pytest.approx(0.0, abs=0.01)


@freeze_time("2024-04-25 12:02:30")  # 150 s in
def test_elapsed_mid_window():
    assert elapsed_in_window() == pytest.approx(150.0, abs=0.01)


def test_elapsed_plus_tau_equals_window_seconds():
    now = 1714046400 + 123.456
    elapsed = elapsed_in_window(now=now)
    tau = time_to_resolution(now=now)
    assert elapsed + tau == pytest.approx(WINDOW_SECONDS, abs=0.001)
