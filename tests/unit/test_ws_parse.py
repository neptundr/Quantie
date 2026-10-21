"""Unit tests for WebSocket message parsers (no network required)."""
import pytest

from quantie.data.binance_ws import _parse_agg_trade, _parse_depth
from quantie.data.event_bus import (
    BinanceDepth,
    BinanceTrade,
    ChainlinkPrice,
    PolyBookUpdate,
    PolyPriceChange,
    PolyTrade,
)
from quantie.data.polymarket_clob_ws import (
    _parse_book,
    _parse_market_message,
    _parse_price_change,
    _parse_trade,
)
from quantie.data.polymarket_rtds_ws import _parse_chainlink


# ---------------------------------------------------------------------------
# Binance aggTrade
# ---------------------------------------------------------------------------

_AGG_TRADE = {
    "e": "aggTrade", "E": 1714046500000, "s": "BTCUSDT",
    "a": 1234567, "p": "60000.50", "q": "0.025",
    "f": 100, "l": 105, "T": 1714046499900, "m": False, "M": True,
}

def test_parse_agg_trade_happy_path():
    ev = _parse_agg_trade(_AGG_TRADE)
    assert isinstance(ev, BinanceTrade)
    assert ev.ts_ms == 1714046500000
    assert ev.trade_ts_ms == 1714046499900
    assert ev.price == pytest.approx(60000.50)
    assert ev.qty == pytest.approx(0.025)
    assert ev.is_buyer_maker is False

def test_parse_agg_trade_buyer_maker_true():
    ev = _parse_agg_trade({**_AGG_TRADE, "m": True})
    assert ev is not None and ev.is_buyer_maker is True

def test_parse_agg_trade_missing_field_returns_none():
    assert _parse_agg_trade({}) is None

def test_parse_agg_trade_invalid_price_returns_none():
    assert _parse_agg_trade({**_AGG_TRADE, "p": "not_a_number"}) is None


# ---------------------------------------------------------------------------
# Binance depth
# ---------------------------------------------------------------------------

_DEPTH = {
    "lastUpdateId": 9999,
    "bids": [["60000.00", "1.5"], ["59999.00", "0.5"]],
    "asks": [["60001.00", "2.0"], ["60002.00", "0.8"]],
}

def test_parse_depth_happy_path():
    ev = _parse_depth(_DEPTH)
    assert isinstance(ev, BinanceDepth)
    assert ev.bids[0] == pytest.approx((60000.0, 1.5))
    assert ev.asks[0] == pytest.approx((60001.0, 2.0))

def test_parse_depth_zero_qty_filtered():
    payload = {
        "lastUpdateId": 1,
        "bids": [["60000.00", "0.0"], ["59999.00", "1.0"]],
        "asks": [["60001.00", "0.0"]],
    }
    ev = _parse_depth(payload)
    assert ev is not None
    assert len(ev.bids) == 1
    assert len(ev.asks) == 0

def test_parse_depth_missing_field_returns_none():
    assert _parse_depth({}) is None


# ---------------------------------------------------------------------------
# Chainlink RTDS
# ---------------------------------------------------------------------------

_RTDS = {
    "topic": "crypto_prices_chainlink", "type": "update",
    "timestamp": 1714046500237,
    "payload": {"symbol": "btc/usd", "timestamp": 1714046500213, "value": 60123.45},
}

def test_parse_chainlink_happy_path():
    ev = _parse_chainlink(_RTDS, received_ts_ms=1714046500300)
    assert isinstance(ev, ChainlinkPrice)
    assert ev.payload_ts_ms == 1714046500213
    assert ev.received_ts_ms == 1714046500300
    assert ev.price == pytest.approx(60123.45)

def test_parse_chainlink_wrong_topic_returns_none():
    assert _parse_chainlink({**_RTDS, "topic": "other"}, received_ts_ms=0) is None

def test_parse_chainlink_missing_payload_returns_none():
    assert _parse_chainlink({"topic": "crypto_prices_chainlink"}, received_ts_ms=0) is None

def test_parse_chainlink_non_json_topic_returns_none():
    # "PONG" responses from the server should not reach _parse_chainlink
    # (filtered upstream), but guard here too.
    assert _parse_chainlink({}, received_ts_ms=0) is None


# ---------------------------------------------------------------------------
# CLOB book
# ---------------------------------------------------------------------------

_BOOK = {
    "event_type": "book",
    "asset_id": "71321045679252212594626385532706912750332728571942532289631379312455583992563",
    "market": "0x5f65177b394277fd294cd75650044e32ba009a95022d88a0c1d565897d72f8f1",
    "bids": [{"price": ".48", "size": "30"}, {"price": ".49", "size": "20"}],
    "asks": [{"price": ".52", "size": "25"}, {"price": ".53", "size": "60"}],
    "timestamp": "1714046500000", "hash": "0xabc123",
}

def test_parse_book_happy_path():
    ev = _parse_book(_BOOK)
    assert isinstance(ev, PolyBookUpdate)
    assert ev.ts_ms == 1714046500000
    assert ev.bids[0] == pytest.approx((0.48, 30.0))
    assert ev.asks[0] == pytest.approx((0.52, 25.0))
    assert ev.book_hash == "0xabc123"

def test_parse_book_missing_field_returns_none():
    assert _parse_book({}) is None


# ---------------------------------------------------------------------------
# CLOB price_change
# ---------------------------------------------------------------------------

_PRICE_CHANGE = {
    "event_type": "price_change",
    "market": "0x5f65177b394277fd294cd75650044e32ba009a95022d88a0c1d565897d72f8f1",
    "price_changes": [{"asset_id": "71321045", "price": "0.5", "size": "200",
                        "side": "BUY", "hash": "abc", "best_bid": "0.5", "best_ask": "0.52"}],
    "timestamp": "1714046500500",
}

def test_parse_price_change_happy_path():
    events = _parse_price_change(_PRICE_CHANGE)
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, PolyPriceChange)
    assert ev.price == pytest.approx(0.5)
    assert ev.best_bid == pytest.approx(0.5)
    assert ev.best_ask == pytest.approx(0.52)
    assert ev.side == "BUY"

def test_parse_price_change_empty_list():
    assert _parse_price_change({**_PRICE_CHANGE, "price_changes": []}) == []

def test_parse_price_change_missing_timestamp_returns_empty():
    msg = {k: v for k, v in _PRICE_CHANGE.items() if k != "timestamp"}
    assert _parse_price_change(msg) == []


# ---------------------------------------------------------------------------
# CLOB last_trade_price
# ---------------------------------------------------------------------------

_TRADE = {
    "event_type": "last_trade_price",
    "asset_id": "71321045679252212594626385532706912750332728571942532289631379312455583992563",
    "market": "0x5f65177b394277fd294cd75650044e32ba009a95022d88a0c1d565897d72f8f1",
    "price": "0.52", "size": "100.5", "side": "BUY",
    "fee_rate_bps": "0", "timestamp": "1714046501000",
}

def test_parse_trade_happy_path():
    ev = _parse_trade(_TRADE)
    assert isinstance(ev, PolyTrade)
    assert ev.ts_ms == 1714046501000
    assert ev.price == pytest.approx(0.52)
    assert ev.size == pytest.approx(100.5)
    assert ev.side == "BUY"

def test_parse_trade_missing_field_returns_none():
    assert _parse_trade({}) is None


# ---------------------------------------------------------------------------
# _parse_market_message dispatcher
# ---------------------------------------------------------------------------

def test_dispatcher_book():
    assert isinstance(_parse_market_message(_BOOK)[0], PolyBookUpdate)

def test_dispatcher_price_change():
    assert isinstance(_parse_market_message(_PRICE_CHANGE)[0], PolyPriceChange)

def test_dispatcher_trade():
    assert isinstance(_parse_market_message(_TRADE)[0], PolyTrade)

def test_dispatcher_unknown_type_returns_empty():
    assert _parse_market_message({"event_type": "tick_size_change"}) == []

def test_dispatcher_list_of_messages():
    events = _parse_market_message([_BOOK, _TRADE])
    assert len(events) == 2
    assert isinstance(events[0], PolyBookUpdate)
    assert isinstance(events[1], PolyTrade)
