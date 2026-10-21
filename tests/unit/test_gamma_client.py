"""Unit tests for GammaClient — no network required (httpx mocked with respx)."""
from __future__ import annotations

import pytest
import httpx
import respx

from quantie.data.gamma_client import (
    GammaAPIError,
    get_market_by_slug,
    resolve_window_tokens,
)
import quantie.data.gamma_client as gc

_BASE = "https://gamma-api.polymarket.com"
_UP_ID = "71321045679252212594626385532706912750332728571942532289631379312455583992563"
_DOWN_ID = "16678291189211314787145083999015737376658799626183230671114808138726389093913"

_GOOD_RESPONSE = [
    {
        "slug": "btc-updown-5m-1714000000",
        "clobTokenIds": [_UP_ID, _DOWN_ID],
        "conditionId": "0xabc",
    }
]


# ---------------------------------------------------------------------------
# get_market_by_slug
# ---------------------------------------------------------------------------

@respx.mock
def test_get_market_by_slug_happy_path():
    respx.get(f"{_BASE}/markets").mock(
        return_value=httpx.Response(200, json=_GOOD_RESPONSE)
    )
    client = httpx.Client(base_url=_BASE)
    market = get_market_by_slug("btc-updown-5m-1714000000", client=client)
    assert market["clobTokenIds"] == [_UP_ID, _DOWN_ID]
    client.close()


@respx.mock
def test_get_market_by_slug_empty_list_raises():
    respx.get(f"{_BASE}/markets").mock(return_value=httpx.Response(200, json=[]))
    client = httpx.Client(base_url=_BASE)
    with pytest.raises(GammaAPIError, match="no market found"):
        get_market_by_slug("btc-updown-5m-0000000000", client=client)
    client.close()


@respx.mock
def test_get_market_by_slug_missing_token_ids_raises():
    respx.get(f"{_BASE}/markets").mock(
        return_value=httpx.Response(
            200, json=[{"slug": "x", "clobTokenIds": ["only_one"]}]
        )
    )
    client = httpx.Client(base_url=_BASE)
    with pytest.raises(GammaAPIError, match="malformed"):
        get_market_by_slug("x", client=client)
    client.close()


@respx.mock
def test_get_market_by_slug_404_raises():
    respx.get(f"{_BASE}/markets").mock(return_value=httpx.Response(404))
    client = httpx.Client(base_url=_BASE)
    with pytest.raises(httpx.HTTPStatusError):
        get_market_by_slug("no-such-slug", client=client)
    client.close()


@respx.mock
def test_get_market_by_slug_no_clob_token_ids_key():
    respx.get(f"{_BASE}/markets").mock(
        return_value=httpx.Response(200, json=[{"slug": "x"}])
    )
    client = httpx.Client(base_url=_BASE)
    with pytest.raises(GammaAPIError, match="malformed"):
        get_market_by_slug("x", client=client)
    client.close()


# ---------------------------------------------------------------------------
# resolve_window_tokens
# ---------------------------------------------------------------------------

def test_resolve_window_tokens_uses_correct_slug():
    """Slug built by make_slug matches expected format."""
    from quantie.clock import make_slug
    assert make_slug(1714000000) == "btc-updown-5m-1714000000"


def test_resolve_window_tokens_returns_up_down(monkeypatch):
    monkeypatch.setattr(
        gc,
        "get_market_by_slug",
        lambda slug, client=None: {"clobTokenIds": [_UP_ID, _DOWN_ID]},
    )
    up, down = resolve_window_tokens(1714000000)
    assert up == _UP_ID
    assert down == _DOWN_ID


def test_resolve_window_tokens_propagates_gamma_error(monkeypatch):
    def _raise(slug, client=None):
        raise GammaAPIError("not found")

    monkeypatch.setattr(gc, "get_market_by_slug", _raise)
    with pytest.raises(GammaAPIError):
        resolve_window_tokens(1714000000)
