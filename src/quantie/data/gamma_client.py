"""
Gamma API client — resolves a window timestamp to Polymarket token IDs.

Endpoint:  https://gamma-api.polymarket.com
Key call:  GET /markets?slug=btc-updown-5m-{window_start_ts}
Response:  list of market dicts; first element has:
             clobTokenIds: ["<up_token_id>", "<down_token_id>"]

UP token is index 0, DOWN token is index 1 per Polymarket convention.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from quantie.clock import make_slug
from quantie.logging_config import get_logger

log = get_logger(__name__)

_BASE_URL = os.environ.get("GAMMA_BASE_URL", "https://gamma-api.polymarket.com")
_TIMEOUT = 10.0  # seconds


class GammaAPIError(Exception):
    """Raised when the Gamma API returns an unexpected response."""


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def get_market_by_slug(slug: str, client: Optional[httpx.Client] = None) -> dict:
    """Fetch a single market dict by slug. Retries on network errors.

    Args:
        slug:   e.g. "btc-updown-5m-1714000000"
        client: optional pre-constructed httpx.Client (for testing / connection reuse)

    Returns:
        Market dict with at minimum ``clobTokenIds`` list.

    Raises:
        GammaAPIError: market not found or malformed response.
        httpx.TransportError / httpx.TimeoutException: after retries exhausted.
    """
    _client = client or httpx.Client(base_url=_BASE_URL, timeout=_TIMEOUT)
    try:
        resp = _client.get("/markets", params={"slug": slug})
        resp.raise_for_status()
        data = resp.json()
    finally:
        if client is None:
            _client.close()

    if not isinstance(data, list) or len(data) == 0:
        raise GammaAPIError(f"no market found for slug={slug!r}")

    market = data[0]
    token_ids = market.get("clobTokenIds")
    # API sometimes returns clobTokenIds as a JSON-encoded string instead of a list
    if isinstance(token_ids, str):
        try:
            token_ids = json.loads(token_ids)
        except json.JSONDecodeError:
            pass
    if not isinstance(token_ids, list) or len(token_ids) < 2:
        raise GammaAPIError(
            f"clobTokenIds missing or malformed for slug={slug!r}: {token_ids!r}"
        )
    # Normalise back so resolve_window_tokens can safely index
    market = {**market, "clobTokenIds": token_ids}
    log.info("gamma_market_resolved", slug=slug, token_ids=token_ids)
    return market


def resolve_window_tokens(window_start_ts: int) -> tuple[str, str]:
    """Return (up_token_id, down_token_id) for a window timestamp.

    Raises:
        GammaAPIError: market not found or token IDs malformed.
        httpx.TransportError / httpx.TimeoutException: network failure after retries.
    """
    slug = make_slug(window_start_ts)
    market = get_market_by_slug(slug)
    token_ids: list[str] = market["clobTokenIds"]
    return token_ids[0], token_ids[1]
