"""
Live CLOB broker — typed wrapper around py-clob-client-v2 v1.0.1.

Credentials from environment variables (never hardcoded):
  POLY_PRIVATE_KEY    — EVM private key exported from Magic wallet
  POLY_API_KEY        — Polymarket API key
  POLY_API_SECRET     — Polymarket API secret
  POLY_API_PASSPHRASE — Polymarket API passphrase
  POLY_FUNDER         — proxy wallet address shown in Polymarket profile (checksummed)
  POLY_CHAIN_ID       — 137 (Polygon mainnet) or 80002 (Amoy testnet)

Signature type: 1 (POLY_PROXY) — for Google/Magic wallet accounts.

Retry policy (tenacity):
  3 attempts, exponential backoff 1→2→4s.
  Retries only on network/timeout errors — NEVER on signature or auth errors.

Order sizing:
  size_usd converted to shares: shares = round(size_usd / price, 2)
  Minimum 1 share enforced to avoid API rejection.
"""
from __future__ import annotations

import math
import os
from typing import Any

import structlog
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from quantie.execution.broker import (
    SIDE_BUY,
    SIDE_SELL,
    STATUS_CANCELLED,
    STATUS_FILLED,
    STATUS_OPEN,
    STATUS_REJECTED,
    OrderResult,
)

log = structlog.get_logger(__name__)

_CLOB_HOST = "https://clob.polymarket.com"
_CHAIN_ID_MAINNET = 137
_RETRY_ATTEMPTS = 3
_RETRY_MIN_WAIT = 1.0
_RETRY_MAX_WAIT = 4.0

_NON_RETRYABLE = ("signature", "auth", "unauthorized", "forbidden", "invalid key")


def _is_retryable(exc: BaseException) -> bool:
    """Only retry transient errors, never auth/signature failures."""
    msg = str(exc).lower()
    return not any(kw in msg for kw in _NON_RETRYABLE)


class ClobClient:
    """Live Polymarket CLOB broker.

    Usage:
        client = ClobClient.from_env()
        result = await client.place_fok_buy(asset_id, price=0.60, size_usd=25.0)
    """

    def __init__(self, inner: Any) -> None:
        self._c = inner

    @classmethod
    def from_env(cls) -> "ClobClient":
        """Construct from environment variables. Raises KeyError if any missing.

        api_secret and api_passphrase are optional — if absent, they are derived
        deterministically from the wallet private key via the CLOB /auth/derive-api-key
        endpoint. This is the standard flow for Google/Magic wallet accounts.
        """
        from py_clob_client_v2.client import ClobClient as _Inner
        from py_clob_client_v2.clob_types import ApiCreds

        pk = os.environ["POLY_PRIVATE_KEY"]
        chain_id = int(os.environ.get("POLY_CHAIN_ID", str(_CHAIN_ID_MAINNET)))
        funder = os.environ["POLY_FUNDER"]

        api_key = os.environ.get("POLY_API_KEY", "")
        api_secret = os.environ.get("POLY_API_SECRET", "")
        api_passphrase = os.environ.get("POLY_API_PASSPHRASE", "")

        if not (api_secret and api_passphrase):
            # Derive credentials deterministically from wallet private key
            log.info("clob.deriving_api_credentials")
            l1_client = _Inner(host=_CLOB_HOST, chain_id=chain_id, key=pk)
            derived: ApiCreds = l1_client.derive_api_key()
            api_key = derived.api_key
            api_secret = derived.api_secret
            api_passphrase = derived.api_passphrase
            log.info("clob.credentials_derived", api_key=api_key)

        creds = ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        )
        inner = _Inner(
            host=_CLOB_HOST,
            chain_id=chain_id,
            key=pk,
            creds=creds,
            signature_type=1,   # POLY_PROXY — Google/Magic wallet
            funder=funder,
        )
        return cls(inner)

    # ------------------------------------------------------------------
    # Broker interface
    # ------------------------------------------------------------------

    async def place_fok_buy(
        self,
        asset_id: str,
        price: float,
        size_usd: float,
        current_ask: float | None = None,  # unused live; kept for interface compat
    ) -> OrderResult:
        from py_clob_client_v2.clob_types import OrderArgs, OrderType
        from py_clob_client_v2.order_builder.constants import BUY

        shares = _usd_to_shares(size_usd, price)
        if shares < 1.0:
            return _rejected(asset_id, price, size_usd)

        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(_RETRY_ATTEMPTS),
            wait=wait_exponential(min=_RETRY_MIN_WAIT, max=_RETRY_MAX_WAIT),
            reraise=True,
        )
        def _place() -> Any:
            order = self._c.create_order(
                OrderArgs(token_id=asset_id, price=price, size=shares, side=BUY)
            )
            return self._c.post_order(order, OrderType.FOK)

        try:
            resp = _place()
            return _parse_response(resp, asset_id, price, size_usd)
        except Exception as exc:
            log.error("clob.fok.failed", asset_id=asset_id, price=price, error=str(exc))
            return _rejected(asset_id, price, size_usd)

    async def place_gtc_buy(
        self,
        asset_id: str,
        price: float,
        size_usd: float,
    ) -> OrderResult:
        from py_clob_client_v2.clob_types import OrderArgs, OrderType
        from py_clob_client_v2.order_builder.constants import BUY

        shares = _usd_to_shares(size_usd, price)
        if shares < 1.0:
            return _rejected(asset_id, price, size_usd)

        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(_RETRY_ATTEMPTS),
            wait=wait_exponential(min=_RETRY_MIN_WAIT, max=_RETRY_MAX_WAIT),
            reraise=True,
        )
        def _place() -> Any:
            order = self._c.create_order(
                OrderArgs(token_id=asset_id, price=price, size=shares, side=BUY)
            )
            return self._c.post_order(order, OrderType.GTC)

        try:
            resp = _place()
            return _parse_response(resp, asset_id, price, size_usd)
        except Exception as exc:
            log.error("clob.gtc.failed", asset_id=asset_id, price=price, error=str(exc))
            return _rejected(asset_id, price, size_usd)

    async def place_fok_sell(
        self,
        asset_id: str,
        shares: float,
        current_bid: float,
    ) -> OrderResult:
        from py_clob_client_v2.clob_types import OrderArgs, OrderType
        from py_clob_client_v2.order_builder.constants import SELL

        # Floor (not round) to avoid submitting more shares than we own
        shares_rounded = math.floor(shares * 100) / 100
        if shares_rounded < 1.0:
            return _rejected_sell(asset_id, current_bid, shares)

        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(_RETRY_ATTEMPTS),
            wait=wait_exponential(min=_RETRY_MIN_WAIT, max=_RETRY_MAX_WAIT),
            reraise=True,
        )
        def _place() -> Any:
            order = self._c.create_order(
                OrderArgs(token_id=asset_id, price=current_bid, size=shares_rounded, side=SELL)
            )
            return self._c.post_order(order, OrderType.FOK)

        try:
            resp = _place()
            return _parse_sell_response(resp, asset_id, current_bid, shares_rounded)
        except Exception as exc:
            log.error("clob.sell.failed", asset_id=asset_id, bid=current_bid, error=str(exc))
            return _rejected_sell(asset_id, current_bid, shares)

    async def cancel_order(self, order_id: str) -> bool:
        try:
            self._c.cancel({"orderID": order_id})
            return True
        except Exception as exc:
            log.warning("clob.cancel.failed", order_id=order_id, error=str(exc))
            return False

    async def cancel_all(self) -> int:
        try:
            resp = self._c.cancel_all()
            cancelled = resp.get("canceled", []) if isinstance(resp, dict) else []
            return len(cancelled)
        except Exception as exc:
            log.error("clob.cancel_all.failed", error=str(exc))
            return 0

    async def get_order(self, order_id: str) -> OrderResult | None:
        try:
            resp = self._c.get_order(order_id)
            if not resp:
                return None
            order_price = float(resp.get("price", 0))
            # original_size and size_matched from Polymarket are in SHARES — convert to USD
            size_usd = float(resp.get("original_size", 0)) * order_price
            filled_usd = float(resp.get("size_matched", 0)) * order_price
            return OrderResult(
                order_id=order_id,
                asset_id=resp.get("asset_id", ""),
                side=SIDE_BUY,
                price=order_price,
                size_usd=size_usd,
                filled_usd=filled_usd,
                status=_map_status(resp.get("status", "")),
                is_paper=False,
            )
        except Exception as exc:
            log.warning("clob.get_order.failed", order_id=order_id, error=str(exc))
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _usd_to_shares(size_usd: float, price: float) -> float:
    if price <= 0:
        return 0.0
    return round(size_usd / price, 2)


def _rejected(asset_id: str, price: float, size_usd: float) -> OrderResult:
    return OrderResult(
        order_id="",
        asset_id=asset_id,
        side=SIDE_BUY,
        price=price,
        size_usd=size_usd,
        filled_usd=0.0,
        status=STATUS_REJECTED,
        is_paper=False,
    )


def _map_status(raw: str) -> str:
    mapping = {
        "live": STATUS_OPEN,
        "open": STATUS_OPEN,
        "matched": STATUS_FILLED,
        "filled": STATUS_FILLED,
        "canceled": STATUS_CANCELLED,
        "cancelled": STATUS_CANCELLED,
    }
    return mapping.get(raw.lower(), STATUS_OPEN)


def _rejected_sell(asset_id: str, bid: float, shares: float) -> OrderResult:
    return OrderResult(
        order_id="",
        asset_id=asset_id,
        side=SIDE_SELL,
        price=bid,
        size_usd=shares * bid,
        filled_usd=0.0,
        status=STATUS_REJECTED,
        is_paper=False,
    )


def _parse_sell_response(
    resp: Any,
    asset_id: str,
    bid: float,
    shares: float,
) -> OrderResult:
    if isinstance(resp, dict):
        order_id = str(resp.get("orderID", resp.get("order_id", "")))
        status_raw = resp.get("status", "live")
        filled_shares = float(resp.get("size_matched", 0))
    else:
        order_id = str(getattr(resp, "order_id", "") or "")
        status_raw = str(getattr(resp, "status", "live") or "live")
        filled_shares = float(getattr(resp, "size_matched", 0) or 0)

    filled_usd = filled_shares * bid
    return OrderResult(
        order_id=order_id,
        asset_id=asset_id,
        side=SIDE_SELL,
        price=bid,
        size_usd=shares * bid,
        filled_usd=filled_usd,
        status=_map_status(status_raw),
        is_paper=False,
    )


def _parse_response(
    resp: Any,
    asset_id: str,
    price: float,
    size_usd: float,
) -> OrderResult:
    if isinstance(resp, dict):
        order_id = str(resp.get("orderID", resp.get("order_id", "")))
        status_raw = resp.get("status", "live")
        filled_shares = float(resp.get("size_matched", 0))
    else:
        order_id = str(getattr(resp, "order_id", "") or "")
        status_raw = str(getattr(resp, "status", "live") or "live")
        filled_shares = float(getattr(resp, "size_matched", 0) or 0)

    # Polymarket returns size_matched in SHARES — convert to USD
    filled_usd = filled_shares * price

    return OrderResult(
        order_id=order_id,
        asset_id=asset_id,
        side=SIDE_BUY,
        price=price,
        size_usd=size_usd,
        filled_usd=filled_usd,
        status=_map_status(status_raw),
        is_paper=False,
    )
