"""Paper-only Alpaca broker adapter."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from .base import Broker, Order, Position


class AlpacaPaperBroker(Broker):
    """Translate the project broker contract to Alpaca's paper API."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        *,
        client: Any | None = None,
        request_factory: Callable[[Order, str], Any] | None = None,
    ) -> None:
        if client is None:
            api_key = api_key or os.getenv("ALPACA_API_KEY")
            secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
            if not api_key or not secret_key:
                raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")
            from alpaca.trading.client import TradingClient

            client = TradingClient(api_key, secret_key, paper=True)
        self._client = client
        self._request_factory = request_factory or _market_order_request
        self.last_response: Any | None = None

    def cash(self) -> float:
        return float(self._client.get_account().cash)

    def account_equity(self) -> float:
        return float(self._client.get_account().equity)

    def positions(self) -> dict[str, Position]:
        return {
            item.symbol: Position(
                symbol=item.symbol,
                quantity=float(item.qty),
                avg_price=float(item.avg_entry_price),
            )
            for item in self._client.get_all_positions()
        }

    def market_is_open(self) -> bool:
        return bool(self._client.get_clock().is_open)

    def submit(self, order: Order, price: float) -> None:
        del price  # Market orders intentionally do not claim a guaranteed fill price.
        if order.quantity == 0:
            return
        side = "buy" if order.quantity > 0 else "sell"
        day = datetime.now(UTC).strftime("%Y%m%d")
        client_order_id = f"quantbot-{order.symbol.lower()}-{day}-{side}"
        request = self._request_factory(order, client_order_id)
        try:
            self.last_response = self._client.submit_order(order_data=request)
        except Exception as exc:  # noqa: BLE001 - narrowed by _is_duplicate_order below
            if _is_duplicate_order(exc):
                # The deterministic client_order_id already exists for today, so this
                # side was submitted earlier. Treat the duplicate as an idempotent no-op.
                self.last_response = None
                return
            raise


def _is_duplicate_order(exc: Exception) -> bool:
    """Detect Alpaca's 'client_order_id must be unique' rejection (code 40010001)."""
    return "client_order_id must be unique" in str(exc) or "40010001" in str(exc)


def _market_order_request(order: Order, client_order_id: str) -> Any:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    side = OrderSide.BUY if order.quantity > 0 else OrderSide.SELL
    return MarketOrderRequest(
        symbol=order.symbol,
        qty=round(abs(order.quantity), 6),
        side=side,
        time_in_force=TimeInForce.DAY,
        client_order_id=client_order_id,
    )
