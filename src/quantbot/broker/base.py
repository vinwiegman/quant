"""Broker interface shared by the simulator and any live implementation.

The backtester and the paper trader must go through the same door, otherwise
the thing you tested is not the thing you run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    symbol: str
    quantity: float
    """Positive to buy, negative to sell. Fractional shares are allowed."""


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    avg_price: float

    def market_value(self, price: float) -> float:
        return self.quantity * price


class Broker(ABC):
    """Minimal execution surface: know your cash, positions, and place orders."""

    @abstractmethod
    def cash(self) -> float:
        """Free cash available to deploy."""

    @abstractmethod
    def positions(self) -> dict[str, Position]:
        """Currently held positions, keyed by symbol."""

    @abstractmethod
    def submit(self, order: Order, price: float) -> None:
        """Execute an order at the given reference price."""

    def equity(self, prices: dict[str, float]) -> float:
        """Total account value: cash plus the marked value of all positions."""
        held = sum(p.market_value(prices[s]) for s, p in self.positions().items() if s in prices)
        return self.cash() + held


def target_weights_to_orders(
    targets: dict[str, float],
    prices: dict[str, float],
    equity: float,
    current: dict[str, Position],
    min_trade_value: float = 1.0,
) -> list[Order]:
    """Translate desired portfolio weights into the orders needed to get there.

    Args:
        targets: Desired weight per symbol, e.g. ``{"AAPL": 0.5}``.
        prices: Current price per symbol.
        equity: Total account value to size against.
        current: Positions held right now.
        min_trade_value: Skip trades smaller than this notional, so rounding
            noise does not generate a stream of pointless commissions.

    Returns:
        The orders that move the account from ``current`` to ``targets``.
    """
    orders: list[Order] = []
    symbols = set(targets) | set(current)

    for symbol in sorted(symbols):
        price = prices.get(symbol)
        if not price:
            continue
        desired_qty = targets.get(symbol, 0.0) * equity / price
        held_qty = current[symbol].quantity if symbol in current else 0.0
        delta = desired_qty - held_qty
        if abs(delta * price) < min_trade_value:
            continue
        orders.append(Order(symbol=symbol, quantity=delta))

    return orders
