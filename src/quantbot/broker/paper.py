"""In-memory paper broker.

Fills every order immediately at the reference price plus a slippage haircut.
It is intentionally optimistic about liquidity and intentionally pessimistic
about price: you pay to cross the spread in whichever direction hurts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import Broker, Order, Position


@dataclass
class Fill:
    symbol: str
    quantity: float
    price: float


@dataclass
class PaperBroker(Broker):
    """A simulated account with cash, positions and a trade log.

    Args:
        starting_cash: Initial account balance.
        slippage_bps: Adverse price move applied to every fill, in basis points.
        allow_short: Whether negative positions may be opened.
    """

    starting_cash: float = 100_000.0
    slippage_bps: float = 2.0
    allow_short: bool = True

    _cash: float = field(init=False)
    _positions: dict[str, Position] = field(init=False, default_factory=dict)
    fills: list[Fill] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._cash = self.starting_cash

    def cash(self) -> float:
        return self._cash

    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def submit(self, order: Order, price: float) -> None:
        if order.quantity == 0:
            return

        existing = self._positions.get(order.symbol)
        held = existing.quantity if existing else 0.0
        if not self.allow_short and held + order.quantity < 0:
            raise ValueError(f"shorting disabled but order would short {order.symbol}")

        # Buys fill above the reference price, sells below it.
        direction = 1.0 if order.quantity > 0 else -1.0
        fill_price = price * (1.0 + direction * self.slippage_bps / 10_000.0)

        self._cash -= order.quantity * fill_price
        self._positions[order.symbol] = _apply(existing, order.symbol, order.quantity, fill_price)
        self.fills.append(Fill(order.symbol, order.quantity, fill_price))

        if self._positions[order.symbol].quantity == 0:
            del self._positions[order.symbol]


def _apply(existing: Position | None, symbol: str, qty: float, price: float) -> Position:
    """Fold a fill into a position, keeping a sane average entry price."""
    if existing is None:
        return Position(symbol=symbol, quantity=qty, avg_price=price)

    new_qty = existing.quantity + qty
    if new_qty == 0:
        return Position(symbol=symbol, quantity=0.0, avg_price=0.0)

    # Adding to a position blends the cost basis; reducing or flipping does not.
    same_direction = existing.quantity * qty > 0
    if same_direction:
        cost = existing.avg_price * existing.quantity + price * qty
        return Position(symbol=symbol, quantity=new_qty, avg_price=cost / new_qty)

    flipped = existing.quantity * new_qty < 0
    avg = price if flipped else existing.avg_price
    return Position(symbol=symbol, quantity=new_qty, avg_price=avg)
