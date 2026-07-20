"""Tests for the paper broker and weight-to-order translation."""

from __future__ import annotations

import pytest

from quantbot.broker.base import Order, Position, target_weights_to_orders
from quantbot.broker.paper import PaperBroker


def test_buy_reduces_cash_and_creates_position():
    broker = PaperBroker(starting_cash=10_000.0, slippage_bps=0.0)

    broker.submit(Order("AAPL", 10), price=100.0)

    assert broker.cash() == pytest.approx(9_000.0)
    assert broker.positions()["AAPL"].quantity == 10


def test_slippage_makes_buys_more_expensive():
    broker = PaperBroker(starting_cash=10_000.0, slippage_bps=100.0)

    broker.submit(Order("AAPL", 10), price=100.0)

    # 100 bps worse means filling at 101 rather than 100.
    assert broker.cash() == pytest.approx(10_000.0 - 1_010.0)


def test_closing_a_position_removes_it():
    broker = PaperBroker(starting_cash=10_000.0, slippage_bps=0.0)

    broker.submit(Order("AAPL", 10), price=100.0)
    broker.submit(Order("AAPL", -10), price=110.0)

    assert "AAPL" not in broker.positions()
    assert broker.cash() == pytest.approx(10_100.0)


def test_averaging_up_blends_the_cost_basis():
    broker = PaperBroker(slippage_bps=0.0)

    broker.submit(Order("AAPL", 10), price=100.0)
    broker.submit(Order("AAPL", 10), price=120.0)

    assert broker.positions()["AAPL"].avg_price == pytest.approx(110.0)


def test_shorting_can_be_disabled():
    broker = PaperBroker(allow_short=False)

    with pytest.raises(ValueError, match="short"):
        broker.submit(Order("AAPL", -5), price=100.0)


def test_equity_marks_positions_at_current_prices():
    broker = PaperBroker(starting_cash=10_000.0, slippage_bps=0.0)
    broker.submit(Order("AAPL", 10), price=100.0)

    assert broker.equity({"AAPL": 150.0}) == pytest.approx(9_000.0 + 1_500.0)


def test_weights_translate_into_the_right_orders():
    orders = target_weights_to_orders(
        targets={"AAPL": 0.5, "MSFT": 0.5},
        prices={"AAPL": 100.0, "MSFT": 200.0},
        equity=10_000.0,
        current={},
    )

    by_symbol = {o.symbol: o.quantity for o in orders}
    assert by_symbol["AAPL"] == pytest.approx(50.0)
    assert by_symbol["MSFT"] == pytest.approx(25.0)


def test_dropped_symbols_are_sold_down_to_zero():
    orders = target_weights_to_orders(
        targets={"AAPL": 1.0},
        prices={"AAPL": 100.0, "MSFT": 200.0},
        equity=10_000.0,
        current={"MSFT": Position("MSFT", 25.0, 200.0)},
    )

    by_symbol = {o.symbol: o.quantity for o in orders}
    assert by_symbol["MSFT"] == pytest.approx(-25.0)


def test_tiny_rebalances_are_skipped():
    orders = target_weights_to_orders(
        targets={"AAPL": 0.5001},
        prices={"AAPL": 100.0},
        equity=10_000.0,
        current={"AAPL": Position("AAPL", 50.0, 100.0)},
        min_trade_value=10.0,
    )

    assert orders == []
