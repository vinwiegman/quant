"""Tests for the paper-only Alpaca adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from quantbot.broker import AlpacaPaperBroker, Order


class FakeTradingClient:
    def __init__(self):
        self.submitted = []

    def get_account(self):
        return SimpleNamespace(cash="25000.50", equity="100000.25")

    def get_all_positions(self):
        return [SimpleNamespace(symbol="SPY", qty="10.5", avg_entry_price="500.25")]

    def get_clock(self):
        return SimpleNamespace(is_open=False)

    def submit_order(self, *, order_data):
        self.submitted.append(order_data)
        return SimpleNamespace(id="paper-order", status="accepted")


def test_alpaca_adapter_reads_account_and_positions():
    broker = AlpacaPaperBroker(client=FakeTradingClient(), request_factory=lambda *_: {})

    assert broker.cash() == pytest.approx(25_000.50)
    assert broker.account_equity() == pytest.approx(100_000.25)
    assert broker.positions()["SPY"].quantity == pytest.approx(10.5)
    assert broker.market_is_open() is False


def test_alpaca_adapter_submits_fractional_order_with_idempotency_key():
    client = FakeTradingClient()
    captured = {}

    def request_factory(order, client_order_id):
        captured["order"] = order
        captured["client_order_id"] = client_order_id
        return {"qty": abs(order.quantity), "client_order_id": client_order_id}

    broker = AlpacaPaperBroker(client=client, request_factory=request_factory)
    broker.submit(Order("SPY", 1.25), price=500.0)

    assert captured["order"].quantity == pytest.approx(1.25)
    assert captured["client_order_id"].startswith("quantbot-spy-")
    assert captured["client_order_id"].endswith("-buy")
    assert len(client.submitted) == 1


def test_alpaca_adapter_requires_paper_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    with pytest.raises(ValueError, match="ALPACA_API_KEY"):
        AlpacaPaperBroker()
