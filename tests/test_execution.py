"""Tests for safe daily signal generation and paper reconciliation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantbot.broker import Broker, Order, Position
from quantbot.execution import run_daily_execution


class FixedProbabilityModel:
    def __init__(self, probability: float):
        self.probability = probability

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.tile([1.0 - self.probability, self.probability], (len(X), 1))


class RecordingBroker(Broker):
    def __init__(self, positions=None):
        self._positions = positions or {}
        self.submitted: list[Order] = []

    def cash(self):
        return 10_000.0

    def account_equity(self):
        return 10_000.0

    def positions(self):
        return dict(self._positions)

    def market_is_open(self):
        return False

    def submit(self, order, price):
        self.submitted.append(order)


def market_data() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=160, freq="B")
    time = np.arange(len(index))
    return pd.DataFrame(
        {
            "close": 100.0 + time * 0.1 + np.sin(time / 5.0),
            "volume": 1_000_000.0 + time * 1_000.0,
        },
        index=index,
    )


def test_daily_execution_is_dry_run_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "quantbot.execution.daily.get_model_factory",
        lambda name: lambda: FixedProbabilityModel(0.60),
    )
    broker = RecordingBroker()
    log = tmp_path / "executions.jsonl"

    result = run_daily_execution(market_data(), broker, log_path=log)

    assert result.target_weight == pytest.approx(0.95)
    assert result.orders[0]["side"] == "buy"
    assert result.submitted is False
    assert result.signal_date == market_data().index[-1].date().isoformat()
    assert result.account_equity == pytest.approx(10_000.0)
    assert result.cash == pytest.approx(10_000.0)
    assert broker.submitted == []
    assert json.loads(log.read_text())["probability"] == pytest.approx(0.60)


def test_daily_execution_persists_to_sqlite(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "quantbot.execution.daily.get_model_factory",
        lambda name: lambda: FixedProbabilityModel(0.60),
    )
    database = tmp_path / "paper.sqlite3"

    run_daily_execution(
        market_data(),
        RecordingBroker(),
        log_path=None,
        database_path=database,
    )

    from quantbot.persistence import ExecutionStore

    assert len(ExecutionStore(database).decisions()) == 1


def test_daily_execution_submits_only_when_explicit(monkeypatch):
    monkeypatch.setattr(
        "quantbot.execution.daily.get_model_factory",
        lambda name: lambda: FixedProbabilityModel(0.60),
    )
    broker = RecordingBroker()

    result = run_daily_execution(
        market_data(),
        broker,
        submit=True,
        log_path=None,
    )

    assert result.submitted is True
    assert len(broker.submitted) == 1
    assert broker.submitted[0].quantity > 0


def test_neutral_probability_keeps_existing_position(monkeypatch):
    monkeypatch.setattr(
        "quantbot.execution.daily.get_model_factory",
        lambda name: lambda: FixedProbabilityModel(0.50),
    )
    latest_price = float(market_data()["close"].iloc[-1])
    desired_quantity = 0.95 * 10_000.0 / latest_price
    broker = RecordingBroker({"SPY": Position("SPY", desired_quantity, avg_price=100.0)})

    result = run_daily_execution(market_data(), broker, log_path=None)

    assert result.current_invested is True
    assert result.target_weight == pytest.approx(0.95)
    assert result.orders == ()


def test_unexpected_account_position_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "quantbot.execution.daily.get_model_factory",
        lambda name: lambda: FixedProbabilityModel(0.60),
    )
    broker = RecordingBroker({"AAPL": Position("AAPL", 1.0, 100.0)})

    with pytest.raises(ValueError, match="outside strategy scope"):
        run_daily_execution(market_data(), broker, log_path=None)
