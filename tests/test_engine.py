"""Tests for the backtest engine.

The lookahead test is the important one. If it ever fails, every performance
number the project has ever produced is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbot.backtest.engine import run_backtest


def make_prices(values: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(values.values())))
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(values, index=index)


def test_flat_weights_produce_flat_equity():
    prices = make_prices({"A": [100, 110, 121, 133.1]})
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    result = run_backtest(prices, weights, cost_bps=0.0, initial_capital=1_000.0)

    assert result.equity.iloc[-1] == pytest.approx(1_000.0)
    assert result.turnover.sum() == pytest.approx(0.0)


def test_full_long_matches_asset_return():
    prices = make_prices({"A": [100, 110, 121]})
    weights = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)

    result = run_backtest(prices, weights, cost_bps=0.0, initial_capital=1_000.0)

    # Weight on day 0 is only active from day 1, so we capture the day-1 and
    # day-2 returns: two consecutive +10% moves.
    assert result.equity.iloc[-1] == pytest.approx(1_000.0 * 1.10 * 1.10)


def test_signal_cannot_see_the_same_day_return():
    """A perfect-foresight signal must not profit on the day it is set."""
    prices = make_prices({"A": [100, 50, 50]})

    # Someone "knows" the crash on day 1 and goes short on that very row.
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.iloc[1] = -1.0

    result = run_backtest(prices, weights, cost_bps=0.0, initial_capital=1_000.0)

    # The -50% move already happened on day 1; the short only applies to day 2,
    # which is flat. So the strategy must end exactly where it started.
    assert result.equity.iloc[-1] == pytest.approx(1_000.0)


def test_transaction_costs_reduce_returns():
    prices = make_prices({"A": [100, 110, 121]})
    weights = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)

    free = run_backtest(prices, weights, cost_bps=0.0)
    costly = run_backtest(prices, weights, cost_bps=100.0)

    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert costly.costs.sum() > 0


def test_short_position_profits_when_price_falls():
    prices = make_prices({"A": [100, 100, 90]})
    weights = pd.DataFrame(-1.0, index=prices.index, columns=prices.columns)

    result = run_backtest(prices, weights, cost_bps=0.0, initial_capital=1_000.0)

    assert result.equity.iloc[-1] == pytest.approx(1_100.0)


def test_mismatched_columns_are_rejected():
    prices = make_prices({"A": [100, 101]})
    weights = pd.DataFrame(0.0, index=prices.index, columns=["B"])

    with pytest.raises(ValueError, match="columns"):
        run_backtest(prices, weights)


def test_mismatched_index_is_rejected():
    prices = make_prices({"A": [100, 101]})
    weights = pd.DataFrame(
        0.0, index=pd.date_range("2021-01-01", periods=2, freq="D"), columns=["A"]
    )

    with pytest.raises(ValueError, match="index"):
        run_backtest(prices, weights)


def test_gross_returns_exclude_costs():
    prices = make_prices({"A": [100, 110, 121]})
    weights = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)

    result = run_backtest(prices, weights, cost_bps=50.0)

    assert np.allclose(result.gross_returns - result.costs, result.returns)
