"""Vectorised portfolio backtester.

The single most important property here is that it cannot look ahead. A signal
computed from data up to and including the close of day *t* is only allowed to
earn the return of day *t+1*. That is enforced in exactly one place --
``weights.shift(1)`` in :func:`run_backtest` -- so it is easy to audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import drawdown_series, summary


@dataclass(frozen=True)
class BacktestResult:
    """Everything a backtest produces, kept separate from how it is displayed."""

    equity: pd.Series
    """Portfolio value over time, starting at ``initial_capital``."""

    returns: pd.Series
    """Net periodic returns, after transaction costs."""

    gross_returns: pd.Series
    """Periodic returns before transaction costs."""

    positions: pd.DataFrame
    """Weights actually held during each period (the shifted signal)."""

    turnover: pd.Series
    """Sum of absolute weight changes at each rebalance."""

    costs: pd.Series
    """Transaction cost charged each period, as a fraction of capital."""

    def drawdown(self) -> pd.Series:
        return drawdown_series(self.returns)

    def summary(self, benchmark: pd.Series | None = None) -> pd.DataFrame:
        return summary(self.returns, benchmark=benchmark)


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 5.0,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Run a weight-based backtest.

    Args:
        prices: Close prices, indexed by date, one column per instrument.
        weights: Target portfolio weights decided using data up to that date's
            close. Same index and columns as ``prices``. A weight of 1.0 means
            100% of capital long that instrument; negative means short.
        cost_bps: Round-trip cost in basis points charged on traded notional.
            5 bps is a reasonable default for liquid US equities.
        initial_capital: Starting portfolio value.

    Returns:
        A :class:`BacktestResult` with the full time series of the run.
    """
    if not prices.index.equals(weights.index):
        raise ValueError("prices and weights must share the same index")
    if list(prices.columns) != list(weights.columns):
        raise ValueError("prices and weights must share the same columns")

    asset_returns = prices.pct_change().fillna(0.0)

    # The only lookahead barrier in the system: today's signal, tomorrow's return.
    positions = weights.shift(1).fillna(0.0)

    gross_returns = (positions * asset_returns).sum(axis=1)

    # Trading happens when the held weight changes from one period to the next.
    turnover = (positions - positions.shift(1).fillna(0.0)).abs().sum(axis=1)
    costs = turnover * (cost_bps / 10_000.0)

    net_returns = gross_returns - costs
    equity = initial_capital * (1.0 + net_returns).cumprod()

    return BacktestResult(
        equity=equity,
        returns=net_returns,
        gross_returns=gross_returns,
        positions=positions,
        turnover=turnover,
        costs=costs,
    )
