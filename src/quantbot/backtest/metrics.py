"""Performance and risk metrics for a strategy return series.

All functions take a series of *periodic* (usually daily) returns and assume
``periods_per_year`` observations per year for annualisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate implied by the return series."""
    if len(returns) == 0:
        return float("nan")
    total_growth = float((1.0 + returns).prod())
    if total_growth <= 0:
        return -1.0
    years = len(returns) / periods_per_year
    return total_growth ** (1.0 / years) - 1.0


def cumulative_return(returns: pd.Series) -> float:
    """Total compounded return over the complete series."""
    if len(returns) == 0:
        return float("nan")
    return float((1.0 + returns).prod() - 1.0)


def annual_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised Sharpe ratio. ``risk_free`` is an annual rate."""
    excess = returns - risk_free / periods_per_year
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Like Sharpe but penalises only downside deviation."""
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    dd = downside.std(ddof=1)
    if len(downside) == 0 or dd == 0 or np.isnan(dd):
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak, at every point in time."""
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return float("nan")
    return float(drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """CAGR divided by the magnitude of the worst drawdown."""
    mdd = abs(max_drawdown(returns))
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return cagr(returns, periods_per_year) / mdd


def hit_rate(returns: pd.Series) -> float:
    """Share of periods with a strictly positive return, ignoring flat periods."""
    active = returns[returns != 0]
    if len(active) == 0:
        return float("nan")
    return float((active > 0).mean())


def summary(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """One-column (or two, with a benchmark) table of headline metrics."""

    def _row(r: pd.Series) -> dict[str, float]:
        return {
            "CAGR": cagr(r, periods_per_year),
            "Cumulative return": cumulative_return(r),
            "Volatility": annual_volatility(r, periods_per_year),
            "Sharpe": sharpe_ratio(r, periods_per_year=periods_per_year),
            "Sortino": sortino_ratio(r, periods_per_year=periods_per_year),
            "Max drawdown": max_drawdown(r),
            "Calmar": calmar_ratio(r, periods_per_year),
            "Hit rate": hit_rate(r),
        }

    data = {"strategy": _row(returns)}
    if benchmark is not None:
        aligned = benchmark.reindex(returns.index).fillna(0.0)
        data["benchmark"] = _row(aligned)
    return pd.DataFrame(data)
