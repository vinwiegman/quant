"""Tests for the performance metrics, checked against hand-computed values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbot.backtest import metrics


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="D"))


def test_cagr_of_doubling_over_one_year():
    daily = series([0.0] * 252)
    daily.iloc[0] = 1.0  # a single +100% day, then flat

    assert metrics.cagr(daily) == pytest.approx(1.0, rel=1e-6)


def test_sharpe_is_zero_for_symmetric_returns():
    daily = series([0.01, -0.01] * 100)

    assert metrics.sharpe_ratio(daily) == pytest.approx(0.0, abs=1e-9)


def test_sharpe_is_nan_for_constant_returns():
    assert np.isnan(metrics.sharpe_ratio(series([0.001] * 50)))


def test_max_drawdown_matches_manual_calculation():
    # +50%, then -40%: peak 1.5, trough 0.9, so drawdown is 0.6/1.5 = 40%.
    daily = series([0.5, -0.4])

    assert metrics.max_drawdown(daily) == pytest.approx(-0.4)


def test_drawdown_is_never_positive():
    rng = np.random.default_rng(42)
    daily = series(list(rng.normal(0.0005, 0.01, 500)))

    assert (metrics.drawdown_series(daily) <= 1e-12).all()


def test_sortino_ignores_upside_volatility():
    calm = series([0.01, -0.005] * 100)
    spiky = series([0.20, -0.005] * 100)

    # Both have the same downside, but the spiky one has far more upside, so
    # its Sortino must be higher even though its plain volatility is worse.
    assert metrics.sortino_ratio(spiky) > metrics.sortino_ratio(calm)


def test_hit_rate_ignores_flat_periods():
    daily = series([0.01, -0.01, 0.0, 0.0, 0.01])

    assert metrics.hit_rate(daily) == pytest.approx(2 / 3)


def test_summary_includes_benchmark_column():
    strat = series([0.01, -0.01, 0.02])
    bench = series([0.00, 0.01, 0.01])

    table = metrics.summary(strat, benchmark=bench)

    assert list(table.columns) == ["strategy", "benchmark"]
    assert "Sharpe" in table.index


def test_metrics_on_empty_series_do_not_crash():
    empty = pd.Series(dtype=float)

    assert np.isnan(metrics.cagr(empty))
    assert np.isnan(metrics.max_drawdown(empty))
