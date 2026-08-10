from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbot.validation import (
    NestedMomentumEnsemble,
    block_bootstrap_sharpe,
    newey_west_mean,
    run_robustness_analysis,
)


class RecordingModel:
    fits: list[pd.DatetimeIndex] = []
    predictions: list[pd.DatetimeIndex] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RecordingModel:
        self.fits.append(X.index.copy())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.predictions.append(X.index.copy())
        probability = np.clip(0.5 + X["return_20d"].to_numpy(), 0.1, 0.9)
        return np.column_stack([1.0 - probability, probability])


def _market() -> tuple[pd.Series, pd.Series]:
    index = pd.date_range("2010-01-01", "2018-12-31", freq="B")
    step = np.arange(len(index))
    close = pd.Series(
        100.0 * np.exp(0.00025 * step) * (1.0 + 0.025 * np.sin(step / 13.0)),
        index=index,
    )
    volume = pd.Series(1_000_000.0 + 20_000.0 * np.cos(step / 9.0), index=index)
    return close, volume


def test_nested_ensemble_tunes_without_outer_test_rows():
    RecordingModel.fits.clear()
    RecordingModel.predictions.clear()
    close, volume = _market()

    result = run_robustness_analysis(
        close,
        volume=volume,
        model_factory=RecordingModel,
        n_bootstrap=20,
        out_dir=None,
    )

    assert {"model_probability", "momentum_probability", "model_blend_weight"} <= set(
        result.ensemble.predictions
    )
    assert set(result.ensemble.predictions["model_blend_weight"].unique()) <= {
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    }
    for fitted, predicted in zip(RecordingModel.fits, RecordingModel.predictions, strict=True):
        assert fitted.max() < predicted.min()


def test_transaction_costs_never_improve_a_trading_strategy():
    close, volume = _market()
    result = run_robustness_analysis(
        close,
        volume=volume,
        model_factory=RecordingModel,
        cost_grid=(0.0, 25.0, 50.0),
        n_bootstrap=20,
        out_dir=None,
    )
    logistic = result.cost_sensitivity.query("strategy == 'Logistic'").sort_values("cost_bps")

    assert logistic["CAGR"].is_monotonic_decreasing
    assert list(logistic["cost_bps"]) == [0.0, 25.0, 50.0]


def test_uncertainty_estimators_are_reproducible_and_ordered():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0004, 0.01, 600))

    first = block_bootstrap_sharpe(returns, n_bootstrap=100, random_state=11)
    second = block_bootstrap_sharpe(returns, n_bootstrap=100, random_state=11)
    nw = newey_west_mean(returns)

    assert first == second
    assert first["Sharpe CI lower"] <= first["Sharpe"] <= first["Sharpe CI upper"]
    assert nw["Mean CI lower"] <= nw["Annualized mean"] <= nw["Mean CI upper"]


def test_invalid_robustness_parameters_are_rejected():
    returns = pd.Series([0.01, -0.01, 0.02])
    with pytest.raises(ValueError, match="block_size"):
        block_bootstrap_sharpe(returns, block_size=4)
    with pytest.raises(ValueError, match="max_lag"):
        newey_west_mean(returns, max_lag=3)

    X = pd.DataFrame({"return_20d": [0.1]})
    with pytest.raises(ValueError, match="blend weights"):
        NestedMomentumEnsemble(RecordingModel, blend_grid=(-0.1,)).fit(
            X, pd.Series([1], index=X.index)
        )
