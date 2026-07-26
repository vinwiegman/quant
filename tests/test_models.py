"""Tests for reproducible ML model construction and comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbot.models import MODEL_NAMES, get_model_factory
from quantbot.validation import probabilities_to_positions, run_model_comparison


def test_supported_models_produce_binary_probabilities():
    X = pd.DataFrame(
        {
            "trend": np.linspace(-1.0, 1.0, 80),
            "noise": np.sin(np.arange(80)),
        }
    )
    y = pd.Series((X["trend"] > 0).astype(int))

    for name in MODEL_NAMES:
        model = get_model_factory(name)()
        model.fit(X, y)
        probabilities = model.predict_proba(X.iloc[-5:])

        assert probabilities.shape == (5, 2)
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        get_model_factory("neural-network")  # type: ignore[arg-type]


def test_hysteresis_preserves_position_inside_neutral_band():
    probability = pd.Series([0.50, 0.56, 0.52, 0.44, 0.50, 0.57])

    position = probabilities_to_positions(
        probability,
        entry_threshold=0.55,
        exit_threshold=0.45,
    )

    assert position.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]


def test_hysteresis_thresholds_must_not_overlap():
    with pytest.raises(ValueError, match="exit threshold"):
        probabilities_to_positions(
            pd.Series([0.5]),
            entry_threshold=0.55,
            exit_threshold=0.60,
        )


def test_model_comparison_uses_the_same_prediction_dates():
    index = pd.date_range("2010-01-01", "2017-12-31", freq="B")
    close = pd.Series(
        100.0
        * np.exp(0.0003 * np.arange(len(index)))
        * (1.0 + 0.02 * np.sin(np.arange(len(index)) / 8.0)),
        index=index,
    )

    comparison = run_model_comparison(close, out_dir=None)

    logistic_dates = comparison.models["logistic"].predictions.index
    boosting_dates = comparison.models["gradient-boosting"].predictions.index
    assert logistic_dates.equals(boosting_dates)
    assert {"Accuracy", "Precision", "Recall", "ROC AUC"} <= set(comparison.metrics.columns)
