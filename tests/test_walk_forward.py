from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbot.features import make_features
from quantbot.validation.walk_forward import (
    build_spy_dataset,
    chronological_folds,
    run_spy_walk_forward,
    walk_forward_predict,
)


class RecordingModel:
    fitted_indices: list[pd.DatetimeIndex] = []
    predicted_indices: list[pd.DatetimeIndex] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RecordingModel:
        self.fitted_indices.append(X.index.copy())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.predicted_indices.append(X.index.copy())
        probability = np.linspace(0.25, 0.75, len(X))
        return np.column_stack([1.0 - probability, probability])


def _data() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2010-01-01", "2017-12-31", freq="B")
    X = pd.DataFrame({"feature": np.arange(len(index))}, index=index)
    y = pd.Series(np.arange(len(index)) % 2, index=index)
    return X, y


def test_every_training_date_precedes_every_test_date():
    X, _ = _data()
    for train, test in chronological_folds(X.index):
        assert X.index[train].max() < X.index[test].min()


def test_test_observations_never_enter_model_fitting():
    RecordingModel.fitted_indices.clear()
    RecordingModel.predicted_indices.clear()
    X, y = _data()
    walk_forward_predict(X, y, RecordingModel)

    for fitted, predicted in zip(
        RecordingModel.fitted_indices, RecordingModel.predicted_indices, strict=True
    ):
        assert fitted.intersection(predicted).empty
        assert fitted.max() < predicted.min()


def test_out_of_sample_predictions_retain_correct_dates():
    RecordingModel.fitted_indices.clear()
    RecordingModel.predicted_indices.clear()
    X, y = _data()
    probabilities = walk_forward_predict(X, y, RecordingModel)
    expected = RecordingModel.predicted_indices[0].append(RecordingModel.predicted_indices[1:])

    assert probabilities.index.equals(expected)


def test_each_test_date_is_predicted_at_most_once():
    X, y = _data()
    probabilities = walk_forward_predict(X, y, RecordingModel)

    assert probabilities.index.is_unique


def test_spy_dataset_uses_the_shared_feature_pipeline():
    index = pd.date_range("2010-01-01", periods=120, freq="B")
    close = pd.Series(100.0 + np.arange(len(index)), index=index)

    X, target, forward_return = build_spy_dataset(close)

    assert list(X.columns) == list(make_features(close).columns)
    assert X.index.equals(target.index)
    assert X.index.equals(forward_return.index)
    assert not X.isna().any().any()
    assert set(target.unique()) <= {0, 1}


def test_spy_dataset_adds_volume_features_when_volume_is_available():
    index = pd.date_range("2010-01-01", periods=120, freq="B")
    close = pd.Series(100.0 + np.arange(len(index)), index=index)
    volume = pd.Series(1_000_000.0 + 100.0 * np.arange(len(index)), index=index)

    X, target, forward_return = build_spy_dataset(close, volume=volume)

    assert {"volume_change_1d", "volume_to_average_20d"} <= set(X.columns)
    assert X.index.equals(target.index)
    assert X.index.equals(forward_return.index)


def test_probability_threshold_must_be_valid():
    index = pd.date_range("2010-01-01", periods=120, freq="B")
    close = pd.Series(100.0 + np.arange(len(index)), index=index)

    with pytest.raises(ValueError, match="threshold"):
        run_spy_walk_forward(close, RecordingModel, threshold=1.01, out_dir=None)


def test_unknown_feature_selection_is_rejected():
    index = pd.date_range("2010-01-01", periods=120, freq="B")
    close = pd.Series(100.0 + np.arange(len(index)), index=index)

    with pytest.raises(ValueError, match="unknown feature"):
        run_spy_walk_forward(
            close,
            RecordingModel,
            feature_columns=("future_return",),
            out_dir=None,
        )
