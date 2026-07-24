"""Tests for point-in-time feature and target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbot.features import build_dataset, make_features, make_target


def sample_close(n: int = 120) -> pd.Series:
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    trend = 100.0 * np.exp(0.0005 * np.arange(n))
    cycle = 1.0 + 0.02 * np.sin(np.arange(n) / 5.0)
    return pd.Series(trend * cycle, index=index, name="SPY")


def test_features_have_expected_columns():
    features = make_features(sample_close())

    assert list(features.columns) == [
        "return_1d",
        "return_5d",
        "return_10d",
        "return_20d",
        "price_to_ma_10d",
        "price_to_ma_20d",
        "price_to_ma_50d",
        "volatility_10d",
        "volatility_20d",
        "rsi_14d",
        "macd",
    ]


def test_features_cannot_see_future_prices():
    close = sample_close()
    cutoff = close.index[79]

    changed_future = close.copy()
    changed_future.loc[changed_future.index > cutoff] *= 100.0

    original = make_features(close).loc[:cutoff]
    modified = make_features(changed_future).loc[:cutoff]

    pd.testing.assert_frame_equal(original, modified)


def test_target_is_next_day_direction_and_last_row_is_missing():
    close = pd.Series(
        [100.0, 102.0, 101.0, 101.0],
        index=pd.date_range("2024-01-01", periods=4),
    )

    target = make_target(close)

    assert target.iloc[:3].tolist() == [1, 0, 0]
    assert pd.isna(target.iloc[-1])
    assert str(target.dtype) == "Int64"


def test_dataset_removes_warmup_and_unknown_target():
    close = sample_close()

    dataset = build_dataset(close)

    assert not dataset.isna().any().any()
    assert dataset.index.max() < close.index.max()
    assert set(dataset["target"].unique()) <= {0, 1}


def test_optional_volume_features_are_finite_after_warmup():
    close = sample_close()
    volume = pd.Series(
        np.linspace(1_000_000, 2_000_000, len(close)),
        index=close.index,
    )

    dataset = build_dataset(close, volume=volume)

    assert {"volume_change_1d", "volume_to_average_20d"} <= set(dataset.columns)
    assert np.isfinite(dataset.to_numpy(dtype=float)).all()


def test_unordered_input_is_rejected():
    close = sample_close().sort_index(ascending=False)

    with pytest.raises(ValueError, match="chronologically"):
        make_features(close)
