"""Leakage-safe technical features for daily equity data.

Every feature on row ``t`` uses observations from row ``t`` or earlier. The
target is kept separate because it deliberately uses the next close and must
never be passed to a model as an input feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_features(close: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    """Construct a compact set of point-in-time daily features.

    Args:
        close: Adjusted closing prices in chronological order.
        volume: Optional daily volume aligned to ``close``. Supplying it adds
            volume change and volume-to-20-day-average features.

    Returns:
        Feature rows on the same index as ``close``. Warm-up rows contain NaNs
        and can be removed after joining with the target.
    """
    close = _validate_series(close, "close")
    if (close <= 0).any():
        raise ValueError("close prices must be strictly positive")

    returns_1d = close.pct_change(fill_method=None)
    features = pd.DataFrame(index=close.index)

    for days in (1, 5, 10, 20):
        features[f"return_{days}d"] = close.pct_change(days, fill_method=None)

    for days in (10, 20, 50):
        moving_average = close.rolling(days, min_periods=days).mean()
        features[f"price_to_ma_{days}d"] = close / moving_average - 1.0

    for days in (10, 20):
        features[f"volatility_{days}d"] = returns_1d.rolling(days, min_periods=days).std()

    features["rsi_14d"] = _rsi(close, window=14)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    features["macd"] = (ema_12 - ema_26) / close

    if volume is not None:
        volume = _validate_series(volume, "volume").reindex(close.index)
        if volume.isna().any():
            raise ValueError("volume must contain a value for every close date")
        if (volume < 0).any():
            raise ValueError("volume cannot be negative")
        features["volume_change_1d"] = volume.pct_change(fill_method=None)
        average_volume = volume.rolling(20, min_periods=20).mean()
        features["volume_to_average_20d"] = volume / average_volume - 1.0

    return features.replace([np.inf, -np.inf], np.nan)


def make_target(close: pd.Series) -> pd.Series:
    """Return the nullable next-day direction target.

    Row ``t`` is one when the close-to-close return from ``t`` to ``t+1`` is
    positive and zero otherwise. The final row is missing because its future
    return is not known; it is never silently mislabeled as a down day.
    """
    close = _validate_series(close, "close")
    future_return = close.shift(-1) / close - 1.0
    target = (future_return > 0).astype("Int64")
    return target.mask(future_return.isna()).rename("target")


def build_dataset(
    close: pd.Series,
    volume: pd.Series | None = None,
    *,
    dropna: bool = True,
) -> pd.DataFrame:
    """Join features and target into a chronological modeling dataset.

    ``dropna=True`` removes indicator warm-up rows and the unlabeled final row.
    Set it to false when generating the latest live feature row for inference.
    """
    dataset = make_features(close, volume=volume).join(make_target(close))
    return dataset.dropna() if dropna else dataset


def _rsi(close: pd.Series, window: int) -> pd.Series:
    change = close.diff()
    gains = change.clip(lower=0.0)
    losses = -change.clip(upper=0.0)
    average_gain = gains.rolling(window, min_periods=window).mean()
    average_loss = losses.rolling(window, min_periods=window).mean()
    relative_strength = average_gain / average_loss
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)

    # Explicitly handle one-sided and completely flat windows.
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    return rsi.mask((average_loss == 0) & (average_gain == 0), 50.0)


def _validate_series(values: pd.Series, name: str) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if values.empty:
        raise ValueError(f"{name} cannot be empty")
    if values.index.has_duplicates:
        raise ValueError(f"{name} index cannot contain duplicates")
    if not values.index.is_monotonic_increasing:
        raise ValueError(f"{name} must be ordered chronologically")
    if values.isna().any():
        raise ValueError(f"{name} cannot contain missing values")
    return values.astype(float).rename(name)
