"""Tests for cached OHLCV market-data loading."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from quantbot.data import load_ohlcv


def test_load_ohlcv_normalises_yfinance_output_and_uses_cache(monkeypatch, tmp_path):
    index = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    columns = pd.MultiIndex.from_product([fields, ["SPY"]])
    raw = pd.DataFrame(
        np.array(
            [
                [99.0, 101.0, 98.0, 100.0, 1_000_000.0],
                [100.0, 102.0, 99.0, 101.0, 1_100_000.0],
                [101.0, 103.0, 100.0, 102.0, 1_200_000.0],
            ]
        ),
        index=index,
        columns=columns,
    )
    fake_yfinance = SimpleNamespace(download=lambda *args, **kwargs: raw)
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)
    cached: dict[str, pd.DataFrame] = {}

    def write_cache(frame, path):
        cached[str(path)] = frame.copy()
        path.touch()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_cache)
    monkeypatch.setattr(pd, "read_parquet", lambda path: cached[str(path)].copy())

    first = load_ohlcv(
        "spy",
        start="2024-01-01",
        end="2024-01-10",
        cache_dir=tmp_path,
    )
    fake_yfinance.download = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("cache was not used")
    )
    second = load_ohlcv(
        "SPY",
        start="2024-01-01",
        end="2024-01-10",
        cache_dir=tmp_path,
    )

    assert list(first.columns) == ["open", "high", "low", "close", "volume"]
    assert first.index.tz is None
    assert first["close"].tolist() == [100.0, 101.0, 102.0]
    pd.testing.assert_frame_equal(first, second)
