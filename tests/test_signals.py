"""Tests for signal generation and weight normalisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbot.signals.base import normalise_weights
from quantbot.signals.momentum import CrossSectionalMomentum


def trending_prices(n: int = 400) -> pd.DataFrame:
    """Four assets with deliberately different, constant drifts."""
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    drifts = {"WINNER": 0.0015, "GOOD": 0.0008, "MEH": 0.0002, "LOSER": -0.0010}
    return pd.DataFrame(
        {name: 100.0 * np.exp(drift * np.arange(n)) for name, drift in drifts.items()},
        index=index,
    )


def test_normalise_weights_scales_to_target_leverage():
    raw = pd.DataFrame({"A": [1.0, 2.0], "B": [-1.0, 0.0]})

    out = normalise_weights(raw, gross_leverage=1.0)

    assert out.abs().sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_normalise_weights_leaves_empty_rows_flat():
    raw = pd.DataFrame({"A": [0.0, 1.0], "B": [0.0, -1.0]})

    out = normalise_weights(raw)

    assert out.iloc[0].tolist() == [0.0, 0.0]
    assert not out.isna().any().any()


def test_momentum_goes_long_the_strongest_trend():
    prices = trending_prices()
    signal = CrossSectionalMomentum(lookback=126, skip=21, n_long=1, n_short=1, rebalance_days=1)

    weights = signal.generate(prices)
    final = weights.iloc[-1]

    assert final["WINNER"] > 0
    assert final["LOSER"] < 0
    assert final[["GOOD", "MEH"]].tolist() == [0.0, 0.0]


def test_momentum_is_flat_during_the_warmup_window():
    prices = trending_prices()
    signal = CrossSectionalMomentum(lookback=126, skip=21)

    weights = signal.generate(prices)

    # Nothing can be known before a full lookback of history exists.
    assert weights.iloc[:126].abs().sum().sum() == pytest.approx(0.0)


def test_weights_align_with_input_shape():
    prices = trending_prices()

    weights = CrossSectionalMomentum(n_long=1, n_short=1).generate(prices)

    assert weights.index.equals(prices.index)
    assert list(weights.columns) == list(prices.columns)


def test_less_frequent_rebalancing_lowers_turnover():
    prices = trending_prices()

    def turnover(days: int) -> float:
        w = CrossSectionalMomentum(n_long=1, n_short=1, rebalance_days=days).generate(prices)
        return float(w.diff().abs().sum().sum())

    assert turnover(63) <= turnover(1)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="lookback"):
        CrossSectionalMomentum(lookback=10, skip=21)
    with pytest.raises(ValueError, match="n_long"):
        CrossSectionalMomentum(n_long=0)
