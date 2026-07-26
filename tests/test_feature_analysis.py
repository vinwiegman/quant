"""Tests for fold-safe feature importance and ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbot.analysis import run_feature_analysis
from quantbot.features import make_features


def sample_close() -> pd.Series:
    index = pd.date_range("2010-01-01", "2016-12-31", freq="B")
    time = np.arange(len(index))
    close = 100.0 * np.exp(0.0003 * time) * (1.0 + 0.02 * np.sin(time / 8.0))
    return pd.Series(close, index=index, name="SPY")


def test_feature_analysis_covers_every_feature():
    close = sample_close()

    result = run_feature_analysis(
        close,
        models=("logistic",),
        n_repeats=1,
        out_dir=None,
    )

    expected = set(make_features(close).columns)
    importance = result.permutation_importance
    ablation = result.ablation
    assert set(importance["feature"]) == expected
    assert set(ablation["dropped_feature"]) == expected | {"(baseline)"}
    assert importance["importance_mean"].notna().all()
    assert {"roc_auc", "sharpe", "delta_roc_auc", "delta_sharpe"} <= set(ablation.columns)


def test_feature_analysis_requires_repeats():
    with pytest.raises(ValueError, match="n_repeats"):
        run_feature_analysis(
            sample_close(),
            models=("logistic",),
            n_repeats=0,
            out_dir=None,
        )
