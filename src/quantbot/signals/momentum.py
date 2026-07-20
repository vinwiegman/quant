"""Cross-sectional momentum -- the benchmark every other signal has to beat.

Rank instruments by their trailing return, go long the winners and short the
losers. It is deliberately simple: if a fancier model cannot beat this, the
fancier model is not adding anything.
"""

from __future__ import annotations

import pandas as pd

from .base import Signal, normalise_weights


class CrossSectionalMomentum(Signal):
    """Long the top ``n_long`` performers, short the bottom ``n_short``.

    Args:
        lookback: Number of trading days used to measure past performance.
        skip: Most recent days to exclude, the standard short-term reversal
            guard. Momentum research typically skips the last month.
        n_long: How many instruments to hold long.
        n_short: How many to hold short. Zero gives a long-only strategy.
        rebalance_days: Trade every N days instead of daily, to cut turnover.
        gross_leverage: Total absolute exposure, 1.0 meaning fully invested.
    """

    name = "cross_sectional_momentum"

    def __init__(
        self,
        lookback: int = 126,
        skip: int = 21,
        n_long: int = 3,
        n_short: int = 3,
        rebalance_days: int = 21,
        gross_leverage: float = 1.0,
    ) -> None:
        if lookback <= skip:
            raise ValueError("lookback must be longer than skip")
        if n_long < 1:
            raise ValueError("n_long must be at least 1")
        self.lookback = lookback
        self.skip = skip
        self.n_long = n_long
        self.n_short = n_short
        self.rebalance_days = max(1, rebalance_days)
        self.gross_leverage = gross_leverage

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        # Return over the lookback window, ending `skip` days ago.
        past = prices.shift(self.skip)
        momentum = past / past.shift(self.lookback - self.skip) - 1.0

        raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for date, row in momentum.iterrows():
            scores = row.dropna()
            if len(scores) < self.n_long + self.n_short:
                continue
            ranked = scores.sort_values(ascending=False)
            raw.loc[date, ranked.index[: self.n_long]] = 1.0
            if self.n_short > 0:
                raw.loc[date, ranked.index[-self.n_short :]] = -1.0

        weights = normalise_weights(raw, self.gross_leverage)
        return _hold_between_rebalances(weights, self.rebalance_days)


def _hold_between_rebalances(weights: pd.DataFrame, every: int) -> pd.DataFrame:
    """Keep weights fixed except on rebalance dates, so turnover stays low."""
    if every <= 1:
        return weights
    mask = pd.Series(False, index=weights.index)
    mask.iloc[::every] = True
    return weights.where(mask, other=pd.NA).ffill().fillna(0.0)
