"""The contract every signal must satisfy.

A signal turns a price history into target portfolio weights. Keeping this
interface tiny is what lets the backtester, the paper broker and the live
executor all consume any strategy without knowing anything about it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Signal(ABC):
    """Base class for all strategies."""

    name: str = "signal"

    @abstractmethod
    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return target weights, indexed and columned exactly like ``prices``.

        The weights on row *t* may only use information available at the close
        of day *t*. Any use of later data is a lookahead bug and will silently
        produce a beautiful, worthless equity curve.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


def normalise_weights(raw: pd.DataFrame, gross_leverage: float = 1.0) -> pd.DataFrame:
    """Scale each row so total absolute exposure equals ``gross_leverage``.

    Rows with no exposure at all are left flat rather than divided by zero.
    """
    gross = raw.abs().sum(axis=1)
    scale = (gross_leverage / gross).where(gross > 0, 0.0)
    return raw.mul(scale, axis=0).fillna(0.0)
