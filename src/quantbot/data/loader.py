"""Price data loading with an on-disk cache.

Every run of a backtest would otherwise hammer the data provider and be slow
and non-reproducible. Downloads are cached as Parquet keyed by the request, so
a repeated backtest reads from disk and gives byte-identical inputs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DEFAULT_CACHE = Path.home() / ".cache" / "quantbot"


def _cache_key(tickers: tuple[str, ...], start: str, end: str, interval: str) -> str:
    payload = "|".join([",".join(tickers), start, end, interval])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    interval: str = "1d",
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download adjusted close prices, or read them from the local cache.

    Args:
        tickers: Instrument symbols, e.g. ``["AAPL", "MSFT"]``.
        start: First date, ``YYYY-MM-DD``.
        end: Last date, ``YYYY-MM-DD``.
        interval: Bar size understood by the provider, e.g. ``1d`` or ``1h``.
        cache_dir: Where to store Parquet files. Defaults to ``~/.cache/quantbot``.
        force_refresh: Ignore any cached copy and download again.

    Returns:
        A DataFrame of close prices indexed by date, one column per ticker,
        with columns in the order given and any all-empty ones dropped.
    """
    tickers = list(dict.fromkeys(tickers))  # de-duplicate, keep order
    cache_dir = cache_dir or DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_cache_key(tuple(tickers), start, end, interval)}.parquet"

    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    import yfinance as yf  # imported lazily so tests need no network stack

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError(f"no data returned for {tickers} between {start} and {end}")

    # yfinance returns a column MultiIndex for multiple tickers, flat for one.
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(raw.columns, pd.MultiIndex):
        close.columns = tickers[:1]

    close = close.reindex(columns=[t for t in tickers if t in close.columns])
    close = close.dropna(axis=1, how="all").ffill().dropna(how="any")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.to_parquet(path)
    return close
