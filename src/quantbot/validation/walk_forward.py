"""Expanding-window, walk-forward validation and SPY experiment reporting."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from quantbot.backtest.engine import BacktestResult, run_backtest
from quantbot.backtest.metrics import (
    annual_volatility,
    cagr,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
)
from quantbot.features import build_dataset


class ProbabilisticClassifier(Protocol):
    """Smallest estimator interface needed by the harness."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> object: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class WalkForwardResult:
    """Out-of-sample predictions and the backtest they drive."""

    predictions: pd.DataFrame
    weights: pd.DataFrame
    backtest: BacktestResult
    benchmark_returns: dict[str, pd.Series]
    metrics: pd.DataFrame


def probabilities_to_positions(
    probability: pd.Series,
    entry_threshold: float = 0.55,
    exit_threshold: float | None = None,
) -> pd.Series:
    """Convert probabilities to long/cash positions.

    With an exit threshold, probabilities in the neutral band preserve the
    previous position. This hysteresis prevents repeated trades caused by
    probabilities oscillating around one decision boundary.
    """
    if not 0.0 <= entry_threshold <= 1.0:
        raise ValueError("entry threshold must be between zero and one")
    if exit_threshold is None:
        return (probability >= entry_threshold).astype(float).rename("position")
    if not 0.0 <= exit_threshold < entry_threshold:
        raise ValueError("exit threshold must be between zero and the entry threshold")

    current = 0.0
    positions: list[float] = []
    for value in probability:
        if value >= entry_threshold:
            current = 1.0
        elif value <= exit_threshold:
            current = 0.0
        positions.append(current)
    return pd.Series(positions, index=probability.index, name="position")


def chronological_folds(
    index: pd.DatetimeIndex,
    min_train_years: int = 5,
    test_years: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield expanding train/test row positions with calendar-year boundaries."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("index must be sorted and unique")
    if min_train_years <= 0 or test_years <= 0:
        raise ValueError("min_train_years and test_years must be positive")
    if len(index) == 0:
        return

    first_test_start = index[0] + pd.DateOffset(years=min_train_years)
    test_start = first_test_start
    while test_start <= index[-1]:
        test_end = test_start + pd.DateOffset(years=test_years)
        train = np.flatnonzero(index < test_start)
        test = np.flatnonzero((index >= test_start) & (index < test_end))
        if len(train) and len(test):
            yield train, test
        test_start = test_end


def fold_metadata(
    index: pd.DatetimeIndex,
    min_train_years: int = 5,
    test_years: int = 1,
) -> pd.DataFrame:
    """Per-test-date fold id and train/test window boundaries, for auditing.

    Reuses :func:`chronological_folds`, so the boundaries reported here are the
    exact windows the models were trained and evaluated on. Indexed by test date
    with one row per prediction; enables verifying ``train_end < test_start`` and
    which fold produced each row directly from ``predictions.csv``.
    """
    records: list[dict[str, object]] = []
    for fold_id, (train_pos, test_pos) in enumerate(
        chronological_folds(index, min_train_years=min_train_years, test_years=test_years)
    ):
        train_dates, test_dates = index[train_pos], index[test_pos]
        for date in test_dates:
            records.append(
                {
                    "date": date,
                    "fold_id": fold_id,
                    "train_start": train_dates.min(),
                    "train_end": train_dates.max(),
                    "test_start": test_dates.min(),
                    "test_end": test_dates.max(),
                }
            )
    frame = pd.DataFrame.from_records(records)
    return frame if frame.empty else frame.set_index("date")


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], ProbabilisticClassifier],
    min_train_years: int = 5,
    test_years: int = 1,
) -> pd.Series:
    """Fit fresh models on past data and return only out-of-sample probabilities."""
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index")
    if X.isna().any().any() or y.isna().any():
        raise ValueError("X and y must not contain missing values")

    pieces: list[pd.Series] = []
    for train_pos, test_pos in chronological_folds(
        X.index, min_train_years=min_train_years, test_years=test_years
    ):
        train_dates, test_dates = X.index[train_pos], X.index[test_pos]
        if train_dates.max() >= test_dates.min():
            raise RuntimeError("walk-forward split is not chronological")
        model = model_factory()
        model.fit(X.iloc[train_pos], y.iloc[train_pos])
        probabilities = np.asarray(model.predict_proba(X.iloc[test_pos]))
        positive = probabilities[:, 1] if probabilities.ndim == 2 else probabilities
        if len(positive) != len(test_pos):
            raise ValueError("model returned the wrong number of predictions")
        pieces.append(pd.Series(positive, index=test_dates, name="probability"))

    if not pieces:
        return pd.Series(dtype=float, name="probability")
    result = pd.concat(pieces).sort_index()
    if result.index.has_duplicates:
        raise RuntimeError("a test date was predicted more than once")
    return result


def build_spy_dataset(
    close: pd.Series,
    volume: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return model features, next-day target, and aligned next-day returns."""
    dataset = build_dataset(close, volume=volume)
    features = dataset.drop(columns="target")
    target = dataset["target"].astype(int)
    next_return = (close.shift(-1) / close - 1.0).rename("market_return")
    return features, target, next_return.reindex(dataset.index)


def _metric_row(
    returns: pd.Series,
    positions: pd.Series,
    turnover: pd.Series,
) -> dict[str, float]:
    years = len(returns) / 252
    return {
        "CAGR": cagr(returns),
        "Cumulative return": float((1.0 + returns).prod() - 1.0),
        "Annualized volatility": annual_volatility(returns),
        "Sharpe": sharpe_ratio(returns),
        "Maximum drawdown": max_drawdown(returns),
        "Hit rate": hit_rate(returns),
        "Position changes": float(positions.ne(positions.shift()).sum() - 1),
        "Total turnover": float(turnover.sum()),
        "Annualized turnover": float(turnover.sum() / years) if years > 0 else float("nan"),
    }


def _classification_metrics(target: pd.Series, probability: pd.Series) -> dict[str, float]:
    predicted = probability.ge(0.5).astype(int)
    positive = target.eq(1)
    predicted_positive = predicted.eq(1)
    true_positive = int((positive & predicted_positive).sum())
    false_positive = int((~positive & predicted_positive).sum())
    false_negative = int((positive & ~predicted_positive).sum())
    n_positive = int(positive.sum())
    n_negative = len(target) - n_positive

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    ranks = probability.rank(method="average")
    auc = (
        (float(ranks[positive].sum()) - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
        if n_positive and n_negative
        else float("nan")
    )
    return {
        "Accuracy": float((predicted == target).mean()),
        "Precision": (
            true_positive / precision_denominator if precision_denominator else float("nan")
        ),
        "Recall": true_positive / recall_denominator if recall_denominator else float("nan"),
        "ROC AUC": auc,
    }


def run_spy_walk_forward(
    close: pd.Series,
    model_factory: Callable[[], ProbabilisticClassifier] | None = None,
    min_train_years: int = 5,
    test_years: int = 1,
    threshold: float = 0.55,
    exit_threshold: float | None = None,
    cost_bps: float = 5.0,
    feature_columns: tuple[str, ...] | None = None,
    volume: pd.Series | None = None,
    out_dir: str | Path | None = "results",
) -> WalkForwardResult:
    """Run and optionally save the complete inspectable SPY experiment."""
    close = close.dropna().sort_index().rename("close")
    if close.index.has_duplicates:
        raise ValueError("close index must contain unique dates")
    probabilities_to_positions(pd.Series(dtype=float), threshold, exit_threshold)
    X, target, forward_return = build_spy_dataset(close, volume=volume)
    if feature_columns is not None:
        missing = sorted(set(feature_columns) - set(X.columns))
        if missing:
            raise ValueError(f"unknown feature columns: {', '.join(missing)}")
        if not feature_columns:
            raise ValueError("at least one feature column is required")
        X = X.loc[:, list(feature_columns)]

    if model_factory is None:
        try:
            from quantbot.models import get_model_factory
        except ImportError as exc:
            raise ImportError(
                "install quantbot with the 'ml' extra to run this experiment"
            ) from exc
        model_factory = get_model_factory("logistic")

    probability = walk_forward_predict(
        X, target, model_factory, min_train_years=min_train_years, test_years=test_years
    )
    if probability.empty:
        raise ValueError("not enough history to create a walk-forward test fold")

    position = probabilities_to_positions(probability, threshold, exit_threshold)
    evaluation_dates = probability.index

    # A prediction made at date t becomes the engine's held position at t+1.
    engine_prices = close.to_frame("SPY")
    target_weights = pd.DataFrame(0.0, index=engine_prices.index, columns=["SPY"])
    target_weights.loc[evaluation_dates, "SPY"] = position
    backtest = run_backtest(engine_prices, target_weights, cost_bps=cost_bps)

    realized_dates = engine_prices.index[engine_prices.index.get_indexer(evaluation_dates) + 1]

    def _forward(series: pd.Series, name: str) -> pd.Series:
        # A position held from t to t+1 realises the t+1 value, labelled at t.
        return pd.Series(
            series.reindex(realized_dates).to_numpy(), index=evaluation_dates, name=name
        )

    strategy_forward = _forward(backtest.returns, "net_strategy_return")
    gross_strategy_forward = _forward(backtest.gross_returns, "gross_strategy_return")
    cost_forward = _forward(backtest.costs, "cost")
    market_forward = forward_return.reindex(evaluation_dates)

    buy_hold = market_forward.rename("SPY buy and hold")
    moving_average_position = (close > close.rolling(50).mean()).astype(float)
    ma_gross_returns = (
        (moving_average_position * forward_return).reindex(evaluation_dates).fillna(0.0)
    )
    ma_turnover = moving_average_position.diff().abs().reindex(evaluation_dates).fillna(0.0)
    ma_costs = ma_turnover * (cost_bps / 10_000.0)
    ma_returns = ma_gross_returns - ma_costs
    ma_returns.name = "SPY above 50-day MA"
    strategy_returns = strategy_forward.rename("Walk-forward strategy")

    strategy_turnover = backtest.turnover.reindex(realized_dates)
    strategy_turnover.index = evaluation_dates
    benchmarks = {"SPY buy and hold": buy_hold, "SPY above 50-day MA": ma_returns}
    strategy_metrics = _metric_row(strategy_returns, position, strategy_turnover)
    strategy_metrics.update(_classification_metrics(target.reindex(evaluation_dates), probability))
    metrics = pd.DataFrame(
        {
            "Walk-forward strategy": strategy_metrics,
            "SPY buy and hold": _metric_row(
                buy_hold,
                pd.Series(1.0, index=evaluation_dates),
                pd.Series(0.0, index=evaluation_dates),
            ),
            "SPY above 50-day MA": _metric_row(
                ma_returns,
                moving_average_position.reindex(evaluation_dates),
                ma_turnover,
            ),
        }
    ).T

    folds = fold_metadata(X.index, min_train_years=min_train_years, test_years=test_years)
    predictions = pd.concat(
        [
            close.reindex(evaluation_dates).rename("close"),
            folds.reindex(evaluation_dates),
            target.reindex(evaluation_dates),
            probability,
            position,
            market_forward,
            gross_strategy_forward,
            pd.Series(float(cost_bps), index=evaluation_dates, name="cost_bps"),
            cost_forward,
            strategy_forward,
            moving_average_position.reindex(evaluation_dates).rename("benchmark_ma50_position"),
            ma_gross_returns.rename("benchmark_ma50_gross_return"),
            ma_costs.rename("benchmark_ma50_cost"),
            ma_returns.reindex(evaluation_dates).rename("benchmark_ma50_return"),
        ],
        axis=1,
    )
    predictions.index.name = "date"

    result = WalkForwardResult(predictions, target_weights, backtest, benchmarks, metrics)
    if out_dir is not None:
        _write_results(result, Path(out_dir))
    return result


def _write_results(result: WalkForwardResult, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(out_dir / "metrics.csv", index_label="portfolio")
    result.predictions.to_csv(out_dir / "predictions.csv")

    curves = {
        "Walk-forward strategy": (1.0 + result.predictions["net_strategy_return"]).cumprod(),
        **{name: (1.0 + returns).cumprod() for name, returns in result.benchmark_returns.items()},
    }
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for name, curve in curves.items():
        top.plot(curve, label=name)
    top.set_title("SPY walk-forward evaluation")
    top.set_ylabel("Growth of $1")
    top.grid(alpha=0.3)
    top.legend()
    strategy_curve = curves["Walk-forward strategy"]
    drawdown = strategy_curve / strategy_curve.cummax() - 1.0
    bottom.fill_between(drawdown.index, drawdown, 0, color="crimson", alpha=0.4)
    bottom.set_ylabel("Strategy drawdown")
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "performance.png", dpi=140)
    plt.close(fig)
