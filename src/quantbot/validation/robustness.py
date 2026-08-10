"""Nested ensemble selection and statistical robustness reporting."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.backtest.metrics import cagr, max_drawdown, sharpe_ratio
from quantbot.models import get_model_factory

from .walk_forward import (
    ProbabilisticClassifier,
    WalkForwardResult,
    run_spy_walk_forward,
    walk_forward_predict,
)

TRADING_DAYS = 252


@dataclass(frozen=True)
class RobustnessResult:
    """All tables and detailed runs produced by the robustness analysis."""

    comparison: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    uncertainty: pd.DataFrame
    logistic: WalkForwardResult
    ensemble: WalkForwardResult


class NestedMomentumEnsemble:
    """Blend an ML probability with trend, tuning the blend on inner OOS rows."""

    def __init__(
        self,
        model_factory: Callable[[], ProbabilisticClassifier],
        *,
        blend_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        inner_train_years: int = 2,
        audit: list[pd.DataFrame] | None = None,
    ) -> None:
        if not blend_grid or any(not 0.0 <= value <= 1.0 for value in blend_grid):
            raise ValueError("blend weights must be between zero and one")
        self.model_factory = model_factory
        self.blend_grid = tuple(float(value) for value in blend_grid)
        self.inner_train_years = inner_train_years
        self.audit = audit
        self.model: ProbabilisticClassifier | None = None
        self.blend_weight = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NestedMomentumEnsemble:
        _require_momentum(X)
        inner_probability = walk_forward_predict(
            X,
            y,
            self.model_factory,
            min_train_years=self.inner_train_years,
            test_years=1,
        )
        if not inner_probability.empty:
            momentum = _momentum_probability(X.loc[inner_probability.index])
            actual = y.loc[inner_probability.index].astype(float)
            scores = {
                weight: float(
                    ((weight * inner_probability + (1.0 - weight) * momentum - actual) ** 2).mean()
                )
                for weight in self.blend_grid
            }
            self.blend_weight = min(
                self.blend_grid,
                key=lambda weight: (scores[weight], abs(weight - 0.5)),
            )
        self.model = self.model_factory()
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ensemble must be fitted before prediction")
        _require_momentum(X)
        raw = np.asarray(self.model.predict_proba(X))
        model_probability = raw[:, 1] if raw.ndim == 2 else raw
        momentum_probability = _momentum_probability(X).to_numpy()
        probability = (
            self.blend_weight * model_probability + (1.0 - self.blend_weight) * momentum_probability
        )
        if self.audit is not None:
            self.audit.append(
                pd.DataFrame(
                    {
                        "model_probability": model_probability,
                        "momentum_probability": momentum_probability,
                        "model_blend_weight": self.blend_weight,
                    },
                    index=X.index,
                )
            )
        return np.column_stack([1.0 - probability, probability])


def _require_momentum(X: pd.DataFrame) -> None:
    if "return_20d" not in X:
        raise ValueError("ensemble requires the point-in-time return_20d feature")


def _momentum_probability(X: pd.DataFrame) -> pd.Series:
    """Map positive/negative 20-day momentum to conservative probabilities."""
    values = np.where(X["return_20d"] >= 0.0, 0.65, 0.35)
    return pd.Series(values, index=X.index, name="momentum_probability")


def newey_west_mean(
    returns: pd.Series,
    *,
    max_lag: int | None = None,
) -> dict[str, float]:
    """Estimate annualized mean return and its HAC confidence interval."""
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("at least two returns are required")
    lag = int(np.floor(4.0 * (len(values) / 100.0) ** (2.0 / 9.0))) if max_lag is None else max_lag
    if lag < 0 or lag >= len(values):
        raise ValueError("max_lag must be between zero and len(returns) - 1")
    centered = values - values.mean()
    long_run_variance = float(np.dot(centered, centered) / len(values))
    for offset in range(1, lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / len(values))
        long_run_variance += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    daily_se = np.sqrt(max(long_run_variance, 0.0) / len(values))
    annualized_mean = float(values.mean() * TRADING_DAYS)
    annualized_se = float(daily_se * TRADING_DAYS)
    return {
        "Annualized mean": annualized_mean,
        "NW standard error": annualized_se,
        "NW t-stat": annualized_mean / annualized_se if annualized_se else float("nan"),
        "Mean CI lower": annualized_mean - 1.96 * annualized_se,
        "Mean CI upper": annualized_mean + 1.96 * annualized_se,
        "NW lags": float(lag),
    }


def block_bootstrap_sharpe(
    returns: pd.Series,
    *,
    n_bootstrap: int = 1_000,
    block_size: int = 20,
    random_state: int = 42,
) -> dict[str, float]:
    """Moving-block bootstrap interval that preserves short-run dependence."""
    values = returns.dropna().to_numpy(dtype=float)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 1 <= block_size <= len(values):
        raise ValueError("block_size must be between one and the sample length")
    rng = np.random.default_rng(random_state)
    blocks_needed = int(np.ceil(len(values) / block_size))
    max_start = len(values) - block_size
    estimates = np.empty(n_bootstrap)
    for sample_number in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        sample = np.concatenate([values[start : start + block_size] for start in starts])[
            : len(values)
        ]
        std = sample.std(ddof=1)
        estimates[sample_number] = sample.mean() / std * np.sqrt(TRADING_DAYS) if std else np.nan
    finite = estimates[np.isfinite(estimates)]
    lower, upper = (
        np.quantile(finite, [0.025, 0.975]) if len(finite) else (float("nan"), float("nan"))
    )
    return {
        "Sharpe": sharpe_ratio(pd.Series(values)),
        "Sharpe CI lower": float(lower),
        "Sharpe CI upper": float(upper),
        "Bootstrap samples": float(n_bootstrap),
        "Block size": float(block_size),
    }


def run_robustness_analysis(
    close: pd.Series,
    *,
    volume: pd.Series | None = None,
    model_factory: Callable[[], ProbabilisticClassifier] | None = None,
    min_train_years: int = 5,
    test_years: int = 1,
    entry_threshold: float = 0.55,
    exit_threshold: float = 0.45,
    cost_bps: float = 5.0,
    cost_grid: Sequence[float] = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0),
    n_bootstrap: int = 1_000,
    random_state: int = 42,
    out_dir: str | Path | None = "results",
) -> RobustnessResult:
    """Compare the base model and nested ensemble under costs and uncertainty."""
    if not cost_grid or any(value < 0.0 for value in cost_grid):
        raise ValueError("cost assumptions must be non-negative")
    model_factory = model_factory or get_model_factory("logistic")
    shared = dict(
        min_train_years=min_train_years,
        test_years=test_years,
        threshold=entry_threshold,
        exit_threshold=exit_threshold,
        cost_bps=cost_bps,
        volume=volume,
        out_dir=None,
    )
    logistic = run_spy_walk_forward(close, model_factory=model_factory, **shared)

    audit: list[pd.DataFrame] = []

    def ensemble_factory() -> NestedMomentumEnsemble:
        return NestedMomentumEnsemble(model_factory, audit=audit)

    raw_ensemble = run_spy_walk_forward(close, model_factory=ensemble_factory, **shared)
    components = pd.concat(audit).sort_index()
    predictions = raw_ensemble.predictions.join(components)
    ensemble = WalkForwardResult(
        predictions,
        raw_ensemble.weights,
        raw_ensemble.backtest,
        raw_ensemble.benchmark_returns,
        raw_ensemble.metrics,
    )

    comparison = pd.DataFrame(
        {
            "Logistic": logistic.metrics.loc["Walk-forward strategy"],
            "Nested ML + momentum": ensemble.metrics.loc["Walk-forward strategy"],
            "SPY buy and hold": ensemble.metrics.loc["SPY buy and hold"],
            "SPY above 50-day MA": ensemble.metrics.loc["SPY above 50-day MA"],
        }
    ).T
    streams = _strategy_streams(logistic, ensemble)
    cost_sensitivity = _cost_sensitivity(streams, cost_grid)
    uncertainty = _uncertainty_table(
        streams,
        cost_bps=cost_bps,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    result = RobustnessResult(comparison, cost_sensitivity, uncertainty, logistic, ensemble)
    if out_dir is not None:
        _write_robustness(result, Path(out_dir), cost_bps)
    return result


def _strategy_streams(
    logistic: WalkForwardResult,
    ensemble: WalkForwardResult,
) -> dict[str, tuple[pd.Series, pd.Series]]:
    index = ensemble.predictions.index

    def model_stream(result: WalkForwardResult) -> tuple[pd.Series, pd.Series]:
        gross = result.predictions["gross_strategy_return"]
        position = result.predictions["position"]
        turnover = position.diff().abs().fillna(position.iloc[0])
        return gross, turnover

    ma_position = ensemble.predictions["benchmark_ma50_position"]
    ma_turnover = ma_position.diff().abs().fillna(ma_position.iloc[0])
    return {
        "Logistic": model_stream(logistic),
        "Nested ML + momentum": model_stream(ensemble),
        "SPY buy and hold": (
            ensemble.predictions["market_return"],
            pd.Series(0.0, index=index),
        ),
        "SPY above 50-day MA": (
            ensemble.predictions["benchmark_ma50_gross_return"],
            ma_turnover,
        ),
    }


def _cost_sensitivity(
    streams: dict[str, tuple[pd.Series, pd.Series]],
    cost_grid: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name, (gross, turnover) in streams.items():
        for cost in cost_grid:
            net = gross - turnover * (float(cost) / 10_000.0)
            rows.append(
                {
                    "strategy": name,
                    "cost_bps": float(cost),
                    "CAGR": cagr(net),
                    "Sharpe": sharpe_ratio(net),
                    "Maximum drawdown": max_drawdown(net),
                    "Total turnover": float(turnover.sum()),
                }
            )
    return pd.DataFrame(rows)


def _uncertainty_table(
    streams: dict[str, tuple[pd.Series, pd.Series]],
    *,
    cost_bps: float,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for offset, (name, (gross, turnover)) in enumerate(streams.items()):
        returns = gross - turnover * (cost_bps / 10_000.0)
        rows[name] = {
            **newey_west_mean(returns),
            **block_bootstrap_sharpe(
                returns,
                n_bootstrap=n_bootstrap,
                random_state=random_state + offset,
            ),
        }
    return pd.DataFrame(rows).T


def _write_robustness(result: RobustnessResult, out_dir: Path, cost_bps: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    result.comparison.to_csv(out_dir / "robustness_comparison.csv", index_label="strategy")
    result.cost_sensitivity.to_csv(out_dir / "cost_sensitivity.csv", index=False)
    result.uncertainty.to_csv(out_dir / "uncertainty.csv", index_label="strategy")
    result.ensemble.predictions.to_csv(out_dir / "ensemble_predictions.csv")

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    for name, group in result.cost_sensitivity.groupby("strategy", sort=False):
        left.plot(group["cost_bps"], group["Sharpe"], marker="o", label=name)
    left.axhline(0.0, color="black", linewidth=0.8)
    left.set(title="Transaction-cost sensitivity", xlabel="Cost (bps)", ylabel="Sharpe")
    left.grid(alpha=0.3)
    left.legend(fontsize=8)

    returns = {
        "Logistic": result.logistic.predictions["net_strategy_return"],
        "Nested ML + momentum": result.ensemble.predictions["net_strategy_return"],
        **result.ensemble.benchmark_returns,
    }
    for name, values in returns.items():
        right.plot((1.0 + values).cumprod(), label=name)
    right.set(title=f"Common-date comparison ({cost_bps:g} bps)", ylabel="Growth of $1")
    right.grid(alpha=0.3)
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "robustness.png", dpi=140)
    plt.close(fig)

    _write_model_card(result, out_dir / "MODEL_CARD.md", cost_bps)


def _write_model_card(result: RobustnessResult, path: Path, cost_bps: float) -> None:
    ml = result.comparison.loc["Logistic"]
    ensemble = result.comparison.loc["Nested ML + momentum"]
    benchmark = result.comparison.loc["SPY buy and hold"]
    trend = result.comparison.loc["SPY above 50-day MA"]
    interval = result.uncertainty.loc["Nested ML + momentum"]
    winner = result.comparison["Sharpe"].astype(float).idxmax()
    conclusion = (
        "The ensemble does not beat buy-and-hold on out-of-sample Sharpe. It remains a "
        "research candidate and is not evidence of deployable alpha."
        if float(ensemble["Sharpe"]) <= float(benchmark["Sharpe"])
        else "The ensemble beats buy-and-hold on point-estimate Sharpe, but uncertainty and "
        "paper-trading evidence are still required before treating the result as durable."
    )
    ml_row = _model_card_row("Logistic", ml)
    ensemble_row = _model_card_row("Nested ML + momentum", ensemble)
    benchmark_row = _model_card_row("SPY buy and hold", benchmark, include_auc=False)
    trend_row = _model_card_row("SPY above 50-day MA", trend, include_auc=False)
    sharpe_interval = (
        f"[{float(interval['Sharpe CI lower']):.3f}, {float(interval['Sharpe CI upper']):.3f}]"
    )
    text = f"""# Model card: SPY daily direction research

## Intended use

Research and Alpaca paper trading only. The system predicts next-session SPY direction from
point-in-time daily price and volume features. It is not financial advice and is not approved
for live-capital execution.

## Validation design

- Expanding outer walk-forward folds with all training dates before every test date.
- The ML/momentum blend is selected using only inner walk-forward predictions from the outer
  training fold; the outer test fold is never used for tuning.
- Logistic, ensemble, 50-day moving average, and buy-and-hold share evaluation dates.
- Headline costs are {cost_bps:g} bps, with a 0--50 bps sensitivity sweep.
- Newey--West uncertainty accounts for autocorrelation in the mean; a 20-session moving-block
  bootstrap estimates the Sharpe interval.

## Headline out-of-sample results

| Portfolio | Sharpe | CAGR | Maximum drawdown | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
{ml_row}
{ensemble_row}
{benchmark_row}
{trend_row}

The highest point-estimate Sharpe belongs to **{winner}**. The ensemble's 95% block-bootstrap
Sharpe interval is **{sharpe_interval}**.

## Conclusion

{conclusion}

## Known limitations

- One liquid ETF and one historical regime do not establish generalization.
- Flat transaction costs omit spread variation, market impact, tax, and slippage shocks.
- Multiple strategy variants create selection bias; the reported uncertainty does not fully
  correct for every experiment attempted by the team.
- Paper fills do not reproduce real queue position or emotional/operational risk.
"""
    path.write_text(text, encoding="utf-8")


def _model_card_row(name: str, values: pd.Series, *, include_auc: bool = True) -> str:
    auc = f"{float(values['ROC AUC']):.3f}" if include_auc else "n/a"
    return (
        f"| {name} | {float(values['Sharpe']):.3f} | {float(values['CAGR']):.1%} | "
        f"{float(values['Maximum drawdown']):.1%} | {auc} |"
    )
