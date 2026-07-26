"""Fair, fold-identical comparison of the supported ML classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantbot.models import MODEL_NAMES, ModelName, get_model_factory

from .walk_forward import WalkForwardResult, run_spy_walk_forward


@dataclass(frozen=True)
class ModelComparisonResult:
    """Metrics and detailed results for every evaluated model."""

    metrics: pd.DataFrame
    models: dict[str, WalkForwardResult]


def run_model_comparison(
    close: pd.Series,
    models: tuple[ModelName, ...] = MODEL_NAMES,
    min_train_years: int = 5,
    test_years: int = 1,
    entry_threshold: float = 0.55,
    exit_threshold: float | None = 0.45,
    cost_bps: float = 5.0,
    volume: pd.Series | None = None,
    out_dir: str | Path | None = "results",
) -> ModelComparisonResult:
    """Evaluate all models with identical data, folds, costs, and policy."""
    if not models:
        raise ValueError("at least one model is required")

    results: dict[str, WalkForwardResult] = {}
    rows: dict[str, pd.Series] = {}
    for name in models:
        result = run_spy_walk_forward(
            close,
            model_factory=get_model_factory(name),
            min_train_years=min_train_years,
            test_years=test_years,
            threshold=entry_threshold,
            exit_threshold=exit_threshold,
            cost_bps=cost_bps,
            volume=volume,
            out_dir=None,
        )
        results[name] = result
        rows[name] = result.metrics.loc["Walk-forward strategy"]

    comparison = ModelComparisonResult(pd.DataFrame(rows).T, results)
    if out_dir is not None:
        _write_comparison(comparison, Path(out_dir))
    return comparison


def _write_comparison(result: ModelComparisonResult, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(out_dir / "model_comparison.csv", index_label="model")
    for name, model_result in result.models.items():
        model_result.predictions.to_csv(out_dir / f"{name.replace('-', '_')}_predictions.csv")

    fig, axis = plt.subplots(figsize=(11, 5))
    for name, model_result in result.models.items():
        equity = (1.0 + model_result.predictions["strategy_return"]).cumprod()
        axis.plot(equity, label=name)
    axis.set_title("Out-of-sample ML strategy comparison")
    axis.set_ylabel("Growth of $1")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=140)
    plt.close(fig)
