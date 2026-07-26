"""Leakage-safe permutation importance and leave-one-feature-out ablation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantbot.models import MODEL_NAMES, ModelName, get_model_factory
from quantbot.validation.walk_forward import (
    build_spy_dataset,
    chronological_folds,
    run_spy_walk_forward,
)


@dataclass(frozen=True)
class FeatureAnalysisResult:
    """Feature importance and ablation results for each evaluated model."""

    permutation_importance: pd.DataFrame
    ablation: pd.DataFrame


def run_feature_analysis(
    close: pd.Series,
    models: tuple[ModelName, ...] = MODEL_NAMES,
    min_train_years: int = 5,
    test_years: int = 1,
    entry_threshold: float = 0.55,
    exit_threshold: float | None = 0.45,
    cost_bps: float = 5.0,
    n_repeats: int = 5,
    random_state: int = 42,
    out_dir: str | Path | None = "results",
) -> FeatureAnalysisResult:
    """Analyze features without fitting or scoring on future observations."""
    if not models:
        raise ValueError("at least one model is required")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least one")

    close = close.dropna().sort_index().rename("close")
    X, target, _ = build_spy_dataset(close)
    permutation_rows: list[dict[str, str | float | int]] = []
    ablation_rows: list[dict[str, str | float]] = []

    for model_name in models:
        permutation_rows.extend(
            _permutation_importance(
                X,
                target,
                model_name,
                min_train_years,
                test_years,
                n_repeats,
                random_state,
            )
        )
        ablation_rows.extend(
            _feature_ablation(
                close,
                tuple(X.columns),
                model_name,
                min_train_years,
                test_years,
                entry_threshold,
                exit_threshold,
                cost_bps,
            )
        )

    result = FeatureAnalysisResult(
        permutation_importance=pd.DataFrame(permutation_rows),
        ablation=pd.DataFrame(ablation_rows),
    )
    if out_dir is not None:
        _write_analysis(result, Path(out_dir))
    return result


def _permutation_importance(
    X: pd.DataFrame,
    target: pd.Series,
    model_name: ModelName,
    min_train_years: int,
    test_years: int,
    n_repeats: int,
    random_state: int,
) -> list[dict[str, str | float | int]]:
    decreases: dict[str, list[float]] = {feature: [] for feature in X.columns}
    folds = list(
        chronological_folds(
            X.index,
            min_train_years=min_train_years,
            test_years=test_years,
        )
    )
    if not folds:
        raise ValueError("not enough history to create a permutation test fold")
    for fold_number, (train_pos, test_pos) in enumerate(folds):
        model = get_model_factory(model_name)()
        X_train, X_test = X.iloc[train_pos], X.iloc[test_pos]
        y_train, y_test = target.iloc[train_pos], target.iloc[test_pos]
        model.fit(X_train, y_train)
        baseline = _roc_auc(y_test, _positive_probability(model.predict_proba(X_test)))

        for feature_number, feature in enumerate(X.columns):
            for repeat in range(n_repeats):
                seed = random_state + fold_number * 10_000 + feature_number * 100 + repeat
                rng = np.random.default_rng(seed)
                permuted = X_test.copy()
                permuted[feature] = rng.permutation(permuted[feature].to_numpy())
                score = _roc_auc(
                    y_test,
                    _positive_probability(model.predict_proba(permuted)),
                )
                decreases[feature].append(baseline - score)

    return [
        {
            "model": model_name,
            "feature": feature,
            "importance_mean": float(np.mean(values)),
            "importance_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "folds": len(folds),
            "repeats_per_fold": n_repeats,
        }
        for feature, values in decreases.items()
    ]


def _feature_ablation(
    close: pd.Series,
    features: tuple[str, ...],
    model_name: ModelName,
    min_train_years: int,
    test_years: int,
    entry_threshold: float,
    exit_threshold: float | None,
    cost_bps: float,
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    baseline = _run_ablation_case(
        close,
        features,
        model_name,
        min_train_years,
        test_years,
        entry_threshold,
        exit_threshold,
        cost_bps,
    )
    rows.append({"model": model_name, "dropped_feature": "(baseline)", **baseline})

    for dropped in features:
        selected = tuple(feature for feature in features if feature != dropped)
        metrics = _run_ablation_case(
            close,
            selected,
            model_name,
            min_train_years,
            test_years,
            entry_threshold,
            exit_threshold,
            cost_bps,
        )
        rows.append(
            {
                "model": model_name,
                "dropped_feature": dropped,
                **metrics,
                "delta_roc_auc": metrics["roc_auc"] - baseline["roc_auc"],
                "delta_sharpe": metrics["sharpe"] - baseline["sharpe"],
            }
        )
    return rows


def _run_ablation_case(
    close: pd.Series,
    feature_columns: tuple[str, ...],
    model_name: ModelName,
    min_train_years: int,
    test_years: int,
    entry_threshold: float,
    exit_threshold: float | None,
    cost_bps: float,
) -> dict[str, float]:
    result = run_spy_walk_forward(
        close,
        model_factory=get_model_factory(model_name),
        min_train_years=min_train_years,
        test_years=test_years,
        threshold=entry_threshold,
        exit_threshold=exit_threshold,
        cost_bps=cost_bps,
        feature_columns=feature_columns,
        out_dir=None,
    )
    row = result.metrics.loc["Walk-forward strategy"]
    return {
        "accuracy": float(row["Accuracy"]),
        "roc_auc": float(row["ROC AUC"]),
        "cagr": float(row["CAGR"]),
        "sharpe": float(row["Sharpe"]),
        "max_drawdown": float(row["Maximum drawdown"]),
        "annualized_turnover": float(row["Annualized turnover"]),
    }


def _positive_probability(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities)
    return values[:, 1] if values.ndim == 2 else values


def _roc_auc(target: pd.Series, probability: np.ndarray) -> float:
    positive = target.eq(1).to_numpy()
    n_positive = int(positive.sum())
    n_negative = len(target) - n_positive
    if not n_positive or not n_negative:
        return float("nan")
    ranks = pd.Series(probability).rank(method="average").to_numpy()
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    )


def _write_analysis(result: FeatureAnalysisResult, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    result.permutation_importance.to_csv(
        out_dir / "permutation_importance.csv",
        index=False,
    )
    result.ablation.to_csv(out_dir / "feature_ablation.csv", index=False)

    importance = result.permutation_importance.pivot(
        index="feature",
        columns="model",
        values="importance_mean",
    ).sort_values(by=list(result.permutation_importance["model"].unique()))
    uncertainty = (
        result.permutation_importance.pivot(
            index="feature",
            columns="model",
            values="importance_std",
        )
        .reindex(index=importance.index, columns=importance.columns)
        .fillna(0.0)
    )
    axis = importance.plot.barh(
        figsize=(10, 7),
        xerr=uncertainty,
        capsize=2,
    )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_title("Out-of-sample permutation importance")
    axis.set_xlabel("Mean decrease in ROC AUC")
    axis.grid(axis="x", alpha=0.3)
    axis.figure.tight_layout()
    axis.figure.savefig(out_dir / "permutation_importance.png", dpi=140)
    plt.close(axis.figure)

    ablation = result.ablation[result.ablation["dropped_feature"] != "(baseline)"]
    delta = ablation.pivot(
        index="dropped_feature",
        columns="model",
        values="delta_roc_auc",
    )
    axis = delta.plot.barh(figsize=(10, 7))
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_title("Leave-one-feature-out ablation")
    axis.set_xlabel("Change in out-of-sample ROC AUC after dropping feature")
    axis.grid(axis="x", alpha=0.3)
    axis.figure.tight_layout()
    axis.figure.savefig(out_dir / "feature_ablation.png", dpi=140)
    plt.close(axis.figure)
