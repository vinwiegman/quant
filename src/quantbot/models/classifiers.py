"""Small, reproducible classifier definitions for fair model comparison."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from quantbot.validation.walk_forward import ProbabilisticClassifier

ModelName = Literal["logistic", "gradient-boosting"]
MODEL_NAMES: tuple[ModelName, ...] = ("logistic", "gradient-boosting")


def get_model_factory(name: ModelName) -> Callable[[], ProbabilisticClassifier]:
    """Return a fresh estimator factory for a supported model name."""
    if name == "logistic":
        return _logistic_factory
    if name == "gradient-boosting":
        return _gradient_boosting_factory
    raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}")


def _logistic_factory() -> ProbabilisticClassifier:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1_000, random_state=42),
    )


def _gradient_boosting_factory() -> ProbabilisticClassifier:
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        random_state=42,
    )
