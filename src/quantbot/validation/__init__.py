"""Leakage-resistant model validation utilities."""

from .comparison import ModelComparisonResult, run_model_comparison
from .robustness import (
    NestedMomentumEnsemble,
    RobustnessResult,
    block_bootstrap_sharpe,
    newey_west_mean,
    run_robustness_analysis,
)
from .walk_forward import (
    WalkForwardResult,
    build_spy_dataset,
    chronological_folds,
    probabilities_to_positions,
    run_spy_walk_forward,
    walk_forward_predict,
)

__all__ = [
    "WalkForwardResult",
    "build_spy_dataset",
    "ModelComparisonResult",
    "NestedMomentumEnsemble",
    "RobustnessResult",
    "block_bootstrap_sharpe",
    "chronological_folds",
    "probabilities_to_positions",
    "newey_west_mean",
    "run_model_comparison",
    "run_robustness_analysis",
    "run_spy_walk_forward",
    "walk_forward_predict",
]
