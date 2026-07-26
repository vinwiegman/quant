"""Leakage-resistant model validation utilities."""

from .comparison import ModelComparisonResult, run_model_comparison
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
    "chronological_folds",
    "probabilities_to_positions",
    "run_model_comparison",
    "run_spy_walk_forward",
    "walk_forward_predict",
]
