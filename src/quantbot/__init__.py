"""quantbot -- signal research, backtesting and paper trading."""

from .backtest.engine import BacktestResult, run_backtest
from .signals.base import Signal
from .signals.momentum import CrossSectionalMomentum

__version__ = "0.1.0"

__all__ = [
    "BacktestResult",
    "CrossSectionalMomentum",
    "Signal",
    "run_backtest",
]
