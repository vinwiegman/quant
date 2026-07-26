"""Command line entry point.

quantbot backtest --tickers AAPL,MSFT,NVDA,JPM,XOM,PG --start 2018-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .analysis import run_feature_analysis
from .backtest.engine import run_backtest
from .data.loader import load_prices
from .models import MODEL_NAMES, get_model_factory
from .signals.momentum import CrossSectionalMomentum
from .validation import run_model_comparison, run_spy_walk_forward

DEFAULT_TICKERS = "AAPL,MSFT,NVDA,AMZN,JPM,XOM,PG,JNJ,KO,CAT"


def _backtest(args: argparse.Namespace) -> int:
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    prices = load_prices(tickers, start=args.start, end=args.end)
    print(f"Loaded {len(prices)} bars for {len(prices.columns)} instruments.\n")

    signal = CrossSectionalMomentum(
        lookback=args.lookback,
        skip=args.skip,
        n_long=args.n_long,
        n_short=args.n_short,
        rebalance_days=args.rebalance,
    )
    weights = signal.generate(prices)
    result = run_backtest(prices, weights, cost_bps=args.cost_bps)

    # Equal-weight buy and hold, the honest thing to compare against.
    benchmark = prices.pct_change().fillna(0.0).mean(axis=1)

    table = result.summary(benchmark=benchmark)
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))
    print(f"\nFinal equity: {result.equity.iloc[-1]:,.0f}")
    print(f"Average annual turnover: {result.turnover.sum() / (len(prices) / 252):,.1f}x")

    if args.out:
        _write_report(result, benchmark, Path(args.out))
        print(f"\nReport written to {args.out}")
    return 0


def _write_report(result, benchmark: pd.Series, out_dir: Path) -> None:
    """Save the equity curve, drawdown chart and metrics table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    bench_equity = (1.0 + benchmark).cumprod() * result.equity.iloc[0]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    top.plot(result.equity, label="Strategy", linewidth=1.4)
    top.plot(bench_equity, label="Equal-weight buy & hold", linewidth=1.2, alpha=0.7)
    top.set_ylabel("Portfolio value")
    top.set_title("Cross-sectional momentum vs buy & hold")
    top.legend()
    top.grid(alpha=0.3)

    bottom.fill_between(result.returns.index, result.drawdown(), 0, alpha=0.4, color="crimson")
    bottom.set_ylabel("Drawdown")
    bottom.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "performance.png", dpi=140)
    plt.close(fig)

    result.summary(benchmark=benchmark).to_csv(out_dir / "metrics.csv")
    result.equity.to_csv(out_dir / "equity_curve.csv", header=["equity"])


def _walk_forward(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    prices = load_prices(["SPY"], start=args.start, end=args.end, cache_dir=cache_dir)
    result = run_spy_walk_forward(
        prices["SPY"],
        model_factory=get_model_factory(args.model),
        min_train_years=args.min_train_years,
        test_years=args.test_years,
        threshold=args.threshold,
        exit_threshold=args.exit_threshold,
        cost_bps=args.cost_bps,
        out_dir=args.out,
    )
    print(result.metrics.to_string(float_format=lambda value: f"{value:,.4f}"))
    print(f"\nResults written to {args.out}")
    return 0


def _compare_models(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    prices = load_prices(["SPY"], start=args.start, end=args.end, cache_dir=cache_dir)
    result = run_model_comparison(
        prices["SPY"],
        min_train_years=args.min_train_years,
        test_years=args.test_years,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        cost_bps=args.cost_bps,
        out_dir=args.out,
    )
    print(result.metrics.to_string(float_format=lambda value: f"{value:,.4f}"))
    print(f"\nComparison written to {args.out}")
    return 0


def _analyze_features(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    prices = load_prices(["SPY"], start=args.start, end=args.end, cache_dir=cache_dir)
    models = MODEL_NAMES if args.model == "all" else (args.model,)
    result = run_feature_analysis(
        prices["SPY"],
        models=models,
        min_train_years=args.min_train_years,
        test_years=args.test_years,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        cost_bps=args.cost_bps,
        n_repeats=args.repeats,
        random_state=args.random_state,
        out_dir=args.out,
    )
    print("\nPermutation importance")
    print(
        result.permutation_importance.to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}",
        )
    )
    print("\nFeature ablation")
    print(
        result.ablation.to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}",
        )
    )
    print(f"\nFeature analysis written to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantbot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="run a historical backtest")
    bt.add_argument("--tickers", default=DEFAULT_TICKERS, help="comma separated symbols")
    bt.add_argument("--start", default="2015-01-01")
    bt.add_argument("--end", default="2024-12-31")
    bt.add_argument("--lookback", type=int, default=126, help="momentum window in days")
    bt.add_argument("--skip", type=int, default=21, help="recent days to skip")
    bt.add_argument("--n-long", type=int, default=3)
    bt.add_argument("--n-short", type=int, default=3)
    bt.add_argument("--rebalance", type=int, default=21, help="rebalance every N days")
    bt.add_argument("--cost-bps", type=float, default=5.0)
    bt.add_argument("--out", default=None, help="directory for charts and CSVs")
    bt.set_defaults(func=_backtest)

    wf = sub.add_parser("walk-forward", help="run a leakage-resistant SPY model evaluation")
    wf.add_argument("--start", default="2010-01-01")
    wf.add_argument("--end", default="2024-12-31")
    wf.add_argument("--min-train-years", type=int, default=5)
    wf.add_argument("--test-years", type=int, default=1)
    wf.add_argument("--model", choices=MODEL_NAMES, default="logistic")
    wf.add_argument("--threshold", type=float, default=0.55)
    wf.add_argument(
        "--exit-threshold",
        type=float,
        default=None,
        help="optional lower threshold that enables long/cash hysteresis",
    )
    wf.add_argument("--cost-bps", type=float, default=5.0)
    wf.add_argument("--cache-dir", default=None, help="optional price-cache directory")
    wf.add_argument("--out", default="results")
    wf.set_defaults(func=_walk_forward)

    compare = sub.add_parser(
        "compare-models",
        help="compare logistic and gradient boosting on identical SPY folds",
    )
    compare.add_argument("--start", default="2010-01-01")
    compare.add_argument("--end", default="2024-12-31")
    compare.add_argument("--min-train-years", type=int, default=5)
    compare.add_argument("--test-years", type=int, default=1)
    compare.add_argument("--entry-threshold", type=float, default=0.55)
    compare.add_argument("--exit-threshold", type=float, default=0.45)
    compare.add_argument("--cost-bps", type=float, default=5.0)
    compare.add_argument("--cache-dir", default=None, help="optional price-cache directory")
    compare.add_argument("--out", default="results")
    compare.set_defaults(func=_compare_models)

    analysis = sub.add_parser(
        "analyze-features",
        help="run fold-safe permutation importance and feature ablation",
    )
    analysis.add_argument("--start", default="2010-01-01")
    analysis.add_argument("--end", default="2024-12-31")
    analysis.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    analysis.add_argument("--min-train-years", type=int, default=5)
    analysis.add_argument("--test-years", type=int, default=1)
    analysis.add_argument("--entry-threshold", type=float, default=0.55)
    analysis.add_argument("--exit-threshold", type=float, default=0.45)
    analysis.add_argument("--cost-bps", type=float, default=5.0)
    analysis.add_argument("--repeats", type=int, default=5)
    analysis.add_argument("--random-state", type=int, default=42)
    analysis.add_argument("--cache-dir", default=None, help="optional price-cache directory")
    analysis.add_argument("--out", default="results")
    analysis.set_defaults(func=_analyze_features)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
