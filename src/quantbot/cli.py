"""Command line entry point.

quantbot backtest --tickers AAPL,MSFT,NVDA,JPM,XOM,PG --start 2018-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .analysis import run_feature_analysis
from .backtest.engine import run_backtest
from .broker import AlpacaPaperBroker
from .data.loader import load_ohlcv, load_prices
from .execution import run_daily_execution
from .models import MODEL_NAMES, get_model_factory
from .persistence import ExecutionStore
from .reporting import write_paper_report
from .signals.momentum import CrossSectionalMomentum
from .validation import run_model_comparison, run_robustness_analysis, run_spy_walk_forward

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
    market = load_ohlcv("SPY", start=args.start, end=args.end, cache_dir=cache_dir)
    result = run_spy_walk_forward(
        market["close"],
        model_factory=get_model_factory(args.model),
        min_train_years=args.min_train_years,
        test_years=args.test_years,
        threshold=args.threshold,
        exit_threshold=args.exit_threshold,
        cost_bps=args.cost_bps,
        volume=market["volume"],
        out_dir=args.out,
    )
    print(result.metrics.to_string(float_format=lambda value: f"{value:,.4f}"))
    print(f"\nResults written to {args.out}")
    return 0


def _compare_models(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    market = load_ohlcv("SPY", start=args.start, end=args.end, cache_dir=cache_dir)
    result = run_model_comparison(
        market["close"],
        min_train_years=args.min_train_years,
        test_years=args.test_years,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        cost_bps=args.cost_bps,
        volume=market["volume"],
        out_dir=args.out,
    )
    print(result.metrics.to_string(float_format=lambda value: f"{value:,.4f}"))
    print(f"\nComparison written to {args.out}")
    return 0


def _analyze_features(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    market = load_ohlcv("SPY", start=args.start, end=args.end, cache_dir=cache_dir)
    models = MODEL_NAMES if args.model == "all" else (args.model,)
    result = run_feature_analysis(
        market["close"],
        models=models,
        min_train_years=args.min_train_years,
        test_years=args.test_years,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        cost_bps=args.cost_bps,
        n_repeats=args.repeats,
        random_state=args.random_state,
        volume=market["volume"],
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


def _robustness(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    market = load_ohlcv("SPY", start=args.start, end=args.end, cache_dir=cache_dir)
    result = run_robustness_analysis(
        market["close"],
        volume=market["volume"],
        min_train_years=args.min_train_years,
        test_years=args.test_years,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        cost_bps=args.cost_bps,
        n_bootstrap=args.bootstrap_samples,
        random_state=args.random_state,
        out_dir=args.out,
    )
    print("\nCommon-date comparison")
    print(result.comparison.to_string(float_format=lambda value: f"{value:,.4f}"))
    print("\nUncertainty estimates")
    print(result.uncertainty.to_string(float_format=lambda value: f"{value:,.4f}"))
    print(f"\nRobustness report and model card written to {args.out}")
    return 0


def _load_dotenv() -> None:
    """Load Alpaca credentials from a local .env if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _trade(args: argparse.Namespace) -> int:
    _load_dotenv()
    end = args.end or (date.today() + timedelta(days=1)).isoformat()
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    market = load_ohlcv(
        "SPY",
        start=args.start,
        end=end,
        cache_dir=cache_dir,
        force_refresh=args.force_refresh,
    )
    broker = AlpacaPaperBroker()
    result = run_daily_execution(
        market,
        broker,
        model=args.model,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        invested_weight=args.invested_weight,
        min_trade_value=args.min_trade_value,
        submit=args.submit,
        log_path=args.log,
        database_path=args.database,
    )
    print(json.dumps(asdict(result), indent=2))
    if not args.submit:
        print("\nDRY RUN: no order was submitted. Add --submit for Alpaca paper execution.")
    return 0


def _paper_report(args: argparse.Namespace) -> int:
    store = ExecutionStore(args.database)
    summary = write_paper_report(
        store,
        args.out,
        stale_after_hours=args.stale_after_hours,
    )
    print(json.dumps(summary, indent=2))
    print(f"\nPaper history report written to {args.out}")
    if args.fail_on_unhealthy and summary["status"] != "healthy":
        print(f"Paper-history health check failed: {summary['status']}", file=sys.stderr)
        return 1
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

    robustness = sub.add_parser(
        "robustness",
        help="run nested ensemble, cost sensitivity, and uncertainty analysis",
    )
    robustness.add_argument("--start", default="2010-01-01")
    robustness.add_argument("--end", default="2024-12-31")
    robustness.add_argument("--min-train-years", type=int, default=5)
    robustness.add_argument("--test-years", type=int, default=1)
    robustness.add_argument("--entry-threshold", type=float, default=0.55)
    robustness.add_argument("--exit-threshold", type=float, default=0.45)
    robustness.add_argument("--cost-bps", type=float, default=5.0)
    robustness.add_argument("--bootstrap-samples", type=int, default=1_000)
    robustness.add_argument("--random-state", type=int, default=42)
    robustness.add_argument("--cache-dir", default=None, help="optional price-cache directory")
    robustness.add_argument("--out", default="results")
    robustness.set_defaults(func=_robustness)

    trade = sub.add_parser(
        "trade",
        help="generate and optionally submit one SPY Alpaca paper decision",
    )
    trade.add_argument("--start", default="2010-01-01")
    trade.add_argument("--end", default=None, help="exclusive data end date; defaults to tomorrow")
    trade.add_argument("--model", choices=MODEL_NAMES, default="logistic")
    trade.add_argument("--entry-threshold", type=float, default=0.55)
    trade.add_argument("--exit-threshold", type=float, default=0.45)
    trade.add_argument("--invested-weight", type=float, default=0.95)
    trade.add_argument("--min-trade-value", type=float, default=25.0)
    trade.add_argument("--cache-dir", default=None)
    trade.add_argument("--force-refresh", action="store_true")
    trade.add_argument("--log", default="logs/executions.jsonl")
    trade.add_argument("--database", default="state/paper_trading.sqlite3")
    trade.add_argument(
        "--submit",
        action="store_true",
        help="submit to Alpaca paper trading; omitted means read-only dry run",
    )
    trade.set_defaults(func=_trade)

    paper_report = sub.add_parser(
        "paper-report",
        help="generate an offline monitoring report from durable paper history",
    )
    paper_report.add_argument("--database", default="state/paper_trading.sqlite3")
    paper_report.add_argument("--out", default="results/paper")
    paper_report.add_argument("--stale-after-hours", type=float, default=48.0)
    paper_report.add_argument("--fail-on-unhealthy", action="store_true")
    paper_report.set_defaults(func=_paper_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
