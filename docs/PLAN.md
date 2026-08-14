# Project plan

We started quantbot as a summer project after finishing our bachelor's degrees
in Information Science and AI. The aim was to build one complete project
together instead of another standalone notebook: market data in, a tested
strategy out, and a safe path to paper trading.

## How we split the work

We roughly split the work around our backgrounds, with some overlap when a
feature touched both research and execution.

### Platform and data

- Python package structure and command-line interface
- Market-data loading and Parquet caching
- Backtest and broker interfaces
- Alpaca paper-trading integration
- SQLite history, reports, Docker, and GitHub Actions

### Machine learning and validation

- Point-in-time features and next-day targets
- Logistic regression and gradient boosting
- Expanding and nested walk-forward evaluation
- Feature importance and ablation
- Momentum ensemble, cost sensitivity, and uncertainty estimates
- Model card and result interpretation

The `Signal` and `Broker` interfaces kept the two sides connected. Research
code produces weights; execution code turns those weights into orders. The
backtester and paper trader therefore use the same basic contracts.

## Milestones

| Milestone | Status |
| --- | --- |
| Package layout, backtest engine, signals, and tests | Complete |
| Leakage-safe feature and target pipeline | Complete |
| Walk-forward model comparison | Complete |
| Feature importance and ablation | Complete |
| Nested ML and momentum ensemble | Complete |
| Transaction-cost and uncertainty analysis | Complete |
| Alpaca paper execution with safety checks | Complete |
| Durable history and monitoring | Complete |
| Docker, CI, coverage badge, and demo | Complete |

## Working rules

- Work on branches and merge through pull requests.
- Keep CI green on Python 3.11, 3.12, and 3.13.
- Do not tune parameters on the final test fold.
- Store enough detail to reproduce each reported result.
- Keep live-capital trading out of scope; Alpaca integration is paper-only.
- Report results as they come out, including strategies that lose to the
  benchmark.

## Current status

The original build is complete. The repository can be installed locally or
with Docker, reproduces the research artifacts, and runs the same decision
pipeline against an Alpaca paper account. Future work would focus on new data
or a genuinely different research question rather than more tuning on the same
SPY sample.
