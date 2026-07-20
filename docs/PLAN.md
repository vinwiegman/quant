# Project plan

Six weeks, three people, one repo. The goal is a portfolio piece: something a hiring manager can clone, run in two minutes, and immediately see was built by people who know what they are doing.

## Why the roles split this way

The `Signal` ABC in `signals/base.py` is the contract between all three workstreams. It is already written, so nobody is blocked on anybody:

- **Vin** owns everything that consumes a `Signal` — the engine, the broker, the CLI, CI.
- **Teammate A** owns everything that *is* a `Signal` — features, models, weight generation.
- **Teammate B** owns everything that *judges* a `Signal` — metrics, statistical tests, the writeup.

If a workstream starts needing changes to that interface, that is the signal to stop and talk, not to widen it quietly.

---

## Vin — Information Science BSc — Platform & infrastructure

You own the thing that makes this look like engineering rather than a notebook. This is also the part that maps most directly onto job descriptions.

| Week | Deliverable |
| ---- | ----------- |
| 1 | ✅ Repo, package layout, `Signal`/`Broker` interfaces, vectorised engine, 33 tests, CI on 3 Python versions |
| 2 | Walk-forward harness: split history into train/test folds so A's models are evaluated out of sample. This is the single highest-value thing you can build for the team. |
| 3 | Live data feed + scheduler. A `run_live` command that pulls today's bars, asks the signal for weights, diffs against the paper account, submits orders. |
| 4 | Alpaca paper broker implementing the existing `Broker` ABC. The point is that only the constructor changes between backtest and paper. |
| 5 | Persistence + observability: SQLite trade log, daily equity snapshot, a `report` command that regenerates all charts. |
| 6 | Docker image, README polish, coverage badge, a two-minute demo GIF at the top of the README |

**Guard the interfaces.** You will be tempted to let a model leak `sklearn` types into the engine, or to let the paper trader diverge from the backtester "just for now". Both destroy the credibility of the whole thing. Every strategy goes through `Signal.generate` and every fill goes through `Broker.submit`.

## Teammate A — AI BSc — Signals & modelling

You own alpha. Everything you build subclasses `Signal` and returns a weights DataFrame; the rest of the system does not care how you got there.

| Week | Deliverable |
| ---- | ----------- |
| 1 | Read `signals/momentum.py` end to end. It is your template. Reproduce its results locally. |
| 2 | Feature library: returns over multiple horizons, realised vol, volume z-scores, moving-average distance, RSI. All strictly point-in-time. |
| 3 | `MLSignal`: predict next-period return ranking with logistic regression, then gradient boosting. Convert predictions to weights by ranking, same as momentum does. |
| 4 | Feature importance and ablation. Which features survive? Which are proxies for "the market went up"? |
| 5 | Ensemble the ML signal with momentum, tune on train folds only |
| 6 | Model card: what it predicts, what it was trained on, where it fails |

**The trap that will get you.** Every lookahead bug feels like a breakthrough. If your Sharpe jumps above ~2 on daily equities, assume a bug before assuming skill — check whether a feature used a future bar, whether you fit the scaler on the full sample, and whether your labels are shifted correctly. Run `pytest` before you believe any number.

## Teammate B — Econometrics student — Validation & analysis

You own the question "is this real?". Your work is what stops the project from being a plot that goes up, and it is genuinely the most interview-friendly part to be able to talk about.

Start here because it is scoped to run before you need heavy theory:

| Week | Deliverable |
| ---- | ----------- |
| 1 | Read `backtest/metrics.py` and `tests/test_metrics.py`. Verify Sharpe and max drawdown by hand in a spreadsheet against the test cases. Now you know the codebase's arithmetic is correct because *you* checked it. |
| 2 | Add `information_ratio`, `beta`, `alpha` vs the benchmark, and `tail_ratio`. Same file, same test style — copy the pattern. |
| 3 | Cost sensitivity: sweep `cost_bps` from 0 to 50 and plot Sharpe against it. Find the cost level where the edge dies. This single chart is often the most revealing thing in a backtest. |
| 4 | Significance testing: is the Sharpe distinguishable from zero? Newey-West standard errors on the mean return (returns are autocorrelated, so plain OLS errors lie). |
| 5 | Deflated Sharpe ratio — adjust for how many strategy variants the team tried. Directly quantifies the overfitting the other two are at risk of. |
| 6 | Write `docs/REPORT.md`: what was tested, what worked, what did not, and what you would need to trust it with real money |

**Ask for help early and often.** Newey-West and the deflated Sharpe are graduate-level topics and you are not expected to derive them. Understanding *why* they are needed — autocorrelated returns, multiple-testing bias — is the part that matters, and it is the part you will be asked about.

---

## Working agreement

- **Branches and PRs, always.** Never push to `main`. Every PR gets one review. This is partly discipline and partly that your commit graph is public and visible to employers.
- **CI must be green to merge.** No exceptions, including for "it's just a plot".
- **Record every strategy variant you test** in a shared sheet. B needs the count for the deflated Sharpe, and it keeps everyone honest about how much searching happened.
- **Weekly 30-minute sync.** Each person demos something that runs. Not slides.

## Definition of done

The repo is finished when someone can clone it, run `pip install -e .` and `quantbot backtest`, and get a chart — with a README that states plainly what works, what does not, and why. A negative result presented rigorously is a stronger portfolio piece than a positive result presented credulously, and it is much harder to fake.
