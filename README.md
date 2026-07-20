# quantbot

A systematic trading research platform: research a signal, backtest it without lying to yourself, then run it against a paper account.

The point of this repo is not to get rich. It is to do the boring things correctly — no lookahead, honest transaction costs, a benchmark you actually have to beat — because that is what separates a real backtest from a plot that goes up.

## Results

Cross-sectional momentum on 10 large-cap US equities, 2015–2024, 5 bps costs, monthly rebalance:

|              | Strategy | Buy & hold |
| ------------ | -------: | ---------: |
| CAGR         |    5.2%  |     24.0%  |
| Volatility   |   12.9%  |     18.6%  |
| Sharpe       |    0.46  |      1.25  |
| Sortino      |    0.64  |      1.56  |
| Max drawdown |  -25.0%  |    -31.1%  |
| Calmar       |    0.21  |      0.77  |
| Hit rate     |   51.8%  |     56.4%  |

![Performance](docs/performance.png)

**The strategy loses to buy and hold, and that is the honest result.** Long/short momentum on ten mega-caps over the strongest equity decade in living memory was never likely to win: the short leg fights a market that went up almost every year, and ten names give the cross-sectional ranking almost nothing to sort.

It is reported as-is on purpose. Tuning the lookback, the universe and the date range until the blue line wins is trivial, takes about ten minutes, and produces a number that means nothing out of sample. The infrastructure is the deliverable here; the signal is the thing the infrastructure is built to evaluate honestly, and this is it doing its job.

Two details worth reading off the chart: risk-adjusted returns are the only fair comparison (the strategy runs at 13% vol against 19%), and the drawdown panel shows the strategy's losses were shallower but far more persistent — it spent 2021–2023 grinding sideways rather than recovering.

## Install

```bash
git clone https://github.com/<you>/quantbot.git
cd quantbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Use

```bash
quantbot backtest --tickers AAPL,MSFT,NVDA,JPM,XOM --start 2018-01-01 --out docs/
```

```python
from quantbot import CrossSectionalMomentum, run_backtest
from quantbot.data import load_prices

prices = load_prices(["AAPL", "MSFT", "NVDA", "JPM", "XOM"], start="2018-01-01")
weights = CrossSectionalMomentum(lookback=126, n_long=2, n_short=2).generate(prices)
result = run_backtest(prices, weights, cost_bps=5.0)

print(result.summary())
```

## Design

```
data/     price loading, cached to Parquet so runs are reproducible
signals/  Signal ABC -> target portfolio weights. Add strategies here.
backtest/ engine (vectorised, cost-aware) + metrics
broker/   Broker ABC, paper implementation, weight -> order translation
```

Three decisions carry most of the weight:

**Signals return weights, not orders.** A signal says "I want to be 30% long NVDA", not "buy 12 shares". Position sizing and order generation live in the broker layer, so the same strategy object drives both the backtester and the paper account. If those two paths diverge, the thing you tested is not the thing you run.

**Lookahead is prevented in exactly one line.** `run_backtest` does `weights.shift(1)` and nothing else in the codebase is allowed to shift anything. A signal computed at the close of day *t* earns the return of day *t+1*. Concentrating this in one auditable place is why `test_signal_cannot_see_the_same_day_return` is a meaningful test rather than a formality.

**Costs are charged on turnover, not on trades.** Cost is `Σ|Δw| × bps`, which makes rebalance frequency show up directly in the P&L. A strategy that only works at zero cost is visible immediately instead of after you have funded an account.

## Testing

```bash
pytest --cov=quantbot
ruff check .
```

The suite covers the arithmetic (metrics against hand-computed values), the invariants (drawdown is never positive, costs always reduce returns), and the one property that matters most — that a perfect-foresight signal cannot profit from the day it was set.

## Limitations

Known and deliberate, listed because a backtest that does not state these is hiding them:

- **Survivorship bias.** Tickers are chosen today, so companies that failed are absent. Real research needs a point-in-time universe.
- **Costs are a flat estimate.** No market impact model, no borrow costs on shorts, no spread that widens in a crisis.
- **Daily close-to-close only.** No intraday execution, no gap risk, no partial fills.
- **Small universe.** Cross-sectional strategies are noisy on 10 names; the ranking has little to sort.

## License

MIT
