# SPY daily direction model card

## Intended use

This experiment predicts next-session SPY direction from daily price and volume features.
We use it for research and Alpaca paper trading; live-capital trading is outside the scope of
this project.

## Validation design

- Expanding outer walk-forward folds with all training dates before every test date.
- The ML/momentum blend is selected using only inner walk-forward predictions from the outer
  training fold; the outer test fold is never used for tuning.
- Logistic, ensemble, 50-day moving average, and buy-and-hold share evaluation dates.
- Headline costs are 5 bps, with a 0--50 bps sensitivity sweep.
- Newey--West uncertainty accounts for autocorrelation in the mean; a 20-session moving-block
  bootstrap estimates the Sharpe interval.

## Headline out-of-sample results

| Portfolio | Sharpe | CAGR | Maximum drawdown | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic | 0.721 | 11.5% | -33.7% | 0.494 |
| Nested ML + momentum | 0.717 | 11.4% | -33.7% | 0.501 |
| SPY buy and hold | 0.789 | 13.2% | -33.7% | n/a |
| SPY above 50-day MA | 0.524 | 5.1% | -21.1% | n/a |

The highest Sharpe in this run is **SPY buy and hold**. The ensemble's 95% block-bootstrap Sharpe
interval is **[0.143, 1.431]**.

## Conclusion

The ensemble does not beat buy-and-hold on out-of-sample Sharpe in this test. We keep logistic regression as the simpler paper-trading baseline, not as a claim that it beats the market.

## Known limitations

- One liquid ETF and one historical regime do not establish generalization.
- Flat transaction costs omit spread variation, market impact, tax, and slippage shocks.
- Multiple strategy variants create selection bias; the reported uncertainty does not fully
  correct for every experiment attempted by the team.
- Paper fills do not reproduce real queue position, slippage, or operational failures.
