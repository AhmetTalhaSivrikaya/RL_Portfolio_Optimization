# Baseline strategies

Run from the repository root after `data/processed/train_prices.csv` and
`data/processed/test_prices.csv` are available:

```powershell
python -m src.baseline.portfolio_baselines
```

The Markowitz maximum-Sharpe weights are fitted only on training prices, then
rebalanced daily on the test period. Buy-and-hold starts with equal weights and
does not rebalance. Both apply a 0.1% commission to traded turnover and write
CSV files under `data/results/` using the RL evaluator's columns:
`Date`, `Portfolio_Value`, `Daily_Return`, `W_<ticker>`, and `W_CASH`.
