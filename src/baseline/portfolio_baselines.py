"""Markowitz and buy-and-hold benchmarks compatible with the RL backtest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

COMMISSION_RATE = 0.001
INITIAL_BALANCE = 100_000.0


def load_price_data(path: str | Path) -> pd.DataFrame:
    """Load the RL module's wide price CSV (Date index, one column per asset)."""
    prices = pd.read_csv(path, index_col="Date", parse_dates=True)
    if prices.empty or prices.columns.empty:
        raise ValueError(f"Fiyat verisi boş: {path}")
    if prices.index.has_duplicates:
        raise ValueError(f"Tarih indeksinde tekrar var: {path}")
    prices = prices.sort_index().apply(pd.to_numeric, errors="raise")
    if prices.isna().any().any() or not np.isfinite(prices.to_numpy()).all():
        raise ValueError(f"Fiyat verisinde eksik veya sonlu olmayan değer var: {path}")
    if (prices <= 0).any().any():
        raise ValueError(f"Fiyatlar pozitif olmalıdır: {path}")
    return prices


def markowitz_weights(train_prices: pd.DataFrame) -> np.ndarray:
    """Find long-only maximum-Sharpe weights using training data only."""
    returns = train_prices.pct_change().dropna()
    n_assets = train_prices.shape[1]
    if len(returns) < 2:
        raise ValueError("Markowitz için eğitim verisinde en az iki getiri gözlemi gerekir.")
    mean_returns = returns.mean().to_numpy(dtype=float) * 252
    covariance = returns.cov().to_numpy(dtype=float) * 252
    covariance = (covariance + covariance.T) / 2

    def negative_sharpe(weights: np.ndarray) -> float:
        variance = float(weights @ covariance @ weights)
        if variance <= 1e-16:
            return 1e6
        return -float(weights @ mean_returns) / np.sqrt(variance)

    result = minimize(
        negative_sharpe,
        x0=np.full(n_assets, 1 / n_assets),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.x).all():
        # Degenerate covariance estimates can prevent the optimizer converging.
        weights = np.full(n_assets, 1 / n_assets)
    else:
        weights = np.clip(result.x, 0.0, 1.0)
        weights /= weights.sum()
    return weights


def _backtest(
    test_prices: pd.DataFrame,
    target_weights: np.ndarray,
    *,
    rebalance_daily: bool,
    initial_balance: float,
    commission_rate: float,
    window_size: int,
) -> pd.DataFrame:
    """Simulate long-only holdings and emit the RL evaluate.py result schema."""
    if len(test_prices) <= window_size + 1:
        raise ValueError("Test verisi, gözlem penceresi ve en az bir getiri günü içermelidir.")
    n_assets = test_prices.shape[1]
    if target_weights.shape != (n_assets,):
        raise ValueError("Ağırlık sayısı ile test fiyatlarındaki hisse sayısı uyuşmuyor.")
    if not np.isclose(target_weights.sum(), 1.0) or (target_weights < 0).any():
        raise ValueError("Hedef ağırlıklar negatif olmamalı ve toplamları 1 olmalıdır.")

    values = float(initial_balance)
    holdings = np.zeros(n_assets + 1, dtype=float)
    holdings[-1] = 1.0  # RL ortamı gibi başlangıçta tüm sermaye nakitte.
    targets = np.append(target_weights, 0.0)
    records: list[dict[str, float | str]] = []
    price_values = test_prices.to_numpy(dtype=float)
    dates = test_prices.index

    # RL starts trading after observing its initial rolling window.
    for i in range(window_size, len(test_prices) - 1):
        desired = targets if (rebalance_daily or i == window_size) else holdings
        turnover = float(np.abs(desired - holdings).sum())
        values *= 1.0 - commission_rate * turnover

        daily_asset_returns = price_values[i + 1] / price_values[i] - 1.0
        all_returns = np.append(daily_asset_returns, 0.0)
        daily_return = float(desired @ all_returns)
        values *= 1.0 + daily_return

        after_return_values = desired * (1.0 + all_returns)
        holdings = after_return_values / after_return_values.sum()
        row: dict[str, float | str] = {
            "Date": dates[i + 1].strftime("%Y-%m-%d"),
            "Portfolio_Value": round(values, 2),
            "Daily_Return": round(daily_return, 6),
        }
        for asset, weight in zip(test_prices.columns, holdings[:-1]):
            row[f"W_{asset}"] = round(float(weight), 4)
        row["W_CASH"] = round(float(holdings[-1]), 4)
        records.append(row)

    return pd.DataFrame(records)


def run_baselines(
    train_prices_path: str | Path = "data/processed/train_prices.csv",
    test_prices_path: str | Path = "data/processed/test_prices.csv",
    output_dir: str | Path = "data/results",
    initial_balance: float = INITIAL_BALANCE,
    commission_rate: float = COMMISSION_RATE,
    window_size: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both benchmarks and save CSVs shaped like ``rl_backtest_results.csv``.

    Markowitz weights are estimated exclusively on the chronological training
    prices and rebalanced to those targets daily during backtesting. Buy-and-hold
    starts with equal asset weights and holds them without further rebalancing.
    Both strategies use the shared 0.1% commission convention.
    """
    train_prices = load_price_data(train_prices_path)
    test_prices = load_price_data(test_prices_path)
    if list(train_prices.columns) != list(test_prices.columns):
        raise ValueError("Eğitim ve test fiyat dosyalarındaki hisse sırası aynı olmalıdır.")
    if train_prices.index.max() >= test_prices.index.min():
        raise ValueError("Train dönemi test döneminden önce tamamlanmalıdır.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights = markowitz_weights(train_prices)
    equal_weights = np.full(test_prices.shape[1], 1 / test_prices.shape[1])

    markowitz = _backtest(
        test_prices, weights, rebalance_daily=True,
        initial_balance=initial_balance, commission_rate=commission_rate, window_size=window_size,
    )
    buy_and_hold = _backtest(
        test_prices, equal_weights, rebalance_daily=False,
        initial_balance=initial_balance, commission_rate=commission_rate, window_size=window_size,
    )
    markowitz.to_csv(output / "markowitz_backtest_results.csv", index=False)
    buy_and_hold.to_csv(output / "buy_and_hold_backtest_results.csv", index=False)
    return markowitz, buy_and_hold


if __name__ == "__main__":
    run_baselines()




