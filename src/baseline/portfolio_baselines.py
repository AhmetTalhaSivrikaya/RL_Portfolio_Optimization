"""Markowitz ve Al-ve-Tut kıyaslama stratejileri (RL backtest ile aynı şemada çıktı üretir).

Ortak kurallar (docs/ARAYUZ_SOZLESMESI.md):
  * Komisyon = portföy_değeri * turnover * commission_rate
  * turnover = Σ|Δw| yalnızca HİSSE bacakları (alış + satış). Nakit bacağı komisyonsuzdur.
      nakit -> hisse %100 = 1.0, hisse A -> B tam rotasyon = 2.0  (RL ortamı ile birebir aynı tanım)
  * Daily_Return NET getiridir (komisyon dahil): Portfolio_Value[t] / Portfolio_Value[t-1] - 1
  * Başlangıçta sermayenin tamamı nakittir; ilk işlem günü test setinin window_size'ıncı indeksidir
    (ilk sonuç satırı index[window_size + 1]).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

COMMISSION_RATE = 0.001
INITIAL_BALANCE = 100_000.0


def load_price_data(path: str | Path) -> pd.DataFrame:
    """RL modülünün geniş fiyat CSV'sini yükler (Date indeksi, her hisse bir sütun)."""
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


def trade_turnover(old_weights: np.ndarray, new_weights: np.ndarray) -> float:
    """İşlem hacmi / portföy değeri. Son eleman nakittir (komisyonsuz); yalnızca hisse bacakları sayılır."""
    old_w = np.asarray(old_weights, dtype=float)
    new_w = np.asarray(new_weights, dtype=float)
    return float(np.abs(new_w[:-1] - old_w[:-1]).sum())


def markowitz_weights(train_prices: pd.DataFrame) -> np.ndarray:
    """Yalnızca eğitim verisiyle long-only maksimum-Sharpe ağırlıklarını bulur."""
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
        print("⚠️ UYARI: Markowitz optimizasyonu yakınsamadı, eşit ağırlıklara dönüldü.")
        return np.full(n_assets, 1 / n_assets)
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def _backtest(
    test_prices: pd.DataFrame,
    target_weights: np.ndarray,
    *,
    rebalance_every: int,
    initial_balance: float,
    commission_rate: float,
    window_size: int,
) -> pd.DataFrame:
    """Long-only portföyü simüle eder ve RL evaluate.py ile aynı sonuç şemasını üretir.

    rebalance_every: 0 -> yalnızca ilk alım (al-ve-tut), 1 -> her gün, 5 -> haftalık (RL rebalance_freq=5 ile aynı takvim).
    """
    if len(test_prices) <= window_size + 1:
        raise ValueError("Test verisi, gözlem penceresi ve en az bir getiri günü içermelidir.")
    n_assets = test_prices.shape[1]
    if target_weights.shape != (n_assets,):
        raise ValueError("Ağırlık sayısı ile test fiyatlarındaki hisse sayısı uyuşmuyor.")
    if not np.isclose(target_weights.sum(), 1.0) or (target_weights < 0).any():
        raise ValueError("Hedef ağırlıklar negatif olmamalı ve toplamları 1 olmalıdır.")
    if rebalance_every < 0:
        raise ValueError("rebalance_every negatif olamaz.")

    value = float(initial_balance)
    holdings = np.zeros(n_assets + 1, dtype=float)
    holdings[-1] = 1.0  # RL ortamı gibi başlangıçta tüm sermaye nakitte.
    targets = np.append(target_weights, 0.0)
    records: list[dict[str, float | str]] = []
    price_values = test_prices.to_numpy(dtype=float)
    dates = test_prices.index

    for i in range(window_size, len(test_prices) - 1):
        k = i - window_size  # RL ortamındaki steps_elapsed ile aynı sayaç
        rebalance = (k == 0) or (rebalance_every > 0 and k % rebalance_every == 0)
        desired = targets if rebalance else holdings

        turnover = trade_turnover(holdings, desired)
        value_before = value
        commission = value_before * turnover * commission_rate
        value = value_before - commission

        asset_returns = price_values[i + 1] / price_values[i] - 1.0
        all_returns = np.append(asset_returns, 0.0)
        value *= 1.0 + float(desired @ all_returns)

        after = desired * (1.0 + all_returns)
        holdings = after / after.sum()

        row: dict[str, float | str] = {
            "Date": dates[i + 1].strftime("%Y-%m-%d"),
            "Portfolio_Value": round(value, 2),
            "Daily_Return": round(value / value_before - 1.0, 8),  # NET (komisyon dahil)
        }
        for asset, weight in zip(test_prices.columns, holdings[:-1]):
            row[f"W_{asset}"] = round(float(weight), 6)
        row["W_CASH"] = round(float(holdings[-1]), 6)
        row["Turnover"] = round(turnover, 6)
        row["Commission"] = round(commission, 4)
        records.append(row)

    return pd.DataFrame(records)


def run_baselines(
    train_prices_path: str | Path = "data/processed/train_prices.csv",
    test_prices_path: str | Path = "data/processed/test_prices.csv",
    output_dir: str | Path = "data/results",
    initial_balance: float = INITIAL_BALANCE,
    commission_rate: float = COMMISSION_RATE,
    window_size: int = 10,
    weekly_every: int = 5,
    include_weekly: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kıyaslamaları çalıştırır ve ``rl_backtest_results.csv`` şemasında CSV'ler yazar.

    Yazılan dosyalar:
      * markowitz_backtest_results.csv         -> eğitimden gelen sabit ağırlıklara HER GÜN dönüş
      * markowitz_weekly_backtest_results.csv  -> aynı ağırlıklar, ``weekly_every`` günde bir dönüş (RL ile adil kıyas)
      * buy_and_hold_backtest_results.csv      -> eşit ağırlıkla ilk gün al, sonra tut

    Markowitz ağırlıkları yalnızca kronolojik olarak önceki eğitim fiyatlarından tahmin edilir.
    Dönüş değeri geriye uyumluluk için (markowitz, buy_and_hold) ikilisidir.
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
    common = dict(initial_balance=initial_balance, commission_rate=commission_rate, window_size=window_size)

    markowitz = _backtest(test_prices, weights, rebalance_every=1, **common)
    buy_and_hold = _backtest(test_prices, equal_weights, rebalance_every=0, **common)
    markowitz.to_csv(output / "markowitz_backtest_results.csv", index=False)
    buy_and_hold.to_csv(output / "buy_and_hold_backtest_results.csv", index=False)
    if include_weekly:
        weekly = _backtest(test_prices, weights, rebalance_every=weekly_every, **common)
        weekly.to_csv(output / "markowitz_weekly_backtest_results.csv", index=False)
    return markowitz, buy_and_hold


def main() -> None:
    ap = argparse.ArgumentParser(description="Markowitz ve Al-ve-Tut kıyaslamalarını çalıştır")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--split", choices=["val", "test"], default="test", help="Hangi dönem üzerinde backtest yapılacak")
    ap.add_argument("--output-dir", default="data/results")
    ap.add_argument("--commission", type=float, default=COMMISSION_RATE)
    ap.add_argument("--window-size", type=int, default=10)
    ap.add_argument("--weekly-every", type=int, default=5)
    a = ap.parse_args()
    run_baselines(
        train_prices_path=f"{a.data_dir}/train_prices.csv",
        test_prices_path=f"{a.data_dir}/{a.split}_prices.csv",
        output_dir=a.output_dir,
        commission_rate=a.commission,
        window_size=a.window_size,
        weekly_every=a.weekly_every,
    )
    print(f"✅ Baseline sonuçları yazıldı: {a.output_dir}/ (split={a.split})")


if __name__ == "__main__":
    main()