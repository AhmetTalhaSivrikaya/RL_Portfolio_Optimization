"""
Sentetik (dummy) veri üretici.

Amaç: Veri Yükleyici (data_loader) hazır olmadan RL/Baseline/Metrik modüllerinin
paralel geliştirilebilmesi için docs/ARAYUZ_SOZLESMESI.md formatında örnek veri üretmek.
DİKKAT: Bu veriyle elde edilen sonuçlar stratejinin başarısını GÖSTERMEZ; yalnızca
boru hattının uçtan uca çalıştığını doğrular.

Kullanım:
    python src/rl_agent/dummy_data_generator.py
"""
import os

import numpy as np
import pandas as pd

# Sözleşmedeki kronolojik bölünme (docs/ARAYUZ_SOZLESMESI.md ile aynı)
SPLIT_RANGES = {
    "train": ("2018-01-01", "2022-12-31"),
    "val": ("2023-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2025-12-31"),
}
TICKERS = ["THYAO.IS", "GARAN.IS", "ASELS.IS", "KCHOL.IS", "BIMAS.IS"]
FEATURE_NAMES = ["ret_1d", "rsi_norm", "macd_norm", "vol_ratio"]


def create_dummy_datasets(output_dir: str = "data/processed", seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    num_assets = len(TICKERS)
    start = SPLIT_RANGES["train"][0]
    end = SPLIT_RANGES["test"][1]

    # İş günlerinden oluşan tarih indeksi (train başı -> test sonu)
    dates = pd.date_range(start=start, end=end, freq="B")
    total_days = len(dates)

    # 1. Geometric Brownian Motion benzeri kapanış fiyatları (pozitif kalması garanti)
    drifts = np.array([0.0008, 0.0005, 0.0006, 0.0004, 0.0007])
    vols = np.array([0.022, 0.025, 0.020, 0.018, 0.016])
    daily_returns = rng.normal(loc=drifts, scale=vols, size=(total_days, num_assets))
    initial_prices = np.array([150.0, 45.0, 35.0, 110.0, 220.0])
    prices = initial_prices * np.cumprod(1 + daily_returns, axis=0)

    prices_df = pd.DataFrame(np.round(prices, 2), index=dates, columns=TICKERS)
    prices_df.index.name = "Date"

    # 2. Normalize edilmiş özellikler: sütun adı {TICKER}_{özellik}
    feature_cols = [f"{t.replace('.IS', '')}_{f}" for t in TICKERS for f in FEATURE_NAMES]
    features = np.clip(rng.standard_normal((total_days, len(feature_cols))), -2.5, 2.5)
    features_df = pd.DataFrame(np.round(features, 4), index=dates, columns=feature_cols)
    features_df.index.name = "Date"

    # 3. Kronolojik bölme ve kaydetme
    print(f"✅ Sahte veri setleri '{output_dir}/' klasörüne kaydedildi:")
    for name, (s, e) in SPLIT_RANGES.items():
        p, f = prices_df.loc[s:e], features_df.loc[s:e]
        p.to_csv(f"{output_dir}/{name}_prices.csv")
        f.to_csv(f"{output_dir}/{name}_features.csv")
        print(f"   - {name:5s}: {len(p)} gün ({p.index[0].date()} -> {p.index[-1].date()})")


if __name__ == "__main__":
    create_dummy_datasets()