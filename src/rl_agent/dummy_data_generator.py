import os
import numpy as np
import pandas as pd


def create_dummy_datasets(
    output_dir: str = "data/processed",
    train_days: int = 1000,
    test_days: int = 250,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    tickers = ["THYAO.IS", "GARAN.IS", "ASELS.IS", "KCHOL.IS", "BIMAS.IS"]
    num_assets = len(tickers)
    total_days = train_days + test_days

    # İş günlerinden oluşan tarih indeksi
    dates = pd.date_range(start="2020-01-01", periods=total_days, freq="B")

    # 1. Gerçekçi hisse kapanış fiyatları üret (Geometric Brownian Motion benzeri)
    # Farklı trendlere sahip hisseler simüle edilir
    drifts = np.array([0.0008, 0.0005, 0.0006, 0.0004, 0.0007])
    vols = np.array([0.022, 0.025, 0.020, 0.018, 0.016])

    daily_returns = np.random.normal(loc=drifts, scale=vols, size=(total_days, num_assets))
    initial_prices = np.array([150.0, 45.0, 35.0, 110.0, 220.0])
    prices = initial_prices * np.cumprod(1 + daily_returns, axis=0)

    prices_df = pd.DataFrame(np.round(prices, 2), index=dates, columns=tickers)
    prices_df.index.name = "Date"

    # 2. Normalize edilmiş teknik indikatör özellikleri üret (Her hisse için 4 özellik)
    feature_cols = []
    for t in tickers:
        base_name = t.replace(".IS", "")
        feature_cols.extend([
            f"{base_name}_ret_1d",
            f"{base_name}_rsi_norm",
            f"{base_name}_macd_norm",
            f"{base_name}_vol_ratio",
        ])

    # [-2.5, +2.5] aralığında standartlaştırılmış sahte sinyaller
    features = np.clip(np.random.randn(total_days, len(feature_cols)), -2.5, 2.5)
    features_df = pd.DataFrame(np.round(features, 4), index=dates, columns=feature_cols)
    features_df.index.name = "Date"

    # 3. Kronolojik olarak Train ve Test setlerine ayır ve kaydet
    train_prices = prices_df.iloc[:train_days]
    test_prices = prices_df.iloc[train_days:]

    train_features = features_df.iloc[:train_days]
    test_features = features_df.iloc[train_days:]

    train_prices.to_csv(f"{output_dir}/train_prices.csv")
    train_features.to_csv(f"{output_dir}/train_features.csv")
    test_prices.to_csv(f"{output_dir}/test_prices.csv")
    test_features.to_csv(f"{output_dir}/test_features.csv")

    print(f"✅ Sahte veri setleri '{output_dir}/' klasörüne kaydedildi:")
    print(f"   - Train: {len(train_prices)} gün ({train_prices.index[0].date()} -> {train_prices.index[-1].date()})")
    print(f"   - Test : {len(test_prices)} gün ({test_prices.index[0].date()} -> {test_prices.index[-1].date()})")


if __name__ == "__main__":
    create_dummy_datasets()