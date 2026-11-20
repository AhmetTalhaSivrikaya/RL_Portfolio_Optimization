# src/data_loader/features.py

import pandas as pd
import numpy as np

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def generate_features(closes, volumes, market_symbol="XU100.IS"):
    features_dict = {}

    for col in closes.columns:
        p = closes[col]
        df = pd.DataFrame(index=closes.index)

        # Piyasa endeksi ise MKT_ önekiyle piyasa göstergesi üret, hisse ise hisse özelliklerini üret
        prefix = "MKT_xu100" if col == market_symbol else col

        # Log Getiri ve Volatilite
        df[f"{prefix}_log_ret"] = np.log(p / p.shift(1))
        df[f"{prefix}_volatility_20"] = df[f"{prefix}_log_ret"].rolling(20).std()

        # Hareketli Ortalamalar
        df[f"{prefix}_sma_20"] = p.rolling(20).mean() / p - 1
        df[f"{prefix}_sma_50"] = p.rolling(50).mean() / p - 1

        # RSI
        df[f"{prefix}_rsi_14"] = compute_rsi(p, 14)

        # Bollinger Bantları
        sma20 = p.rolling(20).mean()
        std20 = p.rolling(20).std()
        df[f"{prefix}_bollinger_up"] = (sma20 + (2 * std20) - p) / p
        df[f"{prefix}_bollinger_low"] = (p - (sma20 - (2 * std20))) / p

        features_dict[col] = df

    features_df = pd.concat(features_dict.values(), axis=1)

    # Isınma (warm-up) periyodundan kaynaklanan NaN satırları temizle
    features_df = features_df.dropna()
    return features_df