import pandas as pd


def load_and_validate_data(prices_path: str, features_path: str):
    prices_df = pd.read_csv(prices_path, index_col="Date", parse_dates=True)
    features_df = pd.read_csv(features_path, index_col="Date", parse_dates=True)

    # 1. Satır ve tarih uyumu kontrolü
    assert len(prices_df) == len(features_df), "HATA: Fiyat ve özellik satır sayıları eşit değil!"
    assert (prices_df.index == features_df.index).all(), "HATA: Tarih indeksleri uyuşmuyor!"

    # 2. Eksik veya sonsuz veri kontrolü
    assert not prices_df.isna().any().any(), "HATA: Fiyat verisinde NaN değer var!"
    assert not features_df.isna().any().any(), "HATA: Özellik verisinde NaN değer var!"

    # 3. Normalizasyon kontrolü (Özellikler aşırı büyük olmamalı)
    max_val = features_df.abs().max().max()
    if max_val > 50:
        print(f"⚠️ UYARI: features_df içinde çok büyük değer var ({max_val:.2f}). Normalizasyon yapılmamış olabilir!")

    print(f"✅ Veri doğrulandı: {len(prices_df)} gün, {prices_df.shape[1]} hisse, {features_df.shape[1]} özellik.")
    return prices_df, features_df