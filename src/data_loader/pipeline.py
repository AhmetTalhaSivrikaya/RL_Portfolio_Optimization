# src/data_loader/pipeline.py

import os
from .config import SYMBOLS, MARKET_SYMBOL, TRAIN_RATIO, VAL_RATIO
from .fetcher import fetch_raw_data
from .cleaner import clean_and_align_data
from .features import generate_features
from .splitter import chronological_split_and_scale

def run_data_pipeline(output_dir="data/processed/"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Ham Veriyi Çek
    raw_df = fetch_raw_data()
    
    # 2. Temizle ve Hizala
    all_symbols = SYMBOLS + [MARKET_SYMBOL]
    closes, volumes = clean_and_align_data(raw_df, all_symbols)
    
    # 3. Teknik Özellikleri Üret
    features = generate_features(closes, volumes)
    
    # Fiyatları özelliklerin tarih hizasına getir
    prices = closes.loc[features.index]
    
    # 4. Sızıntısız Böl ve Ölçeklendir
    train_data, val_data, test_data = chronological_split_and_scale(prices, features, TRAIN_RATIO, VAL_RATIO)
    
    train_prices, train_features = train_data
    val_prices, val_features = val_data
    test_prices, test_features = test_data
    
    # 5. Çıktı Dosyalarını Kaydet
    train_prices.to_csv(os.path.join(output_dir, "train_prices.csv"))
    train_features.to_csv(os.path.join(output_dir, "train_features.csv"))
    
    val_prices.to_csv(os.path.join(output_dir, "val_prices.csv"))
    val_features.to_csv(os.path.join(output_dir, "val_features.csv"))
    
    test_prices.to_csv(os.path.join(output_dir, "test_prices.csv"))
    test_features.to_csv(os.path.join(output_dir, "test_features.csv"))
    
    print("Tüm boru hattı başarıyla tamamlandı ve data/processed/ altına kaydedildi!")

if __name__ == "__main__":
    run_data_pipeline()