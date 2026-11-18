# src/data_loader/cleaner.py

import pandas as pd

def clean_and_align_data(raw_df, symbols):
    closes = pd.DataFrame()
    volumes = pd.DataFrame()
    
    for sym in symbols:
        if sym in raw_df.columns.levels[0]:
            closes[sym] = raw_df[sym]["Close"]
            volumes[sym] = raw_df[sym]["Volume"]
            
    # Eksik verileri doldur ve NaN'leri temizle
    closes = closes.dropna(how="all").ffill().bfill()
    volumes = volumes.fillna(0)
    
    # Sıfır veya negatif fiyat kontrolü
    if (closes <= 0).any().any():
        raise ValueError("Hatalı sıfır veya negatif fiyat tespit edildi!")
        
    return closes, volumes