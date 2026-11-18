# src/data_loader/fetcher.py

import os
import pandas as pd
import yfinance as yf
from .config import START_DATE, END_DATE, SYMBOLS, MARKET_SYMBOL

def fetch_raw_data(save_dir="data/raw"):
    os.makedirs(save_dir, exist_ok=True)
    all_symbols = SYMBOLS + [MARKET_SYMBOL]
    
    print(f"Ham veriler indiriliyor: {all_symbols}")
    # yfinance kilitlenmelerini önlemek için threads=False kullanıyoruz
    data = yf.download(all_symbols, start=START_DATE, end=END_DATE, group_by="ticker", auto_adjust=True, threads=False)
    
    # Ham verileri kaydet
    file_path = os.path.join(save_dir, "raw_market_data.csv")
    data.to_csv(file_path)
    print(f"Ham veri kaydedildi: {file_path}")
    return data