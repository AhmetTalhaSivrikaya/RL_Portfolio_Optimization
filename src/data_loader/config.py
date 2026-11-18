# src/data_loader/config.py

SYMBOLS = ["THYAO.IS", "GARAN.IS", "ASELS.IS", "KCHOL.IS", "BIMAS.IS"]
MARKET_SYMBOL = "XU100.IS"  # BIST 100 Endeksi
START_DATE = "2018-01-01"
END_DATE = "2026-01-01"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# Kalan %15 otomatik olarak test setine ayrılır.