"""
Veri Sözleşmesi Doğrulayıcı (v2)

Ekip içi sözleşmenin (bkz. docs/ARAYUZ_SOZLESMESI.md) kod olarak uygulanmış halidir.
  * Veri Yükleyici (data_loader) çıktılarını doğrular   -> load_and_validate_data, validate_split_continuity
  * Baseline / RL sonuç dosyalarını doğrular            -> validate_results_file

Komut satırı:
    python src/rl_agent/market_data_validator.py
"""
import os
import sys

import numpy as np
import pandas as pd

# --- Ekip genelinde ortak sabitler (docs/ARAYUZ_SOZLESMESI.md ile aynı tutulmalı) ---
INITIAL_BALANCE = 100_000.0
SPLITS = ("train", "val", "test")
GLOBAL_FEATURE_PREFIX = "MKT_"  # Piyasa geneli özellikler için (örn. MKT_xu100_ret_1d)
FEATURE_ERROR_ABS = 50.0  # Üstü: büyük olasılıkla ölçeklenmemiş ham değer -> HATA
FEATURE_WARN_ABS = 10.0  # Üstü: UYARI


class DataContractError(ValueError):
    """Veri sözleşmesi ihlali. (assert yerine kullanılır: python -O ile de çalışır.)"""


def _fail(msg: str):
    raise DataContractError(f"HATA: {msg}")


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        _fail(f"Dosya bulunamadı: {path}")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        _fail(f"{path}: 'Date' sütunu tarih olarak okunamadı (format: YYYY-MM-DD).")
    return df


def _check_index(df: pd.DataFrame, name: str):
    if df.index.hasnans:
        _fail(f"{name}: Date indeksinde geçersiz (NaT) tarih var.")
    if not df.index.is_unique:
        _fail(f"{name}: Date indeksinde tekrar eden tarih var.")
    if not df.index.is_monotonic_increasing:
        _fail(f"{name}: Date indeksi artan sırada (eskiden yeniye) değil.")


def _base_name(ticker: str) -> str:
    return ticker.replace(".IS", "")


# ----------------------------------------------------------------------------------
# 1) Girdi: fiyat + özellik dosyaları
# ----------------------------------------------------------------------------------
def load_and_validate_data(prices_path: str, features_path: str, min_assets: int = 2):
    prices_df = _read_csv(prices_path)
    features_df = _read_csv(features_path)

    # --- İndeks kuralları ---
    _check_index(prices_df, os.path.basename(prices_path))
    _check_index(features_df, os.path.basename(features_path))
    if len(prices_df) != len(features_df):
        _fail("Fiyat ve özellik satır sayıları eşit değil!")
    if not (prices_df.index == features_df.index).all():
        _fail("Fiyat ve özellik tarih indeksleri uyuşmuyor!")

    # --- Fiyat kuralları ---
    if prices_df.shape[1] < min_assets:
        _fail(f"En az {min_assets} varlık gerekli, bulunan: {prices_df.shape[1]}.")
    if prices_df.columns.duplicated().any():
        _fail("Fiyat dosyasında tekrar eden hisse sütunu var.")
    if not all(pd.api.types.is_numeric_dtype(t) for t in prices_df.dtypes):
        _fail("Fiyat sütunları sayısal değil.")
    if prices_df.isna().any().any() or not np.isfinite(prices_df.values).all():
        _fail("Fiyat verisinde NaN/sonsuz değer var!")
    if (prices_df.values <= 0).any():
        _fail("Fiyat verisinde sıfır veya negatif değer var!")

    # --- Özellik kuralları ---
    if features_df.shape[1] == 0:
        _fail("Özellik dosyasında hiç sütun yok.")
    if features_df.columns.duplicated().any():
        _fail("Özellik dosyasında tekrar eden sütun var.")
    if not all(pd.api.types.is_numeric_dtype(t) for t in features_df.dtypes):
        _fail("Özellik sütunları sayısal değil.")
    if features_df.isna().any().any() or not np.isfinite(features_df.values).all():
        _fail("Özellik verisinde NaN/sonsuz değer var! (İndikatör ısınma satırlarını yükleyici silmeli.)")

    # Adlandırma: {TICKER}_{özellik}; her hisse için aynı sayıda özellik
    counts = {}
    known = {_base_name(t) for t in prices_df.columns}
    for col in features_df.columns:
        if col.startswith(GLOBAL_FEATURE_PREFIX):
            continue
        prefix = col.split("_", 1)[0]
        if prefix not in known:
            _fail(
                f"Özellik sütunu '{col}' hiçbir hisseyle eşleşmiyor. "
                f"Beklenen format: <HISSE>_<özellik> (örn. THYAO_rsi_norm) "
                f"veya piyasa geneli için '{GLOBAL_FEATURE_PREFIX}<özellik>'."
            )
        counts[prefix] = counts.get(prefix, 0) + 1
    for base in known:
        if counts.get(base, 0) == 0:
            _fail(f"'{base}' hissesi için hiç özellik sütunu yok.")
    if len(set(counts.values())) != 1:
        _fail(f"Hisse başına özellik sayısı eşit değil: {counts}")

    max_val = float(features_df.abs().max().max())
    if max_val > FEATURE_ERROR_ABS:
        _fail(
            f"Özelliklerde çok büyük değer var ({max_val:.2f} > {FEATURE_ERROR_ABS:g}). "
            "Ham fiyat/hacim seviyesi veya ölçeklenmemiş özellik olabilir."
        )
    if max_val > FEATURE_WARN_ABS:
        print(f"⚠️ UYARI: features_df içinde büyük değer var ({max_val:.2f}). Normalizasyonu kontrol edin.")
    const_cols = [c for c in features_df.columns if features_df[c].std() == 0]
    if const_cols:
        print(f"⚠️ UYARI: Sabit (varyansı 0) özellik sütunları: {const_cols}")

    print(
        f"✅ Veri doğrulandı: {len(prices_df)} gün, {prices_df.shape[1]} hisse, "
        f"{features_df.shape[1]} özellik ({prices_df.index[0].date()} -> {prices_df.index[-1].date()})."
    )
    return prices_df, features_df


def validate_split_continuity(splits: dict):
    """
    splits: {"train": (prices_df, features_df), "val": (...), "test": (...)}
    Kronolojik sıra, çakışmama ve sütun tutarlılığını denetler (sızıntı koruması).
    """
    names = [s for s in SPLITS if s in splits]
    for a, b in zip(names, names[1:]):
        if splits[a][0].index.max() >= splits[b][0].index.min():
            _fail(f"'{a}' seti '{b}' setinden önce BİTMELİ; tarihler çakışıyor/ters sırada.")
    ref_p, ref_f = splits[names[0]][0].columns, splits[names[0]][1].columns
    for n in names[1:]:
        if list(splits[n][0].columns) != list(ref_p):
            _fail(f"'{n}' fiyat sütunları (sıra dahil) '{names[0]}' ile aynı değil.")
        if list(splits[n][1].columns) != list(ref_f):
            _fail(f"'{n}' özellik sütunları (sıra dahil) '{names[0]}' ile aynı değil.")
    print(f"✅ Split sürekliliği doğrulandı: {' -> '.join(names)}")


# ----------------------------------------------------------------------------------
# 2) Çıktı: backtest sonuç dosyaları (RL, Markowitz, Buy&Hold ...)
# ----------------------------------------------------------------------------------
def validate_results_file(path: str, initial_balance: float = INITIAL_BALANCE, tol: float = 1e-4):
    df = pd.read_csv(path, parse_dates=["Date"])
    required = ["Date", "Portfolio_Value", "Daily_Return", "W_CASH"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        _fail(f"{path}: zorunlu sütunlar eksik: {missing}")
    if (not df["Date"].is_monotonic_increasing) or df["Date"].duplicated().any():
        _fail(f"{path}: Date sütunu artan ve tekrarsız olmalı.")
    if df[required[1:]].isna().any().any():
        _fail(f"{path}: NaN değer var.")

    w_cols = [c for c in df.columns if c.startswith("W_")]
    if np.abs(df[w_cols].sum(axis=1) - 1.0).max() > 1e-3:
        _fail(f"{path}: Ağırlıklar (W_*) her satırda toplamı 1 olmalı.")
    if (df[w_cols].values < -1e-9).any():
        _fail(f"{path}: Negatif ağırlık var (yalnızca long pozisyon).")

    # Net getiri tutarlılığı: Daily_Return == PV_t / PV_{t-1} - 1  (komisyon dahil)
    pv = df["Portfolio_Value"].values
    implied = np.empty_like(pv)
    implied[0] = pv[0] / initial_balance - 1.0
    implied[1:] = pv[1:] / pv[:-1] - 1.0
    err = np.abs(implied - df["Daily_Return"].values).max()
    if err > tol:
        _fail(
            f"{path}: Daily_Return, Portfolio_Value ile tutarsız (maks. fark {err:.2e}). "
            "Daily_Return komisyon DAHİL net getiri olmalı."
        )
    print(f"✅ Sonuç dosyası doğrulandı: {os.path.basename(path)} ({len(df)} satır).")
    return df


# ----------------------------------------------------------------------------------
# Komut satırı: data/processed ve data/results altındaki her şeyi denetle
# ----------------------------------------------------------------------------------
def main(processed_dir: str = "data/processed", results_dir: str = "data/results"):
    found = {}
    for s in SPLITS:
        p, f = f"{processed_dir}/{s}_prices.csv", f"{processed_dir}/{s}_features.csv"
        if os.path.exists(p) or os.path.exists(f):
            print(f"--- {s}")
            found[s] = load_and_validate_data(p, f)
    if not found:
        _fail(f"'{processed_dir}' altında hiç split bulunamadı.")
    if len(found) > 1:
        validate_split_continuity(found)
    if os.path.isdir(results_dir):
        for fn in sorted(os.listdir(results_dir)):
            if fn.endswith("_backtest_results.csv"):
                validate_results_file(os.path.join(results_dir, fn))


if __name__ == "__main__":
    try:
        main()
    except DataContractError as e:
        print(e)
        sys.exit(1)