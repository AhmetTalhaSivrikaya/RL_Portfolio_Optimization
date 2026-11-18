"""Finansal risk metrikleri ve sonuç dosyası yükleyici (arayuz_sozlesmesi.md §3).

UI'dan bağımsızdır; sadece pandas/numpy kullanır.
Sharpe/Sortino/drawdown yalnızca `Daily_Return` ve `Portfolio_Value` sütunlarından hesaplanır.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
INITIAL_CAPITAL = 100_000.0
RESULT_SUFFIX = "_backtest_results.csv"
REQUIRED_COLS = ["Portfolio_Value", "Daily_Return"]


# --------------------------------------------------------------------------- #
# Yükleme
# --------------------------------------------------------------------------- #
def load_results(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    """`<yöntem>_backtest_results.csv` dosyalarını okur -> {yöntem: DataFrame(Date indeksli)}."""
    results_dir = Path(results_dir)
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(results_dir.glob(f"*{RESULT_SUFFIX}")):
        name = path.name[: -len(RESULT_SUFFIX)]
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: eksik sütun(lar): {missing}")
        out[name] = df
    return out


def align_common_range(results: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Tüm yöntemleri ortak tarih aralığına kırpar ve Portfolio_Value'yu 100.000'e yeniden tabanlar."""
    if not results:
        return {}
    start = max(df.index[0] for df in results.values())
    end = min(df.index[-1] for df in results.values())
    aligned = {}
    for name, df in results.items():
        sub = df.loc[start:end].copy()
        if sub.empty:
            raise ValueError(f"{name}: ortak tarih aralığı boş ({start.date()} - {end.date()})")
        # Başlangıç noktası = ilk satırdan önceki kapanış (sözleşme §3: ilk satırın önceki değeri 100.000)
        start_value = sub["Portfolio_Value"].iloc[0] / (1 + sub["Daily_Return"].iloc[0])
        sub["Portfolio_Value"] = sub["Portfolio_Value"] / start_value * INITIAL_CAPITAL
        aligned[name] = sub
    return aligned


def with_start(df: pd.DataFrame) -> pd.Series:
    """Portföy değeri serisinin başına başlangıç noktasını (ilk satırdan önceki kapanış) ekler.

    Toplam getiri, yıllık getiri ve drawdown bu seri üzerinden hesaplanır; böylece ilk günün
    getirisi de hesaba katılır (RL ekibinin summary.json değerleriyle uyumlu).
    """
    v, r = df["Portfolio_Value"], df["Daily_Return"]
    start = pd.Series([v.iloc[0] / (1 + r.iloc[0])], index=[v.index[0] - pd.tseries.offsets.BDay(1)])
    return pd.concat([start, v])


# --------------------------------------------------------------------------- #
# Tekil metrikler  (r: günlük net getiri serisi, v: portföy değeri serisi)
# --------------------------------------------------------------------------- #
def total_return(v: pd.Series) -> float:
    return float(v.iloc[-1] / v.iloc[0] - 1)


def annualized_return(v: pd.Series) -> float:
    n = len(v) - 1
    return float((v.iloc[-1] / v.iloc[0]) ** (TRADING_DAYS / n) - 1) if n > 0 else float("nan")


def annualized_volatility(r: pd.Series) -> float:
    return float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(r: pd.Series, rf_annual: float = 0.0) -> float:
    ex = r - rf_annual / TRADING_DAYS
    sd = ex.std(ddof=0)
    return float(np.sqrt(TRADING_DAYS) * ex.mean() / sd) if sd > 0 else float("nan")


def sortino_ratio(r: pd.Series, rf_annual: float = 0.0) -> float:
    ex = r - rf_annual / TRADING_DAYS
    downside = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    return float(np.sqrt(TRADING_DAYS) * ex.mean() / downside) if downside > 0 else float("nan")


def drawdown_series(v: pd.Series) -> pd.Series:
    return v / v.cummax() - 1


def max_drawdown(v: pd.Series) -> float:
    return float(drawdown_series(v).min())


def calmar_ratio(v: pd.Series) -> float:
    mdd = abs(max_drawdown(v))
    return float(annualized_return(v) / mdd) if mdd > 0 else float("nan")


def var_cvar(r: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """Tarihsel günlük VaR ve CVaR (negatif sayı = kayıp)."""
    var = float(r.quantile(alpha))
    tail = r[r <= var]
    return var, float(tail.mean()) if len(tail) else var


def rolling_sharpe(r: pd.Series, window: int = 60, rf_annual: float = 0.0) -> pd.Series:
    ex = r - rf_annual / TRADING_DAYS
    return np.sqrt(TRADING_DAYS) * ex.rolling(window).mean() / ex.rolling(window).std(ddof=0)


# --------------------------------------------------------------------------- #
# Özet
# --------------------------------------------------------------------------- #
def summary(df: pd.DataFrame, rf_annual: float = 0.0) -> dict[str, float]:
    """Tek yöntem için tüm metrikler. `df`: Portfolio_Value + Daily_Return (+ opsiyonel Turnover/Commission)."""
    v, r = with_start(df), df["Daily_Return"]
    var, cvar = var_cvar(r)
    out = {
        "Toplam Getiri": total_return(v),
        "Yıllık Getiri": annualized_return(v),
        "Yıllık Volatilite": annualized_volatility(r),
        "Sharpe": sharpe_ratio(r, rf_annual),
        "Sortino": sortino_ratio(r, rf_annual),
        "Max Drawdown": max_drawdown(v),
        "Calmar": calmar_ratio(v),
        "VaR %95 (günlük)": var,
        "CVaR %95 (günlük)": cvar,
        "Kazanan Gün Oranı": float((r > 0).mean()),
        "En İyi Gün": float(r.max()),
        "En Kötü Gün": float(r.min()),
        "Son Değer": float(v.iloc[-1]),
    }
    if "Turnover" in df.columns:
        out["Ort. Turnover"] = float(df["Turnover"].mean())
    if "Commission" in df.columns:
        out["Toplam Komisyon"] = float(df["Commission"].sum())
    return out


def summary_table(results: dict[str, pd.DataFrame], rf_annual: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({name: summary(df, rf_annual) for name, df in results.items()})


# Tablo biçimlendirme için hangi satır yüzde / sayı / para
PCT_ROWS = {
    "Toplam Getiri", "Yıllık Getiri", "Yıllık Volatilite", "Max Drawdown",
    "VaR %95 (günlük)", "CVaR %95 (günlük)", "Kazanan Gün Oranı", "En İyi Gün", "En Kötü Gün",
}
MONEY_ROWS = {"Son Değer", "Toplam Komisyon"}


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    def fmt(row_name, x):
        if pd.isna(x):
            return "-"
        if row_name in PCT_ROWS:
            return f"{x * 100:.2f}%"
        if row_name in MONEY_ROWS:
            return f"{x:,.0f}"
        return f"{x:.3f}"

    return pd.DataFrame(
        {c: [fmt(idx, table.loc[idx, c]) for idx in table.index] for c in table.columns},
        index=table.index,
    )


# --------------------------------------------------------------------------- #
# Ağırlıklar / aylık getiri
# --------------------------------------------------------------------------- #
def weight_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("W_")]


def monthly_returns(r: pd.Series) -> pd.DataFrame:
    """Yıl x ay tablosu (bileşik aylık getiri)."""
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    t = m.unstack()
    t.index.name = "Yıl"
    return t
