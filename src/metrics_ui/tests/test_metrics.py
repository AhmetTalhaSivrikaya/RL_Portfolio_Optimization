import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metrics as m


def make_df(returns, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(returns))
    r = pd.Series(returns, index=idx)
    v = 100_000 * (1 + r).cumprod()
    return pd.DataFrame({"Portfolio_Value": v, "Daily_Return": r})


def test_constant_series_has_zero_vol_and_drawdown():
    df = make_df([0.0] * 50)
    assert m.annualized_volatility(df.Daily_Return) == 0
    assert m.max_drawdown(df.Portfolio_Value) == 0
    assert np.isnan(m.sharpe_ratio(df.Daily_Return))


def test_monotonic_growth_no_drawdown():
    df = make_df([0.001] * 100)
    v = m.with_start(df)
    assert m.max_drawdown(v) == 0
    assert v.iloc[0] == pytest.approx(100_000)
    assert m.total_return(v) == pytest.approx(1.001**100 - 1)


def test_max_drawdown_known_value():
    v = pd.Series([100, 120, 90, 110, 80.0])
    assert m.max_drawdown(v) == pytest.approx(80 / 120 - 1)


def test_sharpe_matches_formula():
    r = pd.Series([0.01, -0.005, 0.003, 0.002, -0.001])
    expected = np.sqrt(252) * r.mean() / r.std(ddof=0)
    assert m.sharpe_ratio(r) == pytest.approx(expected)


def test_var_cvar_ordering():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 1000))
    var, cvar = m.var_cvar(r)
    assert cvar <= var < 0


def test_align_common_range_and_rebase():
    a = make_df([0.01] * 30, "2024-01-01")
    b = make_df([0.02] * 20, "2024-01-10")
    out = m.align_common_range({"a": a, "b": b})
    assert out["a"].index[0] == out["b"].index[0]
    assert out["a"].index[-1] == out["b"].index[-1]
    start_a = m.with_start(out["a"]).iloc[0]
    assert start_a == pytest.approx(100_000)
    start_b = m.with_start(out["b"]).iloc[0]
    assert start_b == pytest.approx(100_000)


def test_summary_and_format_table():
    df = make_df(np.random.default_rng(1).normal(0.0003, 0.01, 200))
    t = m.summary_table({"x": df})
    f = m.format_table(t)
    assert f.loc["Toplam Getiri", "x"].endswith("%")
    assert "Sharpe" in t.index


def test_matches_team_summary_json():
    """RL ekibinin güncel rl_backtest_summary.json değerleriyle dinamik uyum."""
    results_dir = Path(__file__).resolve().parents[3] / "data" / "results"
    csv_path = results_dir / "rl_backtest_results.csv"
    json_path = results_dir / "rl_backtest_summary.json"
    if not (csv_path.exists() and json_path.exists()):
        pytest.skip("örnek sonuç veya özet dosyası yok")

    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    expected = json.loads(json_path.read_text(encoding="utf-8"))["metrics"]

    s = m.summary(df)
    assert s["Toplam Getiri"] * 100 == pytest.approx(expected["total_return_pct"], abs=1e-2)
    assert s["Sharpe"] == pytest.approx(expected["sharpe"], abs=1e-3)
    assert s["Max Drawdown"] * 100 == pytest.approx(expected["max_drawdown_pct"], abs=1e-2)
    assert s["Toplam Komisyon"] == pytest.approx(expected["total_commission"], abs=1e-1)