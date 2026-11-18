"""RL Portföy Optimizasyonu: Performans Paneli (Streamlit).

Çalıştırma (repo kökünden):
    streamlit run src/metrics_ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as m  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_DIRS = ["data/results", "results_signal"]

st.set_page_config(page_title="RL Portföy Analizi", page_icon="📈", layout="wide")
st.title("📈 RL Portföy Optimizasyonu: Performans Paneli")


# --------------------------------------------------------------------------- #
# Veri
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load(results_dir: str) -> dict[str, pd.DataFrame]:
    return m.load_results(results_dir)


st.sidebar.header("Ayarlar")
choice = st.sidebar.selectbox("Sonuç klasörü", CANDIDATE_DIRS + ["Özel yol..."])
if choice == "Özel yol...":
    results_dir = Path(st.sidebar.text_input("Klasör yolu", str(REPO_ROOT / "data" / "results")))
else:
    results_dir = REPO_ROOT / choice

if not results_dir.exists():
    st.error(f"Klasör bulunamadı: `{results_dir}`")
    st.stop()

try:
    raw = load(str(results_dir))
except ValueError as e:
    st.error(str(e))
    st.stop()

if not raw:
    st.warning(f"`{results_dir}` içinde `*_backtest_results.csv` dosyası yok.")
    st.stop()

names = list(raw)
selected = st.sidebar.multiselect("Stratejiler", names, default=names)
rf = st.sidebar.number_input("Risksiz getiri (yıllık, %)", 0.0, 100.0, 0.0, 0.5) / 100
roll_win = st.sidebar.slider("Rolling Sharpe penceresi (gün)", 20, 120, 60, 5)

if not selected:
    st.info("Soldan en az bir strateji seçin.")
    st.stop()

data = m.align_common_range({k: raw[k] for k in selected})
first = next(iter(data.values()))
st.caption(
    f"Test dönemi: **{first.index[0].date()} → {first.index[-1].date()}** · "
    f"{len(first)} işlem günü · başlangıç sermayesi {m.INITIAL_CAPITAL:,.0f} · "
    f"{len(data)} strateji"
)
if len(data) == 1:
    st.info("Şu an tek strateji var. Baseline (Markowitz, Buy&Hold) CSV'leri gelince karşılaştırma otomatik görünür.")

# --------------------------------------------------------------------------- #
# Üst kartlar (ilk strateji ya da tek seçim)
# --------------------------------------------------------------------------- #
table = m.summary_table(data, rf)
pretty = m.format_table(table)

cols = st.columns(len(data))
for col, (name, _) in zip(cols, data.items()):
    with col:
        st.markdown(f"**{name}**")
        a, b = st.columns(2)
        a.metric("Toplam Getiri", pretty.loc["Toplam Getiri", name])
        b.metric("Sharpe", pretty.loc["Sharpe", name])
        a.metric("Max Drawdown", pretty.loc["Max Drawdown", name])
        b.metric("Yıllık Vol.", pretty.loc["Yıllık Volatilite", name])

tab_perf, tab_risk, tab_w, tab_cost, tab_table = st.tabs(
    ["Performans", "Risk", "Ağırlıklar", "Maliyet", "Tüm Metrikler"]
)

LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15), hovermode="x unified")

# --------------------------------------------------------------------------- #
# Performans
# --------------------------------------------------------------------------- #
with tab_perf:
    fig = go.Figure()
    for name, df in data.items():
        v = m.with_start(df)
        fig.add_trace(go.Scatter(x=v.index, y=v, name=name, mode="lines"))
    fig.update_layout(title="Portföy Değeri (komisyon sonrası)", yaxis_title="TL", **LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for name, df in data.items():
            cum = (1 + df["Daily_Return"]).cumprod() - 1
            fig.add_trace(go.Scatter(x=cum.index, y=cum * 100, name=name))
        fig.update_layout(title="Kümülatif Getiri (%)", **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        mr = {n: m.monthly_returns(d["Daily_Return"]) for n, d in data.items()}
        pick = st.selectbox("Aylık getiri ısı haritası", list(mr), key="heat")
        t = mr[pick] * 100
        fig = go.Figure(
            go.Heatmap(
                z=t.values, x=[f"{c:02d}" for c in t.columns], y=t.index.astype(str),
                colorscale="RdYlGn", zmid=0, text=t.round(1).values, texttemplate="%{text}",
                colorbar=dict(title="%"),
            )
        )
        fig.update_layout(title="Aylık Getiri (%)", margin=dict(l=10, r=10, t=40, b=10))         
        fig.update_yaxes(type="category")
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)
        

# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
with tab_risk:
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for name, df in data.items():
            dd = m.drawdown_series(m.with_start(df)) * 100
            fig.add_trace(go.Scatter(x=dd.index, y=dd, name=name, fill="tozeroy", opacity=0.5))
        fig.update_layout(title="Drawdown (%)", **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        for name, df in data.items():
            rs = m.rolling_sharpe(df["Daily_Return"], roll_win, rf)
            fig.add_trace(go.Scatter(x=rs.index, y=rs, name=name))
        fig.add_hline(y=0, line_dash="dot", opacity=0.5)
        fig.update_layout(title=f"Rolling Sharpe ({roll_win} gün)", **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    for name, df in data.items():
        fig.add_trace(go.Histogram(x=df["Daily_Return"] * 100, name=name, opacity=0.6, nbinsx=50))
    fig.update_layout(barmode="overlay", title="Günlük Getiri Dağılımı (%)", **LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        pretty.loc[["Yıllık Volatilite", "Max Drawdown", "VaR %95 (günlük)", "CVaR %95 (günlük)", "En Kötü Gün"]],
        use_container_width=True,
    )

# --------------------------------------------------------------------------- #
# Ağırlıklar
# --------------------------------------------------------------------------- #
with tab_w:
    with_w = {n: d for n, d in data.items() if m.weight_columns(d)}
    if not with_w:
        st.info("Seçili sonuç dosyalarında `W_*` ağırlık sütunu yok.")
    else:
        pick = st.selectbox("Strateji", list(with_w), key="w_pick")
        d = with_w[pick]
        fig = go.Figure()
        for c in m.weight_columns(d):
            fig.add_trace(
                go.Scatter(x=d.index, y=d[c] * 100, name=c[2:], stackgroup="w", mode="lines", line=dict(width=0.5))
            )
        fig.update_layout(title=f"Portföy Ağırlıkları (%): {pick}", yaxis=dict(range=[0, 100]), **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        avg = (d[m.weight_columns(d)].mean() * 100).rename(lambda c: c[2:]).round(1)
        st.caption("Dönem boyunca ortalama ağırlık (%)")
        st.dataframe(avg.to_frame("Ortalama %").T, use_container_width=True)

# --------------------------------------------------------------------------- #
# Maliyet
# --------------------------------------------------------------------------- #
with tab_cost:
    with_c = {n: d for n, d in data.items() if "Commission" in d.columns}
    if not with_c:
        st.info("`Commission` / `Turnover` sütunu yok (opsiyonel sütunlar).")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for name, d in with_c.items():
                fig.add_trace(go.Scatter(x=d.index, y=d["Commission"].cumsum(), name=name))
            fig.update_layout(title="Kümülatif Komisyon (TL)", **LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            for name, d in with_c.items():
                reb = d[d["Turnover"] > 0]
                fig.add_trace(go.Bar(x=reb.index, y=reb["Turnover"] * 100, name=name))
            fig.update_layout(title="Rebalans Günlerinde Turnover (%)", barmode="group", **LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        for name, d in with_c.items():
            st.write(
                f"**{name}**: {int((d['Turnover'] > 0).sum())} rebalans, "
                f"toplam komisyon {d['Commission'].sum():,.0f} TL "
                f"(başlangıç sermayesinin %{d['Commission'].sum() / m.INITIAL_CAPITAL * 100:.2f}'i)"
            )

# --------------------------------------------------------------------------- #
# Tüm metrikler
# --------------------------------------------------------------------------- #
with tab_table:
    st.dataframe(pretty, use_container_width=True, height=560)
    st.download_button(
        "Metrikleri CSV indir", table.to_csv().encode("utf-8"), "metrics_summary.csv", "text/csv"
    )
    st.caption(
        "Sharpe/Sortino yıllıklandırması 252 işlem günüdür ve std için ddof=0 kullanılır "
        "(RL ekibinin summary.json değerleriyle birebir uyumlu). VaR/CVaR tarihsel yöntemle hesaplanır."
    )
