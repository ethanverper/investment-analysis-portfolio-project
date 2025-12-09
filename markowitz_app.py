# markowitz_app.py
# Streamlit dashboard: Markowitz optimization with automated FRED 1-Month T-Bill (DGS1MO)

import io
import math
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.tseries.offsets import BMonthBegin, MonthEnd
from pandas_datareader import data as pdr
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize

# --------------------------- #
# Streamlit page config
# --------------------------- #
st.set_page_config(
    page_title="Markowitz Optimization (with automated FRED 1-Month T-Bill)",
    layout="wide",
    page_icon="📊",
)

# --------------------------- #
# Helper formatting
# --------------------------- #
def to_csv_bytes(df: pd.DataFrame, index=True) -> bytes:
    return df.to_csv(index=index).encode("utf-8")

def pct(x):  # format helper
    return f"{100*x:,.2f}%"

def nice_weights_text(weights, tickers, title):
    nonzero = [(t, w) for t, w in zip(tickers, weights) if w > 1e-6]
    lines = [f"**{title}**"]
    if not nonzero:
        lines.append("- (All weights are ~0 under constraints.)")
    else:
        for t, w in nonzero:
            lines.append(f"- **{t}** → **{pct(w)}**")
    return "\n".join(lines)

# --------------------------- #
# Sidebar — inputs
# --------------------------- #
st.sidebar.header("Inputs")

tickers_input = st.sidebar.text_input(
    "Tickers (comma separated):",
    value="GOOGL,WMT,AAPL,DAL,VCIT,VCLT",
).upper().replace(" ", "")

append_spy = st.sidebar.checkbox("Append SPY (S&P 500 ETF) to investable list", value=True)
append_gspc = st.sidebar.checkbox("Add ^GSPC benchmark series (index, non-investable)", value=True)
append_bil = st.sidebar.checkbox("Append BIL (1–3M T-Bills ETF) to investable list", value=False)

start_date = st.sidebar.date_input("Start date", value=datetime.today() - relativedelta(years=5))
end_date   = st.sidebar.date_input("End date", value=datetime.today())

sampling_choice = st.sidebar.selectbox(
    "Sampling frequency",
    ["Monthly (first trading day)", "Monthly (month end)", "Monthly (custom day index)"],
    index=0,
)

custom_day = None
if sampling_choice == "Monthly (custom day index)":
    custom_day = st.sidebar.number_input(
        "Day index within month (0 = first valid, 1 = second, ...)",
        min_value=0, max_value=20, value=0, step=1,
        help="Example: 0 picks the first trading day, 1 picks the second, etc."
    )

long_only = st.sidebar.checkbox("Long-only (weights ≥ 0)", value=True)
frontier_points = st.sidebar.slider("Frontier points", min_value=20, max_value=200, value=120)

st.sidebar.markdown(
    "<small>Tip: use SPY as the investable S&P 500 proxy; "
    "^GSPC is shown as a benchmark only.</small>", unsafe_allow_html=True
)

# Risk-free fallback if FRED fails
rf_fallback = st.sidebar.slider(
    "Risk-free (annual) if FRED is unavailable",
    min_value=0.00, max_value=0.10, step=0.005, value=0.02
)

# Compose investable & benchmark lists
base_tickers = [t for t in tickers_input.split(",") if t]
investable = base_tickers.copy()
if append_spy and "SPY" not in investable:
    investable.append("SPY")
if append_bil and "BIL" not in investable:
    investable.append("BIL")

benchmark = ["^GSPC"] if append_gspc else []
all_series = sorted(list(dict.fromkeys(investable + benchmark)))  # unique while preserving order-ish

# --------------------------- #
# Title & Step 1 — Parameters
# --------------------------- #
st.markdown("# 📊 Markowitz Optimization (with automated FRED 1-Month T-Bill)")
st.caption("Investment Analysis & Portfolio Management — step-by-step technical view")

st.markdown("## ① Parameters used")
with st.expander("Parameters used (click to view)"):
    st.write({
        "Investable tickers": investable,
        "Benchmark tickers": benchmark,
        "Date range": [str(start_date), str(end_date)],
        "Sampling": sampling_choice if custom_day is None else f"{sampling_choice} (index={custom_day})",
        "Long-only?": long_only,
        "Frontier points": frontier_points,
        "Risk-free fallback (annual)": rf_fallback,
    })

# --------------------------- #
# Data download (Yahoo Finance)
# --------------------------- #
st.markdown("## ② Data download 🔗")
st.write(
    "We pull **daily Adjusted Close** from Yahoo Finance for the investable assets and "
    "(optionally) the non-investable benchmark (^GSPC). Then we sample to **monthly** series."
)

@st.cache_data(show_spinner=False)
def fetch_prices_yahoo(symbols, start, end):
    """
    Robust Yahoo fetch:
    - Fuerza auto_adjust=False (para que exista 'Adj Close' como antes).
    - Si por alguna razón llega sin MultiIndex, cae elegantemente a 'Close'
      o a columnas directamente con los tickers.
    - Devuelve un DataFrame con columnas = tickers solicitados (los que sí llegaron).
    """
    # Fuerza el formato “clásico” con MultiIndex OHLCV + 'Adj Close'
    raw = yf.download(
        symbols,
        start=pd.to_datetime(start),
        end=pd.to_datetime(end),
        auto_adjust=False,          # <- clave
        group_by="column",          # estándar
        threads=True,
        progress=False,
    )

    # Caso 1: columnas MultiIndex (esperado cuando auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        top = raw.columns.get_level_values(0)
        if "Adj Close" in top:
            data = raw["Adj Close"].copy()
        elif "Close" in top:  # fallback por si algún ticker vino sin Adj Close
            data = raw["Close"].copy()
        else:
            # busca cualquier nivel que contenga 'close' (por seguridad)
            candidates = [lvl for lvl in sorted(set(top)) if "close" in lvl.lower()]
            if not candidates:
                raise KeyError("Yahoo response has no 'Adj Close' or 'Close' levels.")
            data = raw[candidates[0]].copy()

    # Caso 2: columnas de un solo nivel
    else:
        cols = [c.lower() for c in map(str, raw.columns)]
        if "adj close" in cols:
            data = raw.loc[:, raw.columns[cols.index("adj close")]].to_frame()
        elif "close" in cols:
            data = raw.loc[:, raw.columns[cols.index("close")]].to_frame()
        else:
            # A veces (auto_adjust=True) devuelve directamente los tickers como columnas
            if set(symbols).issubset(set(raw.columns)):
                data = raw[symbols].copy()
            else:
                # último intento: si es una Serie (un ticker)
                if isinstance(raw, pd.Series):
                    data = raw.to_frame(name=symbols[0])
                else:
                    raise KeyError("Could not locate 'Adj Close', 'Close', or ticker columns in Yahoo response.")

    # filtra a los símbolos que realmente llegaron
    keep = [s for s in symbols if s in data.columns]
    if not keep:
        raise ValueError("None of the requested symbols returned price data.")
    data = data[keep]

    # limpieza estándar
    data = data.sort_index().dropna(how="all")
    return data


try:
    prices_daily = fetch_prices_yahoo(all_series, start_date, end_date)
    # Sampling to monthly
    if sampling_choice.startswith("Monthly"):
        if sampling_choice == "Monthly (first trading day)":
            # “First business day that has data” per month
            # Group by year-month and pick first index
            prices = prices_daily.groupby([prices_daily.index.year, prices_daily.index.month]).head(1)
        elif sampling_choice == "Monthly (month end)":
            prices = prices_daily.resample("M").last().dropna(how="all")
        else:
            # Monthly custom day index (0=first valid)
            def pick_kth(g):
                g_sorted = g.sort_index()
                if custom_day < len(g_sorted):
                    return g_sorted.iloc[[custom_day]]
                return g_sorted.iloc[[-1]]  # fallback to last in that month if not enough days
            prices = prices_daily.groupby([prices_daily.index.year, prices_daily.index.month], group_keys=False).apply(pick_kth)
    else:
        prices = prices_daily.copy()

    prices.sort_index(inplace=True)
    prices = prices.dropna(axis=1, how="all")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Prices.tail(10)")
        st.dataframe(prices.tail(10), height=320)
    with right:
        st.subheader("Symbols actually present")
        st.write(list(prices.columns))

    # CSV download
    st.download_button(
        "⬇️ Download prices (CSV)",
        data=to_csv_bytes(prices),
        file_name="prices_monthly.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"Data download error: {e}")
    st.stop()

# --------------------------- #
# Step 3 — Returns math
# --------------------------- #
st.markdown("## ③ Return formulas")
st.latex(r"""
\textbf{Asset return (monthly)}\quad
R_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1
""")
st.latex(r"""
\textbf{Excess return (monthly)}\quad
XR_{i,t} = R_{i,t} - r_{f,t}
""")

returns = prices.pct_change().dropna()
st.caption("We use percentage change of sampled prices; first row is dropped to form valid monthly returns.")

# --------------------------- #
# Step 4 — Risk-free (FRED DGS1MO) & Excess returns
# --------------------------- #
st.markdown("## ④ Returns and risk-free (excess returns)")

@st.cache_data(show_spinner=False)
def fetch_dgs1mo_monthly(start, end):
    """Fetch FRED DGS1MO (1-Month T-Bill, % p.a.), return as monthly decimal rate aligned to returns index."""
    try:
        rf = pdr.DataReader("DGS1MO", "fred", start, end).dropna()
        # Convert % to decimal (annual), then to monthly decimal
        rf_annual = rf["DGS1MO"] / 100.0
        # Convert to end-of-month to align, then forward-fill to month
        rf_annual = rf_annual.resample("M").last().ffill()
        rf_monthly = (rf_annual / 12.0)
        rf_monthly.index = rf_monthly.index.tz_localize(None)
        return rf_monthly.rename("RF_1M")
    except Exception:
        return None

rf_m = fetch_dgs1mo_monthly(start_date, end_date)
if rf_m is None or rf_m.empty:
    st.warning("FRED DGS1MO unavailable. Using fallback annual slider converted to monthly.")
    rf_m = pd.Series(rf_fallback/12.0, index=returns.index).rename("RF_1M")
else:
    # align to returns index (month end stamps vs our monthly pick)
    rf_m = rf_m.reindex(returns.index, method="ffill").rename("RF_1M")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Returns.tail(10)")
    st.dataframe(returns.tail(10), height=320)
with right:
    st.subheader("Risk-free (FRED DGS1MO → monthly decimal) tail")
    st.dataframe(rf_m.to_frame().tail(10), height=320)

# Build excess returns: subtract rf from each column (row-wise)
excess_returns = returns.sub(rf_m, axis=0)

st.download_button(
    "⬇️ Download returns (CSV)",
    data=to_csv_bytes(returns),
    file_name="returns_monthly.csv",
    mime="text/csv",
)
st.download_button(
    "⬇️ Download excess returns (CSV)",
    data=to_csv_bytes(excess_returns),
    file_name="excess_returns_monthly.csv",
    mime="text/csv",
)

# --------------------------- #
# Step 5 — Asset statistics
# --------------------------- #
st.markdown("## ⑤ Asset statistics (monthly)")
mu = returns.mean()
var = returns.var()
std = returns.std()
# Sharpe using excess returns mean / std of returns
sharpe = excess_returns.mean() / returns.std()

stats = pd.DataFrame({
    "Expected Return μ": mu,
    "Variance": var,
    "Std Dev σ": std,
    "Sharpe (using RF)": sharpe
}).reindex(returns.columns)

st.dataframe(stats.style.format({"Expected Return μ":"{:.5f}", "Variance":"{:.5f}", "Std Dev σ":"{:.5f}", "Sharpe (using RF)":"{:.3f}"}), height=320)

st.download_button(
    "⬇️ Download asset statistics (CSV)",
    data=to_csv_bytes(stats),
    file_name="asset_stats_monthly.csv",
    mime="text/csv",
)

# --------------------------- #
# Step 6 — Covariance matrix
# --------------------------- #
st.markdown("## ⑥ Covariance matrix")
cov_m = returns.cov()
cov_a = cov_m * 12.0

c1, c2 = st.columns(2)
with c1:
    st.subheader("Covariance (monthly)")
    st.dataframe(cov_m.style.format("{:.4f}"), height=360)
with c2:
    st.subheader("Covariance (annualized)")
    st.dataframe(cov_a.style.format("{:.4f}"), height=360)

# Heatmaps for visual appeal
h1, h2 = st.columns(2)
with h1:
    st.plotly_chart(px.imshow(cov_m, text_auto=".3f", aspect="auto",
                              title="Covariance (monthly) — heatmap"), use_container_width=True)
with h2:
    st.plotly_chart(px.imshow(cov_a, text_auto=".3f", aspect="auto",
                              title="Covariance (annualized) — heatmap"), use_container_width=True)

st.download_button(
    "⬇️ Download covariance (monthly, CSV)",
    data=to_csv_bytes(cov_m),
    file_name="covariance_monthly.csv",
    mime="text/csv",
)
st.download_button(
    "⬇️ Download covariance (annualized, CSV)",
    data=to_csv_bytes(cov_a),
    file_name="covariance_annualized.csv",
    mime="text/csv",
)

# --------------------------- #
# Step 7 — Markowitz optimization
# --------------------------- #
st.markdown("## ⑦ Optimization & efficient frontier")

assets = returns.columns.tolist()
mu_vec = mu.values
cov_mat = cov_m.values
n = len(assets)

def portfolio_perf(w):
    w = np.array(w)
    mu_p = float(np.dot(w, mu_vec))
    var_p = float(np.dot(w, cov_mat @ w))
    std_p = math.sqrt(var_p)
    # Sharpe with mean monthly rf:
    rf_bar = float(rf_m.mean())
    sharpe_p = (mu_p - rf_bar) / std_p if std_p > 0 else np.nan
    return mu_p, std_p, var_p, sharpe_p

def max_sharpe():
    rf_bar = float(rf_m.mean())
    def neg_sharpe(w):
        mu_p, std_p, _, _ = portfolio_perf(w)
        return -(mu_p - rf_bar) / (std_p + 1e-12)
    constraints = [{"type":"eq","fun":lambda w: np.sum(w)-1}]
    bounds = [(0,1)]*n if long_only else [(-1,1)]*n
    w0 = np.ones(n)/n
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x

def min_variance():
    def port_var(w): return float(np.dot(w, cov_mat @ w))
    constraints = [{"type":"eq","fun":lambda w: np.sum(w)-1}]
    bounds = [(0,1)]*n if long_only else [(-1,1)]*n
    w0 = np.ones(n)/n
    res = minimize(port_var, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x

# Efficient frontier (target returns grid)
def efficient_frontier(points=100):
    mu_min, mu_max = mu_vec.min(), mu_vec.max()
    target_grid = np.linspace(mu_min*0.9, mu_max*1.05, points)
    ws, mus, stds = [], [], []
    for target in target_grid:
        constraints = [
            {"type":"eq","fun":lambda w: np.sum(w)-1},
            {"type":"eq","fun":lambda w, t=target: np.dot(w, mu_vec)-t},
        ]
        bounds = [(0,1)]*n if long_only else [(-1,1)]*n
        w0 = np.ones(n)/n
        res = minimize(lambda w: np.dot(w, cov_mat @ w), w0, method="SLSQP",
                       bounds=bounds, constraints=constraints)
        if res.success:
            ws.append(res.x)
            mu_p, std_p, _, _ = portfolio_perf(res.x)
            mus.append(mu_p)
            stds.append(std_p)
    return np.array(ws), np.array(mus), np.array(stds)

w_ms = max_sharpe()
w_mv = min_variance()
mu_ms, sd_ms, _, sh_ms = portfolio_perf(w_ms)
mu_mv, sd_mv, _, sh_mv = portfolio_perf(w_mv)

W, MU, SD = efficient_frontier(frontier_points)

# Frontier plot with asset points
fig = go.Figure()
fig.add_trace(go.Scatter(x=SD, y=MU, mode="lines", name="Frontier",
                         line=dict(color="#2E6BE6", width=2)))
# Assets as points
fig.add_trace(go.Scatter(
    x=returns.std().values, y=mu.values, mode="markers+text",
    text=assets, textposition="top center", name="Assets",
    marker=dict(size=9, color="#7E57C2")
))
# Max-Sharpe
fig.add_trace(go.Scatter(
    x=[sd_ms], y=[mu_ms], mode="markers", name="Max-Sharpe",
    marker=dict(size=12, color="#FF7043")
))
# Min-Var
fig.add_trace(go.Scatter(
    x=[sd_mv], y=[mu_mv], mode="markers", name="Min-Var",
    marker=dict(size=12, color="#26A69A")
))
fig.update_layout(
    title="Efficient frontier",
    xaxis_title="Risk (σ, per period — monthly)",
    yaxis_title="Expected return (μ, per period — monthly)",
    legend=dict(orientation="v"),
    height=420
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------- #
# Step 8 — Portfolio weights & summaries
# --------------------------- #
st.markdown("## ⑧ Portfolio definitions (weights & performance)")

def weights_df(w, label):
    df = pd.DataFrame({"Ticker": assets, "Weight": w})
    df["Weight_%"] = df["Weight"]*100
    df["Portfolio"] = label
    return df

df_ms = weights_df(w_ms, "Max-Sharpe")
df_mv = weights_df(w_mv, "Min-Variance")

left, right = st.columns(2)
with left:
    st.subheader("Max-Sharpe weights")
    st.bar_chart(df_ms.set_index("Ticker")["Weight"], use_container_width=True)
    st.markdown(nice_weights_text(w_ms, assets, "Allocation (Max-Sharpe)"))
    ms_perf = pd.DataFrame({
        "μ (monthly)": [mu_ms],
        "σ (monthly)": [sd_ms],
        "Annualized μ": [mu_ms*12],
        "Annualized σ": [sd_ms*np.sqrt(12)],
        "Sharpe": [sh_ms]
    }).T.rename(columns={0:"Value"})
    st.table(ms_perf.style.format("{:.4f}"))

with right:
    st.subheader("Min-Variance weights")
    st.bar_chart(df_mv.set_index("Ticker")["Weight"], use_container_width=True)
    st.markdown(nice_weights_text(w_mv, assets, "Allocation (Min-Variance)"))
    mv_perf = pd.DataFrame({
        "μ (monthly)": [mu_mv],
        "σ (monthly)": [sd_mv],
        "Annualized μ": [mu_mv*12],
        "Annualized σ": [sd_mv*np.sqrt(12)],
        "Sharpe": [sh_mv]
    }).T.rename(columns={0:"Value"})
    st.table(mv_perf.style.format("{:.4f}"))

# --------------------------- #
# Step 9 — Report downloads
# --------------------------- #
st.markdown("## ⑨ Downloadable report")
# Combine portfolio summary into a single CSV
report = {
    "Parameters": {
        "Investable tickers": ";".join(investable),
        "Benchmark tickers": ";".join(benchmark),
        "Start": str(start_date), "End": str(end_date),
        "Sampling": sampling_choice if custom_day is None else f"{sampling_choice} (index={custom_day})",
        "Long-only": long_only, "Frontier points": frontier_points
    },
    "Max-Sharpe": {
        "mu_monthly": mu_ms, "sd_monthly": sd_ms, "sharpe": sh_ms,
        "mu_annual": mu_ms*12, "sd_annual": sd_ms*np.sqrt(12)
    },
    "Min-Variance": {
        "mu_monthly": mu_mv, "sd_monthly": sd_mv, "sharpe": sh_mv,
        "mu_annual": mu_mv*12, "sd_annual": sd_mv*np.sqrt(12)
    }
}
report_df = pd.DataFrame(report).round(6)

zip_buf = io.BytesIO()
# We’ll write a few CSVs into a simple “bundle” as bytes concatenation (works fine for quick export)
bundle = {
    "prices_monthly.csv": prices,
    "returns_monthly.csv": returns,
    "excess_returns_monthly.csv": excess_returns,
    "covariance_monthly.csv": cov_m,
    "covariance_annualized.csv": cov_a,
    "weights_max_sharpe.csv": df_ms[["Ticker","Weight","Weight_%"]],
    "weights_min_variance.csv": df_mv[["Ticker","Weight","Weight_%"]],
    "portfolio_report.csv": report_df
}
# Build a single text blob (simple multi-file approach without zipfile dependency)
concat_text = []
for name, df in bundle.items():
    concat_text.append(f"### {name}\n")
    concat_text.append(df.to_csv(index=True))
    concat_text.append("\n\n")
zip_bytes = "\n".join(concat_text).encode("utf-8")

st.download_button(
    "⬇️ Download full report bundle (multi-CSV text)",
    data=zip_bytes,
    file_name="markowitz_report_bundle.txt",
    mime="text/plain",
)

st.success("Done. Scroll up to review each step; adjust inputs on the left and the app will recompute.")
