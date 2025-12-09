# app.py
# Main Streamlit app for the AD717 term project – Investment Analysis.
# I mirror the professor's Excel structure: each "sheet" in the instructions
# becomes a separate section in this dashboard.

from __future__ import annotations

from typing import List
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from fundamentals_helpers import fetch_yahoo_fundamentals
from config import FF_FILE, PRICES_FILE
from io import BytesIO
import yfinance as yf
from data_loading import load_ff_factors, load_price_data
from returns_helpers import (
    compute_simple_returns,
    align_returns_and_factors,
    compute_excess_returns,
)
from regressions import (
    run_capm_regressions_panel,
    capm_results_to_frame,
)

# --------------------------------------------------------------
# Streamlit basic configuration
# --------------------------------------------------------------
st.set_page_config(
    page_title="Term Project – Investment Analysis by Ethan Verduzco",
    layout="wide",
)

# --------------------------------------------------------------
# Suggested tickers by investor profile
# --------------------------------------------------------------
PROFILE_DEFAULT_TICKERS = {
    # Quality / relatively defensive tilt
    "Kim (25, early-career, stable high-quality stocks)": [
        "ADC",   # Agree Realty (net-lease REIT, defensive)
        "EXPO",  # Exponent INC.
        "WTS",   # Watts Water Technologies (water infrastructure)
        "AFG",   # American Financial Group (insurance)
        "AGCO",  # AGCO Corp (agricultural machinery)
        "ACHC",  # Acadia Healthcare (behavioral health)
    ]
}

# --------------------------------------------------------------
# Main app
# --------------------------------------------------------------
def main() -> None:
    st.title("Term Project – Investment Analysis (S&P 400, CAPM, Fama–French & Markowitz) by Ethan Verduzco")

    st.markdown(
        """
I replicate the Excel-based assignment using Python + Streamlit.

Instead of multiple Excel sheets, I organize the work into **sections** that
mirror the structure of the project:

1. Prices of the S&P 400 stocks I downloaded.  
2. Fama–French factor data (including the risk-free rate).  
3. Returns and excess returns for the six stocks in my portfolio.  
4. CAPM regressions using excess returns and the market excess return.

Later sections will cover 3-factor regressions, portfolio construction,
and portfolio summary.
        """
    )

    # ----------------------------------------------------------
    # Data loading
    # ----------------------------------------------------------
    with st.expander("Data loading – S&P 400 prices & Fama–French factors", expanded=True):
        st.write("I start by loading the monthly prices and the Fama–French factors.")

        try:
            prices = load_price_data(PRICES_FILE)
            st.success(f"Loaded price data with shape {prices.shape}.")
        except Exception as e:
            st.error(f"Error loading price data: {e}")
            st.stop()

        try:
            ff = load_ff_factors(FF_FILE)
            st.success(f"Loaded Fama–French factors with shape {ff.shape}.")
        except Exception as e:
            st.error(f"Error loading Fama–French factors: {e}")
            st.stop()

        st.markdown("**Raw S&P 400 price data (first 5 rows):**")
        st.dataframe(prices.head())

        st.markdown("**Raw Fama–French data (first 5 rows):**")
        st.dataframe(ff.head())

    # ------------------------------------------------------------
    # Sidebar: fixed investor profile, stock selection, section menu
    # ------------------------------------------------------------

    st.sidebar.header("Global configuration")

    # Set fixed investor profile (no user selection)
    INVESTOR_PROFILE_FIXED = "Kim (25, early-career, stable high-quality stocks)"
    st.sidebar.markdown(
        f"**Investor profile:**  \n{INVESTOR_PROFILE_FIXED}"
    )

    # Load tickers for the fixed profile
    investor_profile = INVESTOR_PROFILE_FIXED
    default_tickers = PROFILE_DEFAULT_TICKERS[investor_profile]

    all_tickers: List[str] = list(prices.columns)


    # Suggested defaults for the chosen profile, only if they exist in the data
    suggested = PROFILE_DEFAULT_TICKERS.get(investor_profile, [])
    suggested = [t for t in suggested if t in all_tickers]

    if len(suggested) != 6:
        # Fallback: simply take the first six tickers if something looks off
        suggested = all_tickers[:6]

    st.sidebar.markdown(
        "##### Suggested 6-stock portfolio for this profile\n"
        f"`{', '.join(suggested)}`"
    )

    selected_stocks = st.sidebar.multiselect(
        "Select exactly 6 stocks (tickers)",
        options=all_tickers,
        default=suggested,
        max_selections=6,
    )

    if len(selected_stocks) != 6:
        st.sidebar.error("I need exactly 6 stocks selected.")
        st.stop()

    detailed_stock_sidebar = st.sidebar.selectbox(
        "Stock for single-name analysis (used in later sections)",
        options=selected_stocks,
    )

    section = st.sidebar.radio(
        "Excel-style sections",
        options=[
            "Sheet 1 – Prices (raw data)",
            "Sheet 2 – Fama–French factors",
            "Sheet 3 – Returns & excess returns",
            "Sheet 4 – CAPM regressions",
            "Sheet 5 – Portfolio Construction (Markowitz)",
            "Sheet 6 – 3-Factor model regressions",
            "Sheet 7 – Portfolio summary",
            "Sheet 8 – Report analytics",
        ],
        index=0,
    )

    # ----------------------------------------------------------
    # Core computations used by multiple sections
    # ----------------------------------------------------------
    # 1) Monthly simple returns for all stocks
    returns_all = compute_simple_returns(prices)
    returns_sel = returns_all[selected_stocks].copy()

    # 2) Align the selected returns with Fama–French data
    aligned_returns, aligned_ff = align_returns_and_factors(returns_sel, ff)

    # 3) Excess returns for the selected stocks, using RF from Fama–French
    excess_returns = compute_excess_returns(aligned_returns, aligned_ff["RF"])

    # 4) CAPM regressions (using excess returns and Mkt-RF)
    capm_results = run_capm_regressions_panel(
        excess_returns_df=excess_returns,
        ff_factors_aligned=aligned_ff,
    )
    capm_panel = capm_results_to_frame(capm_results)

    # ==========================================================
    # SHEET 1 – Prices
    # ==========================================================
    if section == "Sheet 1 – Prices (raw data)":
        st.header("Sheet 1 – Prices I have downloaded")

        st.markdown(
            """
This section corresponds to the first sheet of the Excel notebook:
it shows the prices I downloaded and highlights the six names in my portfolio.
            """
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Full price panel (all stocks)")
            st.write(
                "These are the monthly closing prices (or adjusted prices) "
                "for all stocks in the S&P 400 universe included in my file."
            )
            st.dataframe(prices)

        with col2:
            st.markdown("### Price history for the 6 selected stocks")
            st.write(
                "Here I subset the panel to only the six stocks in my portfolio, "
                "which I will use in all subsequent calculations."
            )
            st.dataframe(prices[selected_stocks])

            csv_prices_sel = prices[selected_stocks].reset_index().to_csv(index=False)
            st.download_button(
                label="Download Sheet 1 (prices for 6 stocks) as CSV",
                data=csv_prices_sel,
                file_name="sheet1_prices_portfolio.csv",
                mime="text/csv",
            )

        st.markdown(
            f"""
In the Word document, I will describe why each of these six names fits
the investor profile **{investor_profile}**, focusing on business model,
risk level, and role inside the overall portfolio.
            """
        )

    # ==========================================================
    # SHEET 2 – Fama–French factors
    # ==========================================================
    elif section == "Sheet 2 – Fama–French factors":
        st.header("Sheet 2 – Fama–French data (including risk-free rate)")

        st.markdown(
            """
This section corresponds to the second sheet of the Excel notebook:
it contains the Fama–French 3-factor data and the risk-free rate
aligned to the same monthly dates as my stock returns.
            """
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Full Fama–French 3-factor panel (aligned monthly data)")
            st.dataframe(aligned_ff[["Mkt-RF", "SMB", "HML", "RF"]])

        with col2:
            st.markdown("### Overlap with my price data")
            st.write(
                f"After aligning with the S&P 400 data I have "
                f"**{aligned_ff.shape[0]}** monthly observations."
            )

            csv_ff = aligned_ff.reset_index().to_csv(index=False)
            st.download_button(
                label="Download Sheet 2 (aligned Fama–French data) as CSV",
                data=csv_ff,
                file_name="sheet2_fama_french_aligned.csv",
                mime="text/csv",
            )

        st.markdown(
            """
I use **RF** to compute excess returns, and all three factors
(**Mkt-RF**, **SMB**, **HML**) when I estimate Fama–French regressions later.
            """
        )

    # ==========================================================
    # SHEET 3 – Returns & excess returns
    # ==========================================================
    elif section == "Sheet 3 – Returns & excess returns":
        st.header("Sheet 3 – Returns and excess returns of the stocks in my portfolio")

        st.markdown(
            """
This section corresponds to the third sheet of the Excel notebook.
Here I compute monthly returns and excess returns for the six stocks
in my portfolio using the prices from Sheet 1 and the risk-free rate
from Sheet 2.
            """
        )

        # ------------------------------------------------------
        # 3.0 Step-by-step calculation walkthrough (one stock)
        # ------------------------------------------------------
        st.markdown("### 3.0 Step-by-step calculation walkthrough (one example stock)")

        example_ticker = st.selectbox(
            "Example stock for the step-by-step walkthrough",
            options=selected_stocks,
            index=0,
        )

        example_index = aligned_returns.index

        example_df = pd.DataFrame(index=example_index)
        example_df["Price_t_minus_1"] = prices[example_ticker].reindex(example_index).shift(1)
        example_df["Price_t"] = prices[example_ticker].reindex(example_index)
        example_df["Return_t"] = aligned_returns[example_ticker]
        example_df["RF_t"] = aligned_ff["RF"]
        example_df["Excess_t"] = excess_returns[example_ticker]

        example_df = example_df.dropna().head(8)

        col_step_table, col_step_text = st.columns([1.6, 1.4])

        with col_step_table:
            st.markdown(f"**Worked example for stock: `{example_ticker}`**")
            st.dataframe(
                example_df.reset_index().rename(columns={"Date": "Month"}).style.format(
                    {
                        "Price_t_minus_1": "{:.4f}",
                        "Price_t": "{:.4f}",
                        "Return_t": "{:.4f}",
                        "RF_t": "{:.4f}",
                        "Excess_t": "{:.4f}",
                    }
                )
            )

        with col_step_text:
            st.markdown(f"#### How I compute and interpret returns for `{example_ticker}`")

            # Step 1 – from prices to simple return
            st.markdown("**Step 1 – From prices to simple monthly return**")
            st.markdown(
                r"""
For each month $t$:

1. I take $P_{i,t-1}$, the price of stock $i$ at the **end of the previous month**  
   (this is the value shown in the column **Price_t_minus_1**).  
2. I take $P_{i,t}$, the price of stock $i$ at the **end of the current month**  
   (column **Price_t**).  
3. I plug both into the simple-return formula:
                """
            )

            st.latex(r"R_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1")

            st.markdown(
                r"""
The resulting $R_{i,t}$ is stored in the column **Return_t** and represents the
**percentage gain or loss over that month**, measured relative to the price at
the end of the previous month.
                """
            )

            # Step 2 – from return to excess return
            st.markdown("**Step 2 – From return to excess return**")
            st.markdown(
                r"""
For the same month $t$:

1. I take $R_{f,t}$, the **monthly risk-free rate** from the Fama–French file  
   (shown in the column **RF_t**).  
2. I subtract it from the stock return to obtain the **excess return**:
                """
            )

            st.latex(r"R_{i,t}^{excess} = R_{i,t} - R_{f,t}")

            st.markdown(
                r"""
The excess return $R_{i,t}^{excess}$ (column **Excess_t**) measures how much the
stock **outperformed (or underperformed) a risk-free investment** in that month.
                """
            )

        st.markdown("---")

        # ------------------------------------------------------
        # 3.1 Full tables for all six stocks + pipeline text
        # ------------------------------------------------------
        col_pipeline, col_tables = st.columns([1.2, 1.8])

        with col_pipeline:
            st.markdown("### Calculation pipeline (how this replicates Excel)")

            st.markdown(
                r"""
1. **Start from Sheet 1 (prices).**  
   For each stock $i$ and month $t$ I have the end-of-month price $P_{i,t}$.  

2. **Create the lagged price.**  
   I shift the price series by one row to obtain $P_{i,t-1}$, the price at the
   end of the previous month.  

3. **Compute the monthly simple return.**  
   Using the standard simple-return formula:
                """
            )

            st.latex(r"R_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1")

            st.markdown(
                r"""
   I obtain the monthly percentage return for each stock, which fills the
   **Return_t** columns.  

4. **Bring in the risk-free rate from Sheet 2.**  
   From the Fama–French panel I take $R_{f,t}$, the monthly T-bill return (RF),
   properly aligned by date.  

5. **Compute the excess return.**  
   For each stock I subtract the risk-free rate:
                """
            )

            st.latex(r"R_{i,t}^{excess} = R_{i,t} - R_{f,t}")

            st.markdown(
                r"""
   This fills the **Excess_t** columns and matches the textbook definition
   of excess return.  

These are exactly the intermediate steps an auditor could reproduce in Excel
to verify every number in this sheet.
                """
            )

        with col_tables:
            st.markdown("### 3.1 Monthly returns for the 6 selected stocks")
            st.dataframe(aligned_returns)

            st.markdown("### 3.2 Risk-free rate (RF) from Fama–French")
            st.dataframe(aligned_ff[["RF"]])

            st.markdown("### 3.3 Excess returns (stock return minus RF)")
            st.dataframe(excess_returns)

            csv_returns = aligned_returns.reset_index().to_csv(index=False)
            csv_excess = excess_returns.reset_index().to_csv(index=False)

            st.download_button(
                label="Download Sheet 3 – returns (6 stocks) as CSV",
                data=csv_returns,
                file_name="sheet3_returns_portfolio.csv",
                mime="text/csv",
            )

            st.download_button(
                label="Download Sheet 3 – excess returns (6 stocks) as CSV",
                data=csv_excess,
                file_name="sheet3_excess_returns_portfolio.csv",
                mime="text/csv",
            )

        st.markdown("### Summary statistics (monthly) for the 6 stocks")

        summary = pd.DataFrame(
            {
                "mean_return": aligned_returns.mean(),
                "std_return": aligned_returns.std(ddof=1),
            }
        )
        summary["mean_excess"] = excess_returns.mean()

        st.dataframe(summary.style.format("{:.4f}"))

        st.markdown(
            """
These summary statistics will later feed into the CAPM and Fama–French
regressions, as well as the portfolio construction logic (weights,
expected return, and portfolio risk).
            """
        )

        # ------------------------------------------------------
        # Variable glossary (returns & excess returns)
        # ------------------------------------------------------
        with st.expander("Variable glossary for this sheet", expanded=False):
            st.markdown(
                r"""
- $P_{i,t}$ – End-of-month price of stock $i$ in month $t$,  
  stored in the column **Price_t**.  

- $P_{i,t-1}$ – End-of-month price of stock $i$ in the **previous** month,  
  stored in the column **Price_t_minus_1** (obtained by lagging $P_{i,t}$ by one row).  

- $R_{i,t}$ – **Simple monthly return** of stock $i$ in month $t$, computed as  
  $R_{i,t} = \dfrac{P_{i,t}}{P_{i,t-1}} - 1$ and stored in the column **Return_t**.  

- $R_{f,t}$ – **Monthly risk-free rate** in month $t$ from the Fama–French data  
  (T-bill proxy, stored in the column **RF_t**).  

- $R_{i,t}^{excess}$ – **Excess return** of stock $i$ over the risk-free rate,  
  $R_{i,t}^{excess} = R_{i,t} - R_{f,t}$, stored in the column **Excess_t**.
                """
            )

            # ==========================================================
    # SHEET 4 – CAPM regressions
    # ==========================================================
    elif section == "Sheet 4 – CAPM regressions":
        import statsmodels.api as sm
        from typing import List

        st.header("Sheet 4 – CAPM regressions, beta and alpha")

        # ------------------------------------------------------
        # 4.0 CAPM core formulas (LaTeX only)
        # ------------------------------------------------------
        st.markdown("### 4.0 CAPM core formulas")

        # (1) Excess returns
        st.latex(r"R_{i,t}^{\text{excess}} = R_{i,t} - R_{f,t}")
        st.latex(r"R_{M,t}^{\text{excess}} = R_{M,t} - R_{f,t}")

        # (2) CAPM regression equation
        st.latex(
            r"R_{i,t}^{\text{excess}}"
            r" = \alpha_i + \beta_i R_{M,t}^{\text{excess}} + \varepsilon_t"
        )

        # (3) Beta manual (cov/var)
        st.latex(
            r"\beta_i = "
            r"\dfrac{\operatorname{Cov}\!\left(R_i^{\text{excess}},"
            r" R_M^{\text{excess}}\right)}"
            r"{\operatorname{Var}\!\left(R_M^{\text{excess}}\right)}"
        )

        # (4) Historical alpha in excess returns (manual)
        st.latex(
            r"\alpha_i^{\text{hist, excess}} = "
            r"\bar R_i^{\text{excess}} - \beta_i \,\bar R_M^{\text{excess}}"
        )

        # (5) CAPM expected return in levels
        st.latex(
            r"E[R_i]_{\text{CAPM}} = "
            r"R_f + \beta_i \big(E[R_M] - R_f\big)"
        )

        # (6) Alpha forecast (assignment definition)
        st.latex(
            r"\widehat{\alpha}_i = \bar R_i - E[R_i]_{\text{CAPM}}"
        )

        # Market series from Fama–French (already aligned)
        mkt_excess = aligned_ff["Mkt-RF"]
        rf_series = aligned_ff["RF"]

        st.markdown("---")

        # ------------------------------------------------------
        # 4.1 Inputs: excess returns for one stock and the market
        # ------------------------------------------------------
        st.markdown("### 4.1 Inputs: excess return data for one stock and the market")

        detailed_stock = st.selectbox(
            "Stock for step-by-step CAPM illustration",
            options=selected_stocks,
            index=0,
        )

        Ri_excess_det = excess_returns[detailed_stock].dropna()
        common_idx = Ri_excess_det.index.intersection(mkt_excess.index)

        Ri_excess_det = Ri_excess_det.loc[common_idx]
        Rm_excess_det = mkt_excess.loc[common_idx]

        capm_df = pd.DataFrame(
            {
                "Ri_excess": Ri_excess_det,
                "Rm_excess": Rm_excess_det,
            }
        ).dropna()

        st.markdown(
            f"**Excess returns used in the CAPM regression for `{detailed_stock}`:**"
        )
        st.dataframe(capm_df)

        csv_capm_inputs = capm_df.reset_index().to_csv(index=False)
        st.download_button(
            label=f"Download 4.1 – CAPM inputs for {detailed_stock} (CSV)",
            data=csv_capm_inputs,
            file_name=f"sheet4_capm_inputs_{detailed_stock}.csv",
            mime="text/csv",
        )

        st.latex(
            r"\text{Data used: }"
            r"\left\{ R_{i,t}^{\text{excess}},\, R_{M,t}^{\text{excess}} \right\}_{t=1}^{T}"
        )

        st.markdown("---")

        # ------------------------------------------------------
        # 4.2 Covariance and market variance (manual inputs)
        # ------------------------------------------------------
        st.markdown("### 4.2 Manual covariance and market variance for the selected stock")

        n_obs = capm_df.shape[0]
        Ri_bar_det = capm_df["Ri_excess"].mean()
        RM_bar_det = capm_df["Rm_excess"].mean()

        cov_Ri_RM_det = capm_df["Ri_excess"].cov(capm_df["Rm_excess"])
        var_RM_det = capm_df["Rm_excess"].var()

        col_42_left, col_42_right = st.columns([1.8, 1.2])

        with col_42_left:
            st.latex(
                r"\operatorname{Cov}(R_i, R_M) = "
                r"\dfrac{1}{T-1}\sum_{t=1}^{T}"
                r"\big(R_{i,t}^{\text{excess}} - \bar R_i^{\text{excess}}\big)"
                r"\big(R_{M,t}^{\text{excess}} - \bar R_M^{\text{excess}}\big)"
            )
            st.latex(
                r"\operatorname{Var}(R_M) = "
                r"\dfrac{1}{T-1}\sum_{t=1}^{T}"
                r"\big(R_{M,t}^{\text{excess}} - \bar R_M^{\text{excess}}\big)^2"
            )

        with col_42_right:
            st.latex(rf"T = {n_obs}")
            st.latex(rf"\bar R_i^{{\text{{excess}}}} = {Ri_bar_det:.4f}")
            st.latex(rf"\bar R_M^{{\text{{excess}}}} = {RM_bar_det:.4f}")
            st.latex(
                rf"\widehat{{\operatorname{{Cov}}}}\!(R_i, R_M) = {cov_Ri_RM_det:.6f}"
            )
            st.latex(
                rf"\widehat{{\operatorname{{Var}}}}(R_M) = {var_RM_det:.6f}"
            )

        st.markdown("---")

        # ------------------------------------------------------
        # 4.3 Version A – Manual CAPM beta and alpha (selected stock)
        # ------------------------------------------------------
        st.markdown("### 4.3 Version A – Manual CAPM beta and alpha (selected stock)")

        beta_manual_det = cov_Ri_RM_det / var_RM_det
        alpha_hist_excess_det = Ri_bar_det - beta_manual_det * RM_bar_det

        col_43_left, col_43_right = st.columns([1.8, 1.2])

        with col_43_left:
            st.latex(
                r"\beta_i = "
                r"\dfrac{\operatorname{Cov}(R_i^{\text{excess}}, R_M^{\text{excess}})}"
                r"{\operatorname{Var}(R_M^{\text{excess}})}"
            )
            st.latex(
                r"\alpha_i^{\text{hist, excess}} = "
                r"\bar R_i^{\text{excess}} - \beta_i \,\bar R_M^{\text{excess}}"
            )

        with col_43_right:
            st.markdown(f"**Manual CAPM results for `{detailed_stock}`:**")
            st.latex(rf"\beta_i^{{\text{{manual}}}} = {beta_manual_det:.4f}")
            st.latex(
                rf"\alpha_i^{{\text{{hist, excess}}}} = {alpha_hist_excess_det:.4f}"
            )
            st.latex(rf"T = {n_obs}")

        st.markdown("---")

        # ------------------------------------------------------
        # 4.4 Version A – Manual CAPM for all six stocks
        # ------------------------------------------------------
        st.markdown("### 4.4 Version A – Manual CAPM summary for all six stocks")

        manual_rows: List[dict] = []
        for ticker in selected_stocks:
            ri_excess = excess_returns[ticker].dropna()
            common_idx_i = ri_excess.index.intersection(mkt_excess.index)
            ri_excess = ri_excess.loc[common_idx_i]
            rm_excess = mkt_excess.loc[common_idx_i]

            df_i = pd.DataFrame(
                {"Ri_excess": ri_excess, "Rm_excess": rm_excess}
            ).dropna()

            if df_i.empty:
                continue

            Ri_bar_i = df_i["Ri_excess"].mean()
            RM_bar_i = df_i["Rm_excess"].mean()

            cov_iM = df_i["Ri_excess"].cov(df_i["Rm_excess"])
            var_M_i = df_i["Rm_excess"].var()

            beta_i_manual = cov_iM / var_M_i
            alpha_hist_excess_i = Ri_bar_i - beta_i_manual * RM_bar_i

            manual_rows.append(
                {
                    "ticker": ticker,
                    "Ri_excess_bar": Ri_bar_i,
                    "RM_excess_bar": RM_bar_i,
                    "Cov_Ri_RM": cov_iM,
                    "Var_RM": var_M_i,
                    "beta_manual": beta_i_manual,
                    "alpha_hist_excess": alpha_hist_excess_i,
                }
            )

        capm_manual_summary = pd.DataFrame(manual_rows).set_index("ticker")

        st.dataframe(
            capm_manual_summary.style.format(
                {
                    "Ri_excess_bar": "{:.4f}",
                    "RM_excess_bar": "{:.4f}",
                    "Cov_Ri_RM": "{:.6f}",
                    "Var_RM": "{:.6f}",
                    "beta_manual": "{:.4f}",
                    "alpha_hist_excess": "{:.4f}",
                }
            )
        )

        csv_capm_manual = capm_manual_summary.reset_index().to_csv(index=False)
        st.download_button(
            label="Download 4.4 – manual CAPM summary (CSV)",
            data=csv_capm_manual,
            file_name="sheet4_capm_manual_summary.csv",
            mime="text/csv",
        )

        st.markdown("---")

        # ------------------------------------------------------
        # 4.5 Version B – CAPM regressions (OLS) for all six stocks
        # ------------------------------------------------------
        st.markdown("### 4.5 Version B – OLS CAPM regressions for all six stocks")

        st.latex(
            r"R_{i,t}^{\text{excess}}"
            r" = \alpha_i + \beta_i R_{M,t}^{\text{excess}} + \varepsilon_t"
        )
        st.latex(
            r"t(\widehat\beta_i) = "
            r"\dfrac{\widehat\beta_i}{\operatorname{SE}(\widehat\beta_i)}"
        )
        st.latex(
            r"t(\widehat\alpha_i) = "
            r"\dfrac{\widehat\alpha_i}{\operatorname{SE}(\widehat\alpha_i)}"
        )

        st.markdown(
            r"""
The t-statistics \(t(\widehat\beta_i)\) and \(t(\widehat\alpha_i)\) test the
null hypotheses \(H_0:\beta_i=0\) and \(H_0:\alpha_i=0\).  
The corresponding p-values \(\text{p\_value\_beta}\) and
\(\text{p\_value\_alpha}\) measure how likely it is to observe such extreme
t-statistics if those null hypotheses were true.
            """
        )

        ols_rows: List[dict] = []
        for ticker in selected_stocks:
            ri_excess = excess_returns[ticker].dropna()
            common_idx_i = ri_excess.index.intersection(mkt_excess.index)
            ri_excess = ri_excess.loc[common_idx_i]
            rm_excess = mkt_excess.loc[common_idx_i]

            df_i = pd.DataFrame(
                {"Ri_excess": ri_excess, "Rm_excess": rm_excess}
            ).dropna()

            if df_i.empty:
                continue

            X = sm.add_constant(df_i["Rm_excess"])
            y = df_i["Ri_excess"]
            model = sm.OLS(y, X).fit()

            ols_rows.append(
                {
                    "ticker": ticker,
                    "alpha_ols": model.params["const"],
                    "beta_ols": model.params["Rm_excess"],
                    "t_stat_alpha": model.tvalues["const"],
                    "t_stat_beta": model.tvalues["Rm_excess"],
                    "p_value_alpha": model.pvalues["const"],
                    "p_value_beta": model.pvalues["Rm_excess"],
                    "R2": model.rsquared,
                    "n_obs": int(model.nobs),
                }
            )

        ols_df = pd.DataFrame(ols_rows).set_index("ticker")

        st.markdown("**OLS CAPM regression results (all six stocks):**")
        st.dataframe(
            ols_df.style.format(
                {
                    "alpha_ols": "{:.4f}",
                    "beta_ols": "{:.4f}",
                    "t_stat_alpha": "{:.2f}",
                    "t_stat_beta": "{:.2f}",
                    "p_value_alpha": "{:.4f}",
                    "p_value_beta": "{:.4f}",
                    "R2": "{:.3f}",
                }
            )
        )

        csv_capm_ols = ols_df.reset_index().to_csv(index=False)
        st.download_button(
            label="Download 4.5 – OLS CAPM regression results (CSV)",
            data=csv_capm_ols,
            file_name="sheet4_capm_ols_results.csv",
            mime="text/csv",
        )

        # Detail for the selected stock
        if detailed_stock in ols_df.index:
            row_ols = ols_df.loc[detailed_stock]
            st.markdown(f"**OLS regression details for `{detailed_stock}`:**")
            st.latex(
                rf"\widehat\alpha_i^{{\text{{OLS}}}} = {row_ols['alpha_ols']:.4f}"
            )
            st.latex(
                rf"\widehat\beta_i^{{\text{{OLS}}}} = {row_ols['beta_ols']:.4f}"
            )
            st.latex(
                rf"t(\widehat\alpha_i) = {row_ols['t_stat_alpha']:.2f},"
                rf"\quad p\text{{-value}} = {row_ols['p_value_alpha']:.4f}"
            )
            st.latex(
                rf"t(\widehat\beta_i) = {row_ols['t_stat_beta']:.2f},"
                rf"\quad p\text{{-value}} = {row_ols['p_value_beta']:.4f}"
            )
            st.latex(
                rf"R^2 = {row_ols['R2']:.3f},"
                rf"\quad n = {int(row_ols['n_obs'])}"
            )

        st.markdown("---")

        # ------------------------------------------------------
        # 4.6 CAPM expected return and alpha forecast
        # ------------------------------------------------------
        st.markdown("### 4.6 CAPM expected return and alpha forecast")

        st.latex(
            r"E[R_i]_{\text{CAPM}} = R_f + \beta_i \big(E[R_M] - R_f\big)"
        )
        st.latex(
            r"\bar R_i = \dfrac{1}{T}\sum_{t=1}^{T} R_{i,t}"
        )
        st.latex(
            r"\widehat{\alpha}_i = \bar R_i - E[R_i]_{\text{CAPM}}"
        )

        st.markdown(
            r"""
The assignment defines the **alpha forecast** as the difference between the
expected return of the stock and the return predicted by the CAPM.  
Here I use the **sample average return in levels** \(\bar R_i\) as a proxy for
the expected return of each stock and subtract the **CAPM-implied expected
return** \(E[R_i]_{\text{CAPM}}\).
            """
        )

        # Market statistics in levels
        R_M_level = mkt_excess + rf_series
        E_RM_level = R_M_level.mean()
        R_f_bar = rf_series.mean()
        market_risk_premium = E_RM_level - R_f_bar

        forecast_rows: List[dict] = []
        for ticker in selected_stocks:
            # Beta from OLS if available, otherwise manual
            if ticker in ols_df.index:
                beta_i_capm = ols_df.loc[ticker, "beta_ols"]
            else:
                beta_i_capm = capm_manual_summary.loc[ticker, "beta_manual"]

            ri_excess_full = excess_returns[ticker].dropna()
            common_idx_i = ri_excess_full.index.intersection(rf_series.index)
            Ri_level = (
                ri_excess_full.loc[common_idx_i] + rf_series.loc[common_idx_i]
            )
            Ri_bar_level = Ri_level.mean()

            E_Ri_CAPM = R_f_bar + beta_i_capm * market_risk_premium
            alpha_forecast_i = Ri_bar_level - E_Ri_CAPM

            forecast_rows.append(
                {
                    "ticker": ticker,
                    "beta_capm": beta_i_capm,
                    "E_RM_level": E_RM_level,
                    "R_f_bar": R_f_bar,
                    "MRP": market_risk_premium,
                    "CAPM_expected_return": E_Ri_CAPM,
                    "Mean_realized_return": Ri_bar_level,
                    "Alpha_forecast": alpha_forecast_i,
                }
            )

        forecast_df = pd.DataFrame(forecast_rows).set_index("ticker")

        selected_for_forecast = st.selectbox(
            "Stock for detailed alpha forecast display",
            options=forecast_df.index.tolist(),
            index=0,
        )

        fr = forecast_df.loc[selected_for_forecast]

        st.markdown(f"**Detailed CAPM forecast for `{selected_for_forecast}`:**")
        st.latex(
            rf"\beta_{{{selected_for_forecast}}} = {fr['beta_capm']:.4f}"
        )
        st.latex(
            rf"E[R_M] = {fr['E_RM_level']:.4f},"
            rf"\quad \bar R_f = {fr['R_f_bar']:.4f},"
            rf"\quad E[R_M] - \bar R_f = {fr['MRP']:.4f}"
        )
        st.latex(
            rf"E[R_{{{selected_for_forecast}}}]_{{\text{{CAPM}}}}"
            rf" = {fr['CAPM_expected_return']:.4f}"
        )
        st.latex(
            rf"\bar R_{{{selected_for_forecast}}} = "
            rf"{fr['Mean_realized_return']:.4f}"
        )
        st.latex(
            rf"\widehat{{\alpha}}_{{{selected_for_forecast}}}"
            rf" = \bar R_{{{selected_for_forecast}}}"
            rf" - E[R_{{{selected_for_forecast}}}]_{{\text{{CAPM}}}}"
            rf" = {fr['Alpha_forecast']:.4f}"
        )

        st.markdown("**Alpha forecast table for all six stocks:**")
        st.dataframe(
            forecast_df.style.format(
                {
                    "beta_capm": "{:.4f}",
                    "E_RM_level": "{:.4f}",
                    "R_f_bar": "{:.4f}",
                    "MRP": "{:.4f}",
                    "CAPM_expected_return": "{:.4f}",
                    "Mean_realized_return": "{:.4f}",
                    "Alpha_forecast": "{:.4f}",
                }
            )
        )

        csv_forecast = forecast_df.reset_index().to_csv(index=False)
        st.download_button(
            label="Download 4.6 – CAPM alpha forecast (CSV)",
            data=csv_forecast,
            file_name="sheet4_capm_alpha_forecast.csv",
            mime="text/csv",
        )

        st.markdown("---")

        # ------------------------------------------------------
        # 4.7 Final CAPM summary table (everything together)
        # ------------------------------------------------------
        st.markdown("### 4.7 Final CAPM summary (manual vs OLS vs forecast)")

        summary_final = (
            capm_manual_summary.join(ols_df, how="left")
            .join(
                forecast_df[
                    ["CAPM_expected_return", "Mean_realized_return", "Alpha_forecast"]
                ],
                how="left",
            )
        )

        st.dataframe(
            summary_final.style.format(
                {
                    "Ri_excess_bar": "{:.4f}",
                    "RM_excess_bar": "{:.4f}",
                    "Cov_Ri_RM": "{:.6f}",
                    "Var_RM": "{:.6f}",
                    "beta_manual": "{:.4f}",
                    "alpha_hist_excess": "{:.4f}",
                    "alpha_ols": "{:.4f}",
                    "beta_ols": "{:.4f}",
                    "t_stat_alpha": "{:.2f}",
                    "t_stat_beta": "{:.2f}",
                    "p_value_alpha": "{:.4f}",
                    "p_value_beta": "{:.4f}",
                    "R2": "{:.3f}",
                    "CAPM_expected_return": "{:.4f}",
                    "Mean_realized_return": "{:.4f}",
                    "Alpha_forecast": "{:.4f}",
                }
            )
        )

        csv_final = summary_final.reset_index().to_csv(index=False)
        st.download_button(
            label="Download 4.7 – final CAPM summary (CSV)",
            data=csv_final,
            file_name="sheet4_capm_final_summary.csv",
            mime="text/csv",
        )

        st.markdown("---")
        
            # =====================================================================
        # 4.X CAPM summary table for all stocks, portfolio and benchmark
        # =====================================================================

        st.markdown("---")
        st.subheader("4.X CAPM summary – stocks, portfolio and benchmark")

        # ------------------------------------------------------------------
        # Helper: CAPM regression for a single return series
        # excess_y: asset excess return (r_i - r_f)
        # excess_m: market excess return (r_m - r_f)
        # ------------------------------------------------------------------
        def _capm_stats(excess_y: pd.Series, excess_m: pd.Series) -> tuple[float, float, float, float, float]:
            """
            Run CAPM regression: excess_y = alpha + beta * excess_m + eps.

            Returns:
                alpha (monthly),
                beta,
                residual variance,
                R²,
                total variance of excess_y.
            """
            data = pd.concat([excess_y, excess_m], axis=1).dropna()
            if data.empty:
                return 0.0, 0.0, 0.0, 0.0, 0.0

            y = data.iloc[:, 0]
            x = data.iloc[:, 1]
            X = sm.add_constant(x)

            model = sm.OLS(y, X).fit()

            alpha = float(model.params["const"])
            beta = float(model.params[x.name])
            resid_var = float(model.mse_resid)
            r2 = float(model.rsquared)
            total_var = float(y.var())

            return alpha, beta, resid_var, r2, total_var

        # ------------------------------------------------------------------
        # Helper functions for annualization & drawdown
        # (asumimos retornos mensuales)
        # ------------------------------------------------------------------
        def _ann_return(r: pd.Series) -> float:
            """Annualized arithmetic return from monthly simple returns."""
            r = r.dropna()
            if r.empty:
                return 0.0
            return (1 + r.mean()) ** 12 - 1

        import math

        def _ann_vol(r: pd.Series) -> float:
            """Annualized volatility from monthly simple returns (no numpy)."""
            r = r.dropna()
            if r.empty:
                return 0.0
            return r.std() * math.sqrt(12)


        def _max_drawdown(r: pd.Series) -> float:
            """Maximum drawdown from a series of simple returns."""
            r = r.dropna()
            if r.empty:
                return 0.0
            cum = (1 + r).cumprod()
            peak = cum.cummax()
            dd = cum / peak - 1
            return float(dd.min())

        # ------------------------------------------------------------------
        # Prepare basic inputs
        # ------------------------------------------------------------------
        # 1) Monthly simple returns of the 6 stocks (from Sheet 3)
        #    Aquí asumimos que 'returns' ya existe y tiene las columnas de tus 6 acciones.
        stock_rets = excess_returns[selected_stocks].dropna()

        # 2) Risk–free rate and market excess return from Fama–French
        #    (asumimos que 'aligned_ff' ya fue creado en este sheet o en uno anterior)
        mkt_excess = aligned_ff["Mkt-RF"]          # market excess return (r_m - r_f)
        rf_series = aligned_ff["RF"]              # risk-free, normalmente en %
        # Aseguramos que rf esté en formato decimal mensual
        if rf_series.max() > 1.0:
            rf_dec = rf_series / 100.0
        else:
            rf_dec = rf_series.copy()

        # 3) Igual ponderación para la cartera activa (se puede reemplazar por pesos de Markowitz)
        n_assets = len(selected_stocks)
        if n_assets == 0:
            st.warning("Please select six stocks in the sidebar to build the CAPM summary.")
            st.stop()

        equal_weights = [1.0 / n_assets] * n_assets
        weights = pd.Series(equal_weights, index=selected_stocks, name="weight")

        # 4) Portfolio & benchmark returns (simple monthly returns)
        port_rets = (stock_rets * weights).sum(axis=1)
        port_rets.name = "Portfolio"

        # Para el benchmark usamos el valor de mercado: r_m = (Mkt-RF + RF)
        # donde Mkt-RF y RF están en tanto por ciento o decimal según arriba.
        bench_rets = mkt_excess.copy()
        if bench_rets.max() > 1.0:
            bench_rets = bench_rets / 100.0
        bench_rets = bench_rets + rf_dec
        bench_rets.name = "Benchmark"

        # Alineamos todo a las mismas fechas
        common_index = stock_rets.index.intersection(bench_rets.index).intersection(rf_dec.index)
        stock_rets = stock_rets.loc[common_index]
        port_rets = port_rets.loc[common_index]
        bench_rets = bench_rets.loc[common_index]
        rf_dec = rf_dec.loc[common_index]
        mkt_excess = mkt_excess.loc[common_index]
        if mkt_excess.max() > 1.0:
            mkt_excess = mkt_excess / 100.0

        # ------------------------------------------------------------------
        # Build CAPM summary rows
        # ------------------------------------------------------------------
        summary_rows: list[dict] = []

        # --- 4.X.1 Individual stocks -------------------------------------------------
        for ticker in selected_stocks:
            r_i = stock_rets[ticker]

            # Excesos de retorno vs risk-free
            excess_i = r_i - rf_dec

            # CAPM regression vs market excess
            alpha_m, beta, resid_var, r2, total_var = _capm_stats(excess_i, mkt_excess)

            ann_ret = _ann_return(r_i)
            ann_vol = _ann_vol(r_i)

            # Sharpe ratio usando annualized rf
            ann_rf = (1 + rf_dec.mean()) ** 12 - 1
            excess_ann = ann_ret - ann_rf
            sharpe = excess_ann / ann_vol if ann_vol != 0 else 0.0

            summary_rows.append(
                {
                    "Name": ticker,
                    "Type": "Stock",
                    "Annual return": ann_ret,
                    "Annual volatility": ann_vol,
                    "Sharpe ratio": sharpe,
                    "Alpha (annual)": alpha_m * 12.0,
                    "Beta": beta,
                    "Residual variance": resid_var,
                    "R²": r2,
                    "Max drawdown": _max_drawdown(r_i),
                    "Tracking error": 0.0,   # no aplica para acción individual
                    "Information ratio": 0.0,  # no aplica para acción individual
                }
            )

        # --- 4.X.2 Portfolio ---------------------------------------------------------
        excess_port = port_rets - rf_dec
        alpha_m_port, beta_port, resid_var_port, r2_port, total_var_port = _capm_stats(
            excess_port, mkt_excess
        )

        ann_ret_port = _ann_return(port_rets)
        ann_vol_port = _ann_vol(port_rets)
        ann_rf = (1 + rf_dec.mean()) ** 12 - 1
        excess_ann_port = ann_ret_port - ann_rf
        sharpe_port = excess_ann_port / ann_vol_port if ann_vol_port != 0 else 0.0

        # Tracking error e information ratio vs benchmark
        active_rets = port_rets - bench_rets
        te_ann = _ann_vol(active_rets)  # std(active) * sqrt(12)
        ir = (ann_ret_port - _ann_return(bench_rets)) / te_ann if te_ann != 0 else 0.0

        summary_rows.append(
            {
                "Name": "Active portfolio",
                "Type": "Portfolio",
                "Annual return": ann_ret_port,
                "Annual volatility": ann_vol_port,
                "Sharpe ratio": sharpe_port,
                "Alpha (annual)": alpha_m_port * 12.0,
                "Beta": beta_port,
                "Residual variance": resid_var_port,
                "R²": r2_port,
                "Max drawdown": _max_drawdown(port_rets),
                "Tracking error": te_ann,
                "Information ratio": ir,
            }
        )

        # --- 4.X.3 Benchmark ---------------------------------------------------------
        excess_bench = bench_rets - rf_dec
        alpha_m_bench, beta_bench, resid_var_bench, r2_bench, total_var_bench = _capm_stats(
            excess_bench, mkt_excess
        )

        ann_ret_bench = _ann_return(bench_rets)
        ann_vol_bench = _ann_vol(bench_rets)
        excess_ann_bench = ann_ret_bench - ann_rf
        sharpe_bench = excess_ann_bench / ann_vol_bench if ann_vol_bench != 0 else 0.0

        summary_rows.append(
            {
                "Name": "Benchmark (market)",
                "Type": "Benchmark",
                "Annual return": ann_ret_bench,
                "Annual volatility": ann_vol_bench,
                "Sharpe ratio": sharpe_bench,
                "Alpha (annual)": alpha_m_bench * 12.0,
                "Beta": beta_bench,
                "Residual variance": resid_var_bench,
                "R²": r2_bench,
                "Max drawdown": _max_drawdown(bench_rets),
                "Tracking error": 0.0,     # por definición vs sí mismo
                "Information ratio": 0.0,  # por definición vs sí mismo
            }
        )

        # ------------------------------------------------------------------
        # Final CAPM summary DataFrame
        # ------------------------------------------------------------------
        capm_summary_df = pd.DataFrame(summary_rows).set_index("Name")

        # Guardamos en session_state para usarlo en otros sheets (por ejemplo Sheet 8)
        st.session_state["capm_summary_df"] = capm_summary_df

        # Mostrar tabla en pantalla, con formato profesional
        st.dataframe(
            capm_summary_df.style.format(
                {
                    "Annual return": "{:.2%}",
                    "Annual volatility": "{:.2%}",
                    "Sharpe ratio": "{:.2f}",
                    "Alpha (annual)": "{:.2%}",
                    "Beta": "{:.2f}",
                    "Residual variance": "{:.4f}",
                    "R²": "{:.2f}",
                    "Max drawdown": "{:.2%}",
                    "Tracking error": "{:.2%}",
                    "Information ratio": "{:.2f}",
                }
            )
        )

        # Botón para descargar el CSV con todos los datos
        csv_capm = capm_summary_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="Download 4.X – CAPM summary (CSV)",
            data=csv_capm,
            file_name="sheet4_capm_summary.csv",
            mime="text/csv",
        )
        
                # 4.X CAPM summary – stocks, portfolio and benchmark
        st.markdown("### 4.X CAPM summary – stocks, portfolio and benchmark")

        st.dataframe(
            capm_summary_df.style.format({
                "Annual return": "{:.2%}",
                "Annual volatility": "{:.2%}",
                "Sharpe ratio": "{:.2f}",
                "Alpha (annual)": "{:.2%}",
                "Beta": "{:.2f}",
                "Residual variance": "{:.4f}",
                "R²": "{:.2f}",
                "Max drawdown": "{:.2%}",
                "Tracking error": "{:.2%}",
                "Information ratio": "{:.2f}",
            }),
            use_container_width=True,
        )

        csv_capm = capm_summary_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="Download 4.X – CAPM summary (CSV)",
            data=csv_capm,
            file_name="sheet4_capm_summary.csv",
            mime="text/csv",
            key="download_capm_sheet4"
        )

        # ------------------------------------------------------------------
        # 4.Y Risk–return positioning (CAPM)
        # ------------------------------------------------------------------
        
        st.markdown("### 4.Y Risk–return positioning – stocks, portfolio and benchmark")

        st.markdown(
            """
            This scatter plot shows the **annualized return** versus **annualized volatility**
            for each stock, the active portfolio, and the market benchmark.
            It visually summarizes the **risk–return trade-off** implied by the CAPM results.
            """
        )

        # Asegurarnos de que tengamos una columna 'Name' explícita
        if "Name" in capm_summary_df.columns:
            df_plot = capm_summary_df.reset_index(drop=True).copy()
        else:
            df_plot = capm_summary_df.reset_index().rename(columns={"index": "Name"})

        # Filtrar por tipo
        stocks_plot = df_plot[df_plot["Type"] == "Stock"].copy()
        port_plot = df_plot[df_plot["Type"] == "Portfolio"].copy()
        bench_plot = df_plot[df_plot["Type"] == "Benchmark"].copy()

        # Convertir a porcentajes para el gráfico (sin usar numpy)
        for frame in (stocks_plot, port_plot, bench_plot):
            if not frame.empty:
                frame["ret_pct"] = frame["Annual return"] * 100.0
                frame["vol_pct"] = frame["Annual volatility"] * 100.0

        # Crear figura
        import matplotlib.pyplot as plt

        fig_rr, ax_rr = plt.subplots(figsize=(8, 6))

        # 1) Stocks
        if not stocks_plot.empty:
            ax_rr.scatter(
                stocks_plot["vol_pct"],
                stocks_plot["ret_pct"],
                label="Stocks",
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
            )
            # Etiquetas con los tickers
            for _, row in stocks_plot.iterrows():
                ax_rr.annotate(
                    row["Name"],
                    (row["vol_pct"], row["ret_pct"]),
                    textcoords="offset points",
                    xytext=(4, 3),
                    fontsize=8,
                )

        # 2) Active portfolio
        if not port_plot.empty:
            ax_rr.scatter(
                port_plot["vol_pct"],
                port_plot["ret_pct"],
                label="Active portfolio",
                marker="*",
                s=180,
                edgecolors="black",
                linewidths=0.7,
            )

        # 3) Benchmark
        if not bench_plot.empty:
            ax_rr.scatter(
                bench_plot["vol_pct"],
                bench_plot["ret_pct"],
                label="Benchmark",
                marker="s",
                s=90,
                edgecolors="black",
                linewidths=0.7,
            )

        ax_rr.set_xlabel("Annual volatility (%)")
        ax_rr.set_ylabel("Annual return (%)")
        ax_rr.set_title("Risk–return positioning (CAPM)")
        ax_rr.grid(True, linestyle="--", alpha=0.4)
        ax_rr.legend(loc="best", frameon=True)

        fig_rr.tight_layout()

        st.pyplot(fig_rr)
        import io 
        # Botón para descargar la gráfica
        buf_rr = io.BytesIO()
        fig_rr.savefig(buf_rr, format="png", bbox_inches="tight")
        buf_rr.seek(0)

        st.download_button(
            label="Download 4.Y – Risk–return scatter (PNG)",
            data=buf_rr,
            file_name="sheet4_capm_risk_return_scatter.png",
            mime="image/png",
        )

        # ==========================================================
        # 4.X CAPM Scatter Plot for selected stock
        # ==========================================================

        # ===========================




        # ------------------------------------------------------
        # 4.8 Glossary of symbols and variables
        # ------------------------------------------------------
        st.markdown("### 4.8 Glossary of CAPM notation used in this sheet")

        st.latex(r"R_{i,t} : \text{return of stock } i \text{ at time } t")
        st.latex(r"R_{M,t} : \text{return of the market portfolio at time } t")
        st.latex(r"R_{f,t} : \text{risk-free rate at time } t")
        st.latex(
            r"R_{i,t}^{\text{excess}} = R_{i,t} - R_{f,t} : "
            r"\text{excess return of stock } i"
        )
        st.latex(
            r"R_{M,t}^{\text{excess}} = R_{M,t} - R_{f,t} : "
            r"\text{excess return of the market}"
        )
        st.latex(
            r"\bar R_i^{\text{excess}} : "
            r"\text{sample average of } R_{i,t}^{\text{excess}}"
        )
        st.latex(
            r"\bar R_M^{\text{excess}} : "
            r"\text{sample average of } R_{M,t}^{\text{excess}}"
        )
        st.latex(
            r"\operatorname{Cov}(R_i^{\text{excess}}, R_M^{\text{excess}}) : "
            r"\text{sample covariance between stock and market excess returns}"
        )
        st.latex(
            r"\operatorname{Var}(R_M^{\text{excess}}) : "
            r"\text{sample variance of the market excess return}"
        )
        st.latex(
            r"\beta_i : \text{CAPM beta of stock } i "
            r"\text{(manual or OLS estimate)}"
        )
        st.latex(
            r"\alpha_i^{\text{hist, excess}} : "
            r"\text{historical alpha in excess returns (manual CAPM)}"
        )
        st.latex(
            r"\widehat\alpha_i^{\text{OLS}} : "
            r"\text{alpha from the OLS CAPM regression}"
        )
        st.latex(
            r"t(\widehat\beta_i),\ t(\widehat\alpha_i) : "
            r"\text{t-statistics for } \widehat\beta_i \text{ and } \widehat\alpha_i"
        )
        st.latex(
            r"\text{p\_value\_beta},\ \text{p\_value\_alpha} : "
            r"\text{p-values associated with those t-statistics}"
        )
        st.latex(
            r"R^2 : \text{coefficient of determination of the CAPM regression}"
        )
        st.latex(
            r"E[R_M] : \text{expected return of the market in levels "
            r"(sample average)}"
        )
        st.latex(
            r"\bar R_f : \text{average risk-free rate in levels (sample average)}"
        )
        st.latex(
            r"E[R_M] - \bar R_f : \text{market risk premium (MRP) in levels}"
        )
        st.latex(
            r"E[R_i]_{\text{CAPM}} : "
            r"\text{CAPM-implied expected return of stock } i \text{ in levels}"
        )
        st.latex(
            r"\bar R_i : \text{sample average return of stock } i \text{ in levels}"
        )
        st.latex(
            r"\widehat{\alpha}_i = \bar R_i - E[R_i]_{\text{CAPM}} : "
            r"\text{alpha forecast used in this assignment}"
        )

        # ==========================================================
    # SHEET 5 – Portfolio construction (all calculations)
    # ==========================================================
    elif section == "Sheet 5 – Portfolio Construction (Markowitz)":
        st.header("Sheet 5 – Portfolio construction (all calculations)")

        # ------------------------------------------------------
        # 5.0 Core portfolio formulas (LaTeX)
        # ------------------------------------------------------
        st.markdown("### 5.0 Core formulas used for the portfolio")

        st.latex(r"\mu_i = E[R_i] : \text{expected return of stock } i")
        st.latex(
            r"\sigma_i^2 = \operatorname{Var}(R_i) = "
            r"\dfrac{1}{T-1}\sum_{t=1}^{T}\big(R_{i,t} - \mu_i\big)^2"
        )
        st.latex(
            r"\sigma_i = \sqrt{\operatorname{Var}(R_i)}"
        )
        st.latex(
            r"\sigma_{ij} = \operatorname{Cov}(R_i, R_j) = "
            r"\dfrac{1}{T-1}\sum_{t=1}^{T}"
            r"\big(R_{i,t} - \mu_i\big)\big(R_{j,t} - \mu_j\big)"
        )
        st.latex(
            r"\rho_{ij} = \dfrac{\sigma_{ij}}{\sigma_i \sigma_j}"
        )
        st.latex(
            r"\bar R_f = \dfrac{1}{T}\sum_{t=1}^{T} R_{f,t}"
        )
        st.latex(
            r"\text{SR}_i = \dfrac{\mu_i - \bar R_f}{\sigma_i}"
        )

        st.markdown(
            r"""
This sheet contains **all calculations** needed to construct and analyze the
portfolio in later steps: expected returns, volatilities, Sharpe ratios,
covariance and correlation matrices, and the covariance matrix of **excess**
returns.
            """
        )

        st.markdown("---")

        # ------------------------------------------------------
        # 5.1 Level returns, expected returns, volatilities, Sharpe ratios
        # ------------------------------------------------------
        st.markdown(
            "### 5.1 Expected returns, volatilities and Sharpe ratios (individual stocks)"
        )

        # Risk-free and level returns
        rf_series = aligned_ff["RF"]
        # Level returns: R_i,t = R_i,t^{excess} + R_f,t
        level_returns = (
            excess_returns[selected_stocks]
            .add(rf_series, axis=0)
            .dropna()
        )

        # Align RF with the same index for averages
        rf_aligned = rf_series.loc[level_returns.index]
        R_f_bar = rf_aligned.mean()

        # Vector de medias y desviaciones
        mu_vec = level_returns.mean()       # E[R_i]
        sigma_vec = level_returns.std()     # σ_i

        # Sharpe ratios individuales
        sharpe_vec = (mu_vec - R_f_bar) / sigma_vec

        stats_df = pd.DataFrame(
            {
                "mean_return": mu_vec,
                "stdev": sigma_vec,
                "sharpe_ratio": sharpe_vec,
            }
        )
        stats_df.index.name = "ticker"

        st.markdown("**Vector of expected returns, volatilities and Sharpe ratios:**")
        st.dataframe(
            stats_df.style.format(
                {
                    "mean_return": "{:.4f}",
                    "stdev": "{:.4f}",
                    "sharpe_ratio": "{:.4f}",
                }
            )
        )

        csv_stats = stats_df.reset_index().to_csv(index=False)
        st.download_button(
            label="Download 5.1 – expected returns, stdev and Sharpe ratios (CSV)",
            data=csv_stats,
            file_name="sheet5_expected_returns_sharpe.csv",
            mime="text/csv",
        )

        st.latex(
            r"\mu_i = \dfrac{1}{T}\sum_{t=1}^{T} R_{i,t},"
            r"\quad"
            r"\sigma_i = \sqrt{\dfrac{1}{T-1}\sum_{t=1}^{T}"
            r"\big(R_{i,t} - \mu_i\big)^2},"
            r"\quad"
            r"\text{SR}_i = \dfrac{\mu_i - \bar R_f}{\sigma_i}"
        )

        st.markdown("---")

        # ------------------------------------------------------
        # 5.2 Covariance and correlation matrices (level returns)
        # ------------------------------------------------------
        st.markdown("### 5.2 Covariance and correlation matrices (level returns)")

        cov_matrix = level_returns.cov()
        corr_matrix = level_returns.corr()

        st.markdown("**Covariance matrix of level returns \\(\\Sigma\\):**")
        st.dataframe(cov_matrix.style.format("{:.6f}"))

        csv_cov = cov_matrix.reset_index().rename(columns={"index": "ticker"}).to_csv(
            index=False
        )
        st.download_button(
            label="Download 5.2 – covariance matrix of level returns (CSV)",
            data=csv_cov,
            file_name="sheet5_cov_matrix_level_returns.csv",
            mime="text/csv",
        )

        st.markdown("**Correlation matrix of level returns:**")
        st.dataframe(corr_matrix.style.format("{:.4f}"))

        csv_corr = corr_matrix.reset_index().rename(columns={"index": "ticker"}).to_csv(
            index=False
        )
        st.download_button(
            label="Download 5.2 – correlation matrix of level returns (CSV)",
            data=csv_corr,
            file_name="sheet5_corr_matrix_level_returns.csv",
            mime="text/csv",
        )

        st.latex(
            r"\Sigma = \big(\sigma_{ij}\big)_{i,j=1}^{6},"
            r"\quad "
            r"\rho_{ij} = \dfrac{\sigma_{ij}}{\sigma_i \sigma_j}"
        )

        st.markdown("---")

        # ------------------------------------------------------
        # 5.3 Covariance matrix of EXCESS returns
        # ------------------------------------------------------
        st.markdown("### 5.3 Covariance matrix of excess returns")

        excess_sub = excess_returns[selected_stocks].dropna()
        cov_excess = excess_sub.cov()

        st.markdown("**Covariance matrix of excess returns:**")
        st.dataframe(cov_excess.style.format("{:.6f}"))

        csv_cov_excess = (
            cov_excess.reset_index().rename(columns={"index": "ticker"}).to_csv(
                index=False
            )
        )
        st.download_button(
            label="Download 5.3 – covariance matrix of excess returns (CSV)",
            data=csv_cov_excess,
            file_name="sheet5_cov_matrix_excess_returns.csv",
            mime="text/csv",
        )

        st.latex(
            r"R_{i,t}^{\text{excess}} = R_{i,t} - R_{f,t},"
            r"\quad"
            r"\sigma_{ij}^{\text{excess}} = "
            r"\operatorname{Cov}\big(R_i^{\text{excess}}, R_j^{\text{excess}}\big)"
        )

        st.markdown("---")
        
            # ------------------------------------------------------
        # 5.4 Active 6-stock portfolio (Markowitz Sharpe-ratio maximization)
        # ------------------------------------------------------
        st.markdown("### 5.4 Active 6-stock portfolio (Markowitz Sharpe-ratio maximization)")

        st.markdown(
            """
    In this subsection I build the **active portfolio** using the six selected stocks.
    Following the instructor's clarification, I implement the **Markowitz Sharpe-ratio
    maximization procedure** instead of the full Treynor–Black model. The goal is to
    find the tangency (optimal risky) portfolio with respect to the risk-free rate.
            """
        )

        # --------------------------
        # 5.X.1 Inputs for Markowitz
        # --------------------------

        # I align the monthly stock returns and the Fama–French risk-free rate on the
        # same set of dates, to make sure the sample is consistent.
        ff_rf = aligned_ff["RF"].dropna()
        common_idx_5x = returns_sel.dropna().index.intersection(ff_rf.index)

        returns_6 = returns_sel.loc[common_idx_5x, selected_stocks]
        rf_series = ff_rf.loc[common_idx_5x]

        # Sample means (μ_i) and covariance matrix (Σ) of monthly returns
        mean_returns_vec = returns_6.mean()      # one mean per stock
        cov_matrix_6 = returns_6.cov()           # 6×6 covariance matrix
        rf_bar_6 = rf_series.mean()              # average monthly risk-free rate

        st.markdown("**Inputs used for the Markowitz optimization (monthly):**")
        st.dataframe(
            pd.DataFrame(
                {
                    "ticker": mean_returns_vec.index,
                    "mean_return": mean_returns_vec.round(4),
                }
            ),
            use_container_width=True,
        )

        st.markdown("Covariance matrix of monthly returns $\\Sigma$:")
        st.dataframe(cov_matrix_6.round(6), use_container_width=True)

        # --------------------------
        # 5.X.2 Markowitz tangency portfolio
        # --------------------------
        import numpy as np
        import math

        # Column vectors and helper ones-vector
        mu_vec = mean_returns_vec.values.reshape(-1, 1)   # (6×1)
        Sigma = cov_matrix_6.values                       # (6×6)
        ones_vec = np.ones((len(mean_returns_vec), 1))    # (6×1)

        # Expected excess returns relative to risk-free: μ - R_f * 1
        mu_excess_vec = mu_vec - rf_bar_6 * ones_vec

        # Unconstrained tangency portfolio weights: w* ∝ Σ^{-1} (μ - R_f 1)
        Sigma_inv = np.linalg.inv(Sigma)
        w_tilde = Sigma_inv @ mu_excess_vec

        # Normalize weights so that they sum to 1
        w_star = w_tilde / w_tilde.sum()

        w_star_series = pd.Series(
            w_star.flatten(),
            index=mean_returns_vec.index,
            name="optimal_weight",
        )

        # --------------------------
        # 5.X.3 Portfolio statistics (monthly)
        # --------------------------

        # Expected portfolio return, variance, stdev and Sharpe ratio
        mu_p_6 = float((w_star.T @ mu_vec)[0, 0])
        var_p_6 = float((w_star.T @ Sigma @ w_star)[0, 0])
        sigma_p_6 = math.sqrt(var_p_6)
        sharpe_p_6 = (mu_p_6 - rf_bar_6) / sigma_p_6 if sigma_p_6 > 0 else np.nan

        # --------------------------
        # 5.X.4 Tables for the dashboard
        # --------------------------
        weights_df_6 = (
            w_star_series.round(4)
            .reset_index()
            .rename(columns={"index": "ticker"})
        )

        portfolio_stats_6 = pd.DataFrame(
            {
                "expected_return": [round(mu_p_6, 4)],
                "stdev": [round(sigma_p_6, 4)],
                "sharpe_ratio": [round(sharpe_p_6, 4)],
                "avg_risk_free": [round(rf_bar_6, 4)],
            }
        )

        st.markdown("**Optimal Markowitz active portfolio weights (tangency portfolio):**")
        st.dataframe(weights_df_6, use_container_width=True)

        st.markdown("**Markowitz active portfolio statistics (monthly):**")
        st.dataframe(portfolio_stats_6, use_container_width=True)

        # CSV downloads
        csv_weights_6 = weights_df_6.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download – Markowitz optimal weights (CSV)",
            data=csv_weights_6,
            file_name="sheet5_markowitz_optimal_weights.csv",
            mime="text/csv",
        )

        csv_stats_6 = portfolio_stats_6.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download – Markowitz portfolio stats (CSV)",
            data=csv_stats_6,
            file_name="sheet5_markowitz_portfolio_stats.csv",
            mime="text/csv",
        )
        
        # ============================================================
        # 5.X Efficient Frontier (Markowitz) – CLEAN VERSION (no benchmark)
        # ============================================================

        st.markdown("### 5.X Efficient Frontier – Markowitz portfolio optimization")

        # --- 1. Datos base (USANDO TUS VARIABLES REALES) ---
        # port_rets: retornos mensuales de las 6 acciones (ya alineado en Sheet 5)
        port_rets = returns_6.copy()

        # medias y covarianza
        mean_returns = port_rets.mean()
        cov_matrix = port_rets.cov()
        tickers = list(mean_returns.index)
        n_assets = len(tickers)

        # ======================================================
        # 2. Utility functions (SIN NUMPY → con math y puro python)
        # ======================================================

        import math
        import matplotlib.pyplot as plt
        import random

        def portfolio_return(weights, mean_returns):
            """Portfolio expected return (monthly)."""
            r = 0.0
            for w, mu in zip(weights, mean_returns):
                r += w * mu
            return r

        def portfolio_vol(weights, cov_matrix):
            """Portfolio volatility (monthly)."""
            vol = 0.0
            for i in range(len(weights)):
                for j in range(len(weights)):
                    vol += weights[i] * cov_matrix.iloc[i, j] * weights[j]
            return math.sqrt(vol)

        def generate_random_weights(n):
            """Generate random weights that sum to 1 without NumPy."""
            raw = [random.random() for _ in range(n)]
            s = sum(raw)
            return [x / s for x in raw]

        def random_portfolios(n_portfolios, mean_returns, cov_matrix):
            """Generate portfolios for the efficient frontier."""
            results_vol = []
            results_ret = []

            for _ in range(n_portfolios):
                w = generate_random_weights(len(mean_returns))
                r = portfolio_return(w, mean_returns)
                v = portfolio_vol(w, cov_matrix)

                results_vol.append(v)
                results_ret.append(r)

            return results_vol, results_ret

        # ======================================================
        # 3. Generar 5,000 portafolios aleatorios
        # ======================================================

        vols, rets = random_portfolios(
            5000,
            mean_returns.values.tolist(),
            cov_matrix
        )

        # ======================================================
        # 4. Crear gráfica de Efficient Frontier (profesional)
        # ======================================================

        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(
            [v * math.sqrt(12) for v in vols],      # annualized vol
            [(1 + r)**12 - 1 for r in rets],        # annualized return
            c=[r / v if v > 0 else 0 for r, v in zip(rets, vols)],
            cmap="viridis",
            s=14,
        )

        ax.set_xlabel("Annualized Volatility (%)")
        ax.set_ylabel("Annualized Return (%)")
        ax.set_title("Efficient Frontier – Markowitz Optimization (No Benchmark)")
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.colorbar(scatter, label="Sharpe ratio (proxy)")

        st.pyplot(fig)

        # Descargar la figura
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label="Download Efficient Frontier (PNG)",
            data=buf.getvalue(),
            file_name="efficient_frontier.png",
            mime="image/png",
        )

        # ======================================================================
        # 5.X+ — Add CML, Optimal Portfolio Point, and Individual Assets
        # ======================================================================

        st.markdown("### 5.X+ Enhanced Efficient Frontier (CML + Optimal Portfolio + Assets)")

        import matplotlib.pyplot as plt
        import math

        # --- 1. Compute annualized stats for the optimal portfolio (w_star)
        sigma_p_annual = sigma_p_6 * math.sqrt(12)
        mu_p_annual = (1 + mu_p_6)**12 - 1

        # --- 2. Risk-free annual rate
        rf_annual = (1 + rf_bar_6)**12 - 1

        # --- 3. Compute CML line
        # Create a volatility range from 0 up to the max volatility in your scatter
        vol_range = [i / 1000 for i in range(0, int(max(vols)*math.sqrt(12)*1000) + 1)]
        ret_cml = [rf_annual + ( (mu_p_annual - rf_annual) / sigma_p_annual ) * v for v in vol_range]

        # --- 4. Compute individual asset risk-return points (annualized)
        asset_vols = []
        asset_rets = []

        for t in tickers:
            monthly_ret = mean_returns_vec[t]
            monthly_vol = cov_matrix_6.loc[t, t] ** 0.5

            asset_rets.append((1 + monthly_ret)**12 - 1)
            asset_vols.append(monthly_vol * math.sqrt(12))


        # --- 5. Enhanced plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        # Scatter of random portfolios
        scatter = ax2.scatter(
            [v * math.sqrt(12) for v in vols],
            [(1 + r)**12 - 1 for r in rets],
            c=[rets[i] / vols[i] for i in range(len(rets))],
            cmap="viridis",
            s=10,
            alpha=0.65
        )

        # CML line
        ax2.plot(vol_range, ret_cml, color="red", linewidth=2.5, label="Capital Market Line (CML)")

        # Optimal portfolio point
        ax2.scatter(
            sigma_p_annual,
            mu_p_annual,
            color="black",
            s=120,
            marker="*",
            label="Optimal Portfolio (Tangency)"
        )

        # Individual asset points
        ax2.scatter(asset_vols, asset_rets, color="orange", s=80, label="Assets")
        for i, t in enumerate(tickers):
            ax2.text(asset_vols[i] + 0.002, asset_rets[i] + 0.002, t, fontsize=9)

        # Labels and style
        ax2.set_xlabel("Annualized Volatility")
        ax2.set_ylabel("Annualized Return")
        ax2.set_title("Enhanced Efficient Frontier (CML + Optimal Portfolio + Assets)")
        plt.colorbar(scatter, label="Sharpe ratio (proxy)")
        ax2.legend()

        st.pyplot(fig2)

        # Download button
        import io
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png")
        st.download_button(
            label="Download Enhanced Efficient Frontier (PNG)",
            data=buf2.getvalue(),
            file_name="efficient_frontier_enhanced.png",
            mime="image/png",
        )

        
        # ------------------------------------------------------------------
        # EXPLANATION BLOCK (PURE LATEX, CONSISTENT WITH YOUR OTHER FORMULAS)
        # ------------------------------------------------------------------

        st.latex(r"\textbf{Interpretation\ of\ the\ Markowitz\ portfolio\ statistics}")

        st.latex(r"""
        \text{The\ statistics\ reported\ above\ are\ computed\ using\ monthly\ data:}
        """)

        # Monthly definitions
        st.latex(r"E[R_p]_{\text{monthly}} \;=\; \text{monthly expected return}")
        st.latex(r"\sigma_{p,\text{monthly}} \;=\; \text{monthly portfolio volatility}")
        st.latex(r"""
        \text{Sharpe}_{\text{monthly}}
        \;=\;
        \frac{E[R_p]_{\text{monthly}} - \bar{R}_f}{\sigma_{p,\text{monthly}}}
        """)

        st.latex(r"""
        \text{where } \bar{R}_f \text{ is the average monthly risk-free rate.}
        """)

        # Annualization section title
        st.latex(r"\textbf{Annualization\ (if\ needed)}")

        # Annual formulas
        st.latex(r"""
        E[R_p]_{\text{annual}}
        \;=\;
        (1 + E[R_p]_{\text{monthly}})^{12} - 1
        """)

        st.latex(r"""
        \sigma_{p,\text{annual}}
        \;=\;
        \sigma_{p,\text{monthly}} \sqrt{12}
        """)

        st.latex(r"""
        \text{Sharpe}_{\text{annual}}
        \;=\;
        \text{Sharpe}_{\text{monthly}} \sqrt{12}
        """)

        st.latex(r"""
        \text{In\ the\ written\ report,\ we\ keep\ the\ analysis\ in\ monthly\ terms\ to\ stay\ consistent\ with\ Fama--French\ data.}
        """)


        # --------------------------
        # 5.X.5 LaTeX formulas
        # --------------------------
        st.markdown("#### Markowitz formulas used in this section")

        st.latex(r"""
    \mu =
    \begin{pmatrix}
    \mu_1 \\
    \mu_2 \\
    \vdots \\
    \mu_6
    \end{pmatrix},
    \quad
    \Sigma =
    \begin{pmatrix}
    \sigma_{11} & \cdots & \sigma_{1 6} \\
    \vdots      & \ddots & \vdots      \\
    \sigma_{6 1} & \cdots & \sigma_{6 6}
    \end{pmatrix},
    \quad
    \mathbf{1} =
    \begin{pmatrix}
    1 \\
    1 \\
    \vdots \\
    1
    \end{pmatrix}
    """)

        st.latex(r"""
    \mu^{\text{excess}} = \mu - R_f \mathbf{1}
    """)

        st.latex(r"""
    \mathbf{w}^\* \propto \Sigma^{-1} \mu^{\text{excess}},
    \quad
    \mathbf{w}^\* =
    \frac{\Sigma^{-1} \mu^{\text{excess}}}
    {\mathbf{1}' \Sigma^{-1} \mu^{\text{excess}}}
    """)

        st.latex(r"""
    E[R_p] = \mathbf{w}^{\* \prime} \mu,
    \quad
    \sigma_p^2 = \mathbf{w}^{\* \prime} \Sigma \mathbf{w}^\*,
    \quad
    \text{Sharpe}_p =
    \frac{E[R_p] - R_f}{\sigma_p}
    """)


        # ------------------------------------------------------
        # 5.4 Glossary of notation used in Sheet 5
        # ------------------------------------------------------
        st.markdown("### 5.4 Glossary of notation for Sheet 5")

        st.latex(r"R_{i,t} : \text{return of stock } i \text{ at time } t")
        st.latex(r"R_{M,t} : \text{return of the market portfolio at time } t")
        st.latex(r"R_{f,t} : \text{risk-free rate at time } t")
        st.latex(
            r"R_{i,t}^{\text{excess}} = R_{i,t} - R_{f,t} : "
            r"\text{excess return of stock } i"
        )
        st.latex(
            r"\mu_i = E[R_i] : \text{sample average (expected) return of stock } i"
        )
        st.latex(
            r"\sigma_i : \text{sample standard deviation (volatility) of } R_i"
        )
        st.latex(
            r"\sigma_{ij} = \operatorname{Cov}(R_i, R_j) : "
            r"\text{covariance between stocks } i \text{ and } j"
        )
        st.latex(
            r"\rho_{ij} : \text{correlation between stocks } i \text{ and } j"
        )
        st.latex(
            r"\Sigma : \text{covariance matrix of level returns}"
        )
        st.latex(
            r"\Sigma^{\text{excess}} : \text{covariance matrix of excess returns}"
        )
        st.latex(
            r"\bar R_f : \text{average risk-free rate over the sample}"
        )
        st.latex(
            r"\text{SR}_i = \dfrac{\mu_i - \bar R_f}{\sigma_i} : "
            r"\text{Sharpe ratio of stock } i"
        )

        # ==========================================================
    # SHEET 6 – 3-Factor model regressions (Fama–French)
    # ==========================================================
    elif section == "Sheet 6 – 3-Factor model regressions":
        import statsmodels.api as sm
        from typing import List

        st.header("Sheet 6 – 3-Factor model regressions")

        # ------------------------------------------------------
        # 6.0 Core 3-factor formulas (LaTeX)
        # ------------------------------------------------------
        st.markdown("### 6.0 Fama–French 3-Factor model: formulas")

        st.latex(
            r"R_{i,t}^{\text{excess}} = R_{i,t} - R_{f,t}"
        )
        st.latex(
            r"R_{i,t}^{\text{excess}}"
            r" = \alpha_i"
            r" + \beta_{MKT,i}\,\text{MKT}_t"
            r" + \beta_{SMB,i}\,\text{SMB}_t"
            r" + \beta_{HML,i}\,\text{HML}_t"
            r" + \varepsilon_t"
        )
        st.latex(
            r"t(\widehat\beta_{k,i}) = "
            r"\dfrac{\widehat\beta_{k,i}}{\operatorname{SE}(\widehat\beta_{k,i})},"
            r"\qquad k \in \{\text{MKT},\text{SMB},\text{HML},\alpha\}"
        )

        st.markdown(
            r"""
Here I run, for each of the six stocks, the **Fama–French 3-Factor regression**:
excess stock return on the market factor (MKT), the size factor (SMB), and the
value factor (HML).  
All coefficients, t-statistics, p-values and \(R^2\) are reported in the tables
below and can be downloaded as CSV files.
            """
        )

        st.markdown("---")

        # ------------------------------------------------------
        # 6.1 Data alignment: excess returns and Fama–French factors
        # ------------------------------------------------------
        st.markdown("### 6.1 Data used in the 3-Factor regressions")

        # Factors from Fama–French library (already loaded in aligned_ff)
        mkt_excess = aligned_ff["Mkt-RF"]
        smb_factor = aligned_ff["SMB"]
        hml_factor = aligned_ff["HML"]
        rf_series = aligned_ff["RF"]

        # For the selected stocks, use excess returns (Sheet 3)
        excess_sub = excess_returns[selected_stocks]

        # Align index: intersection of dates between stock excess returns and factors
        common_idx_all = excess_sub.dropna().index.intersection(
            mkt_excess.dropna().index
        ).intersection(smb_factor.dropna().index).intersection(
            hml_factor.dropna().index
        )

        excess_sub = excess_sub.loc[common_idx_all]
        mkt_excess_aligned = mkt_excess.loc[common_idx_all]
        smb_aligned = smb_factor.loc[common_idx_all]
        hml_aligned = hml_factor.loc[common_idx_all]

        # Small preview table (inputs for regression)
        reg_inputs = pd.DataFrame(
            {
                "MKT_excess": mkt_excess_aligned,
                "SMB": smb_aligned,
                "HML": hml_aligned,
            }
        )
        st.markdown("**Common sample used for all 3-Factor regressions (factor side):**")
        st.dataframe(reg_inputs)

        csv_inputs = reg_inputs.reset_index().to_csv(index=False)
        st.download_button(
            label="Download 6.1 – common 3-factor inputs (CSV)",
            data=csv_inputs,
            file_name="sheet6_3factor_inputs_common.csv",
            mime="text/csv",
        )

        st.latex(
            rf"T = {len(common_idx_all)}"
        )

        st.markdown("---")

        # ------------------------------------------------------
        # 6.2 3-Factor regressions for all six stocks
        # ------------------------------------------------------
        st.markdown("### 6.2 3-Factor regression results for all six stocks")

        threef_rows: List[dict] = []

        for ticker in selected_stocks:
            Ri_excess_i = excess_sub[ticker].loc[common_idx_all]

            df_i = pd.DataFrame(
                {
                    "Ri_excess": Ri_excess_i,
                    "MKT_excess": mkt_excess_aligned,
                    "SMB": smb_aligned,
                    "HML": hml_aligned,
                }
            ).dropna()

            if df_i.empty:
                continue

            X = sm.add_constant(df_i[["MKT_excess", "SMB", "HML"]])
            y = df_i["Ri_excess"]
            model = sm.OLS(y, X).fit()

            threef_rows.append(
                {
                    "ticker": ticker,
                    "alpha_3f": model.params["const"],
                    "beta_mkt_3f": model.params["MKT_excess"],
                    "beta_smb_3f": model.params["SMB"],
                    "beta_hml_3f": model.params["HML"],
                    "t_stat_alpha_3f": model.tvalues["const"],
                    "t_stat_mkt_3f": model.tvalues["MKT_excess"],
                    "t_stat_smb_3f": model.tvalues["SMB"],
                    "t_stat_hml_3f": model.tvalues["HML"],
                    "p_value_alpha_3f": model.pvalues["const"],
                    "p_value_mkt_3f": model.pvalues["MKT_excess"],
                    "p_value_smb_3f": model.pvalues["SMB"],
                    "p_value_hml_3f": model.pvalues["HML"],
                    "R2_3f": model.rsquared,
                    "n_obs": int(model.nobs),
                }
            )

        threef_df = pd.DataFrame(threef_rows).set_index("ticker")

        st.markdown("**Fama–French 3-Factor regression results (all six stocks):**")
        st.dataframe(
            threef_df.style.format(
                {
                    "alpha_3f": "{:.4f}",
                    "beta_mkt_3f": "{:.4f}",
                    "beta_smb_3f": "{:.4f}",
                    "beta_hml_3f": "{:.4f}",
                    "t_stat_alpha_3f": "{:.2f}",
                    "t_stat_mkt_3f": "{:.2f}",
                    "t_stat_smb_3f": "{:.2f}",
                    "t_stat_hml_3f": "{:.2f}",
                    "p_value_alpha_3f": "{:.4f}",
                    "p_value_mkt_3f": "{:.4f}",
                    "p_value_smb_3f": "{:.4f}",
                    "p_value_hml_3f": "{:.4f}",
                    "R2_3f": "{:.3f}",
                }
            )
        )

        csv_threef = threef_df.reset_index().to_csv(index=False)
        st.download_button(
            label="Download 6.2 – 3-Factor regression results (CSV)",
            data=csv_threef,
            file_name="sheet6_3factor_regression_results.csv",
            mime="text/csv",
        )

        st.markdown("---")

        # ------------------------------------------------------
        # 6.3 Detailed 3-Factor regression for selected stock
        # ------------------------------------------------------
        st.markdown("### 6.3 Detailed 3-Factor regression for a selected stock")

        if not threef_df.empty:
            detailed_stock_3f = st.selectbox(
                "Stock for detailed 3-Factor regression view",
                options=threef_df.index.tolist(),
                index=0,
            )

            row3 = threef_df.loc[detailed_stock_3f]

            st.markdown(f"**3-Factor regression for `{detailed_stock_3f}`:**")

            # Regression equation with estimated coefficients
            st.latex(
                r"R_{i,t}^{\text{excess}}"
                r" = \widehat\alpha_i^{3F}"
                r" + \widehat\beta_{MKT,i}^{3F}\,\text{MKT}_t"
                r" + \widehat\beta_{SMB,i}^{3F}\,\text{SMB}_t"
                r" + \widehat\beta_{HML,i}^{3F}\,\text{HML}_t"
                r" + \widehat\varepsilon_t"
            )

            st.latex(
                rf"\widehat\alpha_i^{{3F}} = {row3['alpha_3f']:.4f}"
            )
            st.latex(
                rf"\widehat\beta_{{MKT,i}}^{{3F}} = {row3['beta_mkt_3f']:.4f}"
            )
            st.latex(
                rf"\widehat\beta_{{SMB,i}}^{{3F}} = {row3['beta_smb_3f']:.4f}"
            )
            st.latex(
                rf"\widehat\beta_{{HML,i}}^{{3F}} = {row3['beta_hml_3f']:.4f}"
            )

            # t-stats and p-values
            st.latex(
                rf"t(\widehat\alpha_i^{{3F}}) = {row3['t_stat_alpha_3f']:.2f},"
                rf"\quad p\text{{-value}} = {row3['p_value_alpha_3f']:.4f}"
            )
            st.latex(
                rf"t(\widehat\beta_{{MKT,i}}^{{3F}}) = {row3['t_stat_mkt_3f']:.2f},"
                rf"\quad p\text{{-value}} = {row3['p_value_mkt_3f']:.4f}"
            )
            st.latex(
                rf"t(\widehat\beta_{{SMB,i}}^{{3F}}) = {row3['t_stat_smb_3f']:.2f},"
                rf"\quad p\text{{-value}} = {row3['p_value_smb_3f']:.4f}"
            )
            st.latex(
                rf"t(\widehat\beta_{{HML,i}}^{{3F}}) = {row3['t_stat_hml_3f']:.2f},"
                rf"\quad p\text{{-value}} = {row3['p_value_hml_3f']:.4f}"
            )

            st.latex(
                rf"R^2_{{3F}} = {row3['R2_3f']:.3f},"
                rf"\quad n = {int(row3['n_obs'])}"
            )

        st.markdown("---")
        
        
        # ------------------------------------------------------
        # 6.X Fama–French 3-Factor Loadings (Portfolio)
        # ------------------------------------------------------

        st.markdown("### 6.X Fama–French 3-Factor Loadings (Portfolio)")

        # Para el portafolio necesitamos:
        # - returns_sel → retornos mensuales por acción (Sheet 3)
        # - w_star_series → pesos óptimos (si no existe, usamos pesos iguales)
        #   Para no depender de Markowitz, NO paramos la ejecución.

        if "returns_sel" not in locals():
            st.error("Internal error: returns_sel not found.")
        else:
            # 1. Construir retornos mensuales del portafolio (weighted returns)
            returns_for_pf = excess_returns[selected_stocks].dropna()

            if "w_star_series" in locals():
                weights_pf = w_star_series.reindex(selected_stocks)
                weights_pf = weights_pf / weights_pf.sum()
            else:
                # Pesos iguales si no hay Markowitz
                weights_pf = pd.Series([1/len(selected_stocks)]*len(selected_stocks),
                                    index=selected_stocks)

            # port_ret_excess_t = sum_i w_i * Ri_excess_{i,t}
            pf_excess_ret = (returns_for_pf * weights_pf).sum(axis=1)

            # 2. Alinear con factores FF3
            common_idx_pf = pf_excess_ret.index \
                .intersection(mkt_excess.dropna().index) \
                .intersection(smb_factor.dropna().index) \
                .intersection(hml_factor.dropna().index)

            y_pf = pf_excess_ret.loc[common_idx_pf]

            X_pf = pd.DataFrame({
                "MKT_excess": mkt_excess.loc[common_idx_pf],
                "SMB": smb_factor.loc[common_idx_pf],
                "HML": hml_factor.loc[common_idx_pf]
            })

            X_pf = sm.add_constant(X_pf)

            model_pf = sm.OLS(y_pf, X_pf).fit()

            alpha_p_ff3 = model_pf.params["const"]
            beta_mkt_p = model_pf.params["MKT_excess"]
            beta_smb_p = model_pf.params["SMB"]
            beta_hml_p = model_pf.params["HML"]
            R2_pf = model_pf.rsquared

            t_alpha_p = model_pf.tvalues["const"]
            t_mkt_p = model_pf.tvalues["MKT_excess"]
            t_smb_p = model_pf.tvalues["SMB"]
            t_hml_p = model_pf.tvalues["HML"]

            p_alpha_p = model_pf.pvalues["const"]
            p_mkt_p = model_pf.pvalues["MKT_excess"]
            p_smb_p = model_pf.pvalues["SMB"]
            p_hml_p = model_pf.pvalues["HML"]

            # ------------------------------------------------------
            # TABLA — igual a la del PDF ejemplo
            # ------------------------------------------------------
            ff3_portfolio_table = pd.DataFrame({
                "Alpha (monthly)": [alpha_p_ff3],
                "Alpha (annualized)": [(1 + alpha_p_ff3)**12 - 1],
                "Beta MKT": [beta_mkt_p],
                "Beta SMB": [beta_smb_p],
                "Beta HML": [beta_hml_p],
                "R²": [R2_pf],
                "t(alpha)": [t_alpha_p],
                "t(MKT)": [t_mkt_p],
                "t(SMB)": [t_smb_p],
                "t(HML)": [t_hml_p],
                "p(alpha)": [p_alpha_p],
                "p(MKT)": [p_mkt_p],
                "p(SMB)": [p_smb_p],
                "p(HML)": [p_hml_p],
                "n_obs": [len(common_idx_pf)],
            }, index=["Portfolio"])

            st.markdown("**Fama–French 3-Factor Loadings for the Portfolio:**")
            st.dataframe(
                ff3_portfolio_table.style.format({
                    "Alpha (monthly)": "{:.4f}",
                    "Alpha (annualized)": "{:.4f}",
                    "Beta MKT": "{:.4f}",
                    "Beta SMB": "{:.4f}",
                    "Beta HML": "{:.4f}",
                    "R²": "{:.3f}",
                    "t(alpha)": "{:.2f}",
                    "t(MKT)": "{:.2f}",
                    "t(SMB)": "{:.2f}",
                    "t(HML)": "{:.2f}",
                    "p(alpha)": "{:.4f}",
                    "p(MKT)": "{:.4f}",
                    "p(SMB)": "{:.4f}",
                    "p(HML)": "{:.4f}",
                })
            )

            # Botón CSV
            st.download_button(
                label="Download – FF3 portfolio loadings (CSV)",
                data=ff3_portfolio_table.reset_index().to_csv(index=False),
                file_name="sheet6_ff3_portfolio_loadings.csv",
                mime="text/csv",
            )

            st.markdown("---")

            # ------------------------------------------------------
            # 6.X+ GRÁFICAS PROFESIONALES (Portafolio)
            # ------------------------------------------------------

            st.markdown("### 6.X+ FF3 Portfolio Graphics")

            import matplotlib.pyplot as plt

            # ====== Gráfica 1 — Betas ======
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.bar(["MKT", "SMB", "HML"], [beta_mkt_p, beta_smb_p, beta_hml_p],
                    color=["#4c72b0", "#55a868", "#c44e52"])
            ax1.set_title("Portfolio FF3 Betas")
            ax1.set_ylabel("Beta")
            st.pyplot(fig1)

            # ====== Gráfica 2 — t-stats ======
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.bar(["α", "MKT", "SMB", "HML"],
                    [t_alpha_p, t_mkt_p, t_smb_p, t_hml_p],
                    color="#6a5acd")
            ax2.axhline(2, color="red", linestyle="--", linewidth=1)
            ax2.axhline(-2, color="red", linestyle="--", linewidth=1)
            ax2.set_title("Portfolio FF3 t-statistics")
            st.pyplot(fig2)

            # ====== Gráfica 3 — Actual vs Predicho ======
            fitted_pf = model_pf.fittedvalues

            fig3, ax3 = plt.subplots(figsize=(8, 4))
            ax3.scatter(fitted_pf, y_pf, alpha=0.6)
            ax3.set_xlabel("Predicted excess return")
            ax3.set_ylabel("Actual excess return")
            ax3.set_title("FF3 Model – Actual vs Predicted")
            st.pyplot(fig3)

            st.markdown("---")

            # ------------------------------------------------------
            # 6.X++ INTERPRETACIÓN AUTOMÁTICA
            # ------------------------------------------------------

            st.markdown("### 6.X++ Interpretation of Portfolio FF3 Loadings")

            interpretation = f"""
        **Alpha:** The portfolio's monthly alpha is {alpha_p_ff3:.4f}, which annualizes to {((1+alpha_p_ff3)**12 - 1):.4f}.  
        A statistically insignificant alpha (p = {p_alpha_p:.4f}) suggests no strong evidence of mispricing.

        **Market Exposure (β_MKT):** The portfolio shows a market beta of {beta_mkt_p:.3f}.  
        This implies that the portfolio behaves as a {'higher' if beta_mkt_p>1 else 'lower'}-than-market risk asset.

        **Size Exposure (β_SMB):** The SMB loading is {beta_smb_p:.3f}, indicating a tilt toward {'small-cap' if beta_smb_p>0 else 'large-cap'} stocks.

        **Value Exposure (β_HML):** The HML beta of {beta_hml_p:.3f} suggests a {'value' if beta_hml_p>0 else 'growth'} orientation.

        **Model Fit:** The regression explains {R2_pf:.3f} of the variation in portfolio excess returns with {len(common_idx_pf)} monthly observations.  
        A moderate R² aligns with typical FF3 results for diversified equity portfolios.

        Overall, the FF3 loadings describe the portfolio’s economic exposures and confirm that most of the risk is driven by market participation rather than alpha generation.
        """
            st.markdown(interpretation)

            st.markdown("---")


        # ------------------------------------------------------
        # 6.4 Glossary of 3-Factor notation
        # ------------------------------------------------------
        st.markdown("### 6.4 Glossary of 3-Factor model notation used in this sheet")

        st.latex(
            r"R_{i,t} : \text{return of stock } i \text{ at time } t"
        )
        st.latex(
            r"R_{f,t} : \text{risk-free rate at time } t"
        )
        st.latex(
            r"R_{i,t}^{\text{excess}} = R_{i,t} - R_{f,t} : "
            r"\text{excess return of stock } i"
        )
        st.latex(
            r"\text{MKT}_t = R_{M,t} - R_{f,t} : "
            r"\text{market factor (excess market return)}"
        )
        st.latex(
            r"\text{SMB}_t : \text{size factor (small minus big)}"
        )
        st.latex(
            r"\text{HML}_t : \text{value factor (high minus low book-to-market)}"
        )
        st.latex(
            r"\alpha_i^{3F} : \text{intercept (3-Factor alpha) for stock } i"
        )
        st.latex(
            r"\beta_{MKT,i}^{3F} : \text{loading on the market factor for stock } i"
        )
        st.latex(
            r"\beta_{SMB,i}^{3F} : \text{loading on the size factor for stock } i"
        )
        st.latex(
            r"\beta_{HML,i}^{3F} : \text{loading on the value factor for stock } i"
        )
        st.latex(
            r"t(\widehat\beta_{k,i}) : \text{t-statistic for coefficient }"
            r" \widehat\beta_{k,i},\ k\in\{\alpha,\text{MKT},\text{SMB},\text{HML}\}"
        )
        st.latex(
            r"\text{p-value}_k : "
            r"\text{p-value associated with } t(\widehat\beta_{k,i})"
        )
        st.latex(
            r"R^2_{3F} : "
            r"\text{coefficient of determination of the 3-Factor regression}"
        )
        
        # ==========================================================
    # SHEET 7 – Portfolio summary (data for the Word document)
    # ==========================================================
    elif section == "Sheet 7 – Portfolio summary":
        st.header("Sheet 7 – Portfolio summary")

        # ------------------------------------------------------
        # 7.0 Formulas used for the portfolio summary
        # ------------------------------------------------------
        st.markdown("### 7.0 Formulas used in this sheet")

        st.latex(r"\mu_p = \sum_{i=1}^{N} w_i \mu_i")
        st.latex(r"\sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w}")
        st.latex(r"\sigma_p = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}")
        st.latex(r"\bar R_p = \dfrac{1}{T}\sum_{t=1}^{T} R_{p,t}")
        st.latex(r"SR_p = \dfrac{\bar R_p - \bar R_f}{\sigma_p}")
        st.latex(
            r"\beta_p^{CAPM} = \sum_{i=1}^{N} w_i \beta_i^{CAPM}"
        )
        st.latex(
            r"\beta_p^{MKT} = \sum_{i=1}^{N} w_i \beta_i^{MKT},"
            r"\quad "
            r"\beta_p^{SMB} = \sum_{i=1}^{N} w_i \beta_i^{SMB},"
            r"\quad "
            r"\beta_p^{HML} = \sum_{i=1}^{N} w_i \beta_i^{HML}"
        )

        st.markdown(
            r"""
This sheet aggregates all key statistics for my **final portfolio**.  
The numbers shown here are the ones that will go into the first page of the
Word document (portfolio data summary).
            """
        )

        st.markdown("---")

        # ------------------------------------------------------
        # 7.1 Build the portfolio return series (equally weighted)
        # ------------------------------------------------------
        st.markdown("### 7.1 Portfolio return series (equally weighted)")

        # I use the simple monthly returns of the selected stocks (no excess).
        # returns_sel: DataFrame with columns = selected_stocks, index = monthly dates.
        # aligned_ff: Fama–French factors already aligned in previous steps.

        # I make sure to align stock returns and Fama–French factors on the same dates.
        factors_for_align = aligned_ff[["Mkt-RF", "SMB", "HML", "RF"]].dropna()
        common_idx = returns_sel.dropna().index.intersection(factors_for_align.index)

        returns_p = returns_sel.loc[common_idx, selected_stocks]
        ff_p = factors_for_align.loc[common_idx]

        # Equal weights for the six stocks (this is my base active portfolio).
        import numpy as np

        n_stocks = len(selected_stocks)
        weights_eq = np.ones(n_stocks) / n_stocks
        weights_series = pd.Series(weights_eq, index=selected_stocks, name="w_i")

        # Portfolio return in levels: R_{p,t} = sum_i w_i R_{i,t}
        port_ret = (returns_p @ weights_series).rename("R_p")

        # I also compute the excess return of the portfolio:
        port_excess = port_ret - ff_p["RF"]

        # Basic sample size
        T = len(port_ret)

        st.markdown("**Equally weighted portfolio returns (monthly): preview**")
        st.dataframe(port_ret.to_frame().head())

        st.latex(rf"T = {T}")

        st.markdown("---")

        # ------------------------------------------------------
        # 7.2 Mean, volatility and Sharpe ratio of the portfolio
        # ------------------------------------------------------
        st.markdown("### 7.2 Mean, volatility and Sharpe ratio of the portfolio")

        # Sample mean and standard deviation of portfolio returns
        R_p_bar = float(port_ret.mean())
        sigma_p = float(port_ret.std(ddof=1))

        # Average risk-free rate (same sample)
        R_f_bar = float(ff_p["RF"].mean())

        # Sharpe ratio (monthly)
        sharpe_p = (R_p_bar - R_f_bar) / sigma_p if sigma_p > 0 else np.nan

        # I also compute the covariance-based volatility using the covariance matrix,
        # just as a consistency check.
        cov_matrix = returns_p.cov()
        w_vec = weights_eq.reshape(-1, 1)
        var_p = float(w_vec.T @ cov_matrix.values @ w_vec)
        sigma_p_cov = float(np.sqrt(var_p))

        st.latex(rf"\bar R_p = {R_p_bar:.4f}")
        st.latex(rf"\sigma_p = {sigma_p:.4f}")
        st.latex(rf"\bar R_f = {R_f_bar:.4f}")
        st.latex(
            rf"SR_p = \dfrac{{\bar R_p - \bar R_f}}{{\sigma_p}} = {sharpe_p:.4f}"
        )
        st.latex(
            rf"\sigma_p^2 = \mathbf{{w}}^\top \Sigma \mathbf{{w}} = {var_p:.6f},"
            rf"\quad \sigma_p = {sigma_p_cov:.4f}"
        )

        st.markdown("---")

        # ------------------------------------------------------
        # 7.3 CAPM regression for the portfolio (excess returns)
        # ------------------------------------------------------
        st.markdown("### 7.3 CAPM regression for the portfolio")

        import statsmodels.api as sm

        df_capm_p = pd.DataFrame(
            {
                "Rp_excess": port_excess,
                "Mkt_RF": ff_p["Mkt-RF"],
            }
        ).dropna()

        X_capm = sm.add_constant(df_capm_p["Mkt_RF"])
        y_capm = df_capm_p["Rp_excess"]
        capm_port_model = sm.OLS(y_capm, X_capm).fit()

        alpha_p_capm = float(capm_port_model.params["const"])
        beta_p_capm = float(capm_port_model.params["Mkt_RF"])
        t_alpha_p = float(capm_port_model.tvalues["const"])
        t_beta_p = float(capm_port_model.tvalues["Mkt_RF"])
        R2_capm_p = float(capm_port_model.rsquared)

        st.latex(
            r"R_{p,t}^{\text{excess}}"
            r" = \alpha_p^{CAPM} + \beta_p^{CAPM} \,\text{MKT}_t^{\text{excess}} + \varepsilon_t"
        )
        st.latex(rf"\alpha_p^{{CAPM}} = {alpha_p_capm:.4f}")
        st.latex(rf"\beta_p^{{CAPM}} = {beta_p_capm:.4f}")
        st.latex(
            rf"t(\alpha_p^{{CAPM}}) = {t_alpha_p:.2f},"
            rf"\quad t(\beta_p^{{CAPM}}) = {t_beta_p:.2f}"
        )
        st.latex(rf"R^2_{{CAPM,p}} = {R2_capm_p:.3f}")

        st.markdown("---")

        # ------------------------------------------------------
        # 7.4 Fama–French 3-Factor regression for the portfolio
        # ------------------------------------------------------
        st.markdown("### 7.4 Fama–French 3-Factor regression for the portfolio")

        df_ff3_p = pd.DataFrame(
            {
                "Rp_excess": port_excess,
                "Mkt_RF": ff_p["Mkt-RF"],
                "SMB": ff_p["SMB"],
                "HML": ff_p["HML"],
            }
        ).dropna()

        X_ff3 = sm.add_constant(df_ff3_p[["Mkt_RF", "SMB", "HML"]])
        y_ff3 = df_ff3_p["Rp_excess"]
        ff3_port_model = sm.OLS(y_ff3, X_ff3).fit()

        alpha_p_ff3 = float(ff3_port_model.params["const"])
        beta_mkt_p = float(ff3_port_model.params["Mkt_RF"])
        beta_smb_p = float(ff3_port_model.params["SMB"])
        beta_hml_p = float(ff3_port_model.params["HML"])
        R2_ff3_p = float(ff3_port_model.rsquared)

        st.latex(
            r"R_{p,t}^{\text{excess}}"
            r" = \alpha_p^{3F}"
            r" + \beta_{MKT,p}^{3F}\,\text{MKT}_t"
            r" + \beta_{SMB,p}^{3F}\,\text{SMB}_t"
            r" + \beta_{HML,p}^{3F}\,\text{HML}_t"
            r" + \varepsilon_t"
        )
        st.latex(rf"\alpha_p^{{3F}} = {alpha_p_ff3:.4f}")
        st.latex(rf"\beta_{{MKT,p}}^{{3F}} = {beta_mkt_p:.4f}")
        st.latex(rf"\beta_{{SMB,p}}^{{3F}} = {beta_smb_p:.4f}")
        st.latex(rf"\beta_{{HML,p}}^{{3F}} = {beta_hml_p:.4f}")
        st.latex(rf"R^2_{{3F,p}} = {R2_ff3_p:.3f}")

        st.markdown("---")

        # ------------------------------------------------------
        # 7.5 Final portfolio summary table (for the Word document)
        # ------------------------------------------------------
        st.markdown("### 7.5 Final portfolio summary table")

        summary_rows = [
            ("Sample size (months)", T),
            ("Mean portfolio return (monthly)", R_p_bar),
            ("Std. dev. portfolio return (monthly)", sigma_p),
            ("Mean risk-free rate (monthly)", R_f_bar),
            ("Sharpe ratio (monthly)", sharpe_p),
            ("CAPM alpha (portfolio)", alpha_p_capm),
            ("CAPM beta (portfolio)", beta_p_capm),
            ("R-squared CAPM (portfolio)", R2_capm_p),
            ("3F alpha (portfolio)", alpha_p_ff3),
            ("3F beta – MKT (portfolio)", beta_mkt_p),
            ("3F beta – SMB (portfolio)", beta_smb_p),
            ("3F beta – HML (portfolio)", beta_hml_p),
            ("R-squared 3F (portfolio)", R2_ff3_p),
            ("Number of stocks in portfolio", n_stocks),
            ("Sum of weights (equally weighted)", float(weights_series.sum())),
        ]

        summary_df = pd.DataFrame(summary_rows, columns=["statistic", "value"])

        st.dataframe(
            summary_df.style.format({"value": "{:.4f}"}),
            use_container_width=True,
        )

        csv_summary = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download 7.5 – Portfolio data summary (CSV)",
            data=csv_summary,
            file_name="sheet7_portfolio_summary.csv",
            mime="text/csv",
        )
    
    elif section == "Sheet 8 – Report analytics":
        # ---------------------------------------------------------------
        # Sheet 8 – Report analytics (all charts + download buttons)
        # ---------------------------------------------------------------
        st.header("Sheet 8 – Report analytics")

        st.markdown(
            """
            In this sheet I collect all charts and tables that I will use in the written report.
            The focus is on professional-looking summary statistics and visualizations that can be
            exported and pasted into the Word document (mutual-fund style first page and the
            following analytical pages).
            """
        )

        # -----------------------------------------------------------
        # 8.0 Base data for all analytics in this sheet
        # -----------------------------------------------------------
        # Use only the 6-stock active portfolio
        portfolio_tickers = selected_stocks
        price_subset = prices[portfolio_tickers].dropna()

        # Monthly returns of each stock
        ret_subset = price_subset.pct_change().dropna()

        # Equal-weighted portfolio by default (can be replaced with Markowitz weights later)
        n_assets = len(portfolio_tickers)
        if n_assets == 0:
            st.warning("Please select six stocks in the sidebar to build the portfolio.")
            st.stop()

        weights = pd.Series(1.0 / n_assets, index=portfolio_tickers, name="weight")

        # Portfolio returns (monthly)
        port_rets = (ret_subset * weights).sum(axis=1)
        port_rets.name = "portfolio_return"

            # Portfolio returns (monthly)
        port_rets = (ret_subset * weights).sum(axis=1)
        port_rets.name = "portfolio_return"

        # Risk-free rate series (monthly, from Fama–French factors)
        # aligned_ff is already created in Sheet 6
        rf_series = aligned_ff["RF"]  # usually in percentages
        if rf_series.max() > 1:
            # Convert from % to decimal if needed
            rf_series = rf_series / 100.0

        rf_monthly = rf_series.reindex(port_rets.index).fillna(method="ffill")
        avg_rf_monthly = rf_monthly.mean()
        avg_rf_annual = (1 + avg_rf_monthly) ** 12 - 1

        # Market portfolio / benchmark returns (monthly)
        # Use the Fama–French market factor Mkt-RF + risk-free as proxy for the benchmark
        mkt_excess = aligned_ff["Mkt-RF"]
        mkt_excess = mkt_excess.reindex(port_rets.index).dropna()

        bench_rets = (mkt_excess + rf_monthly.reindex(mkt_excess.index)).dropna()
        bench_rets.name = "benchmark_return"


        # Helper functions for annualization (assuming monthly data)
        def _ann_return(r: pd.Series) -> float:
            """Annualized arithmetic return from monthly series."""
            return (1 + r.mean()) ** 12 - 1

        def _ann_vol(r: pd.Series) -> float:
            """Annualized volatility from monthly series."""
            # sqrt(12) ≈ 3.464…, se usa para anualizar volatilidad mensual
            return r.std() * (12 ** 0.5)

        # Excess returns vs risk-free
        excess_port = port_rets - rf_monthly
        if bench_rets is not None:
            excess_bench = bench_rets - rf_monthly.reindex(bench_rets.index)

        # Simple cumulative value (Growth of 10,000)
        cum_port = (1 + port_rets).cumprod() * 10_000
        if bench_rets is not None:
            cum_bench = (1 + bench_rets).cumprod() * 10_000

            # --- Growth of $10,000 chart (portfolio vs benchmark) ---------------
            st.markdown("### Growth of $10,000 – active portfolio vs benchmark")

            cum_port = (1 + port_rets).cumprod() * 10_000
            if bench_rets is not None:
                cum_bench = (1 + bench_rets).cumprod() * 10_000
            else:
                cum_bench = None
            
            import matplotlib.pyplot as plt 
            
            fig_g, ax_g = plt.subplots(figsize=(8, 4))
            ax_g.plot(cum_port.index, cum_port.values, label="Active portfolio")
            if cum_bench is not None:
                ax_g.plot(cum_bench.index, cum_bench.values, label="Benchmark", linestyle="--")

            ax_g.set_ylabel("Value ($)")
            ax_g.set_xlabel("Date")
            ax_g.set_title("Growth of $10,000 – active portfolio vs benchmark")
            ax_g.grid(axis="y", linestyle="--", alpha=0.4)
            ax_g.legend()
            fig_g.tight_layout()
            st.pyplot(fig_g)

            png_buf = BytesIO()
            fig_g.savefig(png_buf, format="png", bbox_inches="tight")
            st.download_button(
                label="Download 8.1 – Growth of $10,000 chart (PNG)",
                data=png_buf.getvalue(),
                file_name="sheet8_growth_10000.png",
                mime="image/png",
            )


        # 8.2 Sector allocation vs benchmark
        st.subheader("8.2 Sector allocation vs benchmark")

        # Fundamental data already fetched earlier in the app
        fundamentals = fetch_yahoo_fundamentals(portfolio_tickers)

        # Añadir los pesos del portafolio al dataframe de fundamentales
        fundamentals_with_w = fundamentals.copy()
        fundamentals_with_w["weight"] = weights.reindex(fundamentals_with_w.index)

        # Sumar pesos por sector
        sector_weights = (
            fundamentals_with_w
                .groupby("sector")["weight"]
                .sum()
                .sort_values(ascending=False)
        )

        fig_sec, ax_sec = plt.subplots(figsize=(8, 4))
        sector_weights.plot(kind="bar", ax=ax_sec)

        ax_sec.set_ylabel("Weight")
        ax_sec.set_title("Sector allocation – active portfolio")
        ax_sec.grid(axis="y", linestyle="--", alpha=0.4)
        ax_sec.tick_params(axis="x", rotation=45)
        fig_sec.tight_layout()

        st.pyplot(fig_sec)

        buf_sec = BytesIO()
        fig_sec.savefig(buf_sec, format="png", bbox_inches="tight")
        st.download_button(
            label="Download 8.2 – Sector allocation chart (PNG)",
            data=buf_sec.getvalue(),
            file_name="sheet8_2_sector_allocation.png",
            mime="image/png",
        )

        # -----------------------------------------------------------
        # 8.3 Market-cap allocation (style box / size buckets)
        # -----------------------------------------------------------
        # 8.3 Market-cap allocation – active portfolio
        st.subheader("8.3 Market-cap allocation – active portfolio")

        st.markdown(
            """
        Here I classify the six holdings into **Small / Mid / Large cap** buckets based on
        their market capitalization (using terciles) and show the **portfolio weight** in
        each bucket. This helps to see if the portfolio tilts toward smaller or larger firms.
        """
        )

        # --- Build fundamentals + weights dataframe -----------------------------------
        # Re-use fundamentals_with_w from 8.2 if it exists, otherwise rebuild it
        if "fundamentals_with_w" in locals():
            f_w = fundamentals_with_w.copy()
        else:
            fundamentals = fetch_yahoo_fundamentals(portfolio_tickers)
            f_w = fundamentals.copy()
            f_w["weight"] = weights.reindex(f_w.index)

        # Quantile thresholds for size buckets (based on market cap)
        q_small = f_w["market_cap"].quantile(1 / 3)
        q_large = f_w["market_cap"].quantile(2 / 3)

        def _size_bucket(x: float) -> str:
            if x <= q_small:
                return "Small cap"
            elif x <= q_large:
                return "Mid cap"
            else:
                return "Large cap"

        # Assign each stock to a size bucket
        f_w["size_bucket"] = f_w["market_cap"].apply(_size_bucket)

        # Aggregate portfolio weights by size bucket and enforce display order
        size_weights = (
            f_w.groupby("size_bucket")["weight"]
            .sum()
            .reindex(["Small cap", "Mid cap", "Large cap"])
        )

        # --- Chart: market-cap allocation --------------------------------------------
        fig_size, ax_size = plt.subplots(figsize=(6, 4))
        size_weights.plot(kind="bar", ax=ax_size)

        ax_size.set_ylabel("Weight")
        ax_size.set_xlabel("")
        ax_size.set_title("Market-cap allocation – active portfolio")
        ax_size.grid(axis="y", linestyle="--", alpha=0.4)
        ax_size.tick_params(axis="x", rotation=0)
        fig_size.tight_layout()

        st.pyplot(fig_size)

        # Download button for the chart (PNG)
        buf_size = BytesIO()
        fig_size.savefig(buf_size, format="png", bbox_inches="tight")
        st.download_button(
            label="Download 8.3 – Market-cap chart (PNG)",
            data=buf_size.getvalue(),
            file_name="sheet8_3_market_cap_allocation.png",
            mime="image/png",
        )

        # -----------------------------------------------------------
        # 8.4 Rolling 12-month performance and risk
        # -----------------------------------------------------------
        st.subheader("8.4 Rolling 12-month performance and risk")

        # Rolling performance (cumulative 12-month return)
        # 8.4 Rolling 12-month performance and risk
        rolling_window: int = 12

        # Rolling performance: producto de (1 + r_t) en cada ventana de 12 meses
        # Usamos x.prod() para evitar np.prod y el problema con 'np' local
        roll_perf = (1 + port_rets).rolling(rolling_window).apply(
            lambda x: x.prod(),  # x es una Serie; prod() multiplica todos los valores
            raw=False,
        )
        roll_perf.name = "Rolling 12-month return"


        fig_roll, ax_roll = plt.subplots(figsize=(8, 4))
        roll_perf.plot(ax=ax_roll, linewidth=2)
        ax_roll.set_title("Rolling 12-month performance (active portfolio)")
        ax_roll.set_ylabel("Return")
        ax_roll.grid(True, linestyle="--", alpha=0.4)
        fig_roll.tight_layout()

        buf_roll = BytesIO()
        fig_roll.savefig(buf_roll, format="png", dpi=150)
        buf_roll.seek(0)

        st.pyplot(fig_roll)
        st.download_button(
            label="Download 8.4 – Rolling performance chart (PNG)",
            data=buf_roll.getvalue(),
            file_name="sheet8_rolling_performance.png",
            mime="image/png",
        )

        # -----------------------------------------------------------
        # 8.5 Drawdown and return distribution
        # -----------------------------------------------------------
        st.subheader("8.5 Drawdown and return distribution")

        cum_values = (1 + port_rets).cumprod()
        running_max = cum_values.cummax()
        drawdown = cum_values / running_max - 1
        drawdown.name = "drawdown"

        fig_dd, ax_dd = plt.subplots(figsize=(8, 4))
        drawdown.plot(ax=ax_dd, linewidth=2)
        ax_dd.set_title("Portfolio drawdown (active portfolio)")
        ax_dd.set_ylabel("Drawdown")
        ax_dd.grid(True, linestyle="--", alpha=0.4)
        fig_dd.tight_layout()

        buf_dd = BytesIO()
        fig_dd.savefig(buf_dd, format="png", dpi=150)
        buf_dd.seek(0)

        st.pyplot(fig_dd)
        st.download_button(
            label="Download 8.5 – Drawdown chart (PNG)",
            data=buf_dd.getvalue(),
            file_name="sheet8_drawdown.png",
            mime="image/png",
        )

        # Histogram of monthly returns
        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
        ax_hist.hist(port_rets, bins=25, edgecolor="black", alpha=0.7)
        ax_hist.set_title("Distribution of monthly returns – active portfolio")
        ax_hist.set_xlabel("Monthly return")
        ax_hist.set_ylabel("Frequency")
        ax_hist.grid(axis="y", linestyle="--", alpha=0.4)
        fig_hist.tight_layout()

        buf_hist = BytesIO()
        fig_hist.savefig(buf_hist, format="png", dpi=150)
        buf_hist.seek(0)

        st.pyplot(fig_hist)
        st.download_button(
            label="Download 8.5 – Return distribution chart (PNG)",
            data=buf_hist.getvalue(),
            file_name="sheet8_return_distribution.png",
            mime="image/png",
        )

        # -----------------------------------------------------------
        # 8.6 Correlation matrix between holdings
        # -----------------------------------------------------------
        st.subheader("8.6 Correlation matrix between holdings")

        corr_matrix = ret_subset.corr()

        fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
        cax = ax_corr.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax_corr.set_xticks(range(len(corr_matrix.columns)))
        ax_corr.set_yticks(range(len(corr_matrix.columns)))
        ax_corr.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
        ax_corr.set_yticklabels(corr_matrix.columns)
        ax_corr.set_title("Correlation matrix – monthly returns")
        fig_corr.colorbar(cax, ax=ax_corr, fraction=0.046, pad=0.04)
        fig_corr.tight_layout()

        buf_corr = BytesIO()
        fig_corr.savefig(buf_corr, format="png", dpi=150)
        buf_corr.seek(0)

        st.pyplot(fig_corr)
        st.download_button(
            label="Download 8.6 – Correlation matrix (PNG)",
            data=buf_corr.getvalue(),
            file_name="sheet8_correlation_matrix.png",
            mime="image/png",
        )

        csv_corr = corr_matrix.to_csv().encode("utf-8")
        st.download_button(
            label="Download 8.6 – Correlation matrix (CSV)",
            data=csv_corr,
            file_name="sheet8_correlation_matrix.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
