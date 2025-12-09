# data_loading.py
# All the logic related to loading raw data from disk:
# - S&P 400 prices (Excel)
# - Fama–French 3-factor data (CSV)

from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data
def load_price_data(path: Path) -> pd.DataFrame:
    """
    Load the S&P 400 monthly price data from Excel.

    Assumptions:
    - There is a column named 'Date' with month-end dates.
    - All other columns are stock tickers with price levels.

    Returns
    -------
    prices : DataFrame
        Index: DatetimeIndex (month-end).
        Columns: one column per stock, numeric prices.
    """
    df = pd.read_excel(path, sheet_name=0)

    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in the prices file.")

    # Convert to datetime, use it as index, and sort chronologically
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Only keep numeric columns as prices
    numeric_cols = df.select_dtypes(include=["number"]).columns
    prices = df[numeric_cols].copy()

    return prices


@st.cache_data
def load_ff_factors(path: Path) -> pd.DataFrame:
    """
    Load Fama–French 3-factor data directly from the CSV file.

    I:
    - Skip the textual header.
    - Keep only rows where the first column looks like YYYYMM (6 digits).
    - Convert YYYYMM to month-end dates so they line up with my price data,
      which is also end-of-month.
    - Force the factor columns to numeric and convert from percent to decimal.
    """
    import pandas as pd

    # Read CSV, skipping the text header rows
    raw = pd.read_csv(path, skiprows=4)

    # First column is usually the date code (YYYYMM or YYYY)
    first_col = raw.columns[0]
    raw = raw.rename(columns={first_col: "YYYYMM"})

    # Keep only rows where YYYYMM is numeric
    raw = raw[pd.to_numeric(raw["YYYYMM"], errors="coerce").notna()].copy()

    # Work with it as a string and strip spaces
    raw["YYYYMM"] = raw["YYYYMM"].astype(str).str.strip()

    # ✅ Keep only 6-digit codes (YYYYMM = monthly data).
    # This drops the annual section that uses only YYYY.
    raw = raw[raw["YYYYMM"].str.len() == 6].copy()

    # ✅ Convert YYYYMM → *end of month* date
    # Example: "192607" → 1926-07-31
    raw["Date"] = (
        pd.to_datetime(raw["YYYYMM"], format="%Y%m", errors="coerce")
        + pd.offsets.MonthEnd(0)
    )
    raw = raw[pd.notnull(raw["Date"])].copy()
    raw = raw.set_index("Date").sort_index()

    # Keep the factor columns I actually need
    factor_cols = ["Mkt-RF", "SMB", "HML", "RF"]
    missing = [c for c in factor_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing expected factor columns in FF file: {missing}")

    ff = raw[factor_cols].copy()

    # Force to numeric in case they came in as strings with commas/spaces
    ff = ff.apply(lambda col: col.astype(str).str.replace(",", "").str.strip())
    ff = ff.apply(pd.to_numeric, errors="coerce")

    # Convert from percent to decimal
    ff = ff / 100.0

    return ff
