# fundamentals_helpers.py

from typing import List
import pandas as pd
import yfinance as yf


def fetch_yahoo_fundamentals(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch basic fundamental data for each ticker using Yahoo Finance.

    This helper is used to build the "portfolio analysis" style tables
    (sector, market cap, style, etc.) for the written report.
    """
    rows = []

    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            info = tk.info  # dict with fundamentals and company profile

            market_cap = info.get("marketCap")
            pe_ratio = info.get("trailingPE")
            pb_ratio = info.get("priceToBook")
            dividend_yield = info.get("dividendYield")
            beta = info.get("beta")
            sector = info.get("sector")
            industry = info.get("industry")
            long_name = info.get("longName")

            rows.append(
                {
                    "ticker": ticker,
                    "name": long_name,
                    "sector": sector,
                    "industry": industry,
                    "market_cap": market_cap,
                    "pe_ratio": pe_ratio,
                    "pb_ratio": pb_ratio,
                    "dividend_yield": dividend_yield,
                    "beta": beta,
                }
            )
        except Exception:
            # If Yahoo fails for one ticker, we still want the others
            rows.append(
                {
                    "ticker": ticker,
                    "name": None,
                    "sector": None,
                    "industry": None,
                    "market_cap": None,
                    "pe_ratio": None,
                    "pb_ratio": None,
                    "dividend_yield": None,
                    "beta": None,
                }
            )

    df = pd.DataFrame(rows).set_index("ticker")

    # Simple size buckets for market cap (USD):
    def _size_bucket(mcap):
        if mcap is None or pd.isna(mcap):
            return "Unknown"
        if mcap >= 10_000_000_000:
            return "Large cap"
        if mcap >= 2_000_000_000:
            return "Mid cap"
        return "Small cap"

    df["size_bucket"] = df["market_cap"].apply(_size_bucket)

    # Very simple style classification based on P/B:
    # this is just to approximate a Value / Blend / Growth label.
    def _style_bucket(pb):
        if pb is None or pd.isna(pb):
            return "Blend"
        if pb < 1.5:
            return "Value"
        if pb > 3.0:
            return "Growth"
        return "Blend"

    df["style_bucket"] = df["pb_ratio"].apply(_style_bucket)

    return df
