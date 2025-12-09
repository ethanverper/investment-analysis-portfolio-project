# returns_helpers.py
# All the helper functions related to returns and aligning them
# with the Fama–French factor data.

from typing import Tuple

import pandas as pd


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple monthly returns from price data.

    Formula:
        R_t = (P_t / P_{t-1}) - 1

    Implementation detail:
    - I use pandas' pct_change(), then drop the first NaN row.
    """
    prices = prices.sort_index()
    rets = prices.pct_change().dropna(how="all")
    return rets


def align_returns_and_factors(
    returns: pd.DataFrame,
    ff: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align stock returns with Fama–French factors on a common date index.

    I restrict both datasets to the intersection of their date indices.

    Returns
    -------
    aligned_returns : DataFrame
    aligned_ff : DataFrame
    """
    common_idx = returns.index.intersection(ff.index)

    aligned_returns = returns.loc[common_idx].copy()
    aligned_ff = ff.loc[common_idx].copy()

    return aligned_returns, aligned_ff


def compute_excess_returns(
    returns: pd.DataFrame,
    rf: pd.Series,
) -> pd.DataFrame:
    """
    Compute excess returns for each stock:

        R_i,t^excess = R_i,t - R_f,t

    I broadcast RF across columns using pandas' sub with axis=0.
    """
    rf_aligned = rf.reindex(returns.index)
    excess = returns.sub(rf_aligned, axis=0)
    return excess
