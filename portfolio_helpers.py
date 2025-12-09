# portfolio_helpers.py
# All the portfolio-level helpers:
# - Suggested active weights
# - Portfolio returns
# - Annualized stats
# - Sharpe ratio and Information Ratio

from typing import Tuple

import numpy as np
import pandas as pd


def suggest_active_weights(
    alpha_series: pd.Series,
    resid_var_series: pd.Series,
) -> pd.Series:
    """
    Suggest active portfolio weights using a simple
    "alpha / residual variance" scoring rule.

    Intuition:
        w_i ∝ alpha_i / sigma_{epsilon,i}^2

    Implementation details:
    - Replace infinite / NaN scores with zero.
    - Keep only positive scores; set negative scores to zero.
    - If everything ends up non-positive, fall back to 1/N.
    """
    scores = alpha_series / resid_var_series
    scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Only keep positive scores
    scores = scores.clip(lower=0.0)

    if scores.sum() <= 0:
        # Fallback: equal-weight portfolio
        n = len(scores)
        return pd.Series(1.0 / n, index=scores.index)

    weights = scores / scores.sum()
    return weights


def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """
    Compute portfolio returns as a weighted sum of stock returns:

        R_P,t = sum_i w_i * R_i,t

    I reindex the weights to match the return columns and fill any missing
    weights with zero.
    """
    weights = weights.reindex(returns.columns).fillna(0.0)
    port_rets = (returns * weights).sum(axis=1)
    return port_rets


def annualize_return_and_vol(
    monthly_returns: pd.Series,
) -> Tuple[float, float]:
    """
    Annualize monthly mean return and volatility using the standard
    textbook approximations:

        mu_annual   ≈ 12 * E[R_monthly]
        sigma_annual = sqrt(12) * std(R_monthly)

    This keeps the math fully transparent and easy to replicate in Excel.
    """
    mu_m = monthly_returns.mean()
    sigma_m = monthly_returns.std(ddof=1)

    mu_annual = 12.0 * mu_m
    sigma_annual = np.sqrt(12.0) * sigma_m

    return float(mu_annual), float(sigma_annual)


def compute_sharpe(
    mu_annual: float,
    sigma_annual: float,
    rf_annual: float,
) -> float:
    """
    Compute a simple Sharpe ratio:

        Sharpe = (mu_annual - rf_annual) / sigma_annual
    """
    if sigma_annual == 0:
        return np.nan
    return (mu_annual - rf_annual) / sigma_annual


def compute_information_ratio(
    active_returns: pd.Series,
) -> float:
    """
    Compute the Information Ratio for an active portfolio vs its benchmark.

    Definition here:
        IR = mean(R_active) / std(R_active)

    where:
        R_active = R_portfolio - R_benchmark
    """
    mu = active_returns.mean()
    sigma = active_returns.std(ddof=1)
    if sigma == 0:
        return np.nan
    return float(mu / sigma)
