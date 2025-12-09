# prospectus_text.py
# Helper to generate a "mutual fund style" narrative based on
# the portfolio stats and factor exposures.

from typing import List

from regressions import (
    classify_beta_hml,
    classify_beta_mkt,
    classify_beta_smb,
)


def generate_prospectus_text(
    investor_profile: str,
    selected_stocks: List[str],
    port_mu: float,
    port_sigma: float,
    port_beta_mkt: float,
    port_beta_smb: float,
    port_beta_hml: float,
    sharpe: float,
    benchmark_name: str,
) -> str:
    """
    Generate a high-level text summary that I can later refine
    and copy into my written report / prospectus.
    """
    profile_blurb = {
        "Kim (25, early-career, stable high-quality stocks)": (
            "This portfolio is designed for a young professional investor "
            "who already holds broad index exposure and now wants a "
            "concentrated sleeve of well-managed, fundamentally sound "
            "companies."
        ),
        "Nicole (52, retired, dividend focus)": (
            "This portfolio targets a recently retired investor who relies "
            "on dividend income and values stability and dividend-growth "
            "consistency over speculative upside."
        ),
        "Peter (mid-30s, late saver, high risk tolerance)": (
            "This portfolio is tailored for an investor with a late start "
            "in retirement savings, a strong current income, and a very "
            "high tolerance for risk and short-term volatility."
        ),
    }.get(investor_profile, "")

    text = f"""
**Investment Objective**

This concentrated six-stock portfolio is built for: **{investor_profile}**.  
{profile_blurb}

The portfolio allocates capital across the following names:  
**{", ".join(selected_stocks)}**.

**Risk & Return Profile**

Based on historical monthly data, the portfolio exhibits an **annualized expected return** of approximately **{port_mu:.2%}** and an **annualized volatility** of about **{port_sigma:.2%}**.  
The implied **Sharpe ratio**, using a simple annualized risk-free proxy, is **{sharpe:.2f}**, which I interpret as the risk-adjusted payoff per unit of volatility.

From a factor-exposure standpoint (Fama-French 3-factor model), the portfolio has:

- **Market beta (MKT)** ≈ **{port_beta_mkt:.2f}**, indicating that the portfolio is {classify_beta_mkt(port_beta_mkt).lower()} relative to the broad equity market.
- **Size exposure (SMB)** ≈ **{port_beta_smb:.2f}**, which corresponds to a **{classify_beta_smb(port_beta_smb, 0.01).lower()}** profile.
- **Value/Growth tilt (HML)** ≈ **{port_beta_hml:.2f}**, consistent with a **{classify_beta_hml(port_beta_hml, 0.01).lower()}** orientation.

**Benchmarking and Active Strategy**

I benchmark this portfolio against **{benchmark_name}**. Relative to that benchmark, the portfolio intentionally overweights stocks where I see stronger expected alpha, either because of mispricing, superior fundamentals, or better risk/reward trade-offs. The active allocation seeks to convert those idiosyncratic views into **positive active return** while keeping overall risk at a level consistent with the investor’s profile.

The portfolio’s construction is grounded in CAPM and Fama–French diagnostics: I use regression-based estimates of beta and residual risk to tilt the weights toward stocks with stronger historical alpha per unit of residual variance, and then refine those weights qualitatively based on the business models, quality metrics, and growth prospects of each firm.

This summary is meant to be read alongside the detailed tables in the dashboard, which show the step-by-step calculations for monthly returns, excess returns, CAPM regressions, Fama–French regressions, and portfolio-level factor exposures.
    """.strip()

    return text
