import pandas as pd
import statsmodels.api as sm
from typing import Dict, Any


def run_capm_regression_for_stock(
    stock_excess: pd.Series,
    mkt_excess: pd.Series,
) -> Dict[str, Any]:
    """
    Run a single-stock CAPM regression:
    R_i,t^excess = alpha_i + beta_i * (Mkt-RF)_t + eps_t

    I keep everything in a dict so it is easy to turn into a summary table later.
    """
    # Align by date and drop any missing values
    df = pd.concat(
        {"Ri_excess": stock_excess, "Mkt_excess": mkt_excess},
        axis=1
    ).dropna()

    if df.shape[0] == 0:
        # Edge case: no overlap in dates
        return {
            "alpha": float("nan"),
            "beta": float("nan"),
            "alpha_t": float("nan"),
            "beta_t": float("nan"),
            "r2": float("nan"),
            "n_obs": 0,
        }

    y = df["Ri_excess"]
    X = sm.add_constant(df["Mkt_excess"])
    model = sm.OLS(y, X).fit()

    alpha = model.params["const"]
    beta = model.params["Mkt_excess"]
    alpha_t = model.tvalues["const"]
    beta_t = model.tvalues["Mkt_excess"]
    r2 = model.rsquared
    n_obs = int(model.nobs)

    return {
        "alpha": alpha,
        "beta": beta,
        "alpha_t": alpha_t,
        "beta_t": beta_t,
        "r2": r2,
        "n_obs": n_obs,
        "results_obj": model,  # I keep the full statsmodels result for later if needed
    }


def run_capm_regressions_panel(
    excess_returns_df: pd.DataFrame,
    ff_factors_aligned: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """
    Run CAPM regressions for all stocks in the panel.

    excess_returns_df: DataFrame with Date index, columns = tickers,
                       containing R_i,t^excess.
    ff_factors_aligned: DataFrame with Date index and at least the
                        'Mkt-RF' column (already aligned to the same
                        monthly dates as the excess-returns panel).
    """
    mkt_excess = ff_factors_aligned["Mkt-RF"]

    capm_results: Dict[str, Dict[str, Any]] = {}
    for ticker in excess_returns_df.columns:
        stock_excess = excess_returns_df[ticker]
        capm_results[ticker] = run_capm_regression_for_stock(
            stock_excess=stock_excess,
            mkt_excess=mkt_excess,
        )

    return capm_results


def capm_results_to_frame(capm_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert the capm_results dict into a tidy summary table.

    Rows = tickers, columns = alpha, beta, t-stats, R^2, n_obs.
    """
    rows = []
    for ticker, res in capm_results.items():
        rows.append(
            {
                "ticker": ticker,
                "alpha": res["alpha"],
                "beta": res["beta"],
                "alpha_t": res["alpha_t"],
                "beta_t": res["beta_t"],
                "r2": res["r2"],
                "n_obs": res["n_obs"],
            }
        )

    df = pd.DataFrame(rows).set_index("ticker")
    return df
