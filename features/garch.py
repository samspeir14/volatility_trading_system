import logging
from dataclasses import dataclass

import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

logger = logging.getLogger(__name__)

PCT_SCALE = 100.0  # arch fits in pct units for numerical stability


@dataclass(frozen=True)
class GarchFit:
    omega: float           # in pct² units
    alpha: float
    beta: float
    next_forecast_var: float  # σ²_{T+1} in pct² units
    resid_lb_pvalue: float    # Ljung-Box p-value at lag 10


def fit_garch11(returns: pd.Series) -> GarchFit:
    am = arch_model(
        returns.dropna() * PCT_SCALE,
        vol="Garch",
        p=1,
        q=1,
        mean="zero",
        rescale=False,
    )
    res = am.fit(disp="off", show_warning=False)
    fcast = float(res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
    lb = acorr_ljungbox(res.std_resid.dropna(), lags=[10], return_df=True)
    return GarchFit(
        omega=float(res.params["omega"]),
        alpha=float(res.params["alpha[1]"]),
        beta=float(res.params["beta[1]"]),
        next_forecast_var=fcast,
        resid_lb_pvalue=float(lb["lb_pvalue"].iloc[0]),
    )


def garch_features_walk_forward(
    returns: pd.Series,
    refit_every: int = 21,
    min_history: int = 252,
) -> pd.DataFrame:
    """
    Refit GARCH(1,1) parameters every `refit_every` rows starting at index
    `min_history - 1`. Between refits, recurse the conditional variance daily
    using the GARCH(1,1) equation σ²_{t+1} = ω + α·r²_t + β·σ²_t with the
    *current* parameters and the *new* realized return.

    The forecast recorded at row t is σ²_{t+1} (the 1-step-ahead forecast made
    after observing r_t). Point-in-time, no leakage.
    """
    out = pd.DataFrame(
        index=returns.index,
        columns=["garch_forecast_var", "garch_resid_lb_pvalue"],
        dtype=float,
    )
    fit: GarchFit | None = None
    var_pct_sq: float = float("nan")
    days_since_refit = refit_every  # force initial fit

    for i, t in enumerate(returns.index):
        if i < min_history - 1:
            continue

        if days_since_refit >= refit_every:
            try:
                fit = fit_garch11(returns.iloc[: i + 1])
                var_pct_sq = fit.next_forecast_var
                days_since_refit = 0
            except Exception as e:
                logger.warning("GARCH refit failed at %s: %s", t, e)
                if fit is None:
                    days_since_refit += 1
                    continue
                # else: fall through and recurse with stale params
        else:
            r_pct = float(returns.iloc[i]) * PCT_SCALE
            var_pct_sq = fit.omega + fit.alpha * (r_pct ** 2) + fit.beta * var_pct_sq

        days_since_refit += 1
        out.loc[t, "garch_forecast_var"] = var_pct_sq / (PCT_SCALE ** 2)
        out.loc[t, "garch_resid_lb_pvalue"] = fit.resid_lb_pvalue

    return out
