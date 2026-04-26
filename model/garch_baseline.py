import logging
import math
from dataclasses import dataclass

import pandas as pd

from features.garch import PCT_SCALE, GarchFit, fit_garch11

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GarchPathForecast:
    """Multi-step variance forecast and aggregation for a single base date.
    daily_variances are σ²_{t+1..t+H} in decimal² units."""
    daily_variances: list[float]

    @property
    def total_variance(self) -> float:
        return sum(self.daily_variances)

    @property
    def avg_daily_variance(self) -> float:
        return self.total_variance / len(self.daily_variances)

    @property
    def daily_equiv_vol(self) -> float:
        return math.sqrt(self.avg_daily_variance)


def garch_forecast_path(fit: GarchFit, current_var_pct_sq: float, horizon: int) -> GarchPathForecast:
    """Closed-form GARCH(1,1) variance path:
        σ²_{t+1} = current_var_pct_sq
        σ²_{t+h} = ω + (α + β) · σ²_{t+h-1}  for h ≥ 2
    Inputs are in pct² units; output converted to decimal² units."""
    if horizon < 1:
        raise ValueError(f"horizon must be ≥1, got {horizon}")
    persistence = fit.alpha + fit.beta
    var_pct = current_var_pct_sq
    path_pct = [var_pct]
    for _ in range(horizon - 1):
        var_pct = fit.omega + persistence * var_pct
        path_pct.append(var_pct)
    scale = PCT_SCALE ** 2
    return GarchPathForecast(daily_variances=[v / scale for v in path_pct])


class GARCHBaseline:
    def __init__(self, refit_every: int = 21, min_history: int = 252):
        self.refit_every = refit_every
        self.min_history = min_history

    def predict_forward_rv(self, returns: pd.Series, horizon: int) -> float:
        """Single-shot live prediction. Fits, then closed-form multi-step."""
        clean = returns.dropna()
        if len(clean) < self.min_history:
            raise ValueError(
                f"need ≥{self.min_history} observations, got {len(clean)}"
            )
        fit = fit_garch11(returns)
        path = garch_forecast_path(fit, fit.next_forecast_var, horizon)
        return path.daily_equiv_vol

    def walk_forward_evaluate(
        self,
        returns: pd.Series,
        horizons: tuple[int, ...] = (5, 10, 21),
    ) -> pd.DataFrame:
        """Walk forward, refit every `refit_every`, recurse σ² daily, predict
        forward RV at each horizon, look ahead for the actual realized RV."""
        cols: list[str] = []
        for h in horizons:
            cols += [f"pred_rv_{h}", f"actual_rv_{h}"]
        cols += ["omega", "alpha", "beta", "resid_lb_pvalue", "aic", "bic"]
        out = pd.DataFrame(index=returns.index, columns=cols, dtype=float)

        fit: GarchFit | None = None
        var_pct_sq = float("nan")
        days_since_refit = self.refit_every

        n = len(returns)
        for i, t in enumerate(returns.index):
            if i < self.min_history - 1:
                continue

            if days_since_refit >= self.refit_every:
                try:
                    fit = fit_garch11(returns.iloc[: i + 1])
                    var_pct_sq = fit.next_forecast_var
                    days_since_refit = 0
                except Exception as e:
                    logger.warning("GARCH refit failed at %s: %s", t, e)
                    if fit is None:
                        days_since_refit += 1
                        continue
            else:
                r_pct = float(returns.iloc[i]) * PCT_SCALE
                var_pct_sq = fit.omega + fit.alpha * (r_pct ** 2) + fit.beta * var_pct_sq

            days_since_refit += 1

            for h in horizons:
                path = garch_forecast_path(fit, var_pct_sq, h)
                out.at[t, f"pred_rv_{h}"] = path.daily_equiv_vol
                future_end = i + h
                if future_end < n:
                    actual = returns.iloc[i + 1 : future_end + 1].std(ddof=0)
                    out.at[t, f"actual_rv_{h}"] = float(actual)

            out.at[t, "omega"] = fit.omega
            out.at[t, "alpha"] = fit.alpha
            out.at[t, "beta"] = fit.beta
            out.at[t, "resid_lb_pvalue"] = fit.resid_lb_pvalue
            out.at[t, "aic"] = fit.aic
            out.at[t, "bic"] = fit.bic

        return out
