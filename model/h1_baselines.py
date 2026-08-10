"""Parameter-light baseline forecasters for the h=1 deviation target.

Each returns a (symbol, date)-indexed Series of deviation forecasts
dev_hat[(s,t)] for day t+1, computed from data <= t only, in the same units
as the training target (log vol minus the 63-day log-vol baseline b_t).
Level forecasts reconstruct as exp(b_t + dev_hat).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.realized_vol import ewma_vol
from features.target import GK_EPS
from model.garch_baseline import GARCHBaseline


def persistence_deviation(lv: pd.Series, b: pd.Series) -> pd.Series:
    """Random walk in deviation space: dev_hat_t = lv_t - b_t (today's
    deviation persists to tomorrow). lv and b are the (symbol, date) Series
    from features.target.build_h1_deviation_target."""
    return (lv - b).dropna().rename("dev_persistence")


def ewma_deviation(
    returns_by_symbol: dict[str, pd.Series],
    b: pd.Series,
    lam: float = 0.94,
) -> pd.Series:
    """RiskMetrics EWMA vol (lambda=0.94) as a deviation forecast:
    dev_hat_t = log(ewma_vol_t + eps) - b_t. The EWMA at t uses returns <= t."""
    parts: list[pd.Series] = []
    for symbol, returns in returns_by_symbol.items():
        ew = ewma_vol(returns, lam)
        idx = pd.MultiIndex.from_product(
            [[symbol], returns.index], names=["symbol", "date"]
        )
        parts.append(pd.Series(np.log(ew.to_numpy() + GK_EPS), index=idx))
    if not parts:
        return pd.Series(dtype=float, name="dev_ewma")
    lv_ew = pd.concat(parts)
    return (lv_ew - b).dropna().rename("dev_ewma")


def garch_deviation(
    returns_by_symbol: dict[str, pd.Series],
    b: pd.Series,
    refit_every: int = 21,
    min_history: int = 252,
) -> pd.Series:
    """GARCH(1,1) 1-step vol forecast as a deviation:
    dev_hat_t = log(pred_rv_1_t + eps) - b_t, where pred_rv_1 at row t is the
    sigma_{t+1} forecast from the walk-forward GARCH (refit every
    `refit_every` rows, daily variance recursion between refits)."""
    baseline = GARCHBaseline(refit_every=refit_every, min_history=min_history)
    parts: list[pd.Series] = []
    for symbol, returns in returns_by_symbol.items():
        if len(returns.dropna()) < min_history:
            continue
        df = baseline.walk_forward_evaluate(returns, horizons=(1,))
        pred = df["pred_rv_1"]
        idx = pd.MultiIndex.from_product(
            [[symbol], pred.index], names=["symbol", "date"]
        )
        parts.append(pd.Series(np.log(pred.to_numpy() + GK_EPS), index=idx))
    if not parts:
        return pd.Series(dtype=float, name="dev_garch")
    lv_garch = pd.concat(parts)
    return (lv_garch - b).dropna().rename("dev_garch")
