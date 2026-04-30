"""OHLC-based volatility estimators.

Parkinson (high/low only) and Garman-Klass (open/high/low/close) are more
statistically efficient than close-to-close realized vol because they capture
intraday range that a close-to-close estimator misses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Rolling Parkinson estimator: sqrt( mean(ln(H/L)^2) / (4 ln 2) )."""
    log_hl_sq = np.log(high / low) ** 2
    return np.sqrt(log_hl_sq.rolling(window).mean() / (4.0 * np.log(2.0)))


def garman_klass_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling Garman-Klass estimator. Most efficient classical OHLC estimator.

    GK_t = 0.5 * ln(H_t/L_t)^2 - (2 ln 2 - 1) * ln(C_t/O_t)^2
    Returns sqrt(rolling mean of GK), clipped at zero (GK can be slightly
    negative on rare days where close-to-open dominates the range).
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    return np.sqrt(gk.rolling(window).mean().clip(lower=0.0))
