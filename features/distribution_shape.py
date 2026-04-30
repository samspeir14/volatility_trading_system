"""Realized return distribution shape features.

Skewness and kurtosis of recent returns. Negative skew often precedes vol
spikes; high kurtosis flags fat-tailed regimes. Only h=21 selects these in
the production feature set.
"""
from __future__ import annotations

import pandas as pd


def realized_skew(returns: pd.Series, window: int) -> pd.Series:
    """Rolling sample skewness of log returns."""
    return returns.rolling(window).skew()


def realized_kurt(returns: pd.Series, window: int) -> pd.Series:
    """Rolling sample (excess) kurtosis of log returns."""
    return returns.rolling(window).kurt()
