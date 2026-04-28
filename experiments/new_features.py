from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from config import Ticker
from data import HistoricalStore, compute_log_returns


def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    log_hl_sq = np.log(high / low) ** 2
    return np.sqrt(log_hl_sq.rolling(window).mean() / (4.0 * np.log(2.0)))


def garman_klass_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    return np.sqrt(gk.rolling(window).mean().clip(lower=0.0))


def realized_skew(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).skew()


def realized_kurt(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).kurt()


def vol_of_vol(rv_series: pd.Series, window: int = 21) -> pd.Series:
    return rv_series.rolling(window).std()


NEW_FEATURE_COLUMNS: list[str] = [
    "parkinson_5", "parkinson_10", "parkinson_21",
    "gk_5", "gk_10", "gk_21",
    "rskew_21", "rskew_63",
    "rkurt_21", "rkurt_63",
    "rv21_vs_market", "rv21_vs_sector",
    "rv_5_21_ratio", "rv_10_63_ratio",
    "garch_vs_rv21", "ewma_94_97_ratio", "vix_vs_rv21_ann",
    "vov_rv5", "vov_rv21",
]


def build_new_features_for_symbol(
    bars: pd.DataFrame,
    returns: pd.Series,
    baseline_block: pd.DataFrame,
) -> pd.DataFrame:
    """Build the 19 new feature columns for a single symbol.

    bars: open/high/low/close/volume, date-indexed.
    returns: log returns, date-indexed.
    baseline_block: this symbol's slice of the 28-col baseline matrix, date-indexed.

    Returns a DataFrame with NEW_FEATURE_COLUMNS, date-indexed.
    """
    out = pd.DataFrame(index=bars.index, dtype=float)

    # OHLC-based vol estimators
    out["parkinson_5"] = parkinson_vol(bars["high"], bars["low"], 5)
    out["parkinson_10"] = parkinson_vol(bars["high"], bars["low"], 10)
    out["parkinson_21"] = parkinson_vol(bars["high"], bars["low"], 21)

    out["gk_5"] = garman_klass_vol(bars["open"], bars["high"], bars["low"], bars["close"], 5)
    out["gk_10"] = garman_klass_vol(bars["open"], bars["high"], bars["low"], bars["close"], 10)
    out["gk_21"] = garman_klass_vol(bars["open"], bars["high"], bars["low"], bars["close"], 21)

    # Distribution shape
    out["rskew_21"] = realized_skew(returns, 21)
    out["rskew_63"] = realized_skew(returns, 63)
    out["rkurt_21"] = realized_kurt(returns, 21)
    out["rkurt_63"] = realized_kurt(returns, 63)

    # Ratios — use baseline columns aligned to bars index
    base = baseline_block.reindex(bars.index)
    rv_21 = base["rv_21"].replace(0, np.nan)
    rv_63 = base["rv_63"].replace(0, np.nan)
    market_rv = base["market_avg_rv21"].replace(0, np.nan)
    sector_rv = base["sector_avg_rv21"].replace(0, np.nan)
    ewma_97 = base["ewma_vol_97"].replace(0, np.nan)
    garch_var = base["garch_forecast_var"].clip(lower=0.0)

    out["rv21_vs_market"] = base["rv_21"] / market_rv
    out["rv21_vs_sector"] = base["rv_21"] / sector_rv
    out["rv_5_21_ratio"] = base["rv_5"] / rv_21
    out["rv_10_63_ratio"] = base["rv_10"] / rv_63
    out["garch_vs_rv21"] = np.sqrt(garch_var) / rv_21
    out["ewma_94_97_ratio"] = base["ewma_vol_94"] / ewma_97
    out["vix_vs_rv21_ann"] = base["vix_level"] / (base["rv_21"] * np.sqrt(252.0)).replace(0, np.nan)

    # Vol-of-vol: rolling std of an RV series
    out["vov_rv5"] = vol_of_vol(base["rv_5"], 21)
    out["vov_rv21"] = vol_of_vol(base["rv_21"], 21)

    # Replace any inf that snuck through (e.g., near-zero denominators that weren't exactly 0)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out[NEW_FEATURE_COLUMNS]


def attach_new_features(
    baseline_feature_df: pd.DataFrame,
    store: HistoricalStore,
    watchlist: list[Ticker],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Append the 19 new features to the baseline feature matrix.

    baseline_feature_df: MultiIndex (symbol, date) DataFrame with 28 baseline cols.
    Returns: MultiIndex (symbol, date) DataFrame with 47 cols, inner-joined on (symbol, date).
    """
    per_symbol: dict[str, pd.DataFrame] = {}
    symbols_in_baseline = set(baseline_feature_df.index.get_level_values("symbol").unique())

    for ticker in watchlist:
        sym = ticker.symbol
        if sym not in symbols_in_baseline:
            continue
        bars = store.get_bars(sym, start, end)
        if bars.empty:
            continue
        returns = compute_log_returns(bars["close"])
        baseline_block = baseline_feature_df.loc[sym]
        # Align date dtypes — both should be DatetimeIndex but normalize defensively
        bars.index = pd.to_datetime(bars.index)
        baseline_block = baseline_block.copy()
        baseline_block.index = pd.to_datetime(baseline_block.index)
        new_block = build_new_features_for_symbol(bars, returns, baseline_block)
        # Combine baseline + new on the same date axis
        combined = pd.concat([baseline_block, new_block.reindex(baseline_block.index)], axis=1)
        per_symbol[sym] = combined

    if not per_symbol:
        cols = list(baseline_feature_df.columns) + NEW_FEATURE_COLUMNS
        return pd.DataFrame(columns=cols)

    result = pd.concat(per_symbol, names=["symbol", "date"])
    return result
