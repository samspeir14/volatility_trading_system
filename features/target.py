"""Next-day realized-vol target for the h=1 within-stock model.

The target is the deviation of tomorrow's log Garman-Klass vol from the
stock's own trailing 63-day mean log vol:

    lv_t  = log(gk_vol_t + GK_EPS)
    b_t   = mean(lv_{t-62..t})        # >= 40 observations, data <= t only
    y_t   = lv_{t+1} - b_t

Demeaning by b_t makes pooling across tickers legitimate: the model never
gets credit for knowing a ticker's vol level, only for timing deviations
from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.ohlc_vol import garman_klass_vol, parkinson_vol

GK_EPS = 1e-8
BASELINE_WINDOW = 63
BASELINE_MIN_OBS = 40


def _positive_or_nan(s: pd.Series) -> pd.Series:
    return s.where(s > 0)


def daily_ohlc_vol(bars: pd.DataFrame) -> pd.Series:
    """Single-day vol proxy per row of an OHLC frame (HistoricalStore.get_bars
    output). Priority per day: Garman-Klass -> Parkinson -> abs close-to-close
    log return. Nonpositive prices are treated as missing so a bad field falls
    through to the next estimator instead of producing log(<=0). A day where
    every estimator reads exactly zero (high == low == open == close: trading
    halt or vendor placeholder bar) yields NaN rather than 0 — log(0 + eps)
    = -18.4 would poison the 63-day baseline for three months and blow up the
    linear HAR forecast."""
    o = _positive_or_nan(bars["open"])
    h = _positive_or_nan(bars["high"])
    low = _positive_or_nan(bars["low"])
    c = _positive_or_nan(bars["close"])
    # window=1 reduces the rolling mean to the day's own estimate
    gk = _positive_or_nan(garman_klass_vol(o, h, low, c, window=1))
    park = _positive_or_nan(parkinson_vol(h, low, window=1))
    c2c = _positive_or_nan(np.log(c / c.shift(1)).abs())
    return gk.fillna(park).fillna(c2c)


def daily_total_vol(bars: pd.DataFrame) -> pd.Series:
    """Single-day TOTAL vol proxy: sqrt(overnight_gap² + intraday²), with
    intraday = daily_ohlc_vol (GK-first). The GK family measures the session
    only; options price close-to-close, and the overnight gap carries ~40%
    of close-to-close variance on this watchlist (replay_analysis
    2026-08-14). LAB-ONLY until a checkpoint gate vote — the production
    target stays GK. Rows where the intraday estimator is NaN (halt /
    placeholder bar) stay NaN; a missing gap (first row, bad open) degrades
    to intraday-only."""
    o = _positive_or_nan(bars["open"])
    c = _positive_or_nan(bars["close"])
    gap = np.log(o / c.shift(1))
    intraday = daily_ohlc_vol(bars)
    total = np.sqrt(gap.pow(2).fillna(0.0) + intraday.pow(2))
    return total.where(intraday.notna())


def log_vol(vol: pd.Series) -> pd.Series:
    return np.log(vol + GK_EPS)


def rolling_log_vol_baseline(
    lv: pd.Series,
    window: int = BASELINE_WINDOW,
    min_obs: int = BASELINE_MIN_OBS,
) -> pd.Series:
    """b_t: trailing mean of log vol over `window` rows ending AT t (inclusive),
    NaN until `min_obs` observations have accrued. Uses data <= t only."""
    return lv.rolling(window, min_periods=min_obs).mean()


def build_h1_deviation_target(
    bars_by_symbol: dict[str, pd.DataFrame],
    vol_fn=daily_ohlc_vol,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (y, b, lv), each indexed by (symbol, date):

        lv[(s,t)] = log(vol_fn_t + GK_EPS)
        b[(s,t)]  = rolling_log_vol_baseline(lv_s) at t
        y[(s,t)]  = lv[(s,t+1)] - b[(s,t)]

    `vol_fn` is the single-day vol proxy (default GK-first daily_ohlc_vol;
    the lab's Yang-Zhang arm passes daily_total_vol). y is NaN-dropped
    (needs both tomorrow's vol and today's baseline); b and lv are returned
    full-length for level reconstruction and HAR inputs."""
    y_parts: list[pd.Series] = []
    b_parts: list[pd.Series] = []
    lv_parts: list[pd.Series] = []
    for symbol, bars in bars_by_symbol.items():
        if bars.empty:
            continue
        lv = log_vol(vol_fn(bars))
        b = rolling_log_vol_baseline(lv)
        y = lv.shift(-1) - b
        idx = pd.MultiIndex.from_product(
            [[symbol], bars.index], names=["symbol", "date"]
        )
        lv_parts.append(pd.Series(lv.to_numpy(), index=idx))
        b_parts.append(pd.Series(b.to_numpy(), index=idx))
        y_parts.append(pd.Series(y.to_numpy(), index=idx))

    if not y_parts:
        empty_idx = pd.MultiIndex.from_arrays([[], []], names=["symbol", "date"])
        empty = pd.Series(dtype=float, index=empty_idx)
        return (
            empty.rename("target_dev_h1"),
            empty.rename("log_vol_baseline"),
            empty.rename("log_vol"),
        )

    y = pd.concat(y_parts).dropna().rename("target_dev_h1")
    b = pd.concat(b_parts).rename("log_vol_baseline")
    lv = pd.concat(lv_parts).rename("log_vol")
    return y, b, lv
