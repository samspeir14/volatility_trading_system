import logging
from datetime import date

import numpy as np
import pandas as pd

from pathlib import Path

from config import MARKET_INDICES, Ticker
from data import (
    AsyncTradierClient,
    HistoricalStore,
    compute_log_returns,
    fetch_and_cache,
)
from data.macro_calendar import events_by_label
from features.cross_ticker import (
    RATIO_FEATURE_COLUMNS,
    add_ratio_features,
    market_avg_rv,
    rolling_corr_with_spy,
    sector_avg_rv,
    vix_features,
)
from features.distribution_shape import realized_kurt, realized_skew
from features.garch import garch_features_walk_forward
from features.ohlc_vol import garman_klass_vol, parkinson_vol
from features.target import daily_ohlc_vol, log_vol, rolling_log_vol_baseline
from features.realized_vol import (
    acf_squared_returns,
    ewma_vol,
    har_rv_components,
    rolling_rv,
)
from features.technical_indicators import (
    atr,
    bollinger_width,
    intraday_range,
    macd_histogram,
    rsi,
    volume_ratio,
)

logger = logging.getLogger(__name__)

# 28 baseline feature columns — what FeaturePipeline shipped with prior to LightGBM integration.
BASELINE_FEATURE_COLUMNS: list[str] = [
    # Realized vol (12)
    "rv_5", "rv_10", "rv_21", "rv_63",
    "ewma_vol_94", "ewma_vol_97",
    "har_rv_daily", "har_rv_weekly", "har_rv_monthly",
    "acf_sq_ret_lag1", "acf_sq_ret_lag5", "acf_sq_ret_lag10",
    # GARCH (2)
    "garch_forecast_var", "garch_resid_lb_pvalue",
    # Technical (8)
    "bb_width", "bb_width_roc",
    "macd_hist_mag", "rsi_14",
    "volume_ratio",
    "atr_14", "atr_roc",
    "intraday_range",
    # Cross-ticker (6)
    "market_avg_rv21", "sector_avg_rv21",
    "vix_level", "vix9d_to_vix", "vix3m_to_vix",
    "corr_spy_21",
]

# OHLC-based vol estimators (added 2026-04-28 with LightGBM integration)
OHLC_VOL_COLUMNS: list[str] = [
    "parkinson_5", "parkinson_10", "parkinson_21",
    "gk_5", "gk_10", "gk_21",
]

# Distribution shape — not in the h=1 top-20 (rskew_63 aside), but the
# pipeline computes them so offline feature studies see the full matrix.
DISTRIBUTION_SHAPE_COLUMNS: list[str] = [
    "rskew_21", "rskew_63",
    "rkurt_21", "rkurt_63",
]

# Signed-return / leverage-effect features (added 2026-08-14). Everything
# above is magnitude-only; the classic asymmetry — negative returns predict
# higher next-day vol — needs the sign.
LEVERAGE_COLUMNS: list[str] = [
    "ret_1", "ret_5", "ret_21", "ret_1_neg",
]

# Overnight/intraday decomposition (added 2026-08-14): close-to-open gap
# vol carries information the open-to-close range does not.
OVERNIGHT_COLUMNS: list[str] = [
    "overnight_gap", "overnight_vol_21", "overnight_to_intraday_21",
]

# Deterministic calendar dummies (added 2026-08-14): day-of-week and
# expiration effects on next-day vol.
CALENDAR_COLUMNS: list[str] = [
    "dow_mon", "dow_fri", "opex_friday", "month_end",
]

# Macro release dummies (added 2026-08-14): scheduled FOMC/CPI/PPI/PCE/NFP
# days from data/macro_calendar.py (backfilled 2022-2026 from the official
# OMB schedule PDFs). "tomorrow" is the next BUSINESS day after the bar —
# the event the h=1 target's window will contain.
MACRO_FEATURE_COLUMNS: list[str] = [
    "macro_any_tomorrow", "macro_any_today",
    "fomc_tomorrow", "cpi_tomorrow", "nfp_tomorrow",
]

# Earnings-distance features (added 2026-08-14; data/cache/earnings_history.csv
# via scripts/backfill_earnings_history.py). Impact date = the trading day
# containing the reaction (report date +1 BDay for after-market-close
# reports). Uses the realized calendar — in reality dates are announced weeks
# ahead, so treating the near-term schedule as known is point-in-time honest
# enough for a 21-day-capped horizon. NaN for ETFs / missing data.
EARNINGS_FEATURE_COLUMNS: list[str] = [
    "earnings_tomorrow", "days_to_earnings", "days_since_earnings",
]

# Implied-vol features (added 2026-08-14; data/cache/iv_history.csv via
# scripts/backfill_iv_history.py — DoltHub composite ~30d IV, sparse ~3/week
# before 2025). Forward-filled onto bar dates with a 5-day staleness limit.
# Levels have a source-specific offset; changes/ranks carry the signal.
IV_FEATURE_COLUMNS: list[str] = [
    "iv_level", "iv_minus_hv", "iv_chg_5", "iv_pctile_252",
]

# h=1 target-side columns (added with the within-stock deviation model).
# gk_1 is the single-day GK vol proxy (with Parkinson / |c2c return| fallback),
# log_gk_baseline_63 is b_t (the 63-day trailing mean of log gk_1, min 40 obs),
# dev_gk / har_dev_* are demeaned HAR components (inputs to the HAR baseline
# and candidate LGBM features), garch_persistence is alpha+beta for the
# term-structure projection. All trailing / point-in-time.
H1_FEATURE_COLUMNS: list[str] = [
    "gk_1", "log_gk_1", "log_gk_baseline_63",
    "dev_gk", "har_dev_5", "har_dev_22",
    "garch_persistence",
]

# Full feature matrix produced by build_features():
# 28 baseline + 6 OHLC vol + 4 distribution + 7 ratios + 7 h=1
# + 4 leverage + 3 overnight + 4 calendar + 5 macro + 3 earnings + 4 IV = 75.
FEATURE_COLUMNS: list[str] = (
    BASELINE_FEATURE_COLUMNS
    + OHLC_VOL_COLUMNS
    + DISTRIBUTION_SHAPE_COLUMNS
    + RATIO_FEATURE_COLUMNS
    + H1_FEATURE_COLUMNS
    + LEVERAGE_COLUMNS
    + OVERNIGHT_COLUMNS
    + CALENDAR_COLUMNS
    + MACRO_FEATURE_COLUMNS
    + EARNINGS_FEATURE_COLUMNS
    + IV_FEATURE_COLUMNS
)


# The production h=1 feature set. Selection criterion (since 2026-08-13):
# the lab nests top-N candidates by gain-importance mean rank and freezes the
# subset with the best OOS within-ticker deviation R² — the strategy's
# singular metric. FROZEN 2026-08-14 from the EC2 lab run: the FULL 63-column
# set won again (within R² +0.1423 vs +0.1410 for top-20; blend route
# +0.1466). Kept as an explicit list so later feature-matrix additions cannot
# silently change the production set — re-freeze only from a lab WINNER
# printout. Do not load from CSV at runtime.
HORIZON_FEATURE_SETS: dict[int, list[str]] = {
    1: [
        "rv_5", "rv_10", "rv_21", "rv_63",
        "ewma_vol_94", "ewma_vol_97",
        "har_rv_daily", "har_rv_weekly", "har_rv_monthly",
        "acf_sq_ret_lag1", "acf_sq_ret_lag5", "acf_sq_ret_lag10",
        "garch_forecast_var", "garch_resid_lb_pvalue",
        "bb_width", "bb_width_roc",
        "macd_hist_mag", "rsi_14",
        "volume_ratio",
        "atr_14", "atr_roc",
        "intraday_range",
        "market_avg_rv21", "sector_avg_rv21",
        "vix_level", "vix9d_to_vix", "vix3m_to_vix",
        "corr_spy_21",
        "parkinson_5", "parkinson_10", "parkinson_21",
        "gk_5", "gk_10", "gk_21",
        "rskew_21", "rskew_63",
        "rkurt_21", "rkurt_63",
        "rv21_vs_market", "rv21_vs_sector",
        "rv_5_21_ratio", "rv_10_63_ratio",
        "garch_vs_rv21", "ewma_94_97_ratio", "vix_vs_rv21_ann",
        "gk_1", "log_gk_1", "log_gk_baseline_63",
        "dev_gk", "har_dev_5", "har_dev_22",
        "garch_persistence",
        "ret_1", "ret_5", "ret_21", "ret_1_neg",
        "overnight_gap", "overnight_vol_21", "overnight_to_intraday_21",
        "dow_mon", "dow_fri", "opex_friday", "month_end",
    ],
}


def load_iv_history(path: Path) -> pd.DataFrame | None:
    """Load the DoltHub IV backfill CSV (symbol, date, iv_current, hv_current).
    Returns None (features become NaN) when the file is missing/empty."""
    if not path.exists():
        logger.warning("iv history missing at %s — IV features will be NaN", path)
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df if not df.empty else None


def load_earnings_history(path: Path) -> pd.DataFrame | None:
    """Load the DoltHub earnings backfill CSV (symbol, date, when).
    Returns None (features become NaN) when the file is missing/empty."""
    if not path.exists():
        logger.warning(
            "earnings history missing at %s — earnings features will be NaN", path,
        )
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df if not df.empty else None


class FeaturePipeline:
    def __init__(
        self,
        store: HistoricalStore,
        watchlist: list[Ticker],
        market_indices: tuple[str, ...] = MARKET_INDICES,
        garch_min_history: int = 252,
        garch_refit_every: int = 21,
        iv_history: pd.DataFrame | None = None,
        earnings_history: pd.DataFrame | None = None,
    ):
        self._store = store
        self._watchlist = watchlist
        self._indices = market_indices
        self._garch_min_history = garch_min_history
        self._garch_refit_every = garch_refit_every

        # label -> frozenset[date] for the macro dummies
        self._macro_events = events_by_label()
        self._macro_all = frozenset().union(*self._macro_events.values())

        # symbol -> sorted DatetimeIndex of earnings IMPACT days (the trading
        # day containing the reaction: report date +1 BDay for AMC reports).
        self._earnings_impacts: dict[str, pd.DatetimeIndex] = {}
        if earnings_history is not None:
            for sym, grp in earnings_history.groupby("symbol"):
                when = grp["when"].fillna("").str.lower()
                dates = pd.to_datetime(grp["date"])
                impact = dates.where(
                    ~when.str.startswith("after"),
                    dates + pd.tseries.offsets.BDay(1),
                )
                self._earnings_impacts[str(sym)] = pd.DatetimeIndex(
                    sorted(impact.unique())
                )

        # symbol -> (iv Series, hv Series) indexed by observation date
        self._iv_by_symbol: dict[str, tuple[pd.Series, pd.Series]] = {}
        if iv_history is not None:
            for sym, grp in iv_history.groupby("symbol"):
                g = grp.drop_duplicates("date").set_index("date").sort_index()
                self._iv_by_symbol[str(sym)] = (
                    pd.to_numeric(g["iv_current"], errors="coerce"),
                    pd.to_numeric(g["hv_current"], errors="coerce"),
                )

    async def ensure_data(
        self,
        client: AsyncTradierClient,
        end: date,
        lookback_years: int = 2,
    ) -> None:
        """Backfill cache for watchlist + market indices."""
        symbols = [t.symbol for t in self._watchlist] + list(self._indices)
        # +1 year buffer for the longest lookback (63-day RV + 252-day GARCH min_history)
        await fetch_and_cache(
            client, self._store, symbols,
            lookback_years=lookback_years + 1,
            today=end,
        )

    def build_features(self, start: date, end: date) -> pd.DataFrame:
        """Build the full feature matrix from cached bars. No I/O.

        Returns a MultiIndex (symbol, date) DataFrame with the 63
        FEATURE_COLUMNS. The production h=1 model pulls its frozen subset
        via HORIZON_FEATURE_SETS[1].
        """
        watchlist_symbols = [t.symbol for t in self._watchlist]
        all_symbols = watchlist_symbols + list(self._indices)
        bars = {sym: self._store.get_bars(sym, start, end) for sym in all_symbols}
        returns = {
            sym: compute_log_returns(bars[sym]["close"])
            for sym in all_symbols
            if not bars[sym].empty
        }

        per_ticker: dict[str, pd.DataFrame] = {}
        for ticker in self._watchlist:
            sym = ticker.symbol
            if sym not in returns or bars[sym].empty:
                logger.warning("No bars for %s; skipping", sym)
                continue
            per_ticker[sym] = self._build_single_ticker(sym, bars[sym], returns[sym])

        # Cross-ticker features (writes into per_ticker frames in place)
        if per_ticker:
            self._add_cross_ticker_features(per_ticker, bars, returns)

        if not per_ticker:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        result = pd.concat(per_ticker, names=["symbol", "date"])
        # Ratios depend on cross-ticker columns (market_avg_rv21, sector_avg_rv21)
        # being populated, so compute them after the concat on the unified frame.
        ratios = add_ratio_features(result)
        result = pd.concat([result, ratios], axis=1)
        result = result.replace([np.inf, -np.inf], np.nan)
        # Reorder columns to canonical order; dropna(how="all") trims the warm-up
        result = result.reindex(columns=FEATURE_COLUMNS).dropna(how="all")
        return result

    def _build_single_ticker(self, sym: str, b: pd.DataFrame, r: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=b.index, dtype=float)

        # Realized vol
        df["rv_5"] = rolling_rv(r, 5)
        df["rv_10"] = rolling_rv(r, 10)
        df["rv_21"] = rolling_rv(r, 21)
        df["rv_63"] = rolling_rv(r, 63)
        df["ewma_vol_94"] = ewma_vol(r, 0.94)
        df["ewma_vol_97"] = ewma_vol(r, 0.97)
        har = har_rv_components(r)
        df["har_rv_daily"] = har["har_rv_daily"]
        df["har_rv_weekly"] = har["har_rv_weekly"]
        df["har_rv_monthly"] = har["har_rv_monthly"]
        df["acf_sq_ret_lag1"] = acf_squared_returns(r, 1)
        df["acf_sq_ret_lag5"] = acf_squared_returns(r, 5)
        df["acf_sq_ret_lag10"] = acf_squared_returns(r, 10)

        # GARCH (walk-forward with periodic refit, daily recursion)
        garch = garch_features_walk_forward(
            r,
            refit_every=self._garch_refit_every,
            min_history=self._garch_min_history,
        )
        df["garch_forecast_var"] = garch["garch_forecast_var"]
        df["garch_resid_lb_pvalue"] = garch["garch_resid_lb_pvalue"]
        df["garch_persistence"] = garch["garch_persistence"]

        # Technical indicators
        df["bb_width"] = bollinger_width(b["close"])
        df["bb_width_roc"] = df["bb_width"].pct_change()
        df["macd_hist_mag"] = macd_histogram(b["close"])
        df["rsi_14"] = rsi(b["close"])
        df["volume_ratio"] = volume_ratio(b["volume"])
        df["atr_14"] = atr(b["high"], b["low"], b["close"])
        df["atr_roc"] = df["atr_14"].pct_change()
        df["intraday_range"] = intraday_range(b["high"], b["low"], b["close"])

        # OHLC-based vol (Parkinson, Garman-Klass)
        df["parkinson_5"] = parkinson_vol(b["high"], b["low"], 5)
        df["parkinson_10"] = parkinson_vol(b["high"], b["low"], 10)
        df["parkinson_21"] = parkinson_vol(b["high"], b["low"], 21)
        df["gk_5"] = garman_klass_vol(b["open"], b["high"], b["low"], b["close"], 5)
        df["gk_10"] = garman_klass_vol(b["open"], b["high"], b["low"], b["close"], 10)
        df["gk_21"] = garman_klass_vol(b["open"], b["high"], b["low"], b["close"], 21)

        # Realized distribution shape (h=21 selector only; computed for all)
        df["rskew_21"] = realized_skew(r, 21)
        df["rskew_63"] = realized_skew(r, 63)
        df["rkurt_21"] = realized_kurt(r, 21)
        df["rkurt_63"] = realized_kurt(r, 63)

        # h=1 target-side columns: single-day GK vol (with fallbacks), its log,
        # the 63-day baseline b_t, and demeaned HAR components. Computed here
        # (not only at target build) so signal-time inference reads the exact
        # same b_t / dev the model trained on.
        gk1 = daily_ohlc_vol(b)
        lv = log_vol(gk1)
        baseline = rolling_log_vol_baseline(lv)
        df["gk_1"] = gk1
        df["log_gk_1"] = lv
        df["log_gk_baseline_63"] = baseline
        df["dev_gk"] = lv - baseline
        df["har_dev_5"] = lv.rolling(5).mean() - baseline
        df["har_dev_22"] = lv.rolling(22).mean() - baseline

        # Signed returns / leverage effect
        df["ret_1"] = r
        df["ret_5"] = r.rolling(5).sum()
        df["ret_21"] = r.rolling(21).sum()
        df["ret_1_neg"] = r.clip(upper=0.0)

        # Overnight / intraday decomposition
        gap = np.log(b["open"]) - np.log(b["close"].shift(1))
        oc_ret = np.log(b["close"]) - np.log(b["open"])
        df["overnight_gap"] = gap
        df["overnight_vol_21"] = gap.rolling(21).std(ddof=0)
        df["overnight_to_intraday_21"] = (
            df["overnight_vol_21"] / oc_ret.rolling(21).std(ddof=0)
        )

        # Deterministic calendar dummies. month_end uses the next BUSINESS
        # day's month — calendar knowledge, not future bar data.
        dts = pd.DatetimeIndex(df.index)
        dow = dts.dayofweek
        df["dow_mon"] = (dow == 0).astype(float)
        df["dow_fri"] = (dow == 4).astype(float)
        df["opex_friday"] = ((dow == 4) & (dts.day >= 15) & (dts.day <= 21)).astype(float)
        next_bday = dts + pd.tseries.offsets.BDay(1)
        df["month_end"] = (next_bday.month != dts.month).astype(float)

        # Macro release dummies (scheduled dates — calendar knowledge)
        today_d = np.array([d.date() for d in dts])
        tomorrow_d = np.array([d.date() for d in next_bday])

        def _in(day_arr: np.ndarray, events: frozenset) -> np.ndarray:
            return np.array([d in events for d in day_arr], dtype=float)

        df["macro_any_tomorrow"] = _in(tomorrow_d, self._macro_all)
        df["macro_any_today"] = _in(today_d, self._macro_all)
        df["fomc_tomorrow"] = _in(tomorrow_d, self._macro_events.get("FOMC decision", frozenset()))
        df["cpi_tomorrow"] = _in(tomorrow_d, self._macro_events.get("CPI release", frozenset()))
        df["nfp_tomorrow"] = _in(tomorrow_d, self._macro_events.get("NFP release", frozenset()))

        # Earnings-distance features (NaN when no earnings data for symbol)
        impacts = self._earnings_impacts.get(sym)
        if impacts is not None and len(impacts) > 0:
            imp = impacts.values
            t = dts.values
            nxt = np.searchsorted(imp, t, side="right")  # first impact > t
            prv = nxt - 1
            one_day = np.timedelta64(1, "D")
            days_to = np.where(
                nxt < len(imp),
                (imp[np.minimum(nxt, len(imp) - 1)] - t) / one_day,
                np.nan,
            )
            days_since = np.where(
                prv >= 0,
                (t - imp[np.maximum(prv, 0)]) / one_day,
                np.nan,
            )
            df["earnings_tomorrow"] = np.isin(next_bday.values, imp).astype(float)
            df["days_to_earnings"] = np.minimum(days_to, 21.0)
            df["days_since_earnings"] = np.minimum(days_since, 63.0)
        else:
            df["earnings_tomorrow"] = float("nan")
            df["days_to_earnings"] = float("nan")
            df["days_since_earnings"] = float("nan")

        # Implied-vol features (NaN when no IV data for symbol; observations
        # are sparse pre-2025, so ffill onto bar dates with a 5-day limit)
        iv_pair = self._iv_by_symbol.get(sym)
        if iv_pair is not None:
            iv_al = iv_pair[0].reindex(dts, method="ffill", limit=5)
            hv_al = iv_pair[1].reindex(dts, method="ffill", limit=5)
            iv_al.index = df.index
            hv_al.index = df.index
            df["iv_level"] = iv_al
            df["iv_minus_hv"] = iv_al - hv_al
            df["iv_chg_5"] = iv_al.diff(5)
            df["iv_pctile_252"] = iv_al.rolling(252, min_periods=60).rank(pct=True)
        else:
            df["iv_level"] = float("nan")
            df["iv_minus_hv"] = float("nan")
            df["iv_chg_5"] = float("nan")
            df["iv_pctile_252"] = float("nan")

        return df

    def _add_cross_ticker_features(
        self,
        per_ticker: dict[str, pd.DataFrame],
        bars: dict[str, pd.DataFrame],
        returns: dict[str, pd.Series],
    ) -> None:
        rv21_panel = pd.DataFrame({sym: per_ticker[sym]["rv_21"] for sym in per_ticker})
        market_mean = market_avg_rv(rv21_panel)
        sector_mean = sector_avg_rv(rv21_panel, self._watchlist)

        # SPY correlation: build panel including SPY
        corr_panel = {sym: returns[sym] for sym in per_ticker if sym in returns}
        if "SPY" in returns:
            corr_panel["SPY"] = returns["SPY"]
            spy_corr = rolling_corr_with_spy(pd.DataFrame(corr_panel))
        else:
            logger.warning("SPY missing from returns; corr_spy_21 will be NaN")
            spy_corr = pd.DataFrame()

        # VIX features
        vix_close = bars["VIX"]["close"] if not bars.get("VIX", pd.DataFrame()).empty else None
        vix9d_close = bars["VIX9D"]["close"] if not bars.get("VIX9D", pd.DataFrame()).empty else None
        vix3m_close = bars["VIX3M"]["close"] if not bars.get("VIX3M", pd.DataFrame()).empty else None
        if vix_close is not None and vix9d_close is not None and vix3m_close is not None:
            vix_df = vix_features(vix_close, vix9d_close, vix3m_close)
        else:
            logger.warning("VIX series incomplete; VIX features will be NaN")
            vix_df = pd.DataFrame(columns=["vix_level", "vix9d_to_vix", "vix3m_to_vix"])

        for sym, df in per_ticker.items():
            df["market_avg_rv21"] = market_mean.reindex(df.index)
            if sym in sector_mean.columns:
                df["sector_avg_rv21"] = sector_mean[sym].reindex(df.index)
            else:
                df["sector_avg_rv21"] = float("nan")
            if not vix_df.empty:
                df["vix_level"] = vix_df["vix_level"].reindex(df.index)
                df["vix9d_to_vix"] = vix_df["vix9d_to_vix"].reindex(df.index)
                df["vix3m_to_vix"] = vix_df["vix3m_to_vix"].reindex(df.index)
            else:
                df["vix_level"] = float("nan")
                df["vix9d_to_vix"] = float("nan")
                df["vix3m_to_vix"] = float("nan")
            if sym in spy_corr.columns:
                df["corr_spy_21"] = spy_corr[sym].reindex(df.index)
            else:
                df["corr_spy_21"] = float("nan")
