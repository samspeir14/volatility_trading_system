import logging
from datetime import date

import pandas as pd

from config import MARKET_INDICES, Ticker
from data import (
    AsyncTradierClient,
    HistoricalStore,
    compute_log_returns,
    fetch_and_cache,
)
from features.cross_ticker import (
    market_avg_rv,
    rolling_corr_with_spy,
    sector_avg_rv,
    vix_features,
)
from features.garch import garch_features_walk_forward
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

FEATURE_COLUMNS = [
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


class FeaturePipeline:
    def __init__(
        self,
        store: HistoricalStore,
        watchlist: list[Ticker],
        market_indices: tuple[str, ...] = MARKET_INDICES,
        garch_min_history: int = 252,
        garch_refit_every: int = 21,
    ):
        self._store = store
        self._watchlist = watchlist
        self._indices = market_indices
        self._garch_min_history = garch_min_history
        self._garch_refit_every = garch_refit_every

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
        """Build the full feature matrix from cached bars. No I/O."""
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
            per_ticker[sym] = self._build_single_ticker(bars[sym], returns[sym])

        # Cross-ticker features
        if per_ticker:
            self._add_cross_ticker_features(per_ticker, bars, returns)

        if not per_ticker:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        result = pd.concat(per_ticker, names=["symbol", "date"])
        # Reorder columns to canonical order; dropna(how="all") trims the warm-up
        result = result.reindex(columns=FEATURE_COLUMNS).dropna(how="all")
        return result

    def _build_single_ticker(self, b: pd.DataFrame, r: pd.Series) -> pd.DataFrame:
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

        # Technical indicators
        df["bb_width"] = bollinger_width(b["close"])
        df["bb_width_roc"] = df["bb_width"].pct_change()
        df["macd_hist_mag"] = macd_histogram(b["close"])
        df["rsi_14"] = rsi(b["close"])
        df["volume_ratio"] = volume_ratio(b["volume"])
        df["atr_14"] = atr(b["high"], b["low"], b["close"])
        df["atr_roc"] = df["atr_14"].pct_change()
        df["intraday_range"] = intraday_range(b["high"], b["low"], b["close"])

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
