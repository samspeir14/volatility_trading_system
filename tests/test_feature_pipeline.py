import asyncio
import sys
import time
from datetime import date, timedelta

import numpy as np

from config import Ticker, load_settings, load_watchlist
from data import AsyncTradierClient, HistoricalStore
from features import (
    BASELINE_FEATURE_COLUMNS,
    DISTRIBUTION_SHAPE_COLUMNS,
    FEATURE_COLUMNS,
    HORIZON_FEATURE_SETS,
    OHLC_VOL_COLUMNS,
    RATIO_FEATURE_COLUMNS,
    FeaturePipeline,
)


def _last_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


async def test_small_pipeline() -> None:
    settings = load_settings()
    end = _last_weekday(date.today())
    start = end - timedelta(days=730)  # 2 years

    tickers = [
        Ticker("AAPL", "tech"),
        Ticker("MSFT", "tech"),
        Ticker("SPY", "etf"),
    ]

    store = HistoricalStore(settings.cache_db_path)
    try:
        # Lower min_history so a 2y window has GARCH coverage for testing
        pipeline = FeaturePipeline(
            store, tickers,
            garch_min_history=100,
            garch_refit_every=21,
        )
        async with AsyncTradierClient(settings) as client:
            await pipeline.ensure_data(client, end, lookback_years=2)

        t0 = time.monotonic()
        df = pipeline.build_features(start, end)
        elapsed = time.monotonic() - t0

        # Shape
        assert df.index.names == ["symbol", "date"], f"index names = {df.index.names}"
        assert list(df.columns) == FEATURE_COLUMNS, "column order/identity mismatch"
        assert len(df.columns) == 45, f"expected 45 features, got {len(df.columns)}"
        # Confirm new feature blocks all present
        for col in OHLC_VOL_COLUMNS:
            assert col in df.columns, f"missing OHLC vol column {col}"
        for col in DISTRIBUTION_SHAPE_COLUMNS:
            assert col in df.columns, f"missing distribution shape column {col}"
        for col in RATIO_FEATURE_COLUMNS:
            assert col in df.columns, f"missing ratio column {col}"

        # Per-ticker non-NaN row count after warm-up
        for sym in ("AAPL", "MSFT", "SPY"):
            sym_df = df.loc[sym]
            non_nan_full = sym_df.dropna(how="any").shape[0]
            assert non_nan_full >= 100, (
                f"{sym}: only {non_nan_full} fully-populated rows (need ≥100)"
            )
            print(f"  {sym}: {sym_df.shape[0]} rows, {non_nan_full} fully-populated")

        # No infinities
        assert not np.isinf(df.to_numpy(dtype=float, na_value=0.0)).any(), "found infinities"

        # GARCH coverage ≥ 95% of post-warm-up rows
        for sym in ("AAPL", "MSFT", "SPY"):
            sym_df = df.loc[sym]
            post_warmup = sym_df.iloc[100:]
            garch_coverage = post_warmup["garch_forecast_var"].notna().mean()
            assert garch_coverage >= 0.95, (
                f"{sym}: GARCH coverage {garch_coverage:.2%} below 95%"
            )
            print(f"  {sym}: GARCH coverage = {garch_coverage:.1%}")

        # VIX columns identical across tickers on the same date (broadcast correctness)
        any_date = df.loc["AAPL"].dropna(subset=["vix_level"]).index[-1]
        v_aapl = df.loc[("AAPL", any_date), ["vix_level", "vix9d_to_vix", "vix3m_to_vix"]]
        v_msft = df.loc[("MSFT", any_date), ["vix_level", "vix9d_to_vix", "vix3m_to_vix"]]
        v_spy = df.loc[("SPY", any_date), ["vix_level", "vix9d_to_vix", "vix3m_to_vix"]]
        assert (v_aapl.values == v_msft.values).all() and (v_aapl.values == v_spy.values).all(), (
            "VIX broadcast mismatch"
        )
        print(f"  VIX broadcast OK at {any_date}: level={v_aapl['vix_level']:.2f}")

        # GARCH features change daily within a refit window (daily recursion firing)
        aapl_fc = df.loc["AAPL", "garch_forecast_var"].dropna()
        diffs = aapl_fc.diff().dropna()
        assert (diffs == 0).sum() <= 1, (
            f"GARCH forecast was constant on {(diffs == 0).sum()} pairs — recursion broken?"
        )
        print(f"  AAPL GARCH forecast changes on {(diffs != 0).sum()}/{len(diffs)} consecutive rows")

        print(f"\n3-ticker build_features: {elapsed*1000:.0f}ms (rows={len(df)})")
    finally:
        store.close()


async def test_full_watchlist_perf() -> None:
    settings = load_settings()
    end = _last_weekday(date.today())
    start = end - timedelta(days=730)

    watchlist = load_watchlist()

    store = HistoricalStore(settings.cache_db_path)
    try:
        pipeline = FeaturePipeline(
            store, watchlist,
            garch_min_history=100,
            garch_refit_every=21,
        )
        async with AsyncTradierClient(settings) as client:
            await pipeline.ensure_data(client, end, lookback_years=2)
            api_calls = client.rate_limiter.call_count
        print(f"ensure_data: {api_calls} API calls")

        t0 = time.monotonic()
        df = pipeline.build_features(start, end)
        elapsed = time.monotonic() - t0

        assert len(df) > 0, "empty feature matrix"
        unique_symbols = df.index.get_level_values("symbol").unique()
        assert len(unique_symbols) == 20, f"expected 20 symbols, got {len(unique_symbols)}"

        print(f"20-ticker build_features: {elapsed:.2f}s ({len(df)} rows × {len(df.columns)} cols)")
        assert elapsed < 30.0, f"FAIL: build_features took {elapsed:.2f}s, must be <30s"
        print("PASS: build_features under the 30s gate")
    finally:
        store.close()


def test_horizon_feature_sets_structure() -> None:
    """Pure unit test — no API or cache needed."""
    assert set(HORIZON_FEATURE_SETS.keys()) == {5, 10, 21}, (
        f"unexpected horizons {set(HORIZON_FEATURE_SETS.keys())}"
    )
    for h, feats in HORIZON_FEATURE_SETS.items():
        assert len(feats) == 20, f"h={h}: expected 20 features, got {len(feats)}"
        assert len(set(feats)) == 20, f"h={h}: duplicate feature names"
        # Every selected feature must exist in the full FEATURE_COLUMNS set
        unknown = [f for f in feats if f not in FEATURE_COLUMNS]
        assert not unknown, f"h={h}: features not in FEATURE_COLUMNS: {unknown}"

    # h=5 and h=10 must NOT use distribution-shape features (per spec)
    for h in (5, 10):
        dist_overlap = set(HORIZON_FEATURE_SETS[h]) & set(DISTRIBUTION_SHAPE_COLUMNS)
        assert not dist_overlap, (
            f"h={h} unexpectedly includes distribution-shape features: {dist_overlap}"
        )
    # h=21 must include at least one rskew or rkurt
    h21_dist = set(HORIZON_FEATURE_SETS[21]) & set(DISTRIBUTION_SHAPE_COLUMNS)
    assert h21_dist, "h=21 must include at least one rskew/rkurt feature"
    print(f"horizon_feature_sets: 20-feature subsets per horizon, h=21 dist features = {sorted(h21_dist)}")


async def main() -> int:
    # Pure-unit tests run unconditionally
    test_horizon_feature_sets_structure()

    settings = load_settings()
    if settings.env != "sandbox":
        print(f"refusing to run against env={settings.env!r}", file=sys.stderr)
        return 2

    await test_small_pipeline()
    await test_full_watchlist_perf()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
