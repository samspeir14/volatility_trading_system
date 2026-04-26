import math
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import Ticker, load_settings
from data import HistoricalStore, compute_log_returns
from features import FeaturePipeline
from model import (
    DEFAULT_HYPERPARAMS,
    GARCHBaseline,
    build_training_matrix,
    regression_metrics,
    walk_forward_evaluate_xgboost,
)


def _last_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def main() -> int:
    settings = load_settings()
    if settings.env != "sandbox":
        print(f"refusing to run against env={settings.env!r}", file=sys.stderr)
        return 2

    end = _last_weekday(date.today())
    start = end - timedelta(days=730)
    horizon = 21
    train_window_days = 252  # 1 year (the cache only has ~2 years; smaller window leaves room for OOS)
    refit_every = 21

    tickers = [
        Ticker("AAPL", "tech"),
        Ticker("MSFT", "tech"),
        Ticker("SPY", "etf"),
    ]
    symbols = [t.symbol for t in tickers]

    store = HistoricalStore(settings.cache_db_path)
    try:
        # Build features (uses already-cached data; 0 API calls expected)
        pipeline = FeaturePipeline(
            store, tickers,
            garch_min_history=100, garch_refit_every=21,
        )
        t0 = time.monotonic()
        feature_df = pipeline.build_features(start, end)
        feat_elapsed = time.monotonic() - t0
        print(f"built features: {len(feature_df)} rows × {len(feature_df.columns)} cols in {feat_elapsed:.1f}s")

        # Per-symbol returns
        returns_by_symbol: dict[str, pd.Series] = {}
        for sym in symbols:
            bars = store.get_bars(sym, start, end)
            returns_by_symbol[sym] = compute_log_returns(bars["close"])

        # XGBoost walk-forward (default hyperparams; tuning is in the slow test)
        t0 = time.monotonic()
        xgb_results = walk_forward_evaluate_xgboost(
            feature_df, returns_by_symbol, horizon=horizon,
            train_window_days=train_window_days, refit_every=refit_every,
            hyperparams=DEFAULT_HYPERPARAMS,
        )
        xgb_elapsed = time.monotonic() - t0
        print(f"xgboost walk-forward: {len(xgb_results)} OOS rows in {xgb_elapsed:.1f}s")

        xgb_metrics = regression_metrics(xgb_results["actual"], xgb_results["predicted"])

        # GARCH baseline on the same OOS dates
        oos_dates = set(xgb_results.index.get_level_values("date").unique())
        baseline = GARCHBaseline(refit_every=refit_every, min_history=100)
        garch_pooled = []
        for sym in symbols:
            r = returns_by_symbol[sym]
            eval_df = baseline.walk_forward_evaluate(r, horizons=(horizon,))
            eval_df = eval_df.loc[eval_df.index.isin(oos_dates)]
            sub = eval_df[[f"pred_rv_{horizon}", f"actual_rv_{horizon}"]].copy()
            sub["symbol"] = sym
            garch_pooled.append(sub)
        garch_df = pd.concat(garch_pooled)
        garch_metrics = regression_metrics(
            garch_df[f"actual_rv_{horizon}"],
            garch_df[f"pred_rv_{horizon}"],
        )

        # Head-to-head table
        comparison = pd.DataFrame({
            "GARCH": garch_metrics,
            "XGBoost": xgb_metrics,
        }).T[["n", "rmse", "mae", "r2", "bias"]]
        print("\n=== Head-to-head (horizon=21, OOS) ===")
        print(comparison.to_string())

        # Sanity assertions
        assert math.isfinite(xgb_metrics["r2"]), f"XGBoost R² not finite: {xgb_metrics['r2']}"
        assert math.isfinite(xgb_metrics["rmse"]), f"XGBoost RMSE not finite: {xgb_metrics['rmse']}"
        # XGBoost shouldn't be drastically worse than GARCH
        assert xgb_metrics["rmse"] < garch_metrics["rmse"] * 10, (
            f"XGBoost RMSE {xgb_metrics['rmse']} >>> GARCH RMSE {garch_metrics['rmse']}"
        )
        assert len(xgb_results) > 50, f"too few OOS predictions: {len(xgb_results)}"

        winner = "XGBoost" if xgb_metrics["r2"] > garch_metrics["r2"] else "GARCH"
        print(f"\nwinner @ horizon=21: {winner} "
              f"(XGB R²={xgb_metrics['r2']:+.3f} vs GARCH R²={garch_metrics['r2']:+.3f})")

        # Performance gate
        total_elapsed = feat_elapsed + xgb_elapsed
        print(f"\ntotal wall-clock (features + walk-forward): {total_elapsed:.1f}s")
        assert total_elapsed < 300, f"FAIL: took {total_elapsed:.1f}s, must be <300s (5min)"
        print("PASS: under the 5-min gate")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
