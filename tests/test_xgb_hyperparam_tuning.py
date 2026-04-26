"""Slow integration test: full hyperparameter tuning + walk-forward + GARCH comparison.
Guarded by RUN_SLOW_TESTS=1. Runs against the full 20-ticker watchlist so XGBoost
trains on ~10k rows (vs ~1.5k in the fast test) — the more informative comparison."""

import math
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd

from config import load_settings, load_watchlist
from data import HistoricalStore, compute_log_returns
from features import FeaturePipeline
from model import (
    GARCHBaseline,
    regression_metrics,
    walk_forward_evaluate_xgboost,
)


def _last_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def main() -> int:
    if os.environ.get("RUN_SLOW_TESTS") != "1":
        print("skipping: set RUN_SLOW_TESTS=1 to enable", file=sys.stderr)
        return 0

    settings = load_settings()
    if settings.env != "sandbox":
        print(f"refusing to run against env={settings.env!r}", file=sys.stderr)
        return 2

    end = _last_weekday(date.today())
    start = end - timedelta(days=730)
    horizon = 21
    train_window_days = 252
    refit_every = 21

    tickers = load_watchlist()
    symbols = [t.symbol for t in tickers]
    print(f"running against {len(tickers)} tickers: {symbols}")

    store = HistoricalStore(settings.cache_db_path)
    try:
        pipeline = FeaturePipeline(
            store, tickers,
            garch_min_history=100, garch_refit_every=21,
        )
        t0 = time.monotonic()
        feature_df = pipeline.build_features(start, end)
        feat_elapsed = time.monotonic() - t0
        print(f"built features: {len(feature_df)} rows × {len(feature_df.columns)} cols in {feat_elapsed:.1f}s")

        returns_by_symbol = {
            sym: compute_log_returns(store.get_bars(sym, start, end)["close"])
            for sym in symbols
        }

        # hyperparams=None triggers tune_hyperparameters once at the first refit
        t0 = time.monotonic()
        xgb_results = walk_forward_evaluate_xgboost(
            feature_df, returns_by_symbol, horizon=horizon,
            train_window_days=train_window_days, refit_every=refit_every,
            hyperparams=None,
        )
        xgb_elapsed = time.monotonic() - t0

        print(f"\nfull tuning + walk-forward (20 tickers): {xgb_elapsed:.1f}s")
        # 20-min gate: tuning is 50 trials × 5 folds × ~2-3s/fit on 10k rows ≈ 8-12 min
        assert xgb_elapsed < 1200, f"FAIL: took {xgb_elapsed:.1f}s, must be <20 min"

        xgb_metrics = regression_metrics(xgb_results["actual"], xgb_results["predicted"])

        oos_dates = set(xgb_results.index.get_level_values("date").unique())
        baseline = GARCHBaseline(refit_every=refit_every, min_history=100)
        garch_pooled = []
        for sym in symbols:
            eval_df = baseline.walk_forward_evaluate(returns_by_symbol[sym], horizons=(horizon,))
            eval_df = eval_df.loc[eval_df.index.isin(oos_dates)]
            garch_pooled.append(eval_df[[f"pred_rv_{horizon}", f"actual_rv_{horizon}"]])
        garch_df = pd.concat(garch_pooled)
        garch_metrics = regression_metrics(
            garch_df[f"actual_rv_{horizon}"],
            garch_df[f"pred_rv_{horizon}"],
        )

        comparison = pd.DataFrame({
            "GARCH": garch_metrics,
            "XGBoost (tuned)": xgb_metrics,
        }).T[["n", "rmse", "mae", "r2", "bias"]]
        print("\n=== Head-to-head (tuned XGBoost vs GARCH, horizon=21) ===")
        print(comparison.to_string())

        winner = "XGBoost (tuned)" if xgb_metrics["r2"] > garch_metrics["r2"] else "GARCH"
        print(f"\nwinner: {winner} "
              f"(XGB R²={xgb_metrics['r2']:+.3f} vs GARCH R²={garch_metrics['r2']:+.3f})")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
