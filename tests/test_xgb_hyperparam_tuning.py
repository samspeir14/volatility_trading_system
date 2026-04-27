"""Slow integration test: full hyperparameter tuning + walk-forward + GARCH comparison
across all three prediction horizons (5, 10, 21). Guarded by RUN_SLOW_TESTS=1.
Runs against the full 20-ticker watchlist so XGBoost trains on ~10k rows."""

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from config import load_settings, load_watchlist
from data import HistoricalStore, compute_log_returns
from features import FeaturePipeline
from model import (
    GARCHBaseline,
    regression_metrics,
    walk_forward_evaluate_xgboost,
)


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "model" / "artifacts"


def _last_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def evaluate_horizon(
    horizon: int,
    feature_df: pd.DataFrame,
    returns_by_symbol: dict,
    train_window_days: int,
    refit_every: int,
    symbols: list[str],
) -> tuple[dict, dict, float]:
    """Run XGBoost (with tuning) + GARCH walk-forward at a single horizon.
    Returns (garch_metrics, xgb_metrics, wall_clock_seconds)."""
    print(f"\n--- horizon={horizon} ---")
    t0 = time.monotonic()
    xgb_results = walk_forward_evaluate_xgboost(
        feature_df, returns_by_symbol, horizon=horizon,
        train_window_days=train_window_days, refit_every=refit_every,
        hyperparams=None,  # triggers tuning at first refit
        artifact_dir=ARTIFACT_DIR,  # save tuned models for the live signal test
    )
    xgb_elapsed = time.monotonic() - t0
    print(f"  xgboost tuning + walk-forward: {xgb_elapsed:.1f}s ({len(xgb_results)} OOS rows)")

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
    return garch_metrics, xgb_metrics, xgb_elapsed


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
    horizons = (5, 10, 21)
    train_window_days = 252
    refit_every = 21

    tickers = load_watchlist()
    symbols = [t.symbol for t in tickers]
    print(f"running against {len(tickers)} tickers, horizons={horizons}")

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

        # Per-horizon evaluation
        rows = []
        total_xgb_elapsed = 0.0
        for h in horizons:
            garch_m, xgb_m, elapsed = evaluate_horizon(
                h, feature_df, returns_by_symbol,
                train_window_days, refit_every, symbols,
            )
            total_xgb_elapsed += elapsed
            rows.append({"horizon": h, "model": "GARCH", **garch_m})
            rows.append({"horizon": h, "model": "XGBoost (tuned)", **xgb_m})

        # 25-min gate: 3 horizons × ~3-4 min tuning each
        assert total_xgb_elapsed < 1500, (
            f"FAIL: total xgb tuning took {total_xgb_elapsed:.1f}s, must be <25 min"
        )

        # Final summary
        summary = pd.DataFrame(rows).set_index(["horizon", "model"])[
            ["n", "rmse", "mae", "r2", "bias"]
        ]
        print("\n=== Head-to-head per horizon ===")
        print(summary.to_string())

        # Winner per horizon
        print("\n=== Winners ===")
        wins = {}
        for h in horizons:
            g_r2 = summary.loc[(h, "GARCH"), "r2"]
            x_r2 = summary.loc[(h, "XGBoost (tuned)"), "r2"]
            winner = "XGBoost" if x_r2 > g_r2 else "GARCH"
            lift = x_r2 - g_r2
            wins[h] = (winner, x_r2, g_r2, lift)
            print(f"  h={h:2d}: {winner:7s}  (XGB R²={x_r2:+.3f} vs GARCH R²={g_r2:+.3f}, "
                  f"lift={lift:+.3f})")

        print(f"\ntotal xgboost tuning + walk-forward across {len(horizons)} horizons: "
              f"{total_xgb_elapsed:.1f}s")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
