"""Slow integration test: retrains both LightGBM (primary) and XGBoost
(fallback) across all three prediction horizons (5, 10, 21). Saves artifacts
to model/artifacts/ and writes per-horizon OOS R² to
model/artifacts/latest_retrain_r2.json so main.py's _load_predictors can
update BestPredictor routing without manual config edits. Guarded by
RUN_SLOW_TESTS=1.

Replaces tests/test_xgb_hyperparam_tuning.py (XGBoost-only). The cron entry
in deploy/README.md points here now.
"""
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config import load_settings, load_watchlist
from data import HistoricalStore, compute_log_returns
from features import FeaturePipeline
from model import (
    GARCHBaseline,
    regression_metrics,
    walk_forward_evaluate_lightgbm,
    walk_forward_evaluate_xgboost,
)


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "model" / "artifacts"
ROUTING_R2_JSON = ARTIFACT_DIR / "latest_retrain_r2.json"


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
) -> tuple[dict, dict, dict, float, float]:
    """Run LightGBM (with tuning) + XGBoost (with tuning) + GARCH walk-forward
    at a single horizon. Returns (garch_metrics, lgbm_metrics, xgb_metrics,
    lgbm_elapsed, xgb_elapsed)."""
    print(f"\n--- horizon={horizon} ---")

    t0 = time.monotonic()
    lgbm_results = walk_forward_evaluate_lightgbm(
        feature_df, returns_by_symbol, horizon=horizon,
        train_window_days=train_window_days, refit_every=refit_every,
        hyperparams=None,
        artifact_dir=ARTIFACT_DIR,
    )
    lgbm_elapsed = time.monotonic() - t0
    print(f"  lightgbm tuning + walk-forward: {lgbm_elapsed:.1f}s ({len(lgbm_results)} OOS rows)")
    lgbm_metrics = regression_metrics(lgbm_results["actual"], lgbm_results["predicted"])

    t0 = time.monotonic()
    xgb_results = walk_forward_evaluate_xgboost(
        feature_df, returns_by_symbol, horizon=horizon,
        train_window_days=train_window_days, refit_every=refit_every,
        hyperparams=None,
        artifact_dir=ARTIFACT_DIR,
    )
    xgb_elapsed = time.monotonic() - t0
    print(f"  xgboost  tuning + walk-forward: {xgb_elapsed:.1f}s ({len(xgb_results)} OOS rows)")
    xgb_metrics = regression_metrics(xgb_results["actual"], xgb_results["predicted"])

    oos_dates = set(lgbm_results.index.get_level_values("date").unique())
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
    return garch_metrics, lgbm_metrics, xgb_metrics, lgbm_elapsed, xgb_elapsed


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

        rows = []
        r2_by_horizon: dict[int, dict[str, float]] = {}
        total_lgbm_elapsed = 0.0
        total_xgb_elapsed = 0.0
        for h in horizons:
            garch_m, lgbm_m, xgb_m, l_elapsed, x_elapsed = evaluate_horizon(
                h, feature_df, returns_by_symbol,
                train_window_days, refit_every, symbols,
            )
            total_lgbm_elapsed += l_elapsed
            total_xgb_elapsed += x_elapsed
            rows.append({"horizon": h, "model": "GARCH", **garch_m})
            rows.append({"horizon": h, "model": "LightGBM (tuned)", **lgbm_m})
            rows.append({"horizon": h, "model": "XGBoost (tuned)", **xgb_m})
            r2_by_horizon[h] = {
                "lgbm": float(lgbm_m["r2"]),
                "xgb": float(xgb_m["r2"]),
                "garch": float(garch_m["r2"]),
            }

        # 30-min budget: LightGBM is ~3× faster than XGBoost so even with both
        # fits per horizon this should comfortably fit in 30 min.
        total_elapsed = total_lgbm_elapsed + total_xgb_elapsed
        assert total_elapsed < 1800, (
            f"FAIL: total tuning + walk-forward took {total_elapsed:.1f}s, must be <30 min"
        )

        summary = pd.DataFrame(rows).set_index(["horizon", "model"])[
            ["n", "rmse", "mae", "r2", "bias"]
        ]
        print("\n=== Head-to-head per horizon ===")
        print(summary.to_string())

        print("\n=== Winners (per horizon) ===")
        for h in horizons:
            r2s = {
                "GARCH": summary.loc[(h, "GARCH"), "r2"],
                "LightGBM": summary.loc[(h, "LightGBM (tuned)"), "r2"],
                "XGBoost": summary.loc[(h, "XGBoost (tuned)"), "r2"],
            }
            winner = max(r2s, key=r2s.get)
            print(
                f"  h={h:2d}: {winner:10s}  "
                f"(LGBM={r2s['LightGBM']:+.3f}, XGB={r2s['XGBoost']:+.3f}, "
                f"GARCH={r2s['GARCH']:+.3f})"
            )

        print(
            f"\ntotal lightgbm: {total_lgbm_elapsed:.1f}s, "
            f"total xgboost: {total_xgb_elapsed:.1f}s, "
            f"combined: {total_elapsed:.1f}s"
        )
        # Verify we wrote both artifact families
        for h in horizons:
            lgbm_files = list(ARTIFACT_DIR.glob(f"lgbm_h{h}_*.joblib"))
            xgb_files = list(ARTIFACT_DIR.glob(f"xgb_h{h}_*.joblib"))
            assert lgbm_files, f"no LightGBM artifact written for h={h}"
            assert xgb_files, f"no XGBoost artifact written for h={h}"
        print("\nartifacts: LightGBM + XGBoost saved for all horizons in", ARTIFACT_DIR)

        # Write the routing R² table that main.py's _load_predictors consumes.
        # Atomic write via temp file + rename so a partially-written JSON can
        # never confuse the bot during a concurrent restart.
        payload = {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_window_days": train_window_days,
            "refit_every": refit_every,
            "n_tickers": len(tickers),
            "r2_by_horizon": {str(h): r2_by_horizon[h] for h in horizons},
        }
        tmp_path = ROUTING_R2_JSON.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(ROUTING_R2_JSON)
        print(f"wrote routing R² metadata: {ROUTING_R2_JSON}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
