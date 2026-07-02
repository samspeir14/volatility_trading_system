"""Slow integration test: retrains both LightGBM (primary) and XGBoost
(fallback) across all three prediction horizons (5, 10, 21). Saves artifacts
to model/artifacts/ and writes per-horizon OOS R² to
model/artifacts/latest_retrain_r2.json so main.py's _load_predictors can
update BestPredictor routing without manual config edits. Guarded by
RUN_SLOW_TESTS=1.

Replaces tests/test_xgb_hyperparam_tuning.py (XGBoost-only). The cron entry
in deploy/README.md points here now.
"""
import asyncio
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config import load_settings, load_watchlist
from data import AsyncTradierClient, HistoricalStore, compute_log_returns
from features import FeaturePipeline
from model import (
    GARCHBaseline,
    lagged_rv_forecast,
    r2_vs_baseline,
    regression_metrics,
    walk_forward_evaluate_lightgbm,
    walk_forward_evaluate_xgboost,
    within_ticker_r2,
)


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "model" / "artifacts"
ROUTING_R2_JSON = ARTIFACT_DIR / "latest_retrain_r2.json"


def _last_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def compute_horizon_diagnostics(
    frames: dict[str, pd.DataFrame],
    returns_by_symbol: dict,
    horizon: int,
) -> dict:
    """Within-ticker R² and OOS R² vs a lagged-RV random walk for each model's
    (symbol, date)-indexed predicted/actual frame. The naive baseline is scored
    on the LGBM frame's index (all model frames share the same OOS rows)."""
    naive = lagged_rv_forecast(returns_by_symbol, horizon)
    ref = frames["lgbm"]
    return {
        "within_r2": {
            name: within_ticker_r2(df["actual"], df["predicted"])
            for name, df in frames.items()
        },
        "r2_vs_lagged_rv": {
            name: r2_vs_baseline(df["actual"], df["predicted"], naive.reindex(df.index))
            for name, df in frames.items()
        },
        "lagged_rv_pooled_r2": regression_metrics(
            ref["actual"], naive.reindex(ref.index)
        )["r2"],
        "lagged_rv_within_r2": within_ticker_r2(
            ref["actual"], naive.reindex(ref.index)
        ),
    }


def evaluate_horizon(
    horizon: int,
    feature_df: pd.DataFrame,
    returns_by_symbol: dict,
    train_window_days: int,
    refit_every: int,
    symbols: list[str],
) -> tuple[dict, dict, dict, dict, float, float]:
    """Run LightGBM (with tuning) + XGBoost (with tuning) + GARCH walk-forward
    at a single horizon. Returns (garch_metrics, lgbm_metrics, xgb_metrics,
    diagnostics, lgbm_elapsed, xgb_elapsed). `diagnostics` holds within-ticker
    R² (cross-sectional vol-level differences stripped) and OOS R² vs a
    lagged-RV random-walk forecast — the two views that pooled R² can't
    separate: level knowledge vs timing skill."""
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
        garch_pooled.append(
            eval_df[[f"pred_rv_{horizon}", f"actual_rv_{horizon}"]].rename(
                columns={f"pred_rv_{horizon}": "predicted", f"actual_rv_{horizon}": "actual"}
            )
        )
    garch_df = pd.concat(garch_pooled, keys=symbols)
    garch_df.index.names = ["symbol", "date"]
    garch_metrics = regression_metrics(garch_df["actual"], garch_df["predicted"])

    diagnostics = compute_horizon_diagnostics(
        {"lgbm": lgbm_results, "xgb": xgb_results, "garch": garch_df},
        returns_by_symbol, horizon,
    )
    return garch_metrics, lgbm_metrics, xgb_metrics, diagnostics, lgbm_elapsed, xgb_elapsed


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

        # Refresh daily bars before training — build_features is read-only, so
        # without this the retrain silently trains on whatever the cache last
        # saw (frozen at 2026-04-24 for two months before this step existed).
        # End is the last weekday strictly before today so a mid-session manual
        # run can never cache a partial bar. Fail-soft: a transient API outage
        # shouldn't kill the weekly cron, but the staleness must be loud —
        # bars_through goes to stdout and the routing JSON either way.
        async def _refresh_bars() -> None:
            async with AsyncTradierClient(settings) as client:
                await pipeline.ensure_data(
                    client, end=_last_weekday(date.today() - timedelta(days=1))
                )

        try:
            asyncio.run(_refresh_bars())
        except Exception as e:
            print(f"WARNING: daily bar refresh failed ({e}); "
                  f"training on cached bars", file=sys.stderr)
        bars_through = max(
            (d for d in (store.latest_date(s) for s in symbols) if d is not None),
            default=None,
        )
        print(f"daily bars through: {bars_through}")

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
        diagnostics_by_horizon: dict[int, dict] = {}
        total_lgbm_elapsed = 0.0
        total_xgb_elapsed = 0.0
        for h in horizons:
            garch_m, lgbm_m, xgb_m, diag, l_elapsed, x_elapsed = evaluate_horizon(
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
            diagnostics_by_horizon[h] = diag

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

        print("\n=== Diagnostics: within-ticker R² / R² vs lagged-RV random walk ===")
        print("(within strips cross-sectional vol-level differences; vs-naive is")
        print(" skill above 'next h days = last h days' — the bar IV already clears)")
        for h in horizons:
            d = diagnostics_by_horizon[h]
            w, v = d["within_r2"], d["r2_vs_lagged_rv"]
            print(
                f"  h={h:2d}: within  LGBM={w['lgbm']:+.3f} XGB={w['xgb']:+.3f} "
                f"GARCH={w['garch']:+.3f} | naive pooled={d['lagged_rv_pooled_r2']:+.3f} "
                f"within={d['lagged_rv_within_r2']:+.3f}"
            )
            print(
                f"        vs-naive LGBM={v['lgbm']:+.3f} XGB={v['xgb']:+.3f} "
                f"GARCH={v['garch']:+.3f}"
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
        # diagnostics_by_horizon is informational only — main._load_routing_r2
        # reads r2_by_horizon and ignores unknown keys, so routing is unaffected.
        payload = {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bars_through": bars_through.isoformat() if bars_through else None,
            "train_window_days": train_window_days,
            "refit_every": refit_every,
            "n_tickers": len(tickers),
            "r2_by_horizon": {str(h): r2_by_horizon[h] for h in horizons},
            "diagnostics_by_horizon": {
                str(h): {
                    "within_r2": {k: float(x) for k, x in diagnostics_by_horizon[h]["within_r2"].items()},
                    "r2_vs_lagged_rv": {k: float(x) for k, x in diagnostics_by_horizon[h]["r2_vs_lagged_rv"].items()},
                    "lagged_rv_pooled_r2": float(diagnostics_by_horizon[h]["lagged_rv_pooled_r2"]),
                    "lagged_rv_within_r2": float(diagnostics_by_horizon[h]["lagged_rv_within_r2"]),
                }
                for h in horizons
            },
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
