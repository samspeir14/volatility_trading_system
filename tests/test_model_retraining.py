"""Slow integration test doubling as the Sunday retrain job (RUN_SLOW_TESTS=1;
the cron entry in deploy/README.md points here, atomic JSON write + restart
chain preserved).

MODEL_PIPELINE=h1 (default): trains the pooled LightGBM on the next-day
log-GK-vol deviation target plus the HAR-RV benchmark, scores both against
GARCH/EWMA/persistence baselines on identical walk-forward OOS rows, applies
the acceptance gate (LightGBM must beat HAR on OOS QLIKE, else route=har),
persists OOS predictions for the nightly reconciliation, and writes the
schema-v2 routing JSON (previous file kept as .bak).

MODEL_PIPELINE=legacy: the old multi-horizon (5/10/21) LightGBM + XGBoost +
GARCH retrain, unchanged.
"""
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_settings, load_watchlist
from data import (
    AsyncTradierClient,
    HistoricalStore,
    compute_log_returns,
    find_stale_symbols,
)
from features import FeaturePipeline
from features.feature_pipeline import HORIZON_FEATURE_SETS
from features.garch import fit_garch11
from features.target import build_h1_deviation_target
from model import (
    GARCHBaseline,
    HARRVPredictor,
    LightGBMVolPredictor,
    ewma_deviation,
    garch_deviation,
    lagged_rv_forecast,
    per_ticker_r2_median,
    persistence_deviation,
    qlike,
    r2_vs_baseline,
    regression_metrics,
    tune_h1_hyperparams,
    walk_forward_evaluate_h1,
    walk_forward_evaluate_lightgbm,
    walk_forward_evaluate_xgboost,
    within_ticker_r2,
)
from model.term_structure import PHI_MAX, PHI_MIN
from model.training import build_h1_training_matrix


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "model" / "artifacts"
ROUTING_R2_JSON = ARTIFACT_DIR / "latest_retrain_r2.json"
H1_PREDICTIONS_FILE = "h1_oos_predictions.parquet"

H1_TRAIN_WINDOW_DAYS = 504
H1_REFIT_EVERY = 21


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


def _write_routing_json(payload: dict) -> None:
    """Backup the previous routing JSON to .bak, then atomic-write the new
    payload (temp file + rename) so a partially-written JSON can never
    confuse the bot during a concurrent restart."""
    if ROUTING_R2_JSON.exists():
        shutil.copy2(ROUTING_R2_JSON, ROUTING_R2_JSON.with_suffix(".json.bak"))
    tmp_path = ROUTING_R2_JSON.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(ROUTING_R2_JSON)
    print(f"wrote routing metadata: {ROUTING_R2_JSON}")


def run_h1_retrain(settings) -> int:
    """h=1 within-stock deviation retrain: pooled LightGBM (top-20 features)
    + HAR benchmark + GARCH/EWMA/persistence baselines, acceptance-gated
    routing, OOS predictions persisted for nightly reconciliation."""
    from model.evaluation import h1_metrics_from_predictions

    end = _last_weekday(date.today())
    start = end - timedelta(days=4 * 365)

    tickers = load_watchlist()
    print(f"h1 retrain against {len(tickers)} tickers, window {start} → {end}")

    store = HistoricalStore(settings.cache_db_path)
    try:
        pipeline = FeaturePipeline(
            store, tickers,
            garch_min_history=100, garch_refit_every=21,
        )

        # Refresh daily bars first (fail-soft; see legacy comment below).
        async def _refresh_bars() -> None:
            async with AsyncTradierClient(settings) as client:
                await pipeline.ensure_data(
                    client, end=_last_weekday(date.today() - timedelta(days=1)),
                    lookback_years=4,
                )

        try:
            asyncio.run(_refresh_bars())
        except Exception as e:
            print(f"WARNING: daily bar refresh failed ({e}); "
                  f"training on cached bars", file=sys.stderr)

        all_symbols = [t.symbol for t in tickers]
        excluded_symbols = sorted(find_stale_symbols(store, all_symbols))
        if excluded_symbols:
            print(f"excluding stale symbols from training: {excluded_symbols}")
        tickers = [t for t in tickers if t.symbol not in set(excluded_symbols)]
        symbols = [t.symbol for t in tickers]

        bars_through = max(
            (d for d in (store.latest_date(s) for s in symbols) if d is not None),
            default=None,
        )
        print(f"daily bars through: {bars_through}")

        pipeline = FeaturePipeline(
            store, tickers,
            garch_min_history=100, garch_refit_every=21,
        )
        t0 = time.monotonic()
        feature_df = pipeline.build_features(start, end)
        print(f"built features: {feature_df.shape} in {time.monotonic() - t0:.1f}s")

        bars_by_symbol = {s: store.get_bars(s, start, end) for s in symbols}
        bars_by_symbol = {s: b for s, b in bars_by_symbol.items() if not b.empty}
        returns_by_symbol = {
            s: compute_log_returns(b["close"]) for s, b in bars_by_symbol.items()
        }

        top_features = HORIZON_FEATURE_SETS[1]

        # --- tune + walk-forward ---
        t_budget = time.monotonic()
        X, y, _b = build_h1_training_matrix(feature_df, bars_by_symbol, top_features)
        t0 = time.monotonic()
        lgbm_params = tune_h1_hyperparams(
            X, y, train_window_days=H1_TRAIN_WINDOW_DAYS,
        )
        print(f"lgbm tuned in {time.monotonic() - t0:.1f}s: {lgbm_params}")

        t0 = time.monotonic()
        wf_lgbm = walk_forward_evaluate_h1(
            feature_df, bars_by_symbol,
            model_factory=lambda: LightGBMVolPredictor(
                horizon=1, hyperparams=lgbm_params,
            ),
            feature_subset=top_features,
            train_window_days=H1_TRAIN_WINDOW_DAYS, refit_every=H1_REFIT_EVERY,
            artifact_dir=ARTIFACT_DIR, artifact_prefix="lgbm_h1",
        )
        print(f"lgbm walk-forward: {len(wf_lgbm)} OOS rows ({time.monotonic() - t0:.1f}s)")

        t0 = time.monotonic()
        wf_har = walk_forward_evaluate_h1(
            feature_df, bars_by_symbol,
            model_factory=HARRVPredictor,
            train_window_days=H1_TRAIN_WINDOW_DAYS, refit_every=H1_REFIT_EVERY,
            artifact_dir=ARTIFACT_DIR, artifact_prefix="har_h1",
        )
        print(f"har walk-forward ({time.monotonic() - t0:.1f}s)")

        _y_all, b_all, lv_all = build_h1_deviation_target(bars_by_symbol)
        dev_forecasts = {
            "persistence": persistence_deviation(lv_all, b_all),
            "ewma": ewma_deviation(returns_by_symbol, b_all, lam=0.94),
        }
        t0 = time.monotonic()
        dev_forecasts["garch"] = garch_deviation(returns_by_symbol, b_all)
        print(f"garch baseline ({time.monotonic() - t0:.1f}s)")

        # --- identical OOS rows for every model ---
        common = wf_lgbm.index.intersection(wf_har.index)
        for dev in dev_forecasts.values():
            common = common.intersection(dev.dropna().index)
        print(f"common OOS rows: {len(common)}")
        assert len(common) > 0, "no common OOS rows across models"

        ref = wf_lgbm.loc[common]
        long_parts: list[pd.DataFrame] = []
        model_preds: dict[str, pd.Series] = {
            "lgbm": ref["predicted_dev"],
            "har": wf_har.loc[common, "predicted_dev"],
            **{name: dev.loc[common] for name, dev in dev_forecasts.items()},
        }
        for name, pred in model_preds.items():
            part = pd.DataFrame({
                "symbol": common.get_level_values("symbol"),
                "date": pd.to_datetime(common.get_level_values("date")),
                "model": name,
                "predicted_dev": pred.to_numpy(dtype=float),
                "actual_dev": ref["actual_dev"].to_numpy(),
                "baseline_b": ref["baseline_b"].to_numpy(),
                "actual_lv": ref["actual_lv"].to_numpy(),
            })
            long_parts.append(part)
        preds_long = pd.concat(long_parts, ignore_index=True)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        preds_path = ARTIFACT_DIR / H1_PREDICTIONS_FILE
        preds_long.to_parquet(preds_path, index=False)
        print(f"wrote OOS predictions: {preds_path} ({len(preds_long)} rows)")

        # --- metrics (from the persisted predictions, so the nightly
        # reconciliation recomputes the exact same numbers) ---
        metrics = h1_metrics_from_predictions(preds_long)
        print("\n=== h=1 model comparison (identical OOS rows) ===")
        for name, m in metrics.items():
            print(
                f"  {name:<12s} dev_R²={m['dev_r2_pooled']:+.4f} "
                f"(within {m['dev_r2_within']:+.4f}, "
                f"ticker-median {m['dev_r2_ticker_median']:+.4f}) "
                f"QLIKE={m['qlike_level']:.4f}"
            )

        # --- acceptance gate ---
        lgbm_q = metrics["lgbm"]["qlike_level"]
        har_q = metrics["har"]["qlike_level"]
        route = "lgbm" if lgbm_q < har_q else "har"
        passed = route == "lgbm"
        print(f"\nacceptance gate (lgbm QLIKE < har QLIKE): "
              f"{lgbm_q:.4f} vs {har_q:.4f} → route={route}")

        # --- per-ticker GARCH persistence (diagnostics + term-projection
        # fallback; runtime primarily uses the garch_persistence feature) ---
        persistence_by_symbol: dict[str, float] = {}
        for s in symbols:
            try:
                fit = fit_garch11(returns_by_symbol[s])
                persistence_by_symbol[s] = float(
                    np.clip(fit.alpha + fit.beta, PHI_MIN, PHI_MAX)
                )
            except Exception as e:
                print(f"WARNING: GARCH persistence fit failed for {s}: {e}",
                      file=sys.stderr)

        elapsed = time.monotonic() - t_budget
        assert elapsed < 1800, (
            f"FAIL: h1 tuning + walk-forward took {elapsed:.1f}s, must be <30 min"
        )

        # Promote the FINAL refit's artifacts by exact name. Loading newest-
        # by-mtime at runtime would be fragile: a retrain that dies after the
        # walk-forward (or trips the 30-min budget) leaves ungated fresh
        # joblibs on disk that the next unrelated restart would pick up under
        # last week's route. The JSON pins what the gate actually evaluated.
        lgbm_artifacts = list(ARTIFACT_DIR.glob("lgbm_h1_*.joblib"))
        har_artifacts = list(ARTIFACT_DIR.glob("har_h1_*.joblib"))
        assert lgbm_artifacts, "no LightGBM h1 artifact"
        assert har_artifacts, "no HAR h1 artifact"
        promoted = {
            "lgbm": max(lgbm_artifacts, key=lambda p: p.stat().st_mtime).name,
            "har": max(har_artifacts, key=lambda p: p.stat().st_mtime).name,
        }

        payload = {
            "schema_version": 2,
            "pipeline": "h1",
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bars_through": bars_through.isoformat() if bars_through else None,
            "train_window_days": H1_TRAIN_WINDOW_DAYS,
            "refit_every": H1_REFIT_EVERY,
            "n_tickers": len(tickers),
            "excluded_symbols": excluded_symbols,
            "target": {
                "proxy": "garman_klass",
                "log_eps": 1e-8,
                "baseline_window": 63,
                "baseline_min_obs": 40,
            },
            "h1": {
                "deviation_r2_pooled": {
                    n: float(m["dev_r2_pooled"]) for n, m in metrics.items()
                },
                "deviation_r2_within": {
                    n: float(m["dev_r2_within"]) for n, m in metrics.items()
                },
                "deviation_r2_ticker_median": {
                    n: float(m["dev_r2_ticker_median"]) for n, m in metrics.items()
                },
                "qlike_level": {
                    n: float(m["qlike_level"]) for n, m in metrics.items()
                },
                "route": route,
                "acceptance": {"rule": "qlike lgbm < har", "passed": passed},
                "artifacts": promoted,
                "top_features": top_features,
                "lgbm_hyperparams": {k: (v.item() if hasattr(v, "item") else v)
                                     for k, v in lgbm_params.items()},
                "garch_persistence_by_symbol": persistence_by_symbol,
                "predictions_file": H1_PREDICTIONS_FILE,
            },
            "r2_by_horizon": None,
        }
        _write_routing_json(payload)
        print(f"\nh1 retrain total: {elapsed:.1f}s")
    finally:
        store.close()
    return 0


def main() -> int:
    if os.environ.get("RUN_SLOW_TESTS") != "1":
        print("skipping: set RUN_SLOW_TESTS=1 to enable", file=sys.stderr)
        return 0

    settings = load_settings()
    if settings.env != "sandbox":
        print(f"refusing to run against env={settings.env!r}", file=sys.stderr)
        return 2

    if settings.model_pipeline == "h1":
        return run_h1_retrain(settings)
    return run_legacy_retrain(settings)


def run_legacy_retrain(settings) -> int:
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
            "schema_version": 1,
            "pipeline": "legacy",
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
        _write_routing_json(payload)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
