"""Slow integration test doubling as the Sunday retrain job (RUN_SLOW_TESTS=1;
the cron entry in deploy/README.md points here, atomic JSON write + restart
chain preserved).

Trains the pooled LightGBM on the next-day log-GK-vol deviation target plus
the HAR-RV benchmark, scores both against GARCH/EWMA/persistence baselines on
identical walk-forward OOS rows, applies the acceptance gate (route = argmax
OOS within-ticker deviation R² over lgbm / har / their 50/50 blend — the
singular strategy metric), persists OOS predictions for the nightly
reconciliation, and writes the schema-v2 routing JSON (previous file kept
as .bak).
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
    HARRVPredictor,
    LightGBMVolPredictor,
    ewma_deviation,
    garch_deviation,
    persistence_deviation,
    tune_h1_hyperparams,
    walk_forward_evaluate_h1,
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

        from features.feature_pipeline import load_earnings_history, load_iv_history
        cache_dir = Path(settings.cache_db_path).parent
        pipeline = FeaturePipeline(
            store, tickers,
            garch_min_history=100, garch_refit_every=21,
            iv_history=load_iv_history(cache_dir / "iv_history.csv"),
            earnings_history=load_earnings_history(cache_dir / "earnings_history.csv"),
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
        # Unweighted per the 2026-08-14 post-backfill lab verdict (+0.2008 vs
        # +0.1985 IVW). The lab re-tests IVW every run; flip this flag only
        # from a lab IVW VERDICT printout.
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
            "blend": (
                ref["predicted_dev"] + wf_har.loc[common, "predicted_dev"]
            ) / 2.0,
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
                f"QLIKE={m['qlike_level']:.4f} "
                f"sign={m['sign_hit_rate']:.3f} ρ={m['dev_spearman_median']:+.3f}"
            )

        # --- acceptance gate: within-ticker deviation R², the singular
        # strategy metric (everything else stays report-only). The 50/50
        # lgbm+har blend is a servable route, so it competes too. ---
        candidates = ("lgbm", "har", "blend")
        within = {n: metrics[n]["dev_r2_within"] for n in candidates}
        route = max(candidates, key=lambda n: within[n])
        passed = route != "har"
        print("\nacceptance gate (argmax within-ticker R² of "
              + ", ".join(f"{n}={within[n]:+.4f}" for n in candidates)
              + f") → route={route}")

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
                "sign_hit_rate": {
                    n: float(m["sign_hit_rate"]) for n, m in metrics.items()
                },
                "dev_spearman_median": {
                    n: float(m["dev_spearman_median"]) for n, m in metrics.items()
                },
                "route": route,
                "acceptance": {
                    "rule": "argmax within_r2 {lgbm,har,blend}",
                    "passed": passed,
                },
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

    return run_h1_retrain(settings)


if __name__ == "__main__":
    sys.exit(main())
