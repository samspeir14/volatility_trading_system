"""Model experimentation lab — h=1 within-stock deviation model.

Walk-forward evaluation of the production h=1 pipeline: pooled LightGBM
(full feature set and top-20) against HAR-RV, GARCH(1,1), EWMA(0.94), and
persistence, all scored on identical OOS rows with the same target
construction as model.training — results are directly comparable to the
Sunday retrain. Prints the acceptance-gate verdict and refreshes the frozen
top-20 list candidate (HORIZON_FEATURE_SETS[1]).

This script does NOT modify any code outside experiments/. Run as:

    python -m experiments.vol_model_lab
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_settings, load_watchlist
from data import HistoricalStore, compute_log_returns
from features.feature_pipeline import FeaturePipeline
from model.evaluation import regression_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vol_model_lab")

TRAIN_WINDOW_DAYS = 504
REFIT_EVERY = 21
N_TRIALS = 50
TOP_N_FEATURES = 20

RESULTS_DIR = Path(__file__).parent / "results"


def pick_top_features_by_mean_rank(
    importance_acc: list[pd.Series],
    top_n: int = 20,
) -> list[str]:
    """Aggregate per-refit importance Series into a feature list of size `top_n` by
    mean rank (lowest mean rank = most important across refits). Robust to scale
    drift across refits where a single huge-gain refit could dominate raw means."""
    if not importance_acc:
        return []
    rank_frame = pd.concat(
        [s.rank(ascending=False, method="average") for s in importance_acc],
        axis=1,
    )
    mean_rank = rank_frame.mean(axis=1).sort_values(ascending=True)
    return mean_rank.head(top_n).index.tolist()


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _drop_stale_symbols(store: HistoricalStore, watchlist: list) -> list:
    """Exclude frozen/stale series (e.g. IWM, stuck 2025-03-05) — they would
    poison pooled training."""
    from data import find_stale_symbols

    stale = set(find_stale_symbols(store, [t.symbol for t in watchlist]))
    if stale:
        print(f"STALE SYMBOLS DROPPED: {sorted(stale)}")
    return [t for t in watchlist if t.symbol not in stale]


def run_h1() -> None:
    """Walk-forward evaluation for the h=1 within-stock deviation model.

    Compares pooled LightGBM (full feature set and top-20) against HAR-RV,
    GARCH(1,1), EWMA(0.94), and persistence — all scored on identical OOS
    rows, walk-forward only, no shuffle. Prints the acceptance-gate verdict
    (LightGBM must beat HAR on out-of-sample QLIKE) and writes
    results/h1_comparison.csv + results/h1_feature_importance.csv.
    """
    from features.target import build_h1_deviation_target
    from model.evaluation import per_ticker_r2_median, qlike, within_ticker_r2
    from model.h1_baselines import (
        ewma_deviation,
        garch_deviation,
        persistence_deviation,
    )
    from model.har_model import HARRVPredictor
    from model.lightgbm_model import LightGBMVolPredictor as ProdLGBM
    from model.training import (
        build_h1_training_matrix,
        tune_h1_hyperparams,
        walk_forward_evaluate_h1,
    )

    t_start = time.monotonic()
    _print_section("H1 SETUP")

    settings = load_settings()
    watchlist = load_watchlist()
    end = date.today()
    start = end - timedelta(days=4 * 365)
    print(f"watchlist: {len(watchlist)} tickers | eval window: {start} → {end}")

    with HistoricalStore(settings.cache_db_path) as store:
        watchlist = _drop_stale_symbols(store, watchlist)
        print(f"tickers after staleness filter: {len(watchlist)}")

        pipeline = FeaturePipeline(store, watchlist)
        t0 = time.monotonic()
        feature_df = pipeline.build_features(start, end)
        print(f"features: {feature_df.shape} (built in {time.monotonic() - t0:.1f}s)")

        bars_by_symbol: dict[str, pd.DataFrame] = {}
        returns_by_symbol: dict[str, pd.Series] = {}
        for ticker in watchlist:
            bars = store.get_bars(ticker.symbol, start, end)
            if bars.empty:
                continue
            bars_by_symbol[ticker.symbol] = bars
            returns_by_symbol[ticker.symbol] = compute_log_returns(bars["close"])
        print(f"bars for {len(bars_by_symbol)} symbols")

    # ---------------- TUNING ----------------
    _print_section("H1 TUNING (LightGBM, first window only)")
    X_full, y_full, _b = build_h1_training_matrix(feature_df, bars_by_symbol)
    print(f"pooled training matrix: {X_full.shape}, target rows: {len(y_full)}")
    t0 = time.monotonic()
    lgbm_params = tune_h1_hyperparams(
        X_full, y_full, train_window_days=TRAIN_WINDOW_DAYS, n_trials=N_TRIALS
    )
    print(f"lgbm tuned in {time.monotonic() - t0:.1f}s: {lgbm_params}")

    # ---------------- WALK-FORWARD ----------------
    _print_section("H1 WALK-FORWARD")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    acc_full: list[pd.Series] = []
    t0 = time.monotonic()
    wf_lgbm_full = walk_forward_evaluate_h1(
        feature_df, bars_by_symbol,
        model_factory=lambda: ProdLGBM(horizon=1, hyperparams=lgbm_params),
        train_window_days=TRAIN_WINDOW_DAYS, refit_every=REFIT_EVERY,
        importance_acc=acc_full,
    )
    print(f"lgbm-full walk-forward: {len(wf_lgbm_full)} OOS rows ({time.monotonic() - t0:.1f}s)")

    top20 = pick_top_features_by_mean_rank(acc_full, top_n=TOP_N_FEATURES)
    print(f"\ntop-{TOP_N_FEATURES} features for h=1 (freeze into HORIZON_FEATURE_SETS[1]):")
    for i, f in enumerate(top20, 1):
        print(f"  {i:2d}. {f}")

    t0 = time.monotonic()
    wf_lgbm_top = walk_forward_evaluate_h1(
        feature_df, bars_by_symbol,
        model_factory=lambda: ProdLGBM(horizon=1, hyperparams=lgbm_params),
        feature_subset=top20,
        train_window_days=TRAIN_WINDOW_DAYS, refit_every=REFIT_EVERY,
    )
    print(f"lgbm-top20 walk-forward ({time.monotonic() - t0:.1f}s)")

    t0 = time.monotonic()
    wf_har = walk_forward_evaluate_h1(
        feature_df, bars_by_symbol,
        model_factory=HARRVPredictor,
        train_window_days=TRAIN_WINDOW_DAYS, refit_every=REFIT_EVERY,
    )
    print(f"har walk-forward ({time.monotonic() - t0:.1f}s)")

    # Parameter-light baselines: dev forecasts over the full history, then
    # restricted to the walk-forward OOS rows below.
    y_all, b_all, lv_all = build_h1_deviation_target(bars_by_symbol)
    dev_persist = persistence_deviation(lv_all, b_all)
    dev_ewma = ewma_deviation(returns_by_symbol, b_all, lam=0.94)
    t0 = time.monotonic()
    dev_garch = garch_deviation(returns_by_symbol, b_all)
    print(f"garch baseline ({time.monotonic() - t0:.1f}s)")

    # ---------------- METRICS (identical OOS rows) ----------------
    _print_section("H1 METRICS")
    frames = {
        "lgbm": wf_lgbm_top,
        "lgbm_full": wf_lgbm_full,
        "har": wf_har,
    }
    dev_forecasts = {
        "persistence": dev_persist,
        "ewma": dev_ewma,
        "garch": dev_garch,
    }
    common = wf_lgbm_top.index
    for f in frames.values():
        common = common.intersection(f.index)
    for d in dev_forecasts.values():
        common = common.intersection(d.dropna().index)
    print(f"common OOS rows across all models: {len(common)}")

    ref = wf_lgbm_top.loc[common]
    actual_dev = ref["actual_dev"]
    baseline_b = ref["baseline_b"]
    actual_var = np.exp(2.0 * ref["actual_lv"])

    rows: list[dict] = []
    for name in ("lgbm", "lgbm_full", "har", "persistence", "ewma", "garch"):
        if name in frames:
            pred_dev = frames[name].loc[common, "predicted_dev"]
        else:
            pred_dev = dev_forecasts[name].loc[common]
        forecast_var = np.exp(2.0 * (baseline_b + pred_dev))
        m = regression_metrics(actual_dev, pred_dev)
        row = {
            "model": name,
            "n": m["n"],
            "dev_r2_pooled": m["r2"],
            "dev_r2_within": within_ticker_r2(actual_dev, pred_dev),
            "dev_r2_ticker_median": per_ticker_r2_median(actual_dev, pred_dev),
            "qlike_level": qlike(actual_var, forecast_var),
            "rmse_dev": m["rmse"],
        }
        rows.append(row)
        print(
            f"  {name:<12s} dev_R²={row['dev_r2_pooled']:+.4f} "
            f"(within {row['dev_r2_within']:+.4f}, ticker-median {row['dev_r2_ticker_median']:+.4f}) "
            f"QLIKE={row['qlike_level']:.4f}"
        )

    results_df = pd.DataFrame(rows)
    results_path = RESULTS_DIR / "h1_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nwrote {results_path}")

    imp_rows = []
    if acc_full:
        rank_frame = pd.concat(
            [s.rank(ascending=False, method="average") for s in acc_full], axis=1
        )
        mean_rank = rank_frame.mean(axis=1)
        mean_imp = pd.concat(acc_full, axis=1).mean(axis=1)
        for feat in mean_imp.index:
            imp_rows.append({
                "feature": feat,
                "mean_importance": float(mean_imp[feat]),
                "mean_rank": float(mean_rank.get(feat, np.nan)),
            })
    imp_path = RESULTS_DIR / "h1_feature_importance.csv"
    pd.DataFrame(imp_rows).sort_values("mean_rank").to_csv(imp_path, index=False)
    print(f"wrote {imp_path}")

    # ---------------- ACCEPTANCE GATE ----------------
    _print_section("ACCEPTANCE GATE")
    by_model = {r["model"]: r for r in rows}
    lgbm_q = by_model["lgbm"]["qlike_level"]
    har_q = by_model["har"]["qlike_level"]
    passed = lgbm_q < har_q
    print(f"rule: lgbm QLIKE < har QLIKE  →  {lgbm_q:.4f} vs {har_q:.4f}")
    print(f"VERDICT: {'PASS — route h=1 to LightGBM' if passed else 'FAIL — route h=1 to HAR in production'}")
    full_q = by_model["lgbm_full"]["qlike_level"]
    if full_q < lgbm_q - 1e-6:
        print(f"note: full-feature LGBM QLIKE {full_q:.4f} beats top-20 {lgbm_q:.4f} — "
              f"consider revisiting the top-20 list")

    print(f"\ntotal runtime: {(time.monotonic() - t_start) / 60.0:.1f} min")


if __name__ == "__main__":
    run_h1()
