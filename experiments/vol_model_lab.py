"""Model experimentation lab.

Runs 6 experiments × 3 horizons (5/10/21) comparing the live XGBoost baseline
against augmented-feature variants, alternative gradient-boosted models, and
a Ridge stacking ensemble. All evaluation uses walk-forward with the same
target construction as model.training, so results are directly comparable to
the live baseline.

This script does NOT modify any code outside experiments/. Run as:

    python -m experiments.vol_model_lab

Requires: lightgbm, catboost installed in the venv (not in requirements.txt).

Tuning: per (model, horizon), 50 trials × 3 CV splits. XGB tuned twice per
horizon (baseline and augmented feature sets) so exp 1 stays honest. Exp 3
(top-20 augmented) reuses the augmented XGB params — top-20 is a subset of
augmented, and hyperparams that work on the full set generalize fine.
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
from model.training import (
    HYPERPARAM_SPACE,
    build_training_matrix,
    tune_hyperparameters,
)
from model.xgboost_model import XGBoostVolPredictor

from experiments.new_features import attach_new_features
from experiments.predictors import (
    CATBOOST_SPACE,
    LGBM_SPACE,
    CatBoostVolPredictor,
    LightGBMVolPredictor,
)
from experiments.walk_forward import (
    first_window_slice,
    pick_top_features_by_mean_rank,
    ridge_stack,
    tune_generic,
    walk_forward_generic,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vol_model_lab")

HORIZONS: tuple[int, ...] = (5, 10, 21)
TRAIN_WINDOW_DAYS = 504
REFIT_EVERY = 21
N_TRIALS = 50
N_TUNE_SPLITS = 3
TOP_N_FEATURES = 20

RESULTS_DIR = Path(__file__).parent / "results"


def xgb_factory(horizon: int, hyperparams: dict) -> XGBoostVolPredictor:
    return XGBoostVolPredictor(horizon=horizon, hyperparams=hyperparams)


def lgbm_factory(horizon: int, hyperparams: dict) -> LightGBMVolPredictor:
    return LightGBMVolPredictor(horizon=horizon, hyperparams=hyperparams)


def cat_factory(horizon: int, hyperparams: dict) -> CatBoostVolPredictor:
    return CatBoostVolPredictor(horizon=horizon, hyperparams=hyperparams)


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run() -> None:
    t_start = time.monotonic()
    _print_section("SETUP")

    settings = load_settings()
    watchlist = load_watchlist()
    end = date.today()
    start = end - timedelta(days=4 * 365)
    print(f"watchlist: {len(watchlist)} tickers")
    print(f"eval window: {start} → {end}")

    with HistoricalStore(settings.cache_db_path) as store:
        # Sanity check the cache
        spy_rows = store.row_count("SPY")
        print(f"cache SPY rows: {spy_rows}")
        if spy_rows < TRAIN_WINDOW_DAYS:
            raise RuntimeError(
                f"SPY cache has only {spy_rows} rows; need ≥{TRAIN_WINDOW_DAYS}"
            )

        # Build feature matrices ONCE
        t0 = time.monotonic()
        pipeline = FeaturePipeline(store, watchlist)
        baseline_df = pipeline.build_features(start, end)
        print(f"baseline features: {baseline_df.shape} (built in {time.monotonic() - t0:.1f}s)")

        t0 = time.monotonic()
        aug_df = attach_new_features(baseline_df, store, watchlist, start, end)
        print(f"augmented features: {aug_df.shape} (built in {time.monotonic() - t0:.1f}s)")

        # Sanity: no inf in augmented
        if np.isinf(aug_df.to_numpy()).any():
            raise RuntimeError("inf values found in augmented features — check ratio guards")

        # Returns dict ONCE
        returns_by_symbol: dict[str, pd.Series] = {}
        for ticker in watchlist:
            bars = store.get_bars(ticker.symbol, start, end)
            if bars.empty:
                continue
            returns_by_symbol[ticker.symbol] = compute_log_returns(bars["close"])
        print(f"returns built for {len(returns_by_symbol)} symbols")

        # ---------------- TUNING ----------------
        _print_section("TUNING (per model × horizon)")
        tuned: dict[tuple[str, str, int], dict] = {}

        for h in HORIZONS:
            t0 = time.monotonic()
            X_b, y_b = build_training_matrix(baseline_df, returns_by_symbol, h)
            X_a, y_a = build_training_matrix(aug_df, returns_by_symbol, h)
            X_b_first, y_b_first = first_window_slice(X_b, y_b, TRAIN_WINDOW_DAYS)
            X_a_first, y_a_first = first_window_slice(X_a, y_a, TRAIN_WINDOW_DAYS)

            t1 = time.monotonic()
            tuned[("xgb", "baseline", h)], score_xb = tune_hyperparameters(
                X_b_first, y_b_first, n_trials=N_TRIALS, n_splits=N_TUNE_SPLITS
            )
            print(f"[h={h}] xgb-baseline tuned in {time.monotonic() - t1:.1f}s (CV R²={score_xb:.4f})")

            t1 = time.monotonic()
            tuned[("xgb", "augmented", h)], score_xa = tune_hyperparameters(
                X_a_first, y_a_first, n_trials=N_TRIALS, n_splits=N_TUNE_SPLITS
            )
            print(f"[h={h}] xgb-augmented tuned in {time.monotonic() - t1:.1f}s (CV R²={score_xa:.4f})")

            t1 = time.monotonic()
            tuned[("lgbm", "augmented", h)], score_l = tune_generic(
                X_a_first, y_a_first, lgbm_factory, h, LGBM_SPACE,
                n_trials=N_TRIALS, n_splits=N_TUNE_SPLITS,
            )
            print(f"[h={h}] lgbm-augmented tuned in {time.monotonic() - t1:.1f}s (CV R²={score_l:.4f})")

            t1 = time.monotonic()
            tuned[("cat", "augmented", h)], score_c = tune_generic(
                X_a_first, y_a_first, cat_factory, h, CATBOOST_SPACE,
                n_trials=N_TRIALS, n_splits=N_TUNE_SPLITS,
            )
            print(f"[h={h}] catboost-augmented tuned in {time.monotonic() - t1:.1f}s (CV R²={score_c:.4f})")

            print(f"[h={h}] tuning total: {time.monotonic() - t0:.1f}s")

        # ---------------- WALK-FORWARD GRID ----------------
        _print_section("WALK-FORWARD GRID (6 experiments × 3 horizons)")
        results_rows: list[dict] = []
        importance_rows: list[dict] = []

        for h in HORIZONS:
            t_h = time.monotonic()
            print(f"\n--- horizon h={h} ---")

            acc_1: list[pd.Series] = []
            t0 = time.monotonic()
            preds_1 = walk_forward_generic(
                baseline_df, returns_by_symbol, h, xgb_factory,
                tuned[("xgb", "baseline", h)],
                train_window_days=TRAIN_WINDOW_DAYS, refit_every=REFIT_EVERY,
                importance_acc=acc_1,
            )
            m1 = regression_metrics(preds_1["actual"], preds_1["predicted"])
            print(f"[h={h}] exp1 baseline-xgb     R²={m1['r2']:.4f} RMSE={m1['rmse']:.5f} ({time.monotonic() - t0:.1f}s)")

            acc_2: list[pd.Series] = []
            t0 = time.monotonic()
            preds_2 = walk_forward_generic(
                aug_df, returns_by_symbol, h, xgb_factory,
                tuned[("xgb", "augmented", h)],
                train_window_days=TRAIN_WINDOW_DAYS, refit_every=REFIT_EVERY,
                importance_acc=acc_2,
            )
            m2 = regression_metrics(preds_2["actual"], preds_2["predicted"])
            print(f"[h={h}] exp2 aug-xgb          R²={m2['r2']:.4f} RMSE={m2['rmse']:.5f} ({time.monotonic() - t0:.1f}s)")

            top20 = pick_top_features_by_mean_rank(acc_2, top_n=TOP_N_FEATURES)
            acc_3: list[pd.Series] = []
            t0 = time.monotonic()
            preds_3 = walk_forward_generic(
                aug_df[top20], returns_by_symbol, h, xgb_factory,
                tuned[("xgb", "augmented", h)],
                train_window_days=TRAIN_WINDOW_DAYS, refit_every=REFIT_EVERY,
                importance_acc=acc_3,
            )
            m3 = regression_metrics(preds_3["actual"], preds_3["predicted"])
            print(f"[h={h}] exp3 top20-xgb        R²={m3['r2']:.4f} RMSE={m3['rmse']:.5f} ({time.monotonic() - t0:.1f}s)")

            acc_4: list[pd.Series] = []
            t0 = time.monotonic()
            preds_4 = walk_forward_generic(
                aug_df, returns_by_symbol, h, lgbm_factory,
                tuned[("lgbm", "augmented", h)],
                train_window_days=TRAIN_WINDOW_DAYS, refit_every=REFIT_EVERY,
                importance_acc=acc_4,
            )
            m4 = regression_metrics(preds_4["actual"], preds_4["predicted"])
            print(f"[h={h}] exp4 aug-lgbm         R²={m4['r2']:.4f} RMSE={m4['rmse']:.5f} ({time.monotonic() - t0:.1f}s)")

            acc_5: list[pd.Series] = []
            t0 = time.monotonic()
            preds_5 = walk_forward_generic(
                aug_df, returns_by_symbol, h, cat_factory,
                tuned[("cat", "augmented", h)],
                train_window_days=TRAIN_WINDOW_DAYS, refit_every=REFIT_EVERY,
                importance_acc=acc_5,
            )
            m5 = regression_metrics(preds_5["actual"], preds_5["predicted"])
            print(f"[h={h}] exp5 aug-catboost     R²={m5['r2']:.4f} RMSE={m5['rmse']:.5f} ({time.monotonic() - t0:.1f}s)")

            t0 = time.monotonic()
            preds_6 = ridge_stack(
                {"xgb": preds_2, "lgbm": preds_4, "catboost": preds_5},
                n_splits=5, alpha=1.0,
            )
            m6 = regression_metrics(preds_6["actual"], preds_6["predicted"])
            print(f"[h={h}] exp6 stack-ridge      R²={m6['r2']:.4f} RMSE={m6['rmse']:.5f} ({time.monotonic() - t0:.1f}s)")

            for name, metrics, acc in [
                ("1_baseline_xgb", m1, acc_1),
                ("2_aug_xgb", m2, acc_2),
                ("3_top20_xgb", m3, acc_3),
                ("4_aug_lgbm", m4, acc_4),
                ("5_aug_catboost", m5, acc_5),
                ("6_stack_ridge", m6, None),
            ]:
                results_rows.append({"experiment": name, "horizon": h, **metrics})
                if acc:
                    rank_frame = pd.concat(
                        [s.rank(ascending=False, method="average") for s in acc],
                        axis=1,
                    )
                    mean_rank = rank_frame.mean(axis=1)
                    mean_imp = pd.concat(acc, axis=1).mean(axis=1)
                    for feat in mean_imp.index:
                        importance_rows.append({
                            "experiment": name,
                            "horizon": h,
                            "feature": feat,
                            "mean_importance": float(mean_imp[feat]),
                            "mean_rank": float(mean_rank.get(feat, np.nan)),
                            "importance_type": "gain",
                        })

            print(f"[h={h}] horizon total: {time.monotonic() - t_h:.1f}s")

    # ---------------- WRITE OUTPUTS ----------------
    _print_section("OUTPUTS")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results_rows)
    results_path = RESULTS_DIR / "comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"wrote {results_path} ({len(results_df)} rows)")

    importance_df = pd.DataFrame(importance_rows)
    importance_path = RESULTS_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"wrote {importance_path} ({len(importance_df)} rows)")

    # ---------------- WINNER ----------------
    _print_section("RESULTS")
    print("\nComparison (R² by experiment × horizon):")
    pivot = results_df.pivot(index="experiment", columns="horizon", values="r2").round(4)
    print(pivot.to_string())

    print("\nWinners per horizon:")
    baseline_r2 = {
        int(row["horizon"]): row["r2"]
        for _, row in results_df[results_df["experiment"] == "1_baseline_xgb"].iterrows()
    }
    avg_r2_by_exp: dict[str, float] = {}
    for h in HORIZONS:
        h_rows = results_df[results_df["horizon"] == h]
        winner = h_rows.loc[h_rows["r2"].idxmax()]
        delta = winner["r2"] - baseline_r2.get(h, float("nan"))
        print(
            f"  [h={h}] {winner['experiment']:<20s} "
            f"R²={winner['r2']:.4f}  (vs baseline {baseline_r2.get(h, float('nan')):.4f}, Δ={delta:+.4f})"
        )

    for exp in results_df["experiment"].unique():
        rows = results_df[results_df["experiment"] == exp]
        avg_r2_by_exp[exp] = float(rows["r2"].mean())
    overall_winner = max(avg_r2_by_exp, key=avg_r2_by_exp.get)
    print(f"\n[overall] best avg R² across horizons: {overall_winner} (avg={avg_r2_by_exp[overall_winner]:.4f})")

    # Top features for the winning XGB-augmented at h=5 (most-watched horizon)
    if not importance_df.empty:
        focus = importance_df[
            (importance_df["experiment"] == "2_aug_xgb") & (importance_df["horizon"] == 5)
        ].sort_values("mean_rank", ascending=True).head(10)
        if not focus.empty:
            print("\nTop-10 features (exp 2_aug_xgb, h=5, by mean rank):")
            for i, (_, row) in enumerate(focus.iterrows(), 1):
                print(f"  {i:2d}. {row['feature']:<22s} mean_rank={row['mean_rank']:.2f}")

    print(f"\ntotal runtime: {(time.monotonic() - t_start) / 60.0:.1f} min")


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
    import argparse

    parser = argparse.ArgumentParser(description="Vol model lab")
    parser.add_argument(
        "--h1", action="store_true",
        help="run the h=1 within-stock deviation evaluation instead of the legacy 6×3 grid",
    )
    args = parser.parse_args()
    if args.h1:
        run_h1()
    else:
        run()
