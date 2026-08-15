"""Model experimentation lab — h=1 within-stock deviation model.

Walk-forward evaluation of the production h=1 pipeline: pooled LightGBM
against HAR-RV, GARCH(1,1), EWMA(0.94), and persistence, all scored on
identical OOS rows with the same target construction as model.training —
results are directly comparable to the Sunday retrain.

Within-ticker deviation R² is the singular strategy metric, and every
optimization step here targets it: hyperparameters are tuned on it, the
production feature subset is the candidate top-N (nested by gain-importance
mean rank) with the best OOS within-ticker R², and the acceptance gate is
within-ticker R² (LightGBM must beat HAR). Prints the gate verdict and the
frozen-list candidate for HORIZON_FEATURE_SETS[1].

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
from features.target import PRODUCTION_TARGET_PROXY, target_vol_fn
from model.evaluation import regression_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vol_model_lab")

TRAIN_WINDOW_DAYS = 504
REFIT_EVERY = 21
# The retrain job stays at 50 trials (30-min budget); the lab has no budget
# and the 14-dim space deserves a denser search.
N_TRIALS = 100
# Candidate subset sizes for the frozen production list. Features are ordered
# by gain-importance mean rank (nesting), but the WINNER is the candidate with
# the best OOS within-ticker R² — the metric, not importance, decides.
CANDIDATE_TOP_NS = (10, 15, 20, 25, 30)

RESULTS_DIR = Path(__file__).parent / "results"

# The lab measures the DEPLOYED target — always in lockstep with production.
VOL_FN = target_vol_fn(PRODUCTION_TARGET_PROXY)


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

    Compares pooled LightGBM (full feature set and the best top-N subset by
    OOS within-ticker R²) against HAR-RV, GARCH(1,1), EWMA(0.94), and
    persistence — all scored on identical OOS rows, walk-forward only, no
    shuffle. Prints the acceptance-gate verdict (LightGBM must beat HAR on
    OOS within-ticker deviation R²) and writes results/h1_comparison.csv +
    results/h1_feature_importance.csv.
    """
    from features.target import build_h1_deviation_target
    from model.evaluation import (
        per_ticker_r2_median,
        per_ticker_spearman_median,
        qlike,
        sign_hit_rate,
        within_ticker_r2,
    )
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

        from features.feature_pipeline import load_earnings_history, load_iv_history
        cache_dir = Path(settings.cache_db_path).parent
        pipeline = FeaturePipeline(
            store, watchlist,
            iv_history=load_iv_history(cache_dir / "iv_history.csv"),
            earnings_history=load_earnings_history(cache_dir / "earnings_history.csv"),
            target_proxy=PRODUCTION_TARGET_PROXY,
        )
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
    X_full, y_full, _b = build_h1_training_matrix(
        feature_df, bars_by_symbol, vol_fn=VOL_FN,
    )
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
        vol_fn=VOL_FN,
    )
    print(f"lgbm-full walk-forward: {len(wf_lgbm_full)} OOS rows ({time.monotonic() - t0:.1f}s)")

    # Subset selection: nested top-N candidates ordered by importance mean
    # rank, winner = best OOS within-ticker R². All candidates run on the
    # same target rows, so their walk-forward outputs are directly comparable.
    full_within = within_ticker_r2(
        wf_lgbm_full["actual_dev"], wf_lgbm_full["predicted_dev"]
    )
    print(f"\nsubset selection (metric: OOS within-ticker R²):")
    print(f"  full ({len(feature_df.columns):2d} feats)  within_R²={full_within:+.4f}")
    best_subset: list[str] | None = None  # None = full feature set
    best_within = full_within
    wf_lgbm_top = wf_lgbm_full
    for top_n in CANDIDATE_TOP_NS:
        candidate = pick_top_features_by_mean_rank(acc_full, top_n=top_n)
        t0 = time.monotonic()
        wf_candidate = walk_forward_evaluate_h1(
            feature_df, bars_by_symbol,
            model_factory=lambda: ProdLGBM(horizon=1, hyperparams=lgbm_params),
            feature_subset=candidate,
            train_window_days=TRAIN_WINDOW_DAYS, refit_every=REFIT_EVERY,
            vol_fn=VOL_FN,
        )
        candidate_within = within_ticker_r2(
            wf_candidate["actual_dev"], wf_candidate["predicted_dev"]
        )
        print(f"  top-{top_n:2d}           within_R²={candidate_within:+.4f} "
              f"({time.monotonic() - t0:.1f}s)")
        if candidate_within > best_within:
            best_within = candidate_within
            best_subset = candidate
            wf_lgbm_top = wf_candidate

    if best_subset is None:
        print("\nWINNER: full feature set — freeze ALL feature columns into "
              "HORIZON_FEATURE_SETS[1]")
        best_subset = list(feature_df.columns)
    else:
        print(f"\nWINNER: top-{len(best_subset)} "
              f"(within_R²={best_within:+.4f}) — freeze into HORIZON_FEATURE_SETS[1]:")
    for i, f in enumerate(best_subset, 1):
        print(f"  {i:2d}. {f}")

    # Inverse-variance-weighted variant on the winner subset: rows weighted
    # 1/var(dev) per ticker, so the fit objective itself is within-ticker
    # aligned instead of dominated by high-vol-of-vol names.
    t0 = time.monotonic()
    wf_lgbm_ivw = walk_forward_evaluate_h1(
        feature_df, bars_by_symbol,
        model_factory=lambda: ProdLGBM(
            horizon=1, hyperparams=lgbm_params, inverse_variance_weights=True,
        ),
        feature_subset=best_subset,
        train_window_days=TRAIN_WINDOW_DAYS, refit_every=REFIT_EVERY,
        vol_fn=VOL_FN,
    )
    ivw_within = within_ticker_r2(
        wf_lgbm_ivw["actual_dev"], wf_lgbm_ivw["predicted_dev"]
    )
    print(f"\nIVW variant within_R²={ivw_within:+.4f} vs unweighted "
          f"{best_within:+.4f} ({time.monotonic() - t0:.1f}s)")
    print("IVW VERDICT: " + (
        "WEIGHTED wins — set inverse_variance_weights=True in the retrain job"
        if ivw_within > best_within
        else "unweighted wins — keep inverse_variance_weights=False"
    ))

    t0 = time.monotonic()
    wf_har = walk_forward_evaluate_h1(
        feature_df, bars_by_symbol,
        model_factory=HARRVPredictor,
        train_window_days=TRAIN_WINDOW_DAYS, refit_every=REFIT_EVERY,
        vol_fn=VOL_FN,
    )
    print(f"har walk-forward ({time.monotonic() - t0:.1f}s)")

    # Parameter-light baselines: dev forecasts over the full history, then
    # restricted to the walk-forward OOS rows below.
    y_all, b_all, lv_all = build_h1_deviation_target(bars_by_symbol, vol_fn=VOL_FN)
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
        "lgbm_ivw": wf_lgbm_ivw,
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

    # 50/50 lgbm+har blend — a servable production route, so it competes.
    dev_forecasts["blend"] = (
        ref["predicted_dev"] + frames["har"].loc[common, "predicted_dev"]
    ) / 2.0

    rows: list[dict] = []
    for name in ("lgbm", "lgbm_full", "lgbm_ivw", "blend", "har",
                 "persistence", "ewma", "garch"):
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
            "sign_hit": sign_hit_rate(actual_dev, pred_dev),
            "spearman_med": per_ticker_spearman_median(actual_dev, pred_dev),
        }
        rows.append(row)
        print(
            f"  {name:<12s} dev_R²={row['dev_r2_pooled']:+.4f} "
            f"(within {row['dev_r2_within']:+.4f}, ticker-median {row['dev_r2_ticker_median']:+.4f}) "
            f"QLIKE={row['qlike_level']:.4f} "
            f"sign={row['sign_hit']:.3f} ρ={row['spearman_med']:+.3f}"
        )

    # Margin + significance vs HAR: panel-safe Diebold-Mariano on QLIKE
    # losses and squared dev errors (same definition as the retrain report).
    from model.evaluation import h1_dm_tests
    dm_parts = []
    for name in ("lgbm", "har"):
        pred = (frames[name].loc[common, "predicted_dev"] if name in frames
                else dev_forecasts[name].loc[common])
        dm_parts.append(pd.DataFrame({
            "symbol": common.get_level_values("symbol"),
            "date": pd.to_datetime(common.get_level_values("date")),
            "model": name,
            "predicted_dev": pred.to_numpy(dtype=float),
            "actual_dev": actual_dev.to_numpy(),
            "baseline_b": baseline_b.to_numpy(),
            "actual_lv": ref["actual_lv"].to_numpy(),
        }))
    dm = h1_dm_tests(pd.concat(dm_parts, ignore_index=True), "lgbm", "har")
    by_model_row = {r["model"]: r for r in rows}
    qlike_margin = by_model_row["lgbm"]["qlike_level"] - by_model_row["har"]["qlike_level"]
    print(
        f"\nlgbm vs har: QLIKE margin {qlike_margin:+.4f} "
        f"(DM t={dm['qlike']['dm']:+.2f}, p={dm['qlike']['p']:.4f}) | "
        f"dev-MSE DM t={dm['dev_sq']['dm']:+.2f}, p={dm['dev_sq']['p']:.4f} "
        f"(negative t = lgbm better; {dm['qlike']['n_dates']} dates)"
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
    best_cfg = max(
        ("lgbm", "lgbm_ivw", "blend", "har"),
        key=lambda n: by_model[n]["dev_r2_within"],
    )
    best_w = by_model[best_cfg]["dev_r2_within"]
    har_w = by_model["har"]["dev_r2_within"]
    passed = best_cfg != "har"
    print("rule: argmax within-ticker R²  →  " + ", ".join(
        f"{n}={by_model[n]['dev_r2_within']:+.4f}"
        for n in ("lgbm", "lgbm_ivw", "blend", "har")
    ))
    print(f"VERDICT: {'PASS' if passed else 'FAIL'} — route h=1 to "
          f"{best_cfg} ({best_w:+.4f})")

    print(f"\ntotal runtime: {(time.monotonic() - t_start) / 60.0:.1f} min")


if __name__ == "__main__":
    run_h1()
