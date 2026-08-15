"""Replay analysis: does the IV-aware model thin out entry signals?

Compares the pre-backfill feature set ("old", no IV/earnings/macro columns)
against the frozen top-25 ("new") on identical walk-forward OOS rows:

1. DM tests vs HAR (QLIKE + dev-MSE) for both models — margin AND
   significance, same definition as the retrain report.
2. IV tracking: correlation of the term-projected annualized forecast with
   the DoltHub composite IV (levels and 5-day changes), at DTE 7 and 30.
3. Trigger thinning: divergence (forecast − iv) distribution, counts above
   fixed economic thresholds, and VRP-z-gate proxy counts (trailing 120-obs
   z of the divergence, SELL z>=+1.5 / BUY z<=-1.25 — the production gate
   thresholds; cost gate and chain-level filters cannot be replayed offline).

Uses the run-4 tuned hyperparams (fixed) so the only difference is inputs.

Run on the box: python -m experiments.replay_analysis
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_settings, load_watchlist
from data import HistoricalStore
from features.feature_pipeline import (
    FeaturePipeline,
    HORIZON_FEATURE_SETS,
    load_earnings_history,
    load_iv_history,
)
from model.evaluation import h1_dm_tests, qlike, within_ticker_r2
from model.har_model import HARRVPredictor
from model.lightgbm_model import LightGBMVolPredictor
from model.term_structure import PHI_MAX, PHI_MIN, project_term_vol
from model.training import walk_forward_evaluate_h1

TRAIN_WINDOW_DAYS = 504
REFIT_EVERY = 21

# Run-4 winner (lab 2026-08-14, post-backfill tuning).
HYPERPARAMS = {
    "objective": "huber", "max_depth": 3, "num_leaves": 15,
    "learning_rate": 0.05, "n_estimators": 100, "reg_alpha": 1.0,
    "reg_lambda": 0.5, "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_samples": 5, "min_split_gain": 0.01, "max_bin": 63,
    "extra_trees": True, "linear_tree": True,
}

# Pre-backfill feature set: the 63 columns that existed before the
# macro/earnings/IV phase (the previous frozen full-63 winner).
OLD_FEATURES = [
    "rv_5", "rv_10", "rv_21", "rv_63", "ewma_vol_94", "ewma_vol_97",
    "har_rv_daily", "har_rv_weekly", "har_rv_monthly",
    "acf_sq_ret_lag1", "acf_sq_ret_lag5", "acf_sq_ret_lag10",
    "garch_forecast_var", "garch_resid_lb_pvalue",
    "bb_width", "bb_width_roc", "macd_hist_mag", "rsi_14", "volume_ratio",
    "atr_14", "atr_roc", "intraday_range",
    "market_avg_rv21", "sector_avg_rv21",
    "vix_level", "vix9d_to_vix", "vix3m_to_vix", "corr_spy_21",
    "parkinson_5", "parkinson_10", "parkinson_21", "gk_5", "gk_10", "gk_21",
    "rskew_21", "rskew_63", "rkurt_21", "rkurt_63",
    "rv21_vs_market", "rv21_vs_sector", "rv_5_21_ratio", "rv_10_63_ratio",
    "garch_vs_rv21", "ewma_94_97_ratio", "vix_vs_rv21_ann",
    "gk_1", "log_gk_1", "log_gk_baseline_63",
    "dev_gk", "har_dev_5", "har_dev_22", "garch_persistence",
    "ret_1", "ret_5", "ret_21", "ret_1_neg",
    "overnight_gap", "overnight_vol_21", "overnight_to_intraday_21",
    "dow_mon", "dow_fri", "opex_friday", "month_end",
]

SELL_Z, BUY_Z, Z_MIN_OBS, MAX_DIV = 1.5, -1.25, 120, 0.25


def _project(frame: pd.DataFrame, phi: pd.Series, dte: int) -> pd.Series:
    out = np.empty(len(frame))
    b = frame["baseline_b"].to_numpy()
    dev = frame["predicted_dev"].to_numpy()
    ph = phi.to_numpy()
    for i in range(len(frame)):
        out[i] = project_term_vol(b[i], dev[i], ph[i], dte)
    return pd.Series(out, index=frame.index)


def _zgate_masks(div: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(sell_mask, buy_mask) over div's index: trailing-z triggers under the
    production thresholds and the |div| cap."""
    sell = pd.Series(False, index=div.index)
    buy = pd.Series(False, index=div.index)
    for _, g in div.groupby(level="symbol"):
        g = g.dropna()
        if len(g) <= Z_MIN_OBS:
            continue
        roll_mean = g.shift(1).rolling(Z_MIN_OBS, min_periods=Z_MIN_OBS).mean()
        roll_std = g.shift(1).rolling(Z_MIN_OBS, min_periods=Z_MIN_OBS).std()
        z = (g - roll_mean) / roll_std
        ok = g.abs() <= MAX_DIV
        sell.loc[g.index] = ((z >= SELL_Z) & ok).to_numpy()
        buy.loc[g.index] = ((z <= BUY_Z) & ok).to_numpy()
    return sell, buy


def _zgate_counts(div: pd.Series) -> tuple[int, int]:
    sell, buy = _zgate_masks(div)
    return int(sell.sum()), int(buy.sum())


def _earnings_excluded(index: pd.MultiIndex, cache_dir: Path, window_bd: int = 7) -> pd.Series:
    """True where the bot's earnings gate would block entry: an earnings
    impact date inside (t, t + window_bd business days]. Impact date = the
    trading day the reaction lands (AMC -> next session), mirroring the
    pipeline's convention."""
    df = pd.read_csv(cache_dir / "earnings_history.csv", parse_dates=["date"])
    impact_by_symbol: dict[str, np.ndarray] = {}
    for s, g in df.groupby("symbol"):
        when = g["when"].fillna("").astype(str)
        impact = g["date"].where(
            ~when.str.startswith("After"),
            g["date"] + pd.tseries.offsets.BDay(1),
        )
        impact_by_symbol[s] = np.sort(impact.to_numpy(dtype="datetime64[ns]"))
    out = pd.Series(False, index=index)
    dates = pd.DatetimeIndex(index.get_level_values("date"))
    horizon = dates + pd.tseries.offsets.BDay(window_bd)
    syms = index.get_level_values("symbol")
    for s in pd.unique(syms):
        imp = impact_by_symbol.get(s)
        if imp is None or not len(imp):
            continue
        m = syms == s
        lo = np.searchsorted(imp, dates[m].to_numpy(), side="right")
        hi = np.searchsorted(imp, horizon[m].to_numpy(), side="right")
        out.iloc[np.flatnonzero(m)] = hi > lo
    return out


def main() -> int:
    t0 = time.monotonic()
    settings = load_settings()
    watchlist = load_watchlist()
    end = date.today()
    start = end - timedelta(days=4 * 365)
    cache_dir = Path(settings.cache_db_path).parent

    with HistoricalStore(settings.cache_db_path) as store:
        from data import find_stale_symbols
        stale = set(find_stale_symbols(store, [t.symbol for t in watchlist]))
        watchlist = [t for t in watchlist if t.symbol not in stale]
        pipeline = FeaturePipeline(
            store, watchlist,
            iv_history=load_iv_history(cache_dir / "iv_history.csv"),
            earnings_history=load_earnings_history(cache_dir / "earnings_history.csv"),
        )
        feature_df = pipeline.build_features(start, end)
        bars = {t.symbol: store.get_bars(t.symbol, start, end) for t in watchlist}
        bars = {s: b for s, b in bars.items() if not b.empty}
    print(f"features {feature_df.shape} ({time.monotonic() - t0:.0f}s)")

    runs = {}
    for name, subset in (
        ("old", OLD_FEATURES),
        ("new", HORIZON_FEATURE_SETS[1]),
    ):
        t1 = time.monotonic()
        runs[name] = walk_forward_evaluate_h1(
            feature_df, bars,
            model_factory=lambda: LightGBMVolPredictor(horizon=1, hyperparams=HYPERPARAMS),
            feature_subset=subset,
            train_window_days=TRAIN_WINDOW_DAYS, refit_every=REFIT_EVERY,
        )
        print(f"{name} walk-forward: {len(runs[name])} rows ({time.monotonic() - t1:.0f}s)")
    runs["har"] = walk_forward_evaluate_h1(
        feature_df, bars, model_factory=HARRVPredictor,
        train_window_days=TRAIN_WINDOW_DAYS, refit_every=REFIT_EVERY,
    )

    common = runs["old"].index
    for f in runs.values():
        common = common.intersection(f.index)
    print(f"common OOS rows: {len(common)}")

    # --- DM vs HAR for both models ---
    print("\n=== margins + significance vs HAR ===")
    for name in ("old", "new"):
        parts = []
        for m, frame in (("lgbm", runs[name]), ("har", runs["har"])):
            g = frame.loc[common]
            parts.append(pd.DataFrame({
                "symbol": common.get_level_values("symbol"),
                "date": pd.to_datetime(common.get_level_values("date")),
                "model": m,
                "predicted_dev": g["predicted_dev"].to_numpy(),
                "actual_dev": g["actual_dev"].to_numpy(),
                "baseline_b": g["baseline_b"].to_numpy(),
                "actual_lv": g["actual_lv"].to_numpy(),
            }))
        dm = h1_dm_tests(pd.concat(parts, ignore_index=True), "lgbm", "har")
        g = runs[name].loc[common]
        h = runs["har"].loc[common]
        av = np.exp(2.0 * g["actual_lv"])
        q_m = qlike(av, np.exp(2.0 * (g["baseline_b"] + g["predicted_dev"])))
        q_h = qlike(av, np.exp(2.0 * (h["baseline_b"] + h["predicted_dev"])))
        w = within_ticker_r2(g["actual_dev"], g["predicted_dev"])
        print(f"  {name}: within_R2={w:+.4f}  QLIKE={q_m:.4f} vs har {q_h:.4f} "
              f"(margin {q_m - q_h:+.4f}, DM t={dm['qlike']['dm']:+.2f} p={dm['qlike']['p']:.4f}) "
              f"dev-MSE DM t={dm['dev_sq']['dm']:+.2f} p={dm['dev_sq']['p']:.4f}")

    # --- IV tracking + trigger thinning ---
    iv_df = load_iv_history(cache_dir / "iv_history.csv")
    iv_map = {
        s: g.drop_duplicates("date").set_index("date")["iv_current"].sort_index()
        for s, g in iv_df.groupby("symbol")
    }
    phi = feature_df["garch_persistence"].reindex(common).clip(PHI_MIN, PHI_MAX).fillna(0.94)

    iv_al = pd.Series(index=common, dtype=float)
    for s in common.get_level_values("symbol").unique():
        if s not in iv_map:
            continue
        dates = common[common.get_level_values("symbol") == s].get_level_values("date")
        vals = iv_map[s].reindex(pd.DatetimeIndex(dates)).ffill(limit=5)
        iv_al.loc[(s,)] = vals.to_numpy()

    print("\n=== IV tracking / trigger thinning (old vs new) ===")
    for dte in (7, 30):
        print(f"  -- DTE {dte} --")
        for name in ("old", "new"):
            fc = _project(runs[name].loc[common], phi, dte)
            paired = pd.concat([fc.rename("f"), iv_al.rename("iv")], axis=1).dropna()
            div = paired["f"] - paired["iv"]
            lvl_corr = paired["f"].corr(paired["iv"])
            chg = paired.groupby(level="symbol").diff(5).dropna()
            chg_corr = chg["f"].corr(chg["iv"]) if len(chg) else float("nan")
            n = len(div)
            counts = {th: int((div.abs() >= th).sum()) for th in (0.02, 0.05, 0.10)}
            sells, buys = _zgate_counts(div)
            print(f"    {name}: corr(level)={lvl_corr:+.3f} corr(5d-chg)={chg_corr:+.3f} "
                  f"mean|div|={div.abs().mean():.4f} p90|div|={div.abs().quantile(0.9):.4f} | "
                  f"|div|>=2/5/10pts: {counts[0.02]}/{counts[0.05]}/{counts[0.10]} of {n} | "
                  f"z-gate SELL={sells} BUY={buys}")

    # --- performance ON PROXY TRADE DAYS (DTE 7): z-gate triggers minus
    # earnings-window exclusions, i.e. the rows the bot would actually act
    # on (cost gate not replayable) ---
    print("\n=== performance on proxy trade days (DTE 7, earnings-gated) ===")
    from model.evaluation import sign_hit_rate
    blocked = _earnings_excluded(common, cache_dir, window_bd=7)
    for name in ("old", "new"):
        fc = _project(runs[name].loc[common], phi, 7)
        paired = pd.concat([fc.rename("f"), iv_al.rename("iv")], axis=1).dropna()
        div = paired["f"] - paired["iv"]
        sell, buy = _zgate_masks(div)
        trig = (sell | buy) & ~blocked.reindex(div.index, fill_value=False)
        rows_t = runs[name].loc[div.index[trig]]
        h_t = runs["har"].loc[div.index[trig]]
        w = within_ticker_r2(rows_t["actual_dev"], rows_t["predicted_dev"])
        sh = sign_hit_rate(rows_t["actual_dev"], rows_t["predicted_dev"])
        mse_m = float(((rows_t["predicted_dev"] - rows_t["actual_dev"]) ** 2).mean())
        mse_h = float(((h_t["predicted_dev"] - h_t["actual_dev"]) ** 2).mean())
        n_dates = rows_t.index.get_level_values("date").nunique()
        print(f"  {name}: n={len(rows_t)} rows ({int(sell.sum())}+{int(buy.sum())} "
              f"triggers, {int((~blocked.reindex(div.index, fill_value=False) & (sell | buy)).sum())} after earnings gate, "
              f"{n_dates} dates) within_R2={w:+.4f} sign_hit={sh:.3f} "
              f"dev-MSE={mse_m:.4f} vs har {mse_h:.4f}")

    # --- overnight share of close-to-close variance (the GK target ignores
    # the overnight gap; options price it) ---
    print("\n=== overnight share of close-to-close variance ===")
    shares = []
    for s, b in bars.items():
        c, o = b["close"], b["open"]
        r_on = np.log(o / c.shift(1)).dropna()
        r_cc = np.log(c / c.shift(1)).dropna()
        if len(r_cc) > 100 and float(r_cc.var()) > 0:
            shares.append(float(r_on.var() / r_cc.var()))
    print(f"  mean var(overnight)/var(close-to-close) across "
          f"{len(shares)} names: {np.mean(shares):.3f} (median {np.median(shares):.3f})")

    print(f"\ntotal: {(time.monotonic() - t0) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
