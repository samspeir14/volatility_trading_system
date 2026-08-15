"""Yang-Zhang-arm lab: total-vol target vs the production GK target.

Runs BOTH pipelines in parallel on identical bars — the production GK
(intraday-only) arm and the total-vol arm (overnight gap² + GK², the
Yang-Zhang-style single-day proxy) — same frozen top-25 feature names,
same run-4 hyperparams, walk-forward only. LAB-ONLY: nothing here touches
production; promotion happens only at a checkpoint with a fresh gate vote.

The two arms have DIFFERENT targets, so own-target within-R² is context,
not a comparison. The common rulers:
  1. QLIKE + DM against realized TOTAL variance (what options settle on) —
     the GK arm pays its structural bias here.
  2. IV basis: term-projected forecast vs composite IV — mean |div| and the
     SELL/BUY trigger split (the GK arm's SELL tilt should shrink in the
     total arm).
  3. Trade-day conditional metrics per arm (z-gate proxy, earnings-gated).

Run on the box: python -m experiments.yz_lab
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_settings, load_watchlist
from data import HistoricalStore
from experiments.replay_analysis import (
    HYPERPARAMS,
    _earnings_excluded,
    _project,
    _zgate_masks,
)
from features.feature_pipeline import (
    FeaturePipeline,
    HORIZON_FEATURE_SETS,
    load_earnings_history,
    load_iv_history,
)
from features.target import build_h1_deviation_target, daily_total_vol
from model.evaluation import (
    diebold_mariano,
    qlike,
    qlike_losses,
    sign_hit_rate,
    within_ticker_r2,
)
from model.har_model import HARRVPredictor
from model.lightgbm_model import LightGBMVolPredictor
from model.term_structure import PHI_MAX, PHI_MIN
from model.training import walk_forward_evaluate_h1

TRAIN_WINDOW_DAYS = 504
REFIT_EVERY = 21


def main() -> int:
    t0 = time.monotonic()
    settings = load_settings()
    watchlist = load_watchlist()
    end = date.today()
    start = end - timedelta(days=4 * 365)
    cache_dir = Path(settings.cache_db_path).parent
    iv_df = load_iv_history(cache_dir / "iv_history.csv")
    earn_df = load_earnings_history(cache_dir / "earnings_history.csv")

    with HistoricalStore(settings.cache_db_path) as store:
        from data import find_stale_symbols
        stale = set(find_stale_symbols(store, [t.symbol for t in watchlist]))
        watchlist = [t for t in watchlist if t.symbol not in stale]
        bars = {t.symbol: store.get_bars(t.symbol, start, end) for t in watchlist}
        bars = {s: b for s, b in bars.items() if not b.empty}

        arms: dict[str, dict] = {}
        for arm, proxy in (("gk", "gk"), ("total", "total")):
            t1 = time.monotonic()
            pipeline = FeaturePipeline(
                store, watchlist,
                iv_history=iv_df, earnings_history=earn_df,
                target_proxy=proxy,
            )
            fdf = pipeline.build_features(start, end)
            vol_fn = None if arm == "gk" else daily_total_vol
            wf = walk_forward_evaluate_h1(
                fdf, bars,
                model_factory=lambda: LightGBMVolPredictor(
                    horizon=1, hyperparams=HYPERPARAMS,
                ),
                feature_subset=HORIZON_FEATURE_SETS[1],
                train_window_days=TRAIN_WINDOW_DAYS, refit_every=REFIT_EVERY,
                vol_fn=vol_fn,
            )
            har = walk_forward_evaluate_h1(
                fdf, bars, model_factory=HARRVPredictor,
                train_window_days=TRAIN_WINDOW_DAYS, refit_every=REFIT_EVERY,
                vol_fn=vol_fn,
            )
            arms[arm] = {"features": fdf, "wf": wf, "har": har}
            print(f"{arm} arm: features {fdf.shape}, wf {len(wf)} rows "
                  f"({time.monotonic() - t1:.0f}s)", flush=True)

    # Realized TOTAL log-vol (the common ruler): lv_total at t+1, indexed at t.
    _y, b_tot, lv_tot = build_h1_deviation_target(bars, vol_fn=daily_total_vol)
    total_next_lv = _y + b_tot  # = lv_total(t+1), indexed (s, t)

    common = arms["gk"]["wf"].index
    for a in arms.values():
        common = common.intersection(a["wf"].index).intersection(a["har"].index)
    common = common.intersection(total_next_lv.dropna().index)
    print(f"common OOS rows: {len(common)}")
    actual_total_var = np.exp(2.0 * total_next_lv.loc[common])

    # --- common ruler 1: QLIKE vs realized TOTAL variance ---
    print("\n=== QLIKE vs realized TOTAL variance (common ruler) ===")
    losses: dict[str, pd.Series] = {}
    for arm in ("gk", "total"):
        g = arms[arm]["wf"].loc[common]
        forecast_var = np.exp(2.0 * (g["baseline_b"] + g["predicted_dev"]))
        losses[arm] = qlike_losses(actual_total_var, forecast_var)
        w_own = within_ticker_r2(g["actual_dev"], g["predicted_dev"])
        sh = sign_hit_rate(g["actual_dev"], g["predicted_dev"])
        print(f"  {arm:>5s}: QLIKE_total={qlike(actual_total_var, forecast_var):.4f} "
              f"(own-target within_R2={w_own:+.4f}, sign={sh:.3f} — context only)")
    dm = diebold_mariano(losses["total"], losses["gk"])
    print(f"  DM (total vs gk on QLIKE_total): t={dm['dm']:+.2f} p={dm['p']:.4f} "
          f"(negative = total-arm better; {dm['n_dates']} dates)")

    # --- common ruler 2: IV basis + trigger split ---
    print("\n=== IV basis (DTE 7) ===")
    iv_map = {
        s: g.drop_duplicates("date").set_index("date")["iv_current"].sort_index()
        for s, g in iv_df.groupby("symbol")
    }
    iv_al = pd.Series(index=common, dtype=float)
    for s in common.get_level_values("symbol").unique():
        if s not in iv_map:
            continue
        dates = common[common.get_level_values("symbol") == s].get_level_values("date")
        vals = iv_map[s].reindex(pd.DatetimeIndex(dates)).ffill(limit=5)
        iv_al.loc[(s,)] = vals.to_numpy()

    blocked = _earnings_excluded(common, cache_dir, window_bd=7)
    for arm in ("gk", "total"):
        g = arms[arm]["wf"].loc[common]
        phi = (arms[arm]["features"]["garch_persistence"]
               .reindex(common).clip(PHI_MIN, PHI_MAX).fillna(0.94))
        fc = _project(g, phi, 7)
        paired = pd.concat([fc.rename("f"), iv_al.rename("iv")], axis=1).dropna()
        div = paired["f"] - paired["iv"]
        sell, buy = _zgate_masks(div)
        trig = (sell | buy) & ~blocked.reindex(div.index, fill_value=False)
        rows_t = g.loc[div.index[trig]]
        w_t = within_ticker_r2(rows_t["actual_dev"], rows_t["predicted_dev"])
        sh_t = sign_hit_rate(rows_t["actual_dev"], rows_t["predicted_dev"])
        print(f"  {arm:>5s}: mean|div|={div.abs().mean():.4f} "
              f"mean(div)={div.mean():+.4f} "
              f"SELL={int(sell.sum())} BUY={int(buy.sum())} | "
              f"trade-days n={len(rows_t)}: within_R2={w_t:+.4f} sign={sh_t:.3f}")

    print("\nNO ACTION TAKEN — promotion only at a checkpoint with a fresh "
          "gate vote on the retrain's own OOS rows.")
    print(f"total: {(time.monotonic() - t0) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
