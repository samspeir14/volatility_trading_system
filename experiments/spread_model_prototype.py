"""Spread-model prototype: can ML learn when the variance risk premium is
fat vs thin, where the naive formula (iv - trail63) failed across regimes?

Target: y = iv_current - realized vol over the next 21 trading days (the
premium actually collected, in vol points, at the iv_current ~30d tenor).
Features are all computable at scan time: IV level/changes/percentile,
trailing vols and their ratios (vol-momentum = "storm incoming" detectors),
the dead formula itself, return momentum/drawdown, and market-level stress
(VIX level/changes/percentile, VIX9D/VIX and VIX/VIX3M term structure).

Validation: expanding walk-forward by calendar year (test 2020..2026).
Training rows must have their forward window CLOSED before the test year
starts (embargo — no leakage across the boundary). Training uses overlapping
daily rows; all reported stats use per-symbol NON-OVERLAPPING test windows.
No hyperparameter tuning — fixed, conservative LightGBM params declared
upfront, so nothing is fit to the test years.

The bar (from experiments/spread_history.py, 2026-07-06):
  - sell-everything: +0.0028 mean premium/window at this tenor, hit 62.7%
  - dead formula within-name corr: -0.16 (it sells INTO crashes)
Success = positive within-name OOS corr per year, top-vs-bottom quintile
separation, and not stepping on the 2020 mine it never trained on.

Inputs: experiments/results/{dolthub_iv_history,closes_7yr,vix_7yr}.csv
Run locally: python3 experiments/spread_model_prototype.py
Writes experiments/results/spread_model_oos.csv (non-overlapping OOS rows).
"""
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ANN = math.sqrt(252)
FWD = 21
IV_STALE_LIMIT = 5   # ffill DoltHub scrapes (~3/week pre-2025) at most this far
RESULTS = Path(__file__).resolve().parent / "results"
TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

LGB_PARAMS = dict(
    objective="regression", n_estimators=400, learning_rate=0.03,
    num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=7,
)

FEATURES = [
    "iv", "iv_chg_5", "iv_chg_21", "iv_pctl", "hv_dolt",
    "trail5", "trail21", "trail63", "vol_ratio_5_63", "vol_ratio_21_63",
    "spread_trail63", "spread_trail21",
    "ret_5", "ret_21", "dd_63",
    "vix", "vix_chg_5", "vix_pctl", "vix_ts_9d", "vix_ts_3m", "mkt_spread",
    "iv_minus_vix",
]


def build_panel() -> pd.DataFrame:
    iv = pd.read_csv(RESULTS / "dolthub_iv_history.csv")
    closes = pd.read_csv(RESULTS / "closes_7yr.csv")
    vix = (pd.read_csv(RESULTS / "vix_7yr.csv")
           .pivot(index="date", columns="symbol", values="close") / 100.0)

    # Market-level stress block, shared across symbols.
    spy = closes[closes.symbol == "SPY"].set_index("date")["close"].sort_index()
    spy_trail21 = np.log(spy / spy.shift(1)).rolling(21).std(ddof=0) * ANN
    mkt = pd.DataFrame({
        "vix": vix["VIX"],
        "vix_chg_5": vix["VIX"].diff(5),
        "vix_pctl": vix["VIX"].rolling(252, min_periods=126).rank(pct=True),
        "vix_ts_9d": vix["VIX9D"] / vix["VIX"],
        "vix_ts_3m": vix["VIX"] / vix["VIX3M"],
        "mkt_spread": vix["VIX"] - spy_trail21,
    })

    frames = []
    for sym, g in closes.groupby("symbol"):
        g = g.sort_values("date").set_index("date")
        px = g["close"]
        r = np.log(px / px.shift(1))
        f = pd.DataFrame(index=g.index)
        gi = iv[iv.symbol == sym].set_index("date").reindex(g.index)
        f["iv"] = gi["iv_current"].ffill(limit=IV_STALE_LIMIT)
        f["hv_dolt"] = gi["hv_current"].ffill(limit=IV_STALE_LIMIT)
        f["iv_chg_5"] = f["iv"] - f["iv"].shift(5)
        f["iv_chg_21"] = f["iv"] - f["iv"].shift(21)
        f["iv_pctl"] = f["iv"].rolling(252, min_periods=126).rank(pct=True)
        for w in (5, 21, 63):
            f[f"trail{w}"] = r.rolling(w).std(ddof=0) * ANN
        f["vol_ratio_5_63"] = f["trail5"] / f["trail63"]
        f["vol_ratio_21_63"] = f["trail21"] / f["trail63"]
        f["spread_trail63"] = f["iv"] - f["trail63"]
        f["spread_trail21"] = f["iv"] - f["trail21"]
        f["ret_5"] = np.log(px / px.shift(5))
        f["ret_21"] = np.log(px / px.shift(21))
        f["dd_63"] = px / px.rolling(63).max() - 1.0
        f = f.join(mkt)
        f["iv_minus_vix"] = f["iv"] - f["vix"]
        # Forward window: returns t+1 .. t+FWD, and the date it closes.
        f["realized_fwd"] = r.rolling(FWD).std(ddof=0).shift(-FWD) * ANN
        f["window_end"] = pd.Series(g.index, index=g.index).shift(-FWD)
        f["symbol"] = sym
        frames.append(f.reset_index())
    df = pd.concat(frames, ignore_index=True)
    df["y"] = df["iv"] - df["realized_fwd"]
    df = df.dropna(subset=["iv", "trail63", "y", "window_end"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def nonoverlap(g: pd.DataFrame) -> pd.DataFrame:
    """Greedy per-symbol subsample with disjoint forward windows."""
    keep, nxt = [], ""
    for i, (d, we) in enumerate(zip(g["date"], g["window_end"])):
        if d >= nxt:
            keep.append(i)
            nxt = we
    return g.iloc[keep]


def corr_t(x, y):
    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    return r, r * math.sqrt((n - 2) / (1 - r * r)), n


def demean(g: pd.DataFrame, cols) -> pd.DataFrame:
    out = g.copy()
    for c in cols:
        out[c] = out[c] - out.groupby("symbol")[c].transform("mean")
    return out


def main() -> int:
    df = build_panel()
    print(f"panel: {len(df)} rows, {df.symbol.nunique()} symbols, "
          f"{df.date.min()} -> {df.date.max()}")

    oos, importances = [], []
    for yr in TEST_YEARS:
        t0, t1 = f"{yr}-01-01", f"{yr}-12-31"
        train = df[df["window_end"] < t0]
        test = df[(df["date"] >= t0) & (df["date"] <= t1)]
        if len(train) < 500 or len(test) == 0:
            print(f"{yr}: skipped (train={len(train)})")
            continue
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(train[FEATURES], train["y"])
        test = test.copy()
        test["pred"] = model.predict(test[FEATURES])
        test["test_year"] = yr
        oos.append(pd.concat([nonoverlap(g) for _, g in test.groupby("symbol")]))
        importances.append(pd.Series(model.feature_importances_, index=FEATURES))
        print(f"{yr}: trained on {len(train)} rows "
              f"(through {train.date.max()}), test {len(test)} rows")

    o = pd.concat(oos, ignore_index=True)
    print(f"\nOOS non-overlapping windows: {len(o)}")

    print("\n=== headline (all OOS years pooled) ===")
    w = demean(o, ("pred", "y", "spread_trail63"))
    r, t, n = corr_t(w["pred"], w["y"])
    print(f"model  within-name corr(pred, realized premium): {r:+.3f} (t={t:+.2f}, n={n})")
    r, t, n = corr_t(w["spread_trail63"], w["y"])
    print(f"dead formula on same rows:                       {r:+.3f} (t={t:+.2f})")
    print(f"sell-everything mean premium: {o.y.mean():+.4f}, hit {(o.y > 0).mean():.1%}")

    print("\n=== within-name corr by test year: model vs dead formula ===")
    for yr, g in o.groupby("test_year"):
        gw = demean(g, ("pred", "y", "spread_trail63"))
        rm, tm, n = corr_t(gw["pred"], gw["y"])
        rf, _, _ = corr_t(gw["spread_trail63"], gw["y"])
        print(f"  {yr}: model {rm:+.3f} (t={tm:+.2f}, n={n:3d})   formula {rf:+.3f}   "
              f"sell-all {g.y.mean():+.4f}")

    print("\n=== realized premium by PREDICTED-premium quintile (pooled OOS) ===")
    o["q"] = pd.qcut(o["pred"], 5, labels=False, duplicates="drop")
    print(o.groupby("q").agg(n=("y", "size"), pred=("pred", "mean"), got=("y", "mean"),
                             hit=("y", lambda s: (s > 0).mean())).round(4).to_string())
    top, bot = o[o.q == o.q.max()], o[o.q == 0]
    print(f"top-minus-bottom quintile spread: "
          f"{top.y.mean() - bot.y.mean():+.4f} vol pts/window")

    print("\n=== trading rule: SELL when pred > 0.02 (fixed upfront) ===")
    for yr, g in o.groupby("test_year"):
        sel = g[g["pred"] > 0.02]
        flag = f"n={len(sel):3d}  mean {sel.y.mean():+.4f}  hit {(sel.y > 0).mean():5.1%}  " \
               f"min {sel.y.min():+.4f}" if len(sel) else "no selections"
        print(f"  {yr}: {flag}   vs sell-all {g.y.mean():+.4f} (n={len(g)})")

    print("\n=== the 2020 mine: model predictions entering COVID ===")
    feb_mar = o[(o.test_year == 2020) & (o.date >= "2020-02-15") & (o.date <= "2020-03-31")]
    jan = o[(o.test_year == 2020) & (o.date < "2020-02-15")]
    print(f"  Jan-mid-Feb 2020: mean pred {jan.pred.mean():+.4f}, realized {jan.y.mean():+.4f} (n={len(jan)})")
    print(f"  late-Feb-Mar 2020: mean pred {feb_mar.pred.mean():+.4f}, realized {feb_mar.y.mean():+.4f} (n={len(feb_mar)})")
    print(f"  rule exposure late-Feb-Mar: {len(feb_mar[feb_mar.pred > 0.02])} of {len(feb_mar)} windows sold")

    print("\n=== feature importance (mean split gain rank across folds) ===")
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    print(imp.head(10).round(1).to_string())

    out = RESULTS / "spread_model_oos.csv"
    o.drop(columns=["q"]).to_csv(out, index=False)
    print(f"\nsaved {out.name} ({len(o)} rows)")
    return 0


if __name__ == "__main__":
    main()
