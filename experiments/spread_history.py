"""Seven-year spread-formula backtest on DoltHub IV history.

Same formula as experiments/spread_baseline.py, but over 2019->present using
the validated DoltHub composite ~30d IV (iv_current) instead of our two-month
divergence log:

  spread_est       = iv_current - trail63 (trailing 63d realized vol at scan)
  realized_premium = iv_current - realized vol over the NEXT 21 trading days
                     (matching the ~30-calendar-day tenor of iv_current)

The May-June 2026 result to stress-test: within-name corr +0.47, sell rule
tau=0.05 hit 79.7%. The question here is regime robustness — does the formula
survive COVID 2020, the 2021 meme regime, and the 2022 bear, and how badly
does the sell rule bleed when vol explodes?

Overlap handling: adjacent scrape dates share ~95% of their forward window, so
all headline stats use a per-symbol NON-OVERLAPPING subsample (each kept row's
forward window is disjoint from the previous kept row's). Full-sample numbers
are shown once for reference only.

Inputs (untracked, see experiments/dolthub_iv_pull.py and the Tradier
get_history pull documented there):
  experiments/results/dolthub_iv_history.csv   symbol,date,iv_current,hv_current
  experiments/results/closes_7yr.csv           symbol,date,close  (split-adjusted)

Run locally: python3 experiments/spread_history.py
Writes experiments/results/spread_history_rows.csv (non-overlapping sample).
"""
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ANN = math.sqrt(252)
# Forward window in trading days; 21 (~30 calendar) matches the iv_current
# tenor. Other values probe horizon structure but mismatch the IV tenor, so
# read levels with care there (term structure leaks into realized_premium).
FWD = int(sys.argv[1]) if len(sys.argv) > 1 else 21
TRAIL = 63
RESULTS = Path(__file__).resolve().parent / "results"


def corr_t(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    r = float(x[m].corr(y[m]))
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return r, t, n


def within(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby("symbol")[c].transform("mean")
    return out


def main() -> int:
    iv = (pd.read_csv(RESULTS / "dolthub_iv_history.csv")
          .dropna(subset=["iv_current"]))
    closes = pd.read_csv(RESULTS / "closes_7yr.csv")
    closes = closes[closes["date"] < date.today().isoformat()]  # drop partial bar

    recs = []
    for sym, g in closes.groupby("symbol"):
        g = g.sort_values("date")
        px = g["close"].to_numpy()
        rets = np.log(px[1:] / px[:-1])
        rdates = g["date"].to_numpy()[1:]  # return dated by its close
        for r in iv[iv["symbol"] == sym].itertuples():
            pos = np.searchsorted(rdates, r.date, side="right")
            past = rets[max(0, pos - TRAIL):pos]
            fwd = rets[pos:pos + FWD]
            if len(past) < TRAIL or len(fwd) < FWD:
                continue
            trail63 = float(past.std()) * ANN
            realized = float(fwd.std()) * ANN
            recs.append({
                "symbol": sym, "date": r.date, "pos": pos,
                "iv": r.iv_current, "trail63": trail63, "realized": realized,
                "spread_est": r.iv_current - trail63,
                "realized_premium": r.iv_current - realized,
            })
    df = pd.DataFrame(recs).sort_values(["symbol", "date"]).reset_index(drop=True)
    df["year"] = df["date"].str[:4]

    # Non-overlapping subsample: next kept row starts after this window ends.
    keep = []
    for _, g in df.groupby("symbol"):
        nxt = -1
        for i, pos in zip(g.index, g["pos"]):
            if pos >= nxt:
                keep.append(i)
                nxt = pos + FWD
    s = df.loc[keep]
    print(f"{len(df)} scrape rows -> {len(s)} non-overlapping windows, "
          f"{s.symbol.nunique()} symbols, {s.date.min()} -> {s.date.max()}")

    r, t, n = corr_t(df["spread_est"], df["realized_premium"])
    print(f"\nfull overlapping sample (reference only): corr {r:+.3f} (n={n})")

    print("\n=== headline: non-overlapping sample ===")
    r, t, n = corr_t(s["spread_est"], s["realized_premium"])
    print(f"pooled corr(spread_est, realized_premium): {r:+.3f} (t={t:+.2f}, n={n})")
    w = within(s, ("spread_est", "realized_premium"))
    r, t, n = corr_t(w["spread_est"], w["realized_premium"])
    print(f"within-symbol corr:                        {r:+.3f} (t={t:+.2f}, n={n})")
    print(f"sell-everything: mean premium {s.realized_premium.mean():+.4f}, "
          f"hit {(s.realized_premium > 0).mean():.1%}")

    print("\n=== within-symbol corr by year ===")
    for yr, g in w.groupby(s["year"]):
        r, t, n = corr_t(g["spread_est"], g["realized_premium"])
        base = s[s["year"] == yr]["realized_premium"]
        print(f"  {yr}: corr {r:+.3f} (t={t:+.2f}, n={n:3d})   "
              f"sell-all premium {base.mean():+.4f}, hit {(base > 0).mean():.0%}")

    print("\n=== realized premium by spread_est quintile ===")
    s2 = s.copy()
    s2["q"] = pd.qcut(s2["spread_est"], 5, labels=False, duplicates="drop")
    print(s2.groupby("q").agg(n=("realized_premium", "size"),
                              est=("spread_est", "mean"),
                              got=("realized_premium", "mean"),
                              hit=("realized_premium", lambda x: (x > 0).mean()))
          .round(4).to_string())

    print("\n=== SELL rule: spread_est > tau ===")
    for tau in (0.00, 0.03, 0.05, 0.08):
        sel = s[s["spread_est"] > tau]
        if not len(sel):
            continue
        print(f"  tau={tau:.2f}: n={len(sel):4d}  mean {sel.realized_premium.mean():+.4f}  "
              f"hit {(sel.realized_premium > 0).mean():5.1%}  "
              f"p5 {sel.realized_premium.quantile(0.05):+.4f}  "
              f"min {sel.realized_premium.min():+.4f}")

    print("\n=== SELL rule tau=0.05 by year (the regime question) ===")
    for yr, g in s.groupby("year"):
        sel = g[g["spread_est"] > 0.05]
        if not len(sel):
            print(f"  {yr}: no selections")
            continue
        print(f"  {yr}: n={len(sel):3d}  mean {sel.realized_premium.mean():+.4f}  "
              f"hit {(sel.realized_premium > 0).mean():5.1%}  "
              f"min {sel.realized_premium.min():+.4f}   "
              f"vs sell-all {g.realized_premium.mean():+.4f}")

    print("\n=== 10 worst rule-selected outcomes (tau=0.05) ===")
    worst = (s[s["spread_est"] > 0.05]
             .nsmallest(10, "realized_premium")
             [["symbol", "date", "iv", "trail63", "realized", "spread_est",
               "realized_premium"]])
    print(worst.round(3).to_string(index=False))

    print("\n=== BUY side: spread_est < 0 ===")
    buy = s[s["spread_est"] < 0]
    print(f"  n={len(buy)}  mean premium {buy.realized_premium.mean():+.4f}  "
          f"P(realized > IV) {(buy.realized_premium < 0).mean():5.1%}")
    deep = s[s["spread_est"] < -0.05]
    print(f"  deep (<-0.05): n={len(deep)}  mean {deep.realized_premium.mean():+.4f}  "
          f"P(realized > IV) {(deep.realized_premium < 0).mean():5.1%}")

    out = RESULTS / "spread_history_rows.csv"
    s.drop(columns=["pos"]).to_csv(out, index=False)
    print(f"\nsaved {out.name} ({len(s)} rows)")
    return 0


if __name__ == "__main__":
    main()
