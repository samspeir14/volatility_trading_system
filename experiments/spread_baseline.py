"""Spread-game formula baseline: does `atm_iv - trail63` (a no-ML estimate of
the variance risk premium) predict the premium actually realized?

For every logged forecast row (divergence_history.db) whose option has expired:
  spread_est       = atm_iv - trailing-63d realized vol at scan (annualized)
  realized_premium = atm_iv - realized vol over scan -> expiration

If spread_est buckets order realized_premium, the formula already plays the
spread game and a future spread ML model must beat THIS, not just sell-
everything. Clusters by (symbol, expiration) since scan-days overlap heavily.

Run on EC2: venv/bin/python experiments/spread_baseline.py
Reads prod DBs read-only; writes experiments/results/spread_baseline_rows.csv
"""
import math
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ANN = math.sqrt(252)
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT = Path(__file__).resolve().parent / "results" / "spread_baseline_rows.csv"


def load_closes() -> dict[str, pd.Series]:
    conn = sqlite3.connect(CACHE / "market_data.db")
    out: dict[str, list] = {}
    for sym, d, c in conn.execute(
            "SELECT symbol, date, close FROM daily_bars ORDER BY symbol, date"):
        out.setdefault(sym, []).append((d, c))
    conn.close()
    return {s: pd.Series([c for _, c in v], index=[d for d, _ in v]) for s, v in out.items()}


def main() -> int:
    closes = load_closes()
    rets = {s: np.log(px / px.shift(1)).dropna() for s, px in closes.items()}
    bars_max = max(s.index.max() for s in closes.values())
    print(f"bars through {bars_max}")

    div = sqlite3.connect(CACHE / "divergence_history.db")
    log = pd.read_sql_query(
        "SELECT scan_date, symbol, expiration, atm_iv FROM divergence_log "
        "WHERE expiration <= ?", div, params=(bars_max,))
    div.close()
    # One forecast per (symbol, expiration, scan_date)
    log = log.drop_duplicates(subset=["scan_date", "symbol", "expiration"])

    recs = []
    for r in log.itertuples():
        rr = rets.get(r.symbol)
        if rr is None:
            continue
        past = rr[rr.index < r.scan_date]
        window = rr[(rr.index > r.scan_date) & (rr.index <= r.expiration)]
        if len(past) < 63 or len(window) < 2:
            continue
        trail63 = float(past.iloc[-63:].std(ddof=0)) * ANN
        realized = float(window.std(ddof=0)) * ANN
        recs.append({
            "symbol": r.symbol, "scan_date": r.scan_date, "expiration": r.expiration,
            "dte_trading": len(window), "atm_iv": r.atm_iv, "trail63": trail63,
            "realized": realized,
            "spread_est": r.atm_iv - trail63,
            "realized_premium": r.atm_iv - realized,
        })
    df = pd.DataFrame(recs)
    clusters = df.groupby(["symbol", "expiration"])
    print(f"{len(df)} forecast rows, {clusters.ngroups} (symbol, expiration) clusters")

    print(f"\nsell-everything baseline: mean realized premium "
          f"{df.realized_premium.mean():+.4f}, hit rate {(df.realized_premium > 0).mean():.1%}")

    # Cluster-level correlation: does the estimate order the outcome?
    cl = clusters.agg(est=("spread_est", "mean"), got=("realized_premium", "mean"),
                      dte=("dte_trading", "mean")).reset_index()
    r_all = cl["est"].corr(cl["got"])
    n = len(cl)
    t = r_all * math.sqrt((n - 2) / (1 - r_all ** 2))
    print(f"cluster-level corr(spread_est, realized_premium): {r_all:+.3f} (t={t:+.2f}, n={n})")

    print("\n=== realized premium by spread_est quintile (cluster level) ===")
    cl["q"] = pd.qcut(cl["est"], 5, labels=False, duplicates="drop")
    print(cl.groupby("q").agg(n=("got", "size"), est=("est", "mean"), got=("got", "mean"),
                              hit=("got", lambda s: (s > 0).mean())).round(4).to_string())

    print("\n=== SELL rule: spread_est > tau (cluster level) ===")
    for tau in (0.00, 0.03, 0.05, 0.08):
        sel = cl[cl["est"] > tau]
        if len(sel) == 0:
            continue
        print(f"  tau={tau:.2f}: n={len(sel):3d}  mean premium={sel['got'].mean():+.4f}  "
              f"hit={(sel['got'] > 0).mean():5.1%}")

    print("\n=== BUY side: spread_est < 0 (IV below trailing vol) ===")
    buy = cl[cl["est"] < 0]
    if len(buy):
        print(f"  n={len(buy)}  mean realized premium={buy['got'].mean():+.4f}  "
              f"P(realized > IV)={(buy['got'] < 0).mean():5.1%}")
    else:
        print("  no clusters with negative spread_est in this window")

    print("\n=== by DTE bucket, SELL rule tau=0.05 ===")
    for lo, hi in [(2, 5), (6, 10), (11, 21), (22, 99)]:
        sel = cl[(cl["est"] > 0.05) & (cl["dte"] >= lo) & (cl["dte"] <= hi)]
        base = cl[(cl["dte"] >= lo) & (cl["dte"] <= hi)]
        if len(base) == 0:
            continue
        s = f"{sel['got'].mean():+.4f} (n={len(sel)})" if len(sel) else "--"
        print(f"  DTE {lo:2d}-{hi:2d}: rule {s:>22s}   sell-all {base['got'].mean():+.4f} (n={len(base)})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nsaved {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    main()
