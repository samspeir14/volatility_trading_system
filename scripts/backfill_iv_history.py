"""Backfill daily composite IV/HV history for the FULL watchlist from the free
DoltHub community options database (post-no-preference/options,
volatility_history table).

Data notes (vetted 2026-07-02, experiments/dolthub_iv_pull.py):
- Coverage ~2019-02 onward; ~3 scrapes/week 2019-2024, daily from 2025.
- iv_current is a Barchart-style composite ~30d chain-weighted IV, NOT
  per-expiration ATM IV. Levels have an offset vs our logged atm_iv;
  changes/ranks are the tradeable signal.

Strategy: per-symbol pagination keyed on the last date already stored —
`WHERE act_symbol='X' AND date > '<last>' ORDER BY date LIMIT 40` (a full
page is always under the ~41-row API cap). Seeds itself from
experiments/results/dolthub_iv_history.csv (the 2026-07 research pull, 19
symbols) when the output file does not exist yet, so only the ~14 newer
watchlist names pull deep history. Resumable: re-running continues from
the stored frontier per symbol.

Output: data/cache/iv_history.csv  (symbol, date, iv_current, hv_current)

Run: python -m scripts.backfill_iv_history
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
OUT = PROJECT_ROOT / "data" / "cache" / "iv_history.csv"
SEED = PROJECT_ROOT / "experiments" / "results" / "dolthub_iv_history.csv"
START_DATE = "2021-06-01"  # comfortably before the 4y training window
PAGE = 40


def query(q: str, retries: int = 5) -> list[dict]:
    url = f"{API}?q={urllib.parse.quote(q)}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = json.load(resp)
            if payload.get("query_execution_status") in ("Success", "RowLimit"):
                return payload.get("rows", [])
            raise RuntimeError(payload.get("query_execution_message", "unknown error"))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5.0 * (attempt + 1))
    return []


def _load_existing() -> dict[str, str]:
    """Return {symbol: max_date_stored}. Seeds OUT from the research CSV on
    first run."""
    if not OUT.exists() and SEED.exists():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(SEED) as src, open(OUT, "w", newline="") as dst:
            dst.write(src.read())
        print(f"seeded {OUT} from {SEED}")
    frontier: dict[str, str] = defaultdict(str)
    if OUT.exists():
        with open(OUT) as f:
            for row in csv.DictReader(f):
                if row["date"] > frontier[row["symbol"]]:
                    frontier[row["symbol"]] = row["date"]
    return frontier


def main() -> int:
    from config import load_watchlist

    symbols = [t.symbol for t in load_watchlist()]
    frontier = _load_existing()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    total = 0
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if OUT.stat().st_size == 0:
            w.writerow(["symbol", "date", "iv_current", "hv_current"])
        for sym in symbols:
            last = frontier.get(sym) or START_DATE
            pulled = 0
            while True:
                rows = query(
                    "SELECT date, act_symbol, iv_current, hv_current "
                    "FROM volatility_history "
                    f"WHERE act_symbol = '{sym}' AND date > '{last}' "
                    f"ORDER BY date LIMIT {PAGE}"
                )
                if not rows:
                    break
                for r in rows:
                    w.writerow([r["act_symbol"], r["date"],
                                r["iv_current"], r["hv_current"]])
                last = rows[-1]["date"]
                pulled += len(rows)
                f.flush()
                time.sleep(0.3)
                if len(rows) < PAGE:
                    break
            total += pulled
            print(f"  {sym}: +{pulled} rows (through {last})", flush=True)
    print(f"done: +{total} rows -> {OUT} in {time.monotonic() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
