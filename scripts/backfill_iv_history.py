"""Backfill daily composite IV/HV history for the FULL watchlist from the free
DoltHub community options database (post-no-preference/options,
volatility_history table).

Data notes (vetted 2026-07-02, experiments/dolthub_iv_pull.py):
- Coverage ~2019-02 onward; ~3 scrapes/week 2019-2024, daily from 2025.
- iv_current is a Barchart-style composite ~30d chain-weighted IV, NOT
  per-expiration ATM IV. Levels have an offset vs our logged atm_iv;
  changes/ranks are the tradeable signal.

Strategy: the table's PK leads with `date`, so EXACT-DATE lookups are fast
index hits while anything range-shaped full-scans into the server's 30s
deadline (learned the hard way; see experiments/dolthub_iv_pull.py). One
query per weekday date with all watchlist symbols (33 rows, under the ~41-row
API cap). Resumable: continues from the last date already in the CSV.

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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
OUT = PROJECT_ROOT / "data" / "cache" / "iv_history.csv"
START_DATE = "2021-06-01"  # comfortably before the 4y training window


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


def _resume_date() -> str:
    """Last date already in the CSV (dates are written in order), or the
    backfill start."""
    last = ""
    if OUT.exists():
        with open(OUT) as f:
            for row in csv.DictReader(f):
                if row["date"] > last:
                    last = row["date"]
    return last or START_DATE


def main() -> int:
    from datetime import date as ddate, timedelta

    from config import load_watchlist

    symbols = [t.symbol for t in load_watchlist()]
    in_list = ",".join(f"'{s}'" for s in symbols)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    resume = _resume_date()
    d = ddate.fromisoformat(resume) + timedelta(days=1)
    weekdays: list[str] = []
    while d <= ddate.today():
        if d.weekday() < 5:
            weekdays.append(d.isoformat())
        d += timedelta(days=1)
    print(f"pulling {len(weekdays)} dates after {resume} for {len(symbols)} symbols")

    t0 = time.monotonic()
    total = 0
    write_header = not OUT.exists() or OUT.stat().st_size == 0
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["symbol", "date", "iv_current", "hv_current"])
        for i, day in enumerate(weekdays):
            rows = query(
                "SELECT date, act_symbol, iv_current, hv_current "
                "FROM volatility_history "
                f"WHERE date = '{day}' AND act_symbol IN ({in_list}) "
                "ORDER BY act_symbol"
            )
            for r in rows:
                w.writerow([r["act_symbol"], r["date"],
                            r["iv_current"], r["hv_current"]])
            total += len(rows)
            f.flush()
            if i % 100 == 0:
                print(f"  ...{day}: {total} rows ({time.monotonic() - t0:.0f}s)",
                      flush=True)
            time.sleep(0.25)
    print(f"done: +{total} rows -> {OUT} in {time.monotonic() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
