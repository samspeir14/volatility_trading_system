"""Backfill historical earnings dates for the watchlist from the free DoltHub
community earnings database (post-no-preference/earnings, earnings_calendar
table: act_symbol, date, when).

Coverage vetted 2026-08-14: ~April 2022 onward for large caps; `when` is
"After market close" / "Before market open" (occasionally empty). ETFs return
no rows, which is correct. One query per symbol (each returns well under the
~41-row API cap), polite 0.3s sleep between requests.

Output: data/cache/earnings_history.csv  (symbol, date, when)
Idempotent full rewrite — run on the EC2 box (and locally) whenever the
watchlist changes or ahead of a retrain to pick up newly scheduled dates.

Run: python -m scripts.backfill_earnings_history
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

API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/earnings/master"
OUT = PROJECT_ROOT / "data" / "cache" / "earnings_history.csv"


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


def main() -> int:
    from config import load_watchlist

    symbols = [t.symbol for t in load_watchlist()]
    OUT.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[tuple[str, str, str]] = []
    t0 = time.monotonic()
    for sym in symbols:
        rows = query(
            "SELECT act_symbol, date, `when` FROM earnings_calendar "
            f"WHERE act_symbol = '{sym}' AND date >= '2021-06-01' "
            "ORDER BY date"
        )
        for r in rows:
            rows_out.append((r["act_symbol"], r["date"], r.get("when") or ""))
        print(f"  {sym}: {len(rows)} earnings dates", flush=True)
        time.sleep(0.3)

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "date", "when"])
        w.writerows(rows_out)
    print(f"done: {len(rows_out)} rows -> {OUT} in {time.monotonic() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
