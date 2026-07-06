"""Pull daily IV/HV history for the watchlist from the free DoltHub community
options database (post-no-preference/options, volatility_history table).

The SQL API caps responses at ~41 rows, so this paginates by date per symbol.
One-time research pull for spread-model prototyping — polite 0.3s sleep between
requests, ~600 requests total for 20 symbols x ~7 years.

Output: experiments/results/dolthub_iv_history.csv
Columns: symbol, date, iv_current, hv_current

Data notes from vetting (2026-07-02):
- Coverage: ~2019-02 onward; ~3 scrapes/week 2019-2024, daily from 2025.
- iv_current is a Barchart-style composite IV (~30d, chain-weighted), NOT
  per-expiration ATM IV — expect a level offset vs our logged atm_iv,
  especially near earnings. Validate before trusting levels; changes/ranks
  are the useful signal.
- Per-contract history exists in option_chain (~2020 onward, point-queryable
  by (date, act_symbol)) if per-expiration ATM IV is ever needed.
"""
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AMD", "JPM", "GS",
           "BAC", "XOM", "CVX", "JNJ", "UNH", "TSLA", "HD", "CAT", "BA", "SPY", "QQQ"]
OUT = Path(__file__).resolve().parent / "results" / "dolthub_iv_history.csv"


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
            time.sleep(5.0 * (attempt + 1))  # patient backoff: soft rate limits
    return []


def main() -> int:
    """Iterate weekday dates in pairs (2 dates x 20 symbols = 40 rows, under
    the ~41-row response cap). The PK leads with `date`, so exact-date queries
    are fast index lookups; anything range-shaped full-scans and hits the
    server's 30s deadline. Appends and resumes from the last date in the CSV."""
    from datetime import date as ddate, timedelta

    OUT.parent.mkdir(parents=True, exist_ok=True)
    start = ddate(2019, 2, 1)
    if OUT.exists():
        with open(OUT) as f:
            line = ""
            for line in f:
                pass
        parts = line.strip().split(",")
        if len(parts) >= 2 and parts[1] != "date":
            start = ddate.fromisoformat(parts[1]) + timedelta(days=1)
            print(f"resuming from {start}", flush=True)

    weekdays = []
    d = start
    while d <= ddate.today():
        if d.weekday() < 5:
            weekdays.append(d.isoformat())
        d += timedelta(days=1)

    in_list = ",".join(f"'{s}'" for s in SYMBOLS)
    mode = "a" if start != ddate(2019, 2, 1) else "w"
    total = 0
    t0 = time.monotonic()
    with open(OUT, mode, newline="") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["symbol", "date", "iv_current", "hv_current"])
        for i in range(0, len(weekdays), 2):
            pair = ",".join(f"'{x}'" for x in weekdays[i:i + 2])
            rows = query(
                "SELECT date, act_symbol, iv_current, hv_current FROM volatility_history "
                f"WHERE date IN ({pair}) AND act_symbol IN ({in_list}) "
                "ORDER BY date, act_symbol"
            )
            for r in rows:
                w.writerow([r["act_symbol"], r["date"], r["iv_current"], r["hv_current"]])
            f.flush()
            total += len(rows)
            if i % 100 == 0:
                print(f"...{weekdays[i]}: {total} rows ({time.monotonic()-t0:.0f}s)", flush=True)
            time.sleep(0.25)
    print(f"done: {total} rows -> {OUT} in {time.monotonic()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
