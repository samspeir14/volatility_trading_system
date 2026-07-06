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


def query(q: str, retries: int = 3) -> list[dict]:
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
            time.sleep(2.0 * (attempt + 1))
    return []


def pull_symbol(symbol: str) -> list[dict]:
    rows, last = [], "1900-01-01"
    while True:
        page = query(
            "SELECT date, iv_current, hv_current FROM volatility_history "
            f"WHERE act_symbol='{symbol}' AND date > '{last}' ORDER BY date LIMIT 41"
        )
        if not page:
            return rows
        rows.extend(page)
        last = page[-1]["date"]
        time.sleep(0.3)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "date", "iv_current", "hv_current"])
        total = 0
        for sym in SYMBOLS:
            t0 = time.monotonic()
            rows = pull_symbol(sym)
            for r in rows:
                w.writerow([sym, r["date"], r["iv_current"], r["hv_current"]])
            f.flush()
            total += len(rows)
            print(f"{sym}: {len(rows)} rows in {time.monotonic()-t0:.0f}s", flush=True)
    print(f"done: {total} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
