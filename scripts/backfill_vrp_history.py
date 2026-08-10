"""One-time backfill of the vrp_log gap history from the existing
divergence_log.

The VRP calibration gate needs >= VRP_MIN_OBS (default 120) daily gap
observations per ticker before it emits any signal. The bot has been logging
ATM IV per (scan_date, symbol, expiration) since spring 2026 in
divergence_log; this script reconstructs g = log(atm_iv) - log(realized_vol)
for each of those rows, tenor-matched via the row's DTE (expiration -
scan_date) against the bars-derived trailing Garman-Klass vol AS OF that
scan date. Point-in-time: only bars <= scan_date enter each row's realized
vol.

Idempotent (INSERT OR IGNORE — live rows logged by the bot win over
backfill). Prints per-ticker observation counts against the gate threshold.

Run on the box after deploy:
    .venv/bin/python -m scripts.backfill_vrp_history
"""
from __future__ import annotations

import math
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_settings
from data import HistoricalStore
from features.target import GK_EPS, daily_ohlc_vol
from signals.divergence_history import VRP_MIN_OBS
from signals.signal_generator import (
    CALENDAR_DAYS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    VRP_MIN_REALIZED_WINDOW,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIVERGENCE_DB = PROJECT_ROOT / "data" / "cache" / "divergence_history.db"


def backfill(conn: sqlite3.Connection, store: HistoricalStore) -> dict[str, int]:
    """Insert reconstructed gap rows; returns per-symbol inserted counts."""
    rows = conn.execute(
        "SELECT scan_date, symbol, expiration, atm_iv FROM divergence_log "
        "WHERE atm_iv > 0 ORDER BY symbol, scan_date"
    ).fetchall()
    print(f"divergence_log rows with usable atm_iv: {len(rows)}")

    # Cache per-symbol squared daily GK vol series once.
    gk_sq_by_symbol: dict[str, pd.Series] = {}

    def _gk_sq(symbol: str) -> pd.Series | None:
        if symbol not in gk_sq_by_symbol:
            bars = store.get_bars(symbol, date(2000, 1, 1), date.today())
            if bars.empty:
                gk_sq_by_symbol[symbol] = None
            else:
                gk_sq_by_symbol[symbol] = daily_ohlc_vol(bars) ** 2
        return gk_sq_by_symbol[symbol]

    inserted: dict[str, int] = {}
    to_insert: list[tuple] = []
    for scan_date_s, symbol, expiration_s, atm_iv in rows:
        gk_sq = _gk_sq(symbol)
        if gk_sq is None:
            continue
        scan_d = date.fromisoformat(scan_date_s)
        dte = (date.fromisoformat(expiration_s) - scan_d).days
        if dte < 1:
            continue
        window = max(
            VRP_MIN_REALIZED_WINDOW,
            round(dte * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR),
        )
        # Point-in-time AND estimator-matched to the live path: the bot's
        # cache only holds bars through the weekday BEFORE the scan date
        # (partial same-day bars are never cached), so exclude the scan
        # date's own bar here too.
        upto = gk_sq.loc[: pd.Timestamp(scan_d) - pd.Timedelta(days=1)].dropna()
        if len(upto) < window:
            continue
        realized = math.sqrt(
            TRADING_DAYS_PER_YEAR * float(np.mean(upto.iloc[-window:].to_numpy()))
        )
        g = math.log(atm_iv) - math.log(realized + GK_EPS)
        to_insert.append((scan_date_s, symbol, dte, atm_iv, realized, g))
        inserted[symbol] = inserted.get(symbol, 0) + 1

    conn.executemany(
        "INSERT OR IGNORE INTO vrp_log "
        "(scan_date, symbol, dte, atm_iv, realized_vol, g) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        to_insert,
    )
    conn.commit()
    return inserted


def main() -> int:
    settings = load_settings()
    if not DIVERGENCE_DB.exists():
        print(f"no divergence history at {DIVERGENCE_DB}", file=sys.stderr)
        return 1

    # Open via DivergenceHistory first so the vrp_log table + migrations exist.
    from signals.divergence_history import DivergenceHistory

    with DivergenceHistory(DIVERGENCE_DB):
        pass
    conn = sqlite3.connect(str(DIVERGENCE_DB))
    store = HistoricalStore(settings.cache_db_path)
    try:
        backfill(conn, store)
        print("\nper-ticker distinct gap days vs gate threshold "
              f"(need >= {VRP_MIN_OBS}):")
        counts = conn.execute(
            "SELECT symbol, COUNT(DISTINCT scan_date) FROM vrp_log "
            "GROUP BY symbol ORDER BY symbol"
        ).fetchall()
        below = 0
        for symbol, n in counts:
            marker = "OK " if n >= VRP_MIN_OBS else "-- "
            if n < VRP_MIN_OBS:
                below += 1
            print(f"  {marker}{symbol:<6s} {n}")
        print(f"\n{len(counts) - below}/{len(counts)} tickers meet the "
              f"{VRP_MIN_OBS}-obs threshold; the rest accrue from live logging "
              f"(visible in the daily Gates block as vrp_history)")
    finally:
        conn.close()
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
