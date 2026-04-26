import asyncio
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data.async_client import AsyncTradierClient, Bar

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS daily_bars (
    symbol  TEXT NOT NULL,
    date    TEXT NOT NULL,
    open    REAL NOT NULL,
    high    REAL NOT NULL,
    low     REAL NOT NULL,
    close   REAL NOT NULL,
    volume  INTEGER NOT NULL,
    PRIMARY KEY (symbol, date)
);
"""

CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_bars_symbol_date "
    "ON daily_bars(symbol, date DESC);"
)


class HistoricalStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(CREATE_TABLE_SQL)
        self._conn.execute(CREATE_INDEX_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def upsert_bars(self, symbol: str, bars: list[Bar]) -> int:
        if not bars:
            return 0
        rows = [
            (symbol, b.date.isoformat(), b.open, b.high, b.low, b.close, b.volume)
            for b in bars
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO daily_bars "
            "(symbol, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def get_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume "
            "FROM daily_bars WHERE symbol = ? AND date >= ? AND date <= ? "
            "ORDER BY date ASC",
            self._conn,
            params=(symbol, start.isoformat(), end.isoformat()),
            parse_dates=["date"],
        )
        if not df.empty:
            df = df.set_index("date")
        return df

    def latest_date(self, symbol: str) -> date | None:
        cur = self._conn.execute(
            "SELECT MAX(date) FROM daily_bars WHERE symbol = ?",
            (symbol,),
        )
        row = cur.fetchone()
        if row and row[0]:
            return date.fromisoformat(row[0])
        return None

    def row_count(self, symbol: str) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM daily_bars WHERE symbol = ?",
            (symbol,),
        )
        return cur.fetchone()[0]


def compute_log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).dropna()


async def fetch_and_cache(
    client: AsyncTradierClient,
    store: HistoricalStore,
    symbols: list[str],
    lookback_years: int = 3,
    today: date | None = None,
) -> dict[str, pd.DataFrame]:
    end = today or date.today()
    start = end - timedelta(days=lookback_years * 365)

    async def fetch_one(symbol: str) -> tuple[str, pd.DataFrame]:
        latest = store.latest_date(symbol)
        if latest is None:
            fetch_start = start
        else:
            fetch_start = latest + timedelta(days=1)
        if fetch_start <= end:
            bars = await client.get_history(symbol, fetch_start, end)
            store.upsert_bars(symbol, bars)
        return symbol, store.get_bars(symbol, start, end)

    results = await asyncio.gather(*[fetch_one(s) for s in symbols])
    return dict(results)
