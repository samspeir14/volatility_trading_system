from __future__ import annotations

import math
import sqlite3
from datetime import date
from pathlib import Path

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS divergence_log (
    scan_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    expiration TEXT NOT NULL,
    horizon_lower INTEGER NOT NULL,
    horizon_upper INTEGER NOT NULL,
    weight_lower REAL NOT NULL,
    predicted_iv_equivalent REAL NOT NULL,
    atm_iv REAL NOT NULL,
    divergence REAL NOT NULL,
    underlying_price REAL NOT NULL,
    PRIMARY KEY (scan_date, symbol, expiration, horizon_lower, horizon_upper)
);
"""

CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_div_symbol_horizons "
    "ON divergence_log(symbol, horizon_lower, horizon_upper, scan_date DESC);"
)

DEFAULT_LOOKBACK = 60
MIN_HISTORY = 20


class DivergenceHistory:
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

    def log_signals(self, signals, scan_date: date) -> int:
        """Persist a batch of TradeSignal objects. Idempotent on
        (scan_date, symbol, expiration, horizon_lower, horizon_upper)."""
        rows = [
            (
                scan_date.isoformat(),
                s.symbol,
                s.expiration.isoformat(),
                s.horizon_lower,
                s.horizon_upper,
                s.weight_lower,
                s.predicted_iv_equivalent,
                s.atm_iv,
                s.divergence,
                s.underlying_price,
            )
            for s in signals
        ]
        if not rows:
            return 0
        self._conn.executemany(
            "INSERT OR REPLACE INTO divergence_log "
            "(scan_date, symbol, expiration, horizon_lower, horizon_upper, weight_lower, "
            " predicted_iv_equivalent, atm_iv, divergence, underlying_price) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def time_series_z_score(
        self,
        symbol: str,
        horizon_lower: int,
        horizon_upper: int,
        current_divergence: float,
        lookback: int = DEFAULT_LOOKBACK,
        min_history: int = MIN_HISTORY,
    ) -> float | None:
        """Return z-score of `current_divergence` against the most recent
        `lookback` historical divergences for this (symbol, h_lo, h_up).
        Returns None if fewer than `min_history` past observations exist."""
        cur = self._conn.execute(
            "SELECT divergence FROM divergence_log "
            "WHERE symbol = ? AND horizon_lower = ? AND horizon_upper = ? "
            "ORDER BY scan_date DESC LIMIT ?",
            (symbol, horizon_lower, horizon_upper, lookback),
        )
        rows = [r[0] for r in cur.fetchall()]
        if len(rows) < min_history:
            return None
        n = len(rows)
        mean = sum(rows) / n
        var = sum((x - mean) ** 2 for x in rows) / n
        std = math.sqrt(var)
        if std == 0:
            return None
        return (current_divergence - mean) / std

    def row_count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM divergence_log")
        return cur.fetchone()[0]
