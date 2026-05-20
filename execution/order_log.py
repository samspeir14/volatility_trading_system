from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from signals.signal_generator import TradeSignal


CREATE_SUBMITTED_SQL = """
CREATE TABLE IF NOT EXISTS submitted_orders (
    tradier_order_id INTEGER PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    expiration TEXT NOT NULL,
    direction TEXT NOT NULL,
    structure TEXT NOT NULL,
    horizon_lower INTEGER NOT NULL,
    horizon_upper INTEGER NOT NULL,
    weight_lower REAL NOT NULL,
    underlying_price_at_signal REAL NOT NULL,
    atm_iv_at_signal REAL NOT NULL,
    predicted_iv_at_signal REAL NOT NULL,
    divergence_at_signal REAL NOT NULL,
    cross_sectional_z REAL NOT NULL,
    time_series_z REAL,
    submitted_price REAL NOT NULL,
    legs_json TEXT NOT NULL,
    final_status TEXT,
    fill_price REAL,
    filled_at TEXT,
    error_message TEXT
);
"""

CREATE_FAILED_SQL = """
CREATE TABLE IF NOT EXISTS failed_submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    expiration TEXT NOT NULL,
    direction TEXT NOT NULL,
    reason TEXT NOT NULL
);
"""

CREATE_ASSIGNMENT_ALERTS_SQL = """
CREATE TABLE IF NOT EXISTS assignment_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tradier_order_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    expiration TEXT NOT NULL,
    structure TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    stock_quantity REAL,
    UNIQUE(tradier_order_id)
);
"""

EXPIRATION_SENTINEL_ORDER_ID = 0  # closing_order_id used for auto-expired positions

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_orders_symbol_submitted ON submitted_orders(symbol, submitted_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_orders_status ON submitted_orders(final_status);",
    "CREATE INDEX IF NOT EXISTS idx_orders_fingerprint ON submitted_orders(fingerprint, submitted_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_failed_fingerprint ON failed_submissions(fingerprint, submitted_at DESC);",
]

CLOSE_COLUMNS_MIGRATIONS = [
    ("closing_order_id", "ALTER TABLE submitted_orders ADD COLUMN closing_order_id INTEGER"),
    ("closed_at", "ALTER TABLE submitted_orders ADD COLUMN closed_at TEXT"),
    ("exit_trigger", "ALTER TABLE submitted_orders ADD COLUMN exit_trigger TEXT"),
    ("realized_pnl", "ALTER TABLE submitted_orders ADD COLUMN realized_pnl REAL"),
]


class OrderLog:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(CREATE_SUBMITTED_SQL)
        self._conn.execute(CREATE_FAILED_SQL)
        self._conn.execute(CREATE_ASSIGNMENT_ALERTS_SQL)
        self._migrate_close_columns()
        for sql in CREATE_INDEXES_SQL:
            self._conn.execute(sql)
        self._conn.commit()

    def _migrate_close_columns(self) -> None:
        cur = self._conn.execute("PRAGMA table_info(submitted_orders)")
        existing = {row[1] for row in cur.fetchall()}
        for col_name, sql in CLOSE_COLUMNS_MIGRATIONS:
            if col_name not in existing:
                self._conn.execute(sql)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def record_submission(
        self,
        signal: "TradeSignal",
        fingerprint: str,
        structure: str,
        submitted_price: float,
        order_id: int,
        submitted_at: datetime,
    ) -> None:
        legs_json = json.dumps([asdict(leg) for leg in signal.legs])
        self._conn.execute(
            "INSERT OR REPLACE INTO submitted_orders ("
            "tradier_order_id, fingerprint, submitted_at, symbol, expiration, "
            "direction, structure, horizon_lower, horizon_upper, weight_lower, "
            "underlying_price_at_signal, atm_iv_at_signal, predicted_iv_at_signal, "
            "divergence_at_signal, cross_sectional_z, time_series_z, "
            "submitted_price, legs_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                order_id, fingerprint, submitted_at.isoformat(), signal.symbol,
                signal.expiration.isoformat(), signal.direction, structure,
                signal.horizon_lower, signal.horizon_upper, signal.weight_lower,
                signal.underlying_price, signal.atm_iv,
                signal.predicted_iv_equivalent, signal.divergence,
                signal.cross_sectional_z, signal.time_series_z,
                submitted_price, legs_json,
            ),
        )
        self._conn.commit()

    def update_terminal_state(
        self,
        order_id: int,
        status: str,
        fill_price: float | None,
        filled_at: datetime | None,
        error: str | None = None,
    ) -> None:
        self._conn.execute(
            "UPDATE submitted_orders SET final_status = ?, fill_price = ?, "
            "filled_at = ?, error_message = ? WHERE tradier_order_id = ?",
            (
                status,
                fill_price,
                filled_at.isoformat() if filled_at else None,
                error,
                order_id,
            ),
        )
        self._conn.commit()

    def record_failure(
        self,
        signal: "TradeSignal",
        fingerprint: str,
        reason: str,
        submitted_at: datetime,
    ) -> None:
        self._conn.execute(
            "INSERT INTO failed_submissions "
            "(fingerprint, submitted_at, symbol, expiration, direction, reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                fingerprint, submitted_at.isoformat(),
                signal.symbol, signal.expiration.isoformat(),
                signal.direction, reason,
            ),
        )
        self._conn.commit()

    def has_recent_open_order(self, fingerprint: str, hours: int = 24) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        cur = self._conn.execute(
            "SELECT tradier_order_id, final_status FROM submitted_orders "
            "WHERE fingerprint = ? AND submitted_at >= ?",
            (fingerprint, cutoff.isoformat()),
        )
        rows = cur.fetchall()
        # Considered "open" unless the order's terminal state is rejected / canceled / expired
        TERMINAL_FAILED = {"rejected", "canceled", "expired"}
        for _, status in rows:
            if status is None or status not in TERMINAL_FAILED:
                return True
        return False

    def open_orders_by_symbol(self, symbol: str) -> list[dict]:
        cur = self._conn.execute(
            "SELECT tradier_order_id, fingerprint, submitted_at, expiration, "
            "direction, structure, submitted_price, final_status "
            "FROM submitted_orders WHERE symbol = ? "
            "ORDER BY submitted_at DESC",
            (symbol,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def submitted_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM submitted_orders").fetchone()[0]

    def failed_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM failed_submissions").fetchone()[0]

    def record_close(
        self,
        opening_order_id: int,
        closing_order_id: int,
        closed_at: datetime,
        exit_trigger: str,
        realized_pnl: float,
    ) -> None:
        self._conn.execute(
            "UPDATE submitted_orders SET closing_order_id = ?, closed_at = ?, "
            "exit_trigger = ?, realized_pnl = ? WHERE tradier_order_id = ?",
            (closing_order_id, closed_at.isoformat(), exit_trigger, realized_pnl, opening_order_id),
        )
        self._conn.commit()

    def record_expiration(
        self,
        opening_order_id: int,
        expired_at: datetime,
        realized_pnl: float,
        exit_trigger: str = "expired_worthless",
    ) -> None:
        """Mark a position closed by expiration (no closing trade). Uses the
        sentinel closing_order_id=0 so open_unclosed_positions excludes it
        and closed_today_pnl picks up the realized leg."""
        self._conn.execute(
            "UPDATE submitted_orders SET closing_order_id = ?, closed_at = ?, "
            "exit_trigger = ?, realized_pnl = ? WHERE tradier_order_id = ?",
            (EXPIRATION_SENTINEL_ORDER_ID, expired_at.isoformat(),
             exit_trigger, realized_pnl, opening_order_id),
        )
        self._conn.commit()

    def record_assignment_alert(
        self,
        tradier_order_id: int,
        symbol: str,
        expiration: date,
        structure: str,
        detected_at: datetime,
        stock_quantity: float | None = None,
    ) -> None:
        """Idempotent — duplicate inserts for the same order are ignored."""
        self._conn.execute(
            "INSERT OR IGNORE INTO assignment_alerts "
            "(tradier_order_id, symbol, expiration, structure, detected_at, stock_quantity) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (tradier_order_id, symbol, expiration.isoformat(), structure,
             detected_at.isoformat(), stock_quantity),
        )
        self._conn.commit()

    def assignment_alerts_active(self) -> list[dict]:
        """All recorded assignment alerts. Once acknowledged manually, the
        operator can DELETE the row to dismiss it."""
        cur = self._conn.execute(
            "SELECT tradier_order_id, symbol, expiration, structure, "
            "detected_at, stock_quantity FROM assignment_alerts "
            "ORDER BY detected_at DESC"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def positions_opened_on(self, today: date) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM submitted_orders "
            "WHERE date(submitted_at) = ? "
            "AND final_status IN ('filled', 'partially_filled')",
            (today.isoformat(),),
        )
        return int(cur.fetchone()[0])

    def positions_closed_on(self, today: date) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM submitted_orders "
            "WHERE closing_order_id IS NOT NULL "
            "AND date(closed_at) = ?",
            (today.isoformat(),),
        )
        return int(cur.fetchone()[0])

    def exit_triggers_today(self, today: date) -> dict[str, int]:
        cur = self._conn.execute(
            "SELECT exit_trigger, COUNT(*) FROM submitted_orders "
            "WHERE closing_order_id IS NOT NULL "
            "AND date(closed_at) = ? "
            "AND exit_trigger IS NOT NULL "
            "GROUP BY exit_trigger",
            (today.isoformat(),),
        )
        return {row[0]: int(row[1]) for row in cur.fetchall()}

    def closed_today_pnl(self, today: date) -> float:
        """Sum of realized_pnl for orders whose closing trade filled on `today`."""
        cur = self._conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) FROM submitted_orders "
            "WHERE closing_order_id IS NOT NULL "
            "AND date(closed_at) = ?",
            (today.isoformat(),),
        )
        return float(cur.fetchone()[0])

    def timeout_orders(self) -> list[dict]:
        """Rows with final_status='timeout' — submitted to Tradier but the
        submission-time poll didn't reach a terminal state. They may have
        filled / rejected / been canceled in reality; the reconciler queries
        Tradier to find out."""
        cur = self._conn.execute(
            "SELECT tradier_order_id, fingerprint, submitted_at, symbol, expiration, "
            "direction, structure, submitted_price, legs_json "
            "FROM submitted_orders "
            "WHERE final_status = 'timeout' AND closing_order_id IS NULL "
            "ORDER BY submitted_at"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def open_unclosed_positions(self) -> list[dict]:
        """Rows from submitted_orders that filled successfully and haven't been closed yet."""
        cur = self._conn.execute(
            "SELECT tradier_order_id, fingerprint, submitted_at, symbol, expiration, "
            "direction, structure, horizon_lower, horizon_upper, weight_lower, "
            "underlying_price_at_signal, atm_iv_at_signal, predicted_iv_at_signal, "
            "divergence_at_signal, cross_sectional_z, time_series_z, "
            "submitted_price, legs_json, final_status, fill_price "
            "FROM submitted_orders "
            "WHERE final_status IN ('filled', 'partially_filled') "
            "AND closing_order_id IS NULL "
            "ORDER BY submitted_at DESC"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
