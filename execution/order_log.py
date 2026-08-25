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

CREATE_CLOSE_ATTEMPTS_SQL = """
CREATE TABLE IF NOT EXISTS close_attempts (
    closing_order_id INTEGER PRIMARY KEY,
    opening_order_id INTEGER NOT NULL,
    submitted_at TEXT NOT NULL,
    exit_trigger TEXT NOT NULL,
    order_type TEXT NOT NULL,
    submitted_price REAL NOT NULL,
    status TEXT NOT NULL,
    terminal_at TEXT,
    fill_price REAL
);
"""

CREATE_STALE_CLOSE_ALERTS_SQL = """
CREATE TABLE IF NOT EXISTS stale_close_alerts (
    opening_order_id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    expiration TEXT NOT NULL,
    structure TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    last_exit_trigger TEXT,
    detected_at TEXT NOT NULL
);
"""

# One row per (open position, day) holding its lifetime since-entry P&L as of
# that day's EOD summary. Day-over-day diff yields each position's daily P&L.
CREATE_POSITION_PNL_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS position_pnl_snapshots (
    opening_order_id INTEGER NOT NULL,
    as_of_date TEXT NOT NULL,
    lifetime_pnl REAL NOT NULL,
    PRIMARY KEY (opening_order_id, as_of_date)
);
"""

EXPIRATION_SENTINEL_ORDER_ID = 0  # closing_order_id used for auto-expired positions

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_orders_symbol_submitted ON submitted_orders(symbol, submitted_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_orders_status ON submitted_orders(final_status);",
    "CREATE INDEX IF NOT EXISTS idx_orders_fingerprint ON submitted_orders(fingerprint, submitted_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_failed_fingerprint ON failed_submissions(fingerprint, submitted_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_close_attempts_opening ON close_attempts(opening_order_id);",
    "CREATE INDEX IF NOT EXISTS idx_close_attempts_status ON close_attempts(status);",
]

PENDING_CLOSE_STATUS = "pending"
STALE_CANCELED_STATUS = "stale_canceled"
# partially_filled counts as failed: the executed chunk is booked by shrinking
# the opening order's leg quantities to the remainder, and the attempt must
# burn retry budget so a repeatedly-partial close still escalates to the
# stale-close alert instead of retrying forever.
FAILED_CLOSE_STATUSES = {"stale_canceled", "canceled", "rejected", "expired", "partially_filled"}

CLOSE_COLUMNS_MIGRATIONS = [
    ("closing_order_id", "ALTER TABLE submitted_orders ADD COLUMN closing_order_id INTEGER"),
    ("closed_at", "ALTER TABLE submitted_orders ADD COLUMN closed_at TEXT"),
    ("exit_trigger", "ALTER TABLE submitted_orders ADD COLUMN exit_trigger TEXT"),
    ("realized_pnl", "ALTER TABLE submitted_orders ADD COLUMN realized_pnl REAL"),
    # TCA: the theoretical mid at submission time (per contract). Together
    # with fill_price this yields realized slippage per fill.
    ("arrival_mid", "ALTER TABLE submitted_orders ADD COLUMN arrival_mid REAL"),
    # Earnings-exit fail-closed store: the ticker's next earnings date as
    # known at entry, refreshed on every healthy calendar read. When the
    # calendar API dies, the exit manager falls back to this stored value —
    # a short-vol position must never ride through earnings just because
    # Finnhub was down.
    ("next_earnings_date", "ALTER TABLE submitted_orders ADD COLUMN next_earnings_date TEXT"),
    # VRP calibration z at entry (h=1 pipeline) — every signal record logs it.
    ("vrp_z", "ALTER TABLE submitted_orders ADD COLUMN vrp_z REAL"),
]

CLOSE_ATTEMPT_COLUMNS_MIGRATIONS = [
    ("arrival_mid", "ALTER TABLE close_attempts ADD COLUMN arrival_mid REAL"),
]


class OrderLog:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(CREATE_SUBMITTED_SQL)
        self._conn.execute(CREATE_FAILED_SQL)
        self._conn.execute(CREATE_ASSIGNMENT_ALERTS_SQL)
        self._conn.execute(CREATE_CLOSE_ATTEMPTS_SQL)
        self._conn.execute(CREATE_STALE_CLOSE_ALERTS_SQL)
        self._conn.execute(CREATE_POSITION_PNL_SNAPSHOTS_SQL)
        self._migrate_close_columns()
        for sql in CREATE_INDEXES_SQL:
            self._conn.execute(sql)
        self._conn.commit()

    def _migrate_close_columns(self) -> None:
        self._migrate_columns("submitted_orders", CLOSE_COLUMNS_MIGRATIONS)
        self._migrate_columns("close_attempts", CLOSE_ATTEMPT_COLUMNS_MIGRATIONS)

    def _migrate_columns(self, table: str, migrations: list[tuple[str, str]]) -> None:
        cur = self._conn.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cur.fetchall()}
        for col_name, sql in migrations:
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
        arrival_mid: float | None = None,
        next_earnings_date: date | None = None,
    ) -> None:
        legs_json = json.dumps([asdict(leg) for leg in signal.legs])
        self._conn.execute(
            "INSERT OR REPLACE INTO submitted_orders ("
            "tradier_order_id, fingerprint, submitted_at, symbol, expiration, "
            "direction, structure, horizon_lower, horizon_upper, weight_lower, "
            "underlying_price_at_signal, atm_iv_at_signal, predicted_iv_at_signal, "
            "divergence_at_signal, cross_sectional_z, time_series_z, "
            "submitted_price, legs_json, arrival_mid, next_earnings_date, vrp_z"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                order_id, fingerprint, submitted_at.isoformat(), signal.symbol,
                signal.expiration.isoformat(), signal.direction, structure,
                signal.horizon_lower, signal.horizon_upper, signal.weight_lower,
                signal.underlying_price, signal.atm_iv,
                signal.predicted_iv_equivalent, signal.divergence,
                signal.cross_sectional_z, signal.time_series_z,
                submitted_price, legs_json, arrival_mid,
                next_earnings_date.isoformat() if next_earnings_date else None,
                getattr(signal, "vrp_z", None),
            ),
        )
        self._conn.commit()

    def update_next_earnings_date(
        self, order_id: int, next_earnings_date: date | None
    ) -> None:
        """Refresh the fail-closed earnings store for an open position after a
        healthy calendar read (report dates move)."""
        self._conn.execute(
            "UPDATE submitted_orders SET next_earnings_date = ? "
            "WHERE tradier_order_id = ?",
            (
                next_earnings_date.isoformat() if next_earnings_date else None,
                order_id,
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

    def get_submitted_order(self, order_id: int) -> dict | None:
        """Fetch a single submitted_orders row by tradier_order_id."""
        cur = self._conn.execute(
            "SELECT tradier_order_id, fingerprint, submitted_at, symbol, expiration, "
            "direction, structure, horizon_lower, horizon_upper, weight_lower, "
            "underlying_price_at_signal, atm_iv_at_signal, predicted_iv_at_signal, "
            "divergence_at_signal, cross_sectional_z, time_series_z, "
            "submitted_price, legs_json, final_status, fill_price, closing_order_id, "
            "closed_at, exit_trigger, realized_pnl "
            "FROM submitted_orders WHERE tradier_order_id = ?",
            (order_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def record_failed_close_submission(
        self,
        opening_order_id: int,
        exit_trigger: str,
        submitted_at: datetime,
        reason: str,
    ) -> None:
        """A close that never reached the exchange (preview/place rejected)
        still burns retry budget: without a row here, a close that Tradier
        rejects every cycle (e.g. a size mismatch after an unresolved partial
        fill) would retry forever without ever tripping max_close_retries or
        the stale-close alert. Synthetic negative closing_order_id keeps the
        primary key clear of real Tradier ids."""
        cur = self._conn.execute(
            "SELECT COALESCE(MIN(closing_order_id), 0) FROM close_attempts "
            "WHERE closing_order_id < 0",
        )
        synthetic_id = int(cur.fetchone()[0]) - 1
        self._conn.execute(
            "INSERT INTO close_attempts ("
            "closing_order_id, opening_order_id, submitted_at, exit_trigger, "
            "order_type, submitted_price, status, terminal_at, fill_price"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                synthetic_id, opening_order_id, submitted_at.isoformat(),
                f"{exit_trigger} [{reason[:80]}]", "rejected_submission", 0.0,
                "rejected", submitted_at.isoformat(), None,
            ),
        )
        self._conn.commit()

    def partial_entry_orders(self) -> list[dict]:
        """Entry rows stuck at final_status='partially_filled' — unresolved
        partial fills awaiting resolve_partial_entries(). Includes rows the
        reconciler's timeout recovery marked partially_filled."""
        cur = self._conn.execute(
            "SELECT tradier_order_id, symbol, fill_price FROM submitted_orders "
            "WHERE final_status = 'partially_filled' ORDER BY submitted_at",
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def update_leg_quantities(self, order_id: int, units: int) -> bool:
        """Rewrite every leg's quantity in legs_json to `units` — the
        partial-fill resolution path: after a multi-lot order executes only
        `units` structure units (entry) or leaves `units` open (close), the
        stored legs must describe what is actually held so the tracker,
        exit manager, and close orders act on real size. Returns False when
        the row is missing or its legs_json cannot be parsed."""
        row = self.get_submitted_order(order_id)
        if row is None:
            return False
        try:
            legs = json.loads(row["legs_json"])
            for leg in legs:
                leg["quantity"] = int(units)
            new_json = json.dumps(legs)
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            return False
        self._conn.execute(
            "UPDATE submitted_orders SET legs_json = ? WHERE tradier_order_id = ?",
            (new_json, order_id),
        )
        self._conn.commit()
        return True

    def record_close_attempt(
        self,
        opening_order_id: int,
        closing_order_id: int,
        submitted_at: datetime,
        exit_trigger: str,
        order_type: str,
        submitted_price: float,
        status: str = PENDING_CLOSE_STATUS,
        arrival_mid: float | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO close_attempts ("
            "closing_order_id, opening_order_id, submitted_at, exit_trigger, "
            "order_type, submitted_price, status, arrival_mid"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (closing_order_id, opening_order_id, submitted_at.isoformat(),
             exit_trigger, order_type, submitted_price, status, arrival_mid),
        )
        self._conn.commit()

    def update_close_attempt(
        self,
        closing_order_id: int,
        status: str,
        terminal_at: datetime | None = None,
        fill_price: float | None = None,
    ) -> None:
        self._conn.execute(
            "UPDATE close_attempts SET status = ?, terminal_at = ?, fill_price = ? "
            "WHERE closing_order_id = ?",
            (status, terminal_at.isoformat() if terminal_at else None,
             fill_price, closing_order_id),
        )
        self._conn.commit()

    def has_pending_close(self, opening_order_id: int) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM close_attempts WHERE opening_order_id = ? "
            "AND status = ? LIMIT 1",
            (opening_order_id, PENDING_CLOSE_STATUS),
        )
        return cur.fetchone() is not None

    def pending_close_attempts(self) -> list[dict]:
        """All close_attempts currently in 'pending' state. Stale-close-manager
        walks this list each cycle."""
        cur = self._conn.execute(
            "SELECT ca.closing_order_id, ca.opening_order_id, ca.submitted_at, "
            "ca.exit_trigger, ca.order_type, ca.submitted_price, ca.status, "
            "so.symbol, so.expiration, so.structure, so.direction "
            "FROM close_attempts ca "
            "JOIN submitted_orders so ON so.tradier_order_id = ca.opening_order_id "
            "WHERE ca.status = ? "
            "ORDER BY ca.submitted_at",
            (PENDING_CLOSE_STATUS,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def failed_close_attempt_count(self, opening_order_id: int) -> int:
        """Number of close attempts that ended in a failed terminal state
        (stale-canceled, canceled, rejected, expired). Pending and filled
        attempts are NOT counted — only failed attempts count against the
        retry budget."""
        placeholders = ",".join("?" * len(FAILED_CLOSE_STATUSES))
        cur = self._conn.execute(
            f"SELECT COUNT(*) FROM close_attempts "
            f"WHERE opening_order_id = ? AND status IN ({placeholders})",
            (opening_order_id, *sorted(FAILED_CLOSE_STATUSES)),
        )
        return int(cur.fetchone()[0])

    def record_stale_close_alert(
        self,
        opening_order_id: int,
        symbol: str,
        expiration: date,
        structure: str,
        attempts: int,
        last_exit_trigger: str | None,
        detected_at: datetime,
    ) -> bool:
        """Insert a stale-close alert. Returns True if inserted, False if a
        row for this opening_order_id already existed (idempotent)."""
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO stale_close_alerts "
            "(opening_order_id, symbol, expiration, structure, attempts, "
            "last_exit_trigger, detected_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (opening_order_id, symbol, expiration.isoformat(), structure,
             attempts, last_exit_trigger, detected_at.isoformat()),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def stale_close_alerts_active(self) -> list[dict]:
        """Stale-close alerts for positions that are STILL open. An alert is
        auto-resolved once its opening order gets a closing_order_id by any
        path — a real close, a between-cycle fill, or expiration settled by the
        reconciler (sentinel id=0). The LEFT JOIN keeps any orphan alert whose
        opening order row is missing (fail-safe: surface, don't silently drop).
        Matches the `closing_order_id IS NULL` convention in
        open_unclosed_positions()."""
        cur = self._conn.execute(
            "SELECT sca.opening_order_id, sca.symbol, sca.expiration, "
            "sca.structure, sca.attempts, sca.last_exit_trigger, sca.detected_at "
            "FROM stale_close_alerts sca "
            "LEFT JOIN submitted_orders so "
            "ON so.tradier_order_id = sca.opening_order_id "
            "WHERE so.closing_order_id IS NULL "
            "ORDER BY sca.detected_at DESC"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def record_position_pnl_snapshot(
        self, opening_order_id: int, as_of_date: date, lifetime_pnl: float,
    ) -> None:
        """Persist a position's lifetime since-entry P&L as of `as_of_date`.
        Idempotent per (position, day) so re-running the EOD summary overwrites
        rather than duplicates."""
        self._conn.execute(
            "INSERT OR REPLACE INTO position_pnl_snapshots "
            "(opening_order_id, as_of_date, lifetime_pnl) VALUES (?, ?, ?)",
            (opening_order_id, as_of_date.isoformat(), lifetime_pnl),
        )
        self._conn.commit()

    def previous_position_pnl(
        self, opening_order_id: int, before_date: date,
    ) -> float | None:
        """Most recent recorded lifetime P&L for this position strictly before
        `before_date` (the prior trading day, skipping weekend/holiday gaps).
        None if we've never recorded one — caller treats today's lifetime P&L
        as the daily P&L (position is new since the last snapshot)."""
        cur = self._conn.execute(
            "SELECT lifetime_pnl FROM position_pnl_snapshots "
            "WHERE opening_order_id = ? AND as_of_date < ? "
            "ORDER BY as_of_date DESC LIMIT 1",
            (opening_order_id, before_date.isoformat()),
        )
        row = cur.fetchone()
        return float(row[0]) if row is not None else None

    def open_unclosed_positions(self) -> list[dict]:
        """Rows from submitted_orders that filled successfully and haven't been closed yet."""
        cur = self._conn.execute(
            "SELECT tradier_order_id, fingerprint, submitted_at, symbol, expiration, "
            "direction, structure, horizon_lower, horizon_upper, weight_lower, "
            "underlying_price_at_signal, atm_iv_at_signal, predicted_iv_at_signal, "
            "divergence_at_signal, cross_sectional_z, time_series_z, "
            "submitted_price, legs_json, final_status, fill_price, next_earnings_date "
            "FROM submitted_orders "
            "WHERE final_status IN ('filled', 'partially_filled') "
            "AND closing_order_id IS NULL "
            "ORDER BY submitted_at DESC"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
