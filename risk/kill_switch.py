from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


CREATE_KILL_SQL = """
CREATE TABLE IF NOT EXISTS kill_switch_log (
    date TEXT PRIMARY KEY,
    activated_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    realized_pnl_at_trigger REAL NOT NULL,
    starting_equity REAL NOT NULL
);
"""

CREATE_EQUITY_SQL = """
CREATE TABLE IF NOT EXISTS equity_snapshots (
    date TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    captured_at TEXT NOT NULL
);
"""


class DailyKillSwitch:
    """Persistent daily-loss kill switch + starting-equity tracker.

    Once triggered for a given trading day, all subsequent risk-gate calls on
    that day reject signals with reason='kill switch active'. Resets at the
    next trading day (next date row).
    """

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(CREATE_KILL_SQL)
        self._conn.execute(CREATE_EQUITY_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def is_active(self, today: date) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM kill_switch_log WHERE date = ?",
            (today.isoformat(),),
        )
        return cur.fetchone() is not None

    def trigger(
        self,
        today: date,
        reason: str,
        realized_pnl: float,
        starting_equity: float,
    ) -> None:
        """Idempotent — calling twice on the same day is a no-op."""
        if self.is_active(today):
            return
        self._conn.execute(
            "INSERT INTO kill_switch_log "
            "(date, activated_at, reason, realized_pnl_at_trigger, starting_equity) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                today.isoformat(),
                datetime.now(timezone.utc).isoformat(),
                reason,
                realized_pnl,
                starting_equity,
            ),
        )
        self._conn.commit()
        logger.warning(
            "DAILY KILL SWITCH ACTIVATED for %s: %s (pnl=$%.2f, starting_equity=$%.2f)",
            today, reason, realized_pnl, starting_equity,
        )

    def evaluate_and_maybe_trigger(
        self,
        today: date,
        total_pnl_today: float,
        starting_equity: float,
        threshold_pct: float,
    ) -> bool:
        """If today_pnl <= threshold_pct × starting_equity, trigger.
        Returns the post-evaluation `is_active` state."""
        if self.is_active(today):
            return True
        threshold_dollars = threshold_pct * starting_equity
        if total_pnl_today <= threshold_dollars:
            self.trigger(
                today=today,
                reason=(
                    f"daily P&L ${total_pnl_today:.2f} <= threshold ${threshold_dollars:.2f} "
                    f"({threshold_pct:+.1%} of starting equity ${starting_equity:.2f})"
                ),
                realized_pnl=total_pnl_today,
                starting_equity=starting_equity,
            )
            return True
        return False

    def get_starting_equity(self, today: date, current_equity: float) -> float:
        """First call on a given date snapshots current_equity. Subsequent calls
        return that snapshot. Used by PortfolioStateBuilder for daily P&L
        calculations."""
        cur = self._conn.execute(
            "SELECT equity FROM equity_snapshots WHERE date = ?",
            (today.isoformat(),),
        )
        row = cur.fetchone()
        if row is not None:
            return float(row[0])
        # First observation today — record it
        self._conn.execute(
            "INSERT INTO equity_snapshots (date, equity, captured_at) VALUES (?, ?, ?)",
            (today.isoformat(), current_equity, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()
        return current_equity

    def trigger_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM kill_switch_log").fetchone()[0]

    def equity_snapshot_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM equity_snapshots").fetchone()[0]
