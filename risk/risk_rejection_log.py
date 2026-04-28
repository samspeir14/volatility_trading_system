from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from risk.risk_manager import RiskDecision

logger = logging.getLogger(__name__)


CREATE_REJECTIONS_SQL = """
CREATE TABLE IF NOT EXISTS risk_rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rejected_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    expiration TEXT NOT NULL,
    direction TEXT NOT NULL,
    reasons_json TEXT NOT NULL
);
"""

CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_rejections_rejected_at "
    "ON risk_rejections(rejected_at DESC);"
)


# Maps substrings → human-readable categories for summary aggregation.
REASON_PATTERNS = [
    ("kill switch", "kill_switch"),
    ("per-trade budget", "per_trade_budget"),
    ("exposure", "ticker_exposure"),
    ("sector", "sector_concentration"),
    ("portfolio delta", "delta_limit"),
    ("portfolio gamma", "gamma_limit"),
    ("portfolio vega", "vega_limit"),
    ("required margin", "margin_buffer"),
]


def categorize_reason(reason: str) -> str:
    for pattern, category in REASON_PATTERNS:
        if pattern in reason:
            return category
    return "other"


class RiskRejectionLog:
    """Persistent log of risk-gate rejections. Co-located with the kill switch
    in `data/cache/risk_state.db` since both are risk-module state."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(CREATE_REJECTIONS_SQL)
        self._conn.execute(CREATE_INDEX_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def record_rejection(
        self,
        decision: "RiskDecision",
        rejected_at: datetime,
    ) -> None:
        if decision.approved:
            return  # nothing to record
        self._conn.execute(
            "INSERT INTO risk_rejections "
            "(rejected_at, symbol, expiration, direction, reasons_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                rejected_at.isoformat(),
                decision.signal.symbol,
                decision.signal.expiration.isoformat(),
                decision.signal.direction,
                json.dumps(decision.reasons),
            ),
        )
        self._conn.commit()

    def count_today(self, today: date) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM risk_rejections WHERE date(rejected_at) = ?",
            (today.isoformat(),),
        )
        return int(cur.fetchone()[0])

    def reasons_summary_today(self, today: date) -> dict[str, int]:
        cur = self._conn.execute(
            "SELECT reasons_json FROM risk_rejections WHERE date(rejected_at) = ?",
            (today.isoformat(),),
        )
        counts: dict[str, int] = {}
        for (reasons_json,) in cur.fetchall():
            try:
                reasons = json.loads(reasons_json)
            except json.JSONDecodeError:
                continue
            for reason in reasons:
                category = categorize_reason(reason)
                counts[category] = counts.get(category, 0) + 1
        return counts
