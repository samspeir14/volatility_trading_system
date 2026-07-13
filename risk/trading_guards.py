"""Operational guards that sit between the strategy and the market.

Everything here answers one question — "should the bot be allowed to OPEN new
positions right now?" — or proves the process is alive:

  * HaltFlag          — manual master off-switch (a file the operator creates)
  * DrawdownBreaker   — weekly/monthly drawdown circuit breakers, persistent
  * BarsFreshnessGuard— refuses to trade on a frozen daily-bars cache
  * write_heartbeat / read_heartbeat — dead-man switch consumed by
    scripts/heartbeat_check.py from cron

None of these ever block EXITS: managing existing positions must keep working
no matter what tripped. That mirrors the daily kill switch's contract in
MainLoop.run_once.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class HaltFlag:
    """Manual master off-switch.

    The operator creates the flag file (default: data/cache/HALT) to stop new
    entries without touching the process; the file's contents become the
    reason string shown in logs/Slack. Deleting the file resumes trading on
    the next cycle. See deploy/RUNBOOK.md.
    """

    def __init__(self, path: Path):
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def reason(self) -> str | None:
        """Reason string when halted, None when trading is allowed.
        An unreadable flag file fails SAFE (treated as halted)."""
        try:
            if not self._path.exists():
                return None
            text = self._path.read_text().strip()
        except OSError as e:
            return f"HALT file unreadable ({e}) — failing safe"
        return text or "HALT file present"


CREATE_BREAKER_SQL = """
CREATE TABLE IF NOT EXISTS breaker_log (
    kind TEXT NOT NULL,
    date TEXT NOT NULL,
    activated_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    current_equity REAL NOT NULL,
    peak_equity REAL NOT NULL,
    PRIMARY KEY (kind, date)
);
"""


@dataclass(frozen=True)
class _BreakerSpec:
    kind: str            # "weekly" | "monthly"
    lookback_sessions: int
    threshold_pct: float  # positive number; trigger at drawdown <= -threshold


class DrawdownBreaker:
    """Multi-day drawdown circuit breakers on top of the daily kill switch.

    The daily switch only sees today's P&L against today's open; a slow bleed
    (-3% a day for three days) never trips it. These breakers compare current
    equity against the PEAK of the daily starting-equity snapshots that
    DailyKillSwitch.get_starting_equity already records in the same DB:

      * weekly:  -8% from the peak of the last  5 sessions
      * monthly: -12% from the peak of the last 21 sessions

    Once triggered, a breaker blocks new entries for the remainder of its
    calendar window (ISO week / calendar month). It re-evaluates fresh after
    that, so if equity is still deep below the rolling peak it re-triggers —
    trading only resumes once equity stabilizes near the (decayed) peak.
    Early manual clear: delete the breaker_log row (see deploy/RUNBOOK.md).
    """

    def __init__(
        self,
        db_path: Path,
        weekly_threshold_pct: float = 0.08,
        monthly_threshold_pct: float = 0.12,
    ):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(CREATE_BREAKER_SQL)
        self._conn.commit()
        self._specs = (
            _BreakerSpec("weekly", 5, weekly_threshold_pct),
            _BreakerSpec("monthly", 21, monthly_threshold_pct),
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def evaluate_and_maybe_trigger(
        self, today: date, current_equity: float
    ) -> str | None:
        """Returns the active breaker's stored reason (stable across cycles so
        the caller can dedupe alerts), or None when entries are allowed."""
        active = self._active_reason(today)
        if active is not None:
            return active
        if current_equity is None or current_equity <= 0:
            return None
        for spec in self._specs:
            peak = self._peak_equity(spec.lookback_sessions)
            if peak is None or peak <= 0:
                continue
            drawdown = current_equity / peak - 1.0
            if drawdown <= -spec.threshold_pct:
                reason = (
                    f"{spec.kind} drawdown breaker: equity ${current_equity:,.2f} "
                    f"is {drawdown:+.1%} vs ${peak:,.2f} peak of last "
                    f"{spec.lookback_sessions} sessions (threshold "
                    f"-{spec.threshold_pct:.0%}); blocks entries through "
                    f"{self._window_label(spec.kind, today)}"
                )
                self._conn.execute(
                    "INSERT OR IGNORE INTO breaker_log "
                    "(kind, date, activated_at, reason, current_equity, peak_equity) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        spec.kind,
                        today.isoformat(),
                        datetime.now(timezone.utc).isoformat(),
                        reason,
                        current_equity,
                        peak,
                    ),
                )
                self._conn.commit()
                logger.warning("DRAWDOWN BREAKER TRIGGERED: %s", reason)
                return reason
        return None

    def _active_reason(self, today: date) -> str | None:
        """A weekly row blocks through its ISO week; a monthly row through its
        calendar month. Returns the most severe (monthly first) stored reason."""
        cutoff = (today - timedelta(days=45)).isoformat()
        rows = self._conn.execute(
            "SELECT kind, date, reason FROM breaker_log WHERE date >= ? "
            "ORDER BY date DESC",
            (cutoff,),
        ).fetchall()
        weekly_reason = None
        for row in rows:
            try:
                row_date = date.fromisoformat(row["date"])
            except ValueError:
                continue
            if row["kind"] == "monthly":
                if (row_date.year, row_date.month) == (today.year, today.month):
                    return row["reason"]
            elif row["kind"] == "weekly" and weekly_reason is None:
                if row_date.isocalendar()[:2] == today.isocalendar()[:2]:
                    weekly_reason = row["reason"]
        return weekly_reason

    def _peak_equity(self, lookback_sessions: int) -> float | None:
        rows = self._conn.execute(
            "SELECT equity FROM equity_snapshots ORDER BY date DESC LIMIT ?",
            (lookback_sessions,),
        ).fetchall()
        if not rows:
            return None
        return max(float(r["equity"]) for r in rows)

    @staticmethod
    def _window_label(kind: str, today: date) -> str:
        if kind == "weekly":
            iso = today.isocalendar()
            return f"ISO week {iso[0]}-W{iso[1]:02d}"
        return today.strftime("month %Y-%m")

    def trigger_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM breaker_log").fetchone()[0]


class _BarStore(Protocol):
    def latest_date(self, symbol: str) -> date | None: ...


@dataclass(frozen=True)
class FreshnessReport:
    stale_symbols: list[str]
    block_reason: str | None   # set when staleness looks systemic


class BarsFreshnessGuard:
    """Refuses to open positions on a frozen daily-bars cache.

    The failure mode is real: the cache sat frozen for two months (2026-04-24
    to 07-02) while signals and retrains silently consumed stale prices.
    ensure_data now runs every cycle but is allowed to fail soft — this guard
    is the backstop that turns "quietly stale" into "loudly blocked".

    Per-symbol staleness (a halted/delisted name) only drops that symbol from
    the entry universe; when the stale FRACTION crosses max_stale_fraction the
    whole entry side is blocked because the ingest itself is broken.
    """

    def __init__(
        self,
        store: _BarStore,
        symbols: list[str],
        max_stale_fraction: float = 0.5,
        tolerance_days: int = 4,
    ):
        self._store = store
        self._symbols = list(symbols)
        self._max_stale_fraction = max_stale_fraction
        # Calendar-day grace below the expected end date: long weekends and a
        # single missed refresh pass; a frozen cache does not.
        self._tolerance = timedelta(days=tolerance_days)

    def check(self, expected_end: date) -> FreshnessReport:
        if not self._symbols:
            return FreshnessReport(stale_symbols=[], block_reason=None)
        cutoff = expected_end - self._tolerance
        stale: list[str] = []
        for sym in self._symbols:
            try:
                latest = self._store.latest_date(sym)
            except Exception as e:
                logger.warning("freshness check failed for %s: %s", sym, e)
                latest = None
            if latest is None or latest < cutoff:
                stale.append(sym)
        fraction = len(stale) / len(self._symbols)
        block_reason = None
        if fraction >= self._max_stale_fraction:
            sample = ", ".join(stale[:5])
            block_reason = (
                f"daily-bars cache stale for {len(stale)}/{len(self._symbols)} "
                f"symbols (e.g. {sample}); expected bars through {expected_end} "
                f"(tolerance {self._tolerance.days}d) — blocking new entries"
            )
        return FreshnessReport(stale_symbols=stale, block_reason=block_reason)


def write_heartbeat(path: Path, market_state: str) -> None:
    """Atomic write so the cron-side reader never sees a torn file. Never
    raises — a full disk must not kill the trading loop."""
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "market_state": market_state,
    }
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
    except OSError as e:
        logger.warning("heartbeat write failed: %s", e)


def read_heartbeat(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
