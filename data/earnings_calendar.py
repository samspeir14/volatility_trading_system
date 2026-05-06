"""Earnings calendar fetched from Finnhub, cached in SQLite.

The bot uses this to demote option signals whose expiration straddles an
earnings event — the model can't reason about binary event tail risk, so we
filter rather than feature-engineer. Tradier's /markets/fundamentals/calendars
endpoint requires a paid add-on this account doesn't carry, so Finnhub's free
tier (60 req/min) is the source.

Fail-open by design: missing API key, HTTP error, malformed response, or empty
cache all behave the same — the lookup returns None and the caller treats that
as "no earnings known" so trading continues. A flaky earnings API must never
halt the whole bot.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)


FINNHUB_BASE = "https://finnhub.io/api/v1"
DEFAULT_LOOKAHEAD_DAYS = 60

CREATE_EARNINGS_SQL = """
CREATE TABLE IF NOT EXISTS earnings (
    symbol TEXT NOT NULL,
    earnings_date TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (symbol, earnings_date)
);
"""

CREATE_EARNINGS_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_earnings_symbol_date "
    "ON earnings(symbol, earnings_date);"
)

CREATE_META_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class EarningsCalendar:
    def __init__(self, db_path: Path, api_key: str | None):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(CREATE_EARNINGS_SQL)
        self._conn.execute(CREATE_EARNINGS_INDEX_SQL)
        self._conn.execute(CREATE_META_SQL)
        self._conn.commit()
        self._api_key = api_key

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @property
    def api_key_configured(self) -> bool:
        return bool(self._api_key)

    def last_refresh_date(self) -> date | None:
        cur = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'last_refresh_date'"
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            return date.fromisoformat(row[0])
        except ValueError:
            return None

    def has_earnings_in_window(
        self, symbol: str, start: date, end: date
    ) -> bool | None:
        """Return True if there's a known earnings date d with start <= d <= end,
        False if no earnings within the window (and the cache has any data for
        this symbol or has been refreshed today), None if we have no information
        and the caller should fail open."""
        if start > end:
            return False
        # If no refresh has ever happened, we have no information.
        if self.last_refresh_date() is None:
            return None
        cur = self._conn.execute(
            "SELECT 1 FROM earnings WHERE symbol = ? "
            "AND earnings_date >= ? AND earnings_date <= ? LIMIT 1",
            (symbol, start.isoformat(), end.isoformat()),
        )
        if cur.fetchone() is not None:
            return True
        return False

    def next_earnings_on_or_after(self, symbol: str, start: date) -> date | None:
        """Earliest known earnings date for symbol on or after `start`. Used for
        diagnostic logging; returns None when there is no record."""
        cur = self._conn.execute(
            "SELECT earnings_date FROM earnings "
            "WHERE symbol = ? AND earnings_date >= ? "
            "ORDER BY earnings_date ASC LIMIT 1",
            (symbol, start.isoformat()),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            return date.fromisoformat(row[0])
        except ValueError:
            return None

    async def refresh_if_stale(
        self,
        today: date,
        symbols: list[str] | None = None,
        lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
        force: bool = False,
    ) -> bool:
        """Once-a-day refresh. Returns True if a refresh ran (whether or not it
        succeeded against the network), False if it was skipped because the
        cache is already fresh or no API key is configured.

        Fail-open: any network/parse error logs a WARNING and leaves the
        existing cache untouched. A missing API key skips silently — the
        caller is expected to log a single startup WARNING about the missing
        key, so per-cycle warnings here would just spam the journal."""
        if not force and self.last_refresh_date() == today:
            return False

        if not self._api_key:
            return False

        end = today.fromordinal(today.toordinal() + lookahead_days)
        params = {
            "from": today.isoformat(),
            "to": end.isoformat(),
            "token": self._api_key,
        }
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15.0),
                headers={"Accept": "application/json"},
            ) as session:
                async with session.get(
                    f"{FINNHUB_BASE}/calendar/earnings", params=params
                ) as resp:
                    if resp.status != 200:
                        body = (await resp.text())[:200]
                        logger.warning(
                            "Finnhub earnings calendar returned %d: %s — failing open",
                            resp.status, body,
                        )
                        return True
                    payload = await resp.json()
        except Exception as e:
            logger.warning(
                "Finnhub earnings calendar fetch failed (%s) — failing open", e
            )
            return True

        rows = self._parse_payload(payload, symbols)
        self._replace_cache(rows, fetched_at=datetime.now(timezone.utc), today=today)
        logger.info(
            "earnings calendar refreshed: %d entries through %s "
            "(filtered to %s symbols)",
            len(rows), end.isoformat(),
            len(symbols) if symbols else "all",
        )
        return True

    @staticmethod
    def _parse_payload(payload: dict, symbols: list[str] | None) -> list[tuple[str, str]]:
        """Pull (symbol, date) pairs from a Finnhub earnings calendar response.
        Filters to `symbols` when provided. Skips rows that are missing either
        field rather than failing the whole refresh."""
        raw = payload.get("earningsCalendar") or []
        if not isinstance(raw, list):
            return []
        wanted = set(symbols) if symbols else None
        out: list[tuple[str, str]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            sym = entry.get("symbol")
            d = entry.get("date")
            if not sym or not d:
                continue
            if wanted is not None and sym not in wanted:
                continue
            out.append((sym, d))
        return out

    def _replace_cache(
        self, rows: list[tuple[str, str]], fetched_at: datetime, today: date
    ) -> None:
        ts = fetched_at.isoformat()
        with self._conn:
            self._conn.execute("DELETE FROM earnings")
            self._conn.executemany(
                "INSERT INTO earnings (symbol, earnings_date, fetched_at) "
                "VALUES (?, ?, ?)",
                [(s, d, ts) for s, d in rows],
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES "
                "('last_refresh_date', ?)",
                (today.isoformat(),),
            )

    # Test helper: seed the cache directly without going through Finnhub.
    def _seed_for_testing(
        self, rows: list[tuple[str, date]], today: date
    ) -> None:
        self._replace_cache(
            [(sym, d.isoformat()) for sym, d in rows],
            fetched_at=datetime.now(timezone.utc),
            today=today,
        )
