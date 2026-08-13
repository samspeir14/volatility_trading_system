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
    vrp_z REAL,
    blocked_by TEXT,
    PRIMARY KEY (scan_date, symbol, expiration, horizon_lower, horizon_upper)
);
"""

CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_div_symbol_horizons "
    "ON divergence_log(symbol, horizon_lower, horizon_upper, scan_date DESC);"
)

# Per-ticker variance-risk-premium gap history: g = log(IV) - log(realized
# vol over a window matched to the option's tenor). One row per evaluated
# (scan_date, symbol, dte) candidate — logged for EVERY candidate regardless
# of gating so history accrues from day one.
CREATE_VRP_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS vrp_log (
    scan_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    dte INTEGER NOT NULL,
    atm_iv REAL NOT NULL,
    realized_vol REAL NOT NULL,
    g REAL NOT NULL,
    PRIMARY KEY (scan_date, symbol, dte)
);
"""

CREATE_VRP_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_vrp_symbol_date "
    "ON vrp_log(symbol, scan_date DESC);"
)

DEFAULT_LOOKBACK = 60
MIN_HISTORY = 20

VRP_LOOKBACK_DAYS = 252
VRP_MIN_OBS = 120

# The VRP gap is tenor-dependent (fattest at short DTE), so z-scores compare a
# candidate only against gap history from the SAME tenor band. One band covers
# the whole tradeable entry window (DTE 1-14) — finer sub-bands would split
# the gap history and silence the shortest tenors for months at VRP_MIN_OBS
# (rows below ~7 DTE only started accruing at the 2026-08 pivot; study
# offline on vrp_log before any live re-band). The catch-all keeps any
# out-of-window dte from ever borrowing short-tenor history.
VRP_DTE_BANDS: tuple[tuple[int, int], ...] = ((0, 14), (15, 10_000))


def vrp_dte_band(dte: int) -> tuple[int, int]:
    for lo, hi in VRP_DTE_BANDS:
        if lo <= dte <= hi:
            return (lo, hi)
    return VRP_DTE_BANDS[-1]


class DivergenceHistory:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(CREATE_TABLE_SQL)
        self._conn.execute(CREATE_INDEX_SQL)
        self._conn.execute(CREATE_VRP_TABLE_SQL)
        self._conn.execute(CREATE_VRP_INDEX_SQL)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """In-place migration for databases created before the h=1 gates:
        add vrp_z / blocked_by to divergence_log if missing."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(divergence_log)")
        }
        if "vrp_z" not in cols:
            self._conn.execute(
                "ALTER TABLE divergence_log ADD COLUMN vrp_z REAL"
            )
        if "blocked_by" not in cols:
            self._conn.execute(
                "ALTER TABLE divergence_log ADD COLUMN blocked_by TEXT"
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def log_signals(self, signals, scan_date: date) -> int:
        """Persist a batch of TradeSignal objects. Idempotent on
        (scan_date, symbol, expiration, horizon_lower, horizon_upper).
        blocked_by is '' for actionable signals, the gate name otherwise."""
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
                getattr(s, "vrp_z", None),
                ("" if s.is_actionable else (getattr(s, "blocked_by", None) or "legs")),
            )
            for s in signals
        ]
        if not rows:
            return 0
        self._conn.executemany(
            "INSERT OR REPLACE INTO divergence_log "
            "(scan_date, symbol, expiration, horizon_lower, horizon_upper, weight_lower, "
            " predicted_iv_equivalent, atm_iv, divergence, underlying_price, "
            " vrp_z, blocked_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    # ------------------------------------------------------------------ VRP

    def log_vrp(self, rows: list[tuple[date, str, int, float, float, float]]) -> int:
        """Persist (scan_date, symbol, dte, atm_iv, realized_vol, g) rows.
        Idempotent on (scan_date, symbol, dte)."""
        if not rows:
            return 0
        self._conn.executemany(
            "INSERT OR REPLACE INTO vrp_log "
            "(scan_date, symbol, dte, atm_iv, realized_vol, g) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (d.isoformat(), sym, dte, iv, rv, g)
                for d, sym, dte, iv, rv, g in rows
            ],
        )
        self._conn.commit()
        return len(rows)

    def _daily_gap_series(
        self, symbol: str, dte: int, lookback_days: int
    ) -> list[float]:
        """One g per (symbol, scan_date) within the candidate's DTE band over
        the most recent `lookback_days` distinct scan dates — averaged across
        that day's same-band dte rows so a day with many tenors doesn't
        outweigh a day with one."""
        lo, hi = vrp_dte_band(dte)
        cur = self._conn.execute(
            "SELECT AVG(g) FROM vrp_log WHERE symbol = ? "
            "AND dte >= ? AND dte <= ? "
            "GROUP BY scan_date ORDER BY scan_date DESC LIMIT ?",
            (symbol, lo, hi, lookback_days),
        )
        return [r[0] for r in cur.fetchall()]

    def vrp_obs_count(
        self, symbol: str, dte: int,
        lookback_days: int = VRP_LOOKBACK_DAYS,
    ) -> int:
        """Distinct scan dates with a same-band gap observation for `symbol`
        within the rolling lookback."""
        return len(self._daily_gap_series(symbol, dte, lookback_days))

    def vrp_z_score(
        self,
        symbol: str,
        g_now: float,
        dte: int,
        lookback_days: int = VRP_LOOKBACK_DAYS,
        min_obs: int = VRP_MIN_OBS,
    ) -> float | None:
        """z = (g_now - mu_g) / sigma_g over the rolling per-day gap history
        restricted to the candidate's DTE band (see VRP_DTE_BANDS). Returns
        None (→ caller emits no signal for this candidate) when fewer than
        `min_obs` distinct same-band days exist or sigma is zero."""
        gaps = self._daily_gap_series(symbol, dte, lookback_days)
        if len(gaps) < min_obs:
            return None
        n = len(gaps)
        mu = sum(gaps) / n
        var = sum((x - mu) ** 2 for x in gaps) / n
        sigma = math.sqrt(var)
        # 1e-12 (not == 0): a numerically-constant history leaves float dust
        # in sigma, and dividing by it would fabricate a huge z.
        if sigma < 1e-12:
            return None
        return (g_now - mu) / sigma

    # ----------------------------------------------------------- reporting

    def gate_counts_today(self, scan_date: date) -> dict[str, int]:
        """Signal-level gate tallies for the daily report: total candidates,
        per-gate blocked counts (keyed by blocked_by), and 'approved' (signals
        that cleared every signal-level gate). Prevents a silent zero-signal
        deadlock from going unnoticed."""
        cur = self._conn.execute(
            "SELECT COALESCE(blocked_by, ''), COUNT(*) FROM divergence_log "
            "WHERE scan_date = ? GROUP BY COALESCE(blocked_by, '')",
            (scan_date.isoformat(),),
        )
        counts: dict[str, int] = {}
        total = 0
        approved = 0
        for blocked_by, n in cur.fetchall():
            total += n
            if blocked_by == "":
                approved += n
            else:
                counts[blocked_by] = n
        out = {"candidates": total, **dict(sorted(counts.items())), "approved": approved}
        return out

    def row_count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM divergence_log")
        return cur.fetchone()[0]
