"""Tests for the VRP gap store: z-score math, min-obs gate, per-day
averaging, schema migration, and gate pass-rate counts."""
import math
import sqlite3
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from signals.divergence_history import DivergenceHistory


def _seed(history: DivergenceHistory, symbol: str, gaps: list[float],
          start: date = date(2026, 1, 1), dte: int = 10) -> None:
    rows = [
        (start + timedelta(days=i), symbol, dte, 0.25, 0.20, g)
        for i, g in enumerate(gaps)
    ]
    history.log_vrp(rows)


def test_z_matches_numpy():
    rng = np.random.default_rng(0)
    gaps = list(rng.normal(0.1, 0.05, 150))
    with tempfile.TemporaryDirectory() as d:
        h = DivergenceHistory(Path(d) / "h.db")
        _seed(h, "AAPL", gaps)
        g_now = 0.3
        z = h.vrp_z_score("AAPL", g_now, dte=10)
        arr = np.array(gaps)
        expected = (g_now - arr.mean()) / arr.std(ddof=0)
        assert math.isclose(z, expected, rel_tol=1e-9)
        h.close()
    print(f"z_math: {z:.4f} matches numpy")


def test_min_obs_gate():
    with tempfile.TemporaryDirectory() as d:
        h = DivergenceHistory(Path(d) / "h.db")
        _seed(h, "AAPL", [0.1] * 119)
        assert h.vrp_z_score("AAPL", 0.5, dte=10) is None, "119 obs must not emit"
        assert h.vrp_obs_count("AAPL", dte=10) == 119
        _seed(h, "AAPL", [0.1, 0.2], start=date(2026, 6, 1))
        assert h.vrp_obs_count("AAPL", dte=10) == 121
        assert h.vrp_z_score("AAPL", 0.5, dte=10) is not None
        assert h.vrp_z_score("UNKNOWN", 0.5, dte=10) is None
        h.close()
    print("min_obs: silent below 120 distinct days, live at 121")


def test_sigma_zero_returns_none():
    with tempfile.TemporaryDirectory() as d:
        h = DivergenceHistory(Path(d) / "h.db")
        _seed(h, "FLAT", [0.1] * 130)
        assert h.vrp_z_score("FLAT", 0.2, dte=10) is None
        h.close()
    print("sigma_zero: constant history → None")


def test_per_day_averaging_across_tenors():
    """A day with several dte rows contributes ONE observation (their mean),
    so tenor-count never biases mu/sigma."""
    with tempfile.TemporaryDirectory() as d:
        h = DivergenceHistory(Path(d) / "h.db")
        day1 = date(2026, 1, 1)
        day2 = date(2026, 1, 2)
        h.log_vrp([
            (day1, "X", 7, 0.25, 0.20, 0.0),
            (day1, "X", 14, 0.25, 0.20, 0.4),   # same day, different tenor
            (day2, "X", 7, 0.25, 0.20, 1.0),
        ])
        assert h.vrp_obs_count("X", dte=10) == 2
        z = h.vrp_z_score("X", 0.6, dte=10, min_obs=2)
        # daily gaps: mean(0.0, 0.4)=0.2 and 1.0 → mu=0.6, sigma=0.4
        assert math.isclose(z, 0.0, abs_tol=1e-12)
        h.close()
    print("per_day_avg: two tenors on one day count once")


def test_dte_band_separation():
    """A candidate is scored only against gap history from its own tenor
    band. One band spans the whole tradeable entry window (DTE 1-14); the
    catch-all above it exists purely so an out-of-window dte can never
    borrow short-tenor history."""
    from signals.divergence_history import vrp_dte_band

    assert vrp_dte_band(1) == vrp_dte_band(7) == vrp_dte_band(14)
    assert vrp_dte_band(14) != vrp_dte_band(15)
    with tempfile.TemporaryDirectory() as d:
        h = DivergenceHistory(Path(d) / "h.db")
        _seed(h, "X", [0.1, 0.3] * 65, dte=10)      # 130 short-band days
        assert h.vrp_z_score("X", 0.5, dte=12) is not None, "same band scores"
        assert h.vrp_z_score("X", 0.5, dte=40) is None, (
            "long-band candidate must not borrow short-band history"
        )
        assert h.vrp_obs_count("X", dte=40) == 0
        h.close()
    print("dte_bands: short-band history never calibrates a long-band candidate")


def test_migration_adds_columns_to_old_db():
    """A pre-h1 database (no vrp_z/blocked_by, no vrp_log) opens cleanly,
    gains the new columns, and keeps its old rows readable."""
    old_create = """
    CREATE TABLE divergence_log (
        scan_date TEXT NOT NULL, symbol TEXT NOT NULL, expiration TEXT NOT NULL,
        horizon_lower INTEGER NOT NULL, horizon_upper INTEGER NOT NULL,
        weight_lower REAL NOT NULL, predicted_iv_equivalent REAL NOT NULL,
        atm_iv REAL NOT NULL, divergence REAL NOT NULL,
        underlying_price REAL NOT NULL,
        PRIMARY KEY (scan_date, symbol, expiration, horizon_lower, horizon_upper)
    );
    """
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "old.db"
        conn = sqlite3.connect(str(db))
        conn.execute(old_create)
        conn.execute(
            "INSERT INTO divergence_log VALUES "
            "('2026-06-01', 'AAPL', '2026-06-12', 5, 10, 0.4, 0.21, 0.25, -0.04, 190.0)"
        )
        conn.commit()
        conn.close()

        h = DivergenceHistory(db)
        cols = {r[1] for r in h._conn.execute("PRAGMA table_info(divergence_log)")}
        assert {"vrp_z", "blocked_by"} <= cols
        assert h.row_count() == 1
        # vrp_log exists and works
        _seed(h, "AAPL", [0.1, 0.2])
        assert h.vrp_obs_count("AAPL", dte=10) == 2
        h.close()
    print("migration: old DB gains vrp_z/blocked_by + vrp_log, rows intact")


def test_gate_counts_today():
    from signals.signal_generator import TradeLeg, TradeSignal

    def _sig(sym, actionable, blocked_by):
        return TradeSignal(
            symbol=sym, expiration=date(2026, 6, 12), dte=11,
            horizon_lower=1, horizon_upper=1, weight_lower=1.0,
            direction="SELL", underlying_price=100.0, atm_iv=0.25,
            predicted_iv_equivalent=0.20, divergence=-0.05,
            cross_sectional_z=0.0, time_series_z=None, liquidity_score=1.0,
            legs=[TradeLeg(100, "call", "sell", 1, "X")] if actionable else [],
            is_actionable=actionable, vrp_z=1.7, blocked_by=blocked_by,
        )

    today = date(2026, 8, 10)
    with tempfile.TemporaryDirectory() as d:
        h = DivergenceHistory(Path(d) / "h.db")
        h.log_signals([
            _sig("A", True, None),
            _sig("B", False, "vrp_z"),
            _sig("C", False, "vrp_z"),
            _sig("D", False, "cost_gate"),
            _sig("E", False, None),  # demoted without label → bucketed as "legs"
        ], today)
        counts = h.gate_counts_today(today)
        assert counts["candidates"] == 5
        assert counts["approved"] == 1
        assert counts["vrp_z"] == 2
        assert counts["cost_gate"] == 1
        assert counts["legs"] == 1
        assert h.gate_counts_today(today + timedelta(days=1)) == {
            "candidates": 0, "approved": 0,
        }
        h.close()
    print(f"gate_counts: {counts}")


def main() -> int:
    test_z_matches_numpy()
    test_min_obs_gate()
    test_sigma_zero_returns_none()
    test_per_day_averaging_across_tenors()
    test_dte_band_separation()
    test_migration_adds_columns_to_old_db()
    test_gate_counts_today()
    print("all vrp_gate tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
