import sqlite3
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from execution import OrderLog
from signals.signal_generator import TradeLeg, TradeSignal


def _fake_signal(symbol: str = "AAPL", strikes=(100.0,)) -> TradeSignal:
    legs = [
        TradeLeg(strike=k, option_type="call", side="buy", quantity=1, contract_symbol=f"{symbol}_C{int(k)}")
        for k in strikes
    ]
    legs.extend([
        TradeLeg(strike=k, option_type="put", side="buy", quantity=1, contract_symbol=f"{symbol}_P{int(k)}")
        for k in strikes
    ])
    return TradeSignal(
        symbol=symbol, expiration=date(2026, 5, 15), dte=18,
        horizon_lower=10, horizon_upper=21, weight_lower=0.27,
        direction="BUY", underlying_price=100.0, atm_iv=0.30,
        predicted_iv_equivalent=0.40, divergence=0.10,
        cross_sectional_z=2.0, time_series_z=None,
        liquidity_score=12345.0, legs=legs, is_actionable=True,
    )


def test_close_attempts_migration_adds_last_priced_at():
    """An order_log.db from before the in-place reprice (no last_priced_at
    column) must open cleanly and gain the column."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.db"
        conn = sqlite3.connect(str(path))
        conn.execute(
            "CREATE TABLE close_attempts (closing_order_id INTEGER PRIMARY KEY, "
            "opening_order_id INTEGER NOT NULL, submitted_at TEXT NOT NULL, "
            "exit_trigger TEXT NOT NULL, order_type TEXT NOT NULL, "
            "submitted_price REAL NOT NULL, status TEXT NOT NULL, "
            "terminal_at TEXT, fill_price REAL)"
        )
        conn.execute(
            "INSERT INTO close_attempts VALUES (1, 2, '2026-09-03T13:33:20+00:00', "
            "'assignment_risk', 'debit', 5.15, 'pending', NULL, NULL)"
        )
        conn.commit(); conn.close()

        log = OrderLog(path)
        cols = {r[1] for r in log._conn.execute("PRAGMA table_info(close_attempts)")}
        assert {"arrival_mid", "last_priced_at"} <= cols
        row = log.pending_close_attempt(2)
        assert row["closing_order_id"] == 1 and row["last_priced_at"] is None
        log.close()
    print("order_log: close_attempts migration adds last_priced_at ✓")


def test_record_and_retrieve():
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        sig = _fake_signal()
        log.record_submission(
            signal=sig, fingerprint="abc123", structure="straddle",
            submitted_price=2.50, order_id=1001,
            submitted_at=datetime.now(timezone.utc),
        )
        assert log.submitted_count() == 1
        rows = log.open_orders_by_symbol("AAPL")
        assert len(rows) == 1
        assert rows[0]["tradier_order_id"] == 1001
        assert rows[0]["structure"] == "straddle"
        log.close()
    print("record_and_retrieve: OK")


def test_dedup_within_window():
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        sig = _fake_signal()
        # Empty log → no recent order
        assert log.has_recent_open_order("abc123") is False
        # Record one
        log.record_submission(
            signal=sig, fingerprint="abc123", structure="straddle",
            submitted_price=2.50, order_id=2001,
            submitted_at=datetime.now(timezone.utc),
        )
        assert log.has_recent_open_order("abc123") is True
        log.close()
    print("dedup_within_window: OK")


def test_dedup_window_expires():
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        sig = _fake_signal()
        old = datetime.now(timezone.utc) - timedelta(hours=25)
        log.record_submission(
            signal=sig, fingerprint="old123", structure="straddle",
            submitted_price=2.50, order_id=3001, submitted_at=old,
        )
        # 25h-old order is outside the 24h window
        assert log.has_recent_open_order("old123", hours=24) is False
        # But shows up with a wider window
        assert log.has_recent_open_order("old123", hours=48) is True
        log.close()
    print("dedup_window_expires: OK")


def test_terminal_failed_does_not_dedup():
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        sig = _fake_signal()
        log.record_submission(
            signal=sig, fingerprint="rej123", structure="straddle",
            submitted_price=2.50, order_id=4001,
            submitted_at=datetime.now(timezone.utc),
        )
        log.update_terminal_state(
            order_id=4001, status="rejected", fill_price=None,
            filled_at=None, error="not enough buying power",
        )
        # A rejected order should NOT block a re-attempt
        assert log.has_recent_open_order("rej123") is False
        log.close()
    print("terminal_failed_does_not_dedup: OK")


def test_failed_submissions_separate_table():
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        sig = _fake_signal()
        log.record_failure(sig, "fail123", "premium too high", datetime.now(timezone.utc))
        assert log.failed_count() == 1
        assert log.submitted_count() == 0
        # Pre-flight failures don't trigger dedup either (unless we want them to)
        assert log.has_recent_open_order("fail123") is False
        log.close()
    print("failed_submissions_separate_table: OK")


def test_persistence_across_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.db"
        log1 = OrderLog(path)
        sig = _fake_signal()
        log1.record_submission(
            signal=sig, fingerprint="abc", structure="straddle",
            submitted_price=2.50, order_id=5001,
            submitted_at=datetime.now(timezone.utc),
        )
        log1.close()

        log2 = OrderLog(path)
        assert log2.submitted_count() == 1
        assert log2.has_recent_open_order("abc") is True
        log2.close()
    print("persistence_across_reopen: OK")


def main() -> int:
    test_close_attempts_migration_adds_last_priced_at()
    test_record_and_retrieve()
    test_dedup_within_window()
    test_dedup_window_expires()
    test_terminal_failed_does_not_dedup()
    test_failed_submissions_separate_table()
    test_persistence_across_reopen()
    print("all order_log tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
