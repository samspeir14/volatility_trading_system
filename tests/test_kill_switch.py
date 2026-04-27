import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

from risk import DailyKillSwitch


def test_empty_state():
    with tempfile.TemporaryDirectory() as tmp:
        ks = DailyKillSwitch(Path(tmp) / "ks.db")
        assert ks.is_active(date(2026, 4, 27)) is False
        assert ks.trigger_count() == 0
        ks.close()
    print("empty_state: is_active=False, trigger_count=0")


def test_trigger_then_active():
    with tempfile.TemporaryDirectory() as tmp:
        ks = DailyKillSwitch(Path(tmp) / "ks.db")
        today = date(2026, 4, 27)
        ks.trigger(today, reason="test", realized_pnl=-3000.0, starting_equity=100000.0)
        assert ks.is_active(today) is True
        assert ks.trigger_count() == 1
        ks.close()
    print("trigger_then_active: triggers and reports active")


def test_resets_at_next_day():
    with tempfile.TemporaryDirectory() as tmp:
        ks = DailyKillSwitch(Path(tmp) / "ks.db")
        today = date(2026, 4, 27)
        tomorrow = today + timedelta(days=1)
        ks.trigger(today, reason="test", realized_pnl=-3000.0, starting_equity=100000.0)
        assert ks.is_active(today) is True
        assert ks.is_active(tomorrow) is False  # next day clean
        ks.close()
    print("resets_at_next_day: today active, tomorrow clean")


def test_trigger_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        ks = DailyKillSwitch(Path(tmp) / "ks.db")
        today = date(2026, 4, 27)
        ks.trigger(today, reason="first", realized_pnl=-3000, starting_equity=100000)
        ks.trigger(today, reason="second", realized_pnl=-4000, starting_equity=100000)
        assert ks.trigger_count() == 1
        ks.close()
    print("trigger_idempotent: second call same day is no-op")


def test_evaluate_no_trigger():
    with tempfile.TemporaryDirectory() as tmp:
        ks = DailyKillSwitch(Path(tmp) / "ks.db")
        today = date(2026, 4, 27)
        # P&L = -2% of $100k = -$2000, threshold = -3% = -$3000 → no trigger
        active = ks.evaluate_and_maybe_trigger(
            today=today, total_pnl_today=-2000.0,
            starting_equity=100000.0, threshold_pct=-0.03,
        )
        assert active is False
        assert ks.is_active(today) is False
        ks.close()
    print("evaluate_no_trigger: -2% < -3% threshold, no trigger")


def test_evaluate_fires_trigger():
    with tempfile.TemporaryDirectory() as tmp:
        ks = DailyKillSwitch(Path(tmp) / "ks.db")
        today = date(2026, 4, 27)
        # P&L = -4% of $100k = -$4000, threshold = -3% = -$3000 → trigger
        active = ks.evaluate_and_maybe_trigger(
            today=today, total_pnl_today=-4000.0,
            starting_equity=100000.0, threshold_pct=-0.03,
        )
        assert active is True
        assert ks.is_active(today) is True
        ks.close()
    print("evaluate_fires_trigger: -4% > -3% threshold (deeper drawdown), fires")


def test_get_starting_equity_first_call_records():
    with tempfile.TemporaryDirectory() as tmp:
        ks = DailyKillSwitch(Path(tmp) / "ks.db")
        today = date(2026, 4, 27)
        # First call records the equity
        eq = ks.get_starting_equity(today=today, current_equity=100000.0)
        assert eq == 100000.0
        # Second call returns same value even if current_equity differs
        eq2 = ks.get_starting_equity(today=today, current_equity=99000.0)
        assert eq2 == 100000.0
        assert ks.equity_snapshot_count() == 1
        ks.close()
    print("starting_equity: first call records, subsequent calls return snapshot")


def test_starting_equity_per_day():
    with tempfile.TemporaryDirectory() as tmp:
        ks = DailyKillSwitch(Path(tmp) / "ks.db")
        d1 = date(2026, 4, 27)
        d2 = date(2026, 4, 28)
        eq1 = ks.get_starting_equity(today=d1, current_equity=100000.0)
        eq2 = ks.get_starting_equity(today=d2, current_equity=99000.0)
        assert eq1 == 100000.0
        assert eq2 == 99000.0
        assert ks.equity_snapshot_count() == 2
        ks.close()
    print("starting_equity_per_day: each day independently snapshotted")


def test_persistence_across_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ks.db"
        ks1 = DailyKillSwitch(path)
        today = date(2026, 4, 27)
        ks1.trigger(today, reason="persist", realized_pnl=-5000, starting_equity=100000)
        ks1.get_starting_equity(today, current_equity=100000.0)
        ks1.close()

        ks2 = DailyKillSwitch(path)
        assert ks2.is_active(today) is True
        assert ks2.trigger_count() == 1
        assert ks2.equity_snapshot_count() == 1
        ks2.close()
    print("persistence: kill switch + equity snapshot survive close/reopen")


def main() -> int:
    test_empty_state()
    test_trigger_then_active()
    test_resets_at_next_day()
    test_trigger_idempotent()
    test_evaluate_no_trigger()
    test_evaluate_fires_trigger()
    test_get_starting_equity_first_call_records()
    test_starting_equity_per_day()
    test_persistence_across_reopen()
    print("all kill_switch tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
