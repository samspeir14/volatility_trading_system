"""Unit tests for the operational trading guards (risk/trading_guards.py) and
the dead-man-switch decision logic (scripts/heartbeat_check.py)."""
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from risk.kill_switch import DailyKillSwitch
from risk.trading_guards import (
    BarsFreshnessGuard,
    DrawdownBreaker,
    HaltFlag,
    read_heartbeat,
    write_heartbeat,
)
from scripts.heartbeat_check import should_alert


# ---------- HaltFlag ----------

def test_halt_flag():
    with tempfile.TemporaryDirectory() as tmp:
        flag = HaltFlag(Path(tmp) / "HALT")
        assert flag.reason() is None, "no file -> not halted"
        flag.path.write_text("pausing for FOMC\n")
        assert flag.reason() == "pausing for FOMC"
        flag.path.write_text("")
        assert flag.reason() == "HALT file present", "empty file still halts"
        flag.path.unlink()
        assert flag.reason() is None, "deleting the file resumes"
    print("halt_flag: create/read/delete lifecycle verified")


# ---------- DrawdownBreaker ----------

def _seed_snapshots(db_path: Path, equities_by_date: dict[date, float]) -> None:
    """equity_snapshots is owned by DailyKillSwitch — seed through its API so
    the breaker reads exactly what production writes."""
    with DailyKillSwitch(db_path) as ks:
        for d, eq in sorted(equities_by_date.items()):
            ks.get_starting_equity(d, eq)


def test_weekly_breaker_triggers_and_expires():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "risk_state.db"
        # Mon 2026-07-06 .. Fri 2026-07-10 all open at $100k
        week = {date(2026, 7, 6) + timedelta(days=i): 100_000.0 for i in range(5)}
        _seed_snapshots(db, week)

        with DrawdownBreaker(db) as breaker:
            friday = date(2026, 7, 10)
            # -5% intraday: daily kill switch territory, weekly stays quiet
            assert breaker.evaluate_and_maybe_trigger(friday, 95_000.0) is None
            # -9% vs the 5-session peak: weekly trips
            reason = breaker.evaluate_and_maybe_trigger(friday, 91_000.0)
            assert reason is not None and "weekly" in reason, reason
            # Stable reason on subsequent cycles (alert dedupe relies on this)
            again = breaker.evaluate_and_maybe_trigger(friday, 90_500.0)
            assert again == reason, "active breaker must return the STORED reason"

        # Persistence: a fresh instance (process restart) still sees it
        with DrawdownBreaker(db) as breaker2:
            reason3 = breaker2.evaluate_and_maybe_trigger(friday, 99_000.0)
            assert reason3 == reason, "breaker must survive restart"
            # Next ISO week: window expired; equity recovered -> trading resumes
            monday = date(2026, 7, 13)
            assert breaker2.evaluate_and_maybe_trigger(monday, 99_000.0) is None
    print("weekly_breaker: -8% trips, reason stable, survives restart, expires next ISO week")


def test_weekly_breaker_retriggers_if_still_deep_down():
    """After the ISO-week window expires, a still-depressed equity re-triggers
    against the rolling peak — trading resumes only once equity stabilizes."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "risk_state.db"
        week = {date(2026, 7, 6) + timedelta(days=i): 100_000.0 for i in range(5)}
        _seed_snapshots(db, week)
        with DrawdownBreaker(db) as breaker:
            assert breaker.evaluate_and_maybe_trigger(date(2026, 7, 10), 91_000.0)
            # Monday of the next week, equity still at $91k: peak of last 5
            # sessions is still $100k -> re-triggers with a NEW row
            reason = breaker.evaluate_and_maybe_trigger(date(2026, 7, 13), 91_000.0)
            assert reason is not None and "weekly" in reason
            assert breaker.trigger_count() == 2
    print("weekly_breaker: re-triggers while equity sits deep below rolling peak")


def test_monthly_breaker_slow_bleed():
    """A drift the daily and weekly breakers never see: -300/session for 21
    sessions, then a leg down to -12% vs the 21-session peak."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "risk_state.db"
        snaps: dict[date, float] = {}
        d = date(2026, 6, 12)
        eq = 100_000.0
        while len(snaps) < 21:
            if d.weekday() < 5:
                snaps[d] = eq
                eq -= 300.0
            d += timedelta(days=1)
        _seed_snapshots(db, snaps)
        with DrawdownBreaker(db) as breaker:
            today = max(snaps)
            # 5-session peak ≈ $95.2k -> $87.9k is -7.7% (weekly quiet);
            # 21-session peak = $100k -> -12.1% (monthly trips)
            reason = breaker.evaluate_and_maybe_trigger(today, 87_900.0)
            assert reason is not None and "monthly" in reason, reason
            # Active through the calendar month
            assert breaker.evaluate_and_maybe_trigger(date(2026, 7, 31), 99_000.0) == reason
            # August 3rd (next month, recovered): clear
            assert breaker.evaluate_and_maybe_trigger(date(2026, 8, 3), 99_000.0) is None
    print("monthly_breaker: slow bleed trips monthly, blocks through month end")


# ---------- BarsFreshnessGuard ----------

class _StubStore:
    def __init__(self, latest_by_symbol):
        self._latest = latest_by_symbol

    def latest_date(self, symbol):
        return self._latest.get(symbol)


def test_bars_freshness_guard():
    expected_end = date(2026, 7, 10)
    fresh = expected_end
    stale = expected_end - timedelta(days=10)

    # All fresh: nothing stale, no block
    guard = BarsFreshnessGuard(_StubStore({s: fresh for s in "ABCD"}), list("ABCD"))
    report = guard.check(expected_end)
    assert report.stale_symbols == [] and report.block_reason is None

    # One stale name of four: dropped individually, no systemic block
    guard = BarsFreshnessGuard(
        _StubStore({"A": fresh, "B": fresh, "C": fresh, "D": stale}), list("ABCD"),
    )
    report = guard.check(expected_end)
    assert report.stale_symbols == ["D"] and report.block_reason is None

    # Half stale (>= max_stale_fraction 0.5): systemic — block entries
    guard = BarsFreshnessGuard(
        _StubStore({"A": fresh, "B": fresh, "C": stale, "D": None}), list("ABCD"),
    )
    report = guard.check(expected_end)
    assert sorted(report.stale_symbols) == ["C", "D"]
    assert report.block_reason is not None and "blocking new entries" in report.block_reason

    # Weekend tolerance: bars 3 calendar days old are NOT stale
    guard = BarsFreshnessGuard(
        _StubStore({s: expected_end - timedelta(days=3) for s in "ABCD"}), list("ABCD"),
    )
    assert guard.check(expected_end).stale_symbols == []
    print("bars_freshness: per-symbol drop, systemic block at 50%, weekend tolerance")


# ---------- heartbeat + dead-man switch ----------

def test_heartbeat_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "heartbeat.json"
        assert read_heartbeat(path) is None, "missing file reads as None"
        write_heartbeat(path, "open")
        hb = read_heartbeat(path)
        assert hb is not None and hb["market_state"] == "open"
        age = datetime.now(timezone.utc) - datetime.fromisoformat(hb["ts"])
        assert age.total_seconds() < 5
    print("heartbeat: atomic write/read roundtrip verified")


def test_should_alert_thresholds():
    now = datetime(2026, 7, 13, 15, 0, tzinfo=timezone.utc)

    def _hb(minutes_old: int, state: str) -> dict:
        return {"ts": (now - timedelta(minutes=minutes_old)).isoformat(),
                "market_state": state}

    assert should_alert(None, now) is not None, "missing heartbeat alerts"
    assert should_alert({"ts": "garbage"}, now) is not None, "unparsable alerts"
    # Market open: 20-minute limit
    assert should_alert(_hb(5, "open"), now) is None
    assert should_alert(_hb(25, "open"), now) is not None
    # Not open: 75-minute limit (closed-market loop beats at least hourly)
    assert should_alert(_hb(25, "closed"), now) is None
    assert should_alert(_hb(80, "closed"), now) is not None
    assert should_alert(_hb(80, "error"), now) is not None
    print("should_alert: 20min open / 75min otherwise thresholds verified")


def main() -> int:
    test_halt_flag()
    test_weekly_breaker_triggers_and_expires()
    test_weekly_breaker_retriggers_if_still_deep_down()
    test_monthly_breaker_slow_bleed()
    test_bars_freshness_guard()
    test_heartbeat_roundtrip()
    test_should_alert_thresholds()
    print("all trading_guards tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
