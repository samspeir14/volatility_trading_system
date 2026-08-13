"""Unit tests for the EarningsCalendar cache + the SignalGenerator earnings filter.

Covers:
- Cache miss before any refresh → has_earnings_in_window returns None (fail open).
- Cache hit after seed → returns True/False per stored dates.
- refresh_if_stale skips when last_refresh_date == today.
- refresh_if_stale silently skips (no per-cycle WARNING) when API key is missing.
- Entry gate window is the position's life [today, expiration]:
  - a report AFTER expiry does not block a short-DTE entry (the old flat
    7-day buffer over-blocked these);
  - a report inside the life blocks, however far out (the old buffer
    under-blocked 8-14 DTE entries);
  - boundary: report == expiration blocks (that expiry's IV is all event
    premium), report == expiration+1 passes.
- SignalGenerator does NOT demote when no earnings data is available (fail open).
- SignalGenerator does NOT demote when the filter is disabled via constructor flag.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

from data.earnings_calendar import EarningsCalendar
from signals import DivergenceHistory, SignalGenerator
from tests.test_signal_generator_h1 import (
    _FixedH1,
    _GK_SERIES,
    _feature_row,
    _scan,
    _seed_history,
)

TODAY = date(2026, 6, 1)  # matches _scan's fetched_at


def _generate(
    *,
    earnings: EarningsCalendar | None,
    expiration: date,
    enabled: bool = True,
):
    """One RICH (iv=0.30 → SELL) candidate for NVDA at `expiration`, run
    through the full h=1 gate ladder with the given earnings calendar."""
    with tempfile.TemporaryDirectory() as d:
        history = DivergenceHistory(Path(d) / "h.db")
        dte = (expiration - TODAY).days
        _seed_history(history, ["NVDA"], dtes=(dte,))
        gen = SignalGenerator(
            h1_predictor=_FixedH1(0.0),
            history_store=history,
            earnings_calendar=earnings,
            earnings_filter_enabled=enabled,
        )
        actionable, all_signals = gen.generate(
            _scan([("NVDA", 0.30)], expirations=[expiration]),
            feature_rows={"NVDA": _feature_row("NVDA")},
            daily_gk_vol_by_symbol={"NVDA": _GK_SERIES},
        )
        history.close()
    sig = next(s for s in all_signals if s.symbol == "NVDA")
    return actionable, sig


# ---------- EarningsCalendar cache tests ----------

def test_cache_miss_returns_none_until_refresh():
    """Before any refresh has run, lookups must return None so the caller
    fails open. This is the difference between 'no earnings on record' and
    'we never had a chance to look'."""
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key=None)
        result = cal.has_earnings_in_window("NVDA", date(2026, 5, 1), date(2026, 5, 30))
        assert result is None, f"expected None for never-refreshed cache, got {result!r}"
        cal.close()
    print("cache_miss: returns None when never refreshed (caller fails open)")


def test_seeded_cache_returns_true_for_in_window():
    today = date(2026, 5, 6)
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key=None)
        cal._seed_for_testing(
            [("NVDA", date(2026, 5, 22)), ("MSFT", date(2026, 7, 30))],
            today=today,
        )
        # NVDA earnings at 5/22, expiration 5/30 → within window
        assert cal.has_earnings_in_window("NVDA", today, date(2026, 5, 30)) is True
        # NVDA earnings at 5/22, expiration 5/15 → out of window (earnings after exp)
        assert cal.has_earnings_in_window("NVDA", today, date(2026, 5, 15)) is False
        # AAPL has no record but cache was refreshed → False (clean window)
        assert cal.has_earnings_in_window("AAPL", today, date(2026, 5, 30)) is False
        cal.close()
    print("seeded_cache: returns True/False correctly per stored dates")


def test_refresh_skips_when_already_refreshed_today():
    today = date(2026, 5, 6)
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key="fake_key")
        cal._seed_for_testing([("NVDA", date(2026, 5, 22))], today=today)
        # No HTTP mock needed: refresh should bail before hitting the network
        ran = asyncio.run(cal.refresh_if_stale(today=today))
        assert ran is False, "refresh should skip when last_refresh_date == today"
        cal.close()
    print("refresh_skip: no-op when cache is already fresh for today")


def test_refresh_silently_skips_when_api_key_missing():
    """Missing FINNHUB_API_KEY: refresh_if_stale must return False (skipped),
    leave the cache untouched, and not emit a WARNING per call. The single
    startup warning in build_main_loop is the operator's signal — repeating
    it every 5-minute cycle would just bury other logs."""
    today = date(2026, 5, 6)
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key=None)
        with mock.patch.object(logging.getLogger("data.earnings_calendar"), "warning") as warn:
            ran = asyncio.run(cal.refresh_if_stale(today=today))
        assert ran is False, "refresh should skip silently when no API key"
        assert not warn.called, "no per-call warning when key is missing"
        # Cache still empty afterwards → lookups still return None (fail open)
        assert cal.has_earnings_in_window("NVDA", today, date(2026, 5, 30)) is None
        cal.close()
    print("refresh_skip_no_key: missing API key skips silently, lookups fail open")


# ---------- SignalGenerator filter tests (life-of-position window) ----------

def test_short_dte_entry_passes_when_report_lands_after_expiry():
    """The over-block regression: a 2-DTE entry expiring BEFORE the report
    must trade. The old flat 7-day buffer blocked every short-DTE entry for
    a week around each report — exactly the tenors the h=1 model exists to
    trade."""
    expiration = TODAY + timedelta(days=2)          # 6/3
    report = expiration + timedelta(days=2)         # 6/5 — after expiry
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key=None)
        cal._seed_for_testing([("NVDA", report)], today=TODAY)
        actionable, sig = _generate(earnings=cal, expiration=expiration)
        cal.close()
    assert sig.blocked_by != "earnings", sig.diagnostic_notes
    assert sig.is_actionable, sig.diagnostic_notes
    assert [s.symbol for s in actionable] == ["NVDA"]
    print("life_of_position: 2-DTE entry with report 2d after expiry → trades")


def test_report_inside_position_life_blocks():
    """A report anywhere inside [today, expiration] blocks, however far out.
    The old 7-day buffer passed a 14-DTE entry whose report landed on day
    8-14 and left the fail-closed EXIT to clean it up — in a 1-14 DTE book
    the entry gate itself must refuse the event trade."""
    expiration = TODAY + timedelta(days=14)         # 6/15
    report = TODAY + timedelta(days=12)             # 6/13 — inside the life
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key=None)
        cal._seed_for_testing([("NVDA", report)], today=TODAY)
        actionable, sig = _generate(earnings=cal, expiration=expiration)
        cal.close()
    assert sig.blocked_by == "earnings", (sig.blocked_by, sig.diagnostic_notes)
    assert "earnings_within_position_life" in sig.diagnostic_notes
    assert report.isoformat() in sig.diagnostic_notes
    assert actionable == []
    print("life_of_position: report on day 12 of a 14-DTE entry → blocked")


def test_report_on_expiration_day_boundary():
    """Inclusive upper bound: report == expiration blocks (that expiry's IV
    is all event premium); report == expiration + 1 passes (the position no
    longer exists when the report hits)."""
    expiration = TODAY + timedelta(days=10)         # 6/11
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key=None)
        cal._seed_for_testing([("NVDA", expiration)], today=TODAY)
        _, sig_on = _generate(earnings=cal, expiration=expiration)
        cal.close()
    assert sig_on.blocked_by == "earnings", sig_on.diagnostic_notes

    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key=None)
        cal._seed_for_testing(
            [("NVDA", expiration + timedelta(days=1))], today=TODAY,
        )
        _, sig_after = _generate(earnings=cal, expiration=expiration)
        cal.close()
    assert sig_after.blocked_by != "earnings", sig_after.diagnostic_notes
    assert sig_after.is_actionable
    print("boundary: report==expiration blocks, report==expiration+1 passes")


def test_signal_passes_when_no_earnings_data_available():
    """Fail-open: if the cache has no record (cache never refreshed), the
    filter must allow the trade rather than block all activity."""
    expiration = TODAY + timedelta(days=10)
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key=None)
        # Do NOT seed — cache has never been refreshed
        _, sig = _generate(earnings=cal, expiration=expiration)
        cal.close()
    assert sig.blocked_by != "earnings", sig.diagnostic_notes
    print("filter_fail_open: empty cache → trade allowed (fail open)")


def test_filter_disabled_via_flag_skips_lookup():
    """Even when the calendar has data, earnings_filter_enabled=False must
    bypass the check. Useful for debugging without flushing the cache."""
    expiration = TODAY + timedelta(days=10)
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key=None)
        cal._seed_for_testing(
            [("NVDA", TODAY + timedelta(days=5))], today=TODAY,
        )
        _, sig = _generate(earnings=cal, expiration=expiration, enabled=False)
        cal.close()
    assert sig.blocked_by != "earnings", sig.diagnostic_notes
    print("filter_disabled: flag=False bypasses earnings check entirely")


# ---------- payload parsing test ----------

def test_payload_parsing_filters_to_watchlist_and_skips_malformed():
    """The Finnhub payload is parsed with the watchlist as a filter and bad
    rows are skipped rather than failing the whole refresh."""
    payload = {
        "earningsCalendar": [
            {"symbol": "NVDA", "date": "2026-05-22", "epsEstimate": 1.5},
            {"symbol": "MSFT", "date": "2026-07-30"},  # not in watchlist
            {"symbol": "AAPL", "date": ""},            # bad row, skipped
            {"date": "2026-08-01"},                    # missing symbol, skipped
            "garbage",                                  # not a dict, skipped
            {"symbol": "NVDA", "date": "2026-08-21"},   # second NVDA earnings
        ]
    }
    rows = EarningsCalendar._parse_payload(payload, symbols=["NVDA", "AAPL"])
    syms = sorted({r[0] for r in rows})
    assert syms == ["NVDA"], f"expected only NVDA after watchlist filter, got {syms}"
    assert len(rows) == 2, f"expected 2 NVDA rows, got {len(rows)}"
    print("payload_parsing: bad rows skipped, watchlist filter applied")


def main() -> int:
    test_cache_miss_returns_none_until_refresh()
    test_seeded_cache_returns_true_for_in_window()
    test_refresh_skips_when_already_refreshed_today()
    test_refresh_silently_skips_when_api_key_missing()
    test_short_dte_entry_passes_when_report_lands_after_expiry()
    test_report_inside_position_life_blocks()
    test_report_on_expiration_day_boundary()
    test_signal_passes_when_no_earnings_data_available()
    test_filter_disabled_via_flag_skips_lookup()
    test_payload_parsing_filters_to_watchlist_and_skips_malformed()
    print("all earnings_calendar tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
