"""Unit tests for the EarningsCalendar cache + the SignalGenerator earnings filter.

Covers:
- Cache miss before any refresh → has_earnings_in_window returns None (fail open).
- Cache hit after seed → returns True/False per stored dates.
- refresh_if_stale skips when last_refresh_date == today.
- refresh_if_stale silently skips (no per-cycle WARNING) when API key is missing.
- SignalGenerator demotes a signal whose ticker has earnings inside [today, today+buffer].
- SignalGenerator does NOT demote when earnings fall outside the buffer.
- earnings_buffer_days is configurable: same earnings demoted at buffer=10, allowed at buffer=3.
- SignalGenerator does NOT demote when no earnings data is available (fail open).
- SignalGenerator does NOT demote when the filter is disabled via constructor flag.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

from data.async_client import OptionContract
from data.earnings_calendar import EarningsCalendar
from data.market_data import ScanResult, TickerSnapshot
from signals.signal_generator import SignalGenerator


# ---------- shared fixtures ----------

def _mk_contract(
    sym: str, otype: str, strike: float, expiration: date, *,
    bid=1.0, ask=1.05, iv=0.30, vol=200, oi=1000,
) -> OptionContract:
    return OptionContract(
        symbol=f"{sym}_{otype[0].upper()}{int(strike)}",
        underlying=sym, expiration=expiration, strike=strike,
        option_type=otype, bid=bid, ask=ask, last=(bid + ask) / 2,
        volume=vol, open_interest=oi,
        delta=0.5 if otype == "call" else -0.5,
        gamma=0.01, theta=-0.05, vega=0.20, iv=iv,
        fetched_at=datetime.now(timezone.utc),
    )


def _mk_snapshot_for(sym: str, expiration: date) -> TickerSnapshot:
    return TickerSnapshot(
        symbol=sym, sector="tech",
        underlying={"symbol": sym, "last": 100.0},
        contracts=[
            _mk_contract(sym, "call", 100, expiration),
            _mk_contract(sym, "put", 100, expiration),
            _mk_contract(sym, "call", 110, expiration, bid=0.20, ask=0.22, vol=150, oi=600),
            _mk_contract(sym, "put", 90, expiration, bid=0.20, ask=0.22, vol=150, oi=600),
        ],
    )


def _mk_scan(today: date, expiration: date, symbols: list[str]) -> ScanResult:
    snapshots = {sym: _mk_snapshot_for(sym, expiration) for sym in symbols}
    return ScanResult(
        fetched_at=datetime(today.year, today.month, today.day, 16, 0, tzinfo=timezone.utc),
        snapshots=snapshots,
    )


class _FixedPredictor:
    """Returns daily vol matching the IV used in fixtures so divergence ≈ 0."""
    def __init__(self, daily_vol: float = 0.0189):
        self._v = daily_vol

    def predict_forward_rv(self, returns_history, X_row=None):
        return self._v


def _build_generator(
    *,
    earnings: EarningsCalendar | None,
    enabled: bool = True,
    buffer_days: int = 7,
) -> SignalGenerator:
    predictors = {h: _FixedPredictor() for h in (5, 10, 21)}
    return SignalGenerator(
        predictors_by_horizon=predictors,
        history_store=None,
        cross_sectional_z_threshold=0.0,  # any signal passes the z gate
        max_divergence=0.50,              # well above the synthetic divergence
        earnings_calendar=earnings,
        earnings_filter_enabled=enabled,
        earnings_buffer_days=buffer_days,
    )


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


# ---------- SignalGenerator filter tests ----------

def test_signal_demoted_when_earnings_inside_buffer():
    """Earnings 4 days out, default buffer 7 → demoted."""
    today = date(2026, 5, 6)
    expiration = date(2026, 5, 22)  # DTE=16
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key=None)
        cal._seed_for_testing([("NVDA", date(2026, 5, 10))], today=today)
        gen = _build_generator(earnings=cal)  # default buffer_days=7

        scan = _mk_scan(today, expiration, ["NVDA"])
        feature_rows = {"NVDA": pd.DataFrame({"a": [0.0]}, index=[date(2026, 5, 5)])}
        returns_by_symbol = {"NVDA": pd.Series([0.001] * 100)}

        actionable, all_signals = gen.generate(
            scan, feature_rows, returns_by_symbol, top_n=10,
        )
        nvda_signals = [s for s in all_signals if s.symbol == "NVDA"]
        assert nvda_signals, "expected a signal for NVDA"
        sig = nvda_signals[0]
        assert not sig.is_actionable, "NVDA must be demoted by earnings filter"
        assert "earnings_within_window" in sig.diagnostic_notes
        assert "2026-05-10" in sig.diagnostic_notes
        assert "7-day buffer" in sig.diagnostic_notes
        assert "NVDA" not in [s.symbol for s in actionable]
        cal.close()
    print("filter_demote: NVDA earnings 5/10 (4d out) within 7-day buffer → demoted")


def test_signal_passes_when_earnings_outside_buffer_but_inside_expiration():
    """Earnings 14 days out, default buffer 7. Under the OLD rule (earnings ≤
    expiration) this would be demoted because 5/20 ≤ 5/22. Under the NEW rule
    (earnings within buffer) it passes — this is the case that demonstrates
    the rule change."""
    today = date(2026, 5, 6)
    expiration = date(2026, 5, 22)  # earnings 5/20 is BEFORE expiration
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key=None)
        cal._seed_for_testing([("NVDA", date(2026, 5, 20))], today=today)
        gen = _build_generator(earnings=cal)  # default buffer_days=7

        scan = _mk_scan(today, expiration, ["NVDA"])
        feature_rows = {"NVDA": pd.DataFrame({"a": [0.0]}, index=[date(2026, 5, 5)])}
        returns_by_symbol = {"NVDA": pd.Series([0.001] * 100)}

        _, all_signals = gen.generate(scan, feature_rows, returns_by_symbol, top_n=10)
        sig = next(s for s in all_signals if s.symbol == "NVDA")
        assert "earnings_within_window" not in sig.diagnostic_notes
        cal.close()
    print("filter_pass_outside_buffer: earnings 5/20 (14d out) outside 7-day buffer → not demoted")


def test_custom_buffer_days_changes_demotion():
    """A 6-days-out earnings: demoted with buffer=10, allowed with buffer=3."""
    today = date(2026, 5, 6)
    expiration = date(2026, 5, 22)
    earnings_date = date(2026, 5, 12)  # 6 days out
    feature_rows = {"NVDA": pd.DataFrame({"a": [0.0]}, index=[date(2026, 5, 5)])}
    returns_by_symbol = {"NVDA": pd.Series([0.001] * 100)}

    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key=None)
        cal._seed_for_testing([("NVDA", earnings_date)], today=today)

        # buffer=10: 6 days out is inside → demoted
        gen_wide = _build_generator(earnings=cal, buffer_days=10)
        scan = _mk_scan(today, expiration, ["NVDA"])
        _, sigs_wide = gen_wide.generate(scan, feature_rows, returns_by_symbol, top_n=10)
        sig_wide = next(s for s in sigs_wide if s.symbol == "NVDA")
        assert "earnings_within_window" in sig_wide.diagnostic_notes
        assert "10-day buffer" in sig_wide.diagnostic_notes

        # buffer=3: 6 days out is outside → allowed
        gen_narrow = _build_generator(earnings=cal, buffer_days=3)
        scan = _mk_scan(today, expiration, ["NVDA"])
        _, sigs_narrow = gen_narrow.generate(scan, feature_rows, returns_by_symbol, top_n=10)
        sig_narrow = next(s for s in sigs_narrow if s.symbol == "NVDA")
        assert "earnings_within_window" not in sig_narrow.diagnostic_notes
        cal.close()
    print("custom_buffer: 6d-out earnings demoted at buffer=10, allowed at buffer=3")


def test_signal_passes_when_no_earnings_data_available():
    """Fail-open: if the cache has no record (cache never refreshed), the
    filter must allow the trade rather than block all activity."""
    today = date(2026, 5, 6)
    expiration = date(2026, 5, 22)
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key=None)
        # Do NOT seed — cache has never been refreshed
        gen = _build_generator(earnings=cal)

        scan = _mk_scan(today, expiration, ["NVDA"])
        feature_rows = {"NVDA": pd.DataFrame({"a": [0.0]}, index=[date(2026, 5, 5)])}
        returns_by_symbol = {"NVDA": pd.Series([0.001] * 100)}

        _, all_signals = gen.generate(scan, feature_rows, returns_by_symbol, top_n=10)
        sig = next(s for s in all_signals if s.symbol == "NVDA")
        assert "earnings_within_window" not in sig.diagnostic_notes
        cal.close()
    print("filter_fail_open: empty cache → trade allowed (fail open)")


def test_filter_disabled_via_flag_skips_lookup():
    """Even when the calendar has data, earnings_filter_enabled=False must
    bypass the check. Useful for debugging without flushing the cache."""
    today = date(2026, 5, 6)
    expiration = date(2026, 5, 22)
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key=None)
        cal._seed_for_testing([("NVDA", date(2026, 5, 15))], today=today)
        gen = _build_generator(earnings=cal, enabled=False)

        scan = _mk_scan(today, expiration, ["NVDA"])
        feature_rows = {"NVDA": pd.DataFrame({"a": [0.0]}, index=[date(2026, 5, 5)])}
        returns_by_symbol = {"NVDA": pd.Series([0.001] * 100)}

        _, all_signals = gen.generate(scan, feature_rows, returns_by_symbol, top_n=10)
        sig = next(s for s in all_signals if s.symbol == "NVDA")
        assert "earnings_within_window" not in sig.diagnostic_notes
        cal.close()
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
    test_signal_demoted_when_earnings_inside_buffer()
    test_signal_passes_when_earnings_outside_buffer_but_inside_expiration()
    test_custom_buffer_days_changes_demotion()
    test_signal_passes_when_no_earnings_data_available()
    test_filter_disabled_via_flag_skips_lookup()
    test_payload_parsing_filters_to_watchlist_and_skips_malformed()
    print("all earnings_calendar tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
