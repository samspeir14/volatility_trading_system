"""Unit tests for the shared entry filters on the h=1 gate ladder: macro-event
calendar (life-of-position window), VIX term-structure veto, credit-to-width
floor, and the wing-sigma calendar-day convention."""
import math
import sys
from datetime import date, datetime, timezone

from data.async_client import OptionContract
from data.macro_calendar import MACRO_EVENTS, MacroCalendar
from signals.signal_generator import (
    CALENDAR_DAYS_PER_YEAR,
    _pick_iron_condor_wings,
)
from tests.test_signal_generator_h1 import _run


# ---------- macro calendar ----------

def test_macro_calendar_window_and_staleness():
    cal = MacroCalendar()
    # 2026-07-13 .. 2026-08-01 covers CPI 07-14 and FOMC 07-29
    hit = cal.next_event_in_window(date(2026, 7, 13), date(2026, 8, 1))
    assert hit == (date(2026, 7, 14), "CPI release"), hit
    # A quiet stretch: 2026-07-30 .. 2026-08-11 (before CPI 08-12)
    assert cal.next_event_in_window(date(2026, 7, 30), date(2026, 8, 11)) is None
    # Aged-out table fails open
    assert cal.next_event_in_window(date(2050, 1, 1), date(2050, 12, 31)) is None
    assert len(MACRO_EVENTS) == 20, "2026: 8 FOMC + 12 CPI"
    print("macro_calendar: window hits, quiet stretch, aged-out fail-open")


def test_macro_filter_demotes_sensitive_symbol_only():
    """TLT (sensitive) with a macro event inside the position's life is
    demoted; an identical non-sensitive name passes. Fixtures run 06-01 →
    expiry 06-12; CPI 2026-06-10 falls inside that life."""
    actionable, all_signals, _ = _run(
        [("TLT", 0.30), ("XYZ", 0.30)],
        macro_calendar=MacroCalendar(),
        macro_sensitive_symbols={"TLT"},
    )
    by = {s.symbol: s for s in all_signals}

    assert by["TLT"].blocked_by == "macro", by["TLT"].diagnostic_notes
    assert "macro_event_within_position_life" in by["TLT"].diagnostic_notes
    assert "CPI" in by["TLT"].diagnostic_notes
    assert by["XYZ"].is_actionable, (
        f"non-sensitive XYZ demoted: {by['XYZ'].diagnostic_notes}"
    )
    assert [s.symbol for s in actionable] == ["XYZ"]
    print("macro_filter: TLT demoted on CPI-in-life, XYZ untouched")


def test_macro_filter_passes_when_window_clear():
    """Same TLT setup but the position expires before the next event."""
    cal = MacroCalendar(events=((date(2026, 6, 17), "FOMC decision"),))
    actionable, _, _ = _run(
        [("TLT", 0.30)],
        macro_calendar=cal, macro_sensitive_symbols={"TLT"},
    )  # life 06-01..06-12, event 06-17 outside
    assert [s.symbol for s in actionable] == ["TLT"]
    print("macro_filter: clear window passes")


# ---------- VIX term-structure veto ----------

def test_vix_backwardation_vetoes_all_sells():
    rows = [("A", 0.30), ("B", 0.36)]
    # Backwardation: every SELL demoted with the veto label
    actionable, all_signals, _ = _run(rows, vix_term_ratio=1.04)
    assert actionable == [], f"expected none actionable, got {actionable}"
    for s in all_signals:
        assert s.blocked_by == "vix_backwardation", s.diagnostic_notes
    # Contango: everything trades
    actionable, _, _ = _run(rows, vix_term_ratio=0.92)
    assert {s.symbol for s in actionable} == {"A", "B"}
    # Unavailable: fails open
    actionable, _, _ = _run(rows, vix_term_ratio=None)
    assert {s.symbol for s in actionable} == {"A", "B"}
    print("vix_veto: 1.04 blocks all, 0.92 passes, None fails open")


# ---------- credit-to-width floor ----------

def test_credit_to_width_floor():
    """Fixture condor: body mids 1.02+0.94=1.96 credit, wings 0.205 each →
    net 1.55 on $10 wings = 15.5% of width. A 25% floor demotes it; a 12%
    floor and a disabled floor pass."""
    rows = [("X", 0.30)]
    actionable, all_signals, _ = _run(rows, min_credit_to_width=0.25)
    assert actionable == []
    assert all_signals[0].blocked_by == "legs"
    assert "wing width" in all_signals[0].diagnostic_notes, \
        all_signals[0].diagnostic_notes
    # 12% floor: 15.5% clears it
    actionable, _, _ = _run(rows, min_credit_to_width=0.12)
    assert [s.symbol for s in actionable] == ["X"]
    # Disabled floor: passes
    actionable, _, _ = _run(rows)
    assert [s.symbol for s in actionable] == ["X"]
    print("credit_to_width: 25% floor demotes 15.5% condor, 12% floor passes")


# ---------- wing sigma convention ----------

def test_wing_sigma_uses_calendar_day_year():
    """1σ move for IV=0.30, S=100, dte=11 must use sqrt(11/365) — the old
    sqrt(11/252) placed wings ~20% too wide."""
    contracts = [
        OptionContract(
            symbol=f"W{k}{t[0]}", underlying="W", expiration=date(2026, 6, 12),
            strike=k, option_type=t, bid=0.1, ask=0.2, last=0.15, volume=10,
            open_interest=10, delta=0.2, gamma=0.01, theta=-0.01, vega=0.1,
            iv=0.30, fetched_at=datetime.now(timezone.utc),
        )
        for k in (95.0, 100.0, 105.0, 106.0, 94.0)
        for t in ("call", "put")
    ]
    wings = _pick_iron_condor_wings(
        contracts, atm_strike=100.0, predicted_iv_eq=0.30, dte=11,
        underlying_price=100.0,
    )
    assert wings is not None
    long_call, long_put = wings
    expected_move = 100.0 * 0.30 * math.sqrt(11 / CALENDAR_DAYS_PER_YEAR)  # ≈ 5.21
    assert abs(expected_move - 5.21) < 0.01, expected_move
    # Nearest strikes to 105.21 / 94.79 are 105 / 95 (not 106/94, which the
    # old 252 convention's 5.93 move would have picked)
    assert long_call.strike == 105.0, f"got {long_call.strike}"
    assert long_put.strike == 95.0, f"got {long_put.strike}"
    print(f"wing_sigma: 1σ={expected_move:.2f} via /365 -> strikes 105/95")


def main() -> int:
    test_macro_calendar_window_and_staleness()
    test_macro_filter_demotes_sensitive_symbol_only()
    test_macro_filter_passes_when_window_clear()
    test_vix_backwardation_vetoes_all_sells()
    test_credit_to_width_floor()
    test_wing_sigma_uses_calendar_day_year()
    print("all entry_filters tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
