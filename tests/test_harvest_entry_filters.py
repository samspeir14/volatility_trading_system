"""Unit tests for the harvest entry filters added 2026-07: macro-event
calendar, VIX term-structure veto, credit-to-width floor, and the wing-sigma
calendar-day convention."""
import math
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from data.async_client import OptionContract
from data.macro_calendar import MACRO_EVENTS, MacroCalendar
from data.market_data import ScanResult, TickerSnapshot
from signals.signal_generator import (
    CALENDAR_DAYS_PER_YEAR,
    SignalGenerator,
    _pick_iron_condor_wings,
)


class _ConstPredictor:
    def predict_forward_rv(self, returns_history, X_row=None):
        return 0.012  # ~0.19 annualized


# Alternating ±1.2% daily returns → trail63 ≈ 0.19 annualized (no spread veto)
_RETURNS = pd.Series([0.012, -0.012] * 50)


def _chain(sym, expiration, atm_iv, *, wing_bid=0.20, wing_ask=0.22):
    def _c(strike, otype, bid, ask, delta):
        return OptionContract(
            symbol=f"{sym}_{otype[0].upper()}{strike:.0f}_{expiration.isoformat()}",
            underlying=sym, expiration=expiration, strike=strike, option_type=otype,
            bid=bid, ask=ask, last=(bid + ask) / 2, volume=200, open_interest=1000,
            delta=delta, gamma=0.01, theta=-0.05, vega=0.2, iv=atm_iv,
            fetched_at=datetime.now(timezone.utc),
        )
    return [
        _c(100, "call", 1.0, 1.05, 0.5),
        _c(100, "put", 0.9, 0.95, -0.5),
        _c(110, "call", wing_bid, wing_ask, 0.2),
        _c(90, "put", wing_bid, wing_ask, -0.2),
    ]


def _scan(rows, expiration=date(2026, 6, 12), **chain_kwargs):
    snapshots = {
        sym: TickerSnapshot(
            symbol=sym, sector="tech",
            underlying={"symbol": sym, "last": 100.0},
            contracts=_chain(sym, expiration, iv, **chain_kwargs),
        )
        for sym, iv in rows
    }
    return ScanResult(fetched_at=datetime(2026, 6, 1, 16, 0, tzinfo=timezone.utc),
                      snapshots=snapshots)


def _gen(**kwargs) -> SignalGenerator:
    return SignalGenerator(
        predictors_by_horizon={h: _ConstPredictor() for h in (5, 10, 21)},
        history_store=None,
        strategy_mode="harvest",
        **kwargs,
    )


def _run(gen, scan, symbols, **generate_kwargs):
    feature_rows = {s: pd.DataFrame({"a": [0.0]}, index=[date(2026, 5, 29)])
                    for s in symbols}
    returns = {s: _RETURNS for s in symbols}
    return gen.generate(scan, feature_rows, returns, top_n=10, **generate_kwargs)


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
    """TLT (sensitive) with FOMC inside its life is demoted; an identical
    non-sensitive name passes. Window is life-of-position in harvest: scan
    day 2026-06-01, expiry 06-12, FOMC 06-17 is OUTSIDE -> also passes."""
    cal = MacroCalendar()
    # CPI 2026-06-10 falls inside 06-01..06-12 for both names
    scan = _scan([("TLT", 0.20), ("XYZ", 0.20)])
    gen = _gen(macro_calendar=cal, macro_sensitive_symbols={"TLT"})
    actionable, all_signals = _run(gen, scan, ["TLT", "XYZ"])
    by = {s.symbol: s for s in all_signals}

    assert not by["TLT"].is_actionable, "TLT must be macro-demoted"
    assert "macro_event_within_window" in by["TLT"].diagnostic_notes
    assert "CPI" in by["TLT"].diagnostic_notes
    assert by["XYZ"].is_actionable, (
        f"non-sensitive XYZ demoted: {by['XYZ'].diagnostic_notes}"
    )
    assert [s.symbol for s in actionable] == ["XYZ"]
    print("macro_filter: TLT demoted on CPI-in-window, XYZ untouched")


def test_macro_filter_passes_when_window_clear():
    """Same TLT setup but expiring before the next event -> actionable."""
    cal = MacroCalendar(events=((date(2026, 6, 17), "FOMC decision"),))
    scan = _scan([("TLT", 0.20)])  # life 06-01..06-12, event 06-17 outside
    gen = _gen(macro_calendar=cal, macro_sensitive_symbols={"TLT"})
    actionable, _ = _run(gen, scan, ["TLT"])
    assert [s.symbol for s in actionable] == ["TLT"]
    print("macro_filter: clear window passes")


# ---------- VIX term-structure veto ----------

def test_vix_backwardation_vetoes_all_harvest_sells():
    scan = _scan([("A", 0.20), ("B", 0.25)])
    gen = _gen()
    # Backwardation: every SELL demoted with the veto note
    actionable, all_signals = _run(gen, scan, ["A", "B"], vix_term_ratio=1.04)
    assert actionable == [], f"expected none actionable, got {actionable}"
    for s in all_signals:
        assert "vix-term-structure veto" in s.diagnostic_notes, s.diagnostic_notes
    # Contango: everything trades
    actionable, _ = _run(gen, scan, ["A", "B"], vix_term_ratio=0.92)
    assert {s.symbol for s in actionable} == {"A", "B"}
    # Unavailable: fails open
    actionable, _ = _run(gen, scan, ["A", "B"], vix_term_ratio=None)
    assert {s.symbol for s in actionable} == {"A", "B"}
    print("vix_veto: 1.04 blocks all, 0.92 passes, None fails open")


# ---------- credit-to-width floor ----------

def test_credit_to_width_floor():
    """Body mids 1.025+0.925=1.95 credit, wings 0.21 each -> net 1.53 on $10
    wings = 15.3% of width. A 25% floor demotes it; richer wings (net credit
    unchanged names) with tighter floor pass."""
    scan = _scan([("X", 0.20)])
    gen = _gen(min_credit_to_width=0.25)
    actionable, all_signals = _run(gen, scan, ["X"])
    assert actionable == []
    assert "wing width" in all_signals[0].diagnostic_notes, all_signals[0].diagnostic_notes
    # 12% floor: 15.3% clears it
    gen = _gen(min_credit_to_width=0.12)
    actionable, _ = _run(gen, scan, ["X"])
    assert [s.symbol for s in actionable] == ["X"]
    # Disabled floor: passes
    gen = _gen()
    actionable, _ = _run(gen, scan, ["X"])
    assert [s.symbol for s in actionable] == ["X"]
    print("credit_to_width: 25% floor demotes 15.3% condor, 12% floor passes")


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
    test_vix_backwardation_vetoes_all_harvest_sells()
    test_credit_to_width_floor()
    test_wing_sigma_uses_calendar_day_year()
    print("all harvest_entry_filters tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
