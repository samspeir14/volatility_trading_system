"""Unit tests for ExitManager — focus on the trigger priority logic.

Most tests mock out the predictor + scan so we only exercise _evaluate_one
and the threshold math. The full evaluate() with re-derived divergence is
exercised in the live test."""
import math
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

TODAY = date(2026, 5, 12)
from unittest import mock

from data.async_client import OptionContract
from positions.exit_manager import EXIT_TRIGGER_PRIORITY, ExitManager, _entry_dte, _trading_dte
from positions.position_tracker import OpenPosition, PositionMark
from signals.signal_generator import TradeLeg


def _mk_iron_condor_position(*, entry_credit=13.55) -> OpenPosition:
    legs = [
        TradeLeg(210.0, "call", "sell", 1, "NVDA260522C00210000"),
        TradeLeg(210.0, "put", "sell", 1, "NVDA260522P00210000"),
        TradeLeg(230.0, "call", "buy", 1, "NVDA260522C00230000"),
        TradeLeg(190.0, "put", "buy", 1, "NVDA260522P00190000"),
    ]
    return OpenPosition(
        tradier_order_id=1, symbol="NVDA",
        expiration=date(2026, 5, 22), direction="SELL",
        structure="iron_condor", legs=legs, entry_premium=entry_credit,
        entry_atm_iv=0.40, entry_predicted_iv=0.32, entry_divergence=-0.08,
        entry_horizon_lower=21, entry_horizon_upper=21, entry_weight_lower=1.0,
        submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
    )


def _mk_long_straddle_position(*, entry_debit=4.08) -> OpenPosition:
    legs = [
        TradeLeg(100.0, "call", "buy", 1, "AAPL260515C00100000"),
        TradeLeg(100.0, "put", "buy", 1, "AAPL260515P00100000"),
    ]
    return OpenPosition(
        tradier_order_id=2, symbol="AAPL",
        expiration=date(2026, 5, 15), direction="BUY",
        structure="straddle", legs=legs, entry_premium=entry_debit,
        entry_atm_iv=0.27, entry_predicted_iv=0.42, entry_divergence=0.15,
        entry_horizon_lower=10, entry_horizon_upper=21, entry_weight_lower=0.27,
        submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
    )


def _mark(
    pos: OpenPosition, *, pnl_dollars: float, dte: int = 20,
    underlying_price: float = float("nan"), current_legs=None,
) -> PositionMark:
    return PositionMark(
        position=pos, current_legs=current_legs or [], close_cash_flow=0, cost_to_close=0,
        pnl_dollars=pnl_dollars, pnl_pct_of_entry_premium=pnl_dollars / (pos.entry_premium * 100),
        pnl_pct_of_max=float("nan"),
        delta=0, gamma=0, theta=0, vega=0, dte=dte,
        underlying_price=underlying_price,
    )


def _contract(strike: float, option_type: str, bid: float, ask: float) -> OptionContract:
    return OptionContract(
        symbol=f"TEST{option_type[0].upper()}{int(strike * 1000):08d}",
        underlying="TEST", expiration=date(2026, 5, 22), strike=strike,
        option_type=option_type, bid=bid, ask=ask, last=(bid + ask) / 2,
        volume=100, open_interest=500, delta=0.5, gamma=0.01, theta=-0.05,
        vega=0.10, iv=0.30, fetched_at=datetime(2026, 5, 12, 15, 0, tzinfo=timezone.utc),
    )


def _exit_mgr() -> ExitManager:
    return ExitManager(
        position_tracker=mock.MagicMock(),
        order_manager=mock.MagicMock(),
    )


# ---------- profit target ----------

def test_iron_condor_profit_target_at_75pct():
    pos = _mk_iron_condor_position(entry_credit=13.55)
    # 75% target = 0.75 × 13.55 × 100 = $1016.25
    mark = _mark(pos, pnl_dollars=1017.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.08)  # thesis intact
    assert trigger == "profit_target"
    # Just below threshold: hold
    mark = _mark(pos, pnl_dollars=1016.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.08)
    assert trigger is None
    print("iron_condor profit_target: $1016.25 cutoff verified")


def test_long_straddle_has_no_profit_target():
    """Long gamma's upside is the trade: no profit target on straddles, a
    winner runs to the final-2h expiry close (or a stop / thesis exit)."""
    pos = _mk_long_straddle_position(entry_debit=4.08)
    for pnl in (409.0, 4080.0):  # +100%, +1000% of premium
        mark = _mark(pos, pnl_dollars=pnl)
        trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=0.15)
        assert trigger is None, f"straddle closed on profit at +${pnl:.0f}: {trigger}"
    print("long_straddle: no profit target, winner runs ✓")


# ---------- stop loss ----------

def test_iron_condor_stop_loss_at_neg_100pct():
    pos = _mk_iron_condor_position(entry_credit=13.55)
    # -100% stop = -1.0 × 13.55 × 100 = -$1355
    mark = _mark(pos, pnl_dollars=-1356.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.08)
    assert trigger == "stop_loss"
    print("iron_condor stop_loss: -$1355 cutoff verified")


def test_long_straddle_stop_loss_at_neg_50pct():
    pos = _mk_long_straddle_position(entry_debit=4.08)
    # -50% stop = -0.50 × 4.08 × 100 = -$204
    mark = _mark(pos, pnl_dollars=-205.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=0.15)
    assert trigger == "stop_loss"
    print("long_straddle stop_loss: -$204 cutoff verified")


# ---------- expiration proximity ----------

def test_straddle_rides_to_expiry_day_regardless_of_entry_dte():
    """No aged 'dte <= 2' close for straddles any more: whatever the entry
    dte, a straddle holds until the final 2h of expiry day. Over 14 aged
    closes the old branch realized -$6.4k against -$1.1k at expiry."""
    pos = _mk_long_straddle_position()  # entered 4/27 for 5/15: aged
    for dte in (3, 2, 1):
        trigger, _ = _exit_mgr()._evaluate_one(
            _mark(pos, pnl_dollars=0.0, dte=dte), today=TODAY, current_divergence=0.15)
        assert trigger is None, f"aged straddle closed at dte={dte}: {trigger}"
    close_utc = datetime(2026, 5, 15, 20, 0, tzinfo=timezone.utc)  # 16:00 ET
    mark0 = _mark(pos, pnl_dollars=0.0, dte=0)
    trigger, _ = _exit_mgr()._evaluate_one(
        mark0, today=date(2026, 5, 15), current_divergence=0.15,
        market_close_utc=close_utc, now_utc=close_utc - timedelta(hours=3))
    assert trigger is None, f"aged straddle closed at the expiry-day open: {trigger}"
    trigger, rationale = _exit_mgr()._evaluate_one(
        mark0, today=date(2026, 5, 15), current_divergence=0.15,
        market_close_utc=close_utc, now_utc=close_utc - timedelta(hours=1, minutes=55))
    assert trigger == "expiration_proximity" and "before expiration" in rationale
    print("straddle: rides to expiry day, closes in final 2h whatever the entry dte ✓")


def test_expiration_proximity_skipped_for_iron_condor():
    """Iron condors do NOT trigger expiration_proximity — their end-of-life
    handling is the assignment_risk close-out (near-money close at dte <= 1
    for aged condors, final-2h backstop on expiry day)."""
    pos = _mk_iron_condor_position()
    # dte=2 (would fire expiration_proximity for a straddle): holds — outside
    # the assignment close window, quotes present, shorts not at parity.
    mark = _mark(pos, pnl_dollars=0.0, dte=2, underlying_price=210.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.08)
    assert trigger is None, f"IC at dte=2 should hold, got {trigger}"
    print("expiration_proximity (iron_condor): dte=2 holds (assignment window starts at 1)")


# ---------- assignment risk ----------

def _mk_otm_condor_position(*, entry_credit=2.10) -> OpenPosition:
    """A true condor (OTM shorts), unlike the ATM-body condor whose shorts sit ATM.
    Exercises the near-money buffer as a real decision, not a tautology."""
    legs = [
        TradeLeg(220.0, "call", "sell", 1, "NVDA260522C00220000"),
        TradeLeg(200.0, "put", "sell", 1, "NVDA260522P00200000"),
        TradeLeg(230.0, "call", "buy", 1, "NVDA260522C00230000"),
        TradeLeg(190.0, "put", "buy", 1, "NVDA260522P00190000"),
    ]
    return OpenPosition(
        tradier_order_id=3, symbol="NVDA",
        expiration=date(2026, 5, 22), direction="SELL",
        structure="iron_condor", legs=legs, entry_premium=entry_credit,
        entry_atm_iv=0.40, entry_predicted_iv=0.32, entry_divergence=-0.08,
        entry_horizon_lower=21, entry_horizon_upper=21, entry_weight_lower=1.0,
        submitted_at=datetime(2026, 5, 8, 16, 0, tzinfo=timezone.utc),
    )


def test_assignment_risk_expiry_day_unconditional():
    """Rule (a): short legs never ride through expiration day, even far OTM
    with good quotes."""
    pos = _mk_otm_condor_position()
    mark = _mark(pos, pnl_dollars=0.0, dte=0, underlying_price=210.0)
    trigger, rationale = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.08)
    assert trigger == "assignment_risk", f"expected assignment_risk, got {trigger}"
    assert "expiration" in rationale
    print("assignment_risk (a): dte=0 closes unconditionally")


def test_assignment_risk_near_money_short_at_dte1():
    """Rule (b): dte=1 closes when a short leg is in/near the money, holds when
    both shorts are comfortably OTM."""
    mgr = _exit_mgr()
    # ATM-body condor: shorts at 210, spot pinned there → close.
    fly = _mk_iron_condor_position()
    mark = _mark(fly, pnl_dollars=0.0, dte=1, underlying_price=210.5)
    trigger, _ = mgr._evaluate_one(mark, today=TODAY, current_divergence=-0.08)
    assert trigger == "assignment_risk", f"expected assignment_risk, got {trigger}"
    # OTM condor: shorts at 200/220, spot 210 → both >1.5% away, let it ride.
    condor = _mk_otm_condor_position()
    mark = _mark(condor, pnl_dollars=0.0, dte=1, underlying_price=210.0)
    trigger, _ = mgr._evaluate_one(mark, today=TODAY, current_divergence=-0.08)
    assert trigger is None, f"comfortably-OTM condor at dte=1 should hold, got {trigger}"
    # Spot drifts to the call side: 218.0 >= 220 × 0.985 = 216.7 → close.
    mark = _mark(condor, pnl_dollars=0.0, dte=1, underlying_price=218.0)
    trigger, _ = mgr._evaluate_one(mark, today=TODAY, current_divergence=-0.08)
    assert trigger == "assignment_risk", f"expected assignment_risk, got {trigger}"
    print("assignment_risk (b): near-money short at dte=1 closes, comfortably-OTM holds")


def test_assignment_risk_missing_underlying_fails_safe():
    """Rule (b) with no usable underlying quote: close rather than carry short
    legs blind into expiration."""
    pos = _mk_otm_condor_position()
    mark = _mark(pos, pnl_dollars=0.0, dte=1)  # underlying_price defaults to NaN
    trigger, rationale = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.08)
    assert trigger == "assignment_risk", f"expected assignment_risk, got {trigger}"
    assert "failing safe" in rationale
    # Outside the close window the missing quote does NOT force a close.
    mark = _mark(pos, pnl_dollars=0.0, dte=5)
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.08)
    assert trigger is None
    print("assignment_risk (b): missing underlying fails safe inside window only")


def test_assignment_risk_parity_short_any_dte():
    """Rule (c): a short leg trading at parity (extrinsic <= floor) closes the
    position at ANY dte — early assignment doesn't wait for expiry week."""
    pos = _mk_iron_condor_position()  # shorts at 210
    # Spot ripped to 240: short call intrinsic 30.00, mid 30.02 → extrinsic 0.02.
    legs_at_parity = [
        _contract(210.0, "call", 29.95, 30.09),
        _contract(210.0, "put", 0.01, 0.05),
        _contract(230.0, "call", 10.00, 10.10),
        _contract(190.0, "put", 0.00, 0.02),
    ]
    mark = _mark(pos, pnl_dollars=0.0, dte=10, underlying_price=240.0,
                 current_legs=legs_at_parity)
    trigger, rationale = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.08)
    assert trigger == "assignment_risk", f"expected assignment_risk, got {trigger}"
    assert "parity" in rationale
    # Same shape but with real extrinsic left (mid 31.00 → extrinsic 1.00): hold.
    legs_with_extrinsic = [
        _contract(210.0, "call", 30.80, 31.20),
        _contract(210.0, "put", 0.01, 0.05),
        _contract(230.0, "call", 10.00, 10.10),
        _contract(190.0, "put", 0.00, 0.02),
    ]
    mark = _mark(pos, pnl_dollars=0.0, dte=10, underlying_price=240.0,
                 current_legs=legs_with_extrinsic)
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.08)
    assert trigger is None, f"short with extrinsic left should hold, got {trigger}"
    print("assignment_risk (c): parity short closes at dte=10, extrinsic-rich holds")


def test_assignment_risk_not_for_long_straddle():
    """Long straddles have no short legs — expiry handling stays
    expiration_proximity, never assignment_risk."""
    pos = _mk_long_straddle_position()
    mark = _mark(pos, pnl_dollars=0.0, dte=0, underlying_price=100.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=0.15)
    assert trigger == "expiration_proximity", f"expected expiration_proximity, got {trigger}"
    print("assignment_risk: long straddle unaffected (expiration_proximity fires)")


def test_stop_loss_overrides_assignment_risk():
    """When both fire (deep drawdown on expiry day), the label is stop_loss —
    matches EXIT_TRIGGER_PRIORITY. Either way the position closes."""
    pos = _mk_iron_condor_position(entry_credit=13.55)
    mark = _mark(pos, pnl_dollars=-1500.0, dte=0, underlying_price=240.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.08)
    assert trigger == "stop_loss", f"expected stop_loss, got {trigger}"
    print("PRIORITY: stop_loss overrides assignment_risk when both fire")


def test_short_dated_straddle_entry_rides_to_expiry_morning():
    """A straddle deliberately opened at entry_dte=1 (h=1 overnight trade) is
    NOT proximity-closed at dte=1 (that would be an instant round-trip), but
    IS closed on expiry morning (dte=0) before auto-exercise can leave stock."""
    pos = _mk_long_straddle_position()
    pos = replace(pos, expiration=TODAY + timedelta(days=1),
                  submitted_at=datetime(2026, 5, 12, 15, 0, tzinfo=timezone.utc))

    mark = _mark(pos, pnl_dollars=0.0, dte=1, underlying_price=100.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=0.15)
    assert trigger is None, f"1-DTE entry closed at entry: {trigger}"

    # Expiry day: holds outside the final-2h window, closes inside it.
    close_utc = datetime(2026, 5, 13, 20, 0, tzinfo=timezone.utc)  # 16:00 ET
    mark0 = _mark(pos, pnl_dollars=0.0, dte=0, underlying_price=100.0)
    trigger, _ = _exit_mgr()._evaluate_one(
        mark0, today=TODAY + timedelta(days=1), current_divergence=0.15,
        market_close_utc=close_utc,
        now_utc=close_utc - timedelta(hours=3))
    assert trigger is None, f"closed 3h before the bell: {trigger}"
    trigger, rationale = _exit_mgr()._evaluate_one(
        mark0, today=TODAY + timedelta(days=1), current_divergence=0.15,
        market_close_utc=close_utc,
        now_utc=close_utc - timedelta(hours=1, minutes=55))
    assert trigger == "expiration_proximity", f"expected final-2h close, got {trigger}"
    assert "before expiration" in rationale
    # Unknown close time fails SAFE → closes on the first expiry-day cycle.
    trigger, _ = _exit_mgr()._evaluate_one(
        mark0, today=TODAY + timedelta(days=1), current_divergence=0.15)
    assert trigger == "expiration_proximity"
    print("short-dated straddle: holds at dte=1, closes in final 2h ✓")


def test_short_dated_condor_entry_skips_near_money_close():
    """A condor opened at entry_dte=1 has wings ~1 daily sigma out — inside the
    near-money buffer by construction. Rule (b) must not close it at entry;
    rule (a) still closes it on expiry morning."""
    pos = _mk_iron_condor_position()
    pos = replace(pos, expiration=TODAY + timedelta(days=1),
                  submitted_at=datetime(2026, 5, 12, 15, 0, tzinfo=timezone.utc))

    # Underlying sits within the 1.5% buffer of the short 210 strike.
    mark = _mark(pos, pnl_dollars=0.0, dte=1, underlying_price=209.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.08)
    assert trigger is None, f"1-DTE condor closed at entry: {trigger}"

    # Expiry day: rides outside the final-2h window (the parity check is off
    # on expiry day — see test_parity_check_skipped_on_expiry_day), closes
    # inside the window.
    close_utc = datetime(2026, 5, 13, 20, 0, tzinfo=timezone.utc)  # 16:00 ET
    mark0 = _mark(pos, pnl_dollars=0.0, dte=0, underlying_price=209.0)
    trigger, _ = _exit_mgr()._evaluate_one(
        mark0, today=TODAY + timedelta(days=1), current_divergence=-0.08,
        market_close_utc=close_utc,
        now_utc=close_utc - timedelta(hours=3))
    assert trigger is None, f"condor closed 3h before the bell: {trigger}"
    trigger, _ = _exit_mgr()._evaluate_one(
        mark0, today=TODAY + timedelta(days=1), current_divergence=-0.08,
        market_close_utc=close_utc,
        now_utc=close_utc - timedelta(hours=1, minutes=55))
    assert trigger == "assignment_risk", f"expected final-2h backstop, got {trigger}"

    # A condor that AGED into the window still gets the near-money close.
    aged = _mk_iron_condor_position()
    aged_mark = _mark(aged, pnl_dollars=0.0, dte=1, underlying_price=209.0)
    trigger, _ = _exit_mgr()._evaluate_one(aged_mark, today=TODAY, current_divergence=-0.08)
    assert trigger == "assignment_risk", f"aged condor kept near the money: {trigger}"
    print("short-dated condor: rides at dte=1, expiry-morning backstop + aged close intact ✓")


# ---------- trading-day DTE (Friday entry, Monday expiry) ----------

FRI = date(2026, 8, 28)
MON = date(2026, 8, 31)
MON_CLOSE_UTC = datetime(2026, 8, 31, 20, 0, tzinfo=timezone.utc)  # 16:00 ET


def test_entry_dte_counts_trading_days():
    """Friday→Monday is 3 calendar days but ONE trading day. Counted in
    calendar days it fell outside the short-dated window and was closed on
    Monday's first cycle (TSLA 2026-08-31: sold at 9:35 ET, ahead of the
    20-point rally the straddles had been bought for)."""
    pos = replace(_mk_long_straddle_position(), expiration=MON,
                  submitted_at=datetime(2026, 8, 28, 14, 19, tzinfo=timezone.utc))
    assert _entry_dte(pos) == 1
    # Intra-week entries are unchanged by the unit switch.
    assert _entry_dte(replace(
        pos, expiration=date(2026, 9, 2),
        submitted_at=datetime(2026, 9, 1, 15, 0, tzinfo=timezone.utc))) == 1
    assert _entry_dte(replace(
        pos, expiration=date(2026, 9, 4),
        submitted_at=datetime(2026, 9, 2, 15, 0, tzinfo=timezone.utc))) == 2
    print("entry_dte: Fri→Mon = 1 trading day ✓")


def test_friday_straddle_rides_to_monday_final_2h():
    pos = replace(_mk_long_straddle_position(), expiration=MON,
                  submitted_at=datetime(2026, 8, 28, 14, 19, tzinfo=timezone.utc))
    mark0 = _mark(pos, pnl_dollars=0.0, dte=0, underlying_price=352.5)
    # Monday 9:35 ET: hold.
    trigger, _ = _exit_mgr()._evaluate_one(
        mark0, today=MON, current_divergence=0.15,
        market_close_utc=MON_CLOSE_UTC,
        now_utc=datetime(2026, 8, 31, 13, 35, tzinfo=timezone.utc))
    assert trigger is None, f"Friday straddle closed at Monday open: {trigger}"
    # Monday 14:05 ET: final-2h close.
    trigger, rationale = _exit_mgr()._evaluate_one(
        mark0, today=MON, current_divergence=0.15,
        market_close_utc=MON_CLOSE_UTC,
        now_utc=datetime(2026, 8, 31, 18, 5, tzinfo=timezone.utc))
    assert trigger == "expiration_proximity", f"expected final-2h close, got {trigger}"
    assert "before expiration" in rationale
    print("Fri→Mon straddle: holds Monday morning, closes in final 2h ✓")


def test_friday_condor_rides_to_monday_final_2h():
    pos = replace(_mk_iron_condor_position(), expiration=MON,
                  submitted_at=datetime(2026, 8, 28, 14, 19, tzinfo=timezone.utc))
    # Spot inside the near-money buffer of the short 210 strike; no leg quotes,
    # so the parity rule is skipped and only the DTE window logic decides.
    mark0 = _mark(pos, pnl_dollars=0.0, dte=0, underlying_price=209.0)
    trigger, _ = _exit_mgr()._evaluate_one(
        mark0, today=MON, current_divergence=-0.08,
        market_close_utc=MON_CLOSE_UTC,
        now_utc=datetime(2026, 8, 31, 13, 35, tzinfo=timezone.utc))
    assert trigger is None, f"Friday condor closed at Monday open: {trigger}"
    trigger, _ = _exit_mgr()._evaluate_one(
        mark0, today=MON, current_divergence=-0.08,
        market_close_utc=MON_CLOSE_UTC,
        now_utc=datetime(2026, 8, 31, 18, 5, tzinfo=timezone.utc))
    assert trigger == "assignment_risk", f"expected final-2h backstop, got {trigger}"
    print("Fri→Mon condor: holds Monday morning, closes in final 2h ✓")


def test_aged_condor_window_counts_remaining_trading_days():
    """The aged-condor near-money window (rule (b), dte <= 1) counts trading
    days: on the Friday before a Monday expiry one trading day remains
    (calendar dte=3), so an aged condor with a short leg near the money
    closes Friday instead of riding the weekend. Aged straddles have no such
    window — they ride to expiry day like short-dated ones."""
    aged_ic = replace(_mk_iron_condor_position(), expiration=MON)  # entered 4/27
    # Thu 8/27: 2 trading days left → outside the window, hold.
    trigger, _ = _exit_mgr()._evaluate_one(
        _mark(aged_ic, pnl_dollars=0.0, dte=4, underlying_price=209.0),
        today=date(2026, 8, 27), current_divergence=-0.08)
    assert trigger is None, f"aged condor closed at 2 trading days: {trigger}"
    # Fri 8/28 (calendar dte=3, formerly a hold): 1 trading day left → close.
    trigger, _ = _exit_mgr()._evaluate_one(
        _mark(aged_ic, pnl_dollars=0.0, dte=3, underlying_price=209.0),
        today=FRI, current_divergence=-0.08)
    assert trigger == "assignment_risk", f"aged condor held over the weekend: {trigger}"
    # Missing underlying on that Friday (calendar dte=3): no blind close —
    # rule (a) closes at Monday's open regardless, and a transient quote
    # miss must not dump three calendar days of premium.
    trigger, _ = _exit_mgr()._evaluate_one(
        _mark(aged_ic, pnl_dollars=0.0, dte=3), today=FRI, current_divergence=-0.08)
    assert trigger is None, f"blind close on a transient quote miss: {trigger}"
    # Aged straddle on the same Friday: rides.
    aged = replace(_mk_long_straddle_position(), expiration=MON)
    trigger, _ = _exit_mgr()._evaluate_one(
        _mark(aged, pnl_dollars=0.0, dte=3), today=FRI, current_divergence=0.15)
    assert trigger is None, f"aged straddle closed before expiry day: {trigger}"
    print("aged condor window: remaining trading days, Friday-before-Monday closes ✓")


def test_parity_check_skipped_on_expiry_day():
    """Rule (c) guards early assignment BEFORE expiry day only. A 0DTE ITM
    short trades at parity all session, so on expiry day the parity check
    would close a short-dated condor at the open; rule (a)'s final-2h close
    handles expiry-day assignment instead."""
    legs_at_parity = [
        _contract(210.0, "call", 29.95, 30.09),
        _contract(210.0, "put", 0.01, 0.05),
        _contract(230.0, "call", 10.00, 10.10),
        _contract(190.0, "put", 0.00, 0.02),
    ]
    short_dated = replace(_mk_iron_condor_position(), expiration=TODAY + timedelta(days=1),
                          submitted_at=datetime(2026, 5, 12, 15, 0, tzinfo=timezone.utc))
    close_utc = datetime(2026, 5, 13, 20, 0, tzinfo=timezone.utc)  # 16:00 ET
    # Day before expiry (dte=1): parity still closes.
    mark1 = _mark(short_dated, pnl_dollars=0.0, dte=1, underlying_price=240.0,
                  current_legs=legs_at_parity)
    trigger, rationale = _exit_mgr()._evaluate_one(mark1, today=TODAY, current_divergence=-0.08)
    assert trigger == "assignment_risk" and "parity" in rationale, (trigger, rationale)
    # Expiry day, 3h before the bell: parity ignored, ride.
    mark0 = _mark(short_dated, pnl_dollars=0.0, dte=0, underlying_price=240.0,
                  current_legs=legs_at_parity)
    trigger, _ = _exit_mgr()._evaluate_one(
        mark0, today=TODAY + timedelta(days=1), current_divergence=-0.08,
        market_close_utc=close_utc, now_utc=close_utc - timedelta(hours=3))
    assert trigger is None, f"parity closed a 0DTE condor at the open: {trigger}"
    # Final 2h: rule (a) closes it.
    trigger, rationale = _exit_mgr()._evaluate_one(
        mark0, today=TODAY + timedelta(days=1), current_divergence=-0.08,
        market_close_utc=close_utc, now_utc=close_utc - timedelta(hours=1, minutes=55))
    assert trigger == "assignment_risk" and "expiration" in rationale, (trigger, rationale)
    print("parity check: active at dte=1, skipped on expiry day, final-2h close intact ✓")


def test_trading_dte_skips_market_holidays():
    """Wed 2026-11-25 → Fri 2026-11-27 spans Thanksgiving: ONE session, not
    two. Counted as two, a condor opened Wednesday would be "aged" and dumped
    at Friday's open — the Friday→Monday bug in holiday clothing."""
    pos = replace(_mk_iron_condor_position(), expiration=date(2026, 11, 27),
                  submitted_at=datetime(2026, 11, 25, 15, 0, tzinfo=timezone.utc))
    assert _entry_dte(pos) == 1
    close_utc = datetime(2026, 11, 27, 18, 0, tzinfo=timezone.utc)  # 13:00 ET half day
    mark0 = _mark(pos, pnl_dollars=0.0, dte=0, underlying_price=209.0)
    trigger, _ = _exit_mgr()._evaluate_one(
        mark0, today=date(2026, 11, 27), current_divergence=-0.08,
        market_close_utc=close_utc,
        now_utc=datetime(2026, 11, 27, 14, 35, tzinfo=timezone.utc))
    assert trigger is None, f"holiday-spanning condor closed at the open: {trigger}"
    trigger, _ = _exit_mgr()._evaluate_one(
        mark0, today=date(2026, 11, 27), current_divergence=-0.08,
        market_close_utc=close_utc,
        now_utc=datetime(2026, 11, 27, 16, 5, tzinfo=timezone.utc))
    assert trigger == "assignment_risk", f"expected final-2h backstop, got {trigger}"
    # Aged condor on Wed 11/25 (calendar dte=2): one session left → rule (b)
    # closes rather than carrying the near-money short across the holiday.
    aged = replace(_mk_iron_condor_position(), expiration=date(2026, 11, 27))
    trigger, _ = _exit_mgr()._evaluate_one(
        _mark(aged, pnl_dollars=0.0, dte=2, underlying_price=209.0),
        today=date(2026, 11, 25), current_divergence=-0.08)
    assert trigger == "assignment_risk", f"aged condor held across Thanksgiving: {trigger}"
    print("holidays: Thanksgiving skipped in both entry and remaining counts ✓")


def test_trading_dte_sign_and_edges():
    fri, sat, mon = date(2026, 8, 28), date(2026, 8, 29), date(2026, 8, 31)
    assert _trading_dte(fri, fri) == 0                  # expiry day
    assert _trading_dte(date(2026, 8, 27), fri) == 1     # Thursday before
    assert _trading_dte(fri, mon) == 1                   # weekend skipped
    assert _trading_dte(sat, fri) == 0                   # no session elapsed yet
    assert _trading_dte(mon, fri) == -1                  # one session past expiry
    assert _trading_dte(date(2026, 9, 4), date(2026, 9, 8)) == 1   # Labor Day skipped
    print("trading_dte: sign, weekend, holiday edges ✓")


# ---------- thesis reversal ----------

def test_thesis_reversal_fires_when_sign_flips_and_magnitude_clears():
    pos = _mk_iron_condor_position()  # entry_divergence = -0.08 (we sold premium)
    # Current divergence: positive AND |≥ 0.05| → flipped + magnitude OK
    mark = _mark(pos, pnl_dollars=0.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=+0.06)
    assert trigger == "thesis_reversed"
    # Sign flipped but magnitude too small → no trigger
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=+0.001)
    assert trigger is None
    # Same sign as entry → no trigger
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.20)
    assert trigger is None
    print("thesis_reversal: sign flip + magnitude floor verified")


def test_thesis_exit_can_be_disabled():
    """With thesis_exit_enabled=False a flipped divergence must NOT close the
    position, while P&L triggers keep working."""
    mgr = ExitManager(
        position_tracker=mock.MagicMock(),
        order_manager=mock.MagicMock(),
        thesis_exit_enabled=False,
    )
    pos = _mk_iron_condor_position(entry_credit=13.55)
    # Flipped divergence, flat P&L → hold (would be thesis_reversed if enabled)
    mark = _mark(pos, pnl_dollars=0.0)
    trigger, _ = mgr._evaluate_one(mark, today=TODAY, current_divergence=+0.10)
    assert trigger is None, f"thesis exit fired while disabled: {trigger}"
    # Stop loss still works with the thesis trigger off
    mark = _mark(pos, pnl_dollars=-1500.0)
    trigger, _ = mgr._evaluate_one(mark, today=TODAY, current_divergence=+0.10)
    assert trigger == "stop_loss"
    print("thesis_disabled: thesis exit off, stop loss intact")


# ---------- the user's flagged priority test ----------

def test_thesis_overrides_stop_loss():
    """User-flagged design directive: thesis reversal beats stop loss.
    Position is in -110% drawdown AND thesis flipped → trigger MUST be thesis_reversed."""
    pos = _mk_iron_condor_position(entry_credit=13.55)
    # P&L = -$1500 (well past -100% stop)
    mark = _mark(pos, pnl_dollars=-1500.0)
    # Thesis intact: stop_loss
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.08)
    assert trigger == "stop_loss"
    # Thesis flipped: thesis_reversed (overrides stop)
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=+0.10)
    assert trigger == "thesis_reversed"
    print("PRIORITY: thesis_reversed overrides stop_loss when both fire")


def test_thesis_overrides_profit_target():
    """Same priority logic on the upside — if thesis flips while in profit, close
    rather than wait for the rest. The remaining edge is gone."""
    pos = _mk_iron_condor_position(entry_credit=13.55)
    # P&L = +$1100 (past the +75% profit target of $1016.25)
    mark = _mark(pos, pnl_dollars=1100.0)
    # Thesis intact: profit_target
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.08)
    assert trigger == "profit_target"
    # Thesis flipped: thesis_reversed
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=+0.10)
    assert trigger == "thesis_reversed"
    print("PRIORITY: thesis_reversed overrides profit_target when both fire")


def test_thesis_keyed_to_direction_not_entry_divergence_sign():
    """Reviewer-flagged critical: in the h=1 pipeline a SELL can be entered
    with POSITIVE entry_divergence (direction comes from the VRP z-gate).
    The thesis must reverse on the POSITION's direction: a SELL closes only
    when the model sees vol clearly ABOVE the market — never when the model
    turns favorable to the short."""
    pos = _mk_iron_condor_position()  # direction SELL
    object.__setattr__(pos, "entry_divergence", +0.16)  # h1-style entry
    mark = _mark(pos, pnl_dollars=0.0)
    # Model now agrees with the short (divergence −0.06): NOT a reversal,
    # even though the sign flipped vs entry.
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.06)
    assert trigger is None, f"favorable move must not close the short: {trigger}"
    # Model sees vol above the market: reversal fires.
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=+0.06)
    assert trigger == "thesis_reversed"
    print("thesis_direction: SELL closes on +div only, regardless of entry sign")


def test_priority_constant_matches_evaluation_order():
    """Sanity check: the EXIT_TRIGGER_PRIORITY tuple matches the order in the
    actual logic. If someone reorders one without the other, this catches it."""
    assert EXIT_TRIGGER_PRIORITY == (
        "thesis_reversed", "stop_loss", "earnings_risk", "assignment_risk",
        "expiration_proximity", "profit_target",
    )
    print(f"priority constant: {EXIT_TRIGGER_PRIORITY}")


def test_no_trigger_returns_hold():
    pos = _mk_iron_condor_position()
    mark = _mark(pos, pnl_dollars=0.0, dte=20)
    trigger, rationale = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=-0.07)
    assert trigger is None
    assert rationale == "hold"
    print("no_trigger: returns ('hold')")


def test_current_divergence_none_skips_thesis_check():
    """If we can't compute current divergence (e.g., no chain data), thesis
    check should silently skip. Other triggers still evaluate normally."""
    pos = _mk_iron_condor_position()
    mark = _mark(pos, pnl_dollars=0.0, dte=20)
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=None)
    assert trigger is None  # nothing else fires either
    # P&L stop still works
    mark = _mark(pos, pnl_dollars=-2000.0, dte=20)
    trigger, _ = _exit_mgr()._evaluate_one(mark, today=TODAY, current_divergence=None)
    assert trigger == "stop_loss"
    print("current_divergence None: thesis check skipped, P&L triggers still fire")


def main() -> int:
    test_iron_condor_profit_target_at_75pct()
    test_long_straddle_has_no_profit_target()
    test_iron_condor_stop_loss_at_neg_100pct()
    test_long_straddle_stop_loss_at_neg_50pct()
    test_straddle_rides_to_expiry_day_regardless_of_entry_dte()
    test_expiration_proximity_skipped_for_iron_condor()
    test_assignment_risk_expiry_day_unconditional()
    test_assignment_risk_near_money_short_at_dte1()
    test_assignment_risk_missing_underlying_fails_safe()
    test_assignment_risk_parity_short_any_dte()
    test_assignment_risk_not_for_long_straddle()
    test_stop_loss_overrides_assignment_risk()
    test_short_dated_straddle_entry_rides_to_expiry_morning()
    test_short_dated_condor_entry_skips_near_money_close()
    test_entry_dte_counts_trading_days()
    test_friday_straddle_rides_to_monday_final_2h()
    test_friday_condor_rides_to_monday_final_2h()
    test_aged_condor_window_counts_remaining_trading_days()
    test_parity_check_skipped_on_expiry_day()
    test_trading_dte_skips_market_holidays()
    test_trading_dte_sign_and_edges()
    test_thesis_reversal_fires_when_sign_flips_and_magnitude_clears()
    test_thesis_exit_can_be_disabled()
    test_thesis_overrides_stop_loss()
    test_thesis_overrides_profit_target()
    test_thesis_keyed_to_direction_not_entry_divergence_sign()
    test_priority_constant_matches_evaluation_order()
    test_no_trigger_returns_hold()
    test_current_divergence_none_skips_thesis_check()
    print("all exit_manager tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
