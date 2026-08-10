"""Tests for the fail-CLOSED earnings exit: no short-vol position holds
through an earnings report, even when the calendar API is down."""
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from data.earnings_calendar import EarningsCalendar
from positions.exit_manager import ExitManager, _trading_days_between
from positions.position_tracker import OpenPosition, PositionMark
from signals.signal_generator import TradeLeg

TODAY = date(2026, 8, 11)  # a Tuesday


def _condor(next_earnings: date | None = None) -> OpenPosition:
    legs = [
        TradeLeg(210.0, "call", "sell", 1, "NVDA_C210"),
        TradeLeg(210.0, "put", "sell", 1, "NVDA_P210"),
        TradeLeg(230.0, "call", "buy", 1, "NVDA_C230"),
        TradeLeg(190.0, "put", "buy", 1, "NVDA_P190"),
    ]
    return OpenPosition(
        tradier_order_id=7, symbol="NVDA", expiration=date(2026, 9, 4),
        direction="SELL", structure="iron_condor", legs=legs,
        entry_premium=2.0, entry_atm_iv=0.40, entry_predicted_iv=0.32,
        entry_divergence=-0.08, entry_horizon_lower=1, entry_horizon_upper=1,
        entry_weight_lower=1.0,
        submitted_at=datetime(2026, 8, 3, 16, 0, tzinfo=timezone.utc),
        next_earnings_date=next_earnings,
    )


def _straddle() -> OpenPosition:
    legs = [
        TradeLeg(100.0, "call", "buy", 1, "AAPL_C100"),
        TradeLeg(100.0, "put", "buy", 1, "AAPL_P100"),
    ]
    return OpenPosition(
        tradier_order_id=8, symbol="AAPL", expiration=date(2026, 9, 4),
        direction="BUY", structure="straddle", legs=legs,
        entry_premium=4.0, entry_atm_iv=0.27, entry_predicted_iv=0.42,
        entry_divergence=0.15, entry_horizon_lower=1, entry_horizon_upper=1,
        entry_weight_lower=1.0,
        submitted_at=datetime(2026, 8, 3, 16, 0, tzinfo=timezone.utc),
    )


def _mark(pos, *, pnl=0.0, dte=20) -> PositionMark:
    return PositionMark(
        position=pos, current_legs=[], close_cash_flow=0, cost_to_close=0,
        pnl_dollars=pnl, pnl_pct_of_entry_premium=pnl / (pos.entry_premium * 100),
        pnl_pct_of_max=float("nan"), delta=0, gamma=0, theta=0, vega=0,
        dte=dte, underlying_price=210.0,
    )


def _mgr(calendar=None, order_log=None, buffer_td=1) -> ExitManager:
    return ExitManager(
        position_tracker=mock.MagicMock(),
        order_manager=mock.MagicMock(),
        earnings_calendar=calendar,
        earnings_exit_buffer_trading_days=buffer_td,
        order_log=order_log,
    )


def _healthy_calendar(tmp: str, rows) -> EarningsCalendar:
    cal = EarningsCalendar(Path(tmp) / "earnings.db", api_key=None)
    cal._seed_for_testing(rows, today=TODAY)  # sets last_refresh_date=TODAY
    return cal


def test_trading_days_between():
    fri, mon, wed = date(2026, 8, 7), date(2026, 8, 10), date(2026, 8, 12)
    assert _trading_days_between(fri, mon) == 0        # weekend only
    assert _trading_days_between(mon, wed) == 1        # Tuesday
    assert _trading_days_between(mon, mon) == 0
    assert _trading_days_between(mon, date(2026, 8, 17)) == 4  # Tue-Fri
    print("trading_days: weekend-aware arithmetic verified")


def test_short_vol_closes_before_earnings():
    """Earnings Wednesday, today Tuesday (0 trading days between) → close."""
    with tempfile.TemporaryDirectory() as tmp:
        cal = _healthy_calendar(tmp, [("NVDA", date(2026, 8, 12))])
        trigger, rationale = _mgr(cal)._evaluate_one(
            _mark(_condor()), current_divergence=None, today=TODAY,
        )
        cal.close()
    assert trigger == "earnings_risk", trigger
    assert "2026-08-12" in rationale
    print(f"earnings_close: {rationale}")


def test_far_earnings_holds():
    """Earnings 6 trading days out → hold (buffer only demands 1)."""
    with tempfile.TemporaryDirectory() as tmp:
        cal = _healthy_calendar(tmp, [("NVDA", date(2026, 8, 19))])
        trigger, _ = _mgr(cal)._evaluate_one(
            _mark(_condor()), current_divergence=None, today=TODAY,
        )
        cal.close()
    assert trigger is None
    print("earnings_far: 6 trading days out holds")


def test_friday_closes_before_monday_earnings():
    friday = date(2026, 8, 7)
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "e.db", api_key=None)
        cal._seed_for_testing([("NVDA", date(2026, 8, 10))], today=friday)
        trigger, _ = _mgr(cal)._evaluate_one(
            _mark(_condor()), current_divergence=None, today=friday,
        )
        cal.close()
    assert trigger == "earnings_risk"
    print("weekend: Friday closes ahead of Monday earnings")


def test_long_straddle_exempt():
    """No short legs → the rule doesn't apply (long vol WANTS the event)."""
    with tempfile.TemporaryDirectory() as tmp:
        cal = _healthy_calendar(tmp, [("AAPL", date(2026, 8, 12))])
        trigger, _ = _mgr(cal)._evaluate_one(
            _mark(_straddle()), current_divergence=None, today=TODAY,
        )
        cal.close()
    assert trigger is None
    print("long_straddle: exempt from the earnings exit")


def test_earnings_after_expiration_ignored():
    with tempfile.TemporaryDirectory() as tmp:
        cal = _healthy_calendar(tmp, [("NVDA", date(2026, 9, 10))])  # exp 9/4
        trigger, _ = _mgr(cal)._evaluate_one(
            _mark(_condor()), current_divergence=None, today=TODAY,
        )
        cal.close()
    assert trigger is None
    print("post_expiry: report after expiration can't hurt this position")


def test_stale_calendar_uses_stored_date_fail_closed():
    """Calendar last refreshed 10 days ago → unhealthy. The STORED date on
    the position decides — the position still closes."""
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "e.db", api_key=None)
        cal._seed_for_testing([], today=TODAY - timedelta(days=10))
        pos = _condor(next_earnings=date(2026, 8, 12))
        trigger, rationale = _mgr(cal)._evaluate_one(
            _mark(pos), current_divergence=None, today=TODAY,
        )
        cal.close()
    assert trigger == "earnings_risk", trigger
    print(f"fail_closed: stale calendar, stored date closes — {rationale}")


def test_stale_calendar_no_stored_date_flags_manual_review():
    """Unhealthy calendar and NO stored date: cannot verify safety — hold
    (other triggers still run) but log a manual-review ERROR every cycle."""
    import logging

    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "e.db", api_key=None)
        cal._seed_for_testing([], today=TODAY - timedelta(days=10))
        with mock.patch.object(
            logging.getLogger("positions.exit_manager"), "error"
        ) as err:
            trigger, _ = _mgr(cal)._evaluate_one(
                _mark(_condor(next_earnings=None)), current_divergence=None,
                today=TODAY,
            )
        cal.close()
    assert trigger is None
    assert err.called and "MANUAL REVIEW" in err.call_args[0][0]
    print("manual_review: stale calendar + no stored date → loud error, held")


def test_healthy_calendar_refreshes_stored_date():
    """A healthy read that disagrees with the stored date updates the order
    log (report dates move)."""
    order_log = mock.MagicMock()
    with tempfile.TemporaryDirectory() as tmp:
        cal = _healthy_calendar(tmp, [("NVDA", date(2026, 8, 28))])
        pos = _condor(next_earnings=date(2026, 8, 26))  # stale stored value
        trigger, _ = _mgr(cal, order_log=order_log)._evaluate_one(
            _mark(pos), current_divergence=None, today=TODAY,
        )
        cal.close()
    assert trigger is None  # 8/28 is far out
    order_log.update_next_earnings_date.assert_called_once_with(
        7, date(2026, 8, 28),
    )
    print("refresh: healthy read updates the stored date")


def test_healthy_calendar_never_erases_stored_date():
    """Reviewer-flagged critical: a healthy calendar that doesn't list the
    symbol must NOT overwrite the stored date with None — and the stored
    future date still drives the exit."""
    order_log = mock.MagicMock()
    with tempfile.TemporaryDirectory() as tmp:
        # healthy cache with content, but for a DIFFERENT symbol
        cal = _healthy_calendar(tmp, [("MSFT", date(2026, 9, 1))])
        pos = _condor(next_earnings=date(2026, 8, 12))  # NVDA reports tomorrow
        trigger, _ = _mgr(cal, order_log=order_log)._evaluate_one(
            _mark(pos), current_divergence=None, today=TODAY,
        )
        cal.close()
    assert trigger == "earnings_risk", trigger
    order_log.update_next_earnings_date.assert_not_called()
    print("no_erase: unlisted symbol keeps its stored date and still closes")


def test_emptied_cache_counts_as_unhealthy():
    """An HTTP-200 empty payload wipes the cache but stamps a fresh
    last_refresh_date; the health check must see the empty cache and fall
    back to the stored date."""
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "e.db", api_key=None)
        cal._seed_for_testing([], today=TODAY)  # fresh refresh, ZERO rows
        assert cal.cached_row_count() == 0
        pos = _condor(next_earnings=date(2026, 8, 12))
        trigger, _ = _mgr(cal)._evaluate_one(
            _mark(pos), current_divergence=None, today=TODAY,
        )
        cal.close()
    assert trigger == "earnings_risk", trigger
    print("empty_cache: fresh-but-empty calendar treated as unhealthy, stored date closes")


def test_manual_review_fires_alert_callback_once():
    alerts: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        cal = EarningsCalendar(Path(tmp) / "e.db", api_key=None)
        cal._seed_for_testing([], today=TODAY - timedelta(days=10))
        mgr = ExitManager(
            position_tracker=mock.MagicMock(), order_manager=mock.MagicMock(),
            earnings_calendar=cal, alert_cb=alerts.append,
        )
        for _ in range(3):  # three cycles, one alert
            trigger, _ = mgr._evaluate_one(
                _mark(_condor(next_earnings=None)), current_divergence=None,
                today=TODAY,
            )
        cal.close()
    assert trigger is None
    assert len(alerts) == 1 and "MANUAL REVIEW" in alerts[0]
    print("alert_cb: manual review reaches Slack once per position per day")


def test_priority_stop_loss_beats_earnings_beats_assignment():
    with tempfile.TemporaryDirectory() as tmp:
        cal = _healthy_calendar(tmp, [("NVDA", date(2026, 8, 12))])
        mgr = _mgr(cal)
        # deep drawdown + imminent earnings → stop_loss label
        trigger, _ = mgr._evaluate_one(
            _mark(_condor(), pnl=-1500.0), current_divergence=None, today=TODAY,
        )
        assert trigger == "stop_loss"
        # imminent earnings at dte=1 with NaN underlying (assignment would
        # fail safe) → earnings_risk label wins the priority
        m = _mark(_condor(), dte=1)
        m = PositionMark(**{**m.__dict__, "underlying_price": float("nan")})
        trigger, _ = mgr._evaluate_one(m, current_divergence=None, today=TODAY)
        assert trigger == "earnings_risk", trigger
        cal.close()
    print("priority: stop_loss > earnings_risk > assignment_risk")


def test_order_log_round_trip_of_next_earnings_date():
    """record_submission stores the stamped date; open_unclosed_positions
    surfaces it; update_next_earnings_date refreshes it."""
    from execution.order_log import OrderLog
    from signals.signal_generator import TradeSignal

    sig = TradeSignal(
        symbol="NVDA", expiration=date(2026, 9, 4), dte=24,
        horizon_lower=1, horizon_upper=1, weight_lower=1.0, direction="SELL",
        underlying_price=210.0, atm_iv=0.40, predicted_iv_equivalent=0.32,
        divergence=-0.08, cross_sectional_z=-2.0, time_series_z=None,
        liquidity_score=1.0,
        legs=[TradeLeg(210.0, "call", "sell", 1, "NVDA_C210")],
        is_actionable=True, vrp_z=1.9,
    )
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        log.record_submission(
            signal=sig, fingerprint="fp", structure="iron_condor",
            submitted_price=2.0, order_id=42,
            submitted_at=datetime(2026, 8, 11, 15, 0, tzinfo=timezone.utc),
            next_earnings_date=date(2026, 8, 26),
        )
        log.update_terminal_state(42, "filled", 2.0,
                                  datetime(2026, 8, 11, 15, 1, tzinfo=timezone.utc))
        row = log.open_unclosed_positions()[0]
        assert row["next_earnings_date"] == "2026-08-26"
        log.update_next_earnings_date(42, date(2026, 8, 28))
        row = log.open_unclosed_positions()[0]
        assert row["next_earnings_date"] == "2026-08-28"
        log.close()
    print("order_log: next_earnings_date stored, surfaced, refreshed")


def main() -> int:
    test_trading_days_between()
    test_short_vol_closes_before_earnings()
    test_far_earnings_holds()
    test_friday_closes_before_monday_earnings()
    test_long_straddle_exempt()
    test_earnings_after_expiration_ignored()
    test_stale_calendar_uses_stored_date_fail_closed()
    test_stale_calendar_no_stored_date_flags_manual_review()
    test_healthy_calendar_refreshes_stored_date()
    test_healthy_calendar_never_erases_stored_date()
    test_emptied_cache_counts_as_unhealthy()
    test_manual_review_fires_alert_callback_once()
    test_priority_stop_loss_beats_earnings_beats_assignment()
    test_order_log_round_trip_of_next_earnings_date()
    print("all earnings_exit tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
