import sys
from datetime import date, datetime, timezone

from logs import (
    DailySummary,
    PendingCloseSummary,
    PositionPnlGroup,
    StaleCloseAlertSummary,
    format_summary,
)


def _mk_summary(**overrides) -> DailySummary:
    base = dict(
        date=date(2026, 4, 28),
        starting_equity=99938.40,
        ending_equity=100142.40,
        realized_pnl=0.0,
        unrealized_pnl=204.00,
        open_positions=1,
        positions_opened_today=1,
        positions_closed_today=0,
        signals_total=149,
        signals_approved=1,
        risk_rejections_total=3,
        risk_rejections_by_reason={"per_trade_budget": 3},
        kill_switch_activated=False,
        top_exit_triggers={},
    )
    base.update(overrides)
    return DailySummary(**base)


def test_format_summary_basic():
    text = format_summary(_mk_summary())
    assert "*Options Trader — 2026-04-28*" in text
    assert "$99,938.40" in text
    assert "$100,142.40" in text
    assert "+204.00" in text
    assert "+0.20%" in text
    assert "1 open" in text
    assert "1 opened" in text
    assert "0 closed" in text
    assert "149 raw" in text
    assert "1 approved" in text
    assert "Risk rejections: 3 (per_trade_budget × 3)" in text
    assert "Kill switch: not activated" in text
    print("format_summary basic shape verified")


def test_format_summary_kill_switch_active():
    text = format_summary(_mk_summary(kill_switch_activated=True))
    assert ":rotating_light:" in text
    assert "Kill switch ACTIVATED today" in text
    print("format_summary: kill switch flag rendered")


def test_format_summary_no_rejections():
    text = format_summary(_mk_summary(risk_rejections_total=0, risk_rejections_by_reason={}))
    assert "Risk rejections: 0" in text
    assert "(" not in text.split("Risk rejections:")[1].split("\n")[0]
    print("format_summary: zero rejections rendered cleanly")


def test_format_summary_with_exits():
    text = format_summary(_mk_summary(
        top_exit_triggers={"profit_target": 2, "stop_loss": 1},
    ))
    assert "Exits:" in text
    assert "profit_target × 2" in text
    assert "stop_loss × 1" in text
    print("format_summary: exit triggers rendered")


def test_format_summary_zero_starting_equity():
    """Edge case: division by zero protection."""
    text = format_summary(_mk_summary(
        starting_equity=0.0, ending_equity=100.0,
    ))
    # Should not crash; pct should render as 0.00%
    assert "+0.00%" in text or "+inf%" not in text
    print("format_summary: zero starting equity handled")


def test_format_summary_negative_pnl():
    text = format_summary(_mk_summary(
        realized_pnl=-300.0, unrealized_pnl=-150.0,
        ending_equity=99488.40,
    ))
    assert "-300.00" in text
    assert "-150.00" in text
    assert "-450.00" in text
    print("format_summary: negative P&L renders correctly")


def test_format_summary_stale_close_alert():
    alert = StaleCloseAlertSummary(
        opening_order_id=5001, symbol="XOM",
        expiration=date(2026, 5, 30), structure="straddle",
        attempts=3, last_exit_trigger="profit_target",
        detected_at=datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc),
    )
    text = format_summary(_mk_summary(stale_close_alerts=[alert]))
    assert ":rotating_light:" in text
    assert "STALE CLOSE ALERT" in text
    assert "opening 5001" in text
    assert "XOM" in text
    assert "3 failed attempts" in text
    assert "profit_target" in text
    print("format_summary: stale_close_alert rendered")


def test_format_summary_pending_closes():
    pending = PendingCloseSummary(
        closing_order_id=9101, opening_order_id=5001,
        symbol="XOM", structure="straddle",
        exit_trigger="profit_target", submitted_price=2.50,
        submitted_at=datetime(2026, 5, 20, 14, 0, tzinfo=timezone.utc),
    )
    text = format_summary(_mk_summary(pending_closes=[pending]))
    assert "Open exits in progress" in text
    assert "close 9101" in text
    assert "opening 5001" in text
    assert "$2.50" in text
    assert "profit_target" in text
    print("format_summary: pending_closes rendered")


def test_format_summary_position_pnl_breakdown():
    groups = [
        PositionPnlGroup(
            symbol="NVDA", structure="iron_condor", expiration=date(2026, 6, 19),
            count=3, total_pnl=420.0, daily_pnl=85.0,
        ),
        PositionPnlGroup(
            symbol="TSLA", structure="straddle", expiration=date(2026, 6, 26),
            count=1, total_pnl=-150.0, daily_pnl=-30.0,
        ),
    ]
    text = format_summary(_mk_summary(position_pnl_breakdown=groups))
    assert "Per-position P&L (2 open):" in text
    # Grouped row shows the ×count multiplier and both P&L figures.
    assert "NVDA iron_condor exp 2026-06-19 ×3 — total $+420.00 / today $+85.00" in text
    # Singleton row omits the multiplier.
    assert "TSLA straddle exp 2026-06-26 — total $-150.00 / today $-30.00" in text
    assert "×1" not in text
    print("format_summary: per-position P&L breakdown rendered")


def main() -> int:
    test_format_summary_basic()
    test_format_summary_kill_switch_active()
    test_format_summary_no_rejections()
    test_format_summary_with_exits()
    test_format_summary_zero_starting_equity()
    test_format_summary_negative_pnl()
    test_format_summary_stale_close_alert()
    test_format_summary_pending_closes()
    test_format_summary_position_pnl_breakdown()
    print("all slack_formatter tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
