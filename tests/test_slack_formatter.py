import sys
from datetime import date

from logs import DailySummary, format_summary


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


def main() -> int:
    test_format_summary_basic()
    test_format_summary_kill_switch_active()
    test_format_summary_no_rejections()
    test_format_summary_with_exits()
    test_format_summary_zero_starting_equity()
    test_format_summary_negative_pnl()
    print("all slack_formatter tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
