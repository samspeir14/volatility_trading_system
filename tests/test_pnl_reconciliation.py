"""Regression tests for the daily P&L reconciliation bug.

The bug: portfolio_state.snapshot() set today_unrealized_pnl to the *lifetime*
since-entry unrealized P&L summed across all open positions, not the change
since the start of today. With 16 positions held flat over a week and zero
realized P&L, the reported "P&L today" was effectively a slow-moving
cumulative number while equity swung by thousands intraday. On 5/12 the
signs even disagreed: equity +$8.3k, reported P&L -$1.3k.

Fix: today_unrealized = (equity - starting_equity) - today_realized. By
cash conservation (equity = cash + position MV), today's equity delta is
the only "P&L today" that reconciles regardless of opens/closes.

Tests in this file:
  1. Replays the 6 sessions from the bug report through the real
     DailySummaryBuilder and asserts the reconciliation drift is within
     the max($50, 0.1% × equity) tolerance.
  2. Drives the real PortfolioStateBuilder.snapshot() with a synthetic
     5/12 scenario (cumulative -$1,302, intraday +$8,331) and verifies
     today_unrealized matches the equity delta, not the cumulative.
  3. Confirms the reconciliation guard *flags* (logs error) when the
     identity breaks — the guard that should have caught the bug on day 1.
"""
import asyncio
import logging
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

from data.market_data import ScanResult, TickerSnapshot
from execution import OrderLog
from logs import DailySummaryBuilder
from logs.daily_summary import _check_reconciliation
from positions.position_tracker import OpenPosition, PositionMark, PositionTracker
from risk import DailyKillSwitch, RiskRejectionLog
from risk.portfolio_state import PortfolioStateBuilder
from signals import DivergenceHistory
from signals.signal_generator import TradeLeg


# Six sessions from the bug report. Realized P&L = $0 every day, 16 positions
# held constant. The "old_reported_pnl" column is what the buggy code emitted
# (cumulative since-entry unrealized). "expected_total_pnl" under the fix is
# simply the equity delta.
SESSIONS = [
    # date, equity_start, equity_end, old_reported_pnl
    (date(2026, 5, 11), 101_310.40, 102_188.70, -1_428.0),
    (date(2026, 5, 12), 102_376.20, 110_707.70, -1_302.0),
    (date(2026, 5, 13), 110_456.20, 113_033.70, -1_936.0),
    (date(2026, 5, 14), 113_303.20, 112_576.20, -1_961.0),
    (date(2026, 5, 15), 111_629.20, 107_745.20, -1_409.0),
    (date(2026, 5, 18), 109_583.20, 111_036.20,  1_344.0),
]


def _mk_summary_snapshot(*, starting_equity, ending_equity,
                         today_realized, today_unrealized):
    snap = mock.MagicMock()
    snap.starting_equity_today = starting_equity
    snap.equity = ending_equity
    snap.today_realized_pnl = today_realized
    snap.today_unrealized_pnl = today_unrealized
    snap.today_total_pnl = today_realized + today_unrealized
    snap.open_positions = []
    return snap


def test_old_formula_breaches_reconciliation_on_5_sessions():
    """Sanity check that under the OLD buggy formula (today_unrealized =
    cumulative since-entry), 5 of the 6 sessions exceed the tolerance.
    This is the bug the reconciliation guard should have caught."""
    breaches = []
    for d, eq_start, eq_end, old_pnl in SESSIONS:
        equity_delta = eq_end - eq_start
        drift = abs(equity_delta - old_pnl)
        tolerance = max(50.0, 0.001 * eq_end)
        if drift > tolerance:
            breaches.append((d, equity_delta, old_pnl, drift, tolerance))
    assert len(breaches) == 5, (
        f"expected 5/6 sessions to breach under old code, saw {len(breaches)}"
    )
    print(f"old_formula: {len(breaches)}/6 sessions breach reconciliation guard")
    for d, dE, pnl, drift, tol in breaches:
        print(f"  {d}: ΔE=${dE:+,.2f} pnl=${pnl:+,.2f} drift=${drift:,.2f} tol=${tol:,.2f}")


def test_new_formula_reconciles_all_6_sessions():
    """Under the FIX, today_unrealized = (equity - starting_equity) - realized.
    Drift must be within tolerance for every session."""
    with tempfile.TemporaryDirectory() as tmp:
        order_log = OrderLog(Path(tmp) / "orders.db")
        div_history = DivergenceHistory(Path(tmp) / "div.db")
        risk_log = RiskRejectionLog(Path(tmp) / "risk.db")
        kill_switch = DailyKillSwitch(Path(tmp) / "ks.db")
        builder = DailySummaryBuilder(order_log, div_history, risk_log, kill_switch)

        results = []
        for d, eq_start, eq_end, _ in SESSIONS:
            today_realized = 0.0
            today_unrealized = (eq_end - eq_start) - today_realized
            snap = _mk_summary_snapshot(
                starting_equity=eq_start, ending_equity=eq_end,
                today_realized=today_realized,
                today_unrealized=today_unrealized,
            )
            summary = builder.build(d, snap)
            equity_delta = summary.equity_change
            drift = abs(equity_delta - summary.total_pnl)
            tolerance = max(50.0, 0.001 * summary.ending_equity)
            results.append((d, equity_delta, summary.total_pnl, drift, tolerance))
            assert drift <= tolerance, (
                f"{d}: drift ${drift:.2f} > tolerance ${tolerance:.2f}"
            )

        for c in (order_log, div_history, risk_log, kill_switch):
            c.close()

    print("new_formula: all 6 sessions reconcile within tolerance")
    for d, dE, pnl, drift, tol in results:
        print(f"  {d}: ΔE=${dE:+,.2f} pnl=${pnl:+,.2f} drift=${drift:.2f} tol=${tol:.2f} ✓")


def _mk_leg(strike: float, opt_type: str, side: str) -> TradeLeg:
    return TradeLeg(strike=strike, option_type=opt_type, side=side,
                    quantity=1, contract_symbol=f"X{strike:.0f}{opt_type[0].upper()}{side[0].upper()}")


def test_snapshot_today_unrealized_tracks_equity_not_lifetime_pnl():
    """End-to-end drive of PortfolioStateBuilder.snapshot() with a synthetic
    5/12-shape scenario: equity moved +$8,331 today, but the cumulative
    since-entry unrealized of the 16 positions is -$1,302. The fix says
    today_unrealized = +$8,331, not -$1,302."""
    eq_start = 102_376.20
    eq_end = 110_707.70

    # Fake Tradier balances response
    fake_client = mock.AsyncMock()
    fake_client.get_balances.return_value = {
        "total_equity": eq_end,
        "account_type": "margin",
        "current_requirement": 5000.0,
        "margin": {"option_buying_power": 50000.0},
    }

    # Fake position tracker that returns marks whose lifetime P&L sums to -$1,302
    fake_tracker = mock.MagicMock(spec=PositionTracker)

    async def _no_positions():
        return []  # snapshot only iterates positions for exposure/sector counts
    fake_tracker.list_open_positions = mock.AsyncMock(return_value=[])

    # Synthesize 16 PositionMarks summing to -$1,302 of cumulative unrealized.
    # The marks themselves don't drive today_unrealized under the fix — they
    # represent lifetime P&L only. Use a dummy position to satisfy the dataclass.
    dummy_pos = OpenPosition(
        tradier_order_id=1, symbol="AAPL", expiration=date(2026, 6, 19),
        direction="SELL", structure="iron_condor",
        legs=[_mk_leg(100, "call", "sell"), _mk_leg(110, "call", "buy"),
              _mk_leg(90, "put", "sell"), _mk_leg(80, "put", "buy")],
        entry_premium=2.50, entry_atm_iv=0.30, entry_predicted_iv=0.25,
        entry_divergence=-0.05, entry_horizon_lower=21, entry_horizon_upper=21,
        entry_weight_lower=1.0,
        submitted_at=datetime(2026, 4, 15, 16, 0, tzinfo=timezone.utc),
    )
    per_mark_pnl = -1302.0 / 16
    marks = [
        PositionMark(
            position=dummy_pos, current_legs=[], close_cash_flow=0.0,
            cost_to_close=0.0, pnl_dollars=per_mark_pnl,
            pnl_pct_of_entry_premium=0.0, pnl_pct_of_max=float("nan"),
            delta=0, gamma=0, theta=0, vega=0, dte=30,
        )
        for _ in range(16)
    ]
    fake_tracker.mark_to_market = mock.MagicMock(return_value=marks)
    fake_tracker.portfolio_greeks = mock.MagicMock(return_value={
        "delta": 0, "gamma": 0, "theta": 0, "vega": 0,
    })
    # PortfolioStateBuilder calls PositionTracker.portfolio_greeks as a class
    # method via the type, so patch that too.
    with mock.patch.object(PositionTracker, "portfolio_greeks",
                           return_value={"delta": 0, "gamma": 0, "theta": 0, "vega": 0}):

        with tempfile.TemporaryDirectory() as tmp:
            order_log = OrderLog(Path(tmp) / "orders.db")
            kill_switch = DailyKillSwitch(Path(tmp) / "ks.db")
            # Pre-seed starting_equity so it's not the same as ending equity
            today = date(2026, 5, 12)
            kill_switch.get_starting_equity(today, eq_start)

            builder = PortfolioStateBuilder(
                client=fake_client, order_log=order_log,
                position_tracker=fake_tracker, watchlist=[],
                kill_switch=kill_switch,
            )

            scan = ScanResult(
                fetched_at=datetime(2026, 5, 12, 20, 30, tzinfo=timezone.utc),
                snapshots={},
            )
            snapshot = asyncio.run(builder.snapshot(scan))

            expected_total = eq_end - eq_start  # $8,331.50, realized=0
            assert abs(snapshot.today_total_pnl - expected_total) < 0.01, (
                f"today_total_pnl=${snapshot.today_total_pnl:+,.2f} "
                f"!= equity_delta=${expected_total:+,.2f} "
                f"(cumulative-from-marks bug would give ~-$1,302)"
            )
            # And specifically: today_unrealized must NOT be the cumulative -$1,302
            assert snapshot.today_unrealized_pnl > 1000, (
                f"today_unrealized=${snapshot.today_unrealized_pnl:+,.2f} — "
                f"looks like the cumulative-marks bug is back"
            )

            for c in (order_log, kill_switch):
                c.close()

    print(f"snapshot_e2e: today_total_pnl=${snapshot.today_total_pnl:+,.2f} "
          f"matches equity_delta=${expected_total:+,.2f} ✓")


def test_reconciliation_guard_logs_error_on_breach(caplog=None):
    """If the identity breaks (e.g., a future bug reintroduces the cumulative
    sum), the EOD guard must log a clear error. This is the day-1 alarm."""
    # Construct an old-bug-shaped summary: equity moved +$8,331 but reported
    # P&L is the cumulative -$1,302. The guard should flag it.
    summary = mock.MagicMock()
    summary.starting_equity = 102_376.20
    summary.ending_equity = 110_707.70
    summary.realized_pnl = 0.0
    summary.unrealized_pnl = -1302.0
    summary.total_pnl = -1302.0

    logger = logging.getLogger("logs.daily_summary")
    records = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[assignment]
    handler.setLevel(logging.ERROR)
    logger.addHandler(handler)
    try:
        _check_reconciliation(summary)
    finally:
        logger.removeHandler(handler)

    assert any("reconciliation drift" in r.getMessage().lower() for r in records), (
        f"expected reconciliation-drift error log, saw: {[r.getMessage() for r in records]}"
    )
    print("reconciliation_guard: emits ERROR when identity breaks ✓")


def test_reconciliation_guard_silent_within_tolerance():
    """Float-level rounding within $5 tolerance: no log."""
    summary = mock.MagicMock()
    summary.starting_equity = 100_000.0
    summary.ending_equity = 100_500.0
    summary.realized_pnl = 200.0
    summary.unrealized_pnl = 297.50  # total=$497.50, equity_delta=$500, drift=$2.50
    summary.total_pnl = 497.50

    logger = logging.getLogger("logs.daily_summary")
    records = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[assignment]
    handler.setLevel(logging.ERROR)
    logger.addHandler(handler)
    try:
        _check_reconciliation(summary)
    finally:
        logger.removeHandler(handler)

    assert not any("reconciliation drift" in r.getMessage().lower() for r in records), (
        f"unexpected error log: {[r.getMessage() for r in records]}"
    )
    print("reconciliation_guard: silent within tolerance ✓")


def main() -> int:
    test_old_formula_breaches_reconciliation_on_5_sessions()
    test_new_formula_reconciles_all_6_sessions()
    test_snapshot_today_unrealized_tracks_equity_not_lifetime_pnl()
    test_reconciliation_guard_logs_error_on_breach()
    test_reconciliation_guard_silent_within_tolerance()
    print("all pnl reconciliation tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
