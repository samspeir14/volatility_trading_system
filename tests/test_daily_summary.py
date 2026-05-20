import asyncio
import json
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

from config import Settings
from execution import OrderLog, OrderManager
from logs import DailySummary, DailySummaryBuilder
from positions.position_tracker import OpenPosition, PositionMark
from risk import DailyKillSwitch, RiskRejectionLog
from signals import DivergenceHistory
from signals.signal_generator import TradeLeg


def _mk_snapshot(*, today_realized=0.0, today_unrealized=0.0, equity=100000.0,
                 open_positions=()):
    snap = mock.MagicMock()
    snap.starting_equity_today = 100000.0
    snap.equity = equity
    snap.today_realized_pnl = today_realized
    snap.today_unrealized_pnl = today_unrealized
    snap.today_total_pnl = today_realized + today_unrealized
    snap.open_positions = list(open_positions)
    return snap


def test_summary_with_no_activity():
    """Empty SQLite state → all counts zero, no kill switch."""
    with tempfile.TemporaryDirectory() as tmp:
        order_log = OrderLog(Path(tmp) / "orders.db")
        div_history = DivergenceHistory(Path(tmp) / "div.db")
        risk_log = RiskRejectionLog(Path(tmp) / "risk.db")
        kill_switch = DailyKillSwitch(Path(tmp) / "ks.db")

        builder = DailySummaryBuilder(order_log, div_history, risk_log, kill_switch)
        snap = _mk_snapshot()
        summary = builder.build(date(2026, 4, 28), snap)

        assert summary.starting_equity == 100000.0
        assert summary.ending_equity == 100000.0
        assert summary.realized_pnl == 0.0
        assert summary.signals_total == 0
        assert summary.signals_approved == 0
        assert summary.positions_opened_today == 0
        assert summary.positions_closed_today == 0
        assert summary.risk_rejections_total == 0
        assert summary.kill_switch_activated is False
        assert summary.top_exit_triggers == {}

        for c in (order_log, div_history, risk_log, kill_switch):
            c.close()
    print("no_activity: all zeros, no kill switch")


def test_summary_with_filled_positions():
    """Insert a few orders and verify counts."""
    with tempfile.TemporaryDirectory() as tmp:
        order_log = OrderLog(Path(tmp) / "orders.db")
        div_history = DivergenceHistory(Path(tmp) / "div.db")
        risk_log = RiskRejectionLog(Path(tmp) / "risk.db")
        kill_switch = DailyKillSwitch(Path(tmp) / "ks.db")

        # Inject 3 opened-today + 1 closed-today + 1 rejection directly
        today = date(2026, 4, 28)
        today_iso = "2026-04-28T15:00:00+00:00"
        for oid in (1001, 1002, 1003):
            order_log._conn.execute(
                "INSERT INTO submitted_orders ("
                "tradier_order_id, fingerprint, submitted_at, symbol, expiration, "
                "direction, structure, horizon_lower, horizon_upper, weight_lower, "
                "underlying_price_at_signal, atm_iv_at_signal, predicted_iv_at_signal, "
                "divergence_at_signal, cross_sectional_z, time_series_z, "
                "submitted_price, legs_json, final_status"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (oid, f"fp{oid}", today_iso, "AAPL", "2026-05-15",
                 "BUY", "straddle", 10, 21, 0.5, 100.0, 0.30, 0.40, 0.10, 2.0, None,
                 5.00, json.dumps([]), "filled"),
            )
        # One closed today with $200 realized P&L
        order_log._conn.execute(
            "UPDATE submitted_orders SET closing_order_id = ?, closed_at = ?, "
            "exit_trigger = ?, realized_pnl = ? WHERE tradier_order_id = ?",
            (9001, today_iso, "profit_target", 200.0, 1001),
        )
        order_log._conn.commit()

        # One rejection
        risk_log._conn.execute(
            "INSERT INTO risk_rejections (rejected_at, symbol, expiration, "
            "direction, reasons_json) VALUES (?, ?, ?, ?, ?)",
            (today_iso, "META", "2026-05-15", "SELL",
             json.dumps(["max_loss_per_contract $1668 > per-trade budget $999"])),
        )
        risk_log._conn.commit()

        # Insert one signal in divergence_log
        div_history._conn.execute(
            "INSERT INTO divergence_log "
            "(scan_date, symbol, expiration, horizon_lower, horizon_upper, weight_lower, "
            "predicted_iv_equivalent, atm_iv, divergence, underlying_price) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (today.isoformat(), "AAPL", "2026-05-15", 10, 21, 0.5, 0.40, 0.30, 0.10, 100.0),
        )
        div_history._conn.commit()

        builder = DailySummaryBuilder(order_log, div_history, risk_log, kill_switch)
        snap = _mk_snapshot(today_realized=200.0, equity=100200.0)
        summary = builder.build(today, snap)

        assert summary.positions_opened_today == 3
        assert summary.positions_closed_today == 1
        assert summary.realized_pnl == 200.0
        assert summary.signals_total == 1
        assert summary.signals_approved == 3   # = positions_opened_today
        assert summary.risk_rejections_total == 1
        assert summary.top_exit_triggers == {"profit_target": 1}
        assert summary.kill_switch_activated is False

        for c in (order_log, div_history, risk_log, kill_switch):
            c.close()
    print("filled_positions: 3 opened, 1 closed @ $200, 1 rejection, 1 signal")


def test_summary_with_kill_switch():
    with tempfile.TemporaryDirectory() as tmp:
        order_log = OrderLog(Path(tmp) / "orders.db")
        div_history = DivergenceHistory(Path(tmp) / "div.db")
        risk_log = RiskRejectionLog(Path(tmp) / "risk.db")
        kill_switch = DailyKillSwitch(Path(tmp) / "ks.db")

        today = date(2026, 4, 28)
        kill_switch.trigger(today, "test", -3500.0, 100000.0)

        builder = DailySummaryBuilder(order_log, div_history, risk_log, kill_switch)
        snap = _mk_snapshot(today_realized=-3500.0, equity=96500.0)
        summary = builder.build(today, snap)

        assert summary.kill_switch_activated is True
        for c in (order_log, div_history, risk_log, kill_switch):
            c.close()
    print("kill_switch: flagged in summary")


def test_summary_rejection_categories():
    """Multiple rejection reasons get bucketed correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        order_log = OrderLog(Path(tmp) / "orders.db")
        div_history = DivergenceHistory(Path(tmp) / "div.db")
        risk_log = RiskRejectionLog(Path(tmp) / "risk.db")
        kill_switch = DailyKillSwitch(Path(tmp) / "ks.db")

        today = date(2026, 4, 28)
        today_iso = "2026-04-28T15:00:00+00:00"
        # Insert 4 rejections with different reasons
        for reasons in [
            ["max_loss_per_contract $X > per-trade budget $Y"],
            ["NVDA exposure $X would exceed cap $Y"],
            ["tech sector already has 3 positions (cap 3)"],
            ["max_loss_per_contract $X > per-trade budget $Y"],  # second of same kind
        ]:
            risk_log._conn.execute(
                "INSERT INTO risk_rejections (rejected_at, symbol, expiration, "
                "direction, reasons_json) VALUES (?, ?, ?, ?, ?)",
                (today_iso, "X", "2026-05-15", "BUY", json.dumps(reasons)),
            )
        risk_log._conn.commit()

        builder = DailySummaryBuilder(order_log, div_history, risk_log, kill_switch)
        snap = _mk_snapshot()
        summary = builder.build(today, snap)

        assert summary.risk_rejections_total == 4
        assert summary.risk_rejections_by_reason["per_trade_budget"] == 2
        assert summary.risk_rejections_by_reason["ticker_exposure"] == 1
        assert summary.risk_rejections_by_reason["sector_concentration"] == 1
        for c in (order_log, div_history, risk_log, kill_switch):
            c.close()
    print(f"rejection_categories: {summary.risk_rejections_by_reason}")


def _seed_open(log: OrderLog, *, order_id: int, direction: str, structure: str,
               entry_premium: float) -> None:
    log._conn.execute(
        "INSERT INTO submitted_orders ("
        "tradier_order_id, fingerprint, submitted_at, symbol, expiration, "
        "direction, structure, horizon_lower, horizon_upper, weight_lower, "
        "underlying_price_at_signal, atm_iv_at_signal, predicted_iv_at_signal, "
        "divergence_at_signal, cross_sectional_z, time_series_z, "
        "submitted_price, legs_json, final_status, fill_price"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (order_id, f"fp{order_id}", "2026-04-27T16:00:00+00:00",
         "AAPL", "2026-05-15", direction, structure, 5, 5, 1.0,
         100.0, 0.30, 0.40, 0.10, 2.0, None,
         entry_premium, "[]", "filled",
         entry_premium if direction == "BUY" else -entry_premium),
    )
    log._conn.commit()


def _drive_close(log: OrderLog, pos: OpenPosition, close_cash_flow: float,
                 fill_price: float, closing_order_id: int) -> None:
    """Drive a close through the real OrderManager.submit_close path so the
    realized P&L is computed by production code, not injected."""
    fake = mock.AsyncMock()
    fake.preview_order.return_value = {"order": {"status": "ok"}}
    fake.place_order.return_value = {"order": {"id": closing_order_id, "status": "pending"}}
    fake.get_order_status.return_value = {
        "order": {"id": closing_order_id, "status": "filled", "avg_fill_price": fill_price}
    }
    settings = Settings(api_key="fake", account_id="V", base_url="http://x", env="sandbox")
    mgr = OrderManager(client=fake, order_log=log, settings=settings,
                       poll_interval_seconds=0.001, poll_timeout_seconds=1.0,
                       slippage_buffer=0.0)
    mark = PositionMark(
        position=pos, current_legs=[], close_cash_flow=close_cash_flow,
        cost_to_close=abs(close_cash_flow) if close_cash_flow < 0 else 0,
        pnl_dollars=0.0, pnl_pct_of_entry_premium=0.0,
        pnl_pct_of_max=float("nan"),
        delta=0, gamma=0, theta=0, vega=0, dte=10,
    )
    result = asyncio.run(mgr.submit_close(position=pos, mark=mark, exit_trigger="profit_target"))
    assert result.status == "filled", f"close failed: {result.error}"


def test_summary_pnl_matches_equity_delta():
    """Drive real closes through submit_close() and assert that the summary's
    total P&L (realized + unrealized) agrees with the actual equity delta
    within $5. This is the regression net for the realized-P&L double-count
    bug — it would have failed when closed_today_pnl returned -$2,890 but
    real equity moved -$74."""
    with tempfile.TemporaryDirectory() as tmp:
        order_log = OrderLog(Path(tmp) / "orders.db")
        div_history = DivergenceHistory(Path(tmp) / "div.db")
        risk_log = RiskRejectionLog(Path(tmp) / "risk.db")
        kill_switch = DailyKillSwitch(Path(tmp) / "ks.db")

        # Seed two open positions, then close both. Use the real submit_close
        # path so the sign-flip bug would surface here.
        _seed_open(order_log, order_id=7001, direction="BUY",
                   structure="straddle", entry_premium=4.08)
        _seed_open(order_log, order_id=7002, direction="SELL",
                   structure="iron_condor", entry_premium=13.55)

        long_legs = [TradeLeg(100.0, "call", "buy", 1, "C"),
                     TradeLeg(100.0, "put", "buy", 1, "P")]
        long_pos = OpenPosition(
            tradier_order_id=7001, symbol="AAPL",
            expiration=date(2026, 5, 15), direction="BUY",
            structure="straddle", legs=long_legs, entry_premium=4.08,
            entry_atm_iv=0.27, entry_predicted_iv=0.42, entry_divergence=0.15,
            entry_horizon_lower=5, entry_horizon_upper=5, entry_weight_lower=1.0,
            submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        )
        ic_legs = [TradeLeg(210.0, "call", "sell", 1, "SC"),
                   TradeLeg(210.0, "put", "sell", 1, "SP"),
                   TradeLeg(230.0, "call", "buy", 1, "LC"),
                   TradeLeg(190.0, "put", "buy", 1, "LP")]
        ic_pos = OpenPosition(
            tradier_order_id=7002, symbol="AAPL",
            expiration=date(2026, 5, 15), direction="SELL",
            structure="iron_condor", legs=ic_legs, entry_premium=13.55,
            entry_atm_iv=0.40, entry_predicted_iv=0.32, entry_divergence=-0.08,
            entry_horizon_lower=21, entry_horizon_upper=21, entry_weight_lower=1.0,
            submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        )

        # Long straddle: paid $4.08, sold for $2.00 (Tradier credit fill = -2.00)
        # → realized = -$208
        _drive_close(order_log, long_pos, close_cash_flow=200.0,
                     fill_price=-2.00, closing_order_id=8001)
        # Iron condor: received $13.55, bought back $5.00 → realized = +$855
        _drive_close(order_log, ic_pos, close_cash_flow=-500.0,
                     fill_price=5.00, closing_order_id=8002)

        # The actual cash flow that hit the account: -208 + 855 = +647
        actual_realized = 647.0
        starting_equity = 100_000.0
        ending_equity = starting_equity + actual_realized  # no unrealized in this scenario

        # Build the snapshot the way portfolio_state.py does — pulling realized
        # from closed_today_pnl (the real path that exercises the bug).
        today = datetime.now(timezone.utc).date()
        snap = mock.MagicMock()
        snap.starting_equity_today = starting_equity
        snap.equity = ending_equity
        snap.today_realized_pnl = order_log.closed_today_pnl(today)
        snap.today_unrealized_pnl = 0.0
        snap.today_total_pnl = snap.today_realized_pnl + snap.today_unrealized_pnl
        snap.open_positions = []

        builder = DailySummaryBuilder(order_log, div_history, risk_log, kill_switch)
        summary = builder.build(today, snap)

        equity_delta = summary.ending_equity - summary.starting_equity
        diff = abs(summary.total_pnl - equity_delta)
        assert diff <= 5.0, (
            f"summary P&L (${summary.total_pnl:+.2f}) diverges from equity delta "
            f"(${equity_delta:+.2f}) by ${diff:.2f}; the realized-P&L "
            f"double-count bug would have produced this."
        )
        for c in (order_log, div_history, risk_log, kill_switch):
            c.close()
    print(f"summary_vs_equity_delta: total_pnl={summary.total_pnl:+.2f} "
          f"equity_delta={equity_delta:+.2f} diff={diff:.4f} ✓")


def test_summary_surfaces_stale_close_alerts_and_pending():
    """stale_close_alerts rows + pending close_attempts should both surface
    in the built summary so the daily Slack post catches them."""
    with tempfile.TemporaryDirectory() as tmp:
        order_log = OrderLog(Path(tmp) / "orders.db")
        div_history = DivergenceHistory(Path(tmp) / "div.db")
        risk_log = RiskRejectionLog(Path(tmp) / "risk.db")
        kill_switch = DailyKillSwitch(Path(tmp) / "ks.db")

        today = date(2026, 5, 20)
        # Seed one stale-close alert + one open position with a pending close
        order_log.record_stale_close_alert(
            opening_order_id=5001, symbol="XOM",
            expiration=date(2026, 5, 30), structure="straddle",
            attempts=3, last_exit_trigger="profit_target",
            detected_at=datetime(2026, 5, 20, 14, 0, tzinfo=timezone.utc),
        )
        # Need an opening order row for pending_close_attempts JOIN
        order_log._conn.execute(
            "INSERT INTO submitted_orders ("
            "tradier_order_id, fingerprint, submitted_at, symbol, expiration, "
            "direction, structure, horizon_lower, horizon_upper, weight_lower, "
            "underlying_price_at_signal, atm_iv_at_signal, predicted_iv_at_signal, "
            "divergence_at_signal, cross_sectional_z, time_series_z, "
            "submitted_price, legs_json, final_status, fill_price"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (6001, "fp6001", "2026-05-20T14:00:00+00:00", "MSFT",
             "2026-06-19", "BUY", "straddle", 10, 21, 0.5,
             400.0, 0.25, 0.32, 0.07, 1.8, None,
             4.50, "[]", "filled", 4.50),
        )
        order_log._conn.commit()
        order_log.record_close_attempt(
            opening_order_id=6001, closing_order_id=9001,
            submitted_at=datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc),
            exit_trigger="thesis_reversed", order_type="credit",
            submitted_price=3.20,
        )

        builder = DailySummaryBuilder(order_log, div_history, risk_log, kill_switch)
        snap = _mk_snapshot()
        summary = builder.build(today, snap)

        assert len(summary.stale_close_alerts) == 1
        assert summary.stale_close_alerts[0].symbol == "XOM"
        assert summary.stale_close_alerts[0].attempts == 3
        assert len(summary.pending_closes) == 1
        assert summary.pending_closes[0].closing_order_id == 9001
        assert summary.pending_closes[0].symbol == "MSFT"
        assert summary.pending_closes[0].exit_trigger == "thesis_reversed"

        for c in (order_log, div_history, risk_log, kill_switch):
            c.close()
    print("daily_summary: stale_close_alerts + pending_closes surfaced ✓")


def main() -> int:
    test_summary_with_no_activity()
    test_summary_with_filled_positions()
    test_summary_with_kill_switch()
    test_summary_rejection_categories()
    test_summary_pnl_matches_equity_delta()
    test_summary_surfaces_stale_close_alerts_and_pending()
    print("all daily_summary tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
