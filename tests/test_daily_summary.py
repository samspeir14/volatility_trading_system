import json
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

from execution import OrderLog
from logs import DailySummary, DailySummaryBuilder
from risk import DailyKillSwitch, RiskRejectionLog
from signals import DivergenceHistory


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


def main() -> int:
    test_summary_with_no_activity()
    test_summary_with_filled_positions()
    test_summary_with_kill_switch()
    test_summary_rejection_categories()
    print("all daily_summary tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
