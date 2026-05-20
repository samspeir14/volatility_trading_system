"""Reconciler tests: log vs Tradier diff → expire / assign / hold."""
import asyncio
import json
import sys
import tempfile
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from execution import OrderLog
from positions.reconciler import (
    AssignmentAlert,
    PositionReconciler,
    TimeoutResolution,
    is_option_symbol,
    underlying_of_option,
)
from signals.signal_generator import TradeLeg, TradeSignal


# Real OCC symbol for an AAPL 2025-06-20 $150 call.
AAPL_C150 = "AAPL250620C00150000"
AAPL_P140 = "AAPL250620P00140000"
AAPL_C155 = "AAPL250620C00155000"
AAPL_P135 = "AAPL250620P00135000"


def _mk_signal(symbol: str, expiration: date, direction: str, legs: list[TradeLeg]) -> TradeSignal:
    return TradeSignal(
        symbol=symbol, structure="iron_condor", direction=direction,
        expiration=expiration, legs=legs,
        underlying_price=150.0, atm_iv=0.30, predicted_iv=0.25,
        predicted_iv_equivalent=0.25, divergence=-0.05,
        cross_sectional_z=-1.6, time_series_z=-1.2,
        horizon_lower=21, horizon_upper=21, weight_lower=1.0,
    )


def _seed_open_order(log: OrderLog, *, order_id: int, symbol: str, expiration: date,
                     direction: str, structure: str, entry_premium: float,
                     legs: list[TradeLeg], submitted_at: datetime,
                     final_status: str = "filled") -> None:
    """Insert directly — order_log.record_submission requires a TradeSignal but
    we want to test against pre-existing log rows from prior sessions.
    final_status='timeout' simulates an order that submitted but the polling
    timed out before reaching a terminal state."""
    legs_json = json.dumps([asdict(leg) for leg in legs])
    submitted_signed = -entry_premium if direction == "SELL" else entry_premium
    fill_price = submitted_signed if final_status == "filled" else None
    log._conn.execute(
        "INSERT INTO submitted_orders ("
        "tradier_order_id, fingerprint, submitted_at, symbol, expiration, "
        "direction, structure, horizon_lower, horizon_upper, weight_lower, "
        "underlying_price_at_signal, atm_iv_at_signal, predicted_iv_at_signal, "
        "divergence_at_signal, cross_sectional_z, time_series_z, "
        "submitted_price, legs_json, final_status, fill_price"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (order_id, f"fp-{order_id}", submitted_at.isoformat(), symbol,
         expiration.isoformat(), direction, structure, 21, 21, 1.0,
         150.0, 0.30, 0.25, -0.05, -1.6, -1.2,
         submitted_signed, legs_json, final_status, fill_price),
    )
    log._conn.commit()


def _ic_legs() -> list[TradeLeg]:
    """Standard iron condor legs (short inner, long outer wings)."""
    return [
        TradeLeg(strike=150.0, option_type="call", side="sell", quantity=1, contract_symbol=AAPL_C150),
        TradeLeg(strike=155.0, option_type="call", side="buy",  quantity=1, contract_symbol=AAPL_C155),
        TradeLeg(strike=140.0, option_type="put",  side="sell", quantity=1, contract_symbol=AAPL_P140),
        TradeLeg(strike=135.0, option_type="put",  side="buy",  quantity=1, contract_symbol=AAPL_P135),
    ]


def _straddle_legs() -> list[TradeLeg]:
    return [
        TradeLeg(strike=150.0, option_type="call", side="buy", quantity=1, contract_symbol=AAPL_C150),
        TradeLeg(strike=150.0, option_type="put",  side="buy", quantity=1, contract_symbol=AAPL_P140),
    ]


def test_is_option_symbol_recognizes_occ_format():
    assert is_option_symbol("AAPL250620C00150000")
    assert is_option_symbol("SPY260116P00400000")
    assert not is_option_symbol("AAPL")
    assert not is_option_symbol("BRK.B")
    assert not is_option_symbol("")
    print("is_option_symbol: OCC pattern recognized ✓")


def test_underlying_of_option_strips_suffix():
    assert underlying_of_option(AAPL_C150) == "AAPL"
    assert underlying_of_option("SPY260116P00400000") == "SPY"
    print("underlying_of_option: extracts ticker prefix ✓")


def test_reconciler_marks_expired_short_iron_condor():
    """The headline scenario: a SELL iron condor whose expiration has passed
    and all legs are gone from Tradier with no stock position. Should be
    marked expired_worthless with realized = +entry_credit."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 5, 15)
        entry_credit = 1.20  # $1.20 per share = $120 credit
        _seed_open_order(
            log, order_id=9001, symbol="AAPL", expiration=expiration,
            direction="SELL", structure="iron_condor", entry_premium=entry_credit,
            legs=_ic_legs(),
            submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        )
        assert len(log.open_unclosed_positions()) == 1

        # Tradier no longer reports any of the legs, no stock position.
        client = mock.AsyncMock()
        client.get_positions.return_value = []

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        # Run AFTER expiration date so reconciler will act.
        today = expiration + timedelta(days=3)
        result = asyncio.run(reconciler.reconcile(today))

        assert result.expired_closed == [9001], result
        assert result.assignment_alerts == []
        assert result.skipped_premature == []

        # Order log should now show closed with realized = +$120.
        assert len(log.open_unclosed_positions()) == 0
        row = log._conn.execute(
            "SELECT realized_pnl, closing_order_id, exit_trigger "
            "FROM submitted_orders WHERE tradier_order_id = ?", (9001,),
        ).fetchone()
        assert abs(row[0] - 120.0) < 0.01, f"expected +$120, got {row[0]}"
        assert row[1] == 0, "closing_order_id sentinel"
        assert row[2] == "expired_worthless"

        log.close()
    print("reconciler: short iron condor expired worthless → +entry_credit ✓")


def test_reconciler_marks_expired_long_straddle():
    """Long position expired worthless → realized = -entry_debit."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 5, 15)
        entry_debit = 4.50  # $4.50 per share = $450 debit
        _seed_open_order(
            log, order_id=9002, symbol="AAPL", expiration=expiration,
            direction="BUY", structure="straddle", entry_premium=entry_debit,
            legs=_straddle_legs(),
            submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        )

        client = mock.AsyncMock()
        client.get_positions.return_value = []

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        today = expiration + timedelta(days=1)
        result = asyncio.run(reconciler.reconcile(today))

        assert result.expired_closed == [9002]
        row = log._conn.execute(
            "SELECT realized_pnl FROM submitted_orders WHERE tradier_order_id = ?",
            (9002,),
        ).fetchone()
        assert abs(row[0] - (-450.0)) < 0.01, f"expected -$450, got {row[0]}"

        log.close()
    print("reconciler: long straddle expired worthless → -entry_debit ✓")


def test_reconciler_holds_live_position_with_legs_still_in_tradier():
    """Sanity check: position whose legs are still present is left alone."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        _seed_open_order(
            log, order_id=9003, symbol="AAPL", expiration=date(2026, 6, 19),
            direction="SELL", structure="iron_condor", entry_premium=1.20,
            legs=_ic_legs(),
            submitted_at=datetime(2026, 5, 1, 16, 0, tzinfo=timezone.utc),
        )

        client = mock.AsyncMock()
        # All 4 legs still in Tradier
        client.get_positions.return_value = [
            {"symbol": AAPL_C150, "quantity": -1, "cost_basis": -120.0},
            {"symbol": AAPL_C155, "quantity":  1, "cost_basis":   80.0},
            {"symbol": AAPL_P140, "quantity": -1, "cost_basis": -100.0},
            {"symbol": AAPL_P135, "quantity":  1, "cost_basis":   60.0},
        ]

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(date(2026, 5, 20)))

        assert result.expired_closed == []
        assert result.assignment_alerts == []
        assert len(log.open_unclosed_positions()) == 1

        log.close()
    print("reconciler: live position with all legs present → left alone ✓")


def test_reconciler_flags_assignment_when_stock_position_appears():
    """Iron condor whose short call was assigned: stock position appears in
    Tradier. Reconciler must log CRITICAL, persist alert, NOT auto-close."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 5, 15)
        _seed_open_order(
            log, order_id=9004, symbol="AAPL", expiration=expiration,
            direction="SELL", structure="iron_condor", entry_premium=1.20,
            legs=_ic_legs(),
            submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        )

        client = mock.AsyncMock()
        # All option legs gone, but a -100 AAPL stock position appeared
        # (short call assignment → short stock).
        client.get_positions.return_value = [
            {"symbol": "AAPL", "quantity": -100, "cost_basis": -15000.0},
        ]

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(expiration + timedelta(days=1)))

        assert result.assignment_alerts == [
            AssignmentAlert(tradier_order_id=9004, symbol="AAPL",
                            expiration=expiration, structure="iron_condor",
                            stock_quantity=-100.0),
        ], result
        assert result.expired_closed == [], "must NOT auto-close on assignment"
        # Order remains open for manual handling
        assert len(log.open_unclosed_positions()) == 1
        # Alert is persisted
        active = log.assignment_alerts_active()
        assert len(active) == 1
        assert active[0]["tradier_order_id"] == 9004
        assert active[0]["symbol"] == "AAPL"
        assert active[0]["stock_quantity"] == -100.0

        log.close()
    print("reconciler: assignment → CRITICAL alert persisted, no auto-close ✓")


def test_reconciler_idempotent_assignment_alert():
    """Running reconcile twice should not duplicate the alert row."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 5, 15)
        _seed_open_order(
            log, order_id=9005, symbol="AAPL", expiration=expiration,
            direction="SELL", structure="iron_condor", entry_premium=1.20,
            legs=_ic_legs(),
            submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        )
        client = mock.AsyncMock()
        client.get_positions.return_value = [
            {"symbol": "AAPL", "quantity": -100, "cost_basis": -15000.0},
        ]
        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")

        r1 = asyncio.run(reconciler.reconcile(expiration + timedelta(days=1)))
        r2 = asyncio.run(reconciler.reconcile(expiration + timedelta(days=2)))

        # First call raises alert, second call sees it's already raised.
        assert len(r1.assignment_alerts) == 1
        assert len(r2.assignment_alerts) == 0
        assert len(log.assignment_alerts_active()) == 1

        log.close()
    print("reconciler: assignment alert is idempotent ✓")


def test_reconciler_skips_premature_disappearance():
    """If all legs are missing but expiration is still in the future, this
    is a probable API race — leave the log alone."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        future_expiration = date(2027, 1, 15)  # far in the future
        _seed_open_order(
            log, order_id=9006, symbol="AAPL", expiration=future_expiration,
            direction="SELL", structure="iron_condor", entry_premium=1.20,
            legs=_ic_legs(),
            submitted_at=datetime(2026, 12, 1, 16, 0, tzinfo=timezone.utc),
        )
        client = mock.AsyncMock()
        client.get_positions.return_value = []

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        # Today is well before expiration
        result = asyncio.run(reconciler.reconcile(date(2026, 12, 5)))

        assert result.expired_closed == []
        assert result.skipped_premature == [9006]
        # Log entry must still be open.
        assert len(log.open_unclosed_positions()) == 1

        log.close()
    print("reconciler: missing-but-not-yet-expired → no action ✓")


def test_reconciler_handles_get_positions_failure_gracefully():
    """If Tradier API errors, reconciler must NOT close anything — return empty
    and let the next cycle retry."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        _seed_open_order(
            log, order_id=9007, symbol="AAPL", expiration=date(2026, 5, 15),
            direction="SELL", structure="iron_condor", entry_premium=1.20,
            legs=_ic_legs(),
            submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        )
        client = mock.AsyncMock()
        client.get_positions.side_effect = RuntimeError("Tradier 500")

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(date(2026, 5, 20)))

        assert result.expired_closed == []
        assert result.assignment_alerts == []
        # Log entry untouched.
        assert len(log.open_unclosed_positions()) == 1

        log.close()
    print("reconciler: API failure → no-op, retry next cycle ✓")


def test_timeout_recovery_marks_filled_order():
    """The headline scenario for this fix: a timeout-status row whose Tradier
    order actually filled. Reconciler queries get_order_status, updates the
    log to 'filled' with the real avg_fill_price."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 6, 19)
        _seed_open_order(
            log, order_id=9101, symbol="UNH", expiration=expiration,
            direction="BUY", structure="straddle", entry_premium=14.66,
            legs=_straddle_legs(),
            submitted_at=datetime(2026, 5, 4, 15, 0, tzinfo=timezone.utc),
            final_status="timeout",
        )
        # Not in open_unclosed_positions yet (filtered by status)
        assert len(log.open_unclosed_positions()) == 0
        assert len(log.timeout_orders()) == 1

        client = mock.AsyncMock()
        client.get_positions.return_value = [
            {"symbol": AAPL_C150, "quantity": 1, "cost_basis": 1466.0},
            {"symbol": AAPL_P140, "quantity": 1, "cost_basis": 0.0},
        ]
        client.get_order_status.return_value = {
            "order": {"id": 9101, "status": "filled", "avg_fill_price": 14.55},
        }

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(date(2026, 5, 20)))

        assert len(result.timeouts_resolved) == 1
        res = result.timeouts_resolved[0]
        assert res.tradier_order_id == 9101
        assert res.new_status == "filled"
        assert abs(res.fill_price - 14.55) < 1e-9

        # After recovery, the row is now in open_unclosed_positions
        rows = log.open_unclosed_positions()
        assert len(rows) == 1
        assert rows[0]["tradier_order_id"] == 9101
        assert rows[0]["final_status"] == "filled"
        assert abs(rows[0]["fill_price"] - 14.55) < 1e-9
        # Not expired yet — legs still in Tradier, expiration in future
        assert result.expired_closed == []

        log.close()
    print("timeout_recovery: filled order recovered with real fill_price ✓")


def test_timeout_recovery_marks_rejected_order():
    """Timeout that Tradier reports as rejected — no fill price, status updated."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        _seed_open_order(
            log, order_id=9102, symbol="UNH", expiration=date(2026, 6, 19),
            direction="BUY", structure="straddle", entry_premium=14.66,
            legs=_straddle_legs(),
            submitted_at=datetime(2026, 5, 4, 15, 0, tzinfo=timezone.utc),
            final_status="timeout",
        )
        client = mock.AsyncMock()
        client.get_positions.return_value = []
        client.get_order_status.return_value = {
            "order": {"id": 9102, "status": "rejected", "reason_description": "no liquidity"},
        }

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(date(2026, 5, 20)))

        assert len(result.timeouts_resolved) == 1
        assert result.timeouts_resolved[0].new_status == "rejected"
        assert result.timeouts_resolved[0].fill_price is None
        # Rejected orders don't enter open_unclosed_positions (status filter)
        assert len(log.open_unclosed_positions()) == 0
        assert len(log.timeout_orders()) == 0  # no longer a timeout
        # No expirations either
        assert result.expired_closed == []

        log.close()
    print("timeout_recovery: rejected status persisted ✓")


def test_timeout_recovery_handles_api_error():
    """If get_order_status raises, leave the row as 'timeout' and retry later."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        _seed_open_order(
            log, order_id=9103, symbol="UNH", expiration=date(2026, 6, 19),
            direction="BUY", structure="straddle", entry_premium=14.66,
            legs=_straddle_legs(),
            submitted_at=datetime(2026, 5, 4, 15, 0, tzinfo=timezone.utc),
            final_status="timeout",
        )
        client = mock.AsyncMock()
        client.get_positions.return_value = []
        client.get_order_status.side_effect = RuntimeError("Tradier 503")

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(date(2026, 5, 20)))

        assert len(result.timeouts_resolved) == 1
        assert result.timeouts_resolved[0].new_status == "unknown"
        # Row still listed as timeout
        assert len(log.timeout_orders()) == 1
        log.close()
    print("timeout_recovery: API error → leave as timeout, retry next cycle ✓")


def test_timeout_recovery_handles_still_pending():
    """If Tradier returns a non-terminal status (open, pending), keep as timeout."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        _seed_open_order(
            log, order_id=9104, symbol="UNH", expiration=date(2026, 6, 19),
            direction="BUY", structure="straddle", entry_premium=14.66,
            legs=_straddle_legs(),
            submitted_at=datetime(2026, 5, 4, 15, 0, tzinfo=timezone.utc),
            final_status="timeout",
        )
        client = mock.AsyncMock()
        client.get_positions.return_value = []
        client.get_order_status.return_value = {
            "order": {"id": 9104, "status": "pending"},
        }

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(date(2026, 5, 20)))

        assert result.timeouts_resolved[0].new_status == "unknown"
        assert len(log.timeout_orders()) == 1  # untouched
        log.close()
    print("timeout_recovery: non-terminal status → leave as timeout ✓")


def test_timeout_recovery_chains_into_expiration_marking():
    """End-to-end: timeout order that actually filled AND has since expired.
    A single reconcile() call must recover the timeout, observe that Tradier
    no longer reports the legs, and mark the position expired with the
    correct realized P&L. This is the exact UNH 5/15 scenario from prod."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 5, 15)  # already passed (today below = 5/20)
        _seed_open_order(
            log, order_id=9105, symbol="UNH", expiration=expiration,
            direction="BUY", structure="straddle", entry_premium=14.66,
            legs=_straddle_legs(),
            submitted_at=datetime(2026, 5, 4, 15, 0, tzinfo=timezone.utc),
            final_status="timeout",
        )

        client = mock.AsyncMock()
        # Tradier confirms the order filled at the submitted price
        client.get_order_status.return_value = {
            "order": {"id": 9105, "status": "filled", "avg_fill_price": 14.55},
        }
        # No options for this position in current Tradier positions, no stock
        client.get_positions.return_value = []

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(date(2026, 5, 20)))

        # Both phases ran on the same row
        assert result.timeouts_resolved[0].new_status == "filled"
        assert result.expired_closed == [9105]

        # Order is now closed with the right realized P&L:
        # Long straddle (BUY) at 14.55 paid → -1455 realized when expired worthless.
        # Reconciler uses fill_price (now populated) per position_tracker semantics:
        # entry_premium = abs(fill_price), realized = -entry_premium * 100 for long.
        row = log._conn.execute(
            "SELECT realized_pnl, closing_order_id, exit_trigger, final_status, fill_price "
            "FROM submitted_orders WHERE tradier_order_id = ?", (9105,),
        ).fetchone()
        assert abs(row[0] - (-1455.0)) < 0.01, f"expected -$1455, got {row[0]}"
        assert row[1] == 0  # expiration sentinel
        assert row[2] == "expired_worthless"
        assert row[3] == "filled"
        assert abs(row[4] - 14.55) < 1e-9

        log.close()
    print("timeout_recovery: recovered fill → marked expired with correct realized P&L ✓")


def test_reconciliation_surfaces_in_daily_summary():
    """After an assignment alert is persisted, the next daily summary must
    include it in `assignment_alerts`."""
    from logs import DailySummaryBuilder
    from risk import DailyKillSwitch, RiskRejectionLog
    from signals import DivergenceHistory

    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        div = DivergenceHistory(Path(tmp) / "div.db")
        risk = RiskRejectionLog(Path(tmp) / "risk.db")
        ks = DailyKillSwitch(Path(tmp) / "ks.db")

        log.record_assignment_alert(
            tradier_order_id=9999, symbol="AAPL",
            expiration=date(2026, 5, 15), structure="iron_condor",
            detected_at=datetime(2026, 5, 16, 14, 30, tzinfo=timezone.utc),
            stock_quantity=-100.0,
        )

        snap = mock.MagicMock()
        snap.starting_equity_today = 100_000.0
        snap.equity = 100_000.0
        snap.today_realized_pnl = 0.0
        snap.today_unrealized_pnl = 0.0
        snap.today_total_pnl = 0.0
        snap.open_positions = []

        builder = DailySummaryBuilder(log, div, risk, ks)
        summary = builder.build(date(2026, 5, 16), snap)
        assert len(summary.assignment_alerts) == 1
        a = summary.assignment_alerts[0]
        assert a.tradier_order_id == 9999
        assert a.symbol == "AAPL"
        assert a.stock_quantity == -100.0

        # And it should render in the slack message
        from logs.slack import format_summary
        rendered = format_summary(summary)
        assert "ASSIGNMENT ALERT" in rendered
        assert "9999" in rendered
        assert "AAPL" in rendered

        for c in (log, div, risk, ks):
            c.close()
    print("daily_summary: assignment alerts surface in summary + slack ✓")


def main() -> int:
    test_is_option_symbol_recognizes_occ_format()
    test_underlying_of_option_strips_suffix()
    test_reconciler_marks_expired_short_iron_condor()
    test_reconciler_marks_expired_long_straddle()
    test_reconciler_holds_live_position_with_legs_still_in_tradier()
    test_reconciler_flags_assignment_when_stock_position_appears()
    test_reconciler_idempotent_assignment_alert()
    test_reconciler_skips_premature_disappearance()
    test_reconciler_handles_get_positions_failure_gracefully()
    test_timeout_recovery_marks_filled_order()
    test_timeout_recovery_marks_rejected_order()
    test_timeout_recovery_handles_api_error()
    test_timeout_recovery_handles_still_pending()
    test_timeout_recovery_chains_into_expiration_marking()
    test_reconciliation_surfaces_in_daily_summary()
    print("all reconciler tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
