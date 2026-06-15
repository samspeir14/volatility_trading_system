"""Stale-close-management tests covering the loop described in the spec:

  1. submit → poll timeout → close_attempt left 'pending'
  2. next cycle → still pending past threshold → cancel + mark stale_canceled
  3. between-cycle fill → reconcile_pending_closes updates attempt + closes opening
  4. max retries exceeded → CRITICAL log + stale_close_alert + no further submits
"""
import asyncio
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from config import Settings
from execution import OrderLog, OrderManager
from execution.order_log import PENDING_CLOSE_STATUS, STALE_CANCELED_STATUS
from positions.position_tracker import OpenPosition, PositionMark
from signals.signal_generator import TradeLeg


def _mk_settings() -> Settings:
    return Settings(
        api_key="fake", account_id="VA00000000",
        base_url="https://example.invalid/v1", env="sandbox",
    )


def _mk_long_straddle_position(*, order_id: int = 5001) -> OpenPosition:
    legs = [
        TradeLeg(100.0, "call", "buy", 1, "AAPL260515C00100000"),
        TradeLeg(100.0, "put", "buy", 1, "AAPL260515P00100000"),
    ]
    return OpenPosition(
        tradier_order_id=order_id, symbol="AAPL",
        expiration=date(2026, 5, 15), direction="BUY",
        structure="straddle", legs=legs, entry_premium=4.08,
        entry_atm_iv=0.27, entry_predicted_iv=0.42, entry_divergence=0.15,
        entry_horizon_lower=10, entry_horizon_upper=21, entry_weight_lower=0.27,
        submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
    )


def _mk_mark(pos: OpenPosition, *, close_cash_flow: float) -> PositionMark:
    return PositionMark(
        position=pos, current_legs=[],
        close_cash_flow=close_cash_flow,
        cost_to_close=abs(close_cash_flow) if close_cash_flow < 0 else 0,
        pnl_dollars=0.0, pnl_pct_of_entry_premium=0.0,
        pnl_pct_of_max=float("nan"),
        delta=0, gamma=0, theta=0, vega=0, dte=10,
    )


def _seed_open_order(log: OrderLog, pos: OpenPosition) -> None:
    legs_json = "[" + ",".join(
        f'{{"strike": {l.strike}, "option_type": "{l.option_type}", '
        f'"side": "{l.side}", "quantity": {l.quantity}, '
        f'"contract_symbol": "{l.contract_symbol}"}}'
        for l in pos.legs
    ) + "]"
    log._conn.execute(
        "INSERT INTO submitted_orders ("
        "tradier_order_id, fingerprint, submitted_at, symbol, expiration, "
        "direction, structure, horizon_lower, horizon_upper, weight_lower, "
        "underlying_price_at_signal, atm_iv_at_signal, predicted_iv_at_signal, "
        "divergence_at_signal, cross_sectional_z, time_series_z, "
        "submitted_price, legs_json, final_status, fill_price"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            pos.tradier_order_id, f"fp{pos.tradier_order_id}",
            pos.submitted_at.isoformat(), pos.symbol, pos.expiration.isoformat(),
            pos.direction, pos.structure,
            pos.entry_horizon_lower, pos.entry_horizon_upper, pos.entry_weight_lower,
            100.0, pos.entry_atm_iv, pos.entry_predicted_iv,
            pos.entry_divergence, 2.0, None,
            pos.entry_premium, legs_json, "filled",
            -pos.entry_premium if not pos.is_long else pos.entry_premium,
        ),
    )
    log._conn.commit()


def _mk_manager_pending_poll(
    log: OrderLog, *, closing_order_id: int = 9000,
    stale_threshold_min: int = 15, max_retries: int = 3,
) -> tuple[OrderManager, mock.AsyncMock]:
    """OrderManager whose get_order_status keeps returning 'pending' so poll
    never reaches terminal state."""
    fake_client = mock.AsyncMock()
    fake_client.preview_order.return_value = {"order": {"status": "ok"}}
    fake_client.place_order.return_value = {
        "order": {"id": closing_order_id, "status": "pending"}
    }
    fake_client.get_order_status.return_value = {
        "order": {"id": closing_order_id, "status": "pending"}
    }
    fake_client.cancel_order.return_value = {"order": {"id": closing_order_id, "status": "ok"}}
    mgr = OrderManager(
        client=fake_client, order_log=log, settings=_mk_settings(),
        poll_interval_seconds=0.001, poll_timeout_seconds=0.05,
        slippage_buffer=0.0,
        stale_order_threshold_minutes=stale_threshold_min,
        max_close_retries=max_retries,
    )
    return mgr, fake_client


# ---------- Scenario 1: poll timeout → 'pending' status + close_attempt row ----------

def test_submit_close_poll_timeout_records_pending_attempt():
    """When the close doesn't fill within the submission poll window, the
    attempt row must be left as 'pending' (not 'timeout') and the opening
    order must NOT be marked closed."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_long_straddle_position(order_id=6101)
        _seed_open_order(log, pos)
        mark = _mk_mark(pos, close_cash_flow=200.0)
        mgr, fake = _mk_manager_pending_poll(log, closing_order_id=9101)

        result = asyncio.run(mgr.submit_close(
            position=pos, mark=mark, exit_trigger="profit_target",
        ))

        assert result.status == "pending", f"expected pending, got {result.status}"
        assert result.order_id == 9101

        # close_attempt row exists with status='pending'
        pending = log.pending_close_attempts()
        assert len(pending) == 1
        assert pending[0]["closing_order_id"] == 9101
        assert pending[0]["opening_order_id"] == 6101
        assert pending[0]["status"] == PENDING_CLOSE_STATUS
        assert pending[0]["order_type"] == "credit"
        assert pending[0]["exit_trigger"] == "profit_target"

        # Opening order is NOT marked closed
        opening = log.get_submitted_order(6101)
        assert opening["closing_order_id"] is None
        assert opening["closed_at"] is None
        assert opening["realized_pnl"] is None

        # P&L not yet booked
        today = datetime.now(timezone.utc).date()
        assert log.closed_today_pnl(today) == 0.0

        log.close()
    print("submit_close poll timeout: attempt recorded as pending, opening untouched ✓")


# ---------- Scenario 2: stale pending → cancel ----------

def test_reconcile_cancels_stale_pending_close():
    """Next cycle finds the pending close older than threshold; reconcile_pending_closes
    must cancel it via Tradier and mark the attempt as stale_canceled. exit_manager
    can then re-evaluate and resubmit."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_long_straddle_position(order_id=6201)
        _seed_open_order(log, pos)
        mark = _mk_mark(pos, close_cash_flow=200.0)

        # First: submit close (poll times out, attempt becomes pending)
        mgr, fake = _mk_manager_pending_poll(
            log, closing_order_id=9201, stale_threshold_min=15,
        )
        asyncio.run(mgr.submit_close(pos, mark, "profit_target"))
        assert log.has_pending_close(6201) is True

        # Simulate ~20 minutes have passed since the close was submitted
        now = datetime.now(timezone.utc) + timedelta(minutes=20)
        result = asyncio.run(mgr.reconcile_pending_closes(now))

        assert result["canceled"] == 1, f"expected 1 cancel, got {result}"
        assert result["filled"] == 0
        # cancel_order was called
        fake.cancel_order.assert_called_once_with("VA00000000", 9201)

        # Attempt now marked stale_canceled
        pending = log.pending_close_attempts()
        assert len(pending) == 0, "no pending attempts after cancel"
        # Failed count = 1, room for 2 more retries (max=3)
        assert log.failed_close_attempt_count(6201) == 1
        # Opening order still open
        assert log.get_submitted_order(6201)["closing_order_id"] is None

        log.close()
    print("reconcile_pending_closes: stale pending canceled, slot freed for retry ✓")


def test_reconcile_does_not_cancel_fresh_pending():
    """A close that's still within the stale threshold should NOT be canceled."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_long_straddle_position(order_id=6202)
        _seed_open_order(log, pos)
        mark = _mk_mark(pos, close_cash_flow=200.0)

        mgr, fake = _mk_manager_pending_poll(
            log, closing_order_id=9202, stale_threshold_min=15,
        )
        asyncio.run(mgr.submit_close(pos, mark, "profit_target"))

        # Only 5 minutes have passed — should NOT cancel
        now = datetime.now(timezone.utc) + timedelta(minutes=5)
        result = asyncio.run(mgr.reconcile_pending_closes(now))

        assert result["canceled"] == 0
        fake.cancel_order.assert_not_called()
        # Still pending
        assert log.has_pending_close(6202) is True
        log.close()
    print("reconcile_pending_closes: fresh pending not canceled ✓")


# ---------- Scenario 3: between-cycle fill ----------

def test_reconcile_recognizes_between_cycle_fill():
    """A close that was pending in our log filled at Tradier between cycles.
    reconcile_pending_closes must update the attempt to 'filled' AND mark the
    opening as closed with realized P&L computed from the actual fill price."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_long_straddle_position(order_id=6301)
        _seed_open_order(log, pos)
        mark = _mk_mark(pos, close_cash_flow=200.0)

        # Submit → pending
        mgr, fake = _mk_manager_pending_poll(log, closing_order_id=9301)
        asyncio.run(mgr.submit_close(pos, mark, "profit_target"))
        assert log.has_pending_close(6301) is True

        # Between cycles, Tradier reports the close as filled at $2.00 credit
        # (returned as -2.00 per Tradier sign convention)
        fake.get_order_status.return_value = {
            "order": {"id": 9301, "status": "filled", "avg_fill_price": -2.00},
        }

        now = datetime.now(timezone.utc) + timedelta(minutes=1)
        result = asyncio.run(mgr.reconcile_pending_closes(now))

        assert result["filled"] == 1, f"expected 1 fill, got {result}"
        # cancel was NOT called (filled, not stale)
        fake.cancel_order.assert_not_called()

        # Attempt marked filled, no longer pending
        pending = log.pending_close_attempts()
        assert len(pending) == 0

        # Opening order now closed with correct realized P&L
        # Long straddle paid $4.08, sold at $2.00 → realized = -$408 + $200 = -$208
        opening = log.get_submitted_order(6301)
        assert opening["closing_order_id"] == 9301
        assert opening["exit_trigger"] == "profit_target"
        assert opening["realized_pnl"] is not None
        assert abs(opening["realized_pnl"] - (-208.0)) <= 1.0, (
            f"realized_pnl {opening['realized_pnl']} != expected -208"
        )

        log.close()
    print("reconcile_pending_closes: between-cycle fill reconciled correctly ✓")


def test_reconcile_handles_terminal_failed_states():
    """A pending attempt that Tradier later reports as rejected/canceled/expired
    must update the attempt to that status and count against the retry budget."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_long_straddle_position(order_id=6302)
        _seed_open_order(log, pos)
        mark = _mk_mark(pos, close_cash_flow=200.0)

        mgr, fake = _mk_manager_pending_poll(log, closing_order_id=9302)
        asyncio.run(mgr.submit_close(pos, mark, "profit_target"))

        # Next cycle Tradier reports rejected
        fake.get_order_status.return_value = {
            "order": {"id": 9302, "status": "rejected"},
        }
        now = datetime.now(timezone.utc) + timedelta(minutes=1)
        result = asyncio.run(mgr.reconcile_pending_closes(now))

        assert result["failed_terminal"] == 1
        # No longer pending; counts toward retries
        assert log.has_pending_close(6302) is False
        assert log.failed_close_attempt_count(6302) == 1
        # Opening still open
        assert log.get_submitted_order(6302)["closing_order_id"] is None
        log.close()
    print("reconcile_pending_closes: terminal failed updates attempt + retry budget ✓")


# ---------- Scenario 4: max retries exceeded ----------

def test_max_retries_exceeded_logs_critical_and_alerts():
    """After 3 failed close attempts, submit_close refuses, logs CRITICAL,
    and records a stale_close_alert for the daily summary."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_long_straddle_position(order_id=6401)
        _seed_open_order(log, pos)
        mark = _mk_mark(pos, close_cash_flow=200.0)

        # Seed 3 prior failed attempts directly
        for i, oid in enumerate((9401, 9402, 9403)):
            log.record_close_attempt(
                opening_order_id=6401, closing_order_id=oid,
                submitted_at=datetime.now(timezone.utc),
                exit_trigger="profit_target", order_type="credit",
                submitted_price=2.00, status=STALE_CANCELED_STATUS,
            )
        assert log.failed_close_attempt_count(6401) == 3

        mgr, fake = _mk_manager_pending_poll(
            log, closing_order_id=9404, stale_threshold_min=15, max_retries=3,
        )

        result = asyncio.run(mgr.submit_close(pos, mark, "profit_target"))

        assert result.status == "max_retries_exceeded", (
            f"expected max_retries_exceeded, got {result.status}"
        )
        # place_order was NOT called — we refused before submitting
        fake.place_order.assert_not_called()
        # Alert recorded
        alerts = log.stale_close_alerts_active()
        assert len(alerts) == 1
        assert alerts[0]["opening_order_id"] == 6401
        assert alerts[0]["attempts"] == 3
        log.close()
    print("submit_close max_retries: refused + alert recorded ✓")


def test_max_retries_alert_idempotent():
    """Calling submit_close repeatedly after max retries should not duplicate
    the alert row, but should keep returning max_retries_exceeded."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_long_straddle_position(order_id=6402)
        _seed_open_order(log, pos)
        mark = _mk_mark(pos, close_cash_flow=200.0)

        for oid in (9410, 9411, 9412):
            log.record_close_attempt(
                opening_order_id=6402, closing_order_id=oid,
                submitted_at=datetime.now(timezone.utc),
                exit_trigger="stop_loss", order_type="credit",
                submitted_price=2.00, status=STALE_CANCELED_STATUS,
            )

        mgr, fake = _mk_manager_pending_poll(log, max_retries=3)

        for _ in range(3):
            result = asyncio.run(mgr.submit_close(pos, mark, "stop_loss"))
            assert result.status == "max_retries_exceeded"

        assert len(log.stale_close_alerts_active()) == 1
        log.close()
    print("submit_close max_retries: idempotent alert ✓")


def test_stale_close_alert_auto_resolves_when_position_closes():
    """An alert must disappear once its opening order is closed by any path —
    here expiration settled by the reconciler (sentinel closing_order_id=0).
    Regression: near-expiry straddles that exhaust the close-retry cap and then
    expire were re-emitting their stale alert in every daily summary forever."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_long_straddle_position(order_id=6403)
        _seed_open_order(log, pos)

        log.record_stale_close_alert(
            opening_order_id=6403, symbol="AAPL", expiration=date(2026, 5, 15),
            structure="straddle", attempts=3, last_exit_trigger="expiration_proximity",
            detected_at=datetime.now(timezone.utc),
        )
        # Still open → alert is live.
        assert len(log.stale_close_alerts_active()) == 1

        # Reconciler settles it at expiration (sentinel closing_order_id=0).
        log.record_expiration(
            opening_order_id=6403, expired_at=datetime.now(timezone.utc),
            realized_pnl=-408.0, exit_trigger="expired",
        )
        # Closed → alert auto-resolves, even though the row still exists.
        assert log.stale_close_alerts_active() == []
        log.close()
    print("stale_close_alert auto-resolves on position close ✓")


def test_stale_close_alert_orphan_row_still_surfaces():
    """Fail-safe: an alert whose opening order row is missing must still show
    (the LEFT JOIN keeps it) rather than being silently dropped."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        log.record_stale_close_alert(
            opening_order_id=999999, symbol="ZZZ", expiration=date(2026, 5, 15),
            structure="straddle", attempts=3, last_exit_trigger="stop_loss",
            detected_at=datetime.now(timezone.utc),
        )
        assert len(log.stale_close_alerts_active()) == 1
        log.close()
    print("stale_close_alert orphan row surfaces ✓")


# ---------- Scenario 5: prevent duplicate submissions while pending ----------

def test_submit_close_skips_when_pending_attempt_exists():
    """If there's already a pending close for the opening order, submit_close
    must NOT submit another one."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_long_straddle_position(order_id=6501)
        _seed_open_order(log, pos)
        mark = _mk_mark(pos, close_cash_flow=200.0)

        log.record_close_attempt(
            opening_order_id=6501, closing_order_id=9501,
            submitted_at=datetime.now(timezone.utc),
            exit_trigger="profit_target", order_type="credit",
            submitted_price=2.00, status=PENDING_CLOSE_STATUS,
        )

        mgr, fake = _mk_manager_pending_poll(log, closing_order_id=9502)
        result = asyncio.run(mgr.submit_close(pos, mark, "profit_target"))

        assert result.status == "pending_close_exists"
        fake.place_order.assert_not_called()
        log.close()
    print("submit_close: skips when prior attempt still pending ✓")


# ---------- Scenario 6: cancel race → re-query reveals fill ----------

def test_reconcile_cancel_race_reveals_fill():
    """Tradier rejects our cancel because the order just filled. The reconciler
    should re-query, find the fill, and reconcile correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_long_straddle_position(order_id=6601)
        _seed_open_order(log, pos)
        mark = _mk_mark(pos, close_cash_flow=200.0)

        mgr, fake = _mk_manager_pending_poll(log, closing_order_id=9601)
        asyncio.run(mgr.submit_close(pos, mark, "profit_target"))

        # On reconcile: first get_order_status (the age check) says still pending
        # but cancel_order returns an error (order already filled), then a second
        # get_order_status reveals the fill.
        fake.get_order_status.side_effect = [
            {"order": {"id": 9601, "status": "pending"}},
            {"order": {"id": 9601, "status": "filled", "avg_fill_price": -2.00}},
        ]
        fake.cancel_order.return_value = {
            "errors": {"error": "Cannot cancel an order that has already been filled"},
        }

        now = datetime.now(timezone.utc) + timedelta(minutes=20)
        result = asyncio.run(mgr.reconcile_pending_closes(now))

        assert result["filled"] == 1, f"expected 1 filled via race-recover, got {result}"
        # Opening was closed
        opening = log.get_submitted_order(6601)
        assert opening["closing_order_id"] == 9601
        assert abs(opening["realized_pnl"] - (-208.0)) <= 1.0
        log.close()
    print("reconcile cancel race: fill recovered via re-query ✓")


# ---------- Scenario 7: cancel_order network failure → leave pending ----------

def test_reconcile_leaves_pending_on_cancel_network_error():
    """If cancel_order raises (network error / 5xx after retries), the attempt
    stays pending — next cycle will retry."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_long_straddle_position(order_id=6701)
        _seed_open_order(log, pos)
        mark = _mk_mark(pos, close_cash_flow=200.0)

        mgr, fake = _mk_manager_pending_poll(log, closing_order_id=9701)
        asyncio.run(mgr.submit_close(pos, mark, "profit_target"))

        fake.cancel_order.side_effect = RuntimeError("network kerplodied")
        now = datetime.now(timezone.utc) + timedelta(minutes=20)
        result = asyncio.run(mgr.reconcile_pending_closes(now))

        assert result["canceled"] == 0
        assert log.has_pending_close(6701) is True
        log.close()
    print("reconcile: cancel network error leaves attempt pending for retry ✓")


def main() -> int:
    test_submit_close_poll_timeout_records_pending_attempt()
    test_reconcile_cancels_stale_pending_close()
    test_reconcile_does_not_cancel_fresh_pending()
    test_reconcile_recognizes_between_cycle_fill()
    test_reconcile_handles_terminal_failed_states()
    test_max_retries_exceeded_logs_critical_and_alerts()
    test_max_retries_alert_idempotent()
    test_stale_close_alert_auto_resolves_when_position_closes()
    test_stale_close_alert_orphan_row_still_surfaces()
    test_submit_close_skips_when_pending_attempt_exists()
    test_reconcile_cancel_race_reveals_fill()
    test_reconcile_leaves_pending_on_cancel_network_error()
    print("all stale-close tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
