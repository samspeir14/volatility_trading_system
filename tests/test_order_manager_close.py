"""Regression tests for OrderManager.submit_close() — focus on the realized P&L
sign convention. The bug being guarded: Tradier returns avg_fill_price signed
(credits negative); the close path must abs() it and let order_type carry the
sign. Earlier code passed the signed value through, doubling apparent losses on
long-position closes."""
import asyncio
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

from config import Settings
from execution import OrderLog, OrderManager
from positions.position_tracker import OpenPosition, PositionMark
from signals.signal_generator import TradeLeg


def _mk_settings() -> Settings:
    return Settings(
        api_key="fake", account_id="VA00000000",
        base_url="https://example.invalid/v1", env="sandbox",
    )


def _mk_long_straddle_position() -> OpenPosition:
    legs = [
        TradeLeg(100.0, "call", "buy", 1, "AAPL260515C00100000"),
        TradeLeg(100.0, "put", "buy", 1, "AAPL260515P00100000"),
    ]
    return OpenPosition(
        tradier_order_id=5001, symbol="AAPL",
        expiration=date(2026, 5, 15), direction="BUY",
        structure="straddle", legs=legs, entry_premium=4.08,
        entry_atm_iv=0.27, entry_predicted_iv=0.42, entry_divergence=0.15,
        entry_horizon_lower=10, entry_horizon_upper=21, entry_weight_lower=0.27,
        submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
    )


def _mk_iron_condor_position() -> OpenPosition:
    legs = [
        TradeLeg(210.0, "call", "sell", 1, "NVDA260522C00210000"),
        TradeLeg(210.0, "put", "sell", 1, "NVDA260522P00210000"),
        TradeLeg(230.0, "call", "buy", 1, "NVDA260522C00230000"),
        TradeLeg(190.0, "put", "buy", 1, "NVDA260522P00190000"),
    ]
    return OpenPosition(
        tradier_order_id=5002, symbol="NVDA",
        expiration=date(2026, 5, 22), direction="SELL",
        structure="iron_condor", legs=legs, entry_premium=13.55,
        entry_atm_iv=0.40, entry_predicted_iv=0.32, entry_divergence=-0.08,
        entry_horizon_lower=21, entry_horizon_upper=21, entry_weight_lower=1.0,
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
    """Insert an opening order row so submit_close has something to update."""
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


def _mk_manager(log: OrderLog, *, fill_price: float, closing_order_id: int = 9000) -> OrderManager:
    """Build an OrderManager with a fully-mocked client returning the given fill_price."""
    fake_client = mock.AsyncMock()
    fake_client.preview_order.return_value = {"order": {"status": "ok"}}
    fake_client.place_order.return_value = {"order": {"id": closing_order_id, "status": "pending"}}
    fake_client.get_order_status.return_value = {
        "order": {"id": closing_order_id, "status": "filled", "avg_fill_price": fill_price}
    }
    return OrderManager(
        client=fake_client, order_log=log, settings=_mk_settings(),
        poll_interval_seconds=0.001, poll_timeout_seconds=1.0,
        slippage_buffer=0.0,
    )


# ---------- the regression tests ----------

def test_close_long_straddle_at_credit_fill_records_correct_pnl():
    """Long straddle: paid $4.08 debit, sold to close at $2.00.
    Tradier returns avg_fill_price = -2.00 (credit fill, negative per convention).
    Expected realized P&L: -$408 + $200 = -$208."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_long_straddle_position()
        _seed_open_order(log, pos)
        # close_cash_flow > 0 → order_type = "credit"; mid = $2.00 per contract
        mark = _mk_mark(pos, close_cash_flow=200.0)

        manager = _mk_manager(log, fill_price=-2.00, closing_order_id=9001)
        result = asyncio.run(manager.submit_close(
            position=pos, mark=mark, exit_trigger="profit_target",
        ))

        assert result.status == "filled", f"expected filled, got {result.status}: {result.error}"

        today = datetime.now(timezone.utc).date()
        recorded = log.closed_today_pnl(today)
        expected = -208.0
        assert abs(recorded - expected) <= 5.0, (
            f"long straddle close P&L mismatch: recorded={recorded:+.2f} "
            f"expected={expected:+.2f} (sign-flip bug regression)"
        )
        log.close()
    print(f"long_straddle close: realized={recorded:+.2f} (expected {expected:+.2f}) ✓")


def test_close_short_iron_condor_at_debit_fill_records_correct_pnl():
    """Iron condor: sold $13.55 credit, bought back at $5.00.
    Tradier returns avg_fill_price = +5.00 (debit fill, positive).
    Expected realized P&L: +$1355 - $500 = +$855."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_iron_condor_position()
        _seed_open_order(log, pos)
        # close_cash_flow < 0 → order_type = "debit"; we pay $5 per contract
        mark = _mk_mark(pos, close_cash_flow=-500.0)

        manager = _mk_manager(log, fill_price=5.00, closing_order_id=9002)
        result = asyncio.run(manager.submit_close(
            position=pos, mark=mark, exit_trigger="profit_target",
        ))

        assert result.status == "filled", f"expected filled, got {result.status}: {result.error}"

        today = datetime.now(timezone.utc).date()
        recorded = log.closed_today_pnl(today)
        expected = 855.0
        assert abs(recorded - expected) <= 5.0, (
            f"iron condor close P&L mismatch: recorded={recorded:+.2f} "
            f"expected={expected:+.2f}"
        )
        log.close()
    print(f"iron_condor close: realized={recorded:+.2f} (expected {expected:+.2f}) ✓")


def test_close_long_straddle_loss_at_credit_fill():
    """Long straddle held to a partial loss: paid $4.08, sold for $3.00.
    Tradier fill = -3.00. Expected: -$408 + $300 = -$108."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_long_straddle_position()
        _seed_open_order(log, pos)
        mark = _mk_mark(pos, close_cash_flow=300.0)

        manager = _mk_manager(log, fill_price=-3.00, closing_order_id=9003)
        result = asyncio.run(manager.submit_close(
            position=pos, mark=mark, exit_trigger="expiration_proximity",
        ))

        assert result.status == "filled"
        recorded = log.closed_today_pnl(datetime.now(timezone.utc).date())
        expected = -108.0
        assert abs(recorded - expected) <= 5.0, (
            f"long straddle partial-loss P&L mismatch: recorded={recorded:+.2f} "
            f"expected={expected:+.2f}"
        )
        log.close()
    print(f"long_straddle partial-loss close: realized={recorded:+.2f} ✓")


def _mk_iron_condor_position_5lot() -> OpenPosition:
    legs = [
        TradeLeg(210.0, "call", "sell", 5, "NVDA260522C00210000"),
        TradeLeg(210.0, "put", "sell", 5, "NVDA260522P00210000"),
        TradeLeg(230.0, "call", "buy", 5, "NVDA260522C00230000"),
        TradeLeg(190.0, "put", "buy", 5, "NVDA260522P00190000"),
    ]
    return OpenPosition(
        tradier_order_id=5003, symbol="NVDA",
        expiration=date(2026, 5, 22), direction="SELL",
        structure="iron_condor", legs=legs, entry_premium=3.60,
        entry_atm_iv=0.40, entry_predicted_iv=0.32, entry_divergence=-0.08,
        entry_horizon_lower=1, entry_horizon_upper=1, entry_weight_lower=1.0,
        submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
    )


def test_close_multi_lot_condor_prices_per_unit_and_scales_pnl():
    """5-lot condor: sold for $3.60 credit per lot, closing at $3.00 per lot.
    mark.close_cash_flow is whole-position dollars (−$1,500); the close limit
    price must be the PER-UNIT $3.00 (the pre-fix code would submit $15.00 —
    a 5× overpay on a debit close). Realized P&L must scale by lots:
    (3.60 − 3.00) × 100 × 5 = +$300."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_iron_condor_position_5lot()
        _seed_open_order(log, pos)
        mark = _mk_mark(pos, close_cash_flow=-1500.0)  # pay $3.00/lot × 5 × 100

        manager = _mk_manager(log, fill_price=3.00, closing_order_id=9003)
        result = asyncio.run(manager.submit_close(
            position=pos, mark=mark, exit_trigger="profit_target",
        ))

        assert result.status == "filled", f"expected filled, got {result.status}: {result.error}"

        place_kwargs = manager._client.place_order.call_args.kwargs
        assert abs(place_kwargs["price"] - 3.00) < 0.01, (
            f"close limit must be per-unit $3.00, got {place_kwargs['price']}"
        )
        assert all(leg["quantity"] == 5 for leg in place_kwargs["legs"]), (
            f"close legs must carry the full 5-lot quantity: {place_kwargs['legs']}"
        )

        recorded = log.closed_today_pnl(datetime.now(timezone.utc).date())
        expected = 300.0
        assert abs(recorded - expected) <= 5.0, (
            f"multi-lot close P&L mismatch: recorded={recorded:+.2f} "
            f"expected={expected:+.2f}"
        )
        log.close()
    print(f"multi-lot condor close: per-unit price ✓ realized={recorded:+.2f} ✓")


def test_partial_close_rebooks_remainder_and_does_not_close_position():
    """A 5-lot close that fills only 2 units must NOT book the position
    closed: the residual close is canceled, the attempt is recorded as
    partially_filled (burning a retry), the opening order's legs shrink to
    the 3 units still held, and a stale-close alert row surfaces the event."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_iron_condor_position_5lot()
        _seed_open_order(log, pos)
        mark = _mk_mark(pos, close_cash_flow=-1500.0)

        fake_client = mock.AsyncMock()
        fake_client.preview_order.return_value = {"order": {"status": "ok"}}
        fake_client.place_order.return_value = {"order": {"id": 9100, "status": "pending"}}
        # Working partial during the submit poll; confirmed dead (canceled,
        # 2 units executed) once the residual cancel lands.
        working = {"order": {
            "id": 9100, "status": "partially_filled", "avg_fill_price": 3.00,
            "leg": [{"exec_quantity": 2}] * 4,
        }}
        dead = {"order": {
            "id": 9100, "status": "canceled", "avg_fill_price": 3.00,
            "leg": [{"exec_quantity": 2}] * 4,
        }}
        fake_client.get_order_status.side_effect = [working, dead, dead, dead]
        fake_client.cancel_order.return_value = {"order": {"status": "ok"}}
        manager = OrderManager(
            client=fake_client, order_log=log, settings=_mk_settings(),
            poll_interval_seconds=0.001, poll_timeout_seconds=1.0,
            slippage_buffer=0.0,
        )

        result = asyncio.run(manager.submit_close(
            position=pos, mark=mark, exit_trigger="profit_target",
        ))

        assert result.status == "partially_filled", result.status
        fake_client.cancel_order.assert_awaited()

        opening = log.get_submitted_order(pos.tradier_order_id)
        assert opening["closing_order_id"] is None, "position must remain open"
        import json as _json
        stored = _json.loads(opening["legs_json"])
        assert [l["quantity"] for l in stored] == [3, 3, 3, 3], stored

        attempt = log._conn.execute(
            "SELECT status, fill_price FROM close_attempts WHERE closing_order_id = 9100"
        ).fetchone()
        assert attempt[0] == "partially_filled", attempt
        alert = log._conn.execute(
            "SELECT last_exit_trigger FROM stale_close_alerts WHERE opening_order_id = ?",
            (pos.tradier_order_id,),
        ).fetchone()
        assert alert is not None and alert[0] == "partial_close:profit_target", alert
        log.close()
    print("partial_close: remainder rebooked at 3 lots, attempt failed, alert row ✓")


def test_between_cycle_partial_close_not_booked_as_full_close():
    """reconcile_pending_closes finds a pending close that Tradier reports
    partially_filled (2 of 5 units). It must go through the partial handler —
    residual canceled, opening rebooked at 3 lots, attempt failed — NOT
    through _reconcile_between_cycle_fill, which would book a full 5-lot
    close (overstating realized 2.5×) and orphan the 3 open units."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_iron_condor_position_5lot()
        _seed_open_order(log, pos)
        log.record_close_attempt(
            opening_order_id=pos.tradier_order_id, closing_order_id=9200,
            submitted_at=datetime.now(timezone.utc), exit_trigger="stop_loss",
            order_type="debit", submitted_price=3.00,
        )

        fake_client = mock.AsyncMock()
        working = {"order": {
            "id": 9200, "status": "partially_filled", "avg_fill_price": 3.00,
            "leg": [{"exec_quantity": 2}] * 4,
        }}
        dead = {"order": {
            "id": 9200, "status": "canceled", "avg_fill_price": 3.00,
            "leg": [{"exec_quantity": 2}] * 4,
        }}
        fake_client.get_order_status.side_effect = [working, dead, dead]
        fake_client.cancel_order.return_value = {"order": {"status": "ok"}}
        manager = OrderManager(
            client=fake_client, order_log=log, settings=_mk_settings(),
            poll_interval_seconds=0.001, poll_timeout_seconds=1.0,
        )

        counts = asyncio.run(manager.reconcile_pending_closes(datetime.now(timezone.utc)))

        assert counts["filled"] == 0, counts
        opening = log.get_submitted_order(pos.tradier_order_id)
        assert opening["closing_order_id"] is None, "position must remain open"
        import json as _json
        stored = _json.loads(opening["legs_json"])
        assert [l["quantity"] for l in stored] == [3, 3, 3, 3], stored
        attempt = log._conn.execute(
            "SELECT status FROM close_attempts WHERE closing_order_id = 9200"
        ).fetchone()
        assert attempt[0] == "partially_filled", attempt
        log.close()
    print("between-cycle partial close: handler path, remainder rebooked ✓")


def test_partial_close_cancel_unconfirmed_stays_pending():
    """If the close is still WORKING after the cancel attempt (cancel failed
    or raced), nothing may be booked: the attempt stays pending so the next
    reconcile pass re-queries it, and the opening order's legs are untouched."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        pos = _mk_iron_condor_position_5lot()
        _seed_open_order(log, pos)
        mark = _mk_mark(pos, close_cash_flow=-1500.0)

        fake_client = mock.AsyncMock()
        # Order remains partially_filled (working) on every query — the
        # cancel never demonstrably lands.
        fake_client.get_order_status.return_value = {"order": {
            "id": 9300, "status": "partially_filled", "avg_fill_price": 3.00,
            "leg": [{"exec_quantity": 2}] * 4,
        }}
        fake_client.preview_order.return_value = {"order": {"status": "ok"}}
        fake_client.place_order.return_value = {"order": {"id": 9300, "status": "pending"}}
        fake_client.cancel_order.side_effect = RuntimeError("connection dropped")
        manager = OrderManager(
            client=fake_client, order_log=log, settings=_mk_settings(),
            poll_interval_seconds=0.001, poll_timeout_seconds=1.0,
        )

        result = asyncio.run(manager.submit_close(
            position=pos, mark=mark, exit_trigger="stop_loss",
        ))

        assert result.status == "pending", result.status
        opening = log.get_submitted_order(pos.tradier_order_id)
        assert opening["closing_order_id"] is None
        import json as _json
        stored = _json.loads(opening["legs_json"])
        assert [l["quantity"] for l in stored] == [5, 5, 5, 5], (
            f"legs must be untouched while the order may still fill: {stored}"
        )
        attempt = log._conn.execute(
            "SELECT status FROM close_attempts WHERE closing_order_id = 9300"
        ).fetchone()
        assert attempt[0] == "pending", attempt
        log.close()
    print("cancel-unconfirmed partial close: left pending, legs untouched ✓")


def main() -> int:
    test_close_long_straddle_at_credit_fill_records_correct_pnl()
    test_close_short_iron_condor_at_debit_fill_records_correct_pnl()
    test_close_long_straddle_loss_at_credit_fill()
    test_close_multi_lot_condor_prices_per_unit_and_scales_pnl()
    test_partial_close_rebooks_remainder_and_does_not_close_position()
    test_between_cycle_partial_close_not_booked_as_full_close()
    test_partial_close_cancel_unconfirmed_stays_pending()
    print("all order_manager close tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
