"""Reconciler tests: log vs Tradier diff → expire / assign / hold."""
import asyncio
import json
import sys
import tempfile
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from data.async_client import Bar
from execution import OrderLog
from positions.reconciler import (
    AssignmentAlert,
    PositionReconciler,
    TimeoutResolution,
    is_option_symbol,
    max_loss_dollars,
    settle_intrinsic_pnl,
    underlying_of_option,
)
from signals.signal_generator import TradeLeg, TradeSignal


# Real OCC symbol for an AAPL 2025-06-20 $150 call.
AAPL_C150 = "AAPL250620C00150000"
AAPL_P140 = "AAPL250620P00140000"
AAPL_C155 = "AAPL250620C00155000"
AAPL_P135 = "AAPL250620P00135000"


def _bar(d: date, close: float) -> Bar:
    return Bar(date=d, open=close, high=close, low=close, close=close, volume=1)


def _mock_close(client, symbol_to_close: dict[str, float], expiration: date) -> None:
    """Configure client.get_history to return a single-bar history with the
    given close price for the expiration date, keyed by underlying symbol."""
    async def fake_history(symbol, start, end, interval="daily"):
        if symbol not in symbol_to_close:
            return []
        return [_bar(expiration, symbol_to_close[symbol])]
    client.get_history = mock.AsyncMock(side_effect=fake_history)


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
    """SELL iron condor expires with underlying inside both wings → all legs
    OTM → realized = +entry_credit (full credit kept)."""
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

        client = mock.AsyncMock()
        client.get_positions.return_value = []
        # IC has shorts at 150/140, longs at 155/135. Close at 145 → all OTM.
        _mock_close(client, {"AAPL": 145.0}, expiration)

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        today = expiration + timedelta(days=3)
        result = asyncio.run(reconciler.reconcile(today))

        assert result.expired_closed == [9001], result
        assert result.assignment_alerts == []
        assert result.skipped_premature == []
        assert len(log.open_unclosed_positions()) == 0
        row = log._conn.execute(
            "SELECT realized_pnl, closing_order_id, exit_trigger "
            "FROM submitted_orders WHERE tradier_order_id = ?", (9001,),
        ).fetchone()
        assert abs(row[0] - 120.0) < 0.01, f"expected +$120, got {row[0]}"
        assert row[1] == 0, "closing_order_id sentinel"
        # Trigger is "expired" for a non-worthless settlement; +credit IS the
        # full credit which is fine — old name "expired_worthless" reserved
        # for long debit-at-floor outcomes specifically.
        assert row[2] in ("expired", "expired_worthless")

        log.close()
    print("reconciler: short IC, all OTM → +entry_credit ✓")


def test_reconciler_marks_expired_long_straddle_atm_worthless():
    """Long straddle, underlying expires exactly at the strike → both legs
    have zero intrinsic → realized = -entry_debit."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 5, 15)
        entry_debit = 4.50  # $450 debit, strike 150
        _seed_open_order(
            log, order_id=9002, symbol="AAPL", expiration=expiration,
            direction="BUY", structure="straddle", entry_premium=entry_debit,
            legs=_straddle_legs(),
            submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        )

        client = mock.AsyncMock()
        client.get_positions.return_value = []
        _mock_close(client, {"AAPL": 150.0}, expiration)  # ATM → both legs worthless

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(expiration + timedelta(days=1)))

        assert result.expired_closed == [9002]
        row = log._conn.execute(
            "SELECT realized_pnl, exit_trigger FROM submitted_orders WHERE tradier_order_id = ?",
            (9002,),
        ).fetchone()
        assert abs(row[0] - (-450.0)) < 0.01, f"expected -$450, got {row[0]}"
        assert row[1] == "expired_worthless"

        log.close()
    print("reconciler: long straddle ATM at expiration → -entry_debit (worthless) ✓")


def test_reconciler_long_straddle_itm_call_settles_intrinsic():
    """**Headline regression test for the prod bug.** Long straddle, underlying
    expires $15 above the call strike → call intrinsic $15, put worthless.
    Realized must be +intrinsic*100 - entry_debit, not the full -entry_debit."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 5, 15)
        entry_debit = 4.50  # $450 debit
        _seed_open_order(
            log, order_id=9020, symbol="AAPL", expiration=expiration,
            direction="BUY", structure="straddle", entry_premium=entry_debit,
            legs=_straddle_legs(),  # both at strike 150
            submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        )

        client = mock.AsyncMock()
        client.get_positions.return_value = []
        _mock_close(client, {"AAPL": 165.0}, expiration)  # $15 above 150 strike

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(expiration + timedelta(days=1)))

        assert result.expired_closed == [9020]
        row = log._conn.execute(
            "SELECT realized_pnl, exit_trigger FROM submitted_orders WHERE tradier_order_id = ?",
            (9020,),
        ).fetchone()
        # intrinsic 15 × 100 × 1 = +1500; entry debit -450 → realized = +1050
        expected = 1500.0 - 450.0
        assert abs(row[0] - expected) < 0.01, (
            f"expected ${expected:+.2f} (intrinsic - debit), got ${row[0]:+.2f} — "
            f"if this is -$450 the worthless-bug is back"
        )
        assert row[1] == "expired", "ITM settlement should use 'expired' trigger, not '_worthless'"

        log.close()
    print(f"reconciler: long straddle ITM (call) → realized = intrinsic − debit ✓ "
          f"(${expected:+.2f})")


def test_reconciler_long_straddle_itm_put_settles_intrinsic():
    """Underlying expires below the strike → put intrinsic, call worthless."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 5, 15)
        entry_debit = 8.00
        _seed_open_order(
            log, order_id=9021, symbol="AAPL", expiration=expiration,
            direction="BUY", structure="straddle", entry_premium=entry_debit,
            legs=_straddle_legs(),  # strike 150
            submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        )

        client = mock.AsyncMock()
        client.get_positions.return_value = []
        _mock_close(client, {"AAPL": 138.0}, expiration)  # $12 below 150 strike

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(expiration + timedelta(days=1)))

        assert result.expired_closed == [9021]
        row = log._conn.execute(
            "SELECT realized_pnl FROM submitted_orders WHERE tradier_order_id = ?",
            (9021,),
        ).fetchone()
        # put intrinsic = 12 × 100 = $1200; debit = $800 → realized = +$400
        assert abs(row[0] - 400.0) < 0.01, f"expected +$400, got {row[0]}"

        log.close()
    print("reconciler: long straddle ITM (put) → realized = intrinsic − debit ✓")


def test_reconciler_short_ic_breached_short_call_long_covered():
    """Short iron condor where underlying expires between the short call (150)
    and long call (155) → short call ITM by $X, long call still OTM. Tradier
    would assign the short and leave the long; if cash-settled or auto-resolved
    without leaving stock, realized = credit − short_call_payout."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 5, 15)
        entry_credit = 2.00  # $200 credit
        _seed_open_order(
            log, order_id=9030, symbol="AAPL", expiration=expiration,
            direction="SELL", structure="iron_condor", entry_premium=entry_credit,
            legs=_ic_legs(),  # shorts 150/140, longs 155/135
            submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        )

        client = mock.AsyncMock()
        client.get_positions.return_value = []  # no stock left (cash-settled)
        _mock_close(client, {"AAPL": 153.0}, expiration)  # $3 above short call

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(expiration + timedelta(days=1)))

        assert result.expired_closed == [9030]
        row = log._conn.execute(
            "SELECT realized_pnl FROM submitted_orders WHERE tradier_order_id = ?",
            (9030,),
        ).fetchone()
        # credit +200, short call payout -3×100 = -300; long call OTM (close 153 < strike 155)
        # → realized = +200 - 300 = -100
        assert abs(row[0] - (-100.0)) < 0.01, f"expected -$100, got {row[0]}"

        log.close()
    print("reconciler: short IC, short call breached → realized = credit − payout ✓")


def test_reconciler_short_ic_max_loss_both_call_legs_itm():
    """Both short and long call legs ITM → capped max loss at wing distance."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 5, 15)
        entry_credit = 1.20
        _seed_open_order(
            log, order_id=9031, symbol="AAPL", expiration=expiration,
            direction="SELL", structure="iron_condor", entry_premium=entry_credit,
            legs=_ic_legs(),  # 150/155 call wings, 140/135 put wings
            submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        )

        client = mock.AsyncMock()
        client.get_positions.return_value = []
        _mock_close(client, {"AAPL": 170.0}, expiration)  # way above long call

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(expiration + timedelta(days=1)))

        assert result.expired_closed == [9031]
        row = log._conn.execute(
            "SELECT realized_pnl FROM submitted_orders WHERE tradier_order_id = ?",
            (9031,),
        ).fetchone()
        # short call: -20×100 = -2000; long call: +15×100 = +1500; net leg = -500
        # credit +120; realized = -500 + 120 = -380. Max loss bound = -(5×100 - 120) = -380. ✓
        assert abs(row[0] - (-380.0)) < 0.01, f"expected -$380, got {row[0]}"
        floor = max_loss_dollars(json.loads(log._conn.execute(
            "SELECT legs_json FROM submitted_orders WHERE tradier_order_id=?", (9031,)
        ).fetchone()[0]), "SELL", entry_credit)
        assert abs(row[0] - floor) < 0.01, "should hit defined-risk floor exactly"

        log.close()
    print("reconciler: short IC, both call legs ITM → capped max loss = wing − credit ✓")


def test_reconciler_defers_closure_when_history_unavailable():
    """If get_history fails / returns no bars, do NOT auto-close — defer."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 5, 15)
        _seed_open_order(
            log, order_id=9040, symbol="AAPL", expiration=expiration,
            direction="BUY", structure="straddle", entry_premium=4.50,
            legs=_straddle_legs(),
            submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        )
        client = mock.AsyncMock()
        client.get_positions.return_value = []
        client.get_history = mock.AsyncMock(side_effect=RuntimeError("Tradier 500"))

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(expiration + timedelta(days=1)))

        assert result.expired_closed == []
        assert 9040 in result.skipped_premature
        assert len(log.open_unclosed_positions()) == 1  # still open
        log.close()
    print("reconciler: get_history failure → defer closure, retry next cycle ✓")


def test_reconciler_underlying_close_cached_within_one_call():
    """Multiple positions on the same (underlying, expiration) should fetch
    the close price only once per reconcile() call."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 5, 15)
        for i, oid in enumerate((9050, 9051, 9052)):
            _seed_open_order(
                log, order_id=oid, symbol="AAPL", expiration=expiration,
                direction="BUY", structure="straddle", entry_premium=4.50,
                legs=_straddle_legs(),
                submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
            )
        client = mock.AsyncMock()
        client.get_positions.return_value = []
        call_count = 0
        async def counting_history(symbol, start, end, interval="daily"):
            nonlocal call_count
            call_count += 1
            return [_bar(expiration, 150.0)]
        client.get_history = mock.AsyncMock(side_effect=counting_history)

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        asyncio.run(reconciler.reconcile(expiration + timedelta(days=1)))

        assert call_count == 1, f"expected 1 get_history call (cached), got {call_count}"
        log.close()
    print("reconciler: (symbol, exp) close-price cache holds within one cycle ✓")


def test_settle_intrinsic_pnl_unit():
    """Direct unit checks on the settlement helper."""
    long_straddle = [
        {"strike": 100.0, "option_type": "call", "side": "buy", "quantity": 1},
        {"strike": 100.0, "option_type": "put", "side": "buy", "quantity": 1},
    ]
    # ATM: both 0
    assert settle_intrinsic_pnl(long_straddle, 100.0, "BUY", 5.00) == -500.0
    # Call $10 ITM, put OTM: +1000 - 500 = +500
    assert settle_intrinsic_pnl(long_straddle, 110.0, "BUY", 5.00) == 500.0
    # Put $7 ITM, call OTM: +700 - 500 = +200
    assert settle_intrinsic_pnl(long_straddle, 93.0, "BUY", 5.00) == 200.0
    # Quantity scaling: entry_premium is PER LOT (Tradier multileg prices are
    # per unit), so the entry debit scales with the lot count alongside the
    # payoff — 2 lots: payoff 2×$1,000, debit 2×$500.
    long_qty2 = [{**l, "quantity": 2} for l in long_straddle]
    assert settle_intrinsic_pnl(long_qty2, 110.0, "BUY", 5.00) == 2000.0 - 1000.0

    short_ic = [
        {"strike": 150.0, "option_type": "call", "side": "sell", "quantity": 1},
        {"strike": 155.0, "option_type": "call", "side": "buy", "quantity": 1},
        {"strike": 140.0, "option_type": "put", "side": "sell", "quantity": 1},
        {"strike": 135.0, "option_type": "put", "side": "buy", "quantity": 1},
    ]
    # All OTM: +credit
    assert settle_intrinsic_pnl(short_ic, 145.0, "SELL", 1.20) == 120.0
    # Short call $3 ITM, long call OTM: 120 - 300 = -180
    assert settle_intrinsic_pnl(short_ic, 153.0, "SELL", 1.20) == -180.0
    # Both call legs ITM (max-loss territory): 120 - 2000 + 1500 = -380
    assert settle_intrinsic_pnl(short_ic, 170.0, "SELL", 1.20) == -380.0
    print("settle_intrinsic_pnl: all unit cases pass ✓")


def test_reconciler_expiry_fees_netted_and_floor_check_stays_gross():
    """per_contract_fee on expiry settlement: realized = gross intrinsic −
    fee × Σ leg quantities (open side only — expired legs pay no close fees).
    CRITICAL invariant: the max-loss floor sanity check and the trigger
    comparison stay on the GROSS number, so a max-loss expiry with fees on
    nets BELOW the theoretical floor and must still be recorded, not skipped
    as a floor breach."""
    fee = 0.45

    # Scenario 1: short IC expires all OTM → gross +$120, minus 4 × $0.45.
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 5, 15)
        _seed_open_order(
            log, order_id=9060, symbol="AAPL", expiration=expiration,
            direction="SELL", structure="iron_condor", entry_premium=1.20,
            legs=_ic_legs(),
            submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        )
        client = mock.AsyncMock()
        client.get_positions.return_value = []
        _mock_close(client, {"AAPL": 145.0}, expiration)  # all 4 legs OTM

        reconciler = PositionReconciler(client=client, order_log=log,
                                        account_id="VA1", per_contract_fee=fee)
        result = asyncio.run(reconciler.reconcile(expiration + timedelta(days=1)))

        assert result.expired_closed == [9060], result
        row = log._conn.execute(
            "SELECT realized_pnl FROM submitted_orders WHERE tradier_order_id = ?",
            (9060,),
        ).fetchone()
        expected = 120.0 - 4 * fee  # $118.20
        assert abs(row[0] - expected) < 0.01, f"expected {expected:+.2f}, got {row[0]}"
        log.close()

    # Scenario 2: max-loss expiry. Gross = −$380, exactly the defined-risk
    # floor; net −$381.80 sits below it. Must still be marked closed — if
    # this lands in skipped_premature, the floor check is (wrongly) netting.
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 5, 15)
        entry_credit = 1.20
        _seed_open_order(
            log, order_id=9061, symbol="AAPL", expiration=expiration,
            direction="SELL", structure="iron_condor", entry_premium=entry_credit,
            legs=_ic_legs(),  # 150/155 call wings → max loss (5 − 1.20) × 100
            submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        )
        client = mock.AsyncMock()
        client.get_positions.return_value = []
        _mock_close(client, {"AAPL": 170.0}, expiration)  # both call legs ITM

        reconciler = PositionReconciler(client=client, order_log=log,
                                        account_id="VA1", per_contract_fee=fee)
        result = asyncio.run(reconciler.reconcile(expiration + timedelta(days=1)))

        assert result.expired_closed == [9061], (
            f"max-loss expiry with fees skipped as floor breach: {result}"
        )
        assert result.skipped_premature == []
        row = log._conn.execute(
            "SELECT realized_pnl, exit_trigger FROM submitted_orders WHERE tradier_order_id = ?",
            (9061,),
        ).fetchone()
        expected = -380.0 - 4 * fee  # −$381.80, below the gross floor of −$380
        assert abs(row[0] - expected) < 0.01, f"expected {expected:+.2f}, got {row[0]}"
        floor = max_loss_dollars([asdict(l) for l in _ic_legs()], "SELL", entry_credit)
        assert row[0] < floor, "net must land below the gross-only floor"
        assert row[1] == "expired"  # SELL never gets 'expired_worthless'
        log.close()
    print("reconciler: expiry fees netted (open side only), floor check stays gross ✓")


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


def test_timeout_recovery_expires_stuck_pending_past_expiration():
    """An order wedged non-terminal at Tradier (sandbox PendingNew) with zero
    executed quantity, whose legs' expiration has passed, can never fill —
    resolve it as 'expired' instead of warning every cycle forever."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        _seed_open_order(
            log, order_id=9106, symbol="UNH", expiration=date(2026, 5, 15),
            direction="BUY", structure="straddle", entry_premium=14.66,
            legs=_straddle_legs(),
            submitted_at=datetime(2026, 5, 13, 15, 0, tzinfo=timezone.utc),
            final_status="timeout",
        )
        client = mock.AsyncMock()
        client.get_positions.return_value = []
        client.get_order_status.return_value = {
            "order": {"id": 9106, "status": "pending", "exec_quantity": 0.0},
        }

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(date(2026, 5, 20)))

        assert result.timeouts_resolved[0].new_status == "expired"
        assert result.timeouts_resolved[0].fill_price is None
        assert len(log.timeout_orders()) == 0  # resolved, no more warnings
        # Never filled — must not enter open positions or expiration P&L.
        assert len(log.open_unclosed_positions()) == 0
        assert result.expired_closed == []
        log.close()
    print("timeout_recovery: stuck pending past expiration → expired ✓")


def test_timeout_recovery_keeps_stuck_pending_on_expiration_day():
    """Same wedge but on expiration day itself — the order could in theory
    still fill, so leave it as timeout until the day is over."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        _seed_open_order(
            log, order_id=9107, symbol="UNH", expiration=date(2026, 5, 15),
            direction="BUY", structure="straddle", entry_premium=14.66,
            legs=_straddle_legs(),
            submitted_at=datetime(2026, 5, 13, 15, 0, tzinfo=timezone.utc),
            final_status="timeout",
        )
        client = mock.AsyncMock()
        client.get_positions.return_value = []
        client.get_order_status.return_value = {
            "order": {"id": 9107, "status": "pending", "exec_quantity": 0.0},
        }

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(date(2026, 5, 15)))

        assert result.timeouts_resolved[0].new_status == "unknown"
        assert len(log.timeout_orders()) == 1  # untouched
        log.close()
    print("timeout_recovery: stuck pending on expiration day → keep timeout ✓")


def test_timeout_recovery_keeps_stuck_pending_with_partial_exec():
    """Non-terminal past expiration but with executed quantity — something
    partially filled, needs the normal timeout path, not auto-expiry."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        _seed_open_order(
            log, order_id=9108, symbol="UNH", expiration=date(2026, 5, 15),
            direction="BUY", structure="straddle", entry_premium=14.66,
            legs=_straddle_legs(),
            submitted_at=datetime(2026, 5, 13, 15, 0, tzinfo=timezone.utc),
            final_status="timeout",
        )
        client = mock.AsyncMock()
        client.get_positions.return_value = []
        client.get_order_status.return_value = {
            "order": {"id": 9108, "status": "pending", "exec_quantity": 1.0},
        }

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(date(2026, 5, 20)))

        assert result.timeouts_resolved[0].new_status == "unknown"
        assert len(log.timeout_orders()) == 1  # untouched
        log.close()
    print("timeout_recovery: partial exec past expiration → keep timeout ✓")


def test_timeout_recovery_chains_into_expiration_marking():
    """End-to-end: timeout order that filled AND expired ITM. Single
    reconcile() call must recover the timeout, fetch underlying close,
    settle at intrinsic value. The exact UNH 5/15 scenario from prod —
    with UNH closing $5 below the strike, the put has intrinsic and the
    realized is NOT a full debit loss."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "orders.db")
        expiration = date(2026, 5, 15)
        _seed_open_order(
            log, order_id=9105, symbol="UNH", expiration=expiration,
            direction="BUY", structure="straddle", entry_premium=14.66,
            legs=_straddle_legs(),  # strike 150
            submitted_at=datetime(2026, 5, 4, 15, 0, tzinfo=timezone.utc),
            final_status="timeout",
        )

        client = mock.AsyncMock()
        client.get_order_status.return_value = {
            "order": {"id": 9105, "status": "filled", "avg_fill_price": 14.55},
        }
        client.get_positions.return_value = []
        # UNH expires at 145 → put 5 ITM, call worthless. Intrinsic 5*100 = 500.
        # Realized = 500 - 1455 = -955.
        _mock_close(client, {"UNH": 145.0}, expiration)

        reconciler = PositionReconciler(client=client, order_log=log, account_id="VA1")
        result = asyncio.run(reconciler.reconcile(date(2026, 5, 20)))

        assert result.timeouts_resolved[0].new_status == "filled"
        assert result.expired_closed == [9105]
        row = log._conn.execute(
            "SELECT realized_pnl, closing_order_id, exit_trigger, final_status, fill_price "
            "FROM submitted_orders WHERE tradier_order_id = ?", (9105,),
        ).fetchone()
        assert abs(row[0] - (-955.0)) < 0.01, f"expected -$955, got {row[0]}"
        assert row[1] == 0
        assert row[2] == "expired"  # not "expired_worthless" — ITM settled
        assert row[3] == "filled"
        assert abs(row[4] - 14.55) < 1e-9

        log.close()
    print("timeout_recovery: recovered fill → settled at intrinsic, not worthless ✓")


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
    test_settle_intrinsic_pnl_unit()
    test_reconciler_marks_expired_short_iron_condor()
    test_reconciler_marks_expired_long_straddle_atm_worthless()
    test_reconciler_long_straddle_itm_call_settles_intrinsic()
    test_reconciler_long_straddle_itm_put_settles_intrinsic()
    test_reconciler_short_ic_breached_short_call_long_covered()
    test_reconciler_short_ic_max_loss_both_call_legs_itm()
    test_reconciler_defers_closure_when_history_unavailable()
    test_reconciler_underlying_close_cached_within_one_call()
    test_reconciler_expiry_fees_netted_and_floor_check_stays_gross()
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
