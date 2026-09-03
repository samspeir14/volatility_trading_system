"""Unit tests for MainLoop.run_once with all components mocked."""
import asyncio
import sys
from datetime import date, datetime, timedelta, timezone
from unittest import mock

import pandas as pd

import main as main_module
from main import CycleResult, MainLoop


def _mk_loop(*, market_state="open", kill_active_initial=False, snapshot=None,
             actionable_signals=None, risk_decisions=None, exit_decisions=None):
    """Build a MainLoop with mocked components."""
    client = mock.AsyncMock()
    client.get_clock.return_value = {"state": market_state, "next_change": "2026-04-28T20:00:00Z"}

    market_data = mock.AsyncMock()
    scan = mock.MagicMock()
    scan.fetched_at = datetime(2026, 4, 28, 15, 0, tzinfo=timezone.utc)
    scan.total_contracts = 1000
    scan.snapshots = {}
    scan.__getitem__ = lambda self, k: scan.snapshots.get(k, mock.MagicMock())
    market_data.scan.return_value = scan
    # Augmentation is a no-op by default: no open positions, scan unchanged.
    market_data.fetch_missing_position_chains.return_value = scan

    if snapshot is None:
        snapshot = mock.MagicMock()
        snapshot.equity = 100000.0
        snapshot.starting_equity_today = 100000.0
        snapshot.today_total_pnl = 0.0
        snapshot.today_realized_pnl = 0.0
        snapshot.today_unrealized_pnl = 0.0
        snapshot.open_marks = []
        snapshot.open_positions = []
    builder = mock.AsyncMock()
    builder.snapshot.return_value = snapshot

    kill_switch = mock.MagicMock()
    kill_switch.evaluate_and_maybe_trigger.return_value = kill_active_initial

    sig_gen = mock.MagicMock()
    sig_gen.generate.return_value = (actionable_signals or [], actionable_signals or [])

    risk_manager = mock.MagicMock()
    risk_manager.gate.return_value = risk_decisions or []

    order_manager = mock.AsyncMock()
    order_manager.submit.return_value = mock.MagicMock(status="filled")
    order_manager.reconcile_pending_closes = mock.AsyncMock(return_value={
        "canceled": 0, "filled": 0, "failed_terminal": 0,
    })

    exit_manager = mock.MagicMock()
    exit_manager.evaluate.return_value = exit_decisions or []
    exit_manager.execute = mock.AsyncMock(return_value=[(d, mock.MagicMock(status="filled"))
                                                          for d in (exit_decisions or [])])

    feature_pipeline = mock.MagicMock()
    feature_pipeline._watchlist = []
    feature_pipeline.build_features.return_value = pd.DataFrame()
    feature_pipeline.ensure_data = mock.AsyncMock()

    store = mock.MagicMock()
    order_log = mock.MagicMock()
    order_log.open_unclosed_positions.return_value = []
    risk_rejection_log = mock.MagicMock()
    div_history = mock.MagicMock()
    summary_builder = mock.MagicMock()
    position_tracker = mock.MagicMock()
    position_reconciler = mock.MagicMock()
    position_reconciler.reconcile = mock.AsyncMock(return_value=mock.MagicMock(
        expired_closed=[], assignment_alerts=[], skipped_premature=[],
    ))

    loop = MainLoop(
        settings=mock.MagicMock(),
        client=client,
        store=store,
        order_log=order_log,
        kill_switch=kill_switch,
        risk_rejection_log=risk_rejection_log,
        divergence_history=div_history,
        market_data=market_data,
        feature_pipeline=feature_pipeline,
        signal_generator=sig_gen,
        risk_manager=risk_manager,
        order_manager=order_manager,
        exit_manager=exit_manager,
        position_tracker=position_tracker,
        position_reconciler=position_reconciler,
        portfolio_state_builder=builder,
        daily_summary_builder=summary_builder,
        slack_webhook_url=None,
        scan_interval_seconds=300,
    )
    return loop, {
        "client": client, "market_data": market_data, "builder": builder,
        "kill_switch": kill_switch, "sig_gen": sig_gen, "risk_manager": risk_manager,
        "order_manager": order_manager, "exit_manager": exit_manager,
        "risk_rejection_log": risk_rejection_log, "summary_builder": summary_builder,
        "position_reconciler": position_reconciler, "feature_pipeline": feature_pipeline,
    }


def test_market_closed_returns_early():
    loop, mocks = _mk_loop(market_state="closed")
    result = asyncio.run(loop.run_once())
    assert result.market_open is False
    # Scan should NOT have been called
    mocks["market_data"].scan.assert_not_called()
    print("market_closed: returns early without scanning")


def test_normal_cycle_with_no_signals():
    loop, mocks = _mk_loop()
    result = asyncio.run(loop.run_once())
    assert result.market_open is True
    assert result.signals_total == 0
    assert result.signals_approved == 0
    assert result.exits_evaluated == 0
    mocks["sig_gen"].generate.assert_called_once()
    mocks["order_manager"].submit.assert_not_called()
    print("normal_cycle: market open, scan + snapshot + signal-gen, no actions")


def test_approved_signal_submits_order():
    fake_signal = mock.MagicMock()
    fake_signal.symbol = "NVDA"
    approved_decision = mock.MagicMock(
        approved=True, signal=fake_signal, quantity=1,
    )
    loop, mocks = _mk_loop(
        actionable_signals=[fake_signal],
        risk_decisions=[approved_decision],
    )
    # run_once gates submission on the real wall clock (9:45-15:30 ET);
    # pin the window open so this test is deterministic whenever it runs.
    with mock.patch("main._within_entry_window", return_value=True):
        result = asyncio.run(loop.run_once())
    mocks["order_manager"].submit.assert_called_once()
    mocks["risk_rejection_log"].record_rejection.assert_not_called()
    print("approved_signal: order_manager.submit called once")


def test_approved_multi_lot_decision_scales_signal_legs():
    """The risk manager's sized quantity is applied at submission: a
    quantity=3 approval must reach order_manager.submit with every leg's
    quantity multiplied by 3, on a copy (the original signal is untouched)."""
    from signals.signal_generator import TradeLeg, TradeSignal

    legs = [
        TradeLeg(100.0, "call", "buy", 1, "NVDA260904C00100000"),
        TradeLeg(100.0, "put", "buy", 1, "NVDA260904P00100000"),
    ]
    real_signal = TradeSignal(
        symbol="NVDA", expiration=date(2026, 9, 4), dte=10,
        horizon_lower=1, horizon_upper=1, weight_lower=1.0,
        direction="BUY", underlying_price=100.0, atm_iv=0.25,
        predicted_iv_equivalent=0.35, divergence=0.10,
        cross_sectional_z=1.0, time_series_z=None, liquidity_score=1.0,
        legs=legs, is_actionable=True,
    )
    approved_decision = mock.MagicMock(
        approved=True, signal=real_signal, quantity=3,
        projected_max_loss=1200.0,
    )
    loop, mocks = _mk_loop(
        actionable_signals=[real_signal],
        risk_decisions=[approved_decision],
    )
    with mock.patch("main._within_entry_window", return_value=True):
        asyncio.run(loop.run_once())

    submitted = mocks["order_manager"].submit.call_args.args[0]
    assert [l.quantity for l in submitted.legs] == [3, 3], submitted.legs
    assert [l.quantity for l in real_signal.legs] == [1, 1], "original mutated"
    print("approved_multi_lot: legs scaled to sized quantity at submit")


def test_approved_signal_held_outside_entry_window():
    """Outside 9:45-15:30 ET an approved signal is HELD: no submission, but
    also no risk-rejection row (it wasn't rejected, just deferred)."""
    fake_signal = mock.MagicMock()
    fake_signal.symbol = "NVDA"
    approved_decision = mock.MagicMock(approved=True, signal=fake_signal)
    loop, mocks = _mk_loop(
        actionable_signals=[fake_signal],
        risk_decisions=[approved_decision],
    )
    with mock.patch("main._within_entry_window", return_value=False):
        result = asyncio.run(loop.run_once())
    mocks["order_manager"].submit.assert_not_called()
    mocks["risk_rejection_log"].record_rejection.assert_not_called()
    assert result.signals_approved == 0
    print("entry_window: approved signal held outside window, no rejection logged")


def test_rejected_signal_logs_rejection_no_submit():
    fake_signal = mock.MagicMock(symbol="NVDA")
    rejected_decision = mock.MagicMock(approved=False, signal=fake_signal,
                                        reasons=["test rejection"])
    loop, mocks = _mk_loop(
        actionable_signals=[fake_signal],
        risk_decisions=[rejected_decision],
    )
    result = asyncio.run(loop.run_once())
    mocks["order_manager"].submit.assert_not_called()
    mocks["risk_rejection_log"].record_rejection.assert_called_once()
    print("rejected_signal: rejection logged, no order submitted")


def test_kill_switch_active_skips_signal_generation():
    fake_signal = mock.MagicMock(symbol="NVDA")
    decision = mock.MagicMock(approved=True, signal=fake_signal)
    loop, mocks = _mk_loop(
        kill_active_initial=True,
        actionable_signals=[fake_signal],
        risk_decisions=[decision],
    )
    result = asyncio.run(loop.run_once())
    assert result.kill_switch_active is True
    # Signal generation should NOT have been called
    mocks["sig_gen"].generate.assert_not_called()
    mocks["risk_manager"].gate.assert_not_called()
    mocks["order_manager"].submit.assert_not_called()
    print("kill_switch_active: signal generation skipped")


def test_run_once_calls_reconcile_pending_closes_before_snapshot():
    """The stale-close reconciler must run on every cycle BEFORE snapshot, so
    any between-cycle fills or canceled stale closes are reflected in the
    snapshot used for exits + signal gen."""
    loop, mocks = _mk_loop()
    asyncio.run(loop.run_once())
    mocks["order_manager"].reconcile_pending_closes.assert_called_once()

    # And the call must happen before the snapshot builder runs.
    call_order = []
    snapshot_obj = mocks["builder"].snapshot.return_value
    async def record_reconcile(*args, **kwargs):
        call_order.append("reconcile")
        return {"canceled": 0, "filled": 0, "failed_terminal": 0}
    async def record_snapshot(*args, **kwargs):
        call_order.append("snapshot")
        return snapshot_obj
    mocks["order_manager"].reconcile_pending_closes = mock.AsyncMock(side_effect=record_reconcile)
    mocks["builder"].snapshot = mock.AsyncMock(side_effect=record_snapshot)
    asyncio.run(loop.run_once())
    assert call_order == ["reconcile", "snapshot"], (
        f"expected reconcile before snapshot, got {call_order}"
    )
    print("run_once: reconcile_pending_closes called before snapshot")


def test_run_once_augments_scan_and_feeds_it_to_snapshot():
    """Near-expiry positions below the scan window are pulled in by
    fetch_missing_position_chains, and the augmented scan — not the raw one —
    must be what snapshot/exits/signals see."""
    loop, mocks = _mk_loop()

    # open_unclosed_positions drives the expirations we ask the augmenter for.
    mocks_order_log = loop._order_log
    mocks_order_log.open_unclosed_positions.return_value = [
        {"symbol": "AAPL", "expiration": "2026-04-29"},
        {"symbol": "AAPL", "expiration": "bad-date"},  # skipped, no crash
    ]

    augmented_scan = mock.MagicMock(name="augmented_scan")
    mocks["market_data"].fetch_missing_position_chains.return_value = augmented_scan

    asyncio.run(loop.run_once())

    # Augmenter was handed the per-symbol expiration sets from the order log.
    mocks["market_data"].fetch_missing_position_chains.assert_called_once()
    call = mocks["market_data"].fetch_missing_position_chains.call_args
    passed_needed = call.args[1]
    assert passed_needed == {"AAPL": {date(2026, 4, 29)}}

    # Downstream consumers receive the augmented scan, not the original.
    assert mocks["builder"].snapshot.call_args.args[0] is augmented_scan
    print("run_once: augmented scan flows to snapshot")


def test_kill_switch_active_still_runs_exits():
    """Critical: dropdown shouldn't compound by holding losers — exits must still run."""
    fake_position_mark = mock.MagicMock()
    snapshot = mock.MagicMock()
    snapshot.equity = 100000.0
    snapshot.starting_equity_today = 100000.0
    snapshot.today_total_pnl = -4000.0
    snapshot.today_realized_pnl = 0.0
    snapshot.today_unrealized_pnl = -4000.0
    snapshot.open_marks = [fake_position_mark]
    snapshot.open_positions = [mock.MagicMock()]

    fake_exit_decision = mock.MagicMock(action="close", trigger="stop_loss")
    loop, mocks = _mk_loop(
        kill_active_initial=True,
        snapshot=snapshot,
        exit_decisions=[fake_exit_decision],
    )
    with mock.patch("main._exits_allowed", return_value=True):
        result = asyncio.run(loop.run_once())
    assert result.kill_switch_active is True
    # Exit logic MUST still have run
    mocks["exit_manager"].evaluate.assert_called_once()
    mocks["exit_manager"].execute.assert_called_once()
    # Signal generation skipped
    mocks["sig_gen"].generate.assert_not_called()
    print("kill_switch_active + exits: exits ran, signal gen skipped")


def test_close_decisions_held_before_exit_window():
    """Before 09:45 ET exits are evaluated but not submitted: no order goes
    out at the open's prices; the next in-window cycle re-evaluates."""
    fake_position_mark = mock.MagicMock()
    snapshot = mock.MagicMock()
    snapshot.equity = 100000.0
    snapshot.starting_equity_today = 100000.0
    snapshot.today_total_pnl = 0.0
    snapshot.today_realized_pnl = 0.0
    snapshot.today_unrealized_pnl = 0.0
    snapshot.open_marks = [fake_position_mark]
    snapshot.open_positions = [mock.MagicMock()]
    fake_exit_decision = mock.MagicMock(action="close", trigger="assignment_risk")
    loop, mocks = _mk_loop(snapshot=snapshot, exit_decisions=[fake_exit_decision])
    with mock.patch("main._exits_allowed", return_value=False):
        result = asyncio.run(loop.run_once())
    mocks["exit_manager"].evaluate.assert_called_once()
    mocks["exit_manager"].execute.assert_not_called()
    assert result.exits_evaluated == 1
    print("exit_window: close decision held before 09:45 ET, nothing submitted")


def test_exits_allowed_boundary():
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    assert main_module._exits_allowed(datetime(2026, 9, 3, 9, 44, tzinfo=et)) is False
    assert main_module._exits_allowed(datetime(2026, 9, 3, 9, 45, tzinfo=et)) is True
    assert main_module._exits_allowed(datetime(2026, 9, 3, 15, 59, tzinfo=et)) is True
    print("exit_window: opens at 09:45 ET with no close bound")


def test_cycle_refreshes_daily_bars_through_yesterday():
    loop, mocks = _mk_loop()
    mocks["feature_pipeline"].ensure_data = mock.AsyncMock()
    result = asyncio.run(loop.run_once())
    assert result.market_open is True
    mocks["feature_pipeline"].ensure_data.assert_awaited_once()
    end = mocks["feature_pipeline"].ensure_data.await_args.kwargs["end"]
    # run_once derives "yesterday" from the UTC clock (cycles only execute
    # during US market hours, when the UTC and Eastern dates agree).
    utc_today = datetime.now(timezone.utc).date()
    assert end == main_module._last_weekday(utc_today - timedelta(days=1))
    assert end.weekday() < 5, f"end must be a weekday, got {end}"
    print(f"bar refresh: ensure_data awaited once with end={end}")


def test_cycle_survives_bar_refresh_failure():
    loop, mocks = _mk_loop()
    mocks["feature_pipeline"].ensure_data = mock.AsyncMock(
        side_effect=RuntimeError("api down")
    )
    result = asyncio.run(loop.run_once())
    assert result.market_open is True
    assert result.error is None
    mocks["sig_gen"].generate.assert_called_once()
    print("bar refresh fail-soft: cycle completed on cached bars")


def main() -> int:
    test_market_closed_returns_early()
    test_normal_cycle_with_no_signals()
    test_approved_signal_submits_order()
    test_approved_multi_lot_decision_scales_signal_legs()
    test_approved_signal_held_outside_entry_window()
    test_rejected_signal_logs_rejection_no_submit()
    test_kill_switch_active_skips_signal_generation()
    test_run_once_calls_reconcile_pending_closes_before_snapshot()
    test_kill_switch_active_still_runs_exits()
    test_close_decisions_held_before_exit_window()
    test_exits_allowed_boundary()
    test_cycle_refreshes_daily_bars_through_yesterday()
    test_cycle_survives_bar_refresh_failure()
    print("all main_loop tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
