"""Unit tests for MainLoop.run_once with all components mocked."""
import asyncio
import sys
from datetime import date, datetime, timezone
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

    store = mock.MagicMock()
    order_log = mock.MagicMock()
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
        "position_reconciler": position_reconciler,
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
    approved_decision = mock.MagicMock(approved=True, signal=fake_signal)
    loop, mocks = _mk_loop(
        actionable_signals=[fake_signal],
        risk_decisions=[approved_decision],
    )
    result = asyncio.run(loop.run_once())
    mocks["order_manager"].submit.assert_called_once()
    mocks["risk_rejection_log"].record_rejection.assert_not_called()
    print("approved_signal: order_manager.submit called once")


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
    result = asyncio.run(loop.run_once())
    assert result.kill_switch_active is True
    # Exit logic MUST still have run
    mocks["exit_manager"].evaluate.assert_called_once()
    mocks["exit_manager"].execute.assert_called_once()
    # Signal generation skipped
    mocks["sig_gen"].generate.assert_not_called()
    print("kill_switch_active + exits: exits ran, signal gen skipped")


def main() -> int:
    test_market_closed_returns_early()
    test_normal_cycle_with_no_signals()
    test_approved_signal_submits_order()
    test_rejected_signal_logs_rejection_no_submit()
    test_kill_switch_active_skips_signal_generation()
    test_run_once_calls_reconcile_pending_closes_before_snapshot()
    test_kill_switch_active_still_runs_exits()
    print("all main_loop tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
