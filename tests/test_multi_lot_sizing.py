"""Multi-lot sizing regression tests. Since 2026-08-25 the risk manager's
max-loss-normalized quantity is applied to orders (legs scale uniformly), so
every dollar conversion that treated entry_premium / fill_price as
whole-position values must scale by the lot count. These tests pin the
per-lot → whole-position math in the tracker, exit thresholds, reconciler
settlement, close realized P&L, and the order-request premium cap."""
import sys
from datetime import date, datetime, timezone
from unittest import mock

from data.async_client import OptionContract
from data.market_data import ScanResult, TickerSnapshot
from execution.order_manager import OrderManager, signal_to_request
from positions.exit_manager import ExitManager
from positions.position_tracker import OpenPosition, PositionTracker
from positions.reconciler import max_loss_dollars, settle_intrinsic_pnl
from signals.signal_generator import TradeLeg, TradeSignal


def _straddle(lots: int, entry_premium: float = 4.00) -> OpenPosition:
    legs = [
        TradeLeg(100.0, "call", "buy", lots, "AAPL260904C00100000"),
        TradeLeg(100.0, "put", "buy", lots, "AAPL260904P00100000"),
    ]
    return OpenPosition(
        tradier_order_id=7001, symbol="AAPL",
        expiration=date(2026, 9, 4), direction="BUY",
        structure="straddle", legs=legs, entry_premium=entry_premium,
        entry_atm_iv=0.25, entry_predicted_iv=0.35, entry_divergence=0.10,
        entry_horizon_lower=1, entry_horizon_upper=1, entry_weight_lower=1.0,
        submitted_at=datetime(2026, 8, 25, 15, 0, tzinfo=timezone.utc),
    )


def _condor(lots: int, entry_premium: float = 3.60) -> OpenPosition:
    legs = [
        TradeLeg(210.0, "call", "sell", lots, "NVDA260904C00210000"),
        TradeLeg(210.0, "put", "sell", lots, "NVDA260904P00210000"),
        TradeLeg(215.0, "call", "buy", lots, "NVDA260904C00215000"),
        TradeLeg(205.0, "put", "buy", lots, "NVDA260904P00205000"),
    ]
    return OpenPosition(
        tradier_order_id=7002, symbol="NVDA",
        expiration=date(2026, 9, 4), direction="SELL",
        structure="iron_condor", legs=legs, entry_premium=entry_premium,
        entry_atm_iv=0.40, entry_predicted_iv=0.32, entry_divergence=-0.08,
        entry_horizon_lower=1, entry_horizon_upper=1, entry_weight_lower=1.0,
        submitted_at=datetime(2026, 8, 25, 15, 0, tzinfo=timezone.utc),
    )


def test_open_position_lots_and_max_loss():
    """lots = min leg quantity; max_loss_dollars scales by it."""
    s3 = _straddle(3)
    assert s3.lots == 3
    assert abs(s3.max_loss_dollars - 4.00 * 100 * 3) < 1e-9

    c5 = _condor(5)
    assert c5.lots == 5
    # wing 5.0 wide, credit 3.60 → (5.0 − 3.60) × 100 × 5 = $700
    assert abs(c5.max_loss_dollars - 700.0) < 1e-9

    # 1-lot behavior unchanged
    assert _straddle(1).max_loss_dollars == 400.0
    assert abs(_condor(1).max_loss_dollars - 140.0) < 1e-9
    print("lots + max_loss_dollars ✓")


def _contract(leg: TradeLeg, bid: float, ask: float) -> OptionContract:
    return OptionContract(
        symbol=leg.contract_symbol, underlying="X",
        expiration=date(2026, 9, 4), strike=leg.strike,
        option_type=leg.option_type, bid=bid, ask=ask, last=(bid + ask) / 2,
        volume=100, open_interest=100,
        delta=0.5, gamma=0.01, theta=-0.05, vega=0.10, iv=0.30,
        fetched_at=datetime(2026, 8, 25, 15, 0, tzinfo=timezone.utc),
    )


def test_mark_to_market_scales_entry_cash_by_lots():
    """3-lot straddle bought at $4.00/lot, both legs now mid $2.50 → close
    cash = 2 legs × 2.50 × 3 × 100 = +$1,500; entry = −$1,200; pnl = +$300.
    The pre-fix code left entry at −$400 and reported +$1,100."""
    pos = _straddle(3)
    contracts = [_contract(leg, bid=2.40, ask=2.60) for leg in pos.legs]
    scan = ScanResult(
        fetched_at=datetime(2026, 8, 25, 16, 0, tzinfo=timezone.utc),
        snapshots={"AAPL": TickerSnapshot(
            symbol="AAPL", sector="tech",
            underlying={"last": 100.0}, contracts=contracts,
        )},
    )
    tracker = PositionTracker(
        client=mock.Mock(), order_log=mock.Mock(), settings=mock.Mock(),
    )
    marks = tracker.mark_to_market([pos], scan)
    assert len(marks) == 1
    m = marks[0]
    assert abs(m.close_cash_flow - 1500.0) < 1e-6
    assert abs(m.pnl_dollars - 300.0) < 1e-6
    # pct is lot-invariant: +300 / (4.00 × 100 × 3) = +25%
    assert abs(m.pnl_pct_of_entry_premium - 0.25) < 1e-9
    print("mark_to_market entry-cash scaling ✓")


def test_exit_thresholds_scale_with_lots():
    em = ExitManager(position_tracker=mock.Mock(), order_manager=mock.Mock())
    s = _straddle(3)   # pt +100%, sl −50% of 3 × $400
    assert abs(em._profit_target_threshold(s) - 1200.0) < 1e-9
    assert abs(em._stop_loss_threshold(s) - (-600.0)) < 1e-9
    c = _condor(5)     # pt +50%, sl −100% of 5 × $360
    assert abs(em._profit_target_threshold(c) - 900.0) < 1e-9
    assert abs(em._stop_loss_threshold(c) - (-1800.0)) < 1e-9
    print("exit thresholds ✓")


def _leg_dicts(pos: OpenPosition) -> list[dict]:
    return [
        {"strike": l.strike, "option_type": l.option_type,
         "side": l.side, "quantity": l.quantity}
        for l in pos.legs
    ]


def test_settle_intrinsic_and_max_loss_scale_with_lots():
    # 5-lot condor expires with underlying pinned at the body: all legs
    # worthless → keep the whole credit: +3.60 × 100 × 5 = +$1,800.
    c5 = _condor(5)
    pnl = settle_intrinsic_pnl(_leg_dicts(c5), underlying_close=210.0,
                               direction="SELL", entry_premium=3.60)
    assert abs(pnl - 1800.0) < 1e-6
    # Worst case: through the call wing → −(5.00 − 3.60) × 100 × 5 = −$700.
    floor = max_loss_dollars(_leg_dicts(c5), "SELL", 3.60)
    assert abs(floor - (-700.0)) < 1e-6
    deep = settle_intrinsic_pnl(_leg_dicts(c5), underlying_close=300.0,
                                direction="SELL", entry_premium=3.60)
    assert abs(deep - floor) < 1e-6, f"deep-ITM settle {deep} != floor {floor}"

    # 2-lot straddle, call finishes $6 ITM: payoff 6 × 100 × 2 = $1,200,
    # debit 4.00 × 100 × 2 = $800 → +$400.
    s2 = _straddle(2)
    pnl = settle_intrinsic_pnl(_leg_dicts(s2), underlying_close=106.0,
                               direction="BUY", entry_premium=4.00)
    assert abs(pnl - 400.0) < 1e-6
    assert abs(max_loss_dollars(_leg_dicts(s2), "BUY", 4.00) - (-800.0)) < 1e-6
    print("reconciler settle/max-loss ✓")


def test_compute_close_realized_pnl_scales_with_lots():
    # Short condor: entered at 3.60 credit/lot, closed at 3.00 debit/lot, 5 lots.
    pnl = OrderManager._compute_close_realized_pnl(
        is_long=False, entry_premium=3.60, order_type="debit",
        fill_price=3.00, fallback_pnl=0.0, lots=5,
    )
    assert abs(pnl - 300.0) < 1e-6
    # Long straddle: 4.00 debit/lot, sold at 6.50/lot, 2 lots → +$500.
    pnl = OrderManager._compute_close_realized_pnl(
        is_long=True, entry_premium=4.00, order_type="credit",
        fill_price=-6.50, fallback_pnl=0.0, lots=2,
    )
    assert abs(pnl - 500.0) < 1e-6
    # Default lots=1 preserves the old behavior.
    pnl = OrderManager._compute_close_realized_pnl(
        is_long=True, entry_premium=4.08, order_type="credit",
        fill_price=-2.00, fallback_pnl=0.0,
    )
    assert abs(pnl - (-208.0)) < 1e-6
    print("_compute_close_realized_pnl lots ✓")


def test_signal_to_request_carries_lots_and_scales_premium_cap():
    """Legs scaled to 5 lots must reach the API request at quantity 5, and
    the premium-cap estimate must be whole-trade dollars (× lots)."""
    legs = [
        TradeLeg(100.0, "call", "buy", 5, "AAPL260904C00100000"),
        TradeLeg(100.0, "put", "buy", 5, "AAPL260904P00100000"),
    ]
    signal = TradeSignal(
        symbol="AAPL", expiration=date(2026, 9, 4), dte=10,
        horizon_lower=1, horizon_upper=1, weight_lower=1.0,
        direction="BUY", underlying_price=100.0, atm_iv=0.25,
        predicted_iv_equivalent=0.35, divergence=0.10,
        cross_sectional_z=1.0, time_series_z=None, liquidity_score=1.0,
        legs=legs, is_actionable=True,
    )
    contracts = [_contract(leg, bid=1.90, ask=2.10) for leg in legs]
    snapshot = TickerSnapshot(
        symbol="AAPL", sector="tech",
        underlying={"last": 100.0}, contracts=contracts,
    )
    request = signal_to_request(signal, snapshot, slippage=0.0, max_qty=10)
    assert all(leg["quantity"] == 5 for leg in request.legs), request.legs
    # straddle mid = 2.00 + 2.00 = 4.00/lot → estimated premium $2,000 total
    assert abs(request.estimated_premium - 2000.0) < 1e-6
    # limit price stays PER UNIT
    assert abs(request.price - 4.00) < 1e-6
    print("signal_to_request lots + premium cap ✓")


def main() -> int:
    test_open_position_lots_and_max_loss()
    test_mark_to_market_scales_entry_cash_by_lots()
    test_exit_thresholds_scale_with_lots()
    test_settle_intrinsic_and_max_loss_scale_with_lots()
    test_compute_close_realized_pnl_scales_with_lots()
    test_signal_to_request_carries_lots_and_scales_premium_cap()
    print("all multi-lot sizing tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
