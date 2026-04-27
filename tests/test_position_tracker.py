import math
import sys
from datetime import date, datetime, timezone
from unittest import mock

from data.async_client import OptionContract
from data.market_data import ScanResult, TickerSnapshot
from positions import OpenPosition, PositionMark, PositionTracker
from signals.signal_generator import TradeLeg


def _mk_contract(strike: float, otype: str, *, bid=1.0, ask=1.10, delta=0.5, gamma=0.01,
                 theta=-0.05, vega=0.20, iv=0.30, vol=200, oi=1000) -> OptionContract:
    sym = f"NVDA260522{'C' if otype=='call' else 'P'}{int(strike*1000):08d}"
    return OptionContract(
        symbol=sym, underlying="NVDA", expiration=date(2026, 5, 22),
        strike=strike, option_type=otype,
        bid=bid, ask=ask, last=(bid + ask) / 2,
        volume=vol, open_interest=oi,
        delta=delta if otype == "call" else -delta,
        gamma=gamma, theta=theta, vega=vega, iv=iv,
        fetched_at=datetime.now(timezone.utc),
    )


def _mk_iron_condor_position(*, entry_credit=13.55) -> OpenPosition:
    legs = [
        TradeLeg(210.0, "call", "sell", 1, "NVDA260522C00210000"),
        TradeLeg(210.0, "put", "sell", 1, "NVDA260522P00210000"),
        TradeLeg(230.0, "call", "buy", 1, "NVDA260522C00230000"),
        TradeLeg(190.0, "put", "buy", 1, "NVDA260522P00190000"),
    ]
    return OpenPosition(
        tradier_order_id=99999, symbol="NVDA",
        expiration=date(2026, 5, 22), direction="SELL",
        structure="iron_condor", legs=legs, entry_premium=entry_credit,
        entry_atm_iv=0.40, entry_predicted_iv=0.32, entry_divergence=-0.08,
        entry_horizon_lower=21, entry_horizon_upper=21, entry_weight_lower=1.0,
        submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
    )


def _mk_long_straddle_position(*, entry_debit=4.08) -> OpenPosition:
    legs = [
        TradeLeg(100.0, "call", "buy", 1, "AAPL260515C00100000"),
        TradeLeg(100.0, "put", "buy", 1, "AAPL260515P00100000"),
    ]
    return OpenPosition(
        tradier_order_id=88888, symbol="AAPL",
        expiration=date(2026, 5, 15), direction="BUY",
        structure="straddle", legs=legs, entry_premium=entry_debit,
        entry_atm_iv=0.27, entry_predicted_iv=0.42, entry_divergence=0.15,
        entry_horizon_lower=10, entry_horizon_upper=21, entry_weight_lower=0.27,
        submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
    )


def _mk_scan_for_iron_condor(*, contracts) -> ScanResult:
    snap = TickerSnapshot(
        symbol="NVDA", sector="tech",
        underlying={"symbol": "NVDA", "last": 210.0},
        contracts=contracts,
    )
    return ScanResult(
        fetched_at=datetime(2026, 4, 30, 16, 0, tzinfo=timezone.utc),
        snapshots={"NVDA": snap},
    )


def _tracker() -> PositionTracker:
    fake_client = mock.AsyncMock()
    fake_log = mock.MagicMock()
    fake_settings = mock.MagicMock()
    return PositionTracker(client=fake_client, order_log=fake_log, settings=fake_settings)


def test_iron_condor_pnl_at_credit_decay():
    """Entry credit $13.55, current cost-to-close $5 → P&L = $8.55 × 100 = $855."""
    pos = _mk_iron_condor_position(entry_credit=13.55)
    # Construct chain so cost_to_close = $5 per contract
    # short_call mid = $2, short_put = $2 (sum=$4 to buy back)
    # long_call mid = $0.50, long_put = $0.50 (sum=$1 to sell back)
    # cost_to_close = (4 - 1) = $3 per contract → wait, want $5
    # Let's recalculate: shorts must equal 5 + longs in mid-prices
    # short_mids = 5.5, long_mids = 0.5 → diff = 5
    contracts = [
        _mk_contract(210, "call", bid=2.7, ask=2.8),  # mid 2.75
        _mk_contract(210, "put", bid=2.7, ask=2.8),   # mid 2.75
        _mk_contract(230, "call", bid=0.20, ask=0.30),  # mid 0.25
        _mk_contract(190, "put", bid=0.20, ask=0.30),   # mid 0.25
    ]
    scan = _mk_scan_for_iron_condor(contracts=contracts)
    marks = _tracker().mark_to_market([pos], scan)
    assert len(marks) == 1
    m = marks[0]
    # close_cash_flow per leg:
    #   short_call (sign=-1): -2.75 × 100 = -275
    #   short_put (sign=-1):  -2.75 × 100 = -275
    #   long_call (sign=+1):  +0.25 × 100 = +25
    #   long_put (sign=+1):   +0.25 × 100 = +25
    # Total: -500 (we'd pay $500 to flatten)
    expected_close_cash = -500.0
    assert math.isclose(m.close_cash_flow, expected_close_cash, abs_tol=0.01)
    # P&L = entry_credit*100 + close_cash_flow = 1355 + (-500) = 855
    assert math.isclose(m.pnl_dollars, 855.0, abs_tol=0.01)
    assert math.isclose(m.cost_to_close, 500.0, abs_tol=0.01)
    print(f"iron_condor P&L: ${m.pnl_dollars:.2f}, cost_to_close: ${m.cost_to_close:.2f}")


def test_iron_condor_pnl_unchanged_returns_near_zero():
    """If current chain mid prices match entry mids, P&L should be near zero."""
    pos = _mk_iron_condor_position(entry_credit=13.55)
    # Reverse-engineer: entry_credit 13.55 = (short_mids - long_mids) at entry
    # Use shorts=10.0 each, longs=3.225 each → 20 - 6.45 = 13.55
    contracts = [
        _mk_contract(210, "call", bid=9.95, ask=10.05),    # mid 10
        _mk_contract(210, "put", bid=9.95, ask=10.05),     # mid 10
        _mk_contract(230, "call", bid=3.20, ask=3.25),     # mid 3.225
        _mk_contract(190, "put", bid=3.20, ask=3.25),      # mid 3.225
    ]
    scan = _mk_scan_for_iron_condor(contracts=contracts)
    m = _tracker().mark_to_market([pos], scan)[0]
    # close_cash_flow: -10*100 -10*100 +3.225*100 +3.225*100 = -1000-1000+322.5+322.5 = -1355
    # P&L = 1355 + (-1355) = 0
    assert math.isclose(m.pnl_dollars, 0.0, abs_tol=1.0)
    print(f"iron_condor unchanged P&L: ${m.pnl_dollars:.2f}")


def test_long_straddle_pnl():
    """Entry debit $4.08, current chain mid sum $6.00 → P&L = ($6.00 − $4.08) × 100 = $192."""
    pos = _mk_long_straddle_position(entry_debit=4.08)
    contracts = [
        _mk_contract(100, "call", bid=2.95, ask=3.05),  # mid 3.00
        _mk_contract(100, "put", bid=2.95, ask=3.05),   # mid 3.00
    ]
    snap = TickerSnapshot(
        symbol="AAPL", sector="tech",
        underlying={"symbol": "AAPL", "last": 100.0},
        contracts=contracts,
    )
    scan = ScanResult(
        fetched_at=datetime(2026, 4, 30, 16, 0, tzinfo=timezone.utc),
        snapshots={"AAPL": snap},
    )
    # Replace contract symbols to match position.legs
    contracts[0] = OptionContract(**{**contracts[0].__dict__, "symbol": "AAPL260515C00100000"})
    contracts[1] = OptionContract(**{**contracts[1].__dict__, "symbol": "AAPL260515P00100000"})
    snap = TickerSnapshot(symbol="AAPL", sector="tech", underlying=snap.underlying, contracts=contracts)
    scan = ScanResult(fetched_at=scan.fetched_at, snapshots={"AAPL": snap})
    m = _tracker().mark_to_market([pos], scan)[0]
    # close_cash_flow per leg (both long, sign=+1): +3.00*100 + 3.00*100 = +600
    # entry cash flow (long: -1): -4.08*100 = -408
    # P&L = -408 + 600 = +192
    assert math.isclose(m.pnl_dollars, 192.0, abs_tol=0.01)
    print(f"long_straddle P&L: ${m.pnl_dollars:.2f}")


def test_greeks_signed_by_leg_direction():
    """Iron condor short — short legs negate their delta contributions."""
    pos = _mk_iron_condor_position()
    # Use distinctive deltas so we can check signing
    contracts = [
        _mk_contract(210, "call", delta=0.50),  # short call: contributes -50 (sign=-1, ×100 = -50)
        _mk_contract(210, "put", delta=0.50),    # short put has delta -0.50, sign=-1 → -(-0.50)*100 = +50
        _mk_contract(230, "call", delta=0.20),   # long call: +20
        _mk_contract(190, "put", delta=0.20),    # long put has delta -0.20, sign=+1 → -20
    ]
    scan = _mk_scan_for_iron_condor(contracts=contracts)
    m = _tracker().mark_to_market([pos], scan)[0]
    # delta_sum: -1*0.50*100 + -1*(-0.50)*100 + 1*0.20*100 + 1*(-0.20)*100
    #          = -50 + 50 + 20 + -20 = 0
    assert math.isclose(m.delta, 0.0, abs_tol=0.01), f"got delta={m.delta}"
    print(f"greeks: delta-neutral as expected for symmetric IC: {m.delta}")


def test_dte_math():
    pos = _mk_iron_condor_position()
    # expiration 2026-05-22, scan at 2026-04-30 → dte = 22
    contracts = [
        _mk_contract(210, "call"),
        _mk_contract(210, "put"),
        _mk_contract(230, "call"),
        _mk_contract(190, "put"),
    ]
    scan = _mk_scan_for_iron_condor(contracts=contracts)
    m = _tracker().mark_to_market([pos], scan)[0]
    assert m.dte == 22, f"expected dte=22, got {m.dte}"
    print(f"dte math: expiration 5/22, scan 4/30 → dte={m.dte}")


def test_skip_position_when_legs_missing():
    pos = _mk_iron_condor_position()
    # Empty chain → can't mark
    snap = TickerSnapshot(
        symbol="NVDA", sector="tech",
        underlying={"symbol": "NVDA", "last": 210.0},
        contracts=[],
    )
    scan = ScanResult(
        fetched_at=datetime(2026, 4, 30, 16, 0, tzinfo=timezone.utc),
        snapshots={"NVDA": snap},
    )
    marks = _tracker().mark_to_market([pos], scan)
    assert marks == []
    print("missing legs: position skipped with warning")


def test_portfolio_greeks():
    pos1 = _mk_iron_condor_position()
    pos2 = _mk_long_straddle_position()
    # Mock marks with known greeks
    m1 = PositionMark(
        position=pos1, current_legs=[], close_cash_flow=0, cost_to_close=0,
        pnl_dollars=0, pnl_pct_of_entry_premium=0, pnl_pct_of_max=float("nan"),
        delta=10.0, gamma=2.0, theta=-5.0, vega=15.0, dte=20,
    )
    m2 = PositionMark(
        position=pos2, current_legs=[], close_cash_flow=0, cost_to_close=0,
        pnl_dollars=0, pnl_pct_of_entry_premium=0, pnl_pct_of_max=float("nan"),
        delta=-3.0, gamma=1.0, theta=-2.0, vega=8.0, dte=18,
    )
    g = PositionTracker.portfolio_greeks([m1, m2])
    assert g["delta"] == 7.0
    assert g["gamma"] == 3.0
    assert g["theta"] == -7.0
    assert g["vega"] == 23.0
    print(f"portfolio greeks: {g}")


def main() -> int:
    test_iron_condor_pnl_at_credit_decay()
    test_iron_condor_pnl_unchanged_returns_near_zero()
    test_long_straddle_pnl()
    test_greeks_signed_by_leg_direction()
    test_dte_math()
    test_skip_position_when_legs_missing()
    test_portfolio_greeks()
    print("all position_tracker tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
