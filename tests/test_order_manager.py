import asyncio
import math
import os
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

from config import Settings
from data.async_client import OptionContract
from data.market_data import TickerSnapshot
from execution import (
    OrderLog,
    OrderManager,
    compute_iron_condor_credit,
    compute_straddle_debit,
    fingerprint_signal,
    signal_to_request,
)
from signals.signal_generator import TradeLeg, TradeSignal


def _mk_contract(strike: float, otype: str, *, bid=1.0, ask=1.10, vol=200, oi=1000) -> OptionContract:
    return OptionContract(
        symbol=f"AAPL_{int(strike)}{otype[0].upper()}",
        underlying="AAPL",
        expiration=date(2026, 5, 15),
        strike=strike, option_type=otype,
        bid=bid, ask=ask, last=(bid + ask) / 2,
        volume=vol, open_interest=oi,
        delta=0.5 if otype == "call" else -0.5,
        gamma=0.01, theta=-0.05, vega=0.20,
        iv=0.30, fetched_at=datetime.now(timezone.utc),
    )


def _mk_signal(direction: str = "BUY") -> TradeSignal:
    if direction == "BUY":
        legs = [
            TradeLeg(100.0, "call", "buy", 1, "AAPL_100C"),
            TradeLeg(100.0, "put", "buy", 1, "AAPL_100P"),
        ]
    else:
        legs = [
            TradeLeg(100.0, "call", "sell", 1, "AAPL_100C"),
            TradeLeg(100.0, "put", "sell", 1, "AAPL_100P"),
            TradeLeg(105.0, "call", "buy", 1, "AAPL_105C"),
            TradeLeg(95.0, "put", "buy", 1, "AAPL_95P"),
        ]
    return TradeSignal(
        symbol="AAPL", expiration=date(2026, 5, 15), dte=18,
        horizon_lower=10, horizon_upper=21, weight_lower=0.27,
        direction=direction, underlying_price=100.0, atm_iv=0.30,
        predicted_iv_equivalent=0.40, divergence=0.10,
        cross_sectional_z=2.0, time_series_z=None,
        liquidity_score=12345.0, legs=legs, is_actionable=True,
    )


def _mk_snapshot(direction: str) -> TickerSnapshot:
    contracts = [
        _mk_contract(100.0, "call", bid=2.00, ask=2.10),
        _mk_contract(100.0, "put", bid=1.90, ask=2.00),
    ]
    if direction == "SELL":
        contracts.extend([
            _mk_contract(105.0, "call", bid=0.50, ask=0.55),
            _mk_contract(95.0, "put", bid=0.45, ask=0.50),
        ])
    return TickerSnapshot(
        symbol="AAPL", sector="tech",
        underlying={"symbol": "AAPL", "last": 100.0},
        contracts=contracts,
    )


# ----- pricing helpers -----

def test_compute_straddle_debit():
    call = _mk_contract(100.0, "call", bid=2.00, ask=2.10)
    put = _mk_contract(100.0, "put", bid=1.90, ask=2.00)
    # mid_call = 2.05, mid_put = 1.95, sum = 4.00, *1.02 = 4.08
    expected = round(4.00 * 1.02, 2)
    assert math.isclose(compute_straddle_debit(call, put), expected)
    print(f"straddle_debit: {expected}")


def test_compute_iron_condor_credit():
    short_call = _mk_contract(100.0, "call", bid=2.00, ask=2.10)
    short_put = _mk_contract(100.0, "put", bid=1.90, ask=2.00)
    long_call = _mk_contract(105.0, "call", bid=0.50, ask=0.55)
    long_put = _mk_contract(95.0, "put", bid=0.45, ask=0.50)
    # short_mid = 2.05+1.95 = 4.00; long_mid = 0.525+0.475 = 1.00
    # net = 3.00, *0.98 = 2.94
    expected = round(3.00 * 0.98, 2)
    actual = compute_iron_condor_credit(short_call, short_put, long_call, long_put)
    assert math.isclose(actual, expected)
    assert actual > 0, "iron condor credit must be positive"
    print(f"iron_condor_credit: {actual}")


# ----- fingerprint -----

def test_fingerprint_stable_and_distinct():
    sig1 = _mk_signal("BUY")
    sig2 = _mk_signal("BUY")
    sig3 = _mk_signal("SELL")
    scan_date = date(2026, 4, 27)
    fp1 = fingerprint_signal(sig1, scan_date)
    fp2 = fingerprint_signal(sig2, scan_date)
    fp3 = fingerprint_signal(sig3, scan_date)
    assert fp1 == fp2, "identical signals should hash equal"
    assert fp1 != fp3, "different directions should hash differently"
    # Different scan date → different fingerprint
    fp_other_day = fingerprint_signal(sig1, date(2026, 4, 28))
    assert fp1 != fp_other_day
    print(f"fingerprint: stable + direction-sensitive + date-sensitive (e.g. {fp1})")


# ----- signal_to_request -----

def test_signal_to_request_buy_straddle():
    sig = _mk_signal("BUY")
    snap = _mk_snapshot("BUY")
    req = signal_to_request(sig, snap, slippage=0.02, max_qty=10)
    assert req.structure == "straddle"
    assert req.order_type == "debit"
    assert len(req.legs) == 2
    assert all(l["side"] == "buy_to_open" for l in req.legs)
    assert req.price > 0
    print(f"buy_straddle request: type={req.order_type} price={req.price} legs={len(req.legs)}")


def test_signal_to_request_sell_iron_condor():
    sig = _mk_signal("SELL")
    snap = _mk_snapshot("SELL")
    req = signal_to_request(sig, snap, slippage=0.02, max_qty=10)
    assert req.structure == "iron_condor"
    assert req.order_type == "credit"
    assert len(req.legs) == 4
    sells = [l for l in req.legs if l["side"] == "sell_to_open"]
    buys = [l for l in req.legs if l["side"] == "buy_to_open"]
    assert len(sells) == 2 and len(buys) == 2
    assert req.price > 0  # absolute credit value, sign carried by order_type
    print(f"sell_iron_condor request: type={req.order_type} price={req.price} legs={len(req.legs)}")


def test_quantity_clamped():
    sig = _mk_signal("BUY")
    # Force quantity > cap
    big_legs = [
        TradeLeg(100.0, "call", "buy", 50, "AAPL_100C"),
        TradeLeg(100.0, "put", "buy", 50, "AAPL_100P"),
    ]
    # Re-build signal with the big quantities
    sig = TradeSignal(
        symbol=sig.symbol, expiration=sig.expiration, dte=sig.dte,
        horizon_lower=sig.horizon_lower, horizon_upper=sig.horizon_upper,
        weight_lower=sig.weight_lower, direction=sig.direction,
        underlying_price=sig.underlying_price, atm_iv=sig.atm_iv,
        predicted_iv_equivalent=sig.predicted_iv_equivalent,
        divergence=sig.divergence, cross_sectional_z=sig.cross_sectional_z,
        time_series_z=sig.time_series_z, liquidity_score=sig.liquidity_score,
        legs=big_legs, is_actionable=True,
    )
    snap = _mk_snapshot("BUY")
    req = signal_to_request(sig, snap, slippage=0.02, max_qty=10)
    assert all(l["quantity"] == 10 for l in req.legs), "quantity should be clamped"
    print(f"quantity_clamped: 50 → 10")


# ----- production guard -----

def _mk_settings(env: str) -> Settings:
    return Settings(
        api_key="fake", account_id="VA00000000",
        base_url="https://example.invalid/v1", env=env,
    )


def _mk_manager(env: str, log: OrderLog) -> OrderManager:
    settings = _mk_settings(env)
    fake_client = mock.AsyncMock()
    return OrderManager(client=fake_client, order_log=log, settings=settings)


def test_production_guard_blocks_without_confirmation():
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        sig = _mk_signal("BUY")
        snap = _mk_snapshot("BUY")
        # Force production env
        manager = _mk_manager("production", log)
        # Ensure the env var is unset
        prev = os.environ.pop("TRADIER_LIVE_TRADING_CONFIRMED", None)
        try:
            result = asyncio.run(manager.submit(sig, snap, scan_date=date(2026, 4, 27)))
        finally:
            if prev is not None:
                os.environ["TRADIER_LIVE_TRADING_CONFIRMED"] = prev
        assert result.status == "guard_blocked"
        assert "production" in result.error.lower()
        log.close()
    print("production_guard: blocks without confirmation env var")


def test_premium_cap_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        # Build a signal whose computed price > $50/contract → premium > 5000
        legs = [
            TradeLeg(100.0, "call", "buy", 1, "AAPL_100C"),
            TradeLeg(100.0, "put", "buy", 1, "AAPL_100P"),
        ]
        sig = TradeSignal(
            symbol="AAPL", expiration=date(2026, 5, 15), dte=18,
            horizon_lower=10, horizon_upper=21, weight_lower=0.27,
            direction="BUY", underlying_price=100.0, atm_iv=0.30,
            predicted_iv_equivalent=0.40, divergence=0.10,
            cross_sectional_z=2.0, time_series_z=None,
            liquidity_score=12345.0, legs=legs, is_actionable=True,
        )
        # Snapshot with very expensive options
        contracts = [
            _mk_contract(100.0, "call", bid=30.0, ask=32.0),
            _mk_contract(100.0, "put", bid=25.0, ask=27.0),
        ]
        snap = TickerSnapshot(
            symbol="AAPL", sector="tech",
            underlying={"symbol": "AAPL", "last": 100.0},
            contracts=contracts,
        )
        manager = _mk_manager("sandbox", log)
        result = asyncio.run(manager.submit(sig, snap, scan_date=date(2026, 4, 27)))
        assert result.status == "preview_failed"
        assert "premium" in result.error.lower()
        # Failure recorded in failed_submissions
        assert log.failed_count() == 1
        log.close()
    print(f"premium_cap: blocked with status={result.status}")


def main() -> int:
    test_compute_straddle_debit()
    test_compute_iron_condor_credit()
    test_fingerprint_stable_and_distinct()
    test_signal_to_request_buy_straddle()
    test_signal_to_request_sell_iron_condor()
    test_quantity_clamped()
    test_production_guard_blocks_without_confirmation()
    test_premium_cap_blocks()
    print("all order_manager tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
