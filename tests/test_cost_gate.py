"""Tests for the transaction-cost gate (signals/cost_gate.py)."""
import math
import sys
from datetime import date, datetime, timezone

from data.async_client import OptionContract
from signals.cost_gate import SHARES_PER_CONTRACT, evaluate_cost_gate
from signals.signal_generator import TradeLeg


def _c(symbol, strike, otype, bid, ask, vega):
    return OptionContract(
        symbol=symbol, underlying="X", expiration=date(2026, 6, 12),
        strike=strike, option_type=otype, bid=bid, ask=ask,
        last=(bid + ask) / 2, volume=200, open_interest=1000,
        delta=0.5 if otype == "call" else -0.5, gamma=0.01, theta=-0.05,
        vega=vega, iv=0.25, fetched_at=datetime.now(timezone.utc),
    )


def _condor(atm_vega=0.20, wing_vega=0.10, put_ask=0.96):
    chain = [
        _c("AC", 100, "call", 1.00, 1.04, atm_vega),
        _c("AP", 100, "put", 0.92, put_ask, atm_vega),
        _c("WC", 110, "call", 0.20, 0.21, wing_vega),
        _c("WP", 90, "put", 0.20, 0.21, wing_vega),
    ]
    legs = [
        TradeLeg(100, "call", "sell", 1, "AC"),
        TradeLeg(100, "put", "sell", 1, "AP"),
        TradeLeg(110, "call", "buy", 1, "WC"),
        TradeLeg(90, "put", "buy", 1, "WP"),
    ]
    return legs, chain


def test_condor_edge_and_cost_arithmetic():
    legs, chain = _condor()
    fee = 0.10
    check = evaluate_cost_gate(
        legs, chain, forecast_vol=0.19, atm_iv=0.30,
        per_contract_fee=fee, cost_multiple=2.0, max_leg_spread_pct=0.05,
    )
    # net vega = -0.2 - 0.2 + 0.1 + 0.1 = -0.2; edge = 11 pts × 0.2 × 100 shares = $220
    assert math.isclose(check.expected_edge_usd, 220.0, rel_tol=1e-9), check
    # half-spreads: 0.02 + 0.02 + 0.005 + 0.005 = 0.05 → $5 + 4 fees = $5.40
    assert math.isclose(check.total_cost_usd, 5.0 + 4 * fee, rel_tol=1e-9)
    assert check.passed, check.reason
    print(f"condor: edge ${check.expected_edge_usd:.0f} vs cost ${check.total_cost_usd:.2f} → pass")


def test_straddle_long_vega_sign():
    chain = [
        _c("AC", 100, "call", 1.00, 1.04, 0.20),
        _c("AP", 100, "put", 0.92, 0.96, 0.20),
    ]
    legs = [
        TradeLeg(100, "call", "buy", 1, "AC"),
        TradeLeg(100, "put", "buy", 1, "AP"),
    ]
    check = evaluate_cost_gate(
        legs, chain, forecast_vol=0.30, atm_iv=0.25,
        per_contract_fee=0.0, cost_multiple=2.0, max_leg_spread_pct=0.05,
    )
    # net vega = +0.4 → edge = 5 pts * 0.4 * 100 = $200; cost = $4
    assert math.isclose(check.expected_edge_usd, 200.0, rel_tol=1e-9)
    assert math.isclose(check.total_cost_usd, 4.0, rel_tol=1e-9)
    assert check.passed
    print("straddle: long-vega edge positive, passes")


def test_edge_below_multiple_blocks():
    legs, chain = _condor(atm_vega=0.02, wing_vega=0.01)  # tiny vega, tiny edge
    check = evaluate_cost_gate(
        legs, chain, forecast_vol=0.24, atm_iv=0.25,
        per_contract_fee=0.10, cost_multiple=2.0, max_leg_spread_pct=0.05,
    )
    # edge = 1 pt * 0.02 * 100 = $2 < 2 × $5.40
    assert not check.passed
    assert "edge" in check.reason
    print(f"cost_block: {check.reason}")


def test_leg_spread_cap_blocks():
    legs, chain = _condor(put_ask=1.00)  # put quotes 0.92/1.00 → 8.3% of mid
    check = evaluate_cost_gate(
        legs, chain, forecast_vol=0.19, atm_iv=0.30,
        per_contract_fee=0.0, cost_multiple=2.0, max_leg_spread_pct=0.05,
    )
    assert not check.passed
    assert check.worst_leg_spread_pct > 0.05
    assert "spread" in check.reason
    print(f"spread_block: {check.reason}")


def test_missing_quote_fails_closed():
    legs, chain = _condor()
    check = evaluate_cost_gate(
        legs, chain[:-1], forecast_vol=0.19, atm_iv=0.30,
        per_contract_fee=0.0,
    )
    assert not check.passed and "no quote" in check.reason
    print("missing_quote: fails closed")


def test_fees_count_per_leg():
    legs, chain = _condor()
    free = evaluate_cost_gate(legs, chain, 0.19, 0.30, per_contract_fee=0.0)
    paid = evaluate_cost_gate(legs, chain, 0.19, 0.30, per_contract_fee=0.45)
    assert math.isclose(paid.total_cost_usd - free.total_cost_usd, 4 * 0.45, rel_tol=1e-9)
    print("fees: 4 legs × fee added to cost")


def main() -> int:
    test_condor_edge_and_cost_arithmetic()
    test_straddle_long_vega_sign()
    test_edge_below_multiple_blocks()
    test_leg_spread_cap_blocks()
    test_missing_quote_fails_closed()
    test_fees_count_per_leg()
    print("all cost_gate tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
