"""Unit tests for the signal-generator helpers (ATM pairing, composite
liquidity) plus h=1 gate coverage that doesn't fit the end-to-end flow file:
the long-straddle (index-ETF) exclusion and the constructor contract. The
end-to-end h=1 flow lives in test_signal_generator_h1."""
import math
import sys
from datetime import date, datetime, timezone

from data import OptionContract
from signals import SignalGenerator, composite_liquidity, find_atm_iv
from tests.test_signal_generator_h1 import _run


def _mk_contract(strike: float, otype: str, *, bid=1.0, ask=1.1, vol=100, oi=500, iv=0.3) -> OptionContract:
    return OptionContract(
        symbol=f"X{strike:.0f}{otype[0].upper()}",
        underlying="X",
        expiration=date(2026, 5, 22),
        strike=strike,
        option_type=otype,
        bid=bid,
        ask=ask,
        last=(bid + ask) / 2,
        volume=vol,
        open_interest=oi,
        delta=0.5 if otype == "call" else -0.5,
        gamma=0.01,
        theta=-0.05,
        vega=0.20,
        iv=iv,
        fetched_at=datetime.now(timezone.utc),
    )


def test_find_atm_iv():
    chain = [_mk_contract(k, t) for k in (90, 95, 100, 105, 110) for t in ("call", "put")]
    pair = find_atm_iv(chain, underlying_price=102.0)
    assert pair is not None
    call, put = pair
    assert call.strike == 100, f"expected 100, got {call.strike}"
    assert put.strike == 100
    print("find_atm_iv: 102 → 100-strike pair (closest)")


def test_find_atm_iv_returns_none_when_one_side_missing():
    chain = [_mk_contract(k, "call") for k in (90, 100, 110)]
    assert find_atm_iv(chain, underlying_price=100) is None
    print("find_atm_iv: None when one side missing")


def test_composite_liquidity():
    call = _mk_contract(100, "call", bid=1.0, ask=1.1, vol=100, oi=500)
    put = _mk_contract(100, "put", bid=0.9, ask=1.0, vol=200, oi=800)
    # min vol=100, min oi=500, max spread = (1.1-1.0)/1.05 ≈ 0.0952
    expected = 100 * 500 / (1 + max((1.1 - 1.0) / 1.05, (1.0 - 0.9) / 0.95))
    actual = composite_liquidity(call, put)
    assert math.isclose(actual, expected, rel_tol=1e-9)
    print(f"composite_liquidity: {actual:.1f} matches manual calc")


def test_long_straddle_exclusion_demotes_etf_buy_only():
    """Gate 8: an excluded symbol's BUY straddle is demoted; an identical
    non-excluded symbol stays actionable. The exclusion is BUY-side only —
    a SELL on the excluded symbol still passes."""
    # iv=0.13 vs seeded gap history → z ≈ -4.8 → BUY for both symbols;
    # iv=0.30 → z ≈ +3.5 → SELL for the excluded symbol's condor side.
    actionable, all_signals, _ = _run(
        [("SPY", 0.13), ("CTRL", 0.13)],
        long_straddle_excluded_symbols={"SPY"},
    )
    by = {s.symbol: s for s in all_signals}
    assert by["SPY"].direction == "BUY"
    assert not by["SPY"].is_actionable
    assert by["SPY"].blocked_by == "long_straddle_excluded"
    assert "SPY" not in [s.symbol for s in actionable]
    assert by["CTRL"].direction == "BUY" and by["CTRL"].is_actionable, \
        by["CTRL"].diagnostic_notes
    assert len(by["CTRL"].legs) == 2

    sell_actionable, sell_signals, _ = _run(
        [("SPY", 0.30)],
        long_straddle_excluded_symbols={"SPY"},
    )
    assert sell_signals[0].direction == "SELL" and sell_signals[0].is_actionable, \
        "exclusion must not touch the SELL/iron-condor side"
    print("long_straddle_exclusion: SPY BUY demoted, CTRL BUY + SPY SELL actionable")


def test_constructor_requires_h1_predictor():
    try:
        SignalGenerator()
    except ValueError as e:
        assert "h1_predictor" in str(e)
        print("constructor: h1_predictor required")
    else:
        raise AssertionError("missing h1_predictor should raise")


def main() -> int:
    test_find_atm_iv()
    test_find_atm_iv_returns_none_when_one_side_missing()
    test_composite_liquidity()
    test_long_straddle_exclusion_demotes_etf_buy_only()
    test_constructor_requires_h1_predictor()
    print("all signal_generator tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
