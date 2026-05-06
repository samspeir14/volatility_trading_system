"""Unit tests for the balances-parsing helpers in risk/portfolio_state.py.

Tradier returns one of {margin, cash, pdt} sub-objects depending on account type;
PortfolioStateBuilder must extract option_buying_power from the right one. A bug
where it only ever read balances["margin"] caused the bot to read $0 buying power
on a pdt-typed sandbox account and reject every trade with margin_buffer.
Fixtures here are real Tradier-shaped responses captured from sandbox + docs.
"""
import sys

from risk.portfolio_state import (
    _account_sub_object,
    _extract_margin_requirement,
    _extract_option_buying_power,
)


# Captured live from EC2 sandbox on 2026-05-06 (5 open iron condors).
PDT_BALANCES = {
    "option_short_value": -5632.0,
    "total_equity": 99310.2,
    "account_number": "VA26871732",
    "account_type": "pdt",
    "close_pl": 0,
    "current_requirement": 6000.0,
    "equity": 0,
    "long_market_value": 6429.0,
    "market_value": 797.0,
    "open_pl": -713.0,
    "option_long_value": 6429.0,
    "option_requirement": 6000.0,
    "pending_orders_count": 0,
    "short_market_value": -5632.0,
    "stock_long_value": 0,
    "total_cash": 98513.2,
    "uncleared_funds": 0,
    "pending_cash": 0,
    "pdt": {
        "day_trade_buying_power": 360052.8,
        "fed_call": 0,
        "maintenance_call": 0,
        "option_buying_power": 90013.2,
        "stock_buying_power": 180026.4,
        "stock_short_value": 0,
    },
}

# From Tradier docs — standard margin account shape.
MARGIN_BALANCES = {
    "option_short_value": 0,
    "total_equity": 17798.36,
    "account_number": "VA0000000",
    "account_type": "margin",
    "close_pl": 0,
    "current_requirement": 0.0,
    "equity": 0,
    "long_market_value": 0,
    "market_value": 0,
    "open_pl": 0,
    "option_long_value": 0,
    "option_requirement": 0,
    "pending_orders_count": 0,
    "short_market_value": 0,
    "stock_long_value": 0,
    "total_cash": 17798.36,
    "uncleared_funds": 0,
    "pending_cash": 0,
    "margin": {
        "fed_call": 0,
        "maintenance_call": 0,
        "option_buying_power": 35596.72,
        "stock_buying_power": 71193.44,
        "stock_short_value": 0,
        "sweep": 0,
    },
}

# From Tradier docs — cash account shape (no option_buying_power; cash_available is the limit).
CASH_BALANCES = {
    "option_short_value": 0,
    "total_equity": 25000.0,
    "account_number": "VA0000001",
    "account_type": "cash",
    "close_pl": 0,
    "current_requirement": 0.0,
    "equity": 0,
    "long_market_value": 0,
    "market_value": 0,
    "open_pl": 0,
    "option_long_value": 0,
    "option_requirement": 0,
    "pending_orders_count": 0,
    "short_market_value": 0,
    "stock_long_value": 0,
    "total_cash": 25000.0,
    "uncleared_funds": 0,
    "pending_cash": 0,
    "cash": {
        "cash_available": 24500.0,
        "sweep": 0,
        "unsettled_funds": 500.0,
    },
}


def test_pdt_account_extracts_option_buying_power():
    bp = _extract_option_buying_power(PDT_BALANCES)
    assert bp == 90013.2, f"expected 90013.2, got {bp}"
    print("pdt: option_buying_power extracted from pdt sub-object")


def test_margin_account_extracts_option_buying_power():
    bp = _extract_option_buying_power(MARGIN_BALANCES)
    assert bp == 35596.72, f"expected 35596.72, got {bp}"
    print("margin: option_buying_power extracted from margin sub-object")


def test_cash_account_falls_back_to_cash_available():
    bp = _extract_option_buying_power(CASH_BALANCES)
    assert bp == 24500.0, f"expected 24500.0 (cash_available), got {bp}"
    print("cash: falls back to cash_available when option_buying_power missing")


def test_pdt_account_extracts_margin_requirement():
    held = _extract_margin_requirement(PDT_BALANCES)
    assert held == 6000.0, f"expected 6000.0, got {held}"
    print("pdt: current_requirement read from top level")


def test_margin_account_extracts_margin_requirement():
    held = _extract_margin_requirement(MARGIN_BALANCES)
    assert held == 0.0, f"expected 0.0, got {held}"
    print("margin: current_requirement read from top level")


def test_account_sub_object_uses_account_type_field():
    sub = _account_sub_object(PDT_BALANCES)
    assert "option_buying_power" in sub
    assert sub["option_buying_power"] == 90013.2
    print("sub_object: declared account_type=pdt routes to pdt block")


def test_account_sub_object_falls_back_when_account_type_missing():
    # Construct a response without the account_type field but with a margin block
    response = {k: v for k, v in MARGIN_BALANCES.items() if k != "account_type"}
    sub = _account_sub_object(response)
    assert sub["option_buying_power"] == 35596.72
    print("sub_object: falls back to first-known sub-object when account_type missing")


def test_missing_buying_power_raises():
    bogus = {"account_type": "margin", "margin": {}, "total_equity": 100.0}
    try:
        _extract_option_buying_power(bogus)
    except ValueError as e:
        assert "option_buying_power" in str(e)
        print("missing_bp: raises ValueError with helpful message")
        return
    raise AssertionError("expected ValueError")


def main() -> int:
    test_pdt_account_extracts_option_buying_power()
    test_margin_account_extracts_option_buying_power()
    test_cash_account_falls_back_to_cash_available()
    test_pdt_account_extracts_margin_requirement()
    test_margin_account_extracts_margin_requirement()
    test_account_sub_object_uses_account_type_field()
    test_account_sub_object_falls_back_when_account_type_missing()
    test_missing_buying_power_raises()
    print("all portfolio_state tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
