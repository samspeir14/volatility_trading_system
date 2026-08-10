"""DEPLOY BLOCKER for the cost gate: verify Tradier's vega unit convention
against one live option chain before the first sandbox scan trusts
signals/cost_gate.py.

The Black-Scholes ATM vega per share is ~ 0.4 * S * sqrt(T_years) / 100 per
1 PERCENTAGE POINT of IV (and 100x that if quoted per 1.00 of vol). This
script fetches a liquid chain, compares the quoted ATM vega against both
theoretical values, and says which convention Tradier is using. If the
verdict is "per 1.00 of vol", flip VEGA_PER_PCT_POINT to False in
signals/cost_gate.py before deploying.

Run:  .venv/bin/python -m scripts.verify_vega_units [SYMBOL]
"""
from __future__ import annotations

import asyncio
import math
import sys
from datetime import date

from config import load_settings
from data import AsyncTradierClient
from signals.cost_gate import VEGA_PER_PCT_POINT
from signals.signal_generator import find_atm_iv


async def run(symbol: str) -> int:
    settings = load_settings()
    async with AsyncTradierClient(settings) as client:
        quote = await client.get_quote(symbol)
        spot = float(quote.get("last") or 0.0)
        if spot <= 0:
            print(f"no last price for {symbol}", file=sys.stderr)
            return 1
        expirations = await client.get_expirations(symbol)
        # nearest expiration 20-45 days out — the liquid monthly zone
        target = [e for e in expirations if 20 <= (e - date.today()).days <= 45]
        if not target:
            print("no expiration in the 20-45 DTE window", file=sys.stderr)
            return 1
        expiration = target[0]
        dte = (expiration - date.today()).days
        chain = await client.get_chain(symbol, expiration)
        atm = find_atm_iv(chain, spot)
        if atm is None:
            print("no ATM pair found", file=sys.stderr)
            return 1
        atm_call, _atm_put = atm

        t_years = dte / 365.0
        theo_per_point = 0.4 * spot * math.sqrt(t_years) / 100.0
        theo_per_unit = theo_per_point * 100.0
        quoted = atm_call.vega

        print(f"{symbol} spot={spot:.2f} exp={expiration} (dte={dte}) "
              f"ATM call k={atm_call.strike} iv={atm_call.iv:.3f}")
        print(f"quoted vega:            {quoted:.4f}")
        print(f"theoretical per 1% pt:  {theo_per_point:.4f}")
        print(f"theoretical per 1.00:   {theo_per_unit:.4f}")

        per_point = abs(quoted - theo_per_point) < abs(quoted - theo_per_unit)
        verdict = "per 1 PERCENTAGE POINT" if per_point else "per 1.00 of vol"
        print(f"\nverdict: Tradier vega is {verdict}")
        if per_point == VEGA_PER_PCT_POINT:
            print("cost_gate.VEGA_PER_PCT_POINT matches — gate is safe to trust")
            return 0
        print("MISMATCH: flip VEGA_PER_PCT_POINT in signals/cost_gate.py "
              "before deploying the cost gate", file=sys.stderr)
        return 1


def main() -> int:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    return asyncio.run(run(symbol))


if __name__ == "__main__":
    sys.exit(main())
