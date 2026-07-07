"""Screen watchlist-expansion candidates for harvest mode.

Per candidate: price (batched quote), weekly-expiration availability, and for
the nearest 5-15 DTE expiration the ATM relative spread / OI / IV, then the
per-contract max loss (0.4 * S * IV * sqrt(dte/252) * 100) vs the $1,500
per-trade budget. PASS = weeklies + ATM rel spread <= 10% + ATM OI >= 50 +
at least 1 contract inside the budget.

Run on EC2 (needs .env credentials):
  ssh ec2 '~/options-trader/venv/bin/python -' < experiments/screen_watchlist_candidates.py

IMPORTANT: run DURING MARKET HOURS — off-hours quotes make spread/OI columns
unreliable (liquid ETFs showing 100%+ spreads and zero OI are stale-quote
artifacts, not reality). First run 2026-07-06 (off-hours) passed IWM DIA TLT
WFC ORCL QCOM CRM PLTR DIS KO WMT SLB OXY (added to the watchlist); the
wide-spread/thin-oi rejects (GLD MS C MCD SBUX CSCO MRK PEP XLE XLF PFE F GE)
deserve an intraday rerun before final judgment. Gentle on the shared rate
budget: one batched quote call, one expirations + one chain call per
candidate, 0.6s sleeps (~60 calls total).
"""
import asyncio
import math
import sys
from datetime import date

sys.path.insert(0, "/home/ubuntu/options-trader")
import os
os.chdir("/home/ubuntu/options-trader")

from config.settings import load_settings  # noqa: E402
from data.async_client import AsyncTradierClient  # noqa: E402

CANDIDATES = {
    "IWM": "etf", "DIA": "etf", "GLD": "etf", "TLT": "etf", "XLE": "etf",
    "XLF": "etf", "SMH": "etf",
    "MS": "financials", "WFC": "financials", "C": "financials", "SCHW": "financials",
    "PFE": "healthcare", "MRK": "healthcare", "ABBV": "healthcare",
    "INTC": "tech", "MU": "tech", "CSCO": "tech", "ORCL": "tech",
    "QCOM": "tech", "CRM": "tech", "PLTR": "tech",
    "DIS": "consumer", "KO": "consumer", "PEP": "consumer", "WMT": "consumer",
    "MCD": "consumer", "SBUX": "consumer", "F": "consumer",
    "SLB": "energy", "OXY": "energy",
    "GE": "industrials",
}
BUDGET = 1500.0


async def main() -> None:
    settings = load_settings()
    today = date.today()
    async with AsyncTradierClient(settings) as client:
        quotes = await client.get_quotes(list(CANDIDATES))
        print(f"{'sym':>5} {'sector':>11} {'price':>8} {'wkly':>4} {'dte':>3} "
              f"{'atm_iv':>6} {'spr%':>5} {'oi':>6} {'maxloss':>8} {'qty':>3}  verdict")
        for sym, sector in CANDIDATES.items():
            q = quotes.get(sym) or {}
            px = float(q.get("last") or 0)
            if px <= 0:
                print(f"{sym:>5} {sector:>11}   no quote")
                continue
            try:
                exps = await client.get_expirations(sym)
            except Exception as e:
                print(f"{sym:>5} {sector:>11} {px:8.2f}  expirations failed: {e}")
                continue
            near = [e for e in exps if 0 < (e - today).days <= 21]
            weekly = len(near) >= 3
            window = [e for e in exps if 5 <= (e - today).days <= 15]
            if not window:
                print(f"{sym:>5} {sector:>11} {px:8.2f} {'Y' if weekly else 'N':>4}  "
                      f"no 5-15 DTE expiration")
                continue
            exp = window[0]
            dte = (exp - today).days
            try:
                chain = await client.get_chain(sym, exp)
            except Exception as e:
                print(f"{sym:>5} {sector:>11} {px:8.2f}  chain failed: {e}")
                continue
            calls = [c for c in chain if c.option_type == "call"]
            puts = [c for c in chain if c.option_type == "put"]
            if not calls or not puts:
                print(f"{sym:>5} {sector:>11} {px:8.2f}  empty chain side")
                continue
            atm_c = min(calls, key=lambda c: abs(c.strike - px))
            atm_p = min(puts, key=lambda c: abs(c.strike - px))
            ivs = [c.iv for c in (atm_c, atm_p) if c.iv and c.iv > 0]
            iv = sum(ivs) / len(ivs) if ivs else float("nan")
            def rel(c):
                mid = (c.bid + c.ask) / 2
                return (c.ask - c.bid) / mid if mid > 0 else 9.9
            spread = max(rel(atm_c), rel(atm_p))
            oi = min(atm_c.open_interest, atm_p.open_interest)
            maxloss = 0.4 * px * iv * math.sqrt(dte / 252) * 100 if iv == iv else float("nan")
            qty = int(BUDGET // maxloss) if maxloss == maxloss and maxloss > 0 else 0
            ok = weekly and spread <= 0.10 and oi >= 50 and qty >= 1
            note = "PASS" if ok else " ".join(
                w for w, bad in [("no-weekly", not weekly), ("wide-spread", spread > 0.10),
                                 ("thin-oi", oi < 50), ("too-big", qty < 1)] if bad)
            print(f"{sym:>5} {sector:>11} {px:8.2f} {'Y' if weekly else 'N':>4} {dte:3d} "
                  f"{iv:6.2f} {spread:5.1%} {oi:6d} {maxloss:8.0f} {qty:3d}  {note}")
            await asyncio.sleep(0.6)


asyncio.run(main())
