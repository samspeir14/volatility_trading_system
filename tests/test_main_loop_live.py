"""Live end-to-end test of MainLoop.run_once against the Tradier sandbox.

This is the same code path that systemd will run in production, just one
iteration. Mutates sandbox state (places orders, manages exits) — by design.
"""
import asyncio
import logging
import sys
import time

from config import load_settings
from data import AsyncTradierClient
from main import build_main_loop


async def main_async() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    settings = load_settings()
    if settings.env != "sandbox":
        print(f"refusing live test against env={settings.env!r}", file=sys.stderr)
        return 2

    async with AsyncTradierClient(settings) as client:
        loop, closeables = build_main_loop(settings, client)
        try:
            t0 = time.monotonic()
            result = await loop.run_once()
            elapsed = time.monotonic() - t0

            print(f"\nrun_once result:")
            print(f"  market_open: {result.market_open}")
            print(f"  timestamp: {result.timestamp}")
            print(f"  equity: ${result.equity:,.2f}" if result.equity else "  equity: N/A")
            if result.today_total_pnl is not None:
                print(f"  today_total_pnl: ${result.today_total_pnl:+,.2f}")
            print(f"  scan_contracts: {result.scan_contracts}")
            print(f"  signals: {result.signals_total} total / "
                  f"{result.signals_actionable} actionable / {result.signals_approved} approved")
            print(f"  submissions: {result.submissions_filled} filled, "
                  f"{result.submissions_failed} failed")
            print(f"  exits: {result.exits_evaluated} evaluated, "
                  f"{result.exits_closed} closed")
            print(f"  kill_switch_active: {result.kill_switch_active}")
            print(f"  error: {result.error}")
            print(f"  elapsed: {elapsed:.1f}s")

            assert result.error is None, f"cycle reported error: {result.error}"
            if result.market_open:
                assert result.scan_contracts is not None and result.scan_contracts > 0
                assert result.equity is not None and result.equity > 0
                assert result.signals_total is not None
            else:
                print("  (market closed — limited assertions)")

            assert elapsed < 120, f"FAIL: cycle took {elapsed:.1f}s, gate <120s"
            print(f"\nlive main_loop test complete")
        finally:
            for c in closeables:
                try:
                    c.close()
                except Exception:
                    pass
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
