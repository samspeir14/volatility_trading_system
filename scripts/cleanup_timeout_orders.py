"""One-time cleanup: resolve all timeout-status orders against Tradier and
mark any that have since expired.

Why: when the order-submission poll times out (Tradier slow to confirm the
fill), the row is left as final_status='timeout'. The bot's open-position
queries filter by status IN ('filled','partially_filled'), so timeout rows
are invisible — but they may have actually filled, accumulated P&L, and
since expired. This script walks every timeout row, queries Tradier's
authoritative state, updates the log, and runs the reconciler so that
recovered-and-expired positions get their realized P&L recorded.

Runs ONE reconciliation cycle (timeout recovery + expiration check) and
prints a per-order summary plus aggregate counts. Safe to run repeatedly;
each call only acts on remaining ambiguity.

Usage:
    cd ~/options-trader
    PYTHONPATH=. venv/bin/python scripts/cleanup_timeout_orders.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

from config import load_settings
from data.async_client import AsyncTradierClient
from execution.order_log import OrderLog
from positions.reconciler import PositionReconciler


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    settings = load_settings()
    cache_dir = settings.cache_db_path.parent
    db_path = cache_dir / "order_log.db"
    if not db_path.exists():
        print(f"ERROR: order log not found at {db_path}", file=sys.stderr)
        return 2

    order_log = OrderLog(db_path)
    try:
        before_timeout = order_log.timeout_orders()
        before_open_raw = order_log._conn.execute(
            "SELECT COUNT(*) FROM submitted_orders WHERE closing_order_id IS NULL"
        ).fetchone()[0]
        before_open_active = len(order_log.open_unclosed_positions())

        print(f"BEFORE: {len(before_timeout)} timeout rows, "
              f"{before_open_active} active-open, "
              f"{before_open_raw} raw-open (closing_order_id IS NULL)")
        print()

        if before_timeout:
            print("Timeout rows to resolve:")
            for r in before_timeout:
                print(f"  id={r['tradier_order_id']:<11} {r['symbol']:<6} "
                      f"{r['structure']:<13} {r['direction']:<5} "
                      f"exp={r['expiration']} submitted={r['submitted_at']}")
            print()

        async with AsyncTradierClient(settings) as client:
            reconciler = PositionReconciler(
                client=client, order_log=order_log,
                account_id=settings.account_id,
                per_contract_fee=settings.per_contract_fee,
            )
            result = await reconciler.reconcile(date.today())

        print("=" * 60)
        print("RECONCILIATION RESULT")
        print("=" * 60)
        print(f"Timeouts resolved:    {len(result.timeouts_resolved)}")
        for r in result.timeouts_resolved:
            fp = f"@${r.fill_price:.4f}" if r.fill_price is not None else ""
            print(f"  order {r.tradier_order_id} → {r.new_status} {fp}")
        print()
        print(f"Expired/marked closed: {len(result.expired_closed)}")
        for oid in result.expired_closed:
            row = order_log._conn.execute(
                "SELECT symbol, structure, direction, expiration, realized_pnl "
                "FROM submitted_orders WHERE tradier_order_id = ?", (oid,),
            ).fetchone()
            if row:
                print(f"  order {oid} {row[0]} {row[1]} {row[2]} "
                      f"exp={row[3]} realized=${row[4]:+,.2f}")
        print()
        print(f"Assignment alerts (new): {len(result.assignment_alerts)}")
        for a in result.assignment_alerts:
            print(f"  order {a.tradier_order_id} {a.symbol} {a.structure} "
                  f"exp={a.expiration.isoformat()} stock_qty={a.stock_quantity}")
        print()
        print(f"Premature-skipped (missing but expiration in future): "
              f"{len(result.skipped_premature)}")
        for oid in result.skipped_premature:
            print(f"  order {oid}")
        print()

        after_timeout = order_log.timeout_orders()
        after_open_raw = order_log._conn.execute(
            "SELECT COUNT(*) FROM submitted_orders WHERE closing_order_id IS NULL"
        ).fetchone()[0]
        after_open_active = len(order_log.open_unclosed_positions())
        print("=" * 60)
        print(f"AFTER:  {len(after_timeout)} timeout rows, "
              f"{after_open_active} active-open, "
              f"{after_open_raw} raw-open")

        if after_timeout:
            print()
            print("Timeout rows remaining (Tradier could not resolve — retry later):")
            for r in after_timeout:
                print(f"  id={r['tradier_order_id']:<11} {r['symbol']:<6} "
                      f"exp={r['expiration']}")

        net_realized = order_log._conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) FROM submitted_orders "
            "WHERE tradier_order_id IN ({})".format(
                ",".join(str(o) for o in result.expired_closed) or "NULL"
            )
        ).fetchone()[0]
        print(f"\nNet realized P&L added by this cleanup: ${net_realized:+,.2f}")
    finally:
        order_log.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
