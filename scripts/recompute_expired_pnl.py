"""One-time correction: recompute realized P&L for rows that the reconciler
previously closed under the (buggy) "always worthless" assumption.

The earlier reconciler booked every expired position at the loss/credit
extreme: long → −entry_debit, short → +entry_credit. That's wrong whenever
an option ended ITM: a long ITM straddle returns intrinsic value × 100;
a short IC with one breached short leg loses credit − payout, not zero.

This script walks every row with exit_trigger IN ('expired_worthless',
'expired'), fetches the underlying's closing price on the expiration day
from Tradier's history endpoint, recomputes realized via the same
intrinsic-settlement formula the live reconciler now uses, and updates
the row. Prints before/after per row and aggregate so the correction is
auditable.

Safe to re-run — only updates rows whose recomputed realized differs from
the stored value by more than a cent. Caveat: rows carry no record of the
fee regime they traded under, so the fee to net out of expired rows is an
EXPLICIT argument (default 0.0 = gross, correct for paper history), NOT the
live PER_CONTRACT_FEE from .env — otherwise re-running after going live
would silently rewrite paper-era rows with today's fee.

Usage:
    cd ~/options-trader
    PYTHONPATH=. venv/bin/python scripts/recompute_expired_pnl.py
    # live rows only, Tradier Pro pass-throughs:
    PYTHONPATH=. venv/bin/python scripts/recompute_expired_pnl.py --per-contract-fee 0.10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from config import load_settings
from data.async_client import AsyncTradierClient
from execution.order_log import OrderLog
from positions.reconciler import max_loss_dollars, settle_intrinsic_pnl


async def fetch_close(client, symbol: str, expiration: date) -> float | None:
    try:
        bars = await client.get_history(
            symbol, expiration - timedelta(days=7), expiration,
        )
    except Exception as e:
        print(f"  get_history({symbol}, {expiration}) failed: {e}")
        return None
    if not bars:
        print(f"  get_history({symbol}, {expiration}) returned no bars")
        return None
    exact = next((b for b in bars if b.date == expiration), None)
    chosen = exact or max(bars, key=lambda b: b.date)
    return chosen.close


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute expired rows' realized P&L at intrinsic value."
    )
    parser.add_argument(
        "--per-contract-fee", type=float, default=0.0,
        help="fee per contract to net out of each expired row's open side "
             "(default 0.0 = gross; deliberately not read from .env — see "
             "module docstring)",
    )
    args = parser.parse_args()
    if args.per_contract_fee < 0:
        print("ERROR: --per-contract-fee must be >= 0", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=logging.WARNING,  # quiet — script does its own printing
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    settings = load_settings()
    cache_dir = settings.cache_db_path.parent
    db_path = cache_dir / "order_log.db"
    if not db_path.exists():
        print(f"ERROR: order log not found at {db_path}", file=sys.stderr)
        return 2

    order_log = OrderLog(db_path)
    rows = order_log._conn.execute(
        "SELECT tradier_order_id, symbol, structure, direction, expiration, "
        "ABS(fill_price) AS entry_premium, realized_pnl, legs_json, exit_trigger "
        "FROM submitted_orders "
        "WHERE exit_trigger IN ('expired_worthless', 'expired') "
        "ORDER BY expiration, tradier_order_id"
    ).fetchall()
    if not rows:
        print("No expired rows found.")
        order_log.close()
        return 0

    print(f"Examining {len(rows)} expired row(s)…\n")
    print(f"{'order_id':<11} {'sym':<5} {'dir':<5} {'struct':<13} {'exp':<11} "
          f"{'close':>8}  {'old_pnl':>11} → {'new_pnl':>11}  {'Δ':>9}")
    print("-" * 100)

    cache: dict[tuple[str, date], float | None] = {}
    updates: list[tuple[int, float, float, float]] = []  # (oid, old, new, delta)
    skipped: list[int] = []
    floor_breaches: list[int] = []

    async with AsyncTradierClient(settings) as client:
        for r in rows:
            oid, symbol, structure, direction, exp_str = r[0], r[1], r[2], r[3], r[4]
            entry_premium, old_pnl, legs_json, trigger = r[5], r[6], r[7], r[8]
            expiration = date.fromisoformat(exp_str)
            try:
                legs = json.loads(legs_json)
            except (json.JSONDecodeError, TypeError):
                print(f"{oid:<11} {symbol:<5} {direction:<5} {structure:<13} "
                      f"{exp_str:<11}  bad legs_json — skipping")
                skipped.append(oid)
                continue

            key = (symbol, expiration)
            if key in cache:
                close = cache[key]
            else:
                close = await fetch_close(client, symbol, expiration)
                cache[key] = close
            if close is None:
                print(f"{oid:<11} {symbol:<5} {direction:<5} {structure:<13} "
                      f"{exp_str:<11}  no underlying close — skipping")
                skipped.append(oid)
                continue

            gross_pnl = settle_intrinsic_pnl(
                legs=legs, underlying_close=close,
                direction=direction, entry_premium=float(entry_premium),
            )
            floor = max_loss_dollars(legs, direction, float(entry_premium))
            if floor > float("-inf") and gross_pnl < floor - 0.01:
                print(f"{oid:<11} {symbol:<5} {direction:<5} {structure:<13} "
                      f"{exp_str:<11}  ⚠ computed ${gross_pnl:+,.2f} < floor "
                      f"${floor:+,.2f} — skipping")
                floor_breaches.append(oid)
                continue
            # Match the live reconciler: expired positions carry open-side
            # fees only; floor check and trigger stay on the gross number.
            open_fees = args.per_contract_fee * sum(
                int(leg["quantity"]) for leg in legs
            )
            new_pnl = gross_pnl - open_fees

            delta = new_pnl - (old_pnl or 0.0)
            print(f"{oid:<11} {symbol:<5} {direction:<5} {structure:<13} "
                  f"{exp_str:<11}  ${close:>7.2f}  ${old_pnl:>+10,.2f} → "
                  f"${new_pnl:>+10,.2f}  ${delta:>+8,.2f}")

            if abs(delta) > 0.01:
                new_trigger = "expired_worthless" \
                    if abs(gross_pnl - floor) < 0.01 and direction == "BUY" else "expired"
                order_log._conn.execute(
                    "UPDATE submitted_orders SET realized_pnl = ?, exit_trigger = ? "
                    "WHERE tradier_order_id = ?",
                    (new_pnl, new_trigger, oid),
                )
                updates.append((oid, old_pnl, new_pnl, delta))
        order_log._conn.commit()

    print("-" * 100)
    print(f"\nUpdated:  {len(updates)} rows")
    print(f"Skipped:  {len(skipped)} (no underlying close available)")
    print(f"Floor breach (calc bug guard): {len(floor_breaches)}")
    if updates:
        total_delta = sum(d for _, _, _, d in updates)
        old_sum = sum(o for _, o, _, _ in updates)
        new_sum = sum(n for _, _, n, _ in updates)
        print(f"\nAggregate correction across {len(updates)} updated rows:")
        print(f"  old realized total: ${old_sum:+,.2f}")
        print(f"  new realized total: ${new_sum:+,.2f}")
        print(f"  net Δ:              ${total_delta:+,.2f}")
        print(f"\nThis Δ is the bookkeeping correction applied to closed_today_pnl /")
        print(f"the order log. Equity is unchanged — Tradier had the real numbers")
        print(f"all along; only the log was wrong.")

    order_log.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
