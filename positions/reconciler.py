"""Reconciles the order log against Tradier's actual positions.

The order log doesn't know about expirations — when Tradier auto-expires an
option on the third Friday, the position vanishes from /accounts/{id}/positions
but our log still thinks it's open. That bleeds into list_open_positions(),
mark-to-market (NaN marks for missing legs), exit_manager (evaluating phantoms),
and the per-day P&L (cumulative position value frozen at last seen mark).

This module pulls Tradier's positions each cycle and:
  - For log entries with all legs missing AND today >= expiration:
    mark the order as expired_worthless with the appropriate realized P&L.
  - For log entries with all legs missing AND today < expiration:
    log a warning and leave alone — likely a stale-cache race; safer to
    do nothing than to mark closed prematurely.
  - For log entries whose underlying appears as a stock position in Tradier:
    record a persistent assignment alert. Assignment can cash out a covered
    short into shares that the bot doesn't know how to manage; flagging for
    manual intervention is the only safe move.

Realized P&L on worthless expiration:
  - Long position (paid debit): realized = -entry_premium × 100
    (the debit is a sunk cost; nothing comes back)
  - Short position (received credit): realized = +entry_premium × 100
    (full credit retained, no buyback needed)

For multi-leg structures with mixed ITM/OTM outcomes Tradier may auto-exercise
some legs and assign others; that's the assignment-alert path, not this one.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from data.async_client import AsyncTradierClient
from execution.order_log import OrderLog

logger = logging.getLogger(__name__)


# OCC option symbol: underlying (1-6 alphanum) + YYMMDD + C/P + 8-digit strike.
# Example: AAPL250620C00150000 → AAPL 2025-06-20 call strike $150
_OCC_OPTION_RE = re.compile(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$")


def is_option_symbol(symbol: str) -> bool:
    return bool(_OCC_OPTION_RE.match(symbol))


def underlying_of_option(option_symbol: str) -> str:
    """Strip the OCC suffix to get the underlying ticker. Caller must ensure
    this is actually an option symbol."""
    m = re.match(r"^([A-Z]{1,6})\d{6}[CP]\d{8}$", option_symbol)
    return m.group(1) if m else option_symbol


@dataclass(frozen=True)
class AssignmentAlert:
    tradier_order_id: int
    symbol: str
    expiration: date
    structure: str
    stock_quantity: float | None


@dataclass(frozen=True)
class TimeoutResolution:
    """One row per timeout-status order the reconciler asked Tradier about.
    new_status='unknown' means Tradier didn't give us a usable answer (404,
    network error, malformed response, or still non-terminal) — the log row
    is left as 'timeout' and we'll retry next cycle."""
    tradier_order_id: int
    new_status: str           # filled | partially_filled | rejected | canceled | expired | unknown
    fill_price: float | None  # populated when new_status involves a fill


@dataclass(frozen=True)
class ReconciliationResult:
    expired_closed: list[int]              # order IDs marked expired this cycle
    assignment_alerts: list[AssignmentAlert]  # new alerts raised this cycle
    skipped_premature: list[int]           # missing-but-not-yet-expired (warning only)
    timeouts_resolved: list[TimeoutResolution]  # timeouts queried this cycle


class PositionReconciler:
    """Pulls Tradier positions, compares to the order log, and closes out
    expired positions / flags assignments. Intended to run once per cycle
    before snapshot/signal generation."""

    def __init__(
        self,
        client: AsyncTradierClient,
        order_log: OrderLog,
        account_id: str,
    ):
        self._client = client
        self._log = order_log
        self._account_id = account_id

    async def reconcile(self, today: date) -> ReconciliationResult:
        # Phase 1: resolve any timeout-status orders by querying Tradier's
        # authoritative state. Recovered fills flow into the main pass below
        # because open_unclosed_positions() filters on final_status IN
        # ('filled','partially_filled') — without this phase, expired ITM
        # positions that never confirmed their fill stay dark forever.
        timeouts_resolved = await self._recover_timeouts()

        try:
            tradier_positions = await self._client.get_positions(self._account_id)
        except Exception as e:
            logger.error("get_positions failed during reconciliation: %s — "
                         "skipping cycle to avoid spurious closures", e)
            return ReconciliationResult([], [], [], timeouts_resolved)

        # Split Tradier-reported holdings: option leg symbols vs stock underlyings.
        tradier_option_symbols: set[str] = set()
        tradier_stock_positions: dict[str, float] = {}
        for p in tradier_positions:
            sym = p.get("symbol") or ""
            qty = float(p.get("quantity") or 0)
            if is_option_symbol(sym):
                tradier_option_symbols.add(sym)
            elif sym:
                tradier_stock_positions[sym] = qty

        open_rows = self._log.open_unclosed_positions()
        existing_alerts = {a["tradier_order_id"] for a in self._log.assignment_alerts_active()}

        expired_closed: list[int] = []
        new_alerts: list[AssignmentAlert] = []
        skipped_premature: list[int] = []
        now = datetime.now(timezone.utc)

        for row in open_rows:
            order_id = row["tradier_order_id"]
            underlying = row["symbol"]
            expiration = date.fromisoformat(row["expiration"])

            try:
                legs = json.loads(row["legs_json"])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("reconciler: bad legs_json for order %s: %s — skipping",
                               order_id, e)
                continue

            leg_symbols = {leg["contract_symbol"] for leg in legs if "contract_symbol" in leg}
            if not leg_symbols:
                continue

            still_present = leg_symbols & tradier_option_symbols

            # Assignment check — independent of leg presence, because an iron
            # condor with one short ITM may still have OTM legs reported plus
            # an inherited stock position. Trigger on stock holding alone.
            stock_qty = tradier_stock_positions.get(underlying)
            if stock_qty is not None and stock_qty != 0:
                if order_id not in existing_alerts:
                    logger.critical(
                        "ASSIGNMENT ALERT: order %s (%s %s exp=%s) — Tradier "
                        "reports %s stock position of %.0f shares. Manual "
                        "intervention required; bot will not manage stock.",
                        order_id, underlying, row["structure"],
                        expiration.isoformat(), underlying, stock_qty,
                    )
                    alert = AssignmentAlert(
                        tradier_order_id=order_id, symbol=underlying,
                        expiration=expiration, structure=row["structure"],
                        stock_quantity=stock_qty,
                    )
                    self._log.record_assignment_alert(
                        tradier_order_id=order_id, symbol=underlying,
                        expiration=expiration, structure=row["structure"],
                        detected_at=now, stock_quantity=stock_qty,
                    )
                    new_alerts.append(alert)
                continue  # don't auto-close — manual handling

            if still_present:
                # At least one leg still alive in Tradier → position is live.
                continue

            # All legs missing. Was this past expiration?
            if today < expiration:
                logger.warning(
                    "reconciler: order %s (%s exp=%s) has no legs in Tradier "
                    "but expiration is in the future — possible API/cache "
                    "race, leaving log entry untouched",
                    order_id, underlying, expiration.isoformat(),
                )
                skipped_premature.append(order_id)
                continue

            # Past expiration + no legs + no stock → expired worthless.
            raw_fill = row["fill_price"] if row["fill_price"] is not None else row["submitted_price"]
            entry_premium = abs(float(raw_fill))
            is_long = row["direction"] == "BUY"
            # Long: paid the debit, nothing comes back → -debit
            # Short: kept the credit → +credit
            realized = (-entry_premium if is_long else +entry_premium) * 100.0

            self._log.record_expiration(
                opening_order_id=order_id,
                expired_at=now,
                realized_pnl=realized,
            )
            logger.info(
                "reconciler: order %s (%s %s %s exp=%s) marked expired_worthless "
                "realized_pnl=$%+.2f (entry_premium=$%.2f direction=%s)",
                order_id, underlying, row["structure"], row["direction"],
                expiration.isoformat(), realized, entry_premium, row["direction"],
            )
            expired_closed.append(order_id)

        if expired_closed or new_alerts or skipped_premature or timeouts_resolved:
            logger.info(
                "reconciliation cycle: %d expired, %d new alerts, %d premature-skipped, "
                "%d timeouts resolved",
                len(expired_closed), len(new_alerts), len(skipped_premature),
                len(timeouts_resolved),
            )
        return ReconciliationResult(
            expired_closed=expired_closed,
            assignment_alerts=new_alerts,
            skipped_premature=skipped_premature,
            timeouts_resolved=timeouts_resolved,
        )

    async def _recover_timeouts(self) -> list[TimeoutResolution]:
        """Walk every timeout-status row, query Tradier for its real terminal
        state, update the log. Recovered fills then move into the main
        reconciliation pass (the expiration check below will close them out
        if Tradier no longer reports the legs)."""
        timeout_rows = self._log.timeout_orders()
        if not timeout_rows:
            return []
        resolutions: list[TimeoutResolution] = []
        now = datetime.now(timezone.utc)
        for row in timeout_rows:
            order_id = row["tradier_order_id"]
            new_status, fill_price = await self._query_terminal_status(order_id)
            if new_status is None:
                resolutions.append(TimeoutResolution(order_id, "unknown", None))
                continue
            filled_at = now if new_status in {"filled", "partially_filled"} else None
            self._log.update_terminal_state(
                order_id=order_id, status=new_status, fill_price=fill_price,
                filled_at=filled_at,
                error=None if new_status in {"filled", "partially_filled"} else
                      f"recovered from timeout: {new_status}",
            )
            logger.info(
                "reconciler: recovered timeout order %s (%s %s exp=%s) → %s "
                "fill_price=%s",
                order_id, row["symbol"], row["structure"], row["expiration"],
                new_status, f"{fill_price:.4f}" if fill_price is not None else "n/a",
            )
            resolutions.append(TimeoutResolution(order_id, new_status, fill_price))
        return resolutions

    async def _query_terminal_status(self, order_id: int) -> tuple[str | None, float | None]:
        """Returns (status, fill_price). status=None means we couldn't determine
        (404, network error, or Tradier reports a non-terminal state). Caller
        should leave the row as 'timeout' and retry next cycle."""
        try:
            resp = await self._client.get_order_status(self._account_id, order_id)
        except Exception as e:
            logger.warning(
                "get_order_status failed for order %s: %s — leaving as timeout",
                order_id, e,
            )
            return None, None
        order_node = resp.get("order") if isinstance(resp, dict) else None
        if not isinstance(order_node, dict):
            logger.warning("order %s: malformed get_order_status response, leaving as timeout",
                           order_id)
            return None, None
        status = (order_node.get("status") or "").lower()
        TERMINAL = {"filled", "partially_filled", "rejected", "canceled", "expired"}
        if status not in TERMINAL:
            logger.warning(
                "order %s: Tradier reports non-terminal status %r — leaving as timeout",
                order_id, status,
            )
            return None, None
        fill_price = None
        if status in {"filled", "partially_filled"}:
            raw = order_node.get("avg_fill_price") or order_node.get("price")
            try:
                fill_price = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                fill_price = None
        return status, fill_price
