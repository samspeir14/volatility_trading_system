"""Reconciles the order log against Tradier's actual positions.

The order log doesn't know about expirations — when Tradier auto-expires an
option on the third Friday, the position vanishes from /accounts/{id}/positions
but our log still thinks it's open. That bleeds into list_open_positions(),
mark-to-market (NaN marks for missing legs), exit_manager (evaluating phantoms),
and the per-day P&L (cumulative position value frozen at last seen mark).

This module pulls Tradier's positions each cycle and:
  - For log entries with all legs missing AND today >= expiration:
    mark the order as expired with realized P&L derived from intrinsic value
    of each leg at the underlying's closing price on expiration day.
  - For log entries with all legs missing AND today < expiration:
    log a warning and leave alone — likely a stale-cache race; safer to
    do nothing than to mark closed prematurely.
  - For log entries whose underlying appears as a stock position in Tradier:
    record a persistent assignment alert. Assignment can cash out a covered
    short into shares that the bot doesn't know how to manage; flagging for
    manual intervention is the only safe move.

Realized P&L on expiration (computed per-leg from intrinsic value):
  long call:  +max(0, S_close − K) × qty × 100
  long put:   +max(0, K − S_close) × qty × 100
  short call: −max(0, S_close − K) × qty × 100  (we paid out)
  short put:  −max(0, K − S_close) × qty × 100
  realized = entry_cash_flow + Σ leg_payoffs
    (entry_cash_flow = −entry_premium·100 for BUY, +entry_premium·100 for SELL)

This handles all four cases correctly:
  - long position OTM → all leg payoffs 0 → realized = −entry_debit (worthless)
  - long position ITM → leg payoffs > 0 → realized = intrinsic − debit
  - short position all OTM → realized = +entry_credit (full credit kept)
  - short IC with one short breached but long wing covered (or both ITM and
    auto-exercised, leaving no net stock) → realized = credit − payout

If Tradier reports an actual stock position for the underlying we don't
take this path; that's the assignment-alert branch, manual handling only.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
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


def settle_intrinsic_pnl(
    legs: list[dict],
    underlying_close: float,
    direction: str,
    entry_premium: float,
) -> float:
    """Realized P&L at expiration assuming each leg cash-settles to its
    intrinsic value at `underlying_close`. Works for any combination of
    long/short call/put legs. Returns dollars (already × 100 multiplier).

    legs: list of {strike, option_type ('call'|'put'), side ('buy'|'sell'), quantity}
    direction: 'BUY' (debit) or 'SELL' (credit) — sign of entry cash flow
    entry_premium: positive absolute per-contract premium
    """
    entry_sign = -1 if direction == "BUY" else +1
    entry_cash = entry_sign * entry_premium * 100.0
    total_payoff = 0.0
    for leg in legs:
        strike = float(leg["strike"])
        opt = leg["option_type"]
        side = leg["side"]
        qty = int(leg.get("quantity", 1))
        if opt == "call":
            intrinsic = max(0.0, underlying_close - strike)
        elif opt == "put":
            intrinsic = max(0.0, strike - underlying_close)
        else:
            raise ValueError(f"unknown option_type: {opt!r}")
        leg_sign = +1 if side == "buy" else -1  # long collects intrinsic; short pays it
        total_payoff += leg_sign * intrinsic * qty * 100.0
    return entry_cash + total_payoff


def max_loss_dollars(legs: list[dict], direction: str, entry_premium: float) -> float:
    """Worst-case realized P&L at expiration. Used as a sanity bound on the
    intrinsic settlement — a computed realized below this is a calculation
    bug.

    Long (BUY): −entry_premium·100 (debit is the floor).
    Short defined-risk (SELL iron condor): −(wing_distance·100 − credit).
    Short undefined (naked short): unbounded; returns −inf.
    """
    if direction == "BUY":
        return -entry_premium * 100.0
    # Short: look for symmetric wings to bound max loss.
    calls = sorted([l for l in legs if l["option_type"] == "call"], key=lambda l: float(l["strike"]))
    puts = sorted([l for l in legs if l["option_type"] == "put"], key=lambda l: float(l["strike"]))
    if len(calls) == 2 and len(puts) == 2:
        # iron condor: short inner + long outer wings, symmetric
        call_wing = float(calls[1]["strike"]) - float(calls[0]["strike"])
        put_wing = float(puts[1]["strike"]) - float(puts[0]["strike"])
        wing = max(call_wing, put_wing)
        return -(wing * 100.0 - entry_premium * 100.0)
    return float("-inf")  # naked / undefined-risk — no bound


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
        per_contract_fee: float = 0.0,
    ):
        self._client = client
        self._log = order_log
        self._account_id = account_id
        self._per_contract_fee = per_contract_fee
        # Underlying close-price cache, scoped per reconcile() call.
        self._close_cache: dict[tuple[str, date], float | None] = {}

    async def _underlying_close_on(self, symbol: str, expiration: date) -> float | None:
        """Closing price of the underlying on `expiration` (or the most recent
        trading day on/before expiration). Cached for the duration of one
        reconcile() call. Returns None on fetch failure or empty history —
        caller must NOT auto-close the position in that case."""
        key = (symbol, expiration)
        if key in self._close_cache:
            return self._close_cache[key]
        try:
            # Fetch a small window so weekend/holiday expirations still resolve.
            bars = await self._client.get_history(
                symbol, expiration - timedelta(days=7), expiration,
            )
        except Exception as e:
            logger.warning("get_history(%s, %s) failed: %s", symbol, expiration, e)
            self._close_cache[key] = None
            return None
        if not isinstance(bars, list) or not bars:
            logger.warning("get_history(%s, %s) returned no bars", symbol, expiration)
            self._close_cache[key] = None
            return None
        # Prefer the exact expiration date; fall back to most recent ≤ expiration.
        exact = next((b for b in bars if b.date == expiration), None)
        chosen = exact or max(bars, key=lambda b: b.date)
        self._close_cache[key] = chosen.close
        return chosen.close

    async def reconcile(self, today: date) -> ReconciliationResult:
        # Per-call close-price cache: reset every reconcile() so a price that
        # was unavailable last cycle gets retried this cycle.
        self._close_cache = {}
        # Phase 1: resolve any timeout-status orders by querying Tradier's
        # authoritative state. Recovered fills flow into the main pass below
        # because open_unclosed_positions() filters on final_status IN
        # ('filled','partially_filled') — without this phase, expired ITM
        # positions that never confirmed their fill stay dark forever.
        timeouts_resolved = await self._recover_timeouts(today)

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

            # Past expiration + no legs + no stock → settle at intrinsic value.
            raw_fill = row["fill_price"] if row["fill_price"] is not None else row["submitted_price"]
            entry_premium = abs(float(raw_fill))
            underlying_close = await self._underlying_close_on(underlying, expiration)
            if underlying_close is None:
                logger.warning(
                    "reconciler: order %s (%s exp=%s) — no underlying close price "
                    "available, deferring closure to next cycle",
                    order_id, underlying, expiration.isoformat(),
                )
                skipped_premature.append(order_id)
                continue

            realized = settle_intrinsic_pnl(
                legs=legs, underlying_close=underlying_close,
                direction=row["direction"], entry_premium=entry_premium,
            )
            # Sanity bound: a long can't lose more than the debit; a defined-risk
            # short can't lose more than wing-distance × 100 − credit. If we
            # compute outside that band, our intrinsic math has a bug — refuse
            # to write a number that would silently corrupt the books.
            floor = max_loss_dollars(legs, row["direction"], entry_premium)
            if floor > float("-inf") and realized < floor - 0.01:
                logger.error(
                    "reconciler: order %s computed realized $%+.2f below max-loss "
                    "floor $%+.2f — refusing to mark closed (intrinsic calc bug?)",
                    order_id, realized, floor,
                )
                skipped_premature.append(order_id)
                continue

            trigger = "expired_worthless" if abs(realized - floor) < 0.01 and row["direction"] == "BUY" \
                else "expired"
            # Only the opening order ever executed (the legs expired), so net
            # out one side's contract fees. The floor check and trigger above
            # stay on the gross number — the floor is a theoretical payoff
            # bound that doesn't include fees.
            open_fees = self._per_contract_fee * sum(
                int(leg["quantity"]) for leg in legs
            )
            self._log.record_expiration(
                opening_order_id=order_id,
                expired_at=now,
                realized_pnl=realized - open_fees,
                exit_trigger=trigger,
            )
            logger.info(
                "reconciler: order %s (%s %s %s exp=%s) marked %s "
                "realized_pnl=$%+.2f (entry_premium=$%.2f S_close=$%.2f fees=$%.2f)",
                order_id, underlying, row["structure"], row["direction"],
                expiration.isoformat(), trigger, realized - open_fees,
                entry_premium, underlying_close, open_fees,
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

    async def _recover_timeouts(self, today: date) -> list[TimeoutResolution]:
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
            try:
                expiration = date.fromisoformat(row["expiration"])
            except (TypeError, ValueError):
                expiration = None
            new_status, fill_price = await self._query_terminal_status(
                order_id, expiration=expiration, today=today,
            )
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

    async def _query_terminal_status(
        self,
        order_id: int,
        expiration: date | None = None,
        today: date | None = None,
    ) -> tuple[str | None, float | None]:
        """Returns (status, fill_price). status=None means we couldn't determine
        (404, network error, or Tradier reports a non-terminal state). Caller
        should leave the row as 'timeout' and retry next cycle.

        Exception: an order Tradier still reports as non-terminal (e.g. wedged
        in the sandbox's PendingNew state) with zero executed quantity, whose
        legs' expiration is already past, can never fill — resolve it as
        'expired' instead of warning forever."""
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
            try:
                exec_qty = float(order_node.get("exec_quantity") or 0)
            except (TypeError, ValueError):
                exec_qty = 0.0
            if (
                expiration is not None
                and today is not None
                and today > expiration
                and exec_qty == 0
            ):
                logger.info(
                    "order %s: stuck non-terminal (%r) with 0 executed and "
                    "expiration %s past — resolving as expired",
                    order_id, status, expiration.isoformat(),
                )
                return "expired", None
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
