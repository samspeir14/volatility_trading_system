from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

from config import Settings
from data.async_client import AsyncTradierClient, OptionContract
from data.market_data import TickerSnapshot
from execution.order_log import (
    OrderLog,
    PENDING_CLOSE_STATUS,
    STALE_CANCELED_STATUS,
)
from signals.signal_generator import TradeLeg, TradeSignal

if TYPE_CHECKING:
    from positions.position_tracker import OpenPosition, PositionMark

logger = logging.getLogger(__name__)


MAX_QUANTITY_PER_LEG_DEFAULT = 10
MAX_PREMIUM_PER_TRADE_DEFAULT = 5000.0
STALE_ORDER_THRESHOLD_MINUTES_DEFAULT = 15
MAX_CLOSE_RETRIES_DEFAULT = 3
TERMINAL_STATES = {"filled", "partially_filled", "rejected", "canceled", "expired"}
TERMINAL_FAILED = {"rejected", "canceled", "expired"}
# States in which an order can no longer execute additional units. NOTE:
# Tradier's 'partially_filled' is a WORKING state (units filled, remainder
# live) — the partial-fill resolvers must never book executed quantities
# until the order reaches one of these.
DEAD_ORDER_STATES = {"filled", "rejected", "canceled", "expired"}


@dataclass(frozen=True)
class OrderResult:
    signal: TradeSignal | None  # None for closes (which have no opening signal context here)
    status: str  # filled | partially_filled | rejected | preview_failed | duplicate | guard_blocked | timeout | error
    order_id: int | None
    submitted_price: float | None
    fill_price: float | None
    error: str | None


@dataclass
class _OrderRequest:
    legs: list[dict]                # [{"option_symbol", "side", "quantity"}, ...]
    underlying_symbol: str
    order_type: str                 # "debit" | "credit"
    price: float                    # always positive — order_type determines sign
    structure: str                  # "straddle" | "iron_condor"
    estimated_premium: float        # absolute dollar premium, whole trade (× lots)


def fingerprint_signal(signal: TradeSignal, scan_date) -> str:
    """SHA256 (truncated) of canonical signal identity."""
    legs_part = ",".join(
        f"{leg.strike}:{leg.option_type}:{leg.side}"
        for leg in sorted(signal.legs, key=lambda l: (l.strike, l.option_type))
    )
    parts = [
        signal.symbol,
        signal.expiration.isoformat(),
        signal.direction,
        legs_part,
        scan_date.isoformat(),
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return digest[:16]


def compute_straddle_debit(call: OptionContract, put: OptionContract, slippage: float = 0.02) -> float:
    mid = (call.bid + call.ask) / 2 + (put.bid + put.ask) / 2
    return round(mid * (1 + slippage), 2)


def compute_iron_condor_credit(
    short_call: OptionContract, short_put: OptionContract,
    long_call: OptionContract, long_put: OptionContract,
    slippage: float = 0.02,
) -> float:
    short_mid = (short_call.bid + short_call.ask) / 2 + (short_put.bid + short_put.ask) / 2
    long_mid = (long_call.bid + long_call.ask) / 2 + (long_put.bid + long_put.ask) / 2
    net_credit = short_mid - long_mid
    return round(net_credit * (1 - slippage), 2)


def signal_to_request(
    signal: TradeSignal,
    snapshot: TickerSnapshot,
    slippage: float = 0.02,
    max_qty: int = MAX_QUANTITY_PER_LEG_DEFAULT,
) -> _OrderRequest:
    """Translate a TradeSignal into an order request."""
    if not signal.legs:
        raise ValueError("signal has no legs")

    # Map (strike, option_type) → OptionContract from the snapshot
    contracts_by_key: dict[tuple[float, str], OptionContract] = {
        (c.strike, c.option_type): c
        for c in snapshot.contracts
        if c.expiration == signal.expiration
    }

    api_legs: list[dict] = []
    leg_contracts: list[OptionContract] = []
    for leg in signal.legs:
        key = (leg.strike, leg.option_type)
        contract = contracts_by_key.get(key)
        if contract is None:
            raise ValueError(f"contract not found in chain: strike={leg.strike} type={leg.option_type}")
        qty = min(leg.quantity, max_qty)
        if qty < leg.quantity:
            logger.warning("clamping leg quantity %d -> %d (cap)", leg.quantity, qty)
        side_map = {"buy": "buy_to_open", "sell": "sell_to_open"}
        api_side = side_map.get(leg.side)
        if api_side is None:
            raise ValueError(f"unrecognized leg side: {leg.side}")
        api_legs.append({
            "option_symbol": contract.symbol,
            "side": api_side,
            "quantity": qty,
        })
        leg_contracts.append(contract)

    # Structure lot count: legs are scaled uniformly by the risk manager's
    # sized quantity, so min leg quantity = number of structure units. The
    # limit price stays per unit (Tradier multileg semantics); only the
    # premium-cap estimate below needs the whole-trade dollar figure.
    lots = max(1, min(leg["quantity"] for leg in api_legs))

    if signal.direction == "BUY":
        if len(leg_contracts) != 2:
            raise ValueError(f"BUY straddle expects 2 legs, got {len(leg_contracts)}")
        call, put = (leg_contracts[0], leg_contracts[1]) if leg_contracts[0].option_type == "call" else (leg_contracts[1], leg_contracts[0])
        price = compute_straddle_debit(call, put, slippage=slippage)
        structure = "straddle"
        order_type = "debit"
        estimated_premium = price * 100 * lots  # one contract = 100 shares
    else:
        if len(leg_contracts) != 4:
            raise ValueError(f"SELL iron condor expects 4 legs, got {len(leg_contracts)}")
        # Identify shorts vs longs by side
        shorts = [c for c, leg in zip(leg_contracts, signal.legs) if leg.side == "sell"]
        longs = [c for c, leg in zip(leg_contracts, signal.legs) if leg.side == "buy"]
        if len(shorts) != 2 or len(longs) != 2:
            raise ValueError("iron condor must have 2 shorts and 2 longs")
        short_call = next((c for c in shorts if c.option_type == "call"), None)
        short_put = next((c for c in shorts if c.option_type == "put"), None)
        long_call = next((c for c in longs if c.option_type == "call"), None)
        long_put = next((c for c in longs if c.option_type == "put"), None)
        if not all([short_call, short_put, long_call, long_put]):
            raise ValueError("iron condor missing required leg type")
        price = compute_iron_condor_credit(short_call, short_put, long_call, long_put, slippage=slippage)
        if price <= 0:
            raise ValueError(f"iron condor net credit not positive: {price}")
        structure = "iron_condor"
        order_type = "credit"
        # Max risk on a credit spread = (wing distance × 100) − net credit
        # Premium-cap proxy: use absolute credit × 100 × lots for sanity
        estimated_premium = price * 100 * lots

    return _OrderRequest(
        legs=api_legs,
        underlying_symbol=signal.symbol,
        order_type=order_type,
        price=price,
        structure=structure,
        estimated_premium=estimated_premium,
    )


class OrderManager:
    def __init__(
        self,
        client: AsyncTradierClient,
        order_log: OrderLog,
        settings: Settings,
        max_quantity_per_leg: int = MAX_QUANTITY_PER_LEG_DEFAULT,
        max_premium_per_trade: float = MAX_PREMIUM_PER_TRADE_DEFAULT,
        poll_interval_seconds: float = 2.0,
        poll_timeout_seconds: float = 30.0,
        slippage_buffer: float = 0.02,
        stale_order_threshold_minutes: int = STALE_ORDER_THRESHOLD_MINUTES_DEFAULT,
        max_close_retries: int = MAX_CLOSE_RETRIES_DEFAULT,
        ladder_steps: tuple[float, ...] | None = None,
        earnings_calendar=None,
    ):
        self._client = client
        self._log = order_log
        self._settings = settings
        # Earnings-exit fail-closed store: stamp each entry with the ticker's
        # next known earnings date so the exit rule survives calendar outages.
        self._earnings = earnings_calendar
        self._max_qty = max_quantity_per_leg
        self._max_premium = max_premium_per_trade
        self._fee_per_contract = settings.per_contract_fee
        self._poll_interval = poll_interval_seconds
        self._poll_timeout = poll_timeout_seconds
        self._stale_threshold_minutes = stale_order_threshold_minutes
        self._max_close_retries = max_close_retries
        # Price ladder ("walk the book"): orders start AT the theoretical mid
        # and reprice toward the far side in steps, instead of donating a
        # fixed haircut up front. Steps are fractions of mid; each step gets
        # an equal share of poll_timeout. The old static behavior (submit at
        # mid -/+ slippage_buffer and wait) donated slippage_buffer on every
        # fill that would have happened at mid. Default ladder tops out at
        # 1.5x slippage_buffer — a worse worst-case print than the old static
        # price, in exchange for better average fills and fewer misses; the
        # signal-level min_credit floor still bounds entry economics.
        if ladder_steps is None:
            ladder_steps = (
                0.0,
                slippage_buffer / 2,
                slippage_buffer,
                slippage_buffer * 1.5,
            )
        if not ladder_steps:
            raise ValueError("ladder_steps must have at least one step")
        self._ladder_steps = tuple(ladder_steps)
        self._ladder_step_seconds = poll_timeout_seconds / len(self._ladder_steps)

    def _check_production_guard(self) -> str | None:
        """Return error string if blocked, None if OK to proceed."""
        if self._settings.env == "production":
            if os.environ.get("TRADIER_LIVE_TRADING_CONFIRMED") != "YES":
                return (
                    "Refusing to submit orders in production mode. "
                    "Set TRADIER_LIVE_TRADING_CONFIRMED=YES if you really mean it."
                )
        return None

    async def submit(
        self,
        signal: TradeSignal,
        snapshot: TickerSnapshot,
        scan_date,
    ) -> OrderResult:
        now = datetime.now(timezone.utc)

        # 1. Production guard
        guard_error = self._check_production_guard()
        if guard_error is not None:
            logger.error("guard_blocked: %s", guard_error)
            return OrderResult(
                signal=signal, status="guard_blocked", order_id=None,
                submitted_price=None, fill_price=None, error=guard_error,
            )

        if not signal.is_actionable:
            return OrderResult(
                signal=signal, status="not_actionable", order_id=None,
                submitted_price=None, fill_price=None,
                error=signal.diagnostic_notes or "signal marked non-actionable",
            )

        # 2. Build request at the theoretical mid (slippage 0): the price
        # ladder walks from here toward the far side, and this same mid is
        # recorded as arrival_mid for TCA.
        try:
            request = signal_to_request(signal, snapshot, slippage=0.0, max_qty=self._max_qty)
        except ValueError as e:
            return OrderResult(
                signal=signal, status="preview_failed", order_id=None,
                submitted_price=None, fill_price=None,
                error=f"could not build request: {e}",
            )

        fingerprint = fingerprint_signal(signal, scan_date)

        # 3. Premium cap
        if request.estimated_premium > self._max_premium:
            error = (
                f"estimated premium ${request.estimated_premium:.2f} > cap ${self._max_premium:.2f}"
            )
            self._log.record_failure(signal, fingerprint, error, now)
            return OrderResult(
                signal=signal, status="preview_failed", order_id=None,
                submitted_price=request.price, fill_price=None, error=error,
            )

        # 4. Dedup
        if self._log.has_recent_open_order(fingerprint):
            logger.info("dedup hit for fingerprint %s (%s %s)", fingerprint, signal.symbol, signal.direction)
            return OrderResult(
                signal=signal, status="duplicate", order_id=None,
                submitted_price=request.price, fill_price=None,
                error=f"recent open order with fingerprint {fingerprint}",
            )

        # 5. Preview
        preview = await self._client.preview_order(
            account_id=self._settings.account_id,
            legs=request.legs,
            underlying_symbol=request.underlying_symbol,
            order_type=request.order_type,
            price=request.price,
        )
        preview_error = self._extract_error(preview)
        if preview_error is not None:
            self._log.record_failure(signal, fingerprint, f"preview: {preview_error}", now)
            return OrderResult(
                signal=signal, status="preview_failed", order_id=None,
                submitted_price=request.price, fill_price=None, error=preview_error,
            )

        # 6. Place
        place_resp = await self._client.place_order(
            account_id=self._settings.account_id,
            legs=request.legs,
            underlying_symbol=request.underlying_symbol,
            order_type=request.order_type,
            price=request.price,
        )
        place_error = self._extract_error(place_resp)
        order_node = place_resp.get("order") if isinstance(place_resp, dict) else None
        if place_error is not None or order_node is None or "id" not in order_node:
            err = place_error or f"unexpected place response: {place_resp}"
            self._log.record_failure(signal, fingerprint, f"place: {err}", now)
            return OrderResult(
                signal=signal, status="error", order_id=None,
                submitted_price=request.price, fill_price=None, error=err,
            )
        order_id = int(order_node["id"])
        next_earnings = None
        if self._earnings is not None:
            try:
                next_earnings = self._earnings.next_earnings_on_or_after(
                    signal.symbol, now.date(),
                )
            except Exception as e:
                logger.warning("could not stamp next earnings date for %s: %s",
                               signal.symbol, e)
        self._log.record_submission(
            signal=signal, fingerprint=fingerprint, structure=request.structure,
            submitted_price=request.price, order_id=order_id, submitted_at=now,
            arrival_mid=request.price, next_earnings_date=next_earnings,
        )
        logger.info("submitted order %d (%s %s %s) at mid %.2f (ladder %s)",
                    order_id, signal.symbol, signal.direction, request.structure,
                    request.price, self._ladder_steps)

        # 7. Walk the price ladder toward the far side until filled
        terminal_status, fill_price, last_price = await self._run_price_ladder(
            order_id, request.price, request.order_type,
        )
        if terminal_status is None:
            # Unfilled after the full walk: cancel so the order doesn't sit at
            # the exchange all day at a price we've stopped managing. A
            # canceled entry is retryable next cycle (dedup ignores canceled);
            # if the cancel fails or races a fill, fall back to the old
            # "timeout" status, which the reconciler resolves cross-cycle.
            terminal_status, fill_price = await self._cancel_unfilled_entry(order_id)
        if terminal_status is None:
            self._log.update_terminal_state(
                order_id=order_id, status="timeout", fill_price=None,
                filled_at=None, error="unfilled after price ladder; cancel unconfirmed",
            )
            return OrderResult(
                signal=signal, status="timeout", order_id=order_id,
                submitted_price=last_price, fill_price=None,
                error="order did not reach terminal state within ladder",
            )

        if terminal_status == "partially_filled":
            # Reachable only for multi-lot orders: cancel the residual and
            # book the executed units as the position (legs_json shrunk to
            # match), so the tracker manages exactly what is held.
            terminal_status, resolved_fill = await self._resolve_partial_entry(
                order_id, signal.symbol,
            )
            fill_price = resolved_fill if resolved_fill is not None else fill_price

        filled_at = datetime.now(timezone.utc) if terminal_status == "filled" else None
        self._log.update_terminal_state(
            order_id=order_id, status=terminal_status, fill_price=fill_price,
            filled_at=filled_at,
        )
        if terminal_status == "filled":
            self._log_fill_tca(
                "entry", order_id, request.order_type, request.price, fill_price,
            )
        return OrderResult(
            signal=signal, status=terminal_status, order_id=order_id,
            submitted_price=last_price, fill_price=fill_price, error=None,
        )

    async def _cancel_unfilled_entry(
        self, order_id: int,
    ) -> tuple[str | None, float | None]:
        """Cancel an entry order the ladder couldn't fill. Returns the terminal
        (status, fill_price) — 'canceled' on success, the real fill if the
        cancel raced one, or (None, None) when the state is unknown (caller
        falls back to 'timeout' for the reconciler to resolve)."""
        try:
            cancel_resp = await self._client.cancel_order(
                self._settings.account_id, order_id,
            )
        except Exception as e:
            logger.warning("cancel of unfilled entry %d failed: %r", order_id, e)
            return None, None
        if self._extract_error(cancel_resp) is None:
            logger.info("canceled unfilled entry %d after ladder", order_id)
            return "canceled", None
        # Cancel rejected — usually a race with a fill. Re-query once.
        try:
            resp = await self._client.get_order_status(
                self._settings.account_id, order_id,
            )
        except Exception as e:
            logger.warning("post-cancel status query for %d failed: %r", order_id, e)
            return None, None
        order_node = resp.get("order") if isinstance(resp, dict) else None
        if not isinstance(order_node, dict):
            return None, None
        status = (order_node.get("status") or "").lower()
        if status in TERMINAL_STATES:
            return status, self._extract_fill_price(order_node)
        return None, None

    @staticmethod
    def _node_units(order_node: dict) -> tuple[int | None, bool]:
        """(executed structure units, legs_even) from an order node. Units =
        min over legs of exec_quantity — the count of COMPLETE structures;
        uneven leg fills (legs_even=False) leave odd single-leg residue at
        the broker. Returns (None, True) when the response can't be parsed —
        a leg MISSING the exec_quantity field entirely is a parse failure,
        never a zero, so schema drift can't book a fill as 'nothing
        executed'."""
        legs = order_node.get("leg")
        try:
            if isinstance(legs, list) and legs:
                if any("exec_quantity" not in l for l in legs):
                    return None, True
                quantities = [int(float(l["exec_quantity"] or 0)) for l in legs]
                return min(quantities), len(set(quantities)) == 1
            if "exec_quantity" not in order_node:
                return None, True
            return int(float(order_node["exec_quantity"] or 0)), True
        except (TypeError, ValueError):
            return None, True

    async def _executed_units(
        self, order_id: int,
    ) -> tuple[str | None, int | None, float | None, bool]:
        """(order status, executed structure units, avg fill price,
        legs_even) for a (possibly partially) filled multileg order.
        status/units are None when Tradier's response can't be parsed. The
        STATUS is load-bearing for the partial-fill resolvers: only a dead
        order (DEAD_ORDER_STATES) may be booked from these numbers — a
        still-working order can keep filling after we read them."""
        try:
            resp = await self._client.get_order_status(
                self._settings.account_id, order_id,
            )
        except Exception as e:
            logger.warning("_executed_units(%d): status query failed: %r", order_id, e)
            return None, None, None, True
        order_node = resp.get("order") if isinstance(resp, dict) else None
        if not isinstance(order_node, dict):
            return None, None, None, True
        status = (order_node.get("status") or "").lower() or None
        avg_fill = self._extract_fill_price(order_node)
        units, legs_even = self._node_units(order_node)
        return status, units, avg_fill, legs_even

    async def _resolve_partial_entry(
        self, order_id: int, symbol: str,
    ) -> tuple[str, float | None]:
        """A multi-lot entry ended the ladder partially filled: cancel the
        still-working residual, and once the order is CONFIRMED DEAD shrink
        the stored legs to the executed unit count so the tracker manages
        exactly what is held. Returns the (status, fill_price) to book:
        'filled' at the executed size, 'canceled' when nothing executed, or
        'partially_filled' (unresolved) when the order may still be working
        or Tradier's answer is unusable. An unresolved row stays tracked at
        the SUBMITTED size and resolve_partial_entries() retries it every
        cycle until the order is dead and the row is corrected."""
        try:
            cancel_resp = await self._client.cancel_order(
                self._settings.account_id, order_id,
            )
            cancel_err = self._extract_error(cancel_resp)
        except Exception as e:
            cancel_err = repr(e)
        if cancel_err is not None:
            # Either a race with the remaining units filling, or the cancel
            # genuinely failed and the order is still working — the status
            # check below distinguishes the two.
            logger.warning(
                "cancel of partially-filled entry %d returned %r — re-querying",
                order_id, cancel_err,
            )
        status, units, avg_fill, legs_even = await self._executed_units(order_id)
        if status not in DEAD_ORDER_STATES:
            logger.critical(
                "PARTIAL ENTRY FILL UNRESOLVED: order %d (%s) status=%r after "
                "cancel — the order may STILL BE WORKING and filling more "
                "units. Row left partially_filled (tracked at submitted "
                "size); will retry resolution next cycle.",
                order_id, symbol, status,
            )
            return "partially_filled", avg_fill
        if units is None:
            logger.critical(
                "PARTIAL ENTRY FILL UNRESOLVED: order %d (%s) is dead (%s) "
                "but executed quantity could not be read; row left as "
                "partially_filled and tracked at SUBMITTED size. Verify at "
                "the broker (RUNBOOK section 2); will retry next cycle.",
                order_id, symbol, status,
            )
            return "partially_filled", avg_fill
        if units <= 0:
            if status == "filled":
                # Contradiction: dead-filled with zero executed units read.
                logger.critical(
                    "PARTIAL ENTRY FILL UNRESOLVED: order %d (%s) reports "
                    "status=filled but 0 executed units parsed — refusing to "
                    "book; will retry next cycle.", order_id, symbol,
                )
                return "partially_filled", avg_fill
            logger.info(
                "partial entry %d resolved as canceled (order %s, no complete "
                "units executed)", order_id, status,
            )
            return "canceled", None
        if not self._log.update_leg_quantities(order_id, units):
            logger.critical(
                "PARTIAL ENTRY FILL UNRESOLVED: order %d (%s) executed %d "
                "unit(s) but legs_json could not be rewritten — row left as "
                "partially_filled and tracked at SUBMITTED size. Flatten "
                "manually per RUNBOOK section 2.", order_id, symbol, units,
            )
            return "partially_filled", avg_fill
        if not legs_even:
            logger.critical(
                "PARTIAL ENTRY FILL UNEVEN: order %d (%s) legs executed "
                "unevenly — %d complete unit(s) booked, odd single-leg "
                "residue may remain at the broker. Check positions manually.",
                order_id, symbol, units,
            )
        logger.warning(
            "partial entry %d (%s): kept %d executed unit(s), residual "
            "dead (%s), legs rewritten", order_id, symbol, units, status,
        )
        return "filled", avg_fill

    async def resolve_partial_entries(self, now: datetime) -> int:
        """Re-attempt resolution of entry rows stuck at final_status=
        'partially_filled' — the unresolved outcomes of _resolve_partial_entry
        and the reconciler's timeout recovery. Runs every cycle from the main
        loop; each row is re-canceled/re-read until the order is confirmed
        dead and the row is booked at its true executed size. Returns the
        number of rows resolved."""
        rows = self._log.partial_entry_orders()
        resolved = 0
        for row in rows:
            status, fill = await self._resolve_partial_entry(
                row["tradier_order_id"], row["symbol"],
            )
            if status == "partially_filled":
                continue  # still unresolved — retry next cycle
            self._log.update_terminal_state(
                order_id=row["tradier_order_id"], status=status,
                fill_price=fill if fill is not None else row.get("fill_price"),
                filled_at=now if status == "filled" else None,
            )
            resolved += 1
            logger.warning(
                "resolve_partial_entries: order %d (%s) resolved to %s",
                row["tradier_order_id"], row["symbol"], status,
            )
        return resolved

    async def _poll_until_terminal(
        self, order_id: int, timeout_seconds: float | None = None,
    ) -> tuple[str | None, float | None]:
        deadline = time.monotonic() + (
            timeout_seconds if timeout_seconds is not None else self._poll_timeout
        )
        while time.monotonic() < deadline:
            await asyncio.sleep(self._poll_interval)
            resp = await self._client.get_order_status(self._settings.account_id, order_id)
            order_node = resp.get("order") if isinstance(resp, dict) else None
            if not order_node:
                continue
            status = order_node.get("status", "").lower()
            if status in TERMINAL_STATES:
                fill_price = order_node.get("avg_fill_price") or order_node.get("price")
                try:
                    fill_price_f = float(fill_price) if fill_price is not None else None
                except (TypeError, ValueError):
                    fill_price_f = None
                return status, fill_price_f
        return None, None

    @staticmethod
    def _ladder_price(base_mid: float, order_type: str, step: float) -> float:
        """Price at a ladder step: credits concede downward from mid, debits
        upward. step is a fraction of mid (0.0 = at mid)."""
        if order_type == "credit":
            return round(base_mid * (1 - step), 2)
        return round(base_mid * (1 + step), 2)

    async def _run_price_ladder(
        self,
        order_id: int,
        base_mid: float,
        order_type: str,
    ) -> tuple[str | None, float | None, float]:
        """Walk a working order's limit from mid toward the far side, one
        modify per step, polling between. Returns (terminal_status, fill_price,
        last_price); terminal_status None means unfilled after the full walk.
        A failed modify is not fatal — the order keeps working at its current
        price for the remaining step budget."""
        current_price = self._ladder_price(base_mid, order_type, self._ladder_steps[0])
        for i, step in enumerate(self._ladder_steps):
            if i > 0:
                next_price = self._ladder_price(base_mid, order_type, step)
                # Cheap contracts round to the same cent — skip no-op modifies.
                if next_price != current_price and next_price > 0:
                    try:
                        resp = await self._client.modify_order(
                            account_id=self._settings.account_id,
                            order_id=order_id,
                            order_type=order_type,
                            price=next_price,
                        )
                        modify_error = self._extract_error(resp)
                    except Exception as e:
                        modify_error = repr(e)
                    if modify_error is not None:
                        # Often a race: the order went terminal as we repriced.
                        # The poll below picks that up; otherwise keep working
                        # the old price.
                        logger.info(
                            "ladder modify order=%d -> %.2f failed (%s); "
                            "holding %.2f", order_id, next_price, modify_error,
                            current_price,
                        )
                    else:
                        logger.debug(
                            "ladder modify order=%d step=%d price %.2f -> %.2f",
                            order_id, i, current_price, next_price,
                        )
                        current_price = next_price
            status, fill_price = await self._poll_until_terminal(
                order_id, timeout_seconds=self._ladder_step_seconds,
            )
            if status is not None:
                return status, fill_price, current_price
        return None, None, current_price

    def _log_fill_tca(
        self,
        label: str,
        order_id: int,
        order_type: str,
        arrival_mid: float,
        fill_price: float | None,
    ) -> None:
        """One greppable line per fill: realized slippage vs the arrival mid
        (positive = worse than mid). This is the number that decides whether
        paper results survive real fills — reviewed via `grep TCA`."""
        if fill_price is None or arrival_mid <= 0:
            return
        abs_fill = abs(fill_price)
        slip = (
            arrival_mid - abs_fill if order_type == "credit"
            else abs_fill - arrival_mid
        )
        logger.info(
            "TCA %s order=%d arrival_mid=%.2f fill=%.2f slippage=%+.2f/contract "
            "(%+.1f%% of mid, positive=worse)",
            label, order_id, arrival_mid, abs_fill, slip,
            slip / arrival_mid * 100.0,
        )

    @staticmethod
    def _extract_error(resp: dict | None) -> str | None:
        if not isinstance(resp, dict):
            return f"unexpected response type: {type(resp).__name__}"
        if "errors" in resp:
            errs = resp["errors"]
            if isinstance(errs, dict):
                msg = errs.get("error")
                if isinstance(msg, list):
                    return "; ".join(str(m) for m in msg)
                return str(msg) if msg else str(errs)
            return str(errs)
        order = resp.get("order")
        if isinstance(order, dict):
            status = order.get("status", "").lower()
            if status == "rejected":
                return order.get("reason_description") or "rejected"
        return None

    async def submit_close(
        self,
        position: "OpenPosition",
        mark: "PositionMark",
        exit_trigger: str,
    ) -> OrderResult:
        """Build closing leg list with sides inverted, price at current mid +/-
        slippage, submit via the same multileg endpoint, poll for fill, update
        OrderLog with closing context. Bypasses fingerprint dedup (closing has
        a different fingerprint by construction).

        Records a close_attempt row at submission time. On poll timeout the
        attempt is left status='pending' so reconcile_pending_closes() can
        cancel it after the stale threshold elapses and let exit_manager
        retry next cycle with a fresh limit price."""
        now = datetime.now(timezone.utc)

        guard_error = self._check_production_guard()
        if guard_error is not None:
            logger.error("submit_close guard_blocked: %s", guard_error)
            return OrderResult(
                signal=None, status="guard_blocked", order_id=None,
                submitted_price=None, fill_price=None, error=guard_error,
            )

        # Don't pile up duplicate closes — if one is already in flight, defer.
        # The stale-close-manager will cancel it next cycle if it goes stale,
        # at which point exit_manager re-evaluates and resubmits with a fresh
        # limit.
        if self._log.has_pending_close(position.tradier_order_id):
            logger.info(
                "close already pending for opening %s (%s) — skipping submit",
                position.tradier_order_id, position.symbol,
            )
            return OrderResult(
                signal=None, status="pending_close_exists", order_id=None,
                submitted_price=None, fill_price=None,
                error=f"close already pending for opening {position.tradier_order_id}",
            )

        # Max retries: refuse and raise a CRITICAL alert (idempotent — the
        # alert table is keyed on opening_order_id, so re-attempts on the same
        # position don't spam the daily summary).
        failed_count = self._log.failed_close_attempt_count(position.tradier_order_id)
        if failed_count >= self._max_close_retries:
            inserted = self._log.record_stale_close_alert(
                opening_order_id=position.tradier_order_id,
                symbol=position.symbol,
                expiration=position.expiration,
                structure=position.structure,
                attempts=failed_count,
                last_exit_trigger=exit_trigger,
                detected_at=now,
            )
            if inserted:
                logger.critical(
                    "MAX CLOSE RETRIES EXCEEDED: opening %s (%s %s exp=%s) — "
                    "%d failed close attempts (cap %d). Manual intervention required.",
                    position.tradier_order_id, position.symbol,
                    position.structure, position.expiration.isoformat(),
                    failed_count, self._max_close_retries,
                )
            return OrderResult(
                signal=None, status="max_retries_exceeded", order_id=None,
                submitted_price=None, fill_price=None,
                error=(
                    f"max close retries ({self._max_close_retries}) exceeded "
                    f"for opening {position.tradier_order_id}"
                ),
            )

        # Invert sides: buy → sell_to_close, sell → buy_to_close
        side_close_map = {"buy": "sell_to_close", "sell": "buy_to_close"}
        api_legs: list[dict] = []
        for leg in position.legs:
            api_side = side_close_map.get(leg.side)
            if api_side is None:
                err = f"unrecognized leg side: {leg.side}"
                return OrderResult(
                    signal=None, status="error", order_id=None,
                    submitted_price=None, fill_price=None, error=err,
                )
            api_legs.append({
                "option_symbol": leg.contract_symbol,
                "side": api_side,
                "quantity": leg.quantity,
            })

        # Determine close net premium and order type. Priced at the mark's
        # theoretical mid; the price ladder concedes from there. close_cash_flow
        # is whole-position dollars (scaled by leg quantity), but the multileg
        # limit price is PER STRUCTURE UNIT — divide the lots back out.
        close_cash_per_contract = mark.close_cash_flow / (100.0 * position.lots)
        order_type = "credit" if close_cash_per_contract >= 0 else "debit"
        close_mid = round(abs(close_cash_per_contract), 2)
        price = self._ladder_price(close_mid, order_type, self._ladder_steps[0])
        if price <= 0:
            err = f"computed close price not positive: {price}"
            return OrderResult(
                signal=None, status="error", order_id=None,
                submitted_price=None, fill_price=None, error=err,
            )

        # Preview
        preview = await self._client.preview_order(
            account_id=self._settings.account_id,
            legs=api_legs,
            underlying_symbol=position.symbol,
            order_type=order_type,
            price=price,
        )
        preview_error = self._extract_error(preview)
        if preview_error is not None:
            logger.warning("close preview failed for order %s: %s",
                           position.tradier_order_id, preview_error)
            # Burns retry budget: repeated rejections (e.g. size mismatch
            # after an unresolved partial fill) must escalate to the
            # stale-close alert instead of retrying silently forever.
            self._log.record_failed_close_submission(
                opening_order_id=position.tradier_order_id,
                exit_trigger=exit_trigger, submitted_at=now,
                reason=f"preview: {preview_error}",
            )
            return OrderResult(
                signal=None, status="preview_failed", order_id=None,
                submitted_price=price, fill_price=None, error=preview_error,
            )

        # Place
        place_resp = await self._client.place_order(
            account_id=self._settings.account_id,
            legs=api_legs,
            underlying_symbol=position.symbol,
            order_type=order_type,
            price=price,
        )
        place_error = self._extract_error(place_resp)
        order_node = place_resp.get("order") if isinstance(place_resp, dict) else None
        if place_error is not None or order_node is None or "id" not in order_node:
            err = place_error or f"unexpected place response: {place_resp}"
            logger.error("close place failed for order %s: %s",
                         position.tradier_order_id, err)
            self._log.record_failed_close_submission(
                opening_order_id=position.tradier_order_id,
                exit_trigger=exit_trigger, submitted_at=now,
                reason=f"place: {err}",
            )
            return OrderResult(
                signal=None, status="error", order_id=None,
                submitted_price=price, fill_price=None, error=err,
            )
        closing_order_id = int(order_node["id"])
        placed_at = datetime.now(timezone.utc)
        self._log.record_close_attempt(
            opening_order_id=position.tradier_order_id,
            closing_order_id=closing_order_id,
            submitted_at=placed_at,
            exit_trigger=exit_trigger,
            order_type=order_type,
            submitted_price=price,
            arrival_mid=close_mid,
        )
        logger.info("submitted CLOSE %d for opening %d (%s) trigger=%s mid=%.2f type=%s (ladder %s)",
                    closing_order_id, position.tradier_order_id,
                    position.symbol, exit_trigger, close_mid, order_type,
                    self._ladder_steps)

        # Walk the price ladder toward the far side until filled. Unlike
        # entries, an unfilled close is NOT canceled here — it stays working
        # for the stale-close reconciler, which cancels past the threshold and
        # lets exit_manager retry next cycle at a fresh mark.
        terminal_status, fill_price, _ = await self._run_price_ladder(
            closing_order_id, close_mid, order_type,
        )
        if terminal_status is None:
            # Left pending intentionally — reconcile_pending_closes will cancel
            # past the stale threshold and exit_manager will retry next cycle.
            logger.info(
                "close %d poll timed out — left as pending for stale-close "
                "reconciler (threshold=%dmin)",
                closing_order_id, self._stale_threshold_minutes,
            )
            return OrderResult(
                signal=None, status="pending", order_id=closing_order_id,
                submitted_price=price, fill_price=None,
                error="close did not reach terminal state within submit poll",
            )

        terminal_at = datetime.now(timezone.utc)

        if terminal_status == "partially_filled":
            # Multi-lot close executed only some units: cancel the residual,
            # rebook the opening at the remaining size, burn a retry. Only a
            # cancel-raced COMPLETE fill falls through to the filled path.
            resolution, resolved_fill = await self._handle_partial_close(
                closing_order_id=closing_order_id,
                opening_order_id=position.tradier_order_id,
                exit_trigger=exit_trigger,
                now=terminal_at,
            )
            if resolution == "filled":
                terminal_status = "filled"
                fill_price = resolved_fill if resolved_fill is not None else fill_price
            elif resolution == "pending":
                # Cancel unconfirmed — the attempt stays pending and the
                # stale-close reconciler owns it from here (same contract as
                # the poll-timeout path above).
                return OrderResult(
                    signal=None, status="pending", order_id=closing_order_id,
                    submitted_price=price, fill_price=None,
                    error="close partially filled; cancel unconfirmed — left pending",
                )
            else:
                return OrderResult(
                    signal=None, status="partially_filled",
                    order_id=closing_order_id, submitted_price=price,
                    fill_price=resolved_fill,
                    error=f"close partially filled (resolution: {resolution})",
                )

        if terminal_status == "filled":
            # Open + close sides both executed: fees on every leg, twice.
            fees = self._fee_per_contract * 2 * sum(
                leg.quantity for leg in position.legs
            )
            realized_pnl = self._compute_close_realized_pnl(
                is_long=position.is_long,
                entry_premium=position.entry_premium,
                order_type=order_type,
                fill_price=fill_price,
                fallback_pnl=mark.pnl_dollars,
                fees=fees,
                lots=position.lots,
            )
            self._log.update_close_attempt(
                closing_order_id=closing_order_id, status="filled",
                terminal_at=terminal_at, fill_price=fill_price,
            )
            self._log.record_close(
                opening_order_id=position.tradier_order_id,
                closing_order_id=closing_order_id,
                closed_at=terminal_at,
                exit_trigger=exit_trigger,
                realized_pnl=realized_pnl,
            )
            self._log_fill_tca(
                "close", closing_order_id, order_type, close_mid, fill_price,
            )
            return OrderResult(
                signal=None, status="filled", order_id=closing_order_id,
                submitted_price=price, fill_price=fill_price, error=None,
            )

        # Terminal failed (rejected / canceled / expired) — partially_filled
        # was intercepted and resolved above.
        self._log.update_close_attempt(
            closing_order_id=closing_order_id, status=terminal_status,
            terminal_at=terminal_at, fill_price=fill_price,
        )
        return OrderResult(
            signal=None, status=terminal_status, order_id=closing_order_id,
            submitted_price=price, fill_price=fill_price,
            error=f"close terminal status: {terminal_status}",
        )

    async def reconcile_pending_closes(self, now: datetime) -> dict[str, int]:
        """Walk every pending close attempt: cancel ones older than the stale
        threshold, reconcile any that filled between cycles. Should run on
        every MainLoop cycle before signal generation.

        Returns counts: {"canceled", "filled", "failed_terminal"}.

        - Pending older than threshold + still non-terminal at Tradier:
          DELETE the order, mark attempt as stale_canceled. exit_manager will
          re-evaluate next cycle and resubmit with a fresh limit.
        - Pending but Tradier reports filled: update attempt + record close on
          opening order with realized P&L from the actual fill.
        - Pending but Tradier reports rejected/canceled/expired: update attempt
          to that status (counts toward the retry budget)."""
        pending = self._log.pending_close_attempts()
        if not pending:
            return {"canceled": 0, "filled": 0, "failed_terminal": 0}

        threshold = timedelta(minutes=self._stale_threshold_minutes)
        canceled = filled = failed_terminal = 0

        for row in pending:
            closing_order_id = row["closing_order_id"]
            opening_order_id = row["opening_order_id"]
            try:
                submitted_at = datetime.fromisoformat(row["submitted_at"])
            except (TypeError, ValueError):
                logger.warning(
                    "bad submitted_at on close attempt %s — skipping",
                    closing_order_id,
                )
                continue
            age = now - submitted_at

            try:
                resp = await self._client.get_order_status(
                    self._settings.account_id, closing_order_id,
                )
            except Exception as e:
                logger.warning(
                    "get_order_status(%s) failed: %s — leaving pending",
                    closing_order_id, e,
                )
                continue

            order_node = resp.get("order") if isinstance(resp, dict) else None
            if not isinstance(order_node, dict):
                logger.warning(
                    "malformed get_order_status response for close %s — leaving pending",
                    closing_order_id,
                )
                continue

            status = (order_node.get("status") or "").lower()

            if status == "filled":
                fill_price = self._extract_fill_price(order_node)
                self._reconcile_between_cycle_fill(row, fill_price, now)
                filled += 1
                continue

            if status == "partially_filled":
                # Not a complete close: cancel the residual and rebook the
                # opening at its remaining size instead of booking a full
                # close at the full lot count (which would both overstate
                # realized P&L and orphan the still-open units at Tradier).
                resolution, fp = await self._handle_partial_close(
                    closing_order_id=closing_order_id,
                    opening_order_id=opening_order_id,
                    exit_trigger=row.get("exit_trigger") or "unknown",
                    now=now,
                )
                if resolution == "filled":
                    self._reconcile_between_cycle_fill(row, fp, now)
                    filled += 1
                elif resolution == "pending":
                    pass  # cancel unconfirmed — attempt left pending, retry next pass
                else:
                    failed_terminal += 1
                continue

            if status in {"rejected", "canceled", "expired"}:
                # A canceled/expired close can still carry partial executions
                # (some units filled before it died) — those must go through
                # the partial handler, not be booked as a clean failure.
                dead_units, _even = self._node_units(order_node)
                if dead_units is not None and dead_units > 0:
                    resolution, fp = await self._handle_partial_close(
                        closing_order_id=closing_order_id,
                        opening_order_id=opening_order_id,
                        exit_trigger=row.get("exit_trigger") or "unknown",
                        now=now,
                    )
                    if resolution == "filled":
                        self._reconcile_between_cycle_fill(row, fp, now)
                        filled += 1
                    elif resolution == "pending":
                        pass
                    else:
                        failed_terminal += 1
                    continue
                self._log.update_close_attempt(
                    closing_order_id=closing_order_id, status=status,
                    terminal_at=now, fill_price=None,
                )
                logger.info(
                    "close %s for opening %s reached terminal status %s "
                    "(was pending) — will retry next cycle",
                    closing_order_id, opening_order_id, status,
                )
                failed_terminal += 1
                continue

            # Still non-terminal at Tradier — check age
            if age <= threshold:
                continue

            try:
                cancel_resp = await self._client.cancel_order(
                    self._settings.account_id, closing_order_id,
                )
            except Exception as e:
                # %r not %s: transient aiohttp errors (e.g. ServerDisconnectedError)
                # have an empty str(), which would log a blank reason. repr keeps
                # the exception type so a real failure is still diagnosable.
                logger.warning(
                    "cancel_order(%s) failed: %r — will retry next cycle",
                    closing_order_id, e,
                )
                continue

            cancel_err = self._extract_error(cancel_resp)
            if cancel_err is not None:
                # Race: order may have filled just before our cancel.
                logger.warning(
                    "cancel_order(%s) returned error %r — re-querying status",
                    closing_order_id, cancel_err,
                )
                try:
                    resp2 = await self._client.get_order_status(
                        self._settings.account_id, closing_order_id,
                    )
                    node2 = resp2.get("order") if isinstance(resp2, dict) else None
                    if isinstance(node2, dict):
                        s2 = (node2.get("status") or "").lower()
                        if s2 == "filled":
                            fp = self._extract_fill_price(node2)
                            self._reconcile_between_cycle_fill(row, fp, now)
                            filled += 1
                            continue
                        if s2 == "partially_filled":
                            resolution, fp = await self._handle_partial_close(
                                closing_order_id=closing_order_id,
                                opening_order_id=opening_order_id,
                                exit_trigger=row.get("exit_trigger") or "unknown",
                                now=now,
                            )
                            if resolution == "filled":
                                self._reconcile_between_cycle_fill(row, fp, now)
                                filled += 1
                            elif resolution == "pending":
                                pass  # attempt left pending, retry next pass
                            else:
                                failed_terminal += 1
                            continue
                        if s2 in {"rejected", "canceled", "expired"}:
                            self._log.update_close_attempt(
                                closing_order_id=closing_order_id, status=s2,
                                terminal_at=now, fill_price=None,
                            )
                            failed_terminal += 1
                            continue
                except Exception:
                    pass
                continue

            self._log.update_close_attempt(
                closing_order_id=closing_order_id, status=STALE_CANCELED_STATUS,
                terminal_at=now, fill_price=None,
            )
            logger.info(
                "canceled stale close %s for opening %s (age=%.0fs threshold=%dmin) "
                "— exit_manager will retry next cycle",
                closing_order_id, opening_order_id, age.total_seconds(),
                self._stale_threshold_minutes,
            )
            canceled += 1

        if canceled or filled or failed_terminal:
            logger.info(
                "stale-close reconcile: canceled=%d filled=%d failed_terminal=%d",
                canceled, filled, failed_terminal,
            )
        return {"canceled": canceled, "filled": filled, "failed_terminal": failed_terminal}

    async def _handle_partial_close(
        self,
        closing_order_id: int,
        opening_order_id: int,
        exit_trigger: str,
        now: datetime,
    ) -> tuple[str, float | None]:
        """A multi-lot close executed only some units. Cancel the residual,
        shrink the OPENING order's legs to the units still held, and book the
        attempt as failed (partially_filled burns retry budget) so the exit
        manager re-closes the remainder at its true size next cycle. The
        closed chunk's cash lands in equity rather than realized P&L — the
        equity-derived unrealized split absorbs it, and the persisted
        stale-close alert puts the event in the daily summary.

        Returns (resolution, avg_fill): 'filled' when everything executed
        (caller books the normal full close), 'canceled' when nothing did,
        'partial' when the remainder was rebooked, 'pending' when the close
        order may STILL BE WORKING (attempt left pending so the stale-close
        reconciler re-queries it next cycle), or 'unresolved' when the order
        is dead but Tradier's numbers were unusable."""
        try:
            cancel_resp = await self._client.cancel_order(
                self._settings.account_id, closing_order_id,
            )
            cancel_err = self._extract_error(cancel_resp)
        except Exception as e:
            cancel_err = repr(e)
        if cancel_err is not None:
            logger.warning(
                "cancel of partially-filled close %d returned %r — re-querying",
                closing_order_id, cancel_err,
            )
        status, units, avg_fill, legs_even = await self._executed_units(closing_order_id)

        if status not in DEAD_ORDER_STATES:
            # Cancel unconfirmed — the close may keep filling. Do NOT touch
            # the attempt (it stays/returns to 'pending'), so
            # reconcile_pending_closes re-queries it next cycle instead of
            # this event being handled twice or the order being forgotten.
            logger.warning(
                "partial close %d for opening %d: status=%r after cancel — "
                "order may still be working; leaving attempt pending for the "
                "next reconcile pass", closing_order_id, opening_order_id, status,
            )
            return "pending", avg_fill

        opening = self._log.get_submitted_order(opening_order_id)
        held_lots: int | None = None
        symbol, structure, expiration = "?", "?", None
        if opening is not None:
            symbol, structure = opening["symbol"], opening["structure"]
            try:
                expiration = date.fromisoformat(opening["expiration"])
                legs = json.loads(opening["legs_json"])
                held_lots = max(1, min(int(l["quantity"]) for l in legs))
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                held_lots = None

        def _alert() -> None:
            if opening is None or expiration is None:
                return
            self._log.record_stale_close_alert(
                opening_order_id=opening_order_id, symbol=symbol,
                expiration=expiration, structure=structure,
                attempts=self._log.failed_close_attempt_count(opening_order_id),
                last_exit_trigger=f"partial_close:{exit_trigger}",
                detected_at=now,
            )

        def _unresolved(reason: str) -> tuple[str, float | None]:
            self._log.update_close_attempt(
                closing_order_id=closing_order_id, status="partially_filled",
                terminal_at=now, fill_price=avg_fill,
            )
            _alert()
            logger.critical(
                "PARTIAL CLOSE UNRESOLVED: close %d for opening %d (%s) — %s; "
                "position left at recorded size, verify held units at the "
                "broker (RUNBOOK section 2)",
                closing_order_id, opening_order_id, symbol, reason,
            )
            return "unresolved", avg_fill

        if units is None:
            return _unresolved(f"order dead ({status}) but executed quantity unreadable")
        if held_lots is None:
            # Never guess the held size: a bad guess of 1 would let any
            # partial pass the units >= held test and book a full close.
            return _unresolved("opening order's legs_json unreadable")
        if units <= 0:
            if status == "filled":
                return _unresolved("status=filled but 0 executed units parsed")
            self._log.update_close_attempt(
                closing_order_id=closing_order_id, status="canceled",
                terminal_at=now, fill_price=None,
            )
            logger.info(
                "partial close %d resolved as canceled (order %s, no complete "
                "units executed)", closing_order_id, status,
            )
            return "canceled", None
        if units >= held_lots:
            return "filled", avg_fill

        # Attempt status FIRST, leg shrink second: if the process dies
        # between the two writes, the attempt is already failed so the next
        # cycle re-closes at the (unshrunk) full size and gets rejected —
        # loud but not corrupting. The reverse order would leave the attempt
        # pending against already-shrunk legs and shrink them AGAIN on the
        # next reconcile pass.
        remaining = held_lots - units
        self._log.update_close_attempt(
            closing_order_id=closing_order_id, status="partially_filled",
            terminal_at=now, fill_price=avg_fill,
        )
        rewrote = self._log.update_leg_quantities(opening_order_id, remaining)
        _alert()
        if not legs_even:
            logger.critical(
                "PARTIAL CLOSE UNEVEN: close %d (%s) legs executed unevenly — "
                "%d complete unit(s) booked closed; per-leg residue at the "
                "broker will reject the remainder close. Check positions "
                "manually.", closing_order_id, symbol, units,
            )
        logger.critical(
            "PARTIAL CLOSE: close %d for opening %d (%s %s) closed %d of %d "
            "lot(s); opening rebooked at %d lot(s)%s — closed chunk's P&L "
            "shows in equity/unrealized, exit manager re-closes the remainder "
            "next cycle",
            closing_order_id, opening_order_id, symbol, structure,
            units, held_lots, remaining,
            "" if rewrote else " (LEG REWRITE FAILED — size now wrong, fix manually)",
        )
        return "partial", avg_fill

    def _reconcile_between_cycle_fill(
        self,
        attempt_row: dict,
        fill_price: float | None,
        now: datetime,
    ) -> None:
        """A close that was pending in our log filled at Tradier between
        cycles. Update the close attempt + record the opening order as
        closed, computing realized P&L from the actual fill."""
        closing_order_id = attempt_row["closing_order_id"]
        opening_order_id = attempt_row["opening_order_id"]
        order_type = attempt_row["order_type"]
        exit_trigger = attempt_row["exit_trigger"]

        opening = self._log.get_submitted_order(opening_order_id)
        if opening is None:
            logger.error(
                "reconcile: could not find opening order %s for close %s",
                opening_order_id, closing_order_id,
            )
            return

        is_long = opening["direction"] == "BUY"
        raw_fill = (
            opening["fill_price"] if opening["fill_price"] is not None
            else opening["submitted_price"]
        )
        entry_premium = abs(float(raw_fill))

        # Guarded like every other reader of legs_json (reconciler, tracker):
        # a corrupt row must not block booking the close it describes.
        try:
            opening_legs = json.loads(opening["legs_json"])
            contracts = sum(int(leg["quantity"]) for leg in opening_legs)
            lots = max(1, min(int(leg["quantity"]) for leg in opening_legs))
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            logger.warning(
                "reconcile: bad legs_json on opening order %s — booking close "
                "without fee netting", opening_order_id,
            )
            contracts = 0
            lots = 1
        fees = self._fee_per_contract * 2 * contracts
        realized_pnl = self._compute_close_realized_pnl(
            is_long=is_long, entry_premium=entry_premium,
            order_type=order_type, fill_price=fill_price,
            fallback_pnl=0.0, fees=fees, lots=lots,
        )

        self._log.update_close_attempt(
            closing_order_id=closing_order_id, status="filled",
            terminal_at=now, fill_price=fill_price,
        )
        self._log.record_close(
            opening_order_id=opening_order_id,
            closing_order_id=closing_order_id,
            closed_at=now,
            exit_trigger=exit_trigger,
            realized_pnl=realized_pnl,
        )
        logger.info(
            "reconciled between-cycle close fill: closing=%s opening=%s "
            "realized=$%+.2f trigger=%s fill_price=%s",
            closing_order_id, opening_order_id, realized_pnl, exit_trigger,
            f"{fill_price:.4f}" if fill_price is not None else "n/a",
        )

    @staticmethod
    def _compute_close_realized_pnl(
        is_long: bool,
        entry_premium: float,
        order_type: str,
        fill_price: float | None,
        fallback_pnl: float,
        fees: float = 0.0,
        lots: int = 1,
    ) -> float:
        """Realized P&L using Tradier's signed avg_fill_price + order_type.
        Tradier returns credit fills as NEGATIVE; abs() + order_type carries
        the canonical sign so credits add and debits subtract from realized.
        Entry premium and fill price are per structure unit; `lots` scales
        both to whole-position dollars. If fill_price is missing (rare),
        falls back to the mark estimate (already whole-position dollars).
        `fees` (estimated round-trip contract fees) are netted out; the
        equity-derived unrealized split in portfolio_state absorbs the
        difference, so the bookkeeping identity still holds."""
        if fill_price is None:
            return fallback_pnl - fees
        abs_fill = abs(fill_price)
        close_cash_realized = (abs_fill if order_type == "credit" else -abs_fill) * 100 * lots
        entry_sign = -1 if is_long else 1
        entry_cash = entry_sign * entry_premium * 100 * lots
        return entry_cash + close_cash_realized - fees

    @staticmethod
    def _extract_fill_price(order_node: dict) -> float | None:
        raw = order_node.get("avg_fill_price") or order_node.get("price")
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None
