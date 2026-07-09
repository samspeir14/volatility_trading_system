from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
    estimated_premium: float        # absolute dollar premium per contract


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

    if signal.direction == "BUY":
        if len(leg_contracts) != 2:
            raise ValueError(f"BUY straddle expects 2 legs, got {len(leg_contracts)}")
        call, put = (leg_contracts[0], leg_contracts[1]) if leg_contracts[0].option_type == "call" else (leg_contracts[1], leg_contracts[0])
        price = compute_straddle_debit(call, put, slippage=slippage)
        structure = "straddle"
        order_type = "debit"
        estimated_premium = price * 100  # one contract = 100 shares
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
        # Premium-cap proxy: use absolute credit × 100 for sanity
        estimated_premium = price * 100

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
    ):
        self._client = client
        self._log = order_log
        self._settings = settings
        self._max_qty = max_quantity_per_leg
        self._max_premium = max_premium_per_trade
        self._fee_per_contract = settings.per_contract_fee
        self._poll_interval = poll_interval_seconds
        self._poll_timeout = poll_timeout_seconds
        self._slippage = slippage_buffer
        self._stale_threshold_minutes = stale_order_threshold_minutes
        self._max_close_retries = max_close_retries

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

        # 2. Build request
        try:
            request = signal_to_request(signal, snapshot, slippage=self._slippage, max_qty=self._max_qty)
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
        self._log.record_submission(
            signal=signal, fingerprint=fingerprint, structure=request.structure,
            submitted_price=request.price, order_id=order_id, submitted_at=now,
        )
        logger.info("submitted order %d (%s %s %s) at %.2f",
                    order_id, signal.symbol, signal.direction, request.structure, request.price)

        # 7. Poll for terminal state
        terminal_status, fill_price = await self._poll_until_terminal(order_id)
        if terminal_status is None:
            self._log.update_terminal_state(
                order_id=order_id, status="timeout", fill_price=None,
                filled_at=None, error="poll timeout",
            )
            return OrderResult(
                signal=signal, status="timeout", order_id=order_id,
                submitted_price=request.price, fill_price=None,
                error="order did not reach terminal state within timeout",
            )

        filled_at = datetime.now(timezone.utc) if terminal_status == "filled" else None
        self._log.update_terminal_state(
            order_id=order_id, status=terminal_status, fill_price=fill_price,
            filled_at=filled_at,
        )
        return OrderResult(
            signal=signal, status=terminal_status, order_id=order_id,
            submitted_price=request.price, fill_price=fill_price, error=None,
        )

    async def _poll_until_terminal(self, order_id: int) -> tuple[str | None, float | None]:
        deadline = time.monotonic() + self._poll_timeout
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

        # Determine close net premium and order type
        close_cash_per_contract = mark.close_cash_flow / 100.0
        if close_cash_per_contract >= 0:
            order_type = "credit"
            price = round(close_cash_per_contract * (1 - self._slippage), 2)
        else:
            order_type = "debit"
            price = round(abs(close_cash_per_contract) * (1 + self._slippage), 2)
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
        )
        logger.info("submitted CLOSE %d for opening %d (%s) trigger=%s price=%.2f type=%s",
                    closing_order_id, position.tradier_order_id,
                    position.symbol, exit_trigger, price, order_type)

        # Poll for fill
        terminal_status, fill_price = await self._poll_until_terminal(closing_order_id)
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
            return OrderResult(
                signal=None, status="filled", order_id=closing_order_id,
                submitted_price=price, fill_price=fill_price, error=None,
            )

        # Terminal failed (rejected / canceled / expired / partially_filled)
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

            if status in {"filled", "partially_filled"}:
                fill_price = self._extract_fill_price(order_node)
                self._reconcile_between_cycle_fill(row, fill_price, now)
                filled += 1
                continue

            if status in {"rejected", "canceled", "expired"}:
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
                        if s2 in {"filled", "partially_filled"}:
                            fp = self._extract_fill_price(node2)
                            self._reconcile_between_cycle_fill(row, fp, now)
                            filled += 1
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
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            logger.warning(
                "reconcile: bad legs_json on opening order %s — booking close "
                "without fee netting", opening_order_id,
            )
            contracts = 0
        fees = self._fee_per_contract * 2 * contracts
        realized_pnl = self._compute_close_realized_pnl(
            is_long=is_long, entry_premium=entry_premium,
            order_type=order_type, fill_price=fill_price,
            fallback_pnl=0.0, fees=fees,
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
    ) -> float:
        """Realized P&L using Tradier's signed avg_fill_price + order_type.
        Tradier returns credit fills as NEGATIVE; abs() + order_type carries
        the canonical sign so credits add and debits subtract from realized.
        If fill_price is missing (rare), falls back to the mark estimate.
        `fees` (estimated round-trip contract fees) are netted out; the
        equity-derived unrealized split in portfolio_state absorbs the
        difference, so the bookkeeping identity still holds."""
        if fill_price is None:
            return fallback_pnl - fees
        abs_fill = abs(fill_price)
        close_cash_realized = abs_fill * 100 if order_type == "credit" else -abs_fill * 100
        entry_sign = -1 if is_long else 1
        entry_cash = entry_sign * entry_premium * 100
        return entry_cash + close_cash_realized - fees

    @staticmethod
    def _extract_fill_price(order_node: dict) -> float | None:
        raw = order_node.get("avg_fill_price") or order_node.get("price")
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None
