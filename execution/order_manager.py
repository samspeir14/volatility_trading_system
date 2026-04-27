from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from config import Settings
from data.async_client import AsyncTradierClient, OptionContract
from data.market_data import TickerSnapshot
from execution.order_log import OrderLog
from signals.signal_generator import TradeLeg, TradeSignal

logger = logging.getLogger(__name__)


MAX_QUANTITY_PER_LEG_DEFAULT = 10
MAX_PREMIUM_PER_TRADE_DEFAULT = 5000.0
TERMINAL_STATES = {"filled", "partially_filled", "rejected", "canceled", "expired"}
TERMINAL_FAILED = {"rejected", "canceled", "expired"}


@dataclass(frozen=True)
class OrderResult:
    signal: TradeSignal
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
    ):
        self._client = client
        self._log = order_log
        self._settings = settings
        self._max_qty = max_quantity_per_leg
        self._max_premium = max_premium_per_trade
        self._poll_interval = poll_interval_seconds
        self._poll_timeout = poll_timeout_seconds
        self._slippage = slippage_buffer

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
