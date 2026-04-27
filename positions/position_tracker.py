from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from config import Settings
from data.async_client import AsyncTradierClient, OptionContract
from data.market_data import ScanResult
from execution.order_log import OrderLog
from signals.signal_generator import TradeLeg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenPosition:
    tradier_order_id: int
    symbol: str
    expiration: date
    direction: str              # "BUY" | "SELL"
    structure: str              # "straddle" | "iron_condor"
    legs: list[TradeLeg]
    entry_premium: float        # always positive: debit (BUY) or credit (SELL)
    entry_atm_iv: float
    entry_predicted_iv: float
    entry_divergence: float
    entry_horizon_lower: int
    entry_horizon_upper: int
    entry_weight_lower: float
    submitted_at: datetime

    @property
    def is_long(self) -> bool:
        return self.direction == "BUY"

    @property
    def max_loss_dollars(self) -> float:
        """For iron condors: wing distance × 100 − entry credit. For long
        straddles: entry debit × 100 (premium paid is the max loss)."""
        if self.is_long:
            return self.entry_premium * 100
        # iron condor: assume symmetric wings → wing_distance is max(call_wing, put_wing)
        if self.structure == "iron_condor":
            calls = sorted([l for l in self.legs if l.option_type == "call"], key=lambda l: l.strike)
            puts = sorted([l for l in self.legs if l.option_type == "put"], key=lambda l: l.strike)
            if len(calls) == 2 and len(puts) == 2:
                wing_distance = max(calls[1].strike - calls[0].strike, puts[1].strike - puts[0].strike)
                return (wing_distance * 100) - (self.entry_premium * 100)
        return float("nan")


@dataclass(frozen=True)
class PositionMark:
    """Mark-to-market snapshot for an open position."""
    position: OpenPosition
    current_legs: list[OptionContract]    # same order as position.legs
    close_cash_flow: float                # signed: +receive, -pay (in dollars total)
    cost_to_close: float                  # absolute dollars to flatten (positive)
    pnl_dollars: float                    # current P&L in dollars
    pnl_pct_of_entry_premium: float       # pnl / (entry_premium × 100)
    pnl_pct_of_max: float                 # for iron condors only; NaN for straddles
    delta: float
    gamma: float
    theta: float
    vega: float
    dte: int


def _leg_sign(side: str) -> int:
    """+1 for long (we'd sell to close, receive cash); -1 for short (we'd buy to close, pay cash)."""
    return 1 if side == "buy" else -1


class PositionTracker:
    def __init__(
        self,
        client: AsyncTradierClient,
        order_log: OrderLog,
        settings: Settings,
    ):
        self._client = client
        self._log = order_log
        self._settings = settings

    async def list_open_positions(self) -> list[OpenPosition]:
        """Read unclosed positions from our order log. Tradier's positions
        endpoint is consulted if needed but the order log is authoritative."""
        rows = self._log.open_unclosed_positions()
        positions: list[OpenPosition] = []
        for row in rows:
            try:
                legs_data = json.loads(row["legs_json"])
                legs = [TradeLeg(**leg) for leg in legs_data]
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("could not parse legs_json for order %s: %s",
                               row["tradier_order_id"], e)
                continue
            positions.append(OpenPosition(
                tradier_order_id=row["tradier_order_id"],
                symbol=row["symbol"],
                expiration=date.fromisoformat(row["expiration"]),
                direction=row["direction"],
                structure=row["structure"],
                legs=legs,
                entry_premium=float(row["fill_price"] if row["fill_price"] is not None else row["submitted_price"]),
                entry_atm_iv=row["atm_iv_at_signal"],
                entry_predicted_iv=row["predicted_iv_at_signal"],
                entry_divergence=row["divergence_at_signal"],
                entry_horizon_lower=row["horizon_lower"],
                entry_horizon_upper=row["horizon_upper"],
                entry_weight_lower=row["weight_lower"],
                submitted_at=datetime.fromisoformat(row["submitted_at"]),
            ))
        return positions

    def mark_to_market(
        self,
        positions: list[OpenPosition],
        scan: ScanResult,
    ) -> list[PositionMark]:
        """Look up each leg's current quote in the scan and compute P&L + greeks."""
        marks: list[PositionMark] = []
        today = scan.fetched_at.date()
        for pos in positions:
            current_legs = self._lookup_current_legs(pos, scan)
            if current_legs is None:
                logger.warning("could not mark position %s — not all legs found in scan",
                               pos.tradier_order_id)
                continue

            close_cash_flow = 0.0
            delta_sum = gamma_sum = theta_sum = vega_sum = 0.0
            for opened_leg, current in zip(pos.legs, current_legs):
                sign = _leg_sign(opened_leg.side)
                qty = opened_leg.quantity
                mid = (current.bid + current.ask) / 2.0
                close_cash_flow += sign * mid * qty * 100
                delta_sum += sign * current.delta * qty * 100
                gamma_sum += sign * current.gamma * qty * 100
                theta_sum += sign * current.theta * qty * 100
                vega_sum += sign * current.vega * qty * 100

            # entry cash flow: debit (BUY) → -premium, credit (SELL) → +premium
            entry_sign = -1 if pos.is_long else 1
            entry_cash_flow = entry_sign * pos.entry_premium * 100
            pnl = entry_cash_flow + close_cash_flow

            cost_to_close = abs(close_cash_flow) if close_cash_flow < 0 else -close_cash_flow

            entry_dollars = pos.entry_premium * 100
            pnl_pct_premium = pnl / entry_dollars if entry_dollars > 0 else float("nan")
            max_loss = pos.max_loss_dollars
            pnl_pct_max = pnl / max_loss if max_loss > 0 else float("nan")

            dte = (pos.expiration - today).days

            marks.append(PositionMark(
                position=pos,
                current_legs=current_legs,
                close_cash_flow=close_cash_flow,
                cost_to_close=cost_to_close,
                pnl_dollars=pnl,
                pnl_pct_of_entry_premium=pnl_pct_premium,
                pnl_pct_of_max=pnl_pct_max,
                delta=delta_sum,
                gamma=gamma_sum,
                theta=theta_sum,
                vega=vega_sum,
                dte=dte,
            ))
        return marks

    def _lookup_current_legs(
        self,
        pos: OpenPosition,
        scan: ScanResult,
    ) -> list[OptionContract] | None:
        snap = scan.snapshots.get(pos.symbol)
        if snap is None:
            return None
        by_symbol = {c.symbol: c for c in snap.contracts}
        out: list[OptionContract] = []
        for leg in pos.legs:
            current = by_symbol.get(leg.contract_symbol)
            if current is None:
                return None
            out.append(current)
        return out

    @staticmethod
    def portfolio_greeks(marks: list[PositionMark]) -> dict[str, float]:
        return {
            "delta": sum(m.delta for m in marks),
            "gamma": sum(m.gamma for m in marks),
            "theta": sum(m.theta for m in marks),
            "vega": sum(m.vega for m in marks),
        }
