from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from config import Ticker
from data.market_data import ScanResult
from execution.order_manager import (
    MAX_QUANTITY_PER_LEG_DEFAULT,
    compute_iron_condor_credit,
    compute_straddle_debit,
)
from risk.kill_switch import DailyKillSwitch
from risk.portfolio_state import PortfolioSnapshot
from signals.signal_generator import TradeSignal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskDecision:
    signal: TradeSignal
    approved: bool
    quantity: int
    reasons: list[str]
    snapshot: PortfolioSnapshot
    projected_max_loss: float       # dollars at proposed quantity
    sized_quantity: int             # quantity per the per-trade budget alone (before other gates)


class RiskManager:
    def __init__(
        self,
        watchlist: list[Ticker],
        max_per_trade_loss_pct: float = 0.01,
        max_per_ticker_exposure_pct: float = 0.05,
        max_per_sector_positions: int = 3,
        max_portfolio_risk_pct: float = 0.12,
        max_portfolio_delta_pct: float = 0.05,
        max_portfolio_gamma_pct: float = 0.01,
        max_portfolio_vega_pct: float = 0.05,
        daily_loss_kill_switch_pct: float = -0.03,
        min_buying_power_buffer_pct: float = 0.05,
        kill_switch: DailyKillSwitch | None = None,
        max_quantity_per_leg: int = MAX_QUANTITY_PER_LEG_DEFAULT,
    ):
        self._sector_map = {t.symbol: t.sector for t in watchlist}
        self._max_trade_pct = max_per_trade_loss_pct
        self._max_ticker_pct = max_per_ticker_exposure_pct
        self._max_sector_pos = max_per_sector_positions
        # Cap on the SUM of open max-losses (wing risk). In a correlated crash
        # every short-vol position can max out simultaneously — the per-trade
        # and per-sector gates don't bound that sum, this does. It is the
        # portfolio's worst-case drawdown by construction.
        self._max_portfolio_risk_pct = max_portfolio_risk_pct
        self._delta_pct = max_portfolio_delta_pct
        self._gamma_pct = max_portfolio_gamma_pct
        self._vega_pct = max_portfolio_vega_pct
        self._kill_pct = daily_loss_kill_switch_pct
        self._bp_buffer_pct = min_buying_power_buffer_pct
        self._kill_switch = kill_switch
        self._max_qty = max_quantity_per_leg

    def gate(
        self,
        signals: list[TradeSignal],
        scan: ScanResult,
        snapshot: PortfolioSnapshot,
    ) -> list[RiskDecision]:
        # All signals in a cycle are gated against the same snapshot, so
        # earlier approvals must be carried forward here — otherwise a batch
        # of 15 approved signals each sees an empty book and the portfolio,
        # ticker, and sector caps are only enforced against *yesterday's*
        # positions. (Greeks are not accumulated intra-cycle; the loss-based
        # caps bind far earlier for defined-risk structures.)
        committed_risk = 0.0
        committed_ticker: dict[str, float] = {}
        committed_sector: dict[str, int] = {}
        decisions: list[RiskDecision] = []
        for s in signals:
            d = self._evaluate(
                s, scan, snapshot, committed_risk, committed_ticker, committed_sector,
            )
            if d.approved:
                committed_risk += d.projected_max_loss
                committed_ticker[s.symbol] = (
                    committed_ticker.get(s.symbol, 0.0) + d.projected_max_loss
                )
                sector = self._sector_map.get(s.symbol, "unknown")
                committed_sector[sector] = committed_sector.get(sector, 0) + 1
            decisions.append(d)
        return decisions

    def _evaluate(
        self,
        signal: TradeSignal,
        scan: ScanResult,
        snapshot: PortfolioSnapshot,
        committed_risk: float = 0.0,
        committed_ticker: dict[str, float] | None = None,
        committed_sector: dict[str, int] | None = None,
    ) -> RiskDecision:
        committed_ticker = committed_ticker or {}
        committed_sector = committed_sector or {}
        reasons: list[str] = []

        # 1. Kill switch (short-circuit — no other gate matters)
        if self._kill_switch is not None and self._kill_switch.is_active(
            snapshot.fetched_at.date()
        ):
            return RiskDecision(
                signal=signal, approved=False, quantity=0,
                reasons=["kill switch active"],
                snapshot=snapshot, projected_max_loss=0.0, sized_quantity=0,
            )

        # 2. Per-trade max loss → sizing
        max_loss_per_contract = self._max_loss_per_contract(signal, scan)
        if not math.isfinite(max_loss_per_contract) or max_loss_per_contract <= 0:
            return RiskDecision(
                signal=signal, approved=False, quantity=0,
                reasons=[f"could not compute max loss per contract: {max_loss_per_contract}"],
                snapshot=snapshot, projected_max_loss=0.0, sized_quantity=0,
            )

        budget_dollars = self._max_trade_pct * snapshot.equity
        sized_qty = min(
            int(budget_dollars // max_loss_per_contract),
            self._max_qty,
        )
        if sized_qty <= 0:
            reasons.append(
                f"max_loss_per_contract ${max_loss_per_contract:.2f} > "
                f"per-trade budget ${budget_dollars:.2f} ({self._max_trade_pct:.1%} equity)"
            )

        projected_loss = max_loss_per_contract * max(sized_qty, 0)

        # 3. Per-ticker exposure (open positions + earlier approvals this cycle)
        current_ticker_exposure = (
            snapshot.exposure_by_symbol.get(signal.symbol, 0.0)
            + committed_ticker.get(signal.symbol, 0.0)
        )
        new_ticker_total = current_ticker_exposure + projected_loss
        ticker_cap = self._max_ticker_pct * snapshot.equity
        if new_ticker_total > ticker_cap:
            reasons.append(
                f"{signal.symbol} exposure ${new_ticker_total:.2f} would exceed "
                f"cap ${ticker_cap:.2f} ({self._max_ticker_pct:.1%} equity)"
            )

        # 3b. Portfolio wing risk: the sum of open max-losses is what a
        # correlated crash (every name through its put wing at once) costs.
        # Bounding it here makes worst-case drawdown a chosen number instead
        # of whatever the per-position gates happen to allow.
        open_risk = sum(snapshot.exposure_by_symbol.values()) + committed_risk
        portfolio_risk_cap = self._max_portfolio_risk_pct * snapshot.equity
        if open_risk + projected_loss > portfolio_risk_cap:
            reasons.append(
                f"portfolio wing risk ${open_risk + projected_loss:.2f} would exceed "
                f"cap ${portfolio_risk_cap:.2f} ({self._max_portfolio_risk_pct:.1%} equity)"
            )

        # 4. Per-sector concentration (open positions + earlier approvals this cycle)
        sector = self._sector_map.get(signal.symbol, "unknown")
        sector_positions = (
            snapshot.positions_by_sector.get(sector, 0)
            + committed_sector.get(sector, 0)
        )
        if sector_positions + 1 > self._max_sector_pos:
            reasons.append(
                f"{sector} sector already has {sector_positions} positions "
                f"(cap {self._max_sector_pos})"
            )

        # 5. Portfolio Greek limits (project after-trade)
        if sized_qty > 0:
            proposed_greeks = self._proposed_greeks(signal, scan, sized_qty)
            for greek_name, limit_pct in (
                ("delta", self._delta_pct),
                ("gamma", self._gamma_pct),
                ("vega", self._vega_pct),
            ):
                if proposed_greeks is None:
                    continue
                current = snapshot.portfolio_greeks.get(greek_name, 0.0)
                projected = current + proposed_greeks.get(greek_name, 0.0)
                cap = limit_pct * snapshot.equity
                if abs(projected) > cap:
                    reasons.append(
                        f"projected portfolio {greek_name}=${projected:.2f} would exceed "
                        f"cap ${cap:.2f} ({limit_pct:.1%} equity)"
                    )

        # 6. Margin buffer (only meaningful for credit/short positions)
        if signal.direction == "SELL" and sized_qty > 0:
            margin_required = self._margin_required(signal, sized_qty)
            usable_bp = snapshot.buying_power * (1 - self._bp_buffer_pct)
            if margin_required > usable_bp:
                reasons.append(
                    f"required margin ${margin_required:.2f} exceeds usable buying power "
                    f"${usable_bp:.2f} (buffer {self._bp_buffer_pct:.0%} reserved)"
                )

        approved = not reasons
        final_qty = sized_qty if approved else 0
        return RiskDecision(
            signal=signal, approved=approved, quantity=final_qty,
            reasons=reasons, snapshot=snapshot,
            projected_max_loss=projected_loss if approved else 0.0,
            sized_quantity=sized_qty,
        )

    def _max_loss_per_contract(self, signal: TradeSignal, scan: ScanResult) -> float:
        snap = scan.snapshots.get(signal.symbol)
        if snap is None:
            return float("nan")
        contracts_by_key = {
            (c.strike, c.option_type): c
            for c in snap.contracts
            if c.expiration == signal.expiration
        }
        try:
            leg_contracts = [
                contracts_by_key[(leg.strike, leg.option_type)]
                for leg in signal.legs
            ]
        except KeyError:
            return float("nan")

        if signal.direction == "BUY":
            if len(leg_contracts) != 2:
                return float("nan")
            call = next((c for c in leg_contracts if c.option_type == "call"), None)
            put = next((c for c in leg_contracts if c.option_type == "put"), None)
            if call is None or put is None:
                return float("nan")
            debit = compute_straddle_debit(call, put, slippage=0.0)
            return debit * 100

        # SELL iron condor
        if len(leg_contracts) != 4:
            return float("nan")
        shorts = [c for c, leg in zip(leg_contracts, signal.legs) if leg.side == "sell"]
        longs = [c for c, leg in zip(leg_contracts, signal.legs) if leg.side == "buy"]
        if len(shorts) != 2 or len(longs) != 2:
            return float("nan")
        short_call = next((c for c in shorts if c.option_type == "call"), None)
        short_put = next((c for c in shorts if c.option_type == "put"), None)
        long_call = next((c for c in longs if c.option_type == "call"), None)
        long_put = next((c for c in longs if c.option_type == "put"), None)
        if not all([short_call, short_put, long_call, long_put]):
            return float("nan")
        credit = compute_iron_condor_credit(short_call, short_put, long_call, long_put, slippage=0.0)
        wing_distance = max(
            long_call.strike - short_call.strike,
            short_put.strike - long_put.strike,
        )
        # Max loss per contract = wing × 100 - credit × 100
        return (wing_distance - credit) * 100

    def _proposed_greeks(
        self,
        signal: TradeSignal,
        scan: ScanResult,
        quantity: int,
    ) -> dict[str, float] | None:
        snap = scan.snapshots.get(signal.symbol)
        if snap is None:
            return None
        contracts_by_key = {
            (c.strike, c.option_type): c
            for c in snap.contracts
            if c.expiration == signal.expiration
        }
        out = {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
        for leg in signal.legs:
            contract = contracts_by_key.get((leg.strike, leg.option_type))
            if contract is None:
                return None
            sign = 1 if leg.side == "buy" else -1
            out["delta"] += sign * contract.delta * quantity * 100
            out["gamma"] += sign * contract.gamma * quantity * 100
            out["vega"] += sign * contract.vega * quantity * 100
        return out

    def _margin_required(self, signal: TradeSignal, quantity: int) -> float:
        """Conservative margin estimate. For iron condor: wing_distance × 100 × qty."""
        if signal.direction != "SELL":
            return 0.0
        calls = sorted([l for l in signal.legs if l.option_type == "call"], key=lambda l: l.strike)
        puts = sorted([l for l in signal.legs if l.option_type == "put"], key=lambda l: l.strike)
        if len(calls) != 2 or len(puts) != 2:
            return 0.0
        wing = max(calls[1].strike - calls[0].strike, puts[1].strike - puts[0].strike)
        return wing * 100 * quantity
