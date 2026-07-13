from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import pandas as pd

from data.market_data import ScanResult
from execution.order_manager import OrderManager, OrderResult
from features import FeaturePipeline
from model.best_predictor import BestPredictor
from positions.position_tracker import OpenPosition, PositionMark, PositionTracker
from signals.signal_generator import (
    TRADING_DAYS_PER_YEAR,
    blend_prediction,
    find_atm_iv,
    interpolate_horizon,
)

logger = logging.getLogger(__name__)


EXIT_TRIGGER_PRIORITY = (
    "thesis_reversed",
    "stop_loss",
    "assignment_risk",
    "expiration_proximity",
    "profit_target",
)


@dataclass(frozen=True)
class ExitDecision:
    position: OpenPosition
    mark: PositionMark
    trigger: str | None         # None = hold
    action: str                 # "close" | "hold"
    rationale: str
    current_divergence: float | None  # for diagnostics


class ExitManager:
    def __init__(
        self,
        position_tracker: PositionTracker,
        order_manager: OrderManager,
        predictors_by_horizon: dict[int, BestPredictor],
        straddle_profit_target_pct: float = 1.00,
        straddle_stop_loss_pct: float = -0.50,
        iron_condor_profit_target_pct: float = 0.50,
        iron_condor_stop_loss_pct: float = -1.00,
        expiration_proximity_dte: int = 2,
        thesis_reversal_min_magnitude: float = 0.05,
        thesis_exit_enabled: bool = True,
        short_close_dte: int = 1,
        short_strike_buffer_pct: float = 0.015,
        short_extrinsic_floor: float = 0.05,
    ):
        self._tracker = position_tracker
        self._order_manager = order_manager
        self._predictors = predictors_by_horizon
        self._straddle_pt = straddle_profit_target_pct
        self._straddle_sl = straddle_stop_loss_pct
        self._ic_pt = iron_condor_profit_target_pct
        self._ic_sl = iron_condor_stop_loss_pct
        self._exp_dte = expiration_proximity_dte
        self._thesis_min = thesis_reversal_min_magnitude
        # Harvest mode has no model thesis — positions are premium harvests,
        # not divergence bets — so the thesis-reversal trigger is disabled
        # there. Divergence is still computed and recorded on every decision
        # for diagnostics; only the trigger is off.
        self._thesis_enabled = thesis_exit_enabled
        # Assignment/pin-risk close-out for positions with SHORT legs. Wings
        # cap the wing-to-wing loss, but they don't stop early assignment on a
        # short American-style leg or a pin at expiry — either leaves naked
        # stock outside every risk gate. Three rules, all labeled
        # "assignment_risk":
        #   (a) never carry short legs into expiration day (dte <= 0);
        #   (b) at dte <= short_close_dte, close if any short leg is in or
        #       within short_strike_buffer_pct of the money (fails SAFE when
        #       the underlying quote is missing);
        #   (c) at ANY dte, close when a short leg trades at parity —
        #       extrinsic <= short_extrinsic_floor makes early exercise
        #       rational for the counterparty (dividend capture on calls,
        #       cost-of-carry on puts).
        # Note the harvest structure's shorts sit at the entry ATM strike, so
        # near expiry one of them is almost always in the money: in practice
        # (b) closes harvest positions at dte = short_close_dte, one day
        # earlier than the ride-to-expiry backtest assumed. That is the
        # intended trade: the last day of theta is not worth carrying
        # assignment risk the bot cannot manage after its final cycle.
        self._short_close_dte = short_close_dte
        self._short_buffer = short_strike_buffer_pct
        self._extrinsic_floor = short_extrinsic_floor

    def evaluate(
        self,
        marks: list[PositionMark],
        scan: ScanResult,
        feature_rows: dict[str, pd.DataFrame],
        returns_by_symbol: dict[str, pd.Series],
    ) -> list[ExitDecision]:
        decisions: list[ExitDecision] = []
        for mark in marks:
            current_div = self._compute_current_divergence(
                mark, scan, feature_rows, returns_by_symbol,
            )
            trigger, rationale = self._evaluate_one(mark, current_div)
            decisions.append(ExitDecision(
                position=mark.position,
                mark=mark,
                trigger=trigger,
                action="close" if trigger is not None else "hold",
                rationale=rationale,
                current_divergence=current_div,
            ))
        return decisions

    def _compute_current_divergence(
        self,
        mark: PositionMark,
        scan: ScanResult,
        feature_rows: dict[str, pd.DataFrame],
        returns_by_symbol: dict[str, pd.Series],
    ) -> float | None:
        pos = mark.position
        snap = scan.snapshots.get(pos.symbol)
        if snap is None:
            return None
        chain = [c for c in snap.contracts if c.expiration == pos.expiration]
        if not chain:
            return None
        try:
            underlying = float(snap.underlying.get("last") or 0.0)
        except (TypeError, ValueError):
            return None
        if underlying <= 0:
            return None
        atm_pair = find_atm_iv(chain, underlying)
        if atm_pair is None:
            return None
        atm_call, atm_put = atm_pair
        ivs = [iv for iv in (atm_call.iv, atm_put.iv) if iv > 0]
        if not ivs:
            return None
        current_atm_iv = sum(ivs) / len(ivs)

        # Re-derive horizon mix from CURRENT DTE (not entry DTE) — predictions
        # should be calibrated to the position's remaining payoff window.
        horizon_pair = interpolate_horizon(mark.dte)
        if horizon_pair is None:
            return None
        h_lo, h_up, w_lo = horizon_pair

        if pos.symbol not in feature_rows or pos.symbol not in returns_by_symbol:
            return None

        preds: dict[int, float] = {}
        for h in (h_lo, h_up):
            if h not in self._predictors:
                return None
            try:
                preds[h] = float(self._predictors[h].predict_forward_rv(
                    returns_history=returns_by_symbol[pos.symbol],
                    X_row=feature_rows[pos.symbol],
                ))
            except Exception as e:
                logger.warning("prediction failed for %s @ h=%d: %s", pos.symbol, h, e)
                return None

        pred_rv_daily = blend_prediction(preds, h_lo, h_up, w_lo)
        predicted_iv_eq = pred_rv_daily * math.sqrt(TRADING_DAYS_PER_YEAR)
        return predicted_iv_eq - current_atm_iv

    def _evaluate_one(
        self,
        mark: PositionMark,
        current_divergence: float | None,
    ) -> tuple[str | None, str]:
        pos = mark.position

        # 1. Thesis reversed (HIGHEST priority — overrides P&L)
        if self._thesis_enabled and current_divergence is not None:
            entry_sign = 1 if pos.entry_divergence > 0 else -1
            current_sign = 1 if current_divergence > 0 else -1
            if (current_sign != entry_sign
                    and abs(current_divergence) >= self._thesis_min):
                return "thesis_reversed", (
                    f"divergence flipped {pos.entry_divergence:+.4f} -> {current_divergence:+.4f}"
                )

        # 2. Stop loss
        sl_threshold = self._stop_loss_threshold(pos)
        if mark.pnl_dollars <= sl_threshold:
            return "stop_loss", (
                f"pnl ${mark.pnl_dollars:.2f} <= stop ${sl_threshold:.2f} "
                f"({sl_threshold / (pos.entry_premium * 100):+.0%} of entry premium)"
            )

        # 3. Assignment/pin risk (positions with short legs only)
        assignment_rationale = self._assignment_risk(mark)
        if assignment_rationale is not None:
            return "assignment_risk", assignment_rationale

        # 4. Expiration proximity (long straddles: wings don't apply, but the
        # position bleeds its remaining premium into expiry). Short structures
        # are covered by the assignment-risk close-out above.
        if pos.structure != "iron_condor" and mark.dte <= self._exp_dte:
            return "expiration_proximity", f"dte={mark.dte} <= {self._exp_dte}"

        # 5. Profit target
        pt_threshold = self._profit_target_threshold(pos)
        if mark.pnl_dollars >= pt_threshold:
            return "profit_target", (
                f"pnl ${mark.pnl_dollars:.2f} >= target ${pt_threshold:.2f} "
                f"({pt_threshold / (pos.entry_premium * 100):+.0%} of entry premium)"
            )

        return None, "hold"

    def _assignment_risk(self, mark: PositionMark) -> str | None:
        """Assignment/pin-risk close-out (see the constructor comment for the
        three rules). Returns a rationale string when the position must close,
        None otherwise. No-op for positions without short legs."""
        pos = mark.position
        short_legs = [l for l in pos.legs if l.side == "sell"]
        if not short_legs:
            return None

        underlying = mark.underlying_price
        have_underlying = math.isfinite(underlying) and underlying > 0

        # (a) Expiry-day backstop. Whatever premium is left is pennies against
        # a pin that resolves AFTER the bot's last cycle of the day.
        if mark.dte <= 0:
            return (
                f"dte={mark.dte}: short legs never ride through expiration "
                "(pin risk resolves after the last cycle)"
            )

        # (c) Parity check runs at any DTE — early assignment doesn't wait for
        # expiry week. Needs current quotes; skipped when legs didn't mark.
        if have_underlying and len(mark.current_legs) == len(pos.legs):
            for opened, current in zip(pos.legs, mark.current_legs):
                if opened.side != "sell":
                    continue
                intrinsic = (
                    max(0.0, underlying - current.strike)
                    if current.option_type == "call"
                    else max(0.0, current.strike - underlying)
                )
                if intrinsic <= 0:
                    continue
                mid = (current.bid + current.ask) / 2.0
                if mid <= 0:
                    continue
                extrinsic = mid - intrinsic
                if extrinsic <= self._extrinsic_floor:
                    return (
                        f"short {current.option_type} K={current.strike} ITM at "
                        f"parity (extrinsic ${extrinsic:.2f} <= "
                        f"${self._extrinsic_floor:.2f}) — early assignment is "
                        "rational for the counterparty"
                    )

        # (b) Close window: dte <= short_close_dte with a short leg in or near
        # the money. Missing underlying fails SAFE — for the harvest structure
        # a short leg is nearly always near the money here, so closing blind
        # is almost always what the check would have decided anyway.
        if mark.dte <= self._short_close_dte:
            if not have_underlying:
                return (
                    f"dte={mark.dte} <= {self._short_close_dte} and underlying "
                    "quote unavailable — failing safe"
                )
            for leg in short_legs:
                near = (
                    underlying >= leg.strike * (1 - self._short_buffer)
                    if leg.option_type == "call"
                    else underlying <= leg.strike * (1 + self._short_buffer)
                )
                if near:
                    return (
                        f"short {leg.option_type} K={leg.strike} in/near the money "
                        f"(underlying {underlying:.2f}, buffer "
                        f"{self._short_buffer:.1%}) at dte={mark.dte}"
                    )

        return None

    def _profit_target_threshold(self, pos: OpenPosition) -> float:
        if pos.is_long:
            return self._straddle_pt * pos.entry_premium * 100
        return self._ic_pt * pos.entry_premium * 100

    def _stop_loss_threshold(self, pos: OpenPosition) -> float:
        if pos.is_long:
            return self._straddle_sl * pos.entry_premium * 100
        return self._ic_sl * pos.entry_premium * 100

    async def execute(
        self,
        decisions: list[ExitDecision],
        dry_run: bool = True,
    ) -> list[tuple[ExitDecision, OrderResult | None]]:
        """Submit closing orders for decisions with action='close'.
        dry_run=True (default): log what would happen, don't submit."""
        results: list[tuple[ExitDecision, OrderResult | None]] = []
        for decision in decisions:
            if decision.action != "close":
                results.append((decision, None))
                continue
            if dry_run:
                logger.info("DRY RUN: would close %s (%s) trigger=%s",
                            decision.position.tradier_order_id,
                            decision.position.symbol,
                            decision.trigger)
                results.append((decision, None))
                continue
            order_result = await self._order_manager.submit_close(
                position=decision.position,
                mark=decision.mark,
                exit_trigger=decision.trigger,
            )
            results.append((decision, order_result))
        return results
