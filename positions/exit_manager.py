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

        # 3. Expiration proximity (long straddles only — IC wings cap risk)
        if pos.structure != "iron_condor" and mark.dte <= self._exp_dte:
            return "expiration_proximity", f"dte={mark.dte} <= {self._exp_dte}"

        # 4. Profit target
        pt_threshold = self._profit_target_threshold(pos)
        if mark.pnl_dollars >= pt_threshold:
            return "profit_target", (
                f"pnl ${mark.pnl_dollars:.2f} >= target ${pt_threshold:.2f} "
                f"({pt_threshold / (pos.entry_premium * 100):+.0%} of entry premium)"
            )

        return None, "hold"

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
