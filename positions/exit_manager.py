from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from data.earnings_calendar import EarningsCalendar
from data.market_data import ScanResult
from execution.order_log import OrderLog
from execution.order_manager import OrderManager, OrderResult
from features import FeaturePipeline
from model.best_predictor import BestPredictor
from model.h1_predictor import H1DeviationPredictor
from model.term_structure import project_term_vol
from positions.position_tracker import OpenPosition, PositionMark, PositionTracker
from signals.signal_generator import (
    DEFAULT_PHI,
    TRADING_DAYS_PER_YEAR,
    blend_prediction,
    find_atm_iv,
    interpolate_horizon,
)

logger = logging.getLogger(__name__)


# earnings_risk sits between stop_loss and assignment_risk: it is a
# categorical risk exit like assignment_risk (must fire regardless of P&L)
# but can trigger weeks before expiry, so it must beat assignment_risk's
# DTE conditions; it must not outrank stop_loss, which closes anyway.
EXIT_TRIGGER_PRIORITY = (
    "thesis_reversed",
    "stop_loss",
    "earnings_risk",
    "assignment_risk",
    "expiration_proximity",
    "profit_target",
)

# A calendar whose last successful refresh is older than this is treated as
# unhealthy → the earnings exit falls back to each position's stored date.
CALENDAR_STALE_DAYS = 3


def _trading_days_between(a: date, b: date) -> int:
    """Weekdays strictly between a and b (a < d < b). Holidays are counted as
    trading days — the approximation only ever closes a position one weekday
    EARLIER than strictly needed, which is the safe direction."""
    if b <= a:
        return 0
    n = 0
    d = a + timedelta(days=1)
    while d < b:
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n


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
        predictors_by_horizon: dict[int, BestPredictor] | None = None,
        h1_predictor: H1DeviationPredictor | None = None,
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
        earnings_calendar: EarningsCalendar | None = None,
        earnings_exit_buffer_trading_days: int = 1,
        order_log: OrderLog | None = None,
        alert_cb=None,
    ):
        self._tracker = position_tracker
        self._order_manager = order_manager
        # Divergence source: h=1 term-projected forecast (active pipeline) or
        # the legacy multi-horizon predictors. Either may be None; divergence
        # then reports None and the thesis exit simply never fires.
        self._predictors = predictors_by_horizon
        self._h1 = h1_predictor
        self._straddle_pt = straddle_profit_target_pct
        self._straddle_sl = straddle_stop_loss_pct
        self._ic_pt = iron_condor_profit_target_pct
        self._ic_sl = iron_condor_stop_loss_pct
        self._exp_dte = expiration_proximity_dte
        self._thesis_min = thesis_reversal_min_magnitude
        self._thesis_enabled = thesis_exit_enabled
        # Earnings exit (fail-CLOSED): no short-vol position may hold through
        # an earnings report. Primary source is the live calendar; when it is
        # missing or stale the STORED per-position date (stamped at entry,
        # refreshed on healthy reads) decides — an API outage must never
        # leave a short position riding into a report.
        self._earnings = earnings_calendar
        self._earnings_buffer_td = earnings_exit_buffer_trading_days
        self._order_log = order_log
        # Out-of-band alert channel (Slack in production) for conditions the
        # journal alone would bury — currently the earnings manual-review
        # flag. Deduped to one alert per (position, day).
        self._alert_cb = alert_cb
        self._alerted_manual_review: set[tuple[int, date]] = set()
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
        today = scan.fetched_at.date()
        decisions: list[ExitDecision] = []
        for mark in marks:
            current_div = self._compute_current_divergence(
                mark, scan, feature_rows, returns_by_symbol,
            )
            trigger, rationale = self._evaluate_one(mark, current_div, today)
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

        if pos.symbol not in feature_rows:
            return None

        # h=1 pipeline: term-project the deviation forecast to the position's
        # REMAINING life (current DTE, not entry DTE).
        if self._h1 is not None:
            row = feature_rows[pos.symbol]
            if "log_gk_baseline_63" not in row.columns:
                return None
            b_t = float(row["log_gk_baseline_63"].iloc[-1])
            if not math.isfinite(b_t):
                return None
            phi = float("nan")
            if "garch_persistence" in row.columns:
                phi = float(row["garch_persistence"].iloc[-1])
            if not math.isfinite(phi):
                phi = DEFAULT_PHI
            try:
                dev_hat = self._h1.predict_deviation(row)
            except Exception as e:
                logger.warning("h=1 prediction failed for %s: %s", pos.symbol, e)
                return None
            if not math.isfinite(dev_hat):
                return None
            forecast = project_term_vol(b_t, dev_hat, phi, max(1, mark.dte))
            return forecast - current_atm_iv

        # Legacy pipeline: re-derive horizon mix from CURRENT DTE.
        if self._predictors is None or pos.symbol not in returns_by_symbol:
            return None
        horizon_pair = interpolate_horizon(mark.dte)
        if horizon_pair is None:
            return None
        h_lo, h_up, w_lo = horizon_pair

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
        today: date,
    ) -> tuple[str | None, str]:
        pos = mark.position

        # 1. Thesis reversed (HIGHEST priority — overrides P&L). The thesis is
        # the POSITION's direction, not the sign of entry_divergence: a SELL
        # is a bet that vol realizes below the IV sold, so it reverses when
        # the model now sees vol clearly ABOVE the market (divergence >=
        # +thesis_min); a BUY reverses on the mirror image. (In the h=1
        # pipeline direction comes from the VRP z-gate, so entry_divergence's
        # sign no longer encodes the thesis — keying off it could close a
        # position exactly when the model turned favorable.)
        if self._thesis_enabled and current_divergence is not None:
            reversed_now = (
                current_divergence >= self._thesis_min
                if pos.direction == "SELL"
                else current_divergence <= -self._thesis_min
            )
            if reversed_now:
                return "thesis_reversed", (
                    f"{pos.direction} thesis reversed: divergence "
                    f"{pos.entry_divergence:+.4f} -> {current_divergence:+.4f}"
                )

        # 2. Stop loss
        sl_threshold = self._stop_loss_threshold(pos)
        if mark.pnl_dollars <= sl_threshold:
            return "stop_loss", (
                f"pnl ${mark.pnl_dollars:.2f} <= stop ${sl_threshold:.2f} "
                f"({sl_threshold / (pos.entry_premium * 100):+.0%} of entry premium)"
            )

        # 3. Earnings risk (short-vol positions never hold through a report)
        earnings_rationale = self._earnings_risk(mark, today)
        if earnings_rationale is not None:
            return "earnings_risk", earnings_rationale

        # 4. Assignment/pin risk (positions with short legs only)
        assignment_rationale = self._assignment_risk(mark)
        if assignment_rationale is not None:
            return "assignment_risk", assignment_rationale

        # 5. Expiration proximity (long straddles: wings don't apply, but the
        # position bleeds its remaining premium into expiry). Short structures
        # are covered by the assignment-risk close-out above.
        if pos.structure != "iron_condor" and mark.dte <= self._exp_dte:
            return "expiration_proximity", f"dte={mark.dte} <= {self._exp_dte}"

        # 6. Profit target
        pt_threshold = self._profit_target_threshold(pos)
        if mark.pnl_dollars >= pt_threshold:
            return "profit_target", (
                f"pnl ${mark.pnl_dollars:.2f} >= target ${pt_threshold:.2f} "
                f"({pt_threshold / (pos.entry_premium * 100):+.0%} of entry premium)"
            )

        return None, "hold"

    def _earnings_risk(self, mark: PositionMark, today: date) -> str | None:
        """No short-vol position holds through earnings: close when fewer than
        `earnings_exit_buffer_trading_days` trading days remain strictly
        between today and the report date (weekday arithmetic; a holiday only
        makes us close a day early — the safe direction).

        Fail-CLOSED source chain: a healthy calendar (refreshed within
        CALENDAR_STALE_DAYS) is authoritative and refreshes the per-position
        stored date; a dead/stale calendar falls back to the STORED date with
        a loud alert; a position with neither is flagged for manual review
        every cycle — never silently held."""
        pos = mark.position
        if not any(l.side == "sell" for l in pos.legs):
            return None
        # Report already inside the position's window? Only dates on/before
        # expiration matter — a report after expiry can't hurt this position.
        earnings_date: date | None = None
        last_refresh = (
            self._earnings.last_refresh_date() if self._earnings is not None else None
        )
        # Healthy = recently refreshed AND the cache actually has content. An
        # HTTP-200 empty/unparseable payload wipes the cache but still stamps
        # last_refresh_date — without the content check that single bad
        # response would count as "no earnings anywhere" and simultaneously
        # erase the stored fallback dates below.
        calendar_healthy = (
            last_refresh is not None
            and (today - last_refresh).days <= CALENDAR_STALE_DAYS
            and self._earnings.cached_row_count() > 0
        )
        if calendar_healthy:
            earnings_date = self._earnings.next_earnings_on_or_after(pos.symbol, today)
            if earnings_date is None and pos.next_earnings_date is not None:
                # A healthy calendar not listing this symbol usually means no
                # report inside its 60-day lookahead — but never ERASE the
                # stored date over it; keep the belt with the suspenders.
                if pos.next_earnings_date >= today:
                    earnings_date = pos.next_earnings_date
            elif (earnings_date is not None
                    and self._order_log is not None
                    and earnings_date != pos.next_earnings_date):
                # Refresh the fail-closed store on every healthy read (report
                # dates move; the stored value must track them).
                try:
                    self._order_log.update_next_earnings_date(
                        pos.tradier_order_id, earnings_date,
                    )
                except Exception as e:
                    logger.warning("could not refresh stored earnings date for %s: %s",
                                   pos.symbol, e)
        else:
            stored = pos.next_earnings_date
            if stored is not None and stored >= today:
                earnings_date = stored
                logger.error(
                    "earnings calendar unavailable/stale (last refresh %s) — "
                    "using STORED earnings date %s for %s (fail-closed)",
                    last_refresh, stored.isoformat(), pos.symbol,
                )
            elif self._earnings is not None and stored is None:
                message = (
                    f"MANUAL REVIEW: earnings calendar unavailable/stale and no "
                    f"stored earnings date for short-vol position "
                    f"{pos.tradier_order_id} ({pos.symbol}) — "
                    f"cannot verify it is safe to hold"
                )
                logger.error(message)
                key = (pos.tradier_order_id, today)
                if self._alert_cb is not None and key not in self._alerted_manual_review:
                    self._alerted_manual_review.add(key)
                    try:
                        self._alert_cb(f"🚨 {message}")
                    except Exception as e:
                        logger.warning("manual-review alert failed: %s", e)
                return None

        if earnings_date is None or earnings_date < today:
            return None
        if earnings_date > pos.expiration:
            return None
        if _trading_days_between(today, earnings_date) <= self._earnings_buffer_td:
            return (
                f"{pos.symbol} reports {earnings_date.isoformat()} — closing "
                f">= {self._earnings_buffer_td} trading day(s) ahead "
                f"(short-vol never holds through earnings)"
            )
        return None

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
