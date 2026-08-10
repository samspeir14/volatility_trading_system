from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from data.earnings_calendar import EarningsCalendar
from execution.order_log import OrderLog
from risk.kill_switch import DailyKillSwitch
from risk.risk_rejection_log import RiskRejectionLog
from signals.divergence_history import DivergenceHistory

if TYPE_CHECKING:
    from risk.portfolio_state import PortfolioSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EarningsStraddlingPosition:
    symbol: str
    expiration: date
    earnings_date: date
    structure: str


@dataclass(frozen=True)
class AssignmentAlertSummary:
    """One row per outstanding assignment alert, mirrored from
    order_log.assignment_alerts. Persists until manually dismissed."""
    tradier_order_id: int
    symbol: str
    expiration: date
    structure: str
    stock_quantity: float | None
    detected_at: datetime


@dataclass(frozen=True)
class StaleCloseAlertSummary:
    """One row per opening order that hit the close-retry cap. Persists until
    manually dismissed. CRITICAL — these positions need human intervention."""
    opening_order_id: int
    symbol: str
    expiration: date
    structure: str
    attempts: int
    last_exit_trigger: str | None
    detected_at: datetime


@dataclass(frozen=True)
class PendingCloseSummary:
    """One row per close currently in-flight at Tradier (close_attempt.status
    = 'pending'). Informational — surfaced so the operator can see exits in
    progress before they accumulate into stale-close alerts."""
    closing_order_id: int
    opening_order_id: int
    symbol: str
    structure: str
    exit_trigger: str
    submitted_price: float
    submitted_at: datetime


@dataclass(frozen=True)
class PositionPnlGroup:
    """Per-position P&L breakdown line. Identical open positions (same symbol,
    structure, and expiration — e.g. several NVDA iron condors expiring the
    same day) are collapsed into one row with `count` legs-of-the-same-trade.
    total_pnl is lifetime since-entry; daily_pnl is the change since the prior
    EOD snapshot."""
    symbol: str
    structure: str
    expiration: date
    count: int
    total_pnl: float
    daily_pnl: float


@dataclass(frozen=True)
class DailySummary:
    date: date
    starting_equity: float
    ending_equity: float
    realized_pnl: float
    unrealized_pnl: float
    open_positions: int
    positions_opened_today: int
    positions_closed_today: int
    signals_total: int
    signals_approved: int
    risk_rejections_total: int
    risk_rejections_by_reason: dict[str, int]
    kill_switch_activated: bool
    top_exit_triggers: dict[str, int]
    earnings_straddling_positions: list[EarningsStraddlingPosition] = field(default_factory=list)
    assignment_alerts: list[AssignmentAlertSummary] = field(default_factory=list)
    stale_close_alerts: list[StaleCloseAlertSummary] = field(default_factory=list)
    pending_closes: list[PendingCloseSummary] = field(default_factory=list)
    position_pnl_breakdown: list[PositionPnlGroup] = field(default_factory=list)
    # Signal-level gate pass rates for today: {"candidates": N,
    # "<gate name>": blocked count..., "approved": M}. Surfaces a silent
    # zero-signal deadlock (every candidate dying at one gate) immediately.
    gate_pass_rates: dict[str, int] = field(default_factory=dict)

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def total_pnl_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return self.total_pnl / self.starting_equity

    @property
    def equity_change(self) -> float:
        return self.ending_equity - self.starting_equity


def _check_reconciliation(summary: "DailySummary") -> None:
    """End-of-day guard: reported P&L must match the equity change. Identity
    is exact by construction in portfolio_state.snapshot() (realized + unrealized
    = equity - starting_equity); a breach means starting_equity got lost,
    Tradier equity is wonky, or realized P&L was re-broken. Threshold: $5
    absolute drift, which is the spec — float-arithmetic noise is sub-penny.

    NOTE: this guard verifies the *bookkeeping identity*, not per-position
    correctness. If today_realized is wrong (e.g., an expired ITM long got
    booked at -entry_debit when it should have been intrinsic - debit),
    today_unrealized = equity_change - today_realized silently absorbs the
    error and total_pnl still equals equity_change. The split between
    realized vs unrealized is wrong but the sum reconciles, so this guard
    stays silent. Correct per-position P&L is enforced upstream in
    PositionReconciler.reconcile() via intrinsic settlement (see
    positions/reconciler.py) and a max-loss floor sanity check at write time."""
    equity_delta = summary.ending_equity - summary.starting_equity
    drift = abs(equity_delta - summary.total_pnl)
    tolerance = 5.0
    if drift > tolerance:
        logger.error(
            f"EOD reconciliation drift: equity_change=${equity_delta:+,.2f} "
            f"total_pnl=${summary.total_pnl:+,.2f} drift=${drift:,.2f} "
            f"tolerance=${tolerance:,.2f} "
            f"(starting_equity=${summary.starting_equity:,.2f} "
            f"ending_equity=${summary.ending_equity:,.2f} "
            f"realized=${summary.realized_pnl:+,.2f} "
            f"unrealized=${summary.unrealized_pnl:+,.2f})"
        )


class DailySummaryBuilder:
    def __init__(
        self,
        order_log: OrderLog,
        divergence_history: DivergenceHistory,
        risk_rejection_log: RiskRejectionLog,
        kill_switch: DailyKillSwitch,
        earnings_calendar: EarningsCalendar | None = None,
    ):
        self._order_log = order_log
        self._divergence_history = divergence_history
        self._risk_rejection_log = risk_rejection_log
        self._kill_switch = kill_switch
        self._earnings_calendar = earnings_calendar

    def build(self, today: date, snapshot: "PortfolioSnapshot") -> DailySummary:
        signals_total = self._signals_count_today(today)
        positions_opened = self._order_log.positions_opened_on(today)
        positions_closed = self._order_log.positions_closed_on(today)
        exit_triggers = self._order_log.exit_triggers_today(today)
        risk_rejections = self._risk_rejection_log.count_today(today)
        rejection_reasons = self._risk_rejection_log.reasons_summary_today(today)
        earnings_straddling = self._find_earnings_straddling_positions(today, snapshot)
        assignment_alerts = self._load_assignment_alerts()
        stale_close_alerts = self._load_stale_close_alerts()
        pending_closes = self._load_pending_closes()
        position_pnl_breakdown = self._compute_position_pnl_breakdown(today, snapshot)

        summary = DailySummary(
            date=today,
            starting_equity=snapshot.starting_equity_today,
            ending_equity=snapshot.equity,
            realized_pnl=snapshot.today_realized_pnl,
            unrealized_pnl=snapshot.today_unrealized_pnl,
            open_positions=len(snapshot.open_positions),
            positions_opened_today=positions_opened,
            positions_closed_today=positions_closed,
            signals_total=signals_total,
            signals_approved=positions_opened,  # signals approved == orders opened today
            risk_rejections_total=risk_rejections,
            risk_rejections_by_reason=rejection_reasons,
            kill_switch_activated=self._kill_switch.is_active(today),
            top_exit_triggers=exit_triggers,
            earnings_straddling_positions=earnings_straddling,
            assignment_alerts=assignment_alerts,
            stale_close_alerts=stale_close_alerts,
            pending_closes=pending_closes,
            position_pnl_breakdown=position_pnl_breakdown,
            gate_pass_rates=self._divergence_history.gate_counts_today(today),
        )
        _check_reconciliation(summary)
        return summary

    def _compute_position_pnl_breakdown(
        self, today: date, snapshot: "PortfolioSnapshot"
    ) -> list[PositionPnlGroup]:
        """Per-position total (since-entry) and daily (since prior EOD) P&L for
        the open book, grouping identical trades by (symbol, structure,
        expiration). Records today's lifetime P&L per position so tomorrow can
        diff against it."""
        # Aggregate by trade identity. Each value: [count, total_pnl, daily_pnl].
        groups: dict[tuple[str, str, date], list[float]] = {}
        for mark in snapshot.open_marks:
            lifetime = mark.pnl_dollars
            if lifetime != lifetime:  # NaN mark (missing legs) — skip
                continue
            pos = mark.position
            order_id = pos.tradier_order_id
            prior = self._order_log.previous_position_pnl(order_id, today)
            daily = lifetime if prior is None else lifetime - prior
            self._order_log.record_position_pnl_snapshot(order_id, today, lifetime)

            key = (pos.symbol, pos.structure, pos.expiration)
            agg = groups.setdefault(key, [0.0, 0.0, 0.0])
            agg[0] += 1
            agg[1] += lifetime
            agg[2] += daily

        out = [
            PositionPnlGroup(
                symbol=sym, structure=struct, expiration=exp,
                count=int(agg[0]), total_pnl=agg[1], daily_pnl=agg[2],
            )
            for (sym, struct, exp), agg in groups.items()
        ]
        # Biggest absolute total P&L first — surfaces the most material positions.
        out.sort(key=lambda g: abs(g.total_pnl), reverse=True)
        return out

    def _load_stale_close_alerts(self) -> list[StaleCloseAlertSummary]:
        out: list[StaleCloseAlertSummary] = []
        for row in self._order_log.stale_close_alerts_active():
            try:
                detected = datetime.fromisoformat(row["detected_at"])
            except (ValueError, TypeError):
                detected = datetime.now(timezone.utc)
            out.append(StaleCloseAlertSummary(
                opening_order_id=row["opening_order_id"],
                symbol=row["symbol"],
                expiration=date.fromisoformat(row["expiration"]),
                structure=row["structure"],
                attempts=row["attempts"],
                last_exit_trigger=row["last_exit_trigger"],
                detected_at=detected,
            ))
        return out

    def _load_pending_closes(self) -> list[PendingCloseSummary]:
        out: list[PendingCloseSummary] = []
        for row in self._order_log.pending_close_attempts():
            try:
                submitted = datetime.fromisoformat(row["submitted_at"])
            except (ValueError, TypeError):
                submitted = datetime.now(timezone.utc)
            out.append(PendingCloseSummary(
                closing_order_id=row["closing_order_id"],
                opening_order_id=row["opening_order_id"],
                symbol=row["symbol"],
                structure=row["structure"],
                exit_trigger=row["exit_trigger"],
                submitted_price=row["submitted_price"],
                submitted_at=submitted,
            ))
        return out

    def _load_assignment_alerts(self) -> list[AssignmentAlertSummary]:
        out: list[AssignmentAlertSummary] = []
        for row in self._order_log.assignment_alerts_active():
            try:
                detected = datetime.fromisoformat(row["detected_at"])
            except (ValueError, TypeError):
                detected = datetime.now(timezone.utc)
            out.append(AssignmentAlertSummary(
                tradier_order_id=row["tradier_order_id"],
                symbol=row["symbol"],
                expiration=date.fromisoformat(row["expiration"]),
                structure=row["structure"],
                stock_quantity=row["stock_quantity"],
                detected_at=detected,
            ))
        return out

    def _find_earnings_straddling_positions(
        self, today: date, snapshot: "PortfolioSnapshot"
    ) -> list[EarningsStraddlingPosition]:
        """Open positions whose expiration is on or after a known earnings date.
        Surfaced for human review only — exits stay on their normal triggers."""
        if self._earnings_calendar is None:
            return []
        out: list[EarningsStraddlingPosition] = []
        for pos in snapshot.open_positions:
            try:
                hit = self._earnings_calendar.has_earnings_in_window(
                    pos.symbol, today, pos.expiration,
                )
            except Exception as e:
                logger.warning(
                    "earnings lookup failed for %s: %s — skipping", pos.symbol, e
                )
                continue
            if not hit:
                continue
            earnings_date = self._earnings_calendar.next_earnings_on_or_after(
                pos.symbol, today,
            )
            if earnings_date is None:
                continue
            out.append(EarningsStraddlingPosition(
                symbol=pos.symbol,
                expiration=pos.expiration,
                earnings_date=earnings_date,
                structure=pos.structure,
            ))
            logger.info(
                "open position %s %s exp=%s straddles earnings on %s — "
                "surfaced in daily summary, holding to normal exit triggers",
                pos.symbol, pos.structure, pos.expiration.isoformat(),
                earnings_date.isoformat(),
            )
        return out

    def _signals_count_today(self, today: date) -> int:
        # divergence_history doesn't expose a count helper; query directly
        cur = self._divergence_history._conn.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) FROM divergence_log WHERE date(scan_date) = ?",
            (today.isoformat(),),
        )
        return int(cur.fetchone()[0])
