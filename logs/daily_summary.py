from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from execution.order_log import OrderLog
from risk.kill_switch import DailyKillSwitch
from risk.risk_rejection_log import RiskRejectionLog
from signals.divergence_history import DivergenceHistory

if TYPE_CHECKING:
    from risk.portfolio_state import PortfolioSnapshot

logger = logging.getLogger(__name__)


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


class DailySummaryBuilder:
    def __init__(
        self,
        order_log: OrderLog,
        divergence_history: DivergenceHistory,
        risk_rejection_log: RiskRejectionLog,
        kill_switch: DailyKillSwitch,
    ):
        self._order_log = order_log
        self._divergence_history = divergence_history
        self._risk_rejection_log = risk_rejection_log
        self._kill_switch = kill_switch

    def build(self, today: date, snapshot: "PortfolioSnapshot") -> DailySummary:
        signals_total = self._signals_count_today(today)
        positions_opened = self._order_log.positions_opened_on(today)
        positions_closed = self._order_log.positions_closed_on(today)
        exit_triggers = self._order_log.exit_triggers_today(today)
        risk_rejections = self._risk_rejection_log.count_today(today)
        rejection_reasons = self._risk_rejection_log.reasons_summary_today(today)

        return DailySummary(
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
        )

    def _signals_count_today(self, today: date) -> int:
        # divergence_history doesn't expose a count helper; query directly
        cur = self._divergence_history._conn.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) FROM divergence_log WHERE date(scan_date) = ?",
            (today.isoformat(),),
        )
        return int(cur.fetchone()[0])
