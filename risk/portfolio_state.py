from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from config import Ticker
from data.async_client import AsyncTradierClient
from data.market_data import ScanResult
from execution.order_log import OrderLog
from positions.position_tracker import OpenPosition, PositionMark, PositionTracker
from risk.kill_switch import DailyKillSwitch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioSnapshot:
    fetched_at: datetime
    equity: float
    starting_equity_today: float
    buying_power: float
    margin_held: float
    open_positions: list[OpenPosition]
    open_marks: list[PositionMark]
    today_realized_pnl: float
    today_unrealized_pnl: float
    portfolio_greeks: dict[str, float]
    positions_by_sector: dict[str, int]
    exposure_by_symbol: dict[str, float]   # max-loss $ per symbol

    @property
    def today_total_pnl(self) -> float:
        return self.today_realized_pnl + self.today_unrealized_pnl

    @property
    def today_pnl_pct_of_equity(self) -> float:
        if self.starting_equity_today <= 0:
            return 0.0
        return self.today_total_pnl / self.starting_equity_today


class PortfolioStateBuilder:
    def __init__(
        self,
        client: AsyncTradierClient,
        order_log: OrderLog,
        position_tracker: PositionTracker,
        watchlist: list[Ticker],
        kill_switch: DailyKillSwitch,
    ):
        self._client = client
        self._log = order_log
        self._tracker = position_tracker
        self._sector_map = {t.symbol: t.sector for t in watchlist}
        self._kill_switch = kill_switch

    async def snapshot(self, scan: ScanResult) -> PortfolioSnapshot:
        balances_resp = await self._client.get_balances()
        equity = float(balances_resp.get("total_equity", 0.0))
        margin = balances_resp.get("margin", {}) or {}
        buying_power = float(margin.get("option_buying_power", 0.0))
        margin_held = float(balances_resp.get("current_requirement", 0.0))

        positions = await self._tracker.list_open_positions()
        marks = self._tracker.mark_to_market(positions, scan)

        today = scan.fetched_at.date()
        today_realized = self._log.closed_today_pnl(today)
        today_unrealized = sum(m.pnl_dollars for m in marks)

        starting_equity = self._kill_switch.get_starting_equity(today, equity)

        exposure: dict[str, float] = defaultdict(float)
        sector_count: dict[str, int] = defaultdict(int)
        for pos in positions:
            ml = pos.max_loss_dollars
            if ml == ml:  # not NaN
                exposure[pos.symbol] += ml
            sector_count[self._sector_map.get(pos.symbol, "unknown")] += 1

        return PortfolioSnapshot(
            fetched_at=scan.fetched_at,
            equity=equity,
            starting_equity_today=starting_equity,
            buying_power=buying_power,
            margin_held=margin_held,
            open_positions=positions,
            open_marks=marks,
            today_realized_pnl=today_realized,
            today_unrealized_pnl=today_unrealized,
            portfolio_greeks=PositionTracker.portfolio_greeks(marks),
            positions_by_sector=dict(sector_count),
            exposure_by_symbol=dict(exposure),
        )
