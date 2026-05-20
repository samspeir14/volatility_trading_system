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


_ACCOUNT_TYPE_KEYS = ("margin", "cash", "pdt")


def _account_sub_object(balances: dict) -> dict:
    """Return the account-type sub-object (margin/cash/pdt) Tradier nests inside the
    balances response. The key matches `account_type`; if that's missing, fall back
    to whichever known sub-object is present."""
    declared = balances.get("account_type")
    if declared and isinstance(balances.get(declared), dict):
        return balances[declared]
    for key in _ACCOUNT_TYPE_KEYS:
        sub = balances.get(key)
        if isinstance(sub, dict):
            return sub
    return {}


def _extract_option_buying_power(balances: dict) -> float:
    sub = _account_sub_object(balances)
    if "option_buying_power" in sub:
        return float(sub["option_buying_power"])
    # Cash accounts don't expose option_buying_power directly — usable funds are cash_available.
    if "cash_available" in sub:
        return float(sub["cash_available"])
    if "option_buying_power" in balances:
        return float(balances["option_buying_power"])
    raise ValueError(
        f"option_buying_power not found in balances response: keys={sorted(balances.keys())}"
    )


def _extract_margin_requirement(balances: dict) -> float:
    # Tradier reports current_requirement at the top level for all account types.
    if "current_requirement" in balances:
        return float(balances["current_requirement"])
    sub = _account_sub_object(balances)
    if "current_requirement" in sub:
        return float(sub["current_requirement"])
    return 0.0


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
        buying_power = _extract_option_buying_power(balances_resp)
        margin_held = _extract_margin_requirement(balances_resp)

        positions = await self._tracker.list_open_positions()
        marks = self._tracker.mark_to_market(positions, scan)

        today = scan.fetched_at.date()
        starting_equity = self._kill_switch.get_starting_equity(today, equity)
        today_realized = self._log.closed_today_pnl(today)
        # Today's P&L is the day-over-day equity delta. Cash conservation
        # (equity = cash + position MV) makes this exact regardless of
        # opens/closes during the day. The unrealized component is whatever
        # of today's equity change isn't explained by today's realized closes.
        # Previously we summed PositionMark.pnl_dollars, which is the lifetime
        # since-entry P&L per position — that drifts slowly and doesn't track
        # intraday mark moves.
        today_unrealized = (equity - starting_equity) - today_realized

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
