"""Unit tests for the execution price ladder (walk-the-book repricing), the
unfilled-entry cancel, TCA arrival-mid recording, and the entry time window."""
import asyncio
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

from config import Settings
from data.async_client import OptionContract
from data.market_data import TickerSnapshot
from execution import OrderLog, OrderManager, fingerprint_signal
from positions.position_tracker import OpenPosition, PositionMark
from signals.signal_generator import TradeLeg, TradeSignal


def _mk_settings() -> Settings:
    return Settings(
        api_key="fake", account_id="VA00000000",
        base_url="https://example.invalid/v1", env="sandbox",
    )


def _mk_contract(strike: float, otype: str, *, bid=1.0, ask=1.10) -> OptionContract:
    return OptionContract(
        symbol=f"AAPL_{int(strike)}{otype[0].upper()}",
        underlying="AAPL", expiration=date(2026, 5, 15),
        strike=strike, option_type=otype,
        bid=bid, ask=ask, last=(bid + ask) / 2,
        volume=200, open_interest=1000,
        delta=0.5 if otype == "call" else -0.5,
        gamma=0.01, theta=-0.05, vega=0.20,
        iv=0.30, fetched_at=datetime.now(timezone.utc),
    )


def _mk_sell_signal() -> TradeSignal:
    legs = [
        TradeLeg(100.0, "call", "sell", 1, "AAPL_100C"),
        TradeLeg(100.0, "put", "sell", 1, "AAPL_100P"),
        TradeLeg(105.0, "call", "buy", 1, "AAPL_105C"),
        TradeLeg(95.0, "put", "buy", 1, "AAPL_95P"),
    ]
    return TradeSignal(
        symbol="AAPL", expiration=date(2026, 5, 15), dte=10,
        horizon_lower=10, horizon_upper=10, weight_lower=1.0,
        direction="SELL", underlying_price=100.0, atm_iv=0.30,
        predicted_iv_equivalent=0.40, divergence=0.10,
        cross_sectional_z=2.0, time_series_z=None,
        liquidity_score=12345.0, legs=legs, is_actionable=True,
    )


def _mk_snapshot() -> TickerSnapshot:
    # short mids 2.05 + 1.95 = 4.00; long mids 0.525 + 0.475 = 1.00
    # -> net condor mid credit = 3.00
    return TickerSnapshot(
        symbol="AAPL", sector="tech",
        underlying={"symbol": "AAPL", "last": 100.0},
        contracts=[
            _mk_contract(100.0, "call", bid=2.00, ask=2.10),
            _mk_contract(100.0, "put", bid=1.90, ask=2.00),
            _mk_contract(105.0, "call", bid=0.50, ask=0.55),
            _mk_contract(95.0, "put", bid=0.45, ask=0.50),
        ],
    )


class _LadderFakeClient:
    """Stateful fake: order fills only once `fill_after_modifies` modify calls
    have happened (None = never fills). Cancel can succeed, or reject and
    reveal a race-fill."""

    def __init__(
        self,
        fill_after_modifies: int | None,
        fill_price: float = -2.94,
        cancel_ok: bool = True,
        filled_after_cancel: bool = False,
        order_id: int = 7777,
    ):
        self._fill_after = fill_after_modifies
        self._fill_price = fill_price
        self._cancel_ok = cancel_ok
        self._filled_after_cancel = filled_after_cancel
        self._order_id = order_id
        self._canceled = False
        self.place_price: float | None = None
        self.modify_prices: list[float] = []
        self.cancel_calls = 0

    async def preview_order(self, **kwargs):
        return {"order": {"status": "ok"}}

    async def place_order(self, **kwargs):
        self.place_price = float(kwargs["price"])
        return {"order": {"id": self._order_id, "status": "pending"}}

    async def modify_order(self, **kwargs):
        self.modify_prices.append(float(kwargs["price"]))
        return {"order": {"id": self._order_id, "status": "ok"}}

    async def get_order_status(self, account_id, order_id):
        filled = (
            (self._fill_after is not None
             and len(self.modify_prices) >= self._fill_after)
            or (self._canceled and self._filled_after_cancel)
        )
        if filled:
            return {"order": {"id": order_id, "status": "filled",
                              "avg_fill_price": self._fill_price}}
        return {"order": {"id": order_id, "status": "open"}}

    async def cancel_order(self, account_id, order_id):
        self.cancel_calls += 1
        self._canceled = True
        if self._cancel_ok and not self._filled_after_cancel:
            return {"order": {"id": order_id, "status": "ok"}}
        return {"errors": {"error": "order already in a terminal state"}}


def _mk_manager(client, log: OrderLog) -> OrderManager:
    # slippage_buffer 0.02 -> default ladder (0.0, 0.01, 0.02, 0.03)
    return OrderManager(
        client=client, order_log=log, settings=_mk_settings(),
        poll_interval_seconds=0.001, poll_timeout_seconds=0.2,
        slippage_buffer=0.02,
    )


def _submit(manager: OrderManager):
    return asyncio.run(manager.submit(
        _mk_sell_signal(), _mk_snapshot(), date(2026, 7, 13),
    ))


# ---------- entry ladder ----------

def test_entry_fills_at_mid_no_modifies():
    """A fill at step 0 means the order went in AT mid (not mid minus a
    donated haircut) and never repriced."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        client = _LadderFakeClient(fill_after_modifies=0, fill_price=-3.00)
        result = _submit(_mk_manager(client, log))
        assert result.status == "filled", result
        assert client.place_price == 3.00, f"placed at {client.place_price}, want mid 3.00"
        assert client.modify_prices == []
        row = log._conn.execute(
            "SELECT arrival_mid, fill_price FROM submitted_orders",
        ).fetchone()
        assert row == (3.00, -3.00), row
    print("ladder entry: fills at mid, arrival_mid recorded for TCA")


def test_entry_ladder_walks_toward_far_side():
    """Credit order concedes downward: 3.00 -> 2.97 -> 2.94, filling after the
    second modify."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        client = _LadderFakeClient(fill_after_modifies=2, fill_price=-2.94)
        result = _submit(_mk_manager(client, log))
        assert result.status == "filled", result
        assert client.modify_prices == [2.97, 2.94], client.modify_prices
        assert client.cancel_calls == 0
    print("ladder entry: walks 3.00 -> 2.97 -> 2.94 and fills")


def test_entry_unfilled_cancels_and_is_retryable():
    """Ladder exhausted: the order is canceled (no dangling day orders at a
    price nobody manages) and the fingerprint dedup allows a retry."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        client = _LadderFakeClient(fill_after_modifies=None)
        result = _submit(_mk_manager(client, log))
        assert result.status == "canceled", result
        assert client.modify_prices == [2.97, 2.94, 2.91], client.modify_prices
        assert client.cancel_calls == 1
        fp = fingerprint_signal(_mk_sell_signal(), date(2026, 7, 13))
        assert not log.has_recent_open_order(fp), (
            "canceled entry must not dedup-block the retry"
        )
    print("ladder entry: unfilled -> canceled -> retryable next cycle")


def test_entry_cancel_races_fill():
    """Cancel rejected because the order filled in the race: the re-query
    books the fill instead of losing it."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        client = _LadderFakeClient(
            fill_after_modifies=None, fill_price=-2.91,
            cancel_ok=False, filled_after_cancel=True,
        )
        result = _submit(_mk_manager(client, log))
        assert result.status == "filled", result
        assert result.fill_price == -2.91
        row = log._conn.execute(
            "SELECT final_status FROM submitted_orders",
        ).fetchone()
        assert row[0] == "filled"
    print("ladder entry: cancel/fill race books the fill")


# ---------- close ladder ----------

def test_close_ladder_prices_from_mark_mid():
    legs = [
        TradeLeg(100.0, "call", "buy", 1, "AAPL_100C"),
        TradeLeg(100.0, "put", "buy", 1, "AAPL_100P"),
    ]
    pos = OpenPosition(
        tradier_order_id=5001, symbol="AAPL",
        expiration=date(2026, 5, 15), direction="BUY",
        structure="straddle", legs=legs, entry_premium=4.08,
        entry_atm_iv=0.27, entry_predicted_iv=0.42, entry_divergence=0.15,
        entry_horizon_lower=10, entry_horizon_upper=21, entry_weight_lower=0.27,
        submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
    )
    # close_cash_flow +200 -> credit close, mid $2.00/contract
    mark = PositionMark(
        position=pos, current_legs=[], close_cash_flow=200.0, cost_to_close=0,
        pnl_dollars=0.0, pnl_pct_of_entry_premium=0.0, pnl_pct_of_max=float("nan"),
        delta=0, gamma=0, theta=0, vega=0, dte=10,
    )
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        client = _LadderFakeClient(fill_after_modifies=1, fill_price=-1.98)
        manager = _mk_manager(client, log)
        result = asyncio.run(manager.submit_close(
            position=pos, mark=mark, exit_trigger="profit_target",
        ))
        assert result.status == "filled", result
        assert client.place_price == 2.00, f"close placed at {client.place_price}, want mid"
        assert client.modify_prices == [1.98], client.modify_prices
        row = log._conn.execute(
            "SELECT arrival_mid FROM close_attempts",
        ).fetchone()
        assert row == (2.00,), row
    print("ladder close: placed at mark mid, conceded to 1.98, arrival_mid recorded")


def test_ladder_skips_noop_modifies_on_cheap_contracts():
    """A $0.30 mid rounds several steps to the same cent — those modifies are
    skipped rather than sent as no-ops."""
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        client = _LadderFakeClient(fill_after_modifies=None)
        manager = _mk_manager(client, log)
        status, fill, last_price = asyncio.run(manager._run_price_ladder(
            order_id=7777, base_mid=0.30, order_type="credit",
        ))
        assert status is None and fill is None
        # 0.30, 0.297->0.30 (skip), 0.294->0.29 (modify), 0.291->0.29 (skip)
        assert client.modify_prices == [0.29], client.modify_prices
        assert last_price == 0.29
    print("ladder: no-op modifies skipped on cheap contracts")


# ---------- entry time window ----------

def test_entry_window_boundaries():
    from main import _within_entry_window

    def _utc(h, m, month=7):
        # July = EDT (UTC-4); January = EST (UTC-5)
        return datetime(2026, month, 13, h, m, tzinfo=timezone.utc)

    assert not _within_entry_window(_utc(13, 44)), "9:44 ET is before the window"
    assert _within_entry_window(_utc(13, 45)), "9:45 ET opens the window"
    assert _within_entry_window(_utc(19, 30)), "15:30 ET is the last minute"
    assert not _within_entry_window(_utc(19, 31)), "15:31 ET is after the window"
    # Winter (EST): 9:45 ET = 14:45 UTC
    assert _within_entry_window(_utc(14, 45, month=1))
    assert not _within_entry_window(_utc(14, 44, month=1))
    print("entry_window: 9:45-15:30 ET boundaries verified across DST")


def main() -> int:
    test_entry_fills_at_mid_no_modifies()
    test_entry_ladder_walks_toward_far_side()
    test_entry_unfilled_cancels_and_is_retryable()
    test_entry_cancel_races_fill()
    test_close_ladder_prices_from_mark_mid()
    test_ladder_skips_noop_modifies_on_cheap_contracts()
    test_entry_window_boundaries()
    print("all price_ladder tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
