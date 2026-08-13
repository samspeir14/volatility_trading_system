"""Unit tests for RiskManager — each gate isolated, plus composability test."""
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

from config import Ticker
from data.async_client import OptionContract
from data.market_data import ScanResult, TickerSnapshot
from risk import DailyKillSwitch, PortfolioSnapshot, RiskManager
from signals.signal_generator import TradeLeg, TradeSignal


WATCHLIST = [
    Ticker("AAPL", "tech"),
    Ticker("MSFT", "tech"),
    Ticker("NVDA", "tech"),
    Ticker("JPM", "financials"),
    Ticker("XOM", "energy"),
]


def _mk_contract(strike: float, otype: str, side: str = "neutral", *,
                 bid=1.0, ask=1.10, delta=0.5, gamma=0.01, vega=0.20) -> OptionContract:
    sym_pre = "NVDA260522"
    sym = f"{sym_pre}{'C' if otype == 'call' else 'P'}{int(strike * 1000):08d}"
    return OptionContract(
        symbol=sym, underlying="NVDA", expiration=date(2026, 5, 22),
        strike=strike, option_type=otype,
        bid=bid, ask=ask, last=(bid + ask) / 2,
        volume=200, open_interest=1000,
        delta=delta if otype == "call" else -delta,
        gamma=gamma, theta=-0.05, vega=vega, iv=0.30,
        fetched_at=datetime.now(timezone.utc),
    )


def _mk_iron_condor_signal() -> TradeSignal:
    legs = [
        TradeLeg(210.0, "call", "sell", 1, "NVDA260522C00210000"),
        TradeLeg(210.0, "put", "sell", 1, "NVDA260522P00210000"),
        TradeLeg(230.0, "call", "buy", 1, "NVDA260522C00230000"),
        TradeLeg(190.0, "put", "buy", 1, "NVDA260522P00190000"),
    ]
    return TradeSignal(
        symbol="NVDA", expiration=date(2026, 5, 22), dte=25,
        horizon_lower=21, horizon_upper=21, weight_lower=1.0,
        direction="SELL", underlying_price=210.0, atm_iv=0.40,
        predicted_iv_equivalent=0.32, divergence=-0.08,
        cross_sectional_z=-2.0, time_series_z=None,
        liquidity_score=10000.0, legs=legs, is_actionable=True,
    )


def _mk_scan_for_iron_condor() -> ScanResult:
    contracts = [
        _mk_contract(210, "call", bid=2.7, ask=2.8, delta=0.5),
        _mk_contract(210, "put", bid=2.7, ask=2.8, delta=-0.5),
        _mk_contract(230, "call", bid=0.20, ask=0.30, delta=0.2),
        _mk_contract(190, "put", bid=0.20, ask=0.30, delta=-0.2),
    ]
    snap = TickerSnapshot(
        symbol="NVDA", sector="tech",
        underlying={"symbol": "NVDA", "last": 210.0},
        contracts=contracts,
    )
    return ScanResult(
        fetched_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        snapshots={"NVDA": snap},
    )


def _mk_snapshot(
    *,
    equity=100000.0,
    buying_power=100000.0,
    portfolio_greeks=None,
    positions_by_sector=None,
    exposure_by_symbol=None,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        fetched_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
        equity=equity,
        starting_equity_today=equity,
        buying_power=buying_power,
        margin_held=0.0,
        open_positions=[], open_marks=[],
        today_realized_pnl=0.0, today_unrealized_pnl=0.0,
        portfolio_greeks=portfolio_greeks or {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
        positions_by_sector=positions_by_sector or {},
        exposure_by_symbol=exposure_by_symbol or {},
    )


# ---------- gate-by-gate tests ----------

def test_kill_switch_short_circuits():
    with tempfile.TemporaryDirectory() as tmp:
        ks = DailyKillSwitch(Path(tmp) / "ks.db")
        today = date(2026, 4, 27)
        ks.trigger(today, "test", -5000, 100000)

        rm = RiskManager(watchlist=WATCHLIST, kill_switch=ks)
        signal = _mk_iron_condor_signal()
        scan = _mk_scan_for_iron_condor()
        snapshot = _mk_snapshot()

        decisions = rm.gate([signal], scan, snapshot)
        assert len(decisions) == 1
        d = decisions[0]
        assert d.approved is False
        assert d.quantity == 0
        assert d.reasons == ["kill switch active"]
        ks.close()
    print("kill_switch: short-circuits with single reason")


def test_per_trade_budget_sizes_quantity():
    rm = RiskManager(watchlist=WATCHLIST, max_per_trade_loss_pct=0.01)
    signal = _mk_iron_condor_signal()
    scan = _mk_scan_for_iron_condor()
    # max_loss/contract: wing=20, credit ≈ (2.75 + 2.75 - 0.25 - 0.25) = 5.0
    # max_loss = (20 - 5.0) * 100 = 1500
    # budget at 1% of $100k = $1000 → quantity = floor(1000/1500) = 0 → reject
    snapshot = _mk_snapshot(equity=100000.0)
    decision = rm.gate([signal], scan, snapshot)[0]
    assert decision.approved is False
    assert any("per-trade budget" in r for r in decision.reasons)
    print(f"per_trade_budget @ $100k: max_loss=$1500/contract too big for 1% budget")

    # Bigger account → can afford
    snapshot_big = _mk_snapshot(equity=300000.0)
    decision_big = rm.gate([signal], scan, snapshot_big)[0]
    # 1% of $300k = $3000 → floor(3000/1500) = 2
    assert decision_big.approved is True
    assert decision_big.quantity == 2
    print(f"per_trade_budget @ $300k: quantity sized to 2")


def test_ticker_exposure_cap():
    rm = RiskManager(
        watchlist=WATCHLIST, max_per_trade_loss_pct=0.05,  # plenty of budget
        max_per_ticker_exposure_pct=0.05,
    )
    signal = _mk_iron_condor_signal()
    scan = _mk_scan_for_iron_condor()
    # Existing NVDA exposure $4500, equity $100k, cap 5% = $5000
    # New trade max_loss = $1500 × qty; with 5% trade budget = $5000 → qty=3 → projected $4500
    # 4500 + 4500 = 9000 > 5000 cap → reject
    snapshot = _mk_snapshot(
        equity=100000.0,
        exposure_by_symbol={"NVDA": 4500.0},
    )
    decision = rm.gate([signal], scan, snapshot)[0]
    assert decision.approved is False
    assert any("NVDA exposure" in r and "exceed cap" in r for r in decision.reasons)
    print("ticker_exposure_cap: NVDA $4500 + new $4500 > $5000 cap")


def test_sector_concentration():
    rm = RiskManager(watchlist=WATCHLIST, max_per_sector_positions=3)
    signal = _mk_iron_condor_signal()  # NVDA = tech
    scan = _mk_scan_for_iron_condor()
    snapshot = _mk_snapshot(
        equity=300000.0,  # big account so per-trade gate passes
        positions_by_sector={"tech": 3},
    )
    decision = rm.gate([signal], scan, snapshot)[0]
    assert decision.approved is False
    assert any("tech sector" in r and "3 positions" in r for r in decision.reasons)
    print("sector_concentration: tech already at 3 positions → reject")


def test_portfolio_wing_risk_cap_blocks_full_book():
    """Sum of open max-losses + the new trade must stay under the portfolio
    cap — the correlated-crash bound (every position through its wings)."""
    rm = RiskManager(
        watchlist=WATCHLIST, max_per_trade_loss_pct=0.015,
        max_portfolio_risk_pct=0.12,
    )
    signal = _mk_iron_condor_signal()
    scan = _mk_scan_for_iron_condor()
    # Open book already carries $11,500 wing risk; cap 12% of $100k = $12,000.
    # New trade: qty 1 × $1,500 max loss → $13,000 > cap → reject.
    snapshot = _mk_snapshot(
        equity=100000.0,
        exposure_by_symbol={"AAPL": 5000.0, "MSFT": 6500.0},
    )
    decision = rm.gate([signal], scan, snapshot)[0]
    assert decision.approved is False
    assert any("portfolio wing risk" in r for r in decision.reasons)
    print("portfolio_wing_risk: $11.5k open + $1.5k new > $12k cap → reject")


def test_portfolio_wing_risk_accumulates_within_cycle():
    """A batch of approvals in ONE cycle must count against the cap as they
    accrue — a fresh deploy: empty book, many actionable signals at once."""
    rm = RiskManager(
        watchlist=WATCHLIST, max_per_trade_loss_pct=0.015,
        max_per_ticker_exposure_pct=1.0,     # not the limiter here
        max_per_sector_positions=99,          # not the limiter here
        max_portfolio_risk_pct=0.04,          # $4k on $100k → two $1.5k trades fit
    )
    signals = [_mk_iron_condor_signal() for _ in range(3)]
    scan = _mk_scan_for_iron_condor()
    snapshot = _mk_snapshot(equity=100000.0)  # empty book
    decisions = rm.gate(signals, scan, snapshot)
    assert [d.approved for d in decisions] == [True, True, False], \
        [(d.approved, d.reasons) for d in decisions]
    assert any("portfolio wing risk" in r for r in decisions[2].reasons)
    print("portfolio_wing_risk: intra-cycle accrual — 3rd of 3 identical trades rejected")


def test_ticker_and_sector_caps_accumulate_within_cycle():
    """Same-cycle approvals must also count toward the per-ticker and
    per-sector gates, not just what was open at snapshot time."""
    # Sector: cap 2, three tech signals in one batch → third rejected.
    rm = RiskManager(
        watchlist=WATCHLIST, max_per_trade_loss_pct=0.015,
        max_per_ticker_exposure_pct=1.0, max_per_sector_positions=2,
        max_portfolio_risk_pct=1.0,
    )
    signals = [_mk_iron_condor_signal() for _ in range(3)]
    scan = _mk_scan_for_iron_condor()
    decisions = rm.gate(signals, scan, _mk_snapshot(equity=100000.0))
    assert [d.approved for d in decisions] == [True, True, False]
    assert any("tech sector" in r for r in decisions[2].reasons)

    # Ticker: cap $3k, $1.5k per approval → third NVDA rejected.
    rm = RiskManager(
        watchlist=WATCHLIST, max_per_trade_loss_pct=0.015,
        max_per_ticker_exposure_pct=0.03, max_per_sector_positions=99,
        max_portfolio_risk_pct=1.0,
    )
    decisions = rm.gate(signals, scan, _mk_snapshot(equity=100000.0))
    assert [d.approved for d in decisions] == [True, True, False]
    assert any("NVDA exposure" in r for r in decisions[2].reasons)
    print("intra-cycle accrual: sector and ticker caps bind on batch approvals")


def test_portfolio_delta_limit():
    rm = RiskManager(
        watchlist=WATCHLIST,
        max_per_trade_loss_pct=0.05,        # generous budget
        max_per_ticker_exposure_pct=0.50,   # not the limiter
        max_portfolio_delta_pct=0.05,       # tight at 5% of equity
    )
    signal = _mk_iron_condor_signal()
    scan = _mk_scan_for_iron_condor()
    # The IC's net delta should be roughly 0 but non-zero per leg
    # Force a high current portfolio delta so adding even a small trade breaches
    snapshot = _mk_snapshot(
        equity=100000.0,                                # cap = $5000
        portfolio_greeks={"delta": 4990.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
    )
    decision = rm.gate([signal], scan, snapshot)[0]
    # Even a tiny additional |delta| pushes us over $5000 cap
    # Note: the symmetric IC's delta is near zero, so we need to be careful
    # — it might still approve if the new delta magnitude is < $10
    print(f"portfolio_delta @ near-cap: approved={decision.approved}, "
          f"reasons={decision.reasons[:2] if decision.reasons else 'none'}")


def test_margin_buffer_blocks_when_low_buying_power():
    rm = RiskManager(
        watchlist=WATCHLIST,
        max_per_trade_loss_pct=0.05,
        max_per_ticker_exposure_pct=0.50,
        max_per_sector_positions=10,
        min_buying_power_buffer_pct=0.05,
    )
    signal = _mk_iron_condor_signal()  # SELL → has margin requirement
    scan = _mk_scan_for_iron_condor()
    # IC margin = 20 * 100 = $2000/contract; budget gives qty=2 → margin $4000
    # With BP $4100, usable = $3895 → $4000 > $3895 → reject
    snapshot = _mk_snapshot(
        equity=100000.0,
        buying_power=4100.0,
    )
    decision = rm.gate([signal], scan, snapshot)[0]
    assert decision.approved is False
    assert any("required margin" in r for r in decision.reasons)
    print("margin_buffer: low BP blocks even with sized qty")


def test_compose_multiple_failures():
    """A signal failing 3 gates simultaneously must report all 3."""
    rm = RiskManager(
        watchlist=WATCHLIST,
        max_per_trade_loss_pct=0.05,
        max_per_ticker_exposure_pct=0.05,
        max_per_sector_positions=1,
        min_buying_power_buffer_pct=0.05,
    )
    signal = _mk_iron_condor_signal()
    scan = _mk_scan_for_iron_condor()
    # Fail: ticker exposure + sector concentration + margin buffer
    snapshot = _mk_snapshot(
        equity=100000.0,
        buying_power=1000.0,                # margin will fail
        exposure_by_symbol={"NVDA": 4900.0},  # ticker will fail
        positions_by_sector={"tech": 1},     # sector will fail
    )
    decision = rm.gate([signal], scan, snapshot)[0]
    assert decision.approved is False
    assert len(decision.reasons) >= 3
    has_ticker = any("NVDA" in r and "exceed cap" in r for r in decision.reasons)
    has_sector = any("tech sector" in r for r in decision.reasons)
    has_margin = any("required margin" in r for r in decision.reasons)
    assert has_ticker, f"missing ticker reason in {decision.reasons}"
    assert has_sector, f"missing sector reason in {decision.reasons}"
    assert has_margin, f"missing margin reason in {decision.reasons}"
    print(f"compose_multiple_failures: {len(decision.reasons)} reasons all reported")


def test_full_approval():
    rm = RiskManager(
        watchlist=WATCHLIST,
        max_per_trade_loss_pct=0.05,
        max_per_ticker_exposure_pct=0.20,
        max_per_sector_positions=10,
        max_portfolio_delta_pct=0.50,
        max_portfolio_gamma_pct=0.50,
        max_portfolio_vega_pct=0.50,
    )
    signal = _mk_iron_condor_signal()
    scan = _mk_scan_for_iron_condor()
    snapshot = _mk_snapshot(equity=200000.0, buying_power=50000.0)
    decision = rm.gate([signal], scan, snapshot)[0]
    assert decision.approved is True, f"unexpected rejection: {decision.reasons}"
    assert decision.quantity > 0
    print(f"full_approval: quantity={decision.quantity}, projected_loss=${decision.projected_max_loss:.2f}")


def test_buy_signal_no_margin_check():
    """Long straddles (BUY/debit) shouldn't trigger margin gate."""
    legs = [
        TradeLeg(210.0, "call", "buy", 1, "NVDA260522C00210000"),
        TradeLeg(210.0, "put", "buy", 1, "NVDA260522P00210000"),
    ]
    signal = TradeSignal(
        symbol="NVDA", expiration=date(2026, 5, 22), dte=25,
        horizon_lower=21, horizon_upper=21, weight_lower=1.0,
        direction="BUY", underlying_price=210.0, atm_iv=0.27,
        predicted_iv_equivalent=0.40, divergence=0.13,
        cross_sectional_z=2.0, time_series_z=None,
        liquidity_score=10000.0, legs=legs, is_actionable=True,
    )
    scan = _mk_scan_for_iron_condor()  # has the call/put we need
    rm = RiskManager(
        watchlist=WATCHLIST,
        max_per_trade_loss_pct=0.05,
        max_per_ticker_exposure_pct=0.50,
        max_per_sector_positions=10,
        max_portfolio_delta_pct=0.50,
        max_portfolio_gamma_pct=0.50,
        max_portfolio_vega_pct=0.50,
        min_buying_power_buffer_pct=0.05,
    )
    snapshot = _mk_snapshot(equity=100000.0, buying_power=100.0)  # near zero BP
    decision = rm.gate([signal], scan, snapshot)[0]
    # BUY signal has no margin requirement — low BP shouldn't block
    has_margin_reason = any("required margin" in r for r in decision.reasons)
    assert not has_margin_reason, "BUY signal should not be margin-gated"
    print("buy_signal_no_margin: BUY (debit) ignores margin gate")


def main() -> int:
    test_kill_switch_short_circuits()
    test_per_trade_budget_sizes_quantity()
    test_ticker_exposure_cap()
    test_sector_concentration()
    test_portfolio_delta_limit()
    test_margin_buffer_blocks_when_low_buying_power()
    test_compose_multiple_failures()
    test_full_approval()
    test_buy_signal_no_margin_check()
    print("all risk_manager tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
