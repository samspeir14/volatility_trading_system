"""Tests for the small account profile: MODEL_PIPELINE/ACCOUNT_PROFILE/
PER_CONTRACT_FEE parsing, calibration selection, the small watchlist, $10k
sizing, the min-credit floor in SignalGenerator, and fee netting on realized
P&L."""
import os
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

import main as trader_main
from config import SMALL_WATCHLIST_PATH, Settings, Ticker, load_settings, load_watchlist
from data.async_client import OptionContract
from data.market_data import ScanResult, TickerSnapshot
from execution import OrderLog, OrderManager
from risk import PortfolioSnapshot, RiskManager
from signals import SignalGenerator
from signals.signal_generator import TradeLeg, TradeSignal


# ---------- settings parsing ----------

def _settings_env(**overrides):
    """Fully-controlled env for load_settings(): only the required Tradier
    credentials plus explicit overrides. clear=True wipes leftovers
    (STRATEGY_MODE, PER_CONTRACT_FEE, TRADIER_ENV, ...) from the outer shell;
    load_dotenv() is a no-op here (no .env in the repo, only .env.example)."""
    env = {"TRADIER_API_KEY": "fake-key", "TRADIER_ACCOUNT_ID": "VA00000000"}
    env.update(overrides)
    return mock.patch.dict(os.environ, env, clear=True)


def test_pipeline_and_profile_parsing():
    with _settings_env():
        s = load_settings()
    assert (s.model_pipeline, s.account_profile) == ("h1", "standard")

    with _settings_env(MODEL_PIPELINE="legacy", ACCOUNT_PROFILE="small"):
        s = load_settings()
    assert (s.model_pipeline, s.account_profile) == ("legacy", "small")
    print("pipeline/profile: defaults (h1, standard); env overrides parsed")


def test_pipeline_and_profile_invalid_raise():
    with _settings_env(MODEL_PIPELINE="bogus"):
        try:
            load_settings()
        except ValueError as e:
            assert "MODEL_PIPELINE" in str(e)
        else:
            raise AssertionError("MODEL_PIPELINE=bogus should raise ValueError")
    with _settings_env(ACCOUNT_PROFILE="tiny"):
        try:
            load_settings()
        except ValueError as e:
            assert "ACCOUNT_PROFILE" in str(e)
        else:
            raise AssertionError("ACCOUNT_PROFILE=tiny should raise ValueError")
    print("pipeline/profile: invalid values rejected")


def test_strategy_mode_env_now_raises():
    """The old knob must fail LOUDLY, not silently default — a live box with
    STRATEGY_MODE still in .env needs to be reconfigured, not guessed at."""
    with _settings_env(STRATEGY_MODE="harvest"):
        try:
            load_settings()
        except ValueError as e:
            msg = str(e)
            assert "STRATEGY_MODE" in msg and "MODEL_PIPELINE" in msg
        else:
            raise AssertionError("STRATEGY_MODE should raise ValueError")
    print("strategy_mode: legacy env var raises with migration hint")


def test_vrp_z_knobs_must_be_positive_magnitudes():
    """VRP_Z_BUY=-1.25 (the natural misreading of "BUY needs z <= -1.25")
    would flip the gate to z <= +1.25 — reject it loudly."""
    with _settings_env(VRP_Z_BUY="-1.25"):
        try:
            load_settings()
        except ValueError as e:
            assert "VRP_Z" in str(e)
        else:
            raise AssertionError("negative VRP_Z_BUY should raise")
    with _settings_env(VRP_Z_SELL="0"):
        try:
            load_settings()
        except ValueError:
            pass
        else:
            raise AssertionError("zero VRP_Z_SELL should raise")
    print("vrp_z_knobs: non-positive magnitudes rejected")


def test_per_contract_fee_parsing():
    with _settings_env(PER_CONTRACT_FEE="0.45"):
        assert load_settings().per_contract_fee == 0.45
    with _settings_env():  # unset → default
        assert load_settings().per_contract_fee == 0.0
    with _settings_env(PER_CONTRACT_FEE=""):  # blank → default
        assert load_settings().per_contract_fee == 0.0
    # Unlike the operational knobs, a malformed fee must raise, not silently
    # run fee-free: it changes accounting, and 0.0 is log-identical to paper.
    for bad in ("not-a-number", "$0.45", "0,45"):
        with _settings_env(PER_CONTRACT_FEE=bad):
            try:
                load_settings()
            except ValueError as e:
                assert "PER_CONTRACT_FEE" in str(e)
            else:
                raise AssertionError(f"PER_CONTRACT_FEE={bad!r} should raise")
    print("per_contract_fee: 0.45 parsed; unset/blank → 0.0; malformed raises")


def test_per_contract_fee_negative_raises():
    with _settings_env(PER_CONTRACT_FEE="-0.1"):
        try:
            load_settings()
        except ValueError as e:
            assert "PER_CONTRACT_FEE" in str(e)
        else:
            raise AssertionError("negative PER_CONTRACT_FEE should raise ValueError")
    print("per_contract_fee: -0.1 rejected")


# ---------- calibration wiring ----------

def _mk_settings(**overrides) -> Settings:
    return Settings(
        api_key="fake", account_id="VA00000000",
        base_url="https://example.invalid/v1", env="sandbox",
        **overrides,
    )


def test_calibration_selected_by_profile():
    assert trader_main._calibration(_mk_settings(account_profile="small")) \
        is trader_main.CALIBRATION_SMALL
    assert trader_main._calibration(_mk_settings(account_profile="standard")) \
        is trader_main.CALIBRATION_STANDARD
    # Default Settings (no profile set) must land on standard too.
    assert trader_main._calibration(_mk_settings()) is trader_main.CALIBRATION_STANDARD
    print("calibration: profile 'small' → CALIBRATION_SMALL, else CALIBRATION_STANDARD")


def test_calibration_standard_preserves_live_values():
    """Regression guard: the standard profile must keep the exact values that
    were hardcoded in main.py before the calibration refactor."""
    c = trader_main.CALIBRATION_STANDARD
    assert c.max_per_trade_loss_pct == 0.015
    assert c.max_per_ticker_exposure_pct == 0.05
    assert c.max_per_sector_positions == 4
    assert c.max_portfolio_risk_pct == 0.20
    assert c.max_portfolio_delta_pct == 0.05
    assert c.max_portfolio_gamma_pct == 0.01
    assert c.max_portfolio_vega_pct == 0.05
    assert c.daily_loss_kill_switch_pct == -0.05
    assert c.min_buying_power_buffer_pct == 0.05
    assert c.max_premium_per_trade == 5000.0
    assert c.min_credit == 0.0
    assert c.max_contracts_per_trade == 10  # historical RiskManager cap
    print("calibration_standard: all pre-refactor values preserved")


def test_calibration_small_values():
    c = trader_main.CALIBRATION_SMALL
    assert c.max_per_trade_loss_pct == 0.025
    assert c.max_per_ticker_exposure_pct == 0.05
    assert c.max_per_sector_positions == 4
    assert c.max_portfolio_risk_pct == 0.12
    assert c.max_portfolio_delta_pct == 0.05
    assert c.max_portfolio_gamma_pct == 0.10
    assert c.max_portfolio_vega_pct == 0.05
    assert c.daily_loss_kill_switch_pct == -0.05
    assert c.min_buying_power_buffer_pct == 0.05
    assert c.max_premium_per_trade == 500.0
    assert c.min_credit == 0.25
    # Orders always submit 1-lot; capping RiskManager's sized quantity at 1
    # keeps committed-risk/Greek/margin projections equal to what trades.
    assert c.max_contracts_per_trade == 1
    print("calibration_small: 2.5%/12%/gamma 10%/$500 premium/$0.25 credit floor")


# ---------- small watchlist ----------

def test_small_watchlist_loads():
    tickers = load_watchlist(SMALL_WATCHLIST_PATH)
    assert len(tickers) == 19, f"expected 19 names, got {len(tickers)}"
    assert all(isinstance(t, Ticker) for t in tickers)
    assert all(t.symbol and t.sector for t in tickers), "symbol+sector required"
    symbols = [t.symbol for t in tickers]
    assert len(set(symbols)) == len(symbols), "duplicate symbols in watchlist_small"
    print(f"small_watchlist: 19 unique names across "
          f"{len({t.sector for t in tickers})} sectors")


# ---------- sizing at $10k equity ----------

_SMALL_WATCHLIST = [Ticker("NVDA", "tech")]


def _mk_contract(strike: float, otype: str, *,
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


def _mk_condor_signal(wing: float) -> TradeSignal:
    legs = [
        TradeLeg(210.0, "call", "sell", 1, f"NVDA260522C{int(210.0 * 1000):08d}"),
        TradeLeg(210.0, "put", "sell", 1, f"NVDA260522P{int(210.0 * 1000):08d}"),
        TradeLeg(210.0 + wing, "call", "buy", 1, f"NVDA260522C{int((210.0 + wing) * 1000):08d}"),
        TradeLeg(210.0 - wing, "put", "buy", 1, f"NVDA260522P{int((210.0 - wing) * 1000):08d}"),
    ]
    return TradeSignal(
        symbol="NVDA", expiration=date(2026, 5, 22), dte=8,
        horizon_lower=5, horizon_upper=10, weight_lower=0.4,
        direction="SELL", underlying_price=210.0, atm_iv=0.40,
        predicted_iv_equivalent=0.32, divergence=-0.08,
        cross_sectional_z=-2.0, time_series_z=None,
        liquidity_score=10000.0, legs=legs, is_actionable=True,
    )


def _mk_condor_scan(wing: float, wing_bid: float, wing_ask: float) -> ScanResult:
    contracts = [
        _mk_contract(210, "call", bid=0.50, ask=0.60, delta=0.5),
        _mk_contract(210, "put", bid=0.50, ask=0.60, delta=0.5),
        _mk_contract(210 + wing, "call", bid=wing_bid, ask=wing_ask, delta=0.2),
        _mk_contract(210 - wing, "put", bid=wing_bid, ask=wing_ask, delta=0.2),
    ]
    snap = TickerSnapshot(
        symbol="NVDA", sector="tech",
        underlying={"symbol": "NVDA", "last": 210.0},
        contracts=contracts,
    )
    return ScanResult(
        fetched_at=datetime(2026, 5, 14, 16, 0, tzinfo=timezone.utc),
        snapshots={"NVDA": snap},
    )


def _mk_snapshot_10k() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        fetched_at=datetime(2026, 5, 14, 16, 0, tzinfo=timezone.utc),
        equity=10_000.0,
        starting_equity_today=10_000.0,
        buying_power=10_000.0,
        margin_held=0.0,
        open_positions=[], open_marks=[],
        today_realized_pnl=0.0, today_unrealized_pnl=0.0,
        portfolio_greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
        positions_by_sector={}, exposure_by_symbol={},
    )


def test_small_profile_per_trade_budget_at_10k():
    """CALIBRATION_SMALL's 2.5% of $10k = $250 per-trade budget: a ~$220
    max-loss condor gets 1 lot, a ~$300 one is rejected on that budget."""
    cal = trader_main.CALIBRATION_SMALL
    rm = RiskManager(
        watchlist=_SMALL_WATCHLIST,
        max_per_trade_loss_pct=cal.max_per_trade_loss_pct,  # 0.025
    )
    snapshot = _mk_snapshot_10k()

    # 3-wide wings, credit mids (0.55+0.55) − (0.15+0.15) = $0.80
    # → max loss (3 − 0.80) × 100 = $220 ≤ $250 → 1 contract approved.
    decision = rm.gate(
        [_mk_condor_signal(wing=3.0)],
        _mk_condor_scan(wing=3.0, wing_bid=0.10, wing_ask=0.20),
        snapshot,
    )[0]
    assert decision.approved is True, f"unexpected rejection: {decision.reasons}"
    assert decision.quantity == 1
    assert abs(decision.projected_max_loss - 220.0) < 0.01

    # 4-wide wings, credit (0.55+0.55) − (0.05+0.05) = $1.00
    # → max loss (4 − 1.00) × 100 = $300 > $250 → rejected.
    decision = rm.gate(
        [_mk_condor_signal(wing=4.0)],
        _mk_condor_scan(wing=4.0, wing_bid=0.00, wing_ask=0.10),
        snapshot,
    )[0]
    assert decision.approved is False
    assert any("per-trade budget" in r for r in decision.reasons), decision.reasons
    print("small_sizing @ $10k: $220 condor → 1 lot, $300 condor → per-trade reject")


# ---------- min_credit floor in SignalGenerator ----------

class _ConstPredictor:
    """Returns a fixed daily RV for every horizon (annualizes to ~0.19)."""
    def predict_forward_rv(self, returns_history, X_row=None):
        return 0.012


# Alternating ±1.2% daily returns → trail63 ≈ 0.012 × √252 ≈ 0.19 annualized,
# close to atm_iv=0.20 below so the harvest extreme-spread veto stays quiet.
_HARVEST_RETURNS = pd.Series([0.012, -0.012] * 50)


def _thin_condor_chain(sym: str, expiration: date, *, iv=0.20):
    """ATM call+put and OTM wings, all liquid, priced so the condor's mid
    credit is exactly $0.20: (0.30 + 0.30) − (0.20 + 0.20)."""
    def _c(strike, otype, bid, ask, delta):
        return OptionContract(
            symbol=f"{sym}_{otype[0].upper()}{strike:.0f}_{expiration.isoformat()}",
            underlying=sym, expiration=expiration, strike=strike, option_type=otype,
            bid=bid, ask=ask, last=(bid + ask) / 2, volume=200, open_interest=1000,
            delta=delta, gamma=0.01, theta=-0.05, vega=0.2, iv=iv,
            fetched_at=datetime.now(timezone.utc),
        )
    return [
        _c(100, "call", 0.29, 0.31, 0.5),
        _c(100, "put", 0.29, 0.31, -0.5),
        _c(110, "call", 0.19, 0.21, 0.2),   # OTM wing
        _c(90, "put", 0.19, 0.21, -0.2),    # OTM wing
    ]


def _thin_condor_generate(min_credit: float):
    exp = date(2026, 6, 12)  # DTE 11 from the scan date — inside harvest window
    snap = TickerSnapshot(symbol="X", sector="tech",
                          underlying={"symbol": "X", "last": 100.0},
                          contracts=_thin_condor_chain("X", exp))
    scan = ScanResult(fetched_at=datetime(2026, 6, 1, 16, 0, tzinfo=timezone.utc),
                      snapshots={"X": snap})
    predictors = {h: _ConstPredictor() for h in (5, 10, 21)}
    feature_rows = {"X": pd.DataFrame({"a": [0.0]}, index=[date(2026, 5, 29)])}
    returns_by_symbol = {"X": _HARVEST_RETURNS}
    gen = SignalGenerator(predictors_by_horizon=predictors, history_store=None,
                          cross_sectional_z_threshold=0.0, min_credit=min_credit)
    return gen.generate(scan, feature_rows, returns_by_symbol, top_n=10)


def test_min_credit_demotes_thin_condor():
    """Mid credit $0.20 < floor $0.25 → signal demoted with the floor note,
    exactly like other _build_legs failures."""
    actionable, all_signals = _thin_condor_generate(min_credit=0.25)
    by = {s.symbol: s for s in all_signals}
    assert "X" in by, "signal missing entirely"
    sig = by["X"]
    assert not sig.is_actionable, "thin condor should be demoted"
    assert sig.legs == [], "demoted signal must carry no legs"
    assert "below floor" in sig.diagnostic_notes, sig.diagnostic_notes
    assert "$0.20" in sig.diagnostic_notes and "$0.25" in sig.diagnostic_notes, \
        sig.diagnostic_notes
    assert "X" not in [s.symbol for s in actionable]
    print(f"min_credit: thin condor demoted — '{sig.diagnostic_notes}'")


def test_min_credit_zero_leaves_behavior_unchanged():
    """Same chain with min_credit=0.0 builds the condor fine."""
    actionable, all_signals = _thin_condor_generate(min_credit=0.0)
    by = {s.symbol: s for s in actionable}
    assert "X" in by, f"expected actionable condor, got {[(s.symbol, s.diagnostic_notes) for s in all_signals]}"
    assert by["X"].direction == "SELL"
    assert len(by["X"].legs) == 4
    print("min_credit: 0.0 → same thin condor builds 4 legs, actionable")


def test_min_credit_negative_rejected():
    predictors = {h: _ConstPredictor() for h in (5, 10, 21)}
    try:
        SignalGenerator(predictors_by_horizon=predictors, min_credit=-0.1)
    except ValueError as e:
        assert "min_credit" in str(e)
    else:
        raise AssertionError("negative min_credit should raise ValueError")
    print("min_credit: -0.1 rejected at construction")


# ---------- fee netting on close P&L ----------

def test_close_pnl_fee_netting_both_branches():
    """_compute_close_realized_pnl subtracts fees in both the normal and the
    fallback (fill_price=None) branch; delta vs fee-free is exactly the fee."""
    fn = OrderManager._compute_close_realized_pnl
    fees = 0.45 * 2 * 4  # $0.45/contract × open+close sides × 4-leg 1-lot condor

    # Normal branch: short condor, $1.20 credit in, $0.50 debit to close
    # → gross +$120 − $50 = +$70.
    gross = fn(is_long=False, entry_premium=1.20, order_type="debit",
               fill_price=0.50, fallback_pnl=0.0)
    net = fn(is_long=False, entry_premium=1.20, order_type="debit",
             fill_price=0.50, fallback_pnl=0.0, fees=fees)
    assert abs(gross - 70.0) < 1e-9
    assert abs(net - (70.0 - fees)) < 1e-9
    assert abs((gross - net) - fees) < 1e-9

    # Fallback branch: no fill price → mark estimate minus fees.
    gross_fb = fn(is_long=False, entry_premium=1.20, order_type="debit",
                  fill_price=None, fallback_pnl=120.0)
    net_fb = fn(is_long=False, entry_premium=1.20, order_type="debit",
                fill_price=None, fallback_pnl=120.0, fees=fees)
    assert abs(gross_fb - 120.0) < 1e-9
    assert abs(net_fb - (120.0 - fees)) < 1e-9
    assert abs((gross_fb - net_fb) - fees) < 1e-9
    print(f"close_pnl fees: both branches net out ${fees:.2f} exactly")


def test_order_manager_wires_fee_from_settings():
    with tempfile.TemporaryDirectory() as tmp:
        log = OrderLog(Path(tmp) / "log.db")
        manager = OrderManager(
            client=mock.AsyncMock(), order_log=log,
            settings=_mk_settings(per_contract_fee=0.45),
        )
        assert manager._fee_per_contract == 0.45
        log.close()
    print("order_manager: settings.per_contract_fee → _fee_per_contract")


def main() -> int:
    test_pipeline_and_profile_parsing()
    test_pipeline_and_profile_invalid_raise()
    test_strategy_mode_env_now_raises()
    test_vrp_z_knobs_must_be_positive_magnitudes()
    test_per_contract_fee_parsing()
    test_per_contract_fee_negative_raises()
    test_calibration_selected_by_profile()
    test_calibration_standard_preserves_live_values()
    test_calibration_small_values()
    test_small_watchlist_loads()
    test_small_profile_per_trade_budget_at_10k()
    test_min_credit_demotes_thin_condor()
    test_min_credit_zero_leaves_behavior_unchanged()
    test_min_credit_negative_rejected()
    test_close_pnl_fee_netting_both_branches()
    test_order_manager_wires_fee_from_settings()
    print("all small_harvest tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
