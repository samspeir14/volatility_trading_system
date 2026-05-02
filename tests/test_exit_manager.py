"""Unit tests for ExitManager — focus on the trigger priority logic.

Most tests mock out the predictor + scan so we only exercise _evaluate_one
and the threshold math. The full evaluate() with re-derived divergence is
exercised in the live test."""
import math
import sys
from datetime import date, datetime, timezone
from unittest import mock

from positions.exit_manager import EXIT_TRIGGER_PRIORITY, ExitManager
from positions.position_tracker import OpenPosition, PositionMark
from signals.signal_generator import TradeLeg


def _mk_iron_condor_position(*, entry_credit=13.55) -> OpenPosition:
    legs = [
        TradeLeg(210.0, "call", "sell", 1, "NVDA260522C00210000"),
        TradeLeg(210.0, "put", "sell", 1, "NVDA260522P00210000"),
        TradeLeg(230.0, "call", "buy", 1, "NVDA260522C00230000"),
        TradeLeg(190.0, "put", "buy", 1, "NVDA260522P00190000"),
    ]
    return OpenPosition(
        tradier_order_id=1, symbol="NVDA",
        expiration=date(2026, 5, 22), direction="SELL",
        structure="iron_condor", legs=legs, entry_premium=entry_credit,
        entry_atm_iv=0.40, entry_predicted_iv=0.32, entry_divergence=-0.08,
        entry_horizon_lower=21, entry_horizon_upper=21, entry_weight_lower=1.0,
        submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
    )


def _mk_long_straddle_position(*, entry_debit=4.08) -> OpenPosition:
    legs = [
        TradeLeg(100.0, "call", "buy", 1, "AAPL260515C00100000"),
        TradeLeg(100.0, "put", "buy", 1, "AAPL260515P00100000"),
    ]
    return OpenPosition(
        tradier_order_id=2, symbol="AAPL",
        expiration=date(2026, 5, 15), direction="BUY",
        structure="straddle", legs=legs, entry_premium=entry_debit,
        entry_atm_iv=0.27, entry_predicted_iv=0.42, entry_divergence=0.15,
        entry_horizon_lower=10, entry_horizon_upper=21, entry_weight_lower=0.27,
        submitted_at=datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc),
    )


def _mark(pos: OpenPosition, *, pnl_dollars: float, dte: int = 20) -> PositionMark:
    return PositionMark(
        position=pos, current_legs=[], close_cash_flow=0, cost_to_close=0,
        pnl_dollars=pnl_dollars, pnl_pct_of_entry_premium=pnl_dollars / (pos.entry_premium * 100),
        pnl_pct_of_max=float("nan"),
        delta=0, gamma=0, theta=0, vega=0, dte=dte,
    )


def _exit_mgr() -> ExitManager:
    return ExitManager(
        position_tracker=mock.MagicMock(),
        order_manager=mock.MagicMock(),
        predictors_by_horizon={5: mock.MagicMock(), 10: mock.MagicMock(), 21: mock.MagicMock()},
    )


# ---------- profit target ----------

def test_iron_condor_profit_target_at_50pct():
    pos = _mk_iron_condor_position(entry_credit=13.55)
    # 50% target = 0.50 × 13.55 × 100 = $677.50
    mark = _mark(pos, pnl_dollars=678.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=-0.08)  # thesis intact
    assert trigger == "profit_target"
    # Just below threshold: hold
    mark = _mark(pos, pnl_dollars=677.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=-0.08)
    assert trigger is None
    print("iron_condor profit_target: $677.50 cutoff verified")


def test_long_straddle_profit_target_at_100pct():
    pos = _mk_long_straddle_position(entry_debit=4.08)
    # 100% target = 1.0 × 4.08 × 100 = $408
    mark = _mark(pos, pnl_dollars=409.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=0.15)  # thesis intact
    assert trigger == "profit_target"
    print("long_straddle profit_target: $408 cutoff verified")


# ---------- stop loss ----------

def test_iron_condor_stop_loss_at_neg_100pct():
    pos = _mk_iron_condor_position(entry_credit=13.55)
    # -100% stop = -1.0 × 13.55 × 100 = -$1355
    mark = _mark(pos, pnl_dollars=-1356.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=-0.08)
    assert trigger == "stop_loss"
    print("iron_condor stop_loss: -$1355 cutoff verified")


def test_long_straddle_stop_loss_at_neg_50pct():
    pos = _mk_long_straddle_position(entry_debit=4.08)
    # -50% stop = -0.50 × 4.08 × 100 = -$204
    mark = _mark(pos, pnl_dollars=-205.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=0.15)
    assert trigger == "stop_loss"
    print("long_straddle stop_loss: -$204 cutoff verified")


# ---------- expiration proximity ----------

def test_expiration_proximity_long_straddle_dte_2():
    """Long straddles trigger expiration_proximity at dte<=2 (default).
    The proximity exit applies to long straddles only — IC wings cap risk."""
    pos = _mk_long_straddle_position()
    mark = _mark(pos, pnl_dollars=0.0, dte=2)
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=0.15)
    assert trigger == "expiration_proximity"
    # DTE=3 doesn't trigger
    mark = _mark(pos, pnl_dollars=0.0, dte=3)
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=0.15)
    assert trigger is None
    print("expiration_proximity (long_straddle): dte<=2 fires, dte=3 holds")


def test_expiration_proximity_skipped_for_iron_condor():
    """Iron condors do NOT trigger expiration_proximity — wings cap risk so
    we let them ride to expiration unless another trigger fires."""
    pos = _mk_iron_condor_position()
    # dte=2 (would fire for a straddle)
    mark = _mark(pos, pnl_dollars=0.0, dte=2)
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=-0.08)
    assert trigger is None, f"IC at dte=2 should hold, got {trigger}"
    # dte=0 (expiration day)
    mark = _mark(pos, pnl_dollars=0.0, dte=0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=-0.08)
    assert trigger is None, f"IC at dte=0 should hold, got {trigger}"
    print("expiration_proximity (iron_condor): never fires")


def test_min_dte_strictly_greater_than_expiration_proximity_dte():
    """Trade window must leave a buffer between open and forced close.
    Otherwise h=5 trades open at MIN_DTE and immediately exit on the next
    scan via expiration_proximity (the bug this guards against)."""
    from signals.signal_generator import MIN_DTE
    mgr = _exit_mgr()
    assert MIN_DTE > mgr._exp_dte, (
        f"MIN_DTE ({MIN_DTE}) must be > expiration_proximity_dte "
        f"({mgr._exp_dte}) so opened trades aren't immediately exit-eligible"
    )
    print(f"min_dte_invariant: MIN_DTE={MIN_DTE} > exp_dte={mgr._exp_dte} ✓")


# ---------- thesis reversal ----------

def test_thesis_reversal_fires_when_sign_flips_and_magnitude_clears():
    pos = _mk_iron_condor_position()  # entry_divergence = -0.08 (we sold premium)
    # Current divergence: positive AND |≥ 0.05| → flipped + magnitude OK
    mark = _mark(pos, pnl_dollars=0.0)
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=+0.06)
    assert trigger == "thesis_reversed"
    # Sign flipped but magnitude too small → no trigger
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=+0.001)
    assert trigger is None
    # Same sign as entry → no trigger
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=-0.20)
    assert trigger is None
    print("thesis_reversal: sign flip + magnitude floor verified")


# ---------- the user's flagged priority test ----------

def test_thesis_overrides_stop_loss():
    """User-flagged design directive: thesis reversal beats stop loss.
    Position is in -110% drawdown AND thesis flipped → trigger MUST be thesis_reversed."""
    pos = _mk_iron_condor_position(entry_credit=13.55)
    # P&L = -$1500 (well past -100% stop)
    mark = _mark(pos, pnl_dollars=-1500.0)
    # Thesis intact: stop_loss
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=-0.08)
    assert trigger == "stop_loss"
    # Thesis flipped: thesis_reversed (overrides stop)
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=+0.10)
    assert trigger == "thesis_reversed"
    print("PRIORITY: thesis_reversed overrides stop_loss when both fire")


def test_thesis_overrides_profit_target():
    """Same priority logic on the upside — if thesis flips while in profit, close
    rather than wait for the rest. The remaining edge is gone."""
    pos = _mk_iron_condor_position(entry_credit=13.55)
    # P&L = +$1000 (well past +50% profit target)
    mark = _mark(pos, pnl_dollars=1000.0)
    # Thesis intact: profit_target
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=-0.08)
    assert trigger == "profit_target"
    # Thesis flipped: thesis_reversed
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=+0.10)
    assert trigger == "thesis_reversed"
    print("PRIORITY: thesis_reversed overrides profit_target when both fire")


def test_priority_constant_matches_evaluation_order():
    """Sanity check: the EXIT_TRIGGER_PRIORITY tuple matches the order in the
    actual logic. If someone reorders one without the other, this catches it."""
    assert EXIT_TRIGGER_PRIORITY == (
        "thesis_reversed", "stop_loss", "expiration_proximity", "profit_target",
    )
    print(f"priority constant: {EXIT_TRIGGER_PRIORITY}")


def test_no_trigger_returns_hold():
    pos = _mk_iron_condor_position()
    mark = _mark(pos, pnl_dollars=0.0, dte=20)
    trigger, rationale = _exit_mgr()._evaluate_one(mark, current_divergence=-0.07)
    assert trigger is None
    assert rationale == "hold"
    print("no_trigger: returns ('hold')")


def test_current_divergence_none_skips_thesis_check():
    """If we can't compute current divergence (e.g., no chain data), thesis
    check should silently skip. Other triggers still evaluate normally."""
    pos = _mk_iron_condor_position()
    mark = _mark(pos, pnl_dollars=0.0, dte=20)
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=None)
    assert trigger is None  # nothing else fires either
    # P&L stop still works
    mark = _mark(pos, pnl_dollars=-2000.0, dte=20)
    trigger, _ = _exit_mgr()._evaluate_one(mark, current_divergence=None)
    assert trigger == "stop_loss"
    print("current_divergence None: thesis check skipped, P&L triggers still fire")


def main() -> int:
    test_iron_condor_profit_target_at_50pct()
    test_long_straddle_profit_target_at_100pct()
    test_iron_condor_stop_loss_at_neg_100pct()
    test_long_straddle_stop_loss_at_neg_50pct()
    test_expiration_proximity_long_straddle_dte_2()
    test_expiration_proximity_skipped_for_iron_condor()
    test_min_dte_strictly_greater_than_expiration_proximity_dte()
    test_thesis_reversal_fires_when_sign_flips_and_magnitude_clears()
    test_thesis_overrides_stop_loss()
    test_thesis_overrides_profit_target()
    test_priority_constant_matches_evaluation_order()
    test_no_trigger_returns_hold()
    test_current_divergence_none_skips_thesis_check()
    print("all exit_manager tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
