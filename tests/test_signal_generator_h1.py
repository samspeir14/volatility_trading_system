"""End-to-end tests for the h=1 signal flow: direction set by the VRP z-gate,
hard SELL divergence cap, VIX backwardation veto, cost gate, blocked_by
labels, term-projected forecast, and |z| ranking with per-symbol dedupe."""
import math
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data.async_client import OptionContract
from data.market_data import ScanResult, TickerSnapshot
from model.term_structure import project_term_vol
from signals import DivergenceHistory, SignalGenerator

TODAY = date(2026, 6, 1)
EXP = date(2026, 6, 12)          # DTE 11
B_T = math.log(0.012)            # baseline log daily vol → ~0.19 annualized
PHI = 0.90


class _FixedH1:
    """Deviation predictor returning a fixed dev_hat regardless of features."""
    def __init__(self, dev: float = 0.0):
        self.dev = dev

    def predict_deviation(self, X_row: pd.DataFrame) -> float:
        return self.dev


def _chain(sym, expiration, atm_iv):
    def _c(strike, otype, bid, ask, vega, delta):
        return OptionContract(
            symbol=f"{sym}_{otype[0].upper()}{strike:.0f}_{expiration.isoformat()}",
            underlying=sym, expiration=expiration, strike=strike, option_type=otype,
            bid=bid, ask=ask, last=(bid + ask) / 2, volume=200, open_interest=1000,
            delta=delta, gamma=0.01, theta=-0.05, vega=vega, iv=atm_iv,
            fetched_at=datetime.now(timezone.utc),
        )
    return [
        _c(100, "call", 1.00, 1.04, 0.20, 0.5),
        _c(100, "put", 0.92, 0.96, 0.20, -0.5),
        _c(110, "call", 0.20, 0.21, 0.10, 0.2),
        _c(90, "put", 0.20, 0.21, 0.10, -0.2),
    ]


def _scan(rows, expirations=None):
    """rows: [(symbol, atm_iv)]"""
    expirations = expirations or [EXP]
    snapshots = {
        sym: TickerSnapshot(
            symbol=sym, sector="tech",
            underlying={"symbol": sym, "last": 100.0},
            contracts=[c for e in expirations for c in _chain(sym, e, iv)],
        )
        for sym, iv in rows
    }
    return ScanResult(
        fetched_at=datetime(2026, 6, 1, 16, 0, tzinfo=timezone.utc),
        snapshots=snapshots,
    )


def _feature_row(sym, b_t=B_T, phi=PHI):
    return pd.DataFrame(
        {
            "log_gk_baseline_63": [b_t],
            "garch_persistence": [phi],
            "dev_gk": [0.0], "har_dev_5": [0.0], "har_dev_22": [0.0],
        },
        index=[TODAY - timedelta(days=1)],
    )


# Constant daily GK vol 0.012 → tenor-matched realized ≈ 0.1905 annualized
_GK_SERIES = pd.Series(
    [0.012] * 100,
    index=pd.date_range("2026-01-01", periods=100, freq="B"),
)


def _seed_history(history: DivergenceHistory, symbols, n=130,
                  mu=0.1, spread=0.1, dtes=(10, 18)):
    """Seed n daily gap observations alternating mu±spread → sigma=spread,
    in every DTE band the tests use (z-scores are per tenor band)."""
    for sym in symbols:
        for dte in dtes:
            rows = [
                (date(2026, 1, 1) + timedelta(days=i), sym, dte, 0.25, 0.20,
                 mu + (spread if i % 2 == 0 else -spread))
                for i in range(n)
            ]
            history.log_vrp(rows)


def _run(rows, *, seed_symbols=None, vix_term_ratio=None, dev=0.0,
         expirations=None, top_n=10, **gen_kwargs):
    with tempfile.TemporaryDirectory() as d:
        history = DivergenceHistory(Path(d) / "h.db")
        _seed_history(history, seed_symbols if seed_symbols is not None
                      else [sym for sym, _ in rows])
        gen = SignalGenerator(
            h1_predictor=_FixedH1(dev),
            history_store=history,
            **gen_kwargs,
        )
        scan = _scan(rows, expirations)
        actionable, all_signals = gen.generate(
            scan,
            feature_rows={sym: _feature_row(sym) for sym, _ in rows},
            top_n=top_n,
            vix_term_ratio=vix_term_ratio,
            daily_gk_vol_by_symbol={sym: _GK_SERIES for sym, _ in rows},
        )
        counts = history.gate_counts_today(TODAY)
        history.close()
    return actionable, all_signals, counts


def test_direction_from_vrp_z():
    """SELL when IV is unusually rich vs the ticker's own gap history
    (z >= 1.5), BUY when unusually cheap (z <= -1.25), blocked in between."""
    # seeded mu=0.1, sigma=0.1; realized ≈ 0.1905
    # RICH: iv=0.30 → g=ln(0.30/0.1905)=0.454 → z≈3.5 → SELL
    # CHEAP: iv=0.13 → g=-0.382 → z≈-4.8 → BUY
    # MID: iv=0.215 → g=0.121 → z≈0.2 → blocked
    actionable, all_signals, counts = _run(
        [("RICH", 0.30), ("CHEAP", 0.13), ("MID", 0.215)],
    )
    by = {s.symbol: s for s in all_signals}
    assert by["RICH"].direction == "SELL" and by["RICH"].is_actionable, by["RICH"]
    assert by["CHEAP"].direction == "BUY" and by["CHEAP"].is_actionable, by["CHEAP"]
    assert not by["MID"].is_actionable and by["MID"].blocked_by == "vrp_z"
    assert by["RICH"].vrp_z > 1.5 and by["CHEAP"].vrp_z < -1.25
    # z logged on every record
    assert all(s.vrp_z is not None for s in all_signals)
    assert counts["candidates"] == 3 and counts["approved"] == 2
    print(f"direction: RICH z={by['RICH'].vrp_z:+.1f} SELL, "
          f"CHEAP z={by['CHEAP'].vrp_z:+.1f} BUY, MID blocked")


def test_no_history_emits_no_signal():
    actionable, all_signals, counts = _run(
        [("NEW", 0.30)], seed_symbols=[],  # no gap history at all
    )
    assert len(actionable) == 0
    assert all_signals[0].blocked_by == "vrp_history"
    assert all_signals[0].vrp_z is None
    assert counts.get("vrp_history") == 1
    print("vrp_history: unseeded ticker emits nothing")


def test_hard_sell_block_on_divergence_cap():
    """iv=0.60: z is hugely positive (SELL side) but |forecast−iv| = 0.41
    breaches the 0.25 cap → hard block."""
    actionable, all_signals, _ = _run([("EVENT", 0.60)])
    s = all_signals[0]
    assert s.direction == "SELL" and not s.is_actionable
    assert s.blocked_by == "divergence_cap"
    assert "hard SELL block" in s.diagnostic_notes
    assert len(actionable) == 0
    print("divergence_cap: SELL hard-blocked at |div|=0.41")


def test_vix_backwardation_vetoes_sell_not_buy():
    actionable, all_signals, _ = _run(
        [("RICH", 0.30), ("CHEAP", 0.13)], vix_term_ratio=1.02,
    )
    by = {s.symbol: s for s in all_signals}
    assert by["RICH"].blocked_by == "vix_backwardation"
    assert by["CHEAP"].is_actionable, "BUY must pass the SELL-side veto"
    print("vix_veto: SELL blocked in backwardation, BUY unaffected")


def test_forecast_is_term_projected():
    """The signal's predicted_iv_equivalent must equal the DTE-matched term
    projection, not the raw 1-day annualization."""
    dev = 0.4
    _, all_signals, _ = _run([("RICH", 0.30)], dev=dev)
    s = all_signals[0]
    expected = project_term_vol(B_T, dev, PHI, s.dte)
    assert math.isclose(s.predicted_iv_equivalent, expected, rel_tol=1e-12)
    day1 = math.exp(B_T + dev) * math.sqrt(252)
    assert abs(s.predicted_iv_equivalent - day1) > 1e-6, \
        "must not compare a 1-day forecast to a longer-dated IV"
    print(f"term_projection: forecast {s.predicted_iv_equivalent:.4f} "
          f"(1-day would be {day1:.4f})")


def test_cost_gate_blocks_thin_edge():
    """Same setup as RICH but with the forecast nearly equal to IV via a
    baseline matching the IV — tiny divergence, edge can't clear 2× costs."""
    with tempfile.TemporaryDirectory() as d:
        history = DivergenceHistory(Path(d) / "h.db")
        # seed a WIDE sigma so z still clears the SELL bar despite tiny g
        _seed_history(history, ["X"], mu=-1.0, spread=0.4)
        gen = SignalGenerator(h1_predictor=_FixedH1(0.0), history_store=history)
        # atm_iv 0.195 vs forecast ≈ 0.1905 → divergence ≈ 0.004 → edge ≈ $9
        actionable, all_signals = gen.generate(
            _scan([("X", 0.195)]),
            feature_rows={"X": _feature_row("X")},
            daily_gk_vol_by_symbol={"X": _GK_SERIES},
        )
        history.close()
    s = all_signals[0]
    assert s.direction == "SELL"
    assert s.blocked_by == "cost_gate", (s.blocked_by, s.diagnostic_notes)
    assert s.expected_edge_usd is not None and s.total_cost_usd is not None
    assert s.expected_edge_usd < 2.0 * s.total_cost_usd
    print(f"cost_gate: edge ${s.expected_edge_usd:.0f} < 2×${s.total_cost_usd:.2f} → blocked")


def test_ranking_by_abs_z_and_symbol_dedupe():
    """Stronger |z| ranks first; a symbol with two passing expirations gets
    one actionable slot."""
    exp2 = date(2026, 6, 9)  # DTE 8 — inside the entry window alongside EXP
    actionable, all_signals, _ = _run(
        [("HOT", 0.36), ("WARM", 0.27)],
        expirations=[EXP, exp2],
    )
    assert [s.symbol for s in actionable][:2] == ["HOT", "WARM"]
    assert len([s for s in actionable if s.symbol == "HOT"]) == 1
    hot_all = [s for s in all_signals if s.symbol == "HOT" and s.is_actionable]
    assert len(hot_all) == 2, "both expirations pass gates; dedupe is ranking-side"
    print("ranking: HOT (bigger z) first, one slot per symbol")


def test_missing_baseline_skips_symbol():
    with tempfile.TemporaryDirectory() as d:
        history = DivergenceHistory(Path(d) / "h.db")
        _seed_history(history, ["NOBASE"])
        gen = SignalGenerator(h1_predictor=_FixedH1(), history_store=history)
        actionable, all_signals = gen.generate(
            _scan([("NOBASE", 0.30)]),
            feature_rows={"NOBASE": _feature_row("NOBASE", b_t=float("nan"))},
            daily_gk_vol_by_symbol={"NOBASE": _GK_SERIES},
        )
        history.close()
    assert all_signals == [] and actionable == []
    print("missing_baseline: symbol skipped entirely (no b_t → no forecast)")


def test_model_disagreement_blocks_sell_against_forecast():
    """Reviewer-flagged critical: a rich VRP z (SELL side) whose model
    forecast sits ABOVE the IV being sold must be blocked, not admitted with
    a large fake "edge". Elevated baseline (post-event decay shape): forecast
    ≈ 0.40 vs IV 0.30 → divergence +0.10, under the 0.25 cap — only the
    model-agreement gate stands in the way."""
    with tempfile.TemporaryDirectory() as d:
        history = DivergenceHistory(Path(d) / "h.db")
        _seed_history(history, ["DECAY"])
        gen = SignalGenerator(h1_predictor=_FixedH1(0.0), history_store=history)
        # b_t = log(0.025) → flat term projection ≈ 0.025·√252 ≈ 0.397
        actionable, all_signals = gen.generate(
            _scan([("DECAY", 0.30)]),
            feature_rows={"DECAY": _feature_row("DECAY", b_t=math.log(0.025))},
            daily_gk_vol_by_symbol={"DECAY": _GK_SERIES},
        )
        history.close()
    s = all_signals[0]
    assert s.direction == "SELL" and s.vrp_z > 1.5  # z-gate said sell...
    assert s.divergence > 0                          # ...but the model says vol is HIGH
    assert s.blocked_by == "model_disagrees", (s.blocked_by, s.diagnostic_notes)
    assert not actionable
    print(f"model_disagrees: SELL at divergence {s.divergence:+.3f} blocked")


def test_entry_dte_window():
    """The h=1 path prices a 1-DTE option (the model's purest tenor: one
    overnight of vol exposure) and drops anything beyond MAX_ENTRY_DTE, where
    the term-projected forecast has decayed into the retired VRP-level bet."""
    exp_overnight = date(2026, 6, 2)   # DTE 1 — floor, allowed
    exp_beyond = date(2026, 6, 22)     # DTE 21 — beyond the 14-day cap
    _, all_signals, _ = _run(
        [("RICH", 0.30)], expirations=[exp_overnight, EXP, exp_beyond],
    )
    dtes = sorted({s.dte for s in all_signals})
    assert dtes == [1, 11], f"expected DTEs [1, 11], got {dtes}"

    # Window is constructor-tunable: floor 7 re-excludes the overnight entry.
    _, sigs7, _ = _run(
        [("RICH", 0.30)], expirations=[exp_overnight, EXP, exp_beyond],
        min_entry_dte=7,
    )
    assert sorted({s.dte for s in sigs7}) == [11]
    print("entry_dte_window: DTE 1 priced, 21 capped, floor=7 honored")


def main() -> int:
    test_direction_from_vrp_z()
    test_no_history_emits_no_signal()
    test_hard_sell_block_on_divergence_cap()
    test_vix_backwardation_vetoes_sell_not_buy()
    test_forecast_is_term_projected()
    test_cost_gate_blocks_thin_edge()
    test_ranking_by_abs_z_and_symbol_dedupe()
    test_missing_baseline_skips_symbol()
    test_model_disagreement_blocks_sell_against_forecast()
    test_entry_dte_window()
    print("all signal_generator_h1 tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
