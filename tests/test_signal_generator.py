import asyncio
import math
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import Ticker, load_settings, load_watchlist
from data import (
    AsyncTradierClient,
    HistoricalStore,
    MarketData,
    OptionContract,
    compute_log_returns,
)
from features import FeaturePipeline
from model import (
    DEFAULT_HYPERPARAMS,
    BestPredictor,
    GARCHBaseline,
    XGBoostVolPredictor,
    build_training_matrix,
)
from signals import (
    DivergenceHistory,
    SignalGenerator,
    blend_prediction,
    composite_liquidity,
    find_atm_iv,
    interpolate_horizon,
)


def _last_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


# ---------- pure unit tests ----------

def test_interpolate_horizon():
    # boundaries
    assert interpolate_horizon(5) == (5, 5, 1.0)
    assert interpolate_horizon(10) == (10, 10, 1.0)
    assert interpolate_horizon(21) == (21, 21, 1.0)
    # interior 5-10
    assert interpolate_horizon(8) == (5, 10, 0.4)
    # interior 10-21
    h_lo, h_up, w_lo = interpolate_horizon(15)
    assert (h_lo, h_up) == (10, 21)
    assert math.isclose(w_lo, 6 / 11, rel_tol=1e-9)
    # extrapolation
    assert interpolate_horizon(4) == (5, 5, 1.0)
    assert interpolate_horizon(30) == (21, 21, 1.0)
    # out of range (MIN_DTE=4 to give a buffer above expiration_proximity_dte=2)
    assert interpolate_horizon(3) is None
    assert interpolate_horizon(50) is None
    print("interpolate_horizon: OK on boundaries, interior, extrapolation, out-of-range")


def test_blend_prediction():
    preds = {5: 0.010, 10: 0.012, 21: 0.015}
    # boundary
    assert blend_prediction(preds, 10, 10, 1.0) == 0.012
    # 50/50 between 10 and 21
    assert math.isclose(blend_prediction(preds, 10, 21, 0.5), 0.0135, rel_tol=1e-12)
    # 40/60 between 5 and 10
    assert math.isclose(blend_prediction(preds, 5, 10, 0.4), 0.4 * 0.010 + 0.6 * 0.012, rel_tol=1e-12)
    print("blend_prediction: OK")


def _mk_contract(strike: float, otype: str, *, bid=1.0, ask=1.1, vol=100, oi=500, iv=0.3) -> OptionContract:
    return OptionContract(
        symbol=f"X{strike:.0f}{otype[0].upper()}",
        underlying="X",
        expiration=date(2026, 5, 22),
        strike=strike,
        option_type=otype,
        bid=bid,
        ask=ask,
        last=(bid + ask) / 2,
        volume=vol,
        open_interest=oi,
        delta=0.5 if otype == "call" else -0.5,
        gamma=0.01,
        theta=-0.05,
        vega=0.20,
        iv=iv,
        fetched_at=datetime.now(timezone.utc),
    )


def test_find_atm_iv():
    chain = [_mk_contract(k, t) for k in (90, 95, 100, 105, 110) for t in ("call", "put")]
    pair = find_atm_iv(chain, underlying_price=102.0)
    assert pair is not None
    call, put = pair
    assert call.strike == 100, f"expected 100, got {call.strike}"
    assert put.strike == 100
    print("find_atm_iv: 102 → 100-strike pair (closest)")


def test_find_atm_iv_returns_none_when_one_side_missing():
    chain = [_mk_contract(k, "call") for k in (90, 100, 110)]
    assert find_atm_iv(chain, underlying_price=100) is None
    print("find_atm_iv: None when one side missing")


def test_composite_liquidity():
    call = _mk_contract(100, "call", bid=1.0, ask=1.1, vol=100, oi=500)
    put = _mk_contract(100, "put", bid=0.9, ask=1.0, vol=200, oi=800)
    # min vol=100, min oi=500, max spread = (1.1-1.0)/1.05 ≈ 0.0952
    expected = 100 * 500 / (1 + max((1.1 - 1.0) / 1.05, (1.0 - 0.9) / 0.95))
    actual = composite_liquidity(call, put)
    assert math.isclose(actual, expected, rel_tol=1e-9)
    print(f"composite_liquidity: {actual:.1f} matches manual calc")


def test_divergence_cap_demotes_event_suspects():
    """Synthesize a scan with one ticker carrying a 0.41 divergence (AMZN
    earnings shape). With max_divergence=0.30 it should be demoted as
    event-suspect even if its z-score is above threshold."""
    from datetime import datetime
    from data.market_data import ScanResult, TickerSnapshot
    from model.best_predictor import BestPredictor
    from model.garch_baseline import GARCHBaseline
    from signals import SignalGenerator

    # Build a minimal scan: 5 tickers, all DTE=10. One has wildly inflated IV.
    today = date(2026, 4, 26)
    expiration = date(2026, 5, 6)  # DTE=10
    snapshots: dict[str, TickerSnapshot] = {}
    for sym, atm_iv in [("A", 0.20), ("B", 0.22), ("C", 0.21), ("D", 0.19), ("EVENT", 0.65)]:
        atm_call = _mk_contract(100, "call", bid=1.0, ask=1.05, vol=200, oi=1000, iv=atm_iv)
        atm_call = OptionContract(
            symbol=f"{sym}_C100", underlying=sym, expiration=expiration, strike=100,
            option_type="call", bid=1.0, ask=1.05, last=1.025, volume=200,
            open_interest=1000, delta=0.5, gamma=0.01, theta=-0.05, vega=0.2,
            iv=atm_iv, fetched_at=datetime.now(timezone.utc),
        )
        atm_put = OptionContract(
            symbol=f"{sym}_P100", underlying=sym, expiration=expiration, strike=100,
            option_type="put", bid=0.9, ask=0.95, last=0.925, volume=200,
            open_interest=1000, delta=-0.5, gamma=0.01, theta=-0.05, vega=0.2,
            iv=atm_iv, fetched_at=datetime.now(timezone.utc),
        )
        # Add OTM wings for iron condor
        wing_call = OptionContract(
            symbol=f"{sym}_C110", underlying=sym, expiration=expiration, strike=110,
            option_type="call", bid=0.20, ask=0.22, last=0.21, volume=100,
            open_interest=500, delta=0.2, gamma=0.005, theta=-0.02, vega=0.1,
            iv=atm_iv, fetched_at=datetime.now(timezone.utc),
        )
        wing_put = OptionContract(
            symbol=f"{sym}_P90", underlying=sym, expiration=expiration, strike=90,
            option_type="put", bid=0.20, ask=0.22, last=0.21, volume=100,
            open_interest=500, delta=-0.2, gamma=0.005, theta=-0.02, vega=0.1,
            iv=atm_iv, fetched_at=datetime.now(timezone.utc),
        )
        snapshots[sym] = TickerSnapshot(
            symbol=sym, sector="tech",
            underlying={"symbol": sym, "last": 100.0},
            contracts=[atm_call, atm_put, wing_call, wing_put],
        )

    scan = ScanResult(fetched_at=datetime(2026, 4, 26, 16, 0, tzinfo=timezone.utc), snapshots=snapshots)

    # Predictor that returns daily vol = 0.012 (annualized ~0.19) for every horizon
    class _FixedPredictor:
        def predict_forward_rv(self, returns_history, X_row=None):
            return 0.012
    predictors = {h: _FixedPredictor() for h in (5, 10, 21)}

    feature_rows = {sym: pd.DataFrame({"a": [0.0]}, index=[date(2026, 4, 25)]) for sym in snapshots}
    returns_by_symbol = {sym: pd.Series([0.001] * 100) for sym in snapshots}

    generator = SignalGenerator(
        predictors_by_horizon=predictors,
        history_store=None,
        cross_sectional_z_threshold=1.0,
        max_divergence=0.30,
    )
    actionable, all_signals = generator.generate(scan, feature_rows, returns_by_symbol, top_n=10)

    by_sym = {s.symbol: s for s in all_signals}
    # EVENT has divergence ≈ 0.19 - 0.65 = -0.46 (|.46| > 0.30 cap)
    assert "EVENT" in by_sym
    event_sig = by_sym["EVENT"]
    assert not event_sig.is_actionable, "EVENT should be demoted by divergence cap"
    assert "event-suspect" in event_sig.diagnostic_notes
    assert "EVENT" not in [s.symbol for s in actionable], "EVENT must not be in actionable"
    print(f"divergence_cap: EVENT demoted with note '{event_sig.diagnostic_notes}'")


# ---------- entry-DTE gate + long-straddle exclusion ----------

class _ConstPredictor:
    """Returns a fixed daily RV for every horizon (annualizes to ~0.19)."""
    def predict_forward_rv(self, returns_history, X_row=None):
        return 0.012


def _straddle_chain(sym, expiration, atm_iv, underlying=100.0):
    """ATM call+put plus 1σ OTM wings for one (symbol, expiration), all liquid."""
    def _c(strike, otype, bid, ask, delta):
        return OptionContract(
            symbol=f"{sym}_{otype[0].upper()}{strike:.0f}_{expiration.isoformat()}",
            underlying=sym, expiration=expiration, strike=strike, option_type=otype,
            bid=bid, ask=ask, last=(bid + ask) / 2, volume=200, open_interest=1000,
            delta=delta, gamma=0.01, theta=-0.05, vega=0.2, iv=atm_iv,
            fetched_at=datetime.now(timezone.utc),
        )
    return [
        _c(100, "call", 1.0, 1.05, 0.5),
        _c(100, "put", 0.9, 0.95, -0.5),
        _c(110, "call", 0.20, 0.22, 0.2),   # OTM wing
        _c(90, "put", 0.20, 0.22, -0.2),    # OTM wing
    ]


def test_entry_dte_gate_skips_near_expiry():
    """A candidate below MIN_ENTRY_DTE never becomes a signal; one above does.
    Guards the weekend-compression churn: a fresh straddle opened at DTE 4 would
    be force-closed by the expiration_proximity exit the next session."""
    from datetime import datetime
    from data.market_data import ScanResult, TickerSnapshot
    from signals import MIN_ENTRY_DTE

    # scan.fetched_at below is 2026-06-01, which generate() uses as "today".
    near = date(2026, 6, 5)    # DTE 4  (< MIN_ENTRY_DTE)
    far = date(2026, 6, 12)    # DTE 11 (>= MIN_ENTRY_DTE)
    contracts = _straddle_chain("X", near, 0.20) + _straddle_chain("X", far, 0.20)
    snap = TickerSnapshot(symbol="X", sector="tech",
                          underlying={"symbol": "X", "last": 100.0}, contracts=contracts)
    scan = ScanResult(fetched_at=datetime(2026, 6, 1, 16, 0, tzinfo=timezone.utc),
                      snapshots={"X": snap})

    predictors = {h: _ConstPredictor() for h in (5, 10, 21)}
    feature_rows = {"X": pd.DataFrame({"a": [0.0]}, index=[date(2026, 5, 29)])}
    returns_by_symbol = {"X": pd.Series([0.001] * 100)}

    gen = SignalGenerator(predictors_by_horizon=predictors, history_store=None,
                          cross_sectional_z_threshold=1.0)
    _, all_signals = gen.generate(scan, feature_rows, returns_by_symbol, top_n=10)

    dtes = sorted({s.dte for s in all_signals})
    assert all(s.dte >= MIN_ENTRY_DTE for s in all_signals), f"got DTEs {dtes}"
    assert any(s.dte == 11 for s in all_signals), f"expected DTE-11 signal, got {dtes}"
    assert not any(s.dte == 4 for s in all_signals), f"DTE-4 leaked through gate: {dtes}"
    print(f"entry_dte_gate: DTE-4 skipped, DTE-11 kept (MIN_ENTRY_DTE={MIN_ENTRY_DTE})")


def test_long_straddle_exclusion_demotes_etf_buy_only():
    """An excluded symbol's BUY straddle is demoted; an identical non-excluded
    symbol stays actionable. The exclusion is BUY-side only."""
    from datetime import datetime
    from data.market_data import ScanResult, TickerSnapshot

    # scan.fetched_at below is 2026-06-01, which generate() uses as "today".
    exp = date(2026, 6, 12)  # DTE 11
    # SPY + CTRL are BUY outliers (low IV → predicted >> implied); fillers sit
    # near the predicted IV so the cross-sectional z separates the outliers.
    rows = [("SPY", 0.10, "etf"), ("CTRL", 0.10, "tech"),
            ("A", 0.19, "tech"), ("B", 0.19, "tech"),
            ("C", 0.19, "tech"), ("D", 0.19, "tech")]
    snapshots = {
        sym: TickerSnapshot(symbol=sym, sector=sector,
                            underlying={"symbol": sym, "last": 100.0},
                            contracts=_straddle_chain(sym, exp, iv))
        for sym, iv, sector in rows
    }
    scan = ScanResult(fetched_at=datetime(2026, 6, 1, 16, 0, tzinfo=timezone.utc),
                      snapshots=snapshots)

    predictors = {h: _ConstPredictor() for h in (5, 10, 21)}
    feature_rows = {sym: pd.DataFrame({"a": [0.0]}, index=[date(2026, 5, 29)]) for sym in snapshots}
    returns_by_symbol = {sym: pd.Series([0.001] * 100) for sym in snapshots}

    gen = SignalGenerator(predictors_by_horizon=predictors, history_store=None,
                          cross_sectional_z_threshold=1.0,
                          long_straddle_excluded_symbols={"SPY"})
    actionable, all_signals = gen.generate(scan, feature_rows, returns_by_symbol, top_n=10)
    by = {s.symbol: s for s in all_signals}

    # SPY BUY is demoted purely by the exclusion (note says so), not by z.
    assert "SPY" in by, "SPY signal missing"
    assert by["SPY"].direction == "BUY"
    assert not by["SPY"].is_actionable, "excluded SPY BUY should be demoted"
    assert "excluded" in by["SPY"].diagnostic_notes
    assert "SPY" not in [s.symbol for s in actionable]

    # CTRL has the identical setup but isn't excluded → stays actionable.
    assert by["CTRL"].direction == "BUY"
    assert by["CTRL"].is_actionable, f"CTRL should be actionable: {by['CTRL'].diagnostic_notes}"
    assert len(by["CTRL"].legs) == 2
    print("long_straddle_exclusion: SPY BUY demoted, CTRL BUY actionable")


# ---------- live integration ----------

async def _bootstrap_predictors(
    settings,
    store: HistoricalStore,
    tickers,
    feature_df: pd.DataFrame,
    returns_by_symbol: dict[str, pd.Series],
    artifact_dir: Path,
) -> dict[int, BestPredictor]:
    """Train+save default-hyperparam XGBoost models per horizon if missing,
    then load + wrap each in a BestPredictor flipped to XGBoost mode."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictors: dict[int, BestPredictor] = {}

    for horizon in (5, 10, 21):
        # Pick newest existing artifact by mtime (handles mix of bootstrap +
        # date-named tuned files where filename sort would lie).
        existing = list(artifact_dir.glob(f"xgb_h{horizon}_*.joblib"))
        if existing:
            newest = max(existing, key=lambda p: p.stat().st_mtime)
            xgb = XGBoostVolPredictor.load(newest)
            print(f"  loaded h={horizon} from {newest.name}")
        else:
            X, y = build_training_matrix(feature_df, returns_by_symbol, horizon)
            xgb = XGBoostVolPredictor(horizon=horizon, hyperparams=DEFAULT_HYPERPARAMS)
            xgb.fit(X, y)
            artifact_path = artifact_dir / f"xgb_h{horizon}_bootstrap.joblib"
            xgb.save(artifact_path)
            print(f"  trained + saved h={horizon} ({len(X)} rows) → {artifact_path.name}")

        garch = GARCHBaseline(refit_every=21, min_history=100)
        bp = BestPredictor(garch, xgb, horizon=horizon)
        # Per slow tuning test, XGBoost beats GARCH at all horizons — flip on
        bp.update_from_eval(garch_r2=-0.1, xgb_r2=+0.2)
        predictors[horizon] = bp

    return predictors


async def live_test() -> int:
    settings = load_settings()
    end = _last_weekday(date.today())
    start = end - timedelta(days=730)
    tickers = load_watchlist()
    symbols = [t.symbol for t in tickers]

    store = HistoricalStore(settings.cache_db_path)
    try:
        # Build features (uses cached data)
        pipeline = FeaturePipeline(
            store, tickers,
            garch_min_history=100, garch_refit_every=21,
        )
        t0 = time.monotonic()
        feature_df = pipeline.build_features(start, end)
        feat_elapsed = time.monotonic() - t0
        print(f"built features: {len(feature_df)} rows × {len(feature_df.columns)} cols in {feat_elapsed:.1f}s")

        returns_by_symbol = {
            sym: compute_log_returns(store.get_bars(sym, start, end)["close"])
            for sym in symbols
        }

        # Bootstrap predictors
        artifact_dir = Path(__file__).resolve().parent.parent / "model" / "artifacts"
        print(f"bootstrapping predictors (artifact_dir={artifact_dir})")
        predictors = await _bootstrap_predictors(
            settings, store, tickers, feature_df, returns_by_symbol, artifact_dir,
        )

        # Latest feature row per symbol
        feature_rows = {}
        for sym in symbols:
            try:
                sym_df = feature_df.loc[sym]
            except KeyError:
                continue
            latest_date = sym_df.index.max()
            feature_rows[sym] = sym_df.loc[[latest_date]]

        # Live scan
        print("\nrunning live scan...")
        async with AsyncTradierClient(settings) as client:
            md = MarketData(client, tickers)
            t0 = time.monotonic()
            scan = await md.scan(expiration_window=(3, 45))
            scan_elapsed = time.monotonic() - t0
        print(f"scan: {len(scan)} tickers, {scan.total_contracts} contracts in {scan_elapsed:.1f}s")

        # Generate signals
        with tempfile.TemporaryDirectory() as tmp:
            history = DivergenceHistory(Path(tmp) / "h.db")
            generator = SignalGenerator(
                predictors_by_horizon=predictors,
                history_store=history,
                cross_sectional_z_threshold=1.5,
            )
            t0 = time.monotonic()
            actionable, all_signals = generator.generate(
                scan=scan,
                feature_rows=feature_rows,
                returns_by_symbol=returns_by_symbol,
                top_n=5,
            )
            sig_elapsed = time.monotonic() - t0
            print(f"signals: {len(all_signals)} total, {len(actionable)} actionable in {sig_elapsed:.1f}s")
            print(f"  history rows logged: {history.row_count()}")
            history.close()

        # Print divergence + z-score distribution by bucket
        print("\n=== Bucket summary (cross-sectional z by horizon pair) ===")
        from collections import defaultdict
        buckets = defaultdict(list)
        for s in all_signals:
            buckets[(s.horizon_lower, s.horizon_upper)].append(s)
        for (h_lo, h_up), sigs in sorted(buckets.items()):
            divs = [s.divergence for s in sigs]
            z_max = max(abs(s.cross_sectional_z) for s in sigs)
            print(f"  ({h_lo:2d}, {h_up:2d}): n={len(sigs):3d}  "
                  f"div mean={np.mean(divs):+.4f} std={np.std(divs):.4f}  "
                  f"max |z|={z_max:.2f}")

        # Print top-5 table
        print("\n=== Top 5 actionable signals ===")
        if not actionable:
            # If nothing actionable, show top-5 all signals by |z|
            print("  (no actionable signals — showing top 5 by |z|)")
            actionable = sorted(all_signals, key=lambda s: -abs(s.cross_sectional_z))[:5]
        if actionable:
            header = f"{'sym':5s} {'exp':10s} {'dte':>3s} {'h_lo→h_up':10s} {'w_lo':>5s}  {'dir':4s} {'pred_iv':>8s} {'atm_iv':>8s} {'div':>9s} {'cs_z':>6s}  legs"
            print(header)
            print("-" * len(header))
            for s in actionable[:5]:
                pair = f"{s.horizon_lower}→{s.horizon_upper}"
                legs_str = f"{len(s.legs)}-leg {s.direction}" if s.legs else "(no legs)"
                print(f"{s.symbol:5s} {s.expiration.isoformat()} {s.dte:3d} "
                      f"{pair:10s} {s.weight_lower:5.2f}  {s.direction:4s} "
                      f"{s.predicted_iv_equivalent:>+8.4f} {s.atm_iv:>+8.4f} {s.divergence:>+8.4f} "
                      f"{s.cross_sectional_z:>+6.2f}  {legs_str}")
                if s.diagnostic_notes:
                    print(f"  ⚠  {s.diagnostic_notes}")

        # Sanity assertions
        assert len(all_signals) > 0, "expected ≥1 raw signal"
        assert all(isinstance(s.predicted_iv_equivalent, float) for s in all_signals)
        assert all(s.dte >= 3 for s in all_signals)
        for s in actionable:
            if s.is_actionable:
                assert len(s.legs) in (2, 4), f"unexpected leg count {len(s.legs)}"
                assert all(leg.contract_symbol for leg in s.legs)

        total = feat_elapsed + scan_elapsed + sig_elapsed
        print(f"\ntotal wall-clock: {total:.1f}s (gate <60s)")
        assert total < 60, f"FAIL: took {total:.1f}s"
    finally:
        store.close()
    return 0


def main() -> int:
    test_interpolate_horizon()
    test_blend_prediction()
    test_find_atm_iv()
    test_find_atm_iv_returns_none_when_one_side_missing()
    test_composite_liquidity()
    test_divergence_cap_demotes_event_suspects()
    test_entry_dte_gate_skips_near_expiry()
    test_long_straddle_exclusion_demotes_etf_buy_only()

    settings = load_settings()
    if settings.env != "sandbox":
        print(f"refusing to run live against env={settings.env!r}", file=sys.stderr)
        return 0
    return asyncio.run(live_test())


if __name__ == "__main__":
    sys.exit(main())
