"""Live risk management test — observation only, no order submission.

Builds a PortfolioSnapshot from current sandbox state, runs today's
actionable signals through RiskManager.gate(), and prints decisions.
"""
import asyncio
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from config import load_settings, load_watchlist
from data import (
    AsyncTradierClient,
    HistoricalStore,
    MarketData,
    compute_log_returns,
)
from execution import OrderLog
from features import FeaturePipeline
from model import (
    BestPredictor,
    GARCHBaseline,
    XGBoostVolPredictor,
)
from positions import PositionTracker
from risk import DailyKillSwitch, PortfolioStateBuilder, RiskManager
from signals import DivergenceHistory, SignalGenerator


def _last_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _load_predictors(artifact_dir: Path):
    predictors = {}
    for h in (5, 10, 21):
        existing = list(artifact_dir.glob(f"xgb_h{h}_*.joblib"))
        if not existing:
            raise RuntimeError(f"no model artifact for h={h}")
        newest = max(existing, key=lambda p: p.stat().st_mtime)
        xgb = XGBoostVolPredictor.load(newest)
        garch = GARCHBaseline(refit_every=21, min_history=100)
        bp = BestPredictor(garch, xgb, horizon=h)
        bp.update_from_eval(garch_r2=-0.1, xgb_r2=0.2)
        predictors[h] = bp
    return predictors


async def main_async() -> int:
    settings = load_settings()
    if settings.env != "sandbox":
        print(f"refusing live test against env={settings.env!r}", file=sys.stderr)
        return 2

    end = _last_weekday(date.today())
    start = end - timedelta(days=730)
    tickers = load_watchlist()
    symbols = [t.symbol for t in tickers]

    store = HistoricalStore(settings.cache_db_path)
    artifact_dir = Path(__file__).resolve().parent.parent / "model" / "artifacts"
    order_log = OrderLog(settings.cache_db_path.parent / "order_log.db")

    # Use a temp kill-switch DB so this test doesn't pollute the persistent state
    with tempfile.TemporaryDirectory() as tmp:
        kill_switch = DailyKillSwitch(Path(tmp) / "ks.db")

        try:
            # Bootstrap inputs
            pipeline = FeaturePipeline(
                store, tickers, garch_min_history=100, garch_refit_every=21,
            )
            feature_df = pipeline.build_features(start, end)
            returns_by_symbol = {
                sym: compute_log_returns(store.get_bars(sym, start, end)["close"])
                for sym in symbols
            }
            feature_rows = {}
            for sym in symbols:
                try:
                    sym_df = feature_df.loc[sym]
                    feature_rows[sym] = sym_df.loc[[sym_df.index.max()]]
                except KeyError:
                    continue

            predictors = _load_predictors(artifact_dir)

            async with AsyncTradierClient(settings) as client:
                md = MarketData(client, tickers)
                t0 = time.monotonic()
                scan = await md.scan(expiration_window=(3, 60))
                scan_elapsed = time.monotonic() - t0
                print(f"scan: {scan.total_contracts} contracts in {scan_elapsed:.1f}s")

                # Build snapshot
                tracker = PositionTracker(client=client, order_log=order_log, settings=settings)
                builder = PortfolioStateBuilder(
                    client=client, order_log=order_log,
                    position_tracker=tracker, watchlist=tickers,
                    kill_switch=kill_switch,
                )
                snapshot = await builder.snapshot(scan)

                print("\n=== Portfolio Snapshot ===")
                print(f"  equity:           ${snapshot.equity:,.2f}")
                print(f"  starting_equity:  ${snapshot.starting_equity_today:,.2f}")
                print(f"  buying_power:     ${snapshot.buying_power:,.2f}")
                print(f"  margin_held:      ${snapshot.margin_held:,.2f}")
                print(f"  open positions:   {len(snapshot.open_positions)}")
                print(f"  today realized:   ${snapshot.today_realized_pnl:+,.2f}")
                print(f"  today unrealized: ${snapshot.today_unrealized_pnl:+,.2f}")
                print(f"  today total:      ${snapshot.today_total_pnl:+,.2f} "
                      f"({snapshot.today_pnl_pct_of_equity:+.2%})")
                print(f"  greeks: delta={snapshot.portfolio_greeks['delta']:+.2f} "
                      f"gamma={snapshot.portfolio_greeks['gamma']:+.2f} "
                      f"vega={snapshot.portfolio_greeks['vega']:+.2f}")
                print(f"  positions by sector: {dict(snapshot.positions_by_sector)}")
                print(f"  exposure by symbol: {dict(snapshot.exposure_by_symbol)}")

                # Run signal generation
                div_history = DivergenceHistory(settings.cache_db_path.parent / "divergence_history.db")
                generator = SignalGenerator(
                    predictors_by_horizon=predictors,
                    history_store=div_history,
                    cross_sectional_z_threshold=1.5,
                    max_divergence=0.25,
                )
                actionable, all_signals = generator.generate(
                    scan=scan, feature_rows=feature_rows,
                    returns_by_symbol=returns_by_symbol, top_n=10,
                )
                div_history.close()
                print(f"\nsignals: {len(all_signals)} total, {len(actionable)} actionable")

                # Run risk gate
                risk_manager = RiskManager(
                    watchlist=tickers,
                    max_per_trade_loss_pct=0.01,
                    max_per_ticker_exposure_pct=0.05,
                    max_per_sector_positions=3,
                    max_portfolio_delta_pct=0.05,
                    max_portfolio_gamma_pct=0.01,
                    max_portfolio_vega_pct=0.05,
                    min_buying_power_buffer_pct=0.05,
                    kill_switch=kill_switch,
                )
                decisions = risk_manager.gate(actionable, scan, snapshot)

                # Print decisions
                print(f"\n=== Risk gate decisions ===")
                print(f"{'sym':5s} {'dir':4s} {'qty':>3s}  {'approved':>9s}  reasons")
                print("-" * 100)
                approved_count = 0
                for d in decisions:
                    label = "YES" if d.approved else "NO"
                    if d.approved:
                        approved_count += 1
                    reasons = "; ".join(d.reasons[:2]) if d.reasons else "-"
                    print(f"{d.signal.symbol:5s} {d.signal.direction:4s} {d.quantity:>3d}  "
                          f"{label:>9s}  {reasons}")

                print(f"\n{approved_count}/{len(decisions)} signals approved")

                # Sanity assertions
                assert len(decisions) == len(actionable)
                for d in decisions:
                    if d.approved:
                        assert d.quantity > 0
                        assert not d.reasons
                    else:
                        assert d.quantity == 0
                        assert len(d.reasons) >= 1
                # Snapshot should be sane
                assert snapshot.equity > 0
                assert snapshot.buying_power > 0
        finally:
            store.close()
            order_log.close()
            kill_switch.close()
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
