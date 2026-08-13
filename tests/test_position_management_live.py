"""Live position management test — DRY-RUN BY DEFAULT.

Pulls open positions from our order log, marks them to market against the
current scan, runs the exit manager, and prints a decisions table.

Set EXECUTE_EXITS=YES to actually submit closing orders. Default = print only,
preserving the sandbox track record.
"""
import asyncio
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from config import load_settings, load_watchlist
from data import (
    AsyncTradierClient,
    HistoricalStore,
    MarketData,
)
from execution import OrderLog, OrderManager
from features import FeaturePipeline
from main import _load_h1_predictor
from positions import ExitManager, PositionTracker


def _last_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


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
    order_log_path = settings.cache_db_path.parent / "order_log.db"
    order_log = OrderLog(order_log_path)

    try:
        # Build features (cached) + returns + load predictors
        pipeline = FeaturePipeline(
            store, tickers, garch_min_history=100, garch_refit_every=21,
        )
        feature_df = pipeline.build_features(start, end)
        feature_rows = {}
        for sym in symbols:
            try:
                sym_df = feature_df.loc[sym]
                feature_rows[sym] = sym_df.loc[[sym_df.index.max()]]
            except KeyError:
                continue

        h1_predictor, _phi_by_symbol = _load_h1_predictor(artifact_dir)

        # Live scan — wide window so we have prices for any open position
        async with AsyncTradierClient(settings) as client:
            md = MarketData(client, tickers)
            t0 = time.monotonic()
            scan = await md.scan(expiration_window=(1, 45))
            scan_elapsed = time.monotonic() - t0
            print(f"scan: {scan.total_contracts} contracts in {scan_elapsed:.1f}s")

            # Set up tracker + exit manager
            tracker = PositionTracker(client=client, order_log=order_log, settings=settings)
            order_manager = OrderManager(
                client=client, order_log=order_log, settings=settings,
            )
            exit_manager = ExitManager(
                position_tracker=tracker, order_manager=order_manager,
                h1_predictor=h1_predictor,
                # explicit defaults from the plan:
                straddle_profit_target_pct=1.00,
                straddle_stop_loss_pct=-0.50,
                iron_condor_profit_target_pct=0.50,
                iron_condor_stop_loss_pct=-1.00,
                expiration_proximity_dte=2,
                thesis_reversal_min_magnitude=0.05,
            )

            # 1. List open positions
            positions = await tracker.list_open_positions()
            print(f"\nopen positions: {len(positions)}")
            if not positions:
                print("(none) — likely first run, or step-7 trade was closed manually")
                return 0
            for p in positions:
                print(f"  order_id={p.tradier_order_id} {p.symbol} {p.direction} "
                      f"{p.structure} exp={p.expiration} entry_premium=${p.entry_premium:.2f} "
                      f"div_at_entry={p.entry_divergence:+.4f}")

            # 2. Mark to market
            marks = tracker.mark_to_market(positions, scan)
            print(f"\nmarked positions: {len(marks)}")
            if len(marks) < len(positions):
                missing = len(positions) - len(marks)
                print(f"  warning: {missing} position(s) skipped (legs not in scan)")

            # 3. Print mark + greeks per position
            print("\n=== Position marks ===")
            print(f"{'order_id':>10s} {'sym':5s} {'dte':>4s} {'pnl_$':>10s} "
                  f"{'pnl_%':>7s} {'cost_close':>10s} {'delta':>7s} {'theta':>7s} {'vega':>7s}")
            print("-" * 88)
            for m in marks:
                pnl_pct = m.pnl_pct_of_entry_premium * 100 if not (m.pnl_pct_of_entry_premium != m.pnl_pct_of_entry_premium) else float("nan")
                print(f"{m.position.tradier_order_id:>10d} {m.position.symbol:5s} "
                      f"{m.dte:>4d} ${m.pnl_dollars:>+9.2f} "
                      f"{pnl_pct:>+6.1f}%  ${m.cost_to_close:>9.2f} "
                      f"{m.delta:>+7.2f} {m.theta:>+7.2f} {m.vega:>+7.2f}")

            # 4. Portfolio greeks
            port = PositionTracker.portfolio_greeks(marks)
            print(f"\nportfolio greeks: delta={port['delta']:+.2f} gamma={port['gamma']:+.2f} "
                  f"theta={port['theta']:+.2f} vega={port['vega']:+.2f}")

            # 5. Evaluate exit triggers
            decisions = exit_manager.evaluate(marks, scan, feature_rows)
            print(f"\n=== Exit decisions ===")
            print(f"{'order_id':>10s} {'sym':5s} {'action':6s}  {'trigger':>22s}  rationale")
            print("-" * 100)
            for d in decisions:
                trig = d.trigger or "-"
                print(f"{d.position.tradier_order_id:>10d} {d.position.symbol:5s} "
                      f"{d.action:6s}  {trig:>22s}  {d.rationale}")
                if d.current_divergence is not None:
                    print(f"           divergence: entry={d.position.entry_divergence:+.4f} "
                          f"current={d.current_divergence:+.4f}")

            # 6. Execute if env var set
            should_execute = os.environ.get("EXECUTE_EXITS") == "YES"
            print(f"\nEXECUTE_EXITS={'YES' if should_execute else 'NO (dry-run)'}")
            results = await exit_manager.execute(decisions, dry_run=not should_execute)
            for decision, order_result in results:
                if decision.action == "close" and order_result is not None:
                    print(f"  CLOSE submitted for order {decision.position.tradier_order_id}: "
                          f"status={order_result.status} closing_order_id={order_result.order_id} "
                          f"fill={order_result.fill_price}")

            # Sanity assertions
            assert all(isinstance(m.pnl_dollars, float) for m in marks)
            for m in marks:
                # P&L sanity bound: |pnl| < 200% of entry premium
                cap = abs(m.position.entry_premium) * 200
                assert abs(m.pnl_dollars) < cap, f"P&L {m.pnl_dollars} exceeds sanity cap {cap}"
            assert len(decisions) == len(marks)
            for d in decisions:
                assert d.action in ("close", "hold")
                if d.action == "close":
                    assert d.trigger is not None
                else:
                    assert d.trigger is None

        print("\nlive position management test complete")
    finally:
        store.close()
        order_log.close()
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
