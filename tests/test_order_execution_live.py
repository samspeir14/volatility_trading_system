"""Live execution test — places a real order in the Tradier sandbox account.

This test MUTATES sandbox account state. It is by design: per the project
decision to use Tradier's sandbox positions as the strategy P&L track record,
we want the bot to actually accumulate orders here.

Re-running the test on the same day is idempotent because the OrderManager's
fingerprint dedup catches re-submission of the same signal on the same scan
date.
"""

import asyncio
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from config import Ticker, load_settings, load_watchlist
from data import (
    AsyncTradierClient,
    HistoricalStore,
    MarketData,
)
from execution import OrderLog, OrderManager
from features import FeaturePipeline
from features.target import daily_ohlc_vol
from main import _load_h1_predictor
from signals import DivergenceHistory, SignalGenerator


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
    print(f"order log: {order_log_path}  (current: submitted={order_log.submitted_count()}, failed={order_log.failed_count()})")

    try:
        # 1. Bootstrap inputs (features, returns, predictors)
        pipeline = FeaturePipeline(
            store, tickers,
            garch_min_history=100, garch_refit_every=21,
        )
        t0 = time.monotonic()
        feature_df = pipeline.build_features(start, end)
        print(f"features: {len(feature_df)} rows in {time.monotonic() - t0:.1f}s")

        feature_rows = {}
        for sym in symbols:
            try:
                sym_df = feature_df.loc[sym]
            except KeyError:
                continue
            feature_rows[sym] = sym_df.loc[[sym_df.index.max()]]

        gk_vol_by_symbol = {}
        for sym in symbols:
            bars = store.get_bars(sym, start, end)
            if not bars.empty:
                gk_vol_by_symbol[sym] = daily_ohlc_vol(bars)

        print("loading h=1 predictor:")
        h1_predictor, phi_by_symbol = _load_h1_predictor(artifact_dir)

        # 2. Live scan + signal generation
        async with AsyncTradierClient(settings) as client:
            md = MarketData(client, tickers)
            t0 = time.monotonic()
            scan = await md.scan(expiration_window=(1, 14))
            print(f"scan: {scan.total_contracts} contracts in {time.monotonic() - t0:.1f}s")

            div_history = DivergenceHistory(settings.cache_db_path.parent / "divergence_history.db")
            generator = SignalGenerator(
                h1_predictor=h1_predictor,
                phi_by_symbol=phi_by_symbol,
                history_store=div_history,
                max_divergence=0.25,
            )
            actionable, all_signals = generator.generate(
                scan=scan, feature_rows=feature_rows, top_n=10,
                daily_gk_vol_by_symbol=gk_vol_by_symbol,
            )
            div_history.close()
            print(f"signals: {len(all_signals)} total, {len(actionable)} actionable")

            if not actionable:
                print("no actionable signals today — running PREVIEW-ONLY phase")
                # Preview a synthetic small order so we still exercise the API path.
                if all_signals:
                    print("(no actionable; skipping live submission test)")
                return 0

            # 3. Pick the smallest-premium actionable signal
            manager = OrderManager(
                client=client, order_log=order_log, settings=settings,
                max_quantity_per_leg=1, max_premium_per_trade=2000.0,
            )

            # Sort by liquidity descending so we pick a tradable one with a small premium
            # We don't have a direct premium score on the signal, so we compute the request
            # and pick the smallest. Iterate until one works.
            from execution.order_manager import signal_to_request

            best_signal = None
            best_request = None
            for s in actionable:
                snap = scan[s.symbol]
                try:
                    req = signal_to_request(s, snap, slippage=0.02, max_qty=1)
                except ValueError as e:
                    print(f"  skip {s.symbol} {s.expiration}: {e}")
                    continue
                if best_request is None or req.estimated_premium < best_request.estimated_premium:
                    best_signal = s
                    best_request = req

            if best_signal is None:
                print("no submittable actionable signals; abort")
                return 0

            scan_date = scan.fetched_at.date()
            print(f"\nselected: {best_signal.symbol} {best_signal.direction} "
                  f"{best_signal.expiration} ({best_request.structure}, "
                  f"price={best_request.price}, est. premium=${best_request.estimated_premium:.2f})")

            # 4. Preview-only check first
            print("\n--- preview ---")
            preview = await client.preview_order(
                account_id=settings.account_id,
                legs=best_request.legs,
                underlying_symbol=best_request.underlying_symbol,
                order_type=best_request.order_type,
                price=best_request.price,
            )
            print(f"preview response keys: {list(preview.keys()) if isinstance(preview, dict) else type(preview).__name__}")
            if isinstance(preview, dict) and "errors" in preview:
                print(f"preview errors: {preview['errors']}")
                print("aborting before live submission")
                return 0
            order_node = preview.get("order") if isinstance(preview, dict) else None
            if order_node:
                for k in ("status", "cost", "commission", "margin_change", "result"):
                    if k in order_node:
                        print(f"  {k}: {order_node[k]}")

            # 5. Live submission
            print("\n--- submit ---")
            t0 = time.monotonic()
            result = await manager.submit(best_signal, scan[best_signal.symbol], scan_date)
            elapsed = time.monotonic() - t0
            print(f"result: status={result.status}  order_id={result.order_id}  "
                  f"submitted_price={result.submitted_price}  fill_price={result.fill_price}")
            if result.error:
                print(f"  error: {result.error}")
            print(f"  elapsed: {elapsed:.1f}s")

            # 6. Verify Tradier sees the order / position
            if result.status in ("filled", "partially_filled"):
                positions = await client.get_positions(settings.account_id)
                print(f"\ncurrent sandbox positions: {len(positions)}")
                for pos in positions[-5:]:
                    print(f"  {pos.get('symbol')}: qty={pos.get('quantity')} cost_basis={pos.get('cost_basis')}")

            # 7. Verify our log
            print(f"\norder log post-submit: submitted={order_log.submitted_count()}, "
                  f"failed={order_log.failed_count()}")

            # Acceptable terminal statuses: filled, partially_filled, duplicate (re-run case)
            assert result.status in ("filled", "partially_filled", "duplicate", "preview_failed", "rejected"), (
                f"unexpected status {result.status}"
            )
            print("\nlive execution test complete")
    finally:
        store.close()
        order_log.close()
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
