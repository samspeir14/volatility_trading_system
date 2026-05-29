"""Options Volatility Arbitrage Bot — orchestrator.

Entry point for autonomous operation. Wires every module onto a 5-minute
scan cadence during NYSE trading hours, manages exits live, posts a daily
summary at market close.

CLI:
    python -m main                # run forever (production mode under systemd)
    python -m main --once         # single cycle, then exit (verification)
    python -m main --summary-only # build + post daily summary, then exit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config import load_settings, load_watchlist
from data import (
    AsyncTradierClient,
    EarningsCalendar,
    HistoricalStore,
    MarketData,
    compute_log_returns,
)
from execution import OrderLog, OrderManager
from features import FeaturePipeline
from logs import DailySummary, DailySummaryBuilder, post_to_slack, setup_logging
from model import (
    BestPredictor,
    GARCHBaseline,
    LightGBMVolPredictor,
    XGBoostVolPredictor,
)
from positions import ExitManager, PositionReconciler, PositionTracker
from risk import (
    DailyKillSwitch,
    PortfolioStateBuilder,
    RiskManager,
    RiskRejectionLog,
)
from signals import DivergenceHistory, SignalGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CycleResult:
    market_open: bool
    timestamp: datetime
    equity: float | None = None
    today_total_pnl: float | None = None
    scan_contracts: int | None = None
    signals_total: int | None = None
    signals_actionable: int | None = None
    signals_approved: int | None = None
    submissions_filled: int | None = None
    submissions_failed: int | None = None
    exits_evaluated: int | None = None
    exits_closed: int | None = None
    kill_switch_active: bool = False
    error: str | None = None


def _last_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _load_routing_r2(artifact_dir: Path) -> dict[int, dict[str, float]]:
    """Read per-horizon OOS R² from latest_retrain_r2.json (written by
    tests/test_model_retraining.py). Falls back to a conservative table that
    routes to LightGBM at short horizons and treats h=21 as a tie if the JSON
    is missing or malformed — keeps the bot bootable while a retrain is
    pending."""
    fallback = {
        5:  {"lgbm": 0.25, "xgb": 0.23, "garch": -0.27},
        10: {"lgbm": 0.33, "xgb": 0.29, "garch": -0.06},
        21: {"lgbm": 0.23, "xgb": 0.29, "garch":  0.03},
    }
    json_path = artifact_dir / "latest_retrain_r2.json"
    if not json_path.exists():
        logger.warning(
            "%s missing; using fallback routing R² (run tests.test_model_retraining to refresh)",
            json_path.name,
        )
        return fallback
    try:
        with open(json_path) as f:
            payload = json.load(f)
        out: dict[int, dict[str, float]] = {}
        for h_str, r2s in payload["r2_by_horizon"].items():
            out[int(h_str)] = {
                "lgbm": float(r2s["lgbm"]),
                "xgb": float(r2s["xgb"]),
                "garch": float(r2s["garch"]),
            }
        # Guard against partial files: must contain all 3 horizons.
        if {5, 10, 21} - set(out.keys()):
            raise ValueError(f"missing horizons in {json_path.name}: {set(out.keys())}")
        logger.info(
            "routing R² loaded from %s (trained_at=%s)",
            json_path.name, payload.get("trained_at", "unknown"),
        )
        return out
    except Exception as e:
        logger.error("failed to read %s (%s); using fallback routing R²", json_path.name, e)
        return fallback


def _load_predictors(artifact_dir: Path) -> dict[int, BestPredictor]:
    """Load the best available predictor per horizon. Priority: LightGBM,
    then XGBoost, then GARCH-only. The bot keeps running on the previous
    artifact if a future cron retraining fails."""
    routing_r2 = _load_routing_r2(artifact_dir)
    predictors: dict[int, BestPredictor] = {}
    for h in (5, 10, 21):
        lgbm_files = list(artifact_dir.glob(f"lgbm_h{h}_*.joblib"))
        xgb_files = list(artifact_dir.glob(f"xgb_h{h}_*.joblib"))

        lgbm_pred: LightGBMVolPredictor | None = None
        xgb_pred: XGBoostVolPredictor | None = None

        if lgbm_files:
            newest = max(lgbm_files, key=lambda p: p.stat().st_mtime)
            lgbm_pred = LightGBMVolPredictor.load(newest)
            logger.info("loaded %s for h=%d", newest.name, h)
        if xgb_files:
            newest = max(xgb_files, key=lambda p: p.stat().st_mtime)
            xgb_pred = XGBoostVolPredictor.load(newest)
            logger.info("loaded %s for h=%d", newest.name, h)

        if lgbm_pred is None and xgb_pred is None:
            raise RuntimeError(
                f"no model artifact for h={h}; "
                f"run `python -m tests.test_model_retraining` to generate one"
            )

        garch = GARCHBaseline(refit_every=21, min_history=100)
        bp = BestPredictor(lgbm=lgbm_pred, xgb=xgb_pred, garch=garch, horizon=h)
        latest_r2 = routing_r2.get(h, {})
        bp.update_from_eval(
            lgbm_r2=latest_r2.get("lgbm", float("nan")) if lgbm_pred is not None else float("nan"),
            xgb_r2=latest_r2.get("xgb", float("nan")) if xgb_pred is not None else float("nan"),
            garch_r2=latest_r2.get("garch", -0.1),
        )
        predictors[h] = bp
    return predictors


class MainLoop:
    def __init__(
        self,
        settings,
        client: AsyncTradierClient,
        store: HistoricalStore,
        order_log: OrderLog,
        kill_switch: DailyKillSwitch,
        risk_rejection_log: RiskRejectionLog,
        divergence_history: DivergenceHistory,
        market_data: MarketData,
        feature_pipeline: FeaturePipeline,
        signal_generator: SignalGenerator,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        exit_manager: ExitManager,
        position_tracker: PositionTracker,
        position_reconciler: PositionReconciler,
        portfolio_state_builder: PortfolioStateBuilder,
        daily_summary_builder: DailySummaryBuilder,
        earnings_calendar: EarningsCalendar | None = None,
        watchlist_symbols: list[str] | None = None,
        slack_webhook_url: str | None = None,
        scan_interval_seconds: int = 300,
        kill_switch_threshold_pct: float = -0.05,
    ):
        self._settings = settings
        self._client = client
        self._store = store
        self._order_log = order_log
        self._kill_switch = kill_switch
        self._risk_rejection_log = risk_rejection_log
        self._divergence_history = divergence_history
        self._market_data = market_data
        self._feature_pipeline = feature_pipeline
        self._signal_generator = signal_generator
        self._risk_manager = risk_manager
        self._order_manager = order_manager
        self._exit_manager = exit_manager
        self._position_tracker = position_tracker
        self._reconciler = position_reconciler
        self._builder = portfolio_state_builder
        self._summary_builder = daily_summary_builder
        self._earnings_calendar = earnings_calendar
        self._watchlist_symbols = watchlist_symbols or []
        self._slack_url = slack_webhook_url
        self._scan_interval = scan_interval_seconds
        self._kill_pct = kill_switch_threshold_pct
        self._summary_posted_for_date: date | None = None
        self._last_snapshot = None
        self._last_snapshot_date: date | None = None

    async def run_once(self) -> CycleResult:
        """Single cycle. Returns what happened. Tested with mocks."""
        now = datetime.now(timezone.utc)

        # 1. Clock check
        try:
            clock = await self._client.get_clock()
        except Exception as e:
            logger.error("clock check failed: %s", e)
            return CycleResult(market_open=False, timestamp=now, error=str(e))

        state = clock.get("state")
        if state != "open":
            logger.info("market state=%s, skipping cycle", state)
            return CycleResult(market_open=False, timestamp=now)

        # 2. Scan
        scan = await self._market_data.scan(expiration_window=(3, 60))
        scan_contracts = scan.total_contracts
        today = scan.fetched_at.date()

        # 2b. Reconcile log against Tradier's actual positions BEFORE snapshot,
        # so expired / assigned positions don't leak into marks, exposure
        # counts, exit decisions, or risk gates.
        try:
            await self._reconciler.reconcile(today)
        except Exception as e:
            logger.error("reconciliation failed: %s — continuing with stale log", e)

        # 2c. Reconcile pending closes — cancel stale ones, reconcile any that
        # filled between cycles. Runs before snapshot so freshly-reconciled
        # closes don't leave the opening in open_marks for this cycle.
        try:
            await self._order_manager.reconcile_pending_closes(now)
        except Exception as e:
            logger.error("stale-close reconcile failed: %s — continuing", e)

        # 2d. Pull chains for any still-open position whose expiration has aged
        # below the scan's lower window bound. The scan only fetches (3, 60)-day
        # expirations, so a position at DTE < 3 has no legs in the scan — it
        # silently drops out of mark-to-market and never reaches the
        # expiration-proximity exit. Augment the scan in place (after reconcile,
        # so freshly-expired positions are excluded) and let the single scan
        # object flow to snapshot, exits, and signals alike.
        try:
            needed = self._open_position_expirations()
            scan = await self._market_data.fetch_missing_position_chains(
                scan, needed, today,
            )
        except Exception as e:
            logger.error("scan augmentation failed: %s — near-expiry positions "
                         "may not mark this cycle", e)

        # 3. Snapshot
        snapshot = await self._builder.snapshot(scan)

        # Refresh earnings calendar (no-op if already done today)
        if self._earnings_calendar is not None:
            try:
                await self._earnings_calendar.refresh_if_stale(
                    today=today, symbols=self._watchlist_symbols or None,
                )
            except Exception as e:
                logger.warning("earnings calendar refresh raised %s — failing open", e)

        # 4. Kill switch evaluation
        kill_active = self._kill_switch.evaluate_and_maybe_trigger(
            today=today,
            total_pnl_today=snapshot.today_total_pnl,
            starting_equity=snapshot.starting_equity_today,
            threshold_pct=self._kill_pct,
        )

        # 5. Always manage existing positions (even with kill switch active)
        exit_decisions = []
        exits_closed = 0
        if snapshot.open_marks:
            feature_rows = self._build_feature_rows()
            returns_by_symbol = self._build_returns_dict()
            exit_decisions = self._exit_manager.evaluate(
                snapshot.open_marks, scan, feature_rows, returns_by_symbol,
            )
            exit_results = await self._exit_manager.execute(
                exit_decisions, dry_run=False,
            )
            for decision, result in exit_results:
                if result is not None and result.status == "filled":
                    exits_closed += 1

        # 6. New entries only when kill switch NOT active
        signals_total = signals_actionable = signals_approved = 0
        submissions_filled = submissions_failed = 0
        if not kill_active:
            feature_rows = self._build_feature_rows()
            returns_by_symbol = self._build_returns_dict()
            actionable, all_signals = self._signal_generator.generate(
                scan=scan,
                feature_rows=feature_rows,
                returns_by_symbol=returns_by_symbol,
                top_n=10,
            )
            signals_total = len(all_signals)
            signals_actionable = len(actionable)

            risk_decisions = self._risk_manager.gate(actionable, scan, snapshot)
            for d in risk_decisions:
                if d.approved:
                    snap = scan[d.signal.symbol]
                    res = await self._order_manager.submit(d.signal, snap, today)
                    if res.status == "filled":
                        submissions_filled += 1
                        signals_approved += 1
                    else:
                        submissions_failed += 1
                else:
                    self._risk_rejection_log.record_rejection(d, datetime.now(timezone.utc))

        result = CycleResult(
            market_open=True,
            timestamp=now,
            equity=snapshot.equity,
            today_total_pnl=snapshot.today_total_pnl,
            scan_contracts=scan_contracts,
            signals_total=signals_total,
            signals_actionable=signals_actionable,
            signals_approved=signals_approved,
            submissions_filled=submissions_filled,
            submissions_failed=submissions_failed,
            exits_evaluated=len(exit_decisions),
            exits_closed=exits_closed,
            kill_switch_active=kill_active,
        )
        self._log_cycle(result)
        # Stash latest snapshot so run_forever can post the daily summary on close-transition
        self._last_snapshot = snapshot
        self._last_snapshot_date = today
        return result

    async def run_forever(self) -> int:
        """Outer loop: handles market-hours sleeping + per-cycle exception isolation +
        end-of-day summary post on the open→closed state transition."""
        logger.info("MainLoop.run_forever() starting (scan_interval=%ds)", self._scan_interval)
        last_state: str | None = None
        while True:
            try:
                clock = await self._client.get_clock()
                current_state = clock.get("state")

                # State transition from open → not-open: post the daily summary
                if (last_state == "open" and current_state != "open"
                        and self._last_snapshot is not None
                        and self._last_snapshot_date is not None
                        and self._summary_posted_for_date != self._last_snapshot_date):
                    logger.info("market just closed (state %s -> %s), posting daily summary",
                                last_state, current_state)
                    await self.post_daily_summary(
                        self._last_snapshot_date, self._last_snapshot,
                    )
                    self._summary_posted_for_date = self._last_snapshot_date

                last_state = current_state

                if current_state != "open":
                    sleep_seconds = self._sleep_seconds_until_open(clock)
                    logger.info("market %s, sleeping %.0fs until next change",
                                current_state, sleep_seconds)
                    await asyncio.sleep(min(sleep_seconds, 3600))  # cap at 1h re-poll
                    continue

                await self.run_once()
                await asyncio.sleep(self._scan_interval)
            except Exception as e:
                logger.exception("cycle exception, continuing: %s", e)
                await asyncio.sleep(self._scan_interval)

    async def post_daily_summary(self, today: date, snapshot) -> None:
        """Build and post (file + Slack) the daily summary."""
        try:
            summary = self._summary_builder.build(today, snapshot)
            logger.info(
                "daily_summary date=%s equity=$%.2f pnl=$%+.2f opened=%d closed=%d "
                "signals=%d approved=%d rejections=%d kill_switch=%s",
                summary.date, summary.ending_equity, summary.total_pnl,
                summary.positions_opened_today, summary.positions_closed_today,
                summary.signals_total, summary.signals_approved,
                summary.risk_rejections_total, summary.kill_switch_activated,
            )
            if self._slack_url:
                post_to_slack(self._slack_url, summary)
        except Exception as e:
            logger.error("daily summary failed: %s", e)

    def _open_position_expirations(self) -> dict[str, set[date]]:
        """symbol → set of expirations for every still-open logged position.
        Feeds scan augmentation so near-expiry legs (DTE below the scan window)
        can still be marked. Reads the order log directly; it's the authoritative
        open-position source and the read is a cheap local query."""
        out: dict[str, set[date]] = {}
        for row in self._order_log.open_unclosed_positions():
            try:
                exp = date.fromisoformat(row["expiration"])
            except (TypeError, ValueError, KeyError):
                continue
            out.setdefault(row["symbol"], set()).add(exp)
        return out

    def _build_feature_rows(self) -> dict[str, pd.DataFrame]:
        """Build the latest feature row per symbol from the cache."""
        end = _last_weekday(date.today())
        start = end - timedelta(days=730)
        feature_df = self._feature_pipeline.build_features(start, end)
        rows = {}
        for sym in (t.symbol for t in self._feature_pipeline._watchlist):
            try:
                sym_df = feature_df.loc[sym]
                rows[sym] = sym_df.loc[[sym_df.index.max()]]
            except KeyError:
                continue
        return rows

    def _build_returns_dict(self) -> dict[str, pd.Series]:
        end = _last_weekday(date.today())
        start = end - timedelta(days=730)
        out = {}
        for ticker in self._feature_pipeline._watchlist:
            try:
                bars = self._store.get_bars(ticker.symbol, start, end)
                if not bars.empty:
                    out[ticker.symbol] = compute_log_returns(bars["close"])
            except Exception as e:
                logger.warning("could not build returns for %s: %s", ticker.symbol, e)
        return out

    def _log_cycle(self, result: CycleResult) -> None:
        if not result.market_open:
            return
        pnl_pct = (
            result.today_total_pnl / result.equity * 100
            if result.equity and result.equity > 0 else 0.0
        )
        logger.info(
            "cycle_complete equity=$%.2f pnl_today=$%+.2f (%+.2f%%) "
            "scan_contracts=%d signals=%d actionable=%d approved=%d "
            "submitted=%d failed=%d exits=%d kill_switch=%s",
            result.equity or 0.0, result.today_total_pnl or 0.0, pnl_pct,
            result.scan_contracts or 0, result.signals_total or 0,
            result.signals_actionable or 0, result.signals_approved or 0,
            result.submissions_filled or 0, result.submissions_failed or 0,
            result.exits_closed or 0, result.kill_switch_active,
        )

    def _sleep_seconds_until_open(self, clock: dict) -> float:
        """Use Tradier's next_change to sleep precisely. Falls back to 5min if absent."""
        next_change_str = clock.get("next_change")
        if not next_change_str:
            return self._scan_interval
        try:
            # Tradier returns "2026-04-28T13:30:00Z" or similar
            if next_change_str.endswith("Z"):
                next_change_str = next_change_str[:-1] + "+00:00"
            next_change = datetime.fromisoformat(next_change_str)
            now = datetime.now(timezone.utc)
            delta = (next_change - now).total_seconds() + 60  # +60s buffer
            return max(60.0, delta)
        except (ValueError, TypeError):
            return float(self._scan_interval)


def build_main_loop(settings, client: AsyncTradierClient) -> tuple[MainLoop, list]:
    """Wire every component. Returns the loop + list of resources to close on shutdown."""
    closeables = []

    artifact_dir = Path(__file__).resolve().parent / "model" / "artifacts"
    cache_dir = settings.cache_db_path.parent

    store = HistoricalStore(settings.cache_db_path)
    closeables.append(store)

    order_log = OrderLog(cache_dir / "order_log.db")
    closeables.append(order_log)

    kill_switch = DailyKillSwitch(cache_dir / "risk_state.db")
    closeables.append(kill_switch)

    risk_rejection_log = RiskRejectionLog(cache_dir / "risk_state.db")
    closeables.append(risk_rejection_log)

    divergence_history = DivergenceHistory(cache_dir / "divergence_history.db")
    closeables.append(divergence_history)

    earnings_calendar = EarningsCalendar(
        db_path=cache_dir / "earnings_calendar.db",
        api_key=settings.finnhub_api_key,
    )
    closeables.append(earnings_calendar)
    if not settings.finnhub_api_key:
        logger.warning(
            "FINNHUB_API_KEY not set — earnings filter will fail open until "
            "the key is added to .env"
        )

    watchlist = load_watchlist()
    market_data = MarketData(client, watchlist)
    feature_pipeline = FeaturePipeline(
        store, watchlist,
        garch_min_history=100, garch_refit_every=21,
    )
    position_tracker = PositionTracker(client=client, order_log=order_log, settings=settings)
    position_reconciler = PositionReconciler(
        client=client, order_log=order_log, account_id=settings.account_id,
    )
    portfolio_state_builder = PortfolioStateBuilder(
        client=client, order_log=order_log,
        position_tracker=position_tracker, watchlist=watchlist,
        kill_switch=kill_switch,
    )

    predictors = _load_predictors(artifact_dir)

    signal_generator = SignalGenerator(
        predictors_by_horizon=predictors,
        history_store=divergence_history,
        cross_sectional_z_threshold=1.5,
        max_divergence=0.25,
        earnings_calendar=earnings_calendar,
        earnings_filter_enabled=settings.earnings_filter_enabled,
        earnings_buffer_days=settings.earnings_buffer_days,
    )

    risk_manager = RiskManager(
        watchlist=watchlist,
        max_per_trade_loss_pct=0.02,
        max_per_ticker_exposure_pct=0.05,
        max_per_sector_positions=4,
        max_portfolio_delta_pct=0.05,
        max_portfolio_gamma_pct=0.01,
        max_portfolio_vega_pct=0.05,
        daily_loss_kill_switch_pct=-0.05,
        min_buying_power_buffer_pct=0.05,
        kill_switch=kill_switch,
    )

    order_manager = OrderManager(
        client=client, order_log=order_log, settings=settings,
        stale_order_threshold_minutes=settings.stale_order_threshold_minutes,
        max_close_retries=settings.max_close_retries,
    )

    exit_manager = ExitManager(
        position_tracker=position_tracker,
        order_manager=order_manager,
        predictors_by_horizon=predictors,
    )

    daily_summary_builder = DailySummaryBuilder(
        order_log=order_log,
        divergence_history=divergence_history,
        risk_rejection_log=risk_rejection_log,
        kill_switch=kill_switch,
        earnings_calendar=earnings_calendar,
    )

    loop = MainLoop(
        settings=settings,
        client=client,
        store=store,
        order_log=order_log,
        kill_switch=kill_switch,
        risk_rejection_log=risk_rejection_log,
        divergence_history=divergence_history,
        market_data=market_data,
        feature_pipeline=feature_pipeline,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        order_manager=order_manager,
        exit_manager=exit_manager,
        position_tracker=position_tracker,
        position_reconciler=position_reconciler,
        portfolio_state_builder=portfolio_state_builder,
        daily_summary_builder=daily_summary_builder,
        earnings_calendar=earnings_calendar,
        watchlist_symbols=[t.symbol for t in watchlist],
        slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL"),
        scan_interval_seconds=int(os.environ.get("SCAN_INTERVAL_SECONDS", "300")),
    )
    return loop, closeables


async def _run(args) -> int:
    settings = load_settings()
    log_dir = Path(__file__).resolve().parent / "logs"
    setup_logging(log_dir=log_dir, level=os.environ.get("LOG_LEVEL", "INFO"))

    logger.info("starting bot env=%s account=%s", settings.env, settings.account_id)

    async with AsyncTradierClient(settings) as client:
        loop, closeables = build_main_loop(settings, client)
        try:
            if args.once:
                logger.info("running one cycle and exiting (--once)")
                result = await loop.run_once()
                logger.info("cycle result: market_open=%s, error=%s",
                            result.market_open, result.error)
                return 0 if result.error is None else 1

            if args.summary_only:
                logger.info("building + posting daily summary, then exiting (--summary-only)")
                # Need a snapshot for the summary; do a minimal scan
                scan = await loop._market_data.scan(expiration_window=(3, 60))
                snapshot = await loop._builder.snapshot(scan)
                today = scan.fetched_at.date()
                await loop.post_daily_summary(today, snapshot)
                return 0

            return await loop.run_forever()
        finally:
            for c in closeables:
                try:
                    c.close()
                except Exception:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Options Volatility Arbitrage Bot — orchestrator",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle and exit (verification mode)",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Build and post the daily summary, then exit (Slack format check)",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
