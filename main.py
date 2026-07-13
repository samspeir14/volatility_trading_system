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
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from config import SMALL_WATCHLIST_PATH, Settings, load_settings, load_watchlist
from data import (
    AsyncTradierClient,
    EarningsCalendar,
    HistoricalStore,
    MacroCalendar,
    MarketData,
    compute_log_returns,
)
from execution import OrderLog, OrderManager
from features import FeaturePipeline
from logs import DailySummary, DailySummaryBuilder, post_text, post_to_slack, setup_logging
from model import (
    BestPredictor,
    GARCHBaseline,
    LightGBMVolPredictor,
    XGBoostVolPredictor,
)
from positions import ExitManager, PositionReconciler, PositionTracker
from risk import (
    BarsFreshnessGuard,
    DailyKillSwitch,
    DrawdownBreaker,
    HaltFlag,
    PortfolioStateBuilder,
    RiskManager,
    RiskRejectionLog,
    write_heartbeat,
)
from signals import DivergenceHistory, SignalGenerator

logger = logging.getLogger(__name__)

# Scan only the expirations that can actually produce a trade. The signal
# generator won't open a new position below DTE MIN_ENTRY_DTE (7) or above
# MAX_DTE (45) — so fetching chains out to 60 DTE was pure waste that bloated
# the per-cycle option-chain fan-out and exhausted the rate limiter every cycle.
# The lower bound stays at 3 (below MIN_ENTRY_DTE) so already-open positions at
# DTE 3 still mark off the scan; anything below that is pulled in by
# fetch_missing_position_chains.
SCAN_EXPIRATION_WINDOW = (3, 45)
# Harvest mode never enters above DTE 15 and its positions ride to expiry, so
# chains past 16 are dead weight — and the narrower window is what pays the
# rate-limiter bill for the larger watchlist (~5 calls/name at (3,45) vs ~3 at
# (3,16), against the 180/min budget). Legacy model-mode positions above the
# window still mark: fetch_missing_position_chains pulls any open-position
# expiration the scan didn't cover, above or below. Side effect: while in
# harvest mode the divergence log only accumulates DTE ≤ 16 rows — fine for
# the prospective accuracy tests, which target the short tenor anyway.
SCAN_EXPIRATION_WINDOW_HARVEST = (3, 16)


def _scan_window(settings: Settings) -> tuple[int, int]:
    return (SCAN_EXPIRATION_WINDOW_HARVEST if settings.strategy_mode == "harvest"
            else SCAN_EXPIRATION_WINDOW)


# NEW entries only submit inside this ET window. Options spreads are widest in
# the opening minutes (overnight risk repricing, thin books) and erratic into
# the close (MOC imbalances) — entering there donates edge to the market
# maker. Signals are still generated and logged outside the window (the
# divergence history and risk-rejection diagnostics don't stop); only order
# submission is held until the next in-window cycle.
ENTRY_WINDOW_ET = (dt_time(9, 45), dt_time(15, 30))
_EASTERN = ZoneInfo("America/New_York")


def _within_entry_window(now_utc: datetime) -> bool:
    et = now_utc.astimezone(_EASTERN).time()
    return ENTRY_WINDOW_ET[0] <= et <= ENTRY_WINDOW_ET[1]


@dataclass(frozen=True)
class RiskCalibration:
    """The equity-relative risk knobs plus the few absolute-dollar backstops,
    bundled so the two harvest profiles stay side-by-side and comparable."""
    max_per_trade_loss_pct: float
    max_per_ticker_exposure_pct: float
    max_per_sector_positions: int
    max_portfolio_risk_pct: float
    max_portfolio_delta_pct: float
    max_portfolio_gamma_pct: float
    max_portfolio_vega_pct: float
    daily_loss_kill_switch_pct: float
    min_buying_power_buffer_pct: float
    max_premium_per_trade: float
    min_credit: float
    # Relative credit floor: reject condors collecting less than this fraction
    # of their wing width — a per-trade EV test that scales across cheap and
    # expensive names where the absolute min_credit can't.
    min_credit_to_width: float
    # Cap on RiskManager's sized quantity. Orders always submit 1-lot (signal
    # legs are quantity=1 and RiskDecision.quantity is never applied), so any
    # value above 1 books phantom multi-lot risk into the committed-risk /
    # Greek / margin projections. Standard keeps the historical 10 to preserve
    # paper behavior; small uses 1 so the gates account for what actually
    # trades — at a $1,200 wing-risk cap, $130-250 of phantom commitment per
    # approval would otherwise throttle the ladder at ~half its design size.
    max_contracts_per_trade: int


# Per-trade 1.5% + portfolio wing-risk 20%: the cap bounds worst-case
# book drawdown (all condors through their wings at once, March-2020
# shape) by construction — ~13 concurrent full-size positions, entries
# auto-throttle when the ladder is full. 20% is PAPER calibration: more
# positions = faster friction measurement, and fake drawdowns are
# tuition. Revisit (≤12%) before any real-money deployment. At
# 1.5%/$100k, one contract of AMD/CAT/GS (price × IV too big) doesn't
# fit — they were already over the old 2% budget; META/TSLA/UNH fit at
# the short end of the entry window only.
CALIBRATION_STANDARD = RiskCalibration(
    max_per_trade_loss_pct=0.015,
    max_per_ticker_exposure_pct=0.05,
    max_per_sector_positions=4,
    max_portfolio_risk_pct=0.20,
    max_portfolio_delta_pct=0.05,
    max_portfolio_gamma_pct=0.01,
    max_portfolio_vega_pct=0.05,
    daily_loss_kill_switch_pct=-0.05,
    min_buying_power_buffer_pct=0.05,
    max_premium_per_trade=5000.0,
    min_credit=0.0,
    min_credit_to_width=0.25,
    max_contracts_per_trade=10,
)

# small_harvest: same strategy, ~$10k bankroll, watchlist_small.yaml. Orders
# are 1-lot, so the per-trade pct is an eligibility gate, not a size: at 2.5%
# the $250 budget admits the whole cheap watchlist's 1σ condors ($40-$220 max
# loss) with the pricier names (BAC/SLV/B) fitting at the short-DTE end only.
# Portfolio wing risk 12% is the real-money number the 20% comment above
# defers — this profile exists to go live. Gamma cap 10%, not 1%: the gamma
# gate sums raw gamma × 100, and raw ATM gamma scales as 1/(S·σ·√T), so a $14
# name carries ~15× the raw gamma of a $170 one — at 1% the cap would reject
# a cheap-name book after ~2 positions; 10% admits the intended ~10-15
# position ladder while still capping a pile-up. Premium backstop $500 (5% of
# bankroll, mirroring $5k/5% at $100k). Min credit $0.25: the execution ladder
# starts at mid but can concede up to 3%, and a round trip costs ~$1-4 in
# fees, so thinner credits are structurally unprofitable.
CALIBRATION_SMALL = RiskCalibration(
    max_per_trade_loss_pct=0.025,
    max_per_ticker_exposure_pct=0.05,
    max_per_sector_positions=4,
    max_portfolio_risk_pct=0.12,
    max_portfolio_delta_pct=0.05,
    max_portfolio_gamma_pct=0.10,
    max_portfolio_vega_pct=0.05,
    daily_loss_kill_switch_pct=-0.05,
    min_buying_power_buffer_pct=0.05,
    max_premium_per_trade=500.0,
    min_credit=0.25,
    min_credit_to_width=0.25,
    max_contracts_per_trade=1,
)


def _calibration(settings: Settings) -> RiskCalibration:
    return (CALIBRATION_SMALL if settings.harvest_profile == "small"
            else CALIBRATION_STANDARD)


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
    # Every reason the entry side was blocked this cycle (kill switch, manual
    # HALT, drawdown breaker, stale bars). Exits always run regardless.
    entry_blocks: tuple[str, ...] = ()
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
        halt_flag: HaltFlag | None = None,
        drawdown_breaker: DrawdownBreaker | None = None,
        bars_guard: BarsFreshnessGuard | None = None,
        heartbeat_path: Path | None = None,
        error_streak_alert_threshold: int = 5,
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
        self._halt_flag = halt_flag
        self._drawdown_breaker = drawdown_breaker
        self._bars_guard = bars_guard
        self._heartbeat_path = heartbeat_path
        self._summary_posted_for_date: date | None = None
        self._last_snapshot = None
        self._last_snapshot_date: date | None = None
        # Alert dedupe: only Slack a block reason the first cycle it appears.
        self._prev_entry_blocks: set[str] = set()
        # Consecutive failed cycles → one Slack alert per streak.
        self._error_streak = 0
        self._error_streak_alerted = False
        self._error_streak_threshold = error_streak_alert_threshold

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

        # 1b. Refresh daily bars through the last completed trading day.
        # ensure_data is incremental (latest cached date + 1 forward), so this
        # is ~zero API calls when the cache is current and self-heals any gap
        # after an outage. End is the last weekday strictly BEFORE today:
        # cycles only run while the market is open, so today's bar is partial,
        # and caching it would freeze it permanently (INSERT OR REPLACE is
        # never revisited once latest_date moves past it).
        bars_end = _last_weekday(now.date() - timedelta(days=1))
        try:
            await self._feature_pipeline.ensure_data(self._client, end=bars_end)
        except Exception as e:
            # Failing soft is fine for a single miss — the BarsFreshnessGuard
            # below blocks entries if the cache actually goes stale.
            logger.warning(
                "daily bar refresh failed: %s — continuing on cached bars", e
            )

        # 2. Scan
        scan = await self._market_data.scan(expiration_window=_scan_window(self._settings))
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
        # below the scan's lower window bound. The scan only fetches expirations
        # in SCAN_EXPIRATION_WINDOW, so a position at DTE < 3 has no legs in it —
        # it
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

        # 4b. Entry guards: manual HALT, multi-day drawdown breakers, bars
        # freshness. Any hit blocks NEW entries only — exit management below
        # runs unconditionally. Reasons are stable strings per activation so
        # the transition alert fires once, not every cycle.
        entry_blocks: list[str] = []
        if kill_active:
            entry_blocks.append("daily kill switch active")
        if self._halt_flag is not None:
            halt_reason = self._halt_flag.reason()
            if halt_reason is not None:
                entry_blocks.append(f"manual HALT: {halt_reason}")
        if self._drawdown_breaker is not None and snapshot.equity:
            breaker_reason = self._drawdown_breaker.evaluate_and_maybe_trigger(
                today, snapshot.equity,
            )
            if breaker_reason is not None:
                entry_blocks.append(breaker_reason)
        stale_symbols: list[str] = []
        if self._bars_guard is not None:
            freshness = self._bars_guard.check(bars_end)
            stale_symbols = freshness.stale_symbols
            if freshness.block_reason is not None:
                entry_blocks.append(freshness.block_reason)
        self._alert_new_blocks(entry_blocks)

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

        # 6. New entries only when no guard tripped
        signals_total = signals_actionable = signals_approved = 0
        submissions_filled = submissions_failed = 0
        if not entry_blocks:
            feature_rows = self._build_feature_rows()
            returns_by_symbol = self._build_returns_dict()
            # Individually-stale symbols (below the systemic block threshold)
            # are dropped from the ENTRY universe only — their bars are old,
            # so their predictions and spread vetoes would be fiction. Exits
            # above already handled them off live marks.
            for sym in stale_symbols:
                if sym in feature_rows or sym in returns_by_symbol:
                    logger.warning(
                        "excluding %s from entries: daily bars stale", sym,
                    )
                feature_rows.pop(sym, None)
                returns_by_symbol.pop(sym, None)
            actionable, all_signals = self._signal_generator.generate(
                scan=scan,
                feature_rows=feature_rows,
                returns_by_symbol=returns_by_symbol,
                top_n=10,
                vix_term_ratio=self._vix_term_structure_ratio(),
            )
            signals_total = len(all_signals)
            signals_actionable = len(actionable)

            in_entry_window = _within_entry_window(now)
            if actionable and not in_entry_window:
                logger.info(
                    "outside entry window %s-%s ET — holding %d actionable "
                    "signal(s) until the next in-window cycle",
                    ENTRY_WINDOW_ET[0], ENTRY_WINDOW_ET[1], len(actionable),
                )
            risk_decisions = self._risk_manager.gate(actionable, scan, snapshot)
            for d in risk_decisions:
                if d.approved and in_entry_window:
                    snap = scan[d.signal.symbol]
                    res = await self._order_manager.submit(d.signal, snap, today)
                    if res.status == "filled":
                        submissions_filled += 1
                        signals_approved += 1
                    else:
                        submissions_failed += 1
                elif not d.approved:
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
            entry_blocks=tuple(entry_blocks),
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
                # Heartbeat = "the loop is alive and can reach the broker".
                # Written open or closed (closed-market sleeps are capped at
                # 1h, so a beat older than ~75 min means the process is gone —
                # that's the threshold scripts/heartbeat_check.py alerts on).
                if self._heartbeat_path is not None:
                    write_heartbeat(self._heartbeat_path, current_state or "unknown")

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

                result = await self.run_once()
                if result.error is not None:
                    self._note_cycle_error(result.error)
                else:
                    self._note_cycle_success()
                await asyncio.sleep(self._scan_interval)
            except Exception as e:
                logger.exception("cycle exception, continuing: %s", e)
                if self._heartbeat_path is not None:
                    # Loop is alive even though the cycle failed; state="error"
                    # keeps the dead-man switch honest without firing it.
                    write_heartbeat(self._heartbeat_path, "error")
                self._note_cycle_error(str(e))
                await asyncio.sleep(self._scan_interval)

    def _note_cycle_error(self, error: str) -> None:
        """Consecutive-error breaker: failing cycles already mean no trading,
        so this alerts (once per streak) rather than halts."""
        self._error_streak += 1
        if (self._error_streak >= self._error_streak_threshold
                and not self._error_streak_alerted):
            self._error_streak_alerted = True
            message = (
                f"⚠️ {self._error_streak} consecutive cycle errors — bot is up "
                f"but not trading. Last error: {error}"
            )
            logger.error(message)
            if self._slack_url:
                post_text(self._slack_url, message)

    def _note_cycle_success(self) -> None:
        if self._error_streak_alerted and self._slack_url:
            post_text(
                self._slack_url,
                f"✅ cycles recovered after {self._error_streak} consecutive errors",
            )
        self._error_streak = 0
        self._error_streak_alerted = False

    def _alert_new_blocks(self, blocks: list[str]) -> None:
        """Slack-alert each entry-block reason the first cycle it appears.
        Reasons are stable strings per activation, so re-alerts only happen on
        genuinely new conditions (or a changed HALT file message)."""
        new = [b for b in blocks if b not in self._prev_entry_blocks]
        self._prev_entry_blocks = set(blocks)
        for block in new:
            logger.warning("entries blocked: %s", block)
        if new and self._slack_url:
            post_text(self._slack_url, "🛑 Entries blocked: " + " | ".join(new))

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

    def _vix_term_structure_ratio(self) -> float | None:
        """Latest cached VIX/VIX3M close ratio for the harvest backwardation
        veto, or None (fail open) when either series is missing/stale. Daily
        closes are enough: backwardation episodes persist for days, and the
        crash day itself is caught by the per-name extreme-spread veto on
        live quotes. Both series are already in the cache — they're in
        MARKET_INDICES for the vix3m_to_vix feature."""
        try:
            end = date.today()
            start = end - timedelta(days=14)
            vix = self._store.get_bars("VIX", start, end)
            vix3m = self._store.get_bars("VIX3M", start, end)
            if vix.empty or vix3m.empty:
                return None
            vix3m_close = float(vix3m["close"].iloc[-1])
            if vix3m_close <= 0:
                return None
            return float(vix["close"].iloc[-1]) / vix3m_close
        except Exception as e:
            logger.warning("VIX term-structure ratio unavailable: %s", e)
            return None

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
            "submitted=%d failed=%d exits=%d kill_switch=%s entry_blocks=%s",
            result.equity or 0.0, result.today_total_pnl or 0.0, pnl_pct,
            result.scan_contracts or 0, result.signals_total or 0,
            result.signals_actionable or 0, result.signals_approved or 0,
            result.submissions_filled or 0, result.submissions_failed or 0,
            result.exits_closed or 0, result.kill_switch_active,
            list(result.entry_blocks) or "none",
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

    # Multi-day breakers share risk_state.db (they read the equity_snapshots
    # rows the daily kill switch records); the HALT flag and heartbeat live
    # next to the caches so one directory holds all operational state.
    drawdown_breaker = DrawdownBreaker(cache_dir / "risk_state.db")
    closeables.append(drawdown_breaker)
    halt_flag = HaltFlag(cache_dir / "HALT")

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

    watchlist = (load_watchlist(SMALL_WATCHLIST_PATH)
                 if settings.harvest_profile == "small" else load_watchlist())
    market_data = MarketData(client, watchlist)
    feature_pipeline = FeaturePipeline(
        store, watchlist,
        garch_min_history=100, garch_refit_every=21,
    )
    position_tracker = PositionTracker(client=client, order_log=order_log, settings=settings)
    position_reconciler = PositionReconciler(
        client=client, order_log=order_log, account_id=settings.account_id,
        per_contract_fee=settings.per_contract_fee,
    )
    portfolio_state_builder = PortfolioStateBuilder(
        client=client, order_log=order_log,
        position_tracker=position_tracker, watchlist=watchlist,
        kill_switch=kill_switch,
    )

    predictors = _load_predictors(artifact_dir)

    # Index ETFs carry a variance-risk premium (realized < implied), so buying
    # their straddles bleeds — every SPY long straddle in the live log lost.
    # Bar them from the BUY side; they stay eligible for the SELL/iron-condor
    # side that harvests the premium. "bonds" (TLT) is an index product with
    # the same structural premium — same exclusion, different risk bucket for
    # the sector-concentration gate.
    etf_symbols = frozenset(
        t.symbol for t in watchlist if t.sector in ("etf", "bonds")
    )

    cal = _calibration(settings)

    signal_generator = SignalGenerator(
        predictors_by_horizon=predictors,
        history_store=divergence_history,
        cross_sectional_z_threshold=1.5,
        max_divergence=0.25,
        earnings_calendar=earnings_calendar,
        earnings_filter_enabled=settings.earnings_filter_enabled,
        earnings_buffer_days=settings.earnings_buffer_days,
        long_straddle_excluded_symbols=etf_symbols,
        strategy_mode=settings.strategy_mode,
        min_credit=cal.min_credit,
        min_credit_to_width=cal.min_credit_to_width,
        # FOMC/CPI are the "earnings" of the rate/index-linked names — the
        # same etf+bonds set barred from long straddles. Single-name equities
        # are not macro-gated (the VRP was fat through every macro day).
        macro_calendar=MacroCalendar(),
        macro_sensitive_symbols=etf_symbols,
    )
    logger.info(
        "strategy mode: %s (profile: %s, %d-name watchlist)",
        settings.strategy_mode, settings.harvest_profile, len(watchlist),
    )

    # The rationale for both parameter sets lives on CALIBRATION_STANDARD /
    # CALIBRATION_SMALL at the top of this module.
    risk_manager = RiskManager(
        watchlist=watchlist,
        max_per_trade_loss_pct=cal.max_per_trade_loss_pct,
        max_per_ticker_exposure_pct=cal.max_per_ticker_exposure_pct,
        max_per_sector_positions=cal.max_per_sector_positions,
        max_portfolio_risk_pct=cal.max_portfolio_risk_pct,
        max_portfolio_delta_pct=cal.max_portfolio_delta_pct,
        max_portfolio_gamma_pct=cal.max_portfolio_gamma_pct,
        max_portfolio_vega_pct=cal.max_portfolio_vega_pct,
        daily_loss_kill_switch_pct=cal.daily_loss_kill_switch_pct,
        min_buying_power_buffer_pct=cal.min_buying_power_buffer_pct,
        kill_switch=kill_switch,
        max_quantity_per_leg=cal.max_contracts_per_trade,
    )

    order_manager = OrderManager(
        client=client, order_log=order_log, settings=settings,
        max_premium_per_trade=cal.max_premium_per_trade,
        stale_order_threshold_minutes=settings.stale_order_threshold_minutes,
        max_close_retries=settings.max_close_retries,
    )

    exit_manager = ExitManager(
        position_tracker=position_tracker,
        order_manager=order_manager,
        predictors_by_horizon=predictors,
        thesis_exit_enabled=(settings.strategy_mode != "harvest"),
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
        # The kill switch that actually trips lives here, not in RiskManager
        # (whose daily_loss_kill_switch_pct is stored but never read) — wire
        # the calibration through so the knob is real for both profiles.
        kill_switch_threshold_pct=cal.daily_loss_kill_switch_pct,
        halt_flag=halt_flag,
        drawdown_breaker=drawdown_breaker,
        bars_guard=BarsFreshnessGuard(store, [t.symbol for t in watchlist]),
        heartbeat_path=cache_dir / "heartbeat.json",
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
                scan = await loop._market_data.scan(expiration_window=_scan_window(settings))
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
