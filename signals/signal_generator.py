from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.async_client import OptionContract
from data.earnings_calendar import EarningsCalendar
from data.macro_calendar import MacroCalendar
from data.market_data import ScanResult
from features.target import GK_EPS
from model.best_predictor import BestPredictor
from model.h1_predictor import H1DeviationPredictor
from model.term_structure import project_term_vol
from signals.cost_gate import evaluate_cost_gate
from signals.divergence_history import DivergenceHistory

logger = logging.getLogger(__name__)


# Legacy-only (MODEL_PIPELINE=legacy): the multi-horizon models the h=1
# pipeline replaced. interpolate_horizon / blend_prediction below serve only
# that path.
TRAINED_HORIZONS: tuple[int, ...] = (5, 10, 21)
TRADING_DAYS_PER_YEAR = 252
# Quoted (Black-Scholes) IV annualizes over calendar time, so horizon scaling
# with a CALENDAR dte must divide by 365. Mixing calendar dte with 252
# overstates sqrt(T) by ~20% — the old wing-placement bug. The h=1 term
# projection (model.term_structure) converts calendar DTE to trading days
# internally for the same reason.
CALENDAR_DAYS_PER_YEAR = 365
MIN_DTE = 4
MAX_DTE = 45

# Entry DTE window for NEW positions, [MIN_ENTRY_DTE, MAX_ENTRY_DTE] inclusive.
# Entry-side only; neither bound gates horizon mapping or exits for existing
# positions.
#
# Floor = 1: the h=1 model's purest expression is the shortest-dated option —
# a 1-DTE entry is one overnight of vol exposure, closed on expiry morning.
# The old floor of 7 dated from the 2026-06 straddle-churn bug, where the
# expiration_proximity exit (dte <= 2) force-closed fresh near-expiry entries
# within a session. The exit manager now scopes that exit (and the
# assignment-risk near-money close) by ENTRY dte, so a deliberately
# short-dated entry rides to its expiry-morning close instead of
# round-tripping into bid/ask + theta. Floor of 0 stays forbidden: the
# expiry-day assignment backstop (dte <= 0) would close a same-day entry on
# the spot.
#
# Cap = 14: the 1-day forecast decays toward the stock's 63-day mean vol at
# the GARCH persistence rate, so by ~1 month the projection is mostly "IV vs
# this stock's normal level" — the VRP-level trade that 7 years of research
# refuted and Sam retired. Trade only where the timing signal survives.
MIN_ENTRY_DTE = 1
MAX_ENTRY_DTE = 14

# Fallback GARCH persistence when the feature row has none (fresh symbol, GARCH
# warm-up) and no per-symbol value came from the retrain JSON.
DEFAULT_PHI = 0.94

# Tenor floor (trading days) for the realized-vol window that the VRP gap is
# measured against — below ~5 days the estimate is too noisy to log.
VRP_MIN_REALIZED_WINDOW = 5


@dataclass(frozen=True)
class TradeLeg:
    strike: float
    option_type: str   # "call" | "put"
    side: str          # "buy" | "sell"
    quantity: int
    contract_symbol: str


@dataclass(frozen=True)
class TradeSignal:
    symbol: str
    expiration: date
    dte: int
    horizon_lower: int
    horizon_upper: int
    weight_lower: float
    direction: str             # "BUY" | "SELL"
    underlying_price: float
    atm_iv: float              # annualized
    predicted_iv_equivalent: float  # annualized model forecast (h=1: term-projected to this DTE)
    divergence: float          # predicted_iv_equivalent − atm_iv
    cross_sectional_z: float
    time_series_z: float | None
    liquidity_score: float
    legs: list[TradeLeg]
    is_actionable: bool
    diagnostic_notes: str = ""
    # h=1 gate fields (None on legacy-pipeline signals)
    vrp_z: float | None = None
    expected_edge_usd: float | None = None
    total_cost_usd: float | None = None
    blocked_by: str | None = None   # gate name; None/"" = cleared all gates

    @property
    def is_interpolated(self) -> bool:
        return self.horizon_lower != self.horizon_upper


def interpolate_horizon(dte: int) -> tuple[int, int, float] | None:
    """LEGACY PIPELINE ONLY: map DTE to (horizon_lower, horizon_upper,
    weight_lower) over the (5, 10, 21) model grid. The h=1 path projects a
    single next-day forecast along the GARCH persistence term structure
    instead. Returns None if DTE is outside [MIN_DTE, MAX_DTE]."""
    if dte < MIN_DTE or dte > MAX_DTE:
        return None
    if dte <= 5:
        return (5, 5, 1.0)
    if dte == 10:
        return (10, 10, 1.0)
    if dte >= 21:
        return (21, 21, 1.0)
    if dte < 10:  # 6-9: between 5 and 10
        weight_lower = (10 - dte) / 5.0
        return (5, 10, weight_lower)
    # 11-20: between 10 and 21
    weight_lower = (21 - dte) / 11.0
    return (10, 21, weight_lower)


def blend_prediction(
    preds_by_horizon: dict[int, float],
    horizon_lower: int,
    horizon_upper: int,
    weight_lower: float,
) -> float:
    if horizon_lower == horizon_upper:
        return preds_by_horizon[horizon_lower]
    return weight_lower * preds_by_horizon[horizon_lower] + (1.0 - weight_lower) * preds_by_horizon[horizon_upper]


def find_atm_iv(
    contracts: list[OptionContract],
    underlying_price: float,
) -> tuple[OptionContract, OptionContract] | None:
    """Return (atm_call, atm_put) — both nearest to the underlying price.
    Returns None if either side is missing."""
    calls = [c for c in contracts if c.option_type == "call"]
    puts = [c for c in contracts if c.option_type == "put"]
    if not calls or not puts:
        return None
    atm_call = min(calls, key=lambda c: abs(c.strike - underlying_price))
    atm_put = min(puts, key=lambda p: abs(p.strike - underlying_price))
    return (atm_call, atm_put)


def composite_liquidity(call: OptionContract, put: OptionContract) -> float:
    vol = min(call.volume, put.volume)
    oi = min(call.open_interest, put.open_interest)

    def _spread(c: OptionContract) -> float:
        mid = (c.bid + c.ask) / 2.0
        if mid <= 0:
            return 1.0
        return (c.ask - c.bid) / mid
    spread = max(_spread(call), _spread(put))
    return vol * oi / (1.0 + spread)


def _passes_liquidity_filters(
    contract: OptionContract,
    min_volume: int,
    min_open_interest: int,
    max_relative_spread: float,
) -> bool:
    if contract.volume < min_volume:
        return False
    if contract.open_interest < min_open_interest:
        return False
    mid = (contract.bid + contract.ask) / 2.0
    if mid <= 0:
        return False
    rel_spread = (contract.ask - contract.bid) / mid
    return rel_spread <= max_relative_spread


def _pick_iron_condor_wings(
    contracts: list[OptionContract],
    atm_strike: float,
    predicted_iv_eq: float,
    dte: int,
    underlying_price: float,
) -> tuple[OptionContract, OptionContract] | None:
    """Pick long call (above ATM) and long put (below ATM) at ~1σ OTM,
    using PREDICTED iv (our view) for the σ estimate. Calendar-day dte over a
    calendar-day year — see CALENDAR_DAYS_PER_YEAR."""
    one_sigma_move = underlying_price * predicted_iv_eq * math.sqrt(dte / CALENDAR_DAYS_PER_YEAR)
    target_call_strike = atm_strike + one_sigma_move
    target_put_strike = atm_strike - one_sigma_move

    upper_calls = [c for c in contracts if c.option_type == "call" and c.strike > atm_strike]
    lower_puts = [p for p in contracts if p.option_type == "put" and p.strike < atm_strike]

    if not upper_calls or not lower_puts:
        return None
    long_call = min(upper_calls, key=lambda c: abs(c.strike - target_call_strike))
    long_put = min(lower_puts, key=lambda p: abs(p.strike - target_put_strike))
    return (long_call, long_put)


def trailing_gk_realized_vol(
    daily_gk_vol: pd.Series, window: int
) -> float | None:
    """Annualized realized vol over the trailing `window` trading days from
    the single-day GK vol series: sqrt(252 * mean(gk²)). None if not enough
    history."""
    clean = daily_gk_vol.dropna()
    if len(clean) < window:
        return None
    tail = clean.iloc[-window:].to_numpy(dtype=float)
    return float(math.sqrt(TRADING_DAYS_PER_YEAR * float(np.mean(tail ** 2))))


@dataclass
class _Candidate:
    """Internal: a (symbol, expiration) divergence before gating."""
    symbol: str
    expiration: date
    dte: int
    horizon_lower: int
    horizon_upper: int
    weight_lower: float
    underlying_price: float
    atm_call: OptionContract
    atm_put: OptionContract
    chain: list[OptionContract]
    predicted_iv_equivalent: float
    atm_iv: float
    divergence: float
    # h=1 only
    g_now: float | None = None
    realized_vol: float | None = None
    vrp_z: float | None = None


class SignalGenerator:
    def __init__(
        self,
        h1_predictor: H1DeviationPredictor | None = None,
        predictors_by_horizon: dict[int, BestPredictor] | None = None,
        history_store: DivergenceHistory | None = None,
        min_volume: int = 10,
        min_open_interest: int = 50,
        max_relative_spread: float = 0.10,
        cross_sectional_z_threshold: float = 1.5,
        max_divergence: float = 0.25,
        earnings_calendar: EarningsCalendar | None = None,
        earnings_filter_enabled: bool = True,
        earnings_buffer_days: int = 7,
        long_straddle_excluded_symbols: frozenset[str] | set[str] | None = None,
        min_credit: float = 0.0,
        min_credit_to_width: float = 0.0,
        macro_calendar: MacroCalendar | None = None,
        macro_sensitive_symbols: frozenset[str] | set[str] | None = None,
        vix_backwardation_threshold: float = 1.0,
        vrp_z_sell: float = 1.5,
        vrp_z_buy: float = 1.25,
        vrp_min_obs: int = 120,
        cost_multiple: float = 2.0,
        max_leg_spread_pct: float = 0.05,
        per_contract_fee: float = 0.0,
        phi_by_symbol: dict[str, float] | None = None,
        min_entry_dte: int = MIN_ENTRY_DTE,
        max_entry_dte: int = MAX_ENTRY_DTE,
    ):
        if not 1 <= min_entry_dte <= max_entry_dte:
            raise ValueError(
                f"need 1 <= min_entry_dte <= max_entry_dte, got "
                f"{min_entry_dte}/{max_entry_dte}"
            )
        self._min_entry_dte = min_entry_dte
        self._max_entry_dte = max_entry_dte
        # Exactly one prediction source: the h=1 deviation predictor (active
        # pipeline) or the legacy multi-horizon BestPredictors.
        if (h1_predictor is None) == (predictors_by_horizon is None):
            raise ValueError(
                "provide exactly one of h1_predictor / predictors_by_horizon"
            )
        if predictors_by_horizon is not None:
            for h in TRAINED_HORIZONS:
                if h not in predictors_by_horizon:
                    raise ValueError(f"missing predictor for horizon {h}")
        self._h1 = h1_predictor
        self._predictors = predictors_by_horizon
        self._history = history_store
        self._min_volume = min_volume
        self._min_oi = min_open_interest
        self._max_rel_spread = max_relative_spread
        self._z_threshold = cross_sectional_z_threshold
        # Divergences above this absolute magnitude are almost certainly event-driven
        # (earnings, FDA, FOMC, product launch) — the model can't distinguish event
        # premium from vol mispricing. Hard block on SELL (unbounded loss if the
        # event is real); demote on BUY.
        self._max_divergence = max_divergence
        self._earnings = earnings_calendar
        self._earnings_filter_enabled = earnings_filter_enabled
        self._earnings_buffer_days = earnings_buffer_days
        # Symbols barred from the long-straddle (BUY) side. Index ETFs (SPY/QQQ)
        # carry a structural variance-risk premium — realized vol sits below
        # implied, so buying their premium bleeds. They stay eligible for the
        # SELL (iron condor) side.
        self._no_long_straddle = frozenset(long_straddle_excluded_symbols or ())
        # SELL floor on the condor's mid-price net credit — sub-$0.25 credits
        # are fee-and-slippage food on a small account; 0.0 disables.
        if min_credit < 0:
            raise ValueError(f"min_credit must be >= 0, got {min_credit}")
        self._min_credit = min_credit
        # Relative credit floor: reject condors whose mid credit is below this
        # fraction of the wing width — a per-trade expected-value test that
        # scales across cheap and expensive names. 0.0 disables.
        if min_credit_to_width < 0:
            raise ValueError(
                f"min_credit_to_width must be >= 0, got {min_credit_to_width}"
            )
        self._min_credit_to_width = min_credit_to_width
        # Macro-event filter: for rate/index-linked symbols (TLT, SLV, index
        # ETFs) an FOMC decision or CPI print IS their earnings — demote when
        # one falls inside the entry window.
        self._macro = macro_calendar
        self._macro_sensitive = frozenset(macro_sensitive_symbols or ())
        # Mode-agnostic SELL guard: VIX at or above VIX3M (term-structure
        # backwardation) is the regime where every short-vol book bleeds at
        # once. In early stress IV rises above trailing RV, so the VRP z-gate
        # passes SELL at exactly the wrong time and the 0.25 cap only blocks
        # large spikes — this vetoes the moderate-stress gap between them.
        # Ratio comes from the caller (cached daily closes); None fails open.
        self._vix_backwardation = vix_backwardation_threshold
        # VRP calibration gate thresholds (see divergence_history.vrp_z_score).
        self._vrp_z_sell = vrp_z_sell
        self._vrp_z_buy = vrp_z_buy
        self._vrp_min_obs = vrp_min_obs
        # Cost gate (signals.cost_gate).
        self._cost_multiple = cost_multiple
        self._max_leg_spread_pct = max_leg_spread_pct
        self._fee = per_contract_fee
        # Per-symbol GARCH persistence fallback (from the retrain JSON) for
        # rows where the garch_persistence feature is NaN (warm-up).
        self._phi_by_symbol = dict(phi_by_symbol or {})

    # ------------------------------------------------------------------ API

    def generate(
        self,
        scan: ScanResult,
        feature_rows: dict[str, pd.DataFrame],
        returns_by_symbol: dict[str, pd.Series],
        top_n: int = 10,
        vix_term_ratio: float | None = None,
        daily_gk_vol_by_symbol: dict[str, pd.Series] | None = None,
    ) -> tuple[list[TradeSignal], list[TradeSignal]]:
        """Returns (actionable_top_n, all_signals). vix_term_ratio is the
        latest VIX/VIX3M close ratio (None = unavailable, fails open); at or
        above vix_backwardation_threshold every SELL is demoted.
        daily_gk_vol_by_symbol (h=1 pipeline) carries each symbol's
        single-day GK vol series for the tenor-matched VRP gap."""
        if self._h1 is not None:
            return self._generate_h1(
                scan, feature_rows, top_n, vix_term_ratio,
                daily_gk_vol_by_symbol or {},
            )
        return self._generate_legacy(
            scan, feature_rows, returns_by_symbol, top_n, vix_term_ratio,
        )

    # ------------------------------------------------------------- h=1 path

    def _generate_h1(
        self,
        scan: ScanResult,
        feature_rows: dict[str, pd.DataFrame],
        top_n: int,
        vix_term_ratio: float | None,
        daily_gk_vol_by_symbol: dict[str, pd.Series],
    ) -> tuple[list[TradeSignal], list[TradeSignal]]:
        today = scan.fetched_at.date()

        # 1. Per ticker: baseline b_t, deviation forecast, GARCH persistence
        h1_by_symbol: dict[str, tuple[float, float, float]] = {}
        for symbol in scan.snapshots:
            row = feature_rows.get(symbol)
            if row is None or row.empty:
                continue
            if "log_gk_baseline_63" not in row.columns:
                logger.warning("no log_gk_baseline_63 column for %s — skipping", symbol)
                continue
            b_t = float(row["log_gk_baseline_63"].iloc[-1])
            if not math.isfinite(b_t):
                # <40 obs of GK history (or a frozen series) — no baseline,
                # no signal. This is what keeps stale names like IWM out.
                logger.info("no h=1 baseline for %s (insufficient history)", symbol)
                continue
            phi = float("nan")
            if "garch_persistence" in row.columns:
                phi = float(row["garch_persistence"].iloc[-1])
            if not math.isfinite(phi):
                phi = self._phi_by_symbol.get(symbol, DEFAULT_PHI)
            try:
                dev_hat = self._h1.predict_deviation(row)
            except Exception as e:
                logger.warning("h=1 prediction failed for %s: %s", symbol, e)
                continue
            if not math.isfinite(dev_hat):
                continue
            h1_by_symbol[symbol] = (b_t, dev_hat, phi)

        # 2. Per (ticker, expiration): term-projected forecast vs that
        # option's ATM IV — never the raw 1-day forecast vs long-dated IV.
        candidates: list[_Candidate] = []
        for symbol, snap in scan.snapshots.items():
            if symbol not in h1_by_symbol:
                continue
            b_t, dev_hat, phi = h1_by_symbol[symbol]
            try:
                underlying_price = float(snap.underlying.get("last") or 0.0)
            except (TypeError, ValueError):
                underlying_price = 0.0
            if underlying_price <= 0:
                continue

            chains_by_exp: dict[date, list[OptionContract]] = defaultdict(list)
            for c in snap.contracts:
                chains_by_exp[c.expiration].append(c)

            gk_series = daily_gk_vol_by_symbol.get(symbol)

            for expiration, chain in chains_by_exp.items():
                dte = (expiration - today).days
                # Entry DTE window (see MIN_ENTRY_DTE/MAX_ENTRY_DTE above).
                # MAX_DTE stays as the horizon-mapping ceiling. Skip silently.
                if not self._min_entry_dte <= dte <= min(self._max_entry_dte, MAX_DTE):
                    continue

                forecast_d = project_term_vol(b_t, dev_hat, phi, dte)

                atm_pair = find_atm_iv(chain, underlying_price)
                if atm_pair is None:
                    continue
                atm_call, atm_put = atm_pair
                ivs = [iv for iv in (atm_call.iv, atm_put.iv) if iv > 0]
                if not ivs:
                    continue
                atm_iv = sum(ivs) / len(ivs)

                # Tenor-matched VRP gap: g = log(IV) − log(realized vol over
                # ~the option's life).
                g_now: float | None = None
                realized: float | None = None
                if gk_series is not None:
                    window = max(
                        VRP_MIN_REALIZED_WINDOW,
                        round(dte * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR),
                    )
                    realized = trailing_gk_realized_vol(gk_series, window)
                    if realized is not None and atm_iv > 0:
                        g_now = math.log(atm_iv) - math.log(realized + GK_EPS)

                candidates.append(_Candidate(
                    symbol=symbol,
                    expiration=expiration,
                    dte=dte,
                    horizon_lower=1,
                    horizon_upper=1,
                    weight_lower=1.0,
                    underlying_price=underlying_price,
                    atm_call=atm_call,
                    atm_put=atm_put,
                    chain=chain,
                    predicted_iv_equivalent=forecast_d,
                    atm_iv=atm_iv,
                    divergence=forecast_d - atm_iv,
                    g_now=g_now,
                    realized_vol=realized,
                ))

        # 3. Log the VRP gap for EVERY evaluated candidate (history accrues
        # even while the gate stays silent), then compute each candidate's z.
        if self._history is not None:
            vrp_rows = [
                (today, c.symbol, c.dte, c.atm_iv, c.realized_vol, c.g_now)
                for c in candidates
                if c.g_now is not None and c.realized_vol is not None
            ]
            self._history.log_vrp(vrp_rows)
            for c in candidates:
                if c.g_now is not None:
                    c.vrp_z = self._history.vrp_z_score(
                        c.symbol, c.g_now, dte=c.dte, min_obs=self._vrp_min_obs,
                    )

        # 4. Cross-sectional z on divergence — diagnostic only in the h=1
        # path (ranking uses |vrp_z|), still logged for continuity.
        divs = np.array([c.divergence for c in candidates])
        cs_mean = float(divs.mean()) if len(divs) else 0.0
        cs_std = float(divs.std(ddof=0)) if len(divs) else 0.0

        all_signals: list[TradeSignal] = []
        for c in candidates:
            cs_z = (c.divergence - cs_mean) / cs_std if cs_std > 0 else 0.0
            ts_z: float | None = None
            if self._history is not None:
                ts_z = self._history.time_series_z_score(
                    c.symbol, c.horizon_lower, c.horizon_upper, c.divergence,
                )

            def _signal(
                direction: str,
                *,
                legs: list[TradeLeg] | None = None,
                actionable: bool = False,
                notes: str = "",
                blocked_by: str | None = None,
                liquidity: float = 0.0,
                edge: float | None = None,
                cost: float | None = None,
            ) -> TradeSignal:
                return TradeSignal(
                    symbol=c.symbol, expiration=c.expiration, dte=c.dte,
                    horizon_lower=c.horizon_lower, horizon_upper=c.horizon_upper,
                    weight_lower=c.weight_lower, direction=direction,
                    underlying_price=c.underlying_price, atm_iv=c.atm_iv,
                    predicted_iv_equivalent=c.predicted_iv_equivalent,
                    divergence=c.divergence, cross_sectional_z=cs_z,
                    time_series_z=ts_z, liquidity_score=liquidity,
                    legs=legs or [], is_actionable=actionable,
                    diagnostic_notes=notes, vrp_z=c.vrp_z,
                    expected_edge_usd=edge, total_cost_usd=cost,
                    blocked_by=blocked_by,
                )

            # Gate 1 — VRP history: no calibrated gap distribution, no signal.
            if c.vrp_z is None:
                fallback_direction = "BUY" if c.divergence > 0 else "SELL"
                all_signals.append(_signal(
                    fallback_direction, blocked_by="vrp_history",
                    notes=(
                        f"vrp gate: < {self._vrp_min_obs} gap observations "
                        f"for {c.symbol} (or missing realized vol) — no signal"
                    ),
                ))
                continue

            # Gate 2 — VRP z thresholds set the direction. SELL needs the IV
            # premium unusually rich vs this ticker's own history; BUY needs
            # it unusually cheap (lower bar: long-vol loss is bounded).
            if c.vrp_z >= self._vrp_z_sell:
                direction = "SELL"
            elif c.vrp_z <= -self._vrp_z_buy:
                direction = "BUY"
            else:
                fallback_direction = "BUY" if c.divergence > 0 else "SELL"
                all_signals.append(_signal(
                    fallback_direction, blocked_by="vrp_z",
                    notes=(
                        f"vrp gate: z={c.vrp_z:+.2f} inside "
                        f"[-{self._vrp_z_buy}, +{self._vrp_z_sell}] — no edge vs "
                        f"this ticker's own premium history"
                    ),
                ))
                continue

            # Gate 3 — VIX backwardation veto on every SELL: book-level crash
            # regime guard (see constructor comment).
            if (direction == "SELL" and vix_term_ratio is not None
                    and vix_term_ratio >= self._vix_backwardation):
                all_signals.append(_signal(
                    direction, blocked_by="vix_backwardation",
                    notes=(
                        f"vix-term-structure veto: VIX/VIX3M={vix_term_ratio:.3f} "
                        f">= {self._vix_backwardation} (backwardation — crash regime)"
                    ),
                ))
                continue

            # Gate 4 — model agreement. The z-gate sets the direction from the
            # ticker's own premium history, but the trade must also agree with
            # the model's term-projected view: a SELL needs the forecast BELOW
            # the IV being sold, a BUY needs it above. Without this, a
            # decayed-baseline forecast far above IV would read as the LARGEST
            # "edge" in the cost gate precisely when the model says don't sell.
            if (direction == "SELL" and c.divergence >= 0) or (
                    direction == "BUY" and c.divergence <= 0):
                all_signals.append(_signal(
                    direction, blocked_by="model_disagrees",
                    notes=(
                        f"model disagrees with {direction}: forecast "
                        f"{c.predicted_iv_equivalent:.3f} vs IV {c.atm_iv:.3f} "
                        f"(divergence {c.divergence:+.3f})"
                    ),
                ))
                continue

            # Gate 5 — absolute divergence cap. Hard block on SELL (an event
            # the market is pricing and the model can't see = unbounded short
            # risk); demote on BUY (event-suspect, loss bounded).
            if abs(c.divergence) > self._max_divergence:
                all_signals.append(_signal(
                    direction, blocked_by="divergence_cap",
                    notes=(
                        f"event-suspect: |divergence|={abs(c.divergence):.3f} "
                        f"> cap {self._max_divergence}"
                        + (" (hard SELL block)" if direction == "SELL" else "")
                    ),
                ))
                continue

            # Gate 6 — earnings entry block (fixed 7-day buffer; the
            # earnings_risk EXIT closes short vol before any report that
            # lands later in the position's life).
            earnings_demote = self._check_earnings(c.symbol, today)
            if earnings_demote is not None:
                _earnings_date, note = earnings_demote
                all_signals.append(_signal(
                    direction, blocked_by="earnings", notes=note,
                ))
                continue

            # Gate 7 — macro events for rate/index-linked names.
            macro_note = self._check_macro(c.symbol, today)
            if macro_note is not None:
                all_signals.append(_signal(
                    direction, blocked_by="macro", notes=macro_note,
                ))
                continue

            # Gate 8 — long-straddle exclusion (index ETFs).
            if direction == "BUY" and c.symbol in self._no_long_straddle:
                all_signals.append(_signal(
                    direction, blocked_by="long_straddle_excluded",
                    notes=(
                        f"long-straddle excluded for {c.symbol} "
                        f"(index ETF / variance-risk premium)"
                    ),
                ))
                continue

            # Gate 9 — structure legs (liquidity + credit floors).
            legs, notes = self._build_legs(c, direction)
            if not legs or notes:
                all_signals.append(_signal(
                    direction, blocked_by="legs", notes=notes,
                ))
                continue

            # Gate 10 — cost gate: the priced edge must clear the friction.
            check = evaluate_cost_gate(
                legs, c.chain,
                forecast_vol=c.predicted_iv_equivalent, atm_iv=c.atm_iv,
                per_contract_fee=self._fee,
                cost_multiple=self._cost_multiple,
                max_leg_spread_pct=self._max_leg_spread_pct,
            )
            if not check.passed:
                all_signals.append(_signal(
                    direction, blocked_by=check.blocked_label or "cost_gate",
                    notes=check.reason,
                    edge=check.expected_edge_usd, cost=check.total_cost_usd,
                ))
                continue

            liquidity = composite_liquidity(c.atm_call, c.atm_put)
            all_signals.append(_signal(
                direction, legs=legs, actionable=True, liquidity=liquidity,
                edge=check.expected_edge_usd, cost=check.total_cost_usd,
            ))

        # 5. Persist all signals (including demoted, with their gate labels).
        if self._history is not None and all_signals:
            self._history.log_signals(all_signals, today)

        # 6. Rank actionable by how far each signal cleared ITS OWN direction's
        # z bar (SELL: z − z_sell; BUY: −z − z_buy) — raw |z| would let a BUY
        # barely past its lower 1.25 bar outrank a SELL comfortably past 1.5.
        # Liquidity tiebreak; one signal per symbol per cycle so a single name
        # can't stack correlated expirations in one pass.
        def _z_excess(s: TradeSignal) -> float:
            z = s.vrp_z or 0.0
            return (z - self._vrp_z_sell) if s.direction == "SELL" else (-z - self._vrp_z_buy)

        actionable = [s for s in all_signals if s.is_actionable]
        actionable.sort(key=lambda s: (-_z_excess(s), -s.liquidity_score))
        seen: set[str] = set()
        deduped: list[TradeSignal] = []
        for s in actionable:
            if s.symbol not in seen:
                seen.add(s.symbol)
                deduped.append(s)
        return deduped[:top_n], all_signals

    # ---------------------------------------------------------- legacy path

    def _generate_legacy(
        self,
        scan: ScanResult,
        feature_rows: dict[str, pd.DataFrame],
        returns_by_symbol: dict[str, pd.Series],
        top_n: int,
        vix_term_ratio: float | None,
    ) -> tuple[list[TradeSignal], list[TradeSignal]]:
        """MODEL_PIPELINE=legacy: the multi-horizon divergence flow (z-gated,
        both directions), kept runnable for comparison. Harvest mode is gone;
        the VIX backwardation veto now guards SELLs here too."""
        today = scan.fetched_at.date()

        preds: dict[str, dict[int, float]] = {}
        for symbol in scan.snapshots:
            if symbol not in feature_rows or symbol not in returns_by_symbol:
                continue
            preds[symbol] = {}
            for h, predictor in self._predictors.items():
                try:
                    p = predictor.predict_forward_rv(
                        returns_history=returns_by_symbol[symbol],
                        X_row=feature_rows[symbol],
                    )
                    preds[symbol][h] = float(p)
                except Exception as e:
                    logger.warning("prediction failed for %s @ h=%d: %s", symbol, h, e)
                    preds[symbol][h] = float("nan")

        candidates: list[_Candidate] = []
        for symbol, snap in scan.snapshots.items():
            if symbol not in preds:
                continue
            try:
                underlying_price = float(snap.underlying.get("last") or 0.0)
            except (TypeError, ValueError):
                underlying_price = 0.0
            if underlying_price <= 0:
                continue

            chains_by_exp: dict[date, list[OptionContract]] = defaultdict(list)
            for c in snap.contracts:
                chains_by_exp[c.expiration].append(c)

            for expiration, chain in chains_by_exp.items():
                dte = (expiration - today).days
                # Same entry window as the h1 path; DTEs below MIN_DTE are
                # additionally dropped by interpolate_horizon (the legacy
                # models have no horizon short enough to price them).
                if not self._min_entry_dte <= dte <= min(self._max_entry_dte, MAX_DTE):
                    continue
                horizon_pair = interpolate_horizon(dte)
                if horizon_pair is None:
                    continue
                h_lo, h_up, w_lo = horizon_pair
                if any(math.isnan(preds[symbol].get(h, float("nan"))) for h in (h_lo, h_up)):
                    continue

                pred_rv_daily = blend_prediction(preds[symbol], h_lo, h_up, w_lo)
                predicted_iv_eq = pred_rv_daily * math.sqrt(TRADING_DAYS_PER_YEAR)

                atm_pair = find_atm_iv(chain, underlying_price)
                if atm_pair is None:
                    continue
                atm_call, atm_put = atm_pair
                ivs = [iv for iv in (atm_call.iv, atm_put.iv) if iv > 0]
                if not ivs:
                    continue
                atm_iv = sum(ivs) / len(ivs)

                candidates.append(_Candidate(
                    symbol=symbol,
                    expiration=expiration,
                    dte=dte,
                    horizon_lower=h_lo,
                    horizon_upper=h_up,
                    weight_lower=w_lo,
                    underlying_price=underlying_price,
                    atm_call=atm_call,
                    atm_put=atm_put,
                    chain=chain,
                    predicted_iv_equivalent=predicted_iv_eq,
                    atm_iv=atm_iv,
                    divergence=predicted_iv_eq - atm_iv,
                ))

        buckets: dict[tuple[int, int], list[_Candidate]] = defaultdict(list)
        for c in candidates:
            buckets[(c.horizon_lower, c.horizon_upper)].append(c)
        z_by_id: dict[int, float] = {}
        for _bucket, members in buckets.items():
            divs = np.array([m.divergence for m in members])
            mean = divs.mean()
            std = divs.std(ddof=0)
            for m in members:
                z = (m.divergence - mean) / std if std > 0 else 0.0
                z_by_id[id(m)] = float(z)

        all_signals: list[TradeSignal] = []
        for c in candidates:
            cs_z = z_by_id[id(c)]
            ts_z: float | None = None
            if self._history is not None:
                ts_z = self._history.time_series_z_score(
                    c.symbol, c.horizon_lower, c.horizon_upper, c.divergence,
                )

            direction = "BUY" if c.divergence > 0 else "SELL"

            def _demoted(note: str, blocked_by: str) -> TradeSignal:
                return TradeSignal(
                    symbol=c.symbol, expiration=c.expiration, dte=c.dte,
                    horizon_lower=c.horizon_lower, horizon_upper=c.horizon_upper,
                    weight_lower=c.weight_lower, direction=direction,
                    underlying_price=c.underlying_price, atm_iv=c.atm_iv,
                    predicted_iv_equivalent=c.predicted_iv_equivalent,
                    divergence=c.divergence, cross_sectional_z=cs_z,
                    time_series_z=ts_z, liquidity_score=0.0, legs=[],
                    is_actionable=False, diagnostic_notes=note,
                    blocked_by=blocked_by,
                )

            if (direction == "SELL" and vix_term_ratio is not None
                    and vix_term_ratio >= self._vix_backwardation):
                all_signals.append(_demoted(
                    f"vix-term-structure veto: VIX/VIX3M={vix_term_ratio:.3f} "
                    f">= {self._vix_backwardation} (backwardation — crash regime)",
                    "vix_backwardation",
                ))
                continue

            if direction == "BUY" and c.symbol in self._no_long_straddle:
                all_signals.append(_demoted(
                    f"long-straddle excluded for {c.symbol} "
                    f"(index ETF / variance-risk premium)",
                    "long_straddle_excluded",
                ))
                continue

            if abs(c.divergence) > self._max_divergence:
                all_signals.append(_demoted(
                    f"event-suspect: |divergence|={abs(c.divergence):.3f} "
                    f"> cap {self._max_divergence}",
                    "divergence_cap",
                ))
                continue

            earnings_demote = self._check_earnings(c.symbol, today)
            if earnings_demote is not None:
                _earnings_date, note = earnings_demote
                all_signals.append(_demoted(note, "earnings"))
                continue

            macro_note = self._check_macro(c.symbol, today)
            if macro_note is not None:
                all_signals.append(_demoted(macro_note, "macro"))
                continue

            if abs(cs_z) < self._z_threshold:
                all_signals.append(_demoted(
                    f"|z|={abs(cs_z):.2f} below threshold {self._z_threshold}",
                    "cross_sectional_z",
                ))
                continue

            legs, notes = self._build_legs(c, direction)
            actionable = bool(legs) and not notes
            liquidity = composite_liquidity(c.atm_call, c.atm_put) if c.atm_call and c.atm_put else 0.0

            all_signals.append(TradeSignal(
                symbol=c.symbol, expiration=c.expiration, dte=c.dte,
                horizon_lower=c.horizon_lower, horizon_upper=c.horizon_upper,
                weight_lower=c.weight_lower, direction=direction,
                underlying_price=c.underlying_price, atm_iv=c.atm_iv,
                predicted_iv_equivalent=c.predicted_iv_equivalent,
                divergence=c.divergence, cross_sectional_z=cs_z,
                time_series_z=ts_z, liquidity_score=liquidity, legs=legs,
                is_actionable=actionable, diagnostic_notes=notes,
                blocked_by=None if actionable else "legs",
            ))

        if self._history is not None and all_signals:
            self._history.log_signals(all_signals, today)

        actionable = [s for s in all_signals if s.is_actionable]
        actionable.sort(key=lambda s: (-abs(s.cross_sectional_z), -s.liquidity_score))
        return actionable[:top_n], all_signals

    # -------------------------------------------------------------- filters

    def _check_earnings(
        self, symbol: str, today: date, window_end: date | None = None
    ) -> tuple[date, str] | None:
        """Return (earnings_date, diagnostic_note) if the signal should be demoted
        for earnings risk, None if it should pass. The window runs from today to
        the fixed buffer (or `window_end` if later). Fails open: missing
        calendar, no API key, or no data for the symbol all return None — the
        earnings_risk EXIT (which fails closed via the stored date) is the
        backstop for positions."""
        if not self._earnings_filter_enabled or self._earnings is None:
            return None
        end = today + timedelta(days=self._earnings_buffer_days)
        if window_end is not None and window_end > end:
            end = window_end
        result = self._earnings.has_earnings_in_window(symbol, today, end)
        if result is None:
            # No information — fail open. The calendar logs a single WARNING on
            # refresh failure; no need to spam per signal.
            return None
        if not result:
            return None
        earnings_date = self._earnings.next_earnings_on_or_after(symbol, today)
        if earnings_date is None:
            return None
        return (
            earnings_date,
            f"earnings_within_window: {symbol} reports {earnings_date.isoformat()} "
            f"on/before window end {end.isoformat()}",
        )

    def _check_macro(
        self, symbol: str, today: date, window_end: date | None = None,
    ) -> str | None:
        """Diagnostic note if a scheduled macro event (FOMC/CPI) falls inside
        the window and `symbol` is macro-sensitive, None otherwise. Window
        matches _check_earnings. Fails open when no calendar is wired or the
        hand-maintained table has aged out."""
        if self._macro is None or symbol not in self._macro_sensitive:
            return None
        end = today + timedelta(days=self._earnings_buffer_days)
        if window_end is not None and window_end > end:
            end = window_end
        event = self._macro.next_event_in_window(today, end)
        if event is None:
            return None
        event_date, label = event
        return (
            f"macro_event_within_window: {label} on {event_date.isoformat()} "
            f"inside window ending {end.isoformat()} ({symbol} is rate/index-linked)"
        )

    def _build_legs(self, c: _Candidate, direction: str) -> tuple[list[TradeLeg], str]:
        """Construct the trade legs for the chosen structure. Returns (legs, notes).
        notes is empty on success; populated with reason on failure (signal demoted)."""
        atm_call, atm_put = c.atm_call, c.atm_put

        if direction == "BUY":
            # Long ATM straddle: 2 legs
            for leg in (atm_call, atm_put):
                if not _passes_liquidity_filters(leg, self._min_volume, self._min_oi, self._max_rel_spread):
                    return [], f"BUY straddle: {leg.option_type} k={leg.strike} fails liquidity"
            return [
                TradeLeg(atm_call.strike, "call", "buy", 1, atm_call.symbol),
                TradeLeg(atm_put.strike, "put", "buy", 1, atm_put.symbol),
            ], ""

        # SELL → short iron condor: short ATM call + short ATM put + long OTM
        # wings, placed at ~1σ using the MODEL's forecast (h=1: the
        # term-projected vol for this DTE).
        wings = _pick_iron_condor_wings(
            c.chain, atm_call.strike, c.predicted_iv_equivalent, c.dte,
            c.underlying_price,
        )
        if wings is None:
            return [], "SELL: insufficient OTM strikes for iron condor wings"
        long_call, long_put = wings
        legs_to_check = [atm_call, atm_put, long_call, long_put]
        for leg in legs_to_check:
            if not _passes_liquidity_filters(leg, self._min_volume, self._min_oi, self._max_rel_spread):
                return [], f"SELL iron condor: {leg.option_type} k={leg.strike} fails liquidity"
        if self._min_credit > 0 or self._min_credit_to_width > 0:
            credit_mid = (
                (atm_call.bid + atm_call.ask) / 2.0
                + (atm_put.bid + atm_put.ask) / 2.0
                - (long_call.bid + long_call.ask) / 2.0
                - (long_put.bid + long_put.ask) / 2.0
            )
            if self._min_credit > 0 and credit_mid < self._min_credit:
                return [], (
                    f"SELL iron condor: mid credit ${credit_mid:.2f} below "
                    f"floor ${self._min_credit:.2f}"
                )
            if self._min_credit_to_width > 0:
                wing_width = max(
                    long_call.strike - atm_call.strike,
                    atm_put.strike - long_put.strike,
                )
                if wing_width > 0 and credit_mid < self._min_credit_to_width * wing_width:
                    return [], (
                        f"SELL iron condor: mid credit ${credit_mid:.2f} < "
                        f"{self._min_credit_to_width:.0%} of ${wing_width:.2f} "
                        f"wing width (credit/width "
                        f"{credit_mid / wing_width:.1%})"
                    )
        return [
            TradeLeg(atm_call.strike, "call", "sell", 1, atm_call.symbol),
            TradeLeg(atm_put.strike, "put", "sell", 1, atm_put.symbol),
            TradeLeg(long_call.strike, "call", "buy", 1, long_call.symbol),
            TradeLeg(long_put.strike, "put", "buy", 1, long_put.symbol),
        ], ""
