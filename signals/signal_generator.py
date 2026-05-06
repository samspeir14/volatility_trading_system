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
from data.market_data import ScanResult
from model.best_predictor import BestPredictor
from signals.divergence_history import DivergenceHistory

logger = logging.getLogger(__name__)


TRAINED_HORIZONS: tuple[int, ...] = (5, 10, 21)
TRADING_DAYS_PER_YEAR = 252
MIN_DTE = 4
MAX_DTE = 45


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
    predicted_iv_equivalent: float  # annualized predicted RV × √252
    divergence: float          # predicted_iv_equivalent − atm_iv
    cross_sectional_z: float
    time_series_z: float | None
    liquidity_score: float
    legs: list[TradeLeg]
    is_actionable: bool
    diagnostic_notes: str = ""

    @property
    def is_interpolated(self) -> bool:
        return self.horizon_lower != self.horizon_upper


def interpolate_horizon(dte: int) -> tuple[int, int, float] | None:
    """Map DTE to (horizon_lower, horizon_upper, weight_lower).
    Returns None if DTE is outside [MIN_DTE, MAX_DTE]."""
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
    using PREDICTED iv (our view) for the σ estimate."""
    one_sigma_move = underlying_price * predicted_iv_eq * math.sqrt(dte / TRADING_DAYS_PER_YEAR)
    target_call_strike = atm_strike + one_sigma_move
    target_put_strike = atm_strike - one_sigma_move

    upper_calls = [c for c in contracts if c.option_type == "call" and c.strike > atm_strike]
    lower_puts = [p for p in contracts if p.option_type == "put" and p.strike < atm_strike]

    if not upper_calls or not lower_puts:
        return None
    long_call = min(upper_calls, key=lambda c: abs(c.strike - target_call_strike))
    long_put = min(lower_puts, key=lambda p: abs(p.strike - target_put_strike))
    return (long_call, long_put)


@dataclass
class _Candidate:
    """Internal: a (symbol, expiration) divergence before z-scoring/filtering."""
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


class SignalGenerator:
    def __init__(
        self,
        predictors_by_horizon: dict[int, BestPredictor],
        history_store: DivergenceHistory | None = None,
        min_volume: int = 10,
        min_open_interest: int = 50,
        max_relative_spread: float = 0.10,
        cross_sectional_z_threshold: float = 1.5,
        max_divergence: float = 0.25,
        earnings_calendar: EarningsCalendar | None = None,
        earnings_filter_enabled: bool = True,
        earnings_buffer_days: int = 7,
    ):
        for h in TRAINED_HORIZONS:
            if h not in predictors_by_horizon:
                raise ValueError(f"missing predictor for horizon {h}")
        self._predictors = predictors_by_horizon
        self._history = history_store
        self._min_volume = min_volume
        self._min_oi = min_open_interest
        self._max_rel_spread = max_relative_spread
        self._z_threshold = cross_sectional_z_threshold
        # Divergences above this absolute magnitude are almost certainly event-driven
        # (earnings, FDA, FOMC, product launch) — the model can't distinguish event
        # premium from vol mispricing, so we demote rather than trade them.
        self._max_divergence = max_divergence
        self._earnings = earnings_calendar
        self._earnings_filter_enabled = earnings_filter_enabled
        self._earnings_buffer_days = earnings_buffer_days

    def generate(
        self,
        scan: ScanResult,
        feature_rows: dict[str, pd.DataFrame],
        returns_by_symbol: dict[str, pd.Series],
        top_n: int = 10,
    ) -> tuple[list[TradeSignal], list[TradeSignal]]:
        """Returns (actionable_top_n, all_signals)."""
        today = scan.fetched_at.date()

        # 1. Per ticker: predict at each horizon
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

        # 2. For each (ticker, expiration): build candidate
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

            # Group contracts by expiration
            chains_by_exp: dict[date, list[OptionContract]] = defaultdict(list)
            for c in snap.contracts:
                chains_by_exp[c.expiration].append(c)

            for expiration, chain in chains_by_exp.items():
                dte = (expiration - today).days
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

                # ATM IV: average of call+put if both have non-zero IV, else whichever is non-zero
                ivs = [iv for iv in (atm_call.iv, atm_put.iv) if iv > 0]
                if not ivs:
                    continue
                atm_iv = sum(ivs) / len(ivs)

                divergence = predicted_iv_eq - atm_iv

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
                    divergence=divergence,
                ))

        # 3. Cross-sectional z-score, bucketed by (h_lo, h_up) pair
        buckets: dict[tuple[int, int], list[_Candidate]] = defaultdict(list)
        for c in candidates:
            buckets[(c.horizon_lower, c.horizon_upper)].append(c)
        z_by_id: dict[id, float] = {}
        for bucket, members in buckets.items():
            divs = np.array([m.divergence for m in members])
            mean = divs.mean()
            std = divs.std(ddof=0)
            for m in members:
                z = (m.divergence - mean) / std if std > 0 else 0.0
                z_by_id[id(m)] = float(z)

        # 4. Build TradeSignal per candidate
        all_signals: list[TradeSignal] = []
        for c in candidates:
            cs_z = z_by_id[id(c)]
            ts_z: float | None = None
            if self._history is not None:
                ts_z = self._history.time_series_z_score(
                    c.symbol, c.horizon_lower, c.horizon_upper, c.divergence,
                )

            direction = "BUY" if c.divergence > 0 else "SELL"

            # Event-suspect: divergences above the cap are almost certainly
            # driven by a known event (earnings, FDA, FOMC, product launch)
            # that the market is pricing and the model structurally cannot
            # capture. Demote rather than trade.
            if abs(c.divergence) > self._max_divergence:
                all_signals.append(TradeSignal(
                    symbol=c.symbol, expiration=c.expiration, dte=c.dte,
                    horizon_lower=c.horizon_lower, horizon_upper=c.horizon_upper,
                    weight_lower=c.weight_lower, direction=direction,
                    underlying_price=c.underlying_price, atm_iv=c.atm_iv,
                    predicted_iv_equivalent=c.predicted_iv_equivalent,
                    divergence=c.divergence, cross_sectional_z=cs_z,
                    time_series_z=ts_z, liquidity_score=0.0, legs=[],
                    is_actionable=False,
                    diagnostic_notes=(
                        f"event-suspect: |divergence|={abs(c.divergence):.3f} "
                        f"> cap {self._max_divergence}"
                    ),
                ))
                continue

            # Earnings filter: demote if a known earnings date falls within
            # `earnings_buffer_days` of today. The 0.25 divergence cap catches
            # obvious earnings-day IV spikes; this catches the gradual pre-
            # earnings ramp where divergence sits at 0.18-0.24. Failing open
            # is intentional — a flaky earnings API must not halt trading.
            earnings_demote = self._check_earnings(c.symbol, today)
            if earnings_demote is not None:
                earnings_date, note = earnings_demote
                all_signals.append(TradeSignal(
                    symbol=c.symbol, expiration=c.expiration, dte=c.dte,
                    horizon_lower=c.horizon_lower, horizon_upper=c.horizon_upper,
                    weight_lower=c.weight_lower, direction=direction,
                    underlying_price=c.underlying_price, atm_iv=c.atm_iv,
                    predicted_iv_equivalent=c.predicted_iv_equivalent,
                    divergence=c.divergence, cross_sectional_z=cs_z,
                    time_series_z=ts_z, liquidity_score=0.0, legs=[],
                    is_actionable=False,
                    diagnostic_notes=note,
                ))
                continue

            # Below z threshold: emit non-actionable signal for diagnostics
            if abs(cs_z) < self._z_threshold:
                all_signals.append(TradeSignal(
                    symbol=c.symbol, expiration=c.expiration, dte=c.dte,
                    horizon_lower=c.horizon_lower, horizon_upper=c.horizon_upper,
                    weight_lower=c.weight_lower, direction=direction,
                    underlying_price=c.underlying_price, atm_iv=c.atm_iv,
                    predicted_iv_equivalent=c.predicted_iv_equivalent,
                    divergence=c.divergence, cross_sectional_z=cs_z,
                    time_series_z=ts_z, liquidity_score=0.0, legs=[],
                    is_actionable=False, diagnostic_notes=f"|z|={abs(cs_z):.2f} below threshold {self._z_threshold}",
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
            ))

        # 5. Persist all signals to history
        if self._history is not None and all_signals:
            self._history.log_signals(all_signals, today)

        # 6. Rank actionable
        actionable = [s for s in all_signals if s.is_actionable]
        actionable.sort(key=lambda s: (-abs(s.cross_sectional_z), -s.liquidity_score))
        return actionable[:top_n], all_signals

    def _check_earnings(
        self, symbol: str, today: date
    ) -> tuple[date, str] | None:
        """Return (earnings_date, diagnostic_note) if the signal should be demoted
        for earnings risk, None if it should pass. Fails open: missing calendar,
        no API key, or no data for the symbol all return None."""
        if not self._earnings_filter_enabled or self._earnings is None:
            return None
        end = today + timedelta(days=self._earnings_buffer_days)
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
            f"within {self._earnings_buffer_days}-day buffer",
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

        # SELL → short iron condor: short ATM call + short ATM put + long OTM wings
        wings = _pick_iron_condor_wings(
            c.chain, atm_call.strike, c.predicted_iv_equivalent, c.dte, c.underlying_price,
        )
        if wings is None:
            return [], "SELL: insufficient OTM strikes for iron condor wings"
        long_call, long_put = wings
        legs_to_check = [atm_call, atm_put, long_call, long_put]
        for leg in legs_to_check:
            if not _passes_liquidity_filters(leg, self._min_volume, self._min_oi, self._max_rel_spread):
                return [], f"SELL iron condor: {leg.option_type} k={leg.strike} fails liquidity"
        return [
            TradeLeg(atm_call.strike, "call", "sell", 1, atm_call.symbol),
            TradeLeg(atm_put.strike, "put", "sell", 1, atm_put.symbol),
            TradeLeg(long_call.strike, "call", "buy", 1, long_call.symbol),
            TradeLeg(long_put.strike, "put", "buy", 1, long_put.symbol),
        ], ""
