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

# Lowest DTE at which a NEW position may be opened. This is deliberately higher
# than MIN_DTE (the horizon-mapping floor used by interpolate_horizon, which the
# exit manager also relies on to mark aged positions). A fresh straddle opened
# near MIN_DTE gets force-closed by the expiration_proximity exit (dte <= 2)
# within a session or two — and a weekend collapses the gap entirely (a Friday
# DTE-4 entry is at DTE 1 the next session). Requiring >= 7 calendar days keeps
# at least three trading sessions before the proximity exit can fire, so the
# entry isn't an instant round-trip into bid/ask + theta. Entry-side only; it
# does NOT gate horizon mapping for existing positions.
MIN_ENTRY_DTE = 7


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
        long_straddle_excluded_symbols: frozenset[str] | set[str] | None = None,
        strategy_mode: str = "model",
        extreme_spread_veto: float = 0.12,
        harvest_min_entry_dte: int = 5,
        harvest_max_entry_dte: int = 15,
    ):
        if strategy_mode not in ("model", "harvest"):
            raise ValueError(f"strategy_mode must be 'model' or 'harvest', got {strategy_mode!r}")
        if harvest_min_entry_dte < MIN_DTE:
            raise ValueError(
                f"harvest_min_entry_dte must be >= {MIN_DTE} (horizon-mapping floor), "
                f"got {harvest_min_entry_dte}"
            )
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
        # Symbols barred from the long-straddle (BUY) side. Index ETFs (SPY/QQQ)
        # carry a structural variance-risk premium — realized vol sits below
        # implied, so buying their premium bleeds. They stay eligible for the
        # SELL (iron condor) side, which harvests that premium.
        self._no_long_straddle = frozenset(long_straddle_excluded_symbols or ())
        # "model": trade the model-vs-IV divergence (z-scored, both directions).
        # "harvest": sell iron condors on every eligible name — the 2026-07
        # research verdict: the short-tenor variance risk premium is fat and
        # unconditional, and no model/formula orders it, so there is nothing to
        # gate entries on. Model predictions still run and log to the
        # divergence history for the prospective accuracy tests.
        self._mode = strategy_mode
        # Harvest-mode SELL veto: when ATM IV sits more than this above the
        # trailing 63d realized vol, the gap is usually the market pricing REAL
        # incoming vol (COVID 2020-03-06 shape), not extra premium — the top
        # spread decile was the only reliably losing sell bucket in 7 years.
        # Crash insurance, not alpha: it costs a little in calm years.
        self._spread_veto = extreme_spread_veto
        # Entry window for harvest sells: premium concentrates near expiry
        # (DTE 2-10 in the live log), and the ranking prefers the nearest
        # expiration, so the FLOOR sets where the book actually lives — at 5,
        # entries cluster in DTE 5-8 and ride to zero, covering the fat zone.
        # The model-mode floor (MIN_ENTRY_DTE=7) exists for straddles the
        # proximity exit would force-close; condors are exempt from that exit,
        # so harvest can start closer in. Ceiling: DTE 22+ short vol lost
        # money May-June; just above the fat zone keeps every position in it.
        self._harvest_min_dte = harvest_min_entry_dte
        self._harvest_max_dte = harvest_max_entry_dte

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
                # Entry floor. Model mode: below MIN_ENTRY_DTE a fresh straddle
                # would just be force-closed by the expiration_proximity exit
                # before its thesis can play out. Harvest mode: condors are
                # exempt from that exit, so the floor drops to the fat zone;
                # the ceiling keeps entries inside it. Skip silently — same as
                # an out-of-window DTE — so these don't clutter the signal log.
                if self._mode == "harvest":
                    if dte < self._harvest_min_dte or dte > self._harvest_max_dte:
                        continue
                elif dte < MIN_ENTRY_DTE:
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
        # Harvest mode vetoes on ATM IV minus trailing 63d realized vol; compute
        # it once per symbol. Missing return history fails open (no veto) —
        # consistent with the earnings filter, and post-ingest-fix bars are
        # fresh, so this only fires on genuinely new symbols.
        trail63_by_symbol: dict[str, float] = {}
        if self._mode == "harvest":
            for symbol in {c.symbol for c in candidates}:
                rets = returns_by_symbol.get(symbol)
                if rets is not None and len(rets) >= 63:
                    trail63_by_symbol[symbol] = float(
                        rets.iloc[-63:].std(ddof=0)
                    ) * math.sqrt(TRADING_DAYS_PER_YEAR)
                else:
                    logger.warning(
                        "harvest: no 63d return history for %s — spread veto disabled",
                        symbol,
                    )

        all_signals: list[TradeSignal] = []
        for c in candidates:
            cs_z = z_by_id[id(c)]
            ts_z: float | None = None
            if self._history is not None:
                ts_z = self._history.time_series_z_score(
                    c.symbol, c.horizon_lower, c.horizon_upper, c.divergence,
                )

            # Harvest mode sells premium unconditionally — direction never
            # comes from the model. The divergence is still computed above and
            # logged below so the prospective model-accuracy tests continue.
            if self._mode == "harvest":
                direction = "SELL"
            else:
                direction = "BUY" if c.divergence > 0 else "SELL"

            # Extreme-spread veto (harvest only): IV far above trailing
            # realized usually means the market is pricing real incoming vol,
            # not extra premium — the one sell bucket that reliably lost
            # across 7 years of history. Demote, keep auditable.
            if self._mode == "harvest":
                trail63 = trail63_by_symbol.get(c.symbol)
                if trail63 is not None and (c.atm_iv - trail63) > self._spread_veto:
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
                            f"extreme-spread veto: atm_iv-trail63="
                            f"{c.atm_iv - trail63:.3f} > {self._spread_veto} "
                            "(market likely pricing incoming vol)"
                        ),
                    ))
                    continue

            # Long-straddle exclusion: never buy premium on the configured
            # symbols (index ETFs). Demote the BUY to non-actionable but keep it
            # in all_signals so the skip is auditable. SELL signals pass through.
            if direction == "BUY" and c.symbol in self._no_long_straddle:
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
                        f"long-straddle excluded for {c.symbol} "
                        f"(index ETF / variance-risk premium)"
                    ),
                ))
                continue

            # Event-suspect: divergences above the cap are almost certainly
            # driven by a known event (earnings, FDA, FOMC, product launch)
            # that the market is pricing and the model structurally cannot
            # capture. Demote rather than trade. Model mode only — harvest
            # doesn't trade the divergence, so its analogue is the
            # extreme-spread veto above plus the earnings filter below.
            if self._mode == "model" and abs(c.divergence) > self._max_divergence:
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

            # Earnings filter. Model mode: fixed buffer only — the old
            # today->expiration rule over-blocked 45-DTE entries for reports a
            # month out with no IV ramp yet (see 9364f85). Harvest mode: the
            # window extends to the candidate's EXPIRATION — entries go out to
            # DTE 15 and ride to expiry, so a report on day 10 is exactly what
            # kills the premium seller; "imminent" is the wrong test when you
            # can't exit before the event. Failing open is intentional — a
            # flaky earnings API must not halt trading.
            earnings_window_end = c.expiration if self._mode == "harvest" else None
            earnings_demote = self._check_earnings(c.symbol, today, earnings_window_end)
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

            # Below z threshold: emit non-actionable signal for diagnostics.
            # Model mode only — harvest sells every eligible name; there is no
            # ranking signal worth gating on (nothing ordered the premium in
            # any historical test), so the z-score is diagnostic-only there.
            if self._mode == "model" and abs(cs_z) < self._z_threshold:
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
        if self._mode == "harvest":
            # Nearest expiration first (premium concentrates near expiry),
            # liquidity breaks ties. One signal per symbol per cycle — the
            # weekly re-entry cadence builds the ladder; multiple same-name
            # expirations in one cycle would just stack correlated risk that
            # the per-ticker exposure gate has to unwind.
            actionable.sort(key=lambda s: (s.dte, -s.liquidity_score))
            seen: set[str] = set()
            deduped = []
            for s in actionable:
                if s.symbol not in seen:
                    seen.add(s.symbol)
                    deduped.append(s)
            actionable = deduped
        else:
            actionable.sort(key=lambda s: (-abs(s.cross_sectional_z), -s.liquidity_score))
        return actionable[:top_n], all_signals

    def _check_earnings(
        self, symbol: str, today: date, window_end: date | None = None
    ) -> tuple[date, str] | None:
        """Return (earnings_date, diagnostic_note) if the signal should be demoted
        for earnings risk, None if it should pass. The window runs from today to
        `window_end` (the candidate's expiration, for held-to-expiry structures)
        or the fixed buffer, whichever is later. Fails open: missing calendar,
        no API key, or no data for the symbol all return None."""
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

        # SELL → short iron condor: short ATM call + short ATM put + long OTM wings.
        # σ for wing placement: the model's view in model mode; the market's own
        # ATM IV in harvest mode (there is no model thesis to size wings off,
        # and market σ places wings wider exactly when vol is priced higher).
        wing_iv = c.predicted_iv_equivalent if self._mode == "model" else c.atm_iv
        wings = _pick_iron_condor_wings(
            c.chain, atm_call.strike, wing_iv, c.dte, c.underlying_price,
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
