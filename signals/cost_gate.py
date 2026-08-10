"""Transaction-cost gate: only trade when the priced edge clears costs.

expected_edge_usd = |forecast_D - ATM IV| x |net vega| of the structure —
the dollar value of the vol mispricing IF the model is right. (The caller's
model-agreement gate guarantees the divergence points the same way as the
trade before this runs, so the abs() here is sign-safe.) It must cover
COST_MULT x the ENTRY-side friction — per spec, one half-spread per leg plus
fees; COST_MULT=2 therefore roughly covers the full round trip — and no
individual leg may quote wider than MAX_SPREAD_PCT of its mid.
"""
from __future__ import annotations

from dataclasses import dataclass

from data.async_client import OptionContract

SHARES_PER_CONTRACT = 100

# Vega unit convention. Tradier (ORATS) greeks quote vega as $/share per ONE
# PERCENTAGE POINT of IV; divergence is a decimal annualized vol (0.05 = 5
# points), so edge = |div| * 100 points x vega x 100 shares.
# DEPLOY BLOCKER: verify against one live chain via
# `python -m scripts.verify_vega_units` before trusting the gate in sandbox;
# if vega turns out to be per 1.00 of vol, set this to False (edge formula
# then drops the x100 points scaling).
VEGA_PER_PCT_POINT = True


@dataclass(frozen=True)
class CostCheck:
    passed: bool
    expected_edge_usd: float
    total_cost_usd: float
    worst_leg_spread_pct: float
    reason: str  # "" on pass
    # Gate label for the pass-rate tally: "" on pass, else one of
    # "no_quote" | "leg_spread" | "cost_gate" — so the daily Gates line can
    # tell wide markets from chain gaps from thin edge.
    blocked_label: str = ""


def _half_spread(c: OptionContract) -> float:
    return max(c.ask - c.bid, 0.0) / 2.0


def _rel_spread(c: OptionContract) -> float:
    mid = (c.bid + c.ask) / 2.0
    if mid <= 0:
        return float("inf")
    return max(c.ask - c.bid, 0.0) / mid


def evaluate_cost_gate(
    legs,
    chain: list[OptionContract],
    forecast_vol: float,
    atm_iv: float,
    per_contract_fee: float,
    cost_multiple: float = 2.0,
    max_leg_spread_pct: float = 0.05,
) -> CostCheck:
    """`legs` are TradeLeg objects; `chain` supplies their quoted contracts
    (matched by contract symbol). Fails closed on a leg whose contract is
    missing from the chain — cost can't be assessed without a quote."""
    by_symbol = {c.symbol: c for c in chain}
    contracts: list[tuple[object, OptionContract]] = []
    for leg in legs:
        c = by_symbol.get(leg.contract_symbol)
        if c is None:
            return CostCheck(
                False, 0.0, 0.0, float("inf"),
                f"cost gate: no quote for leg {leg.contract_symbol}",
                blocked_label="no_quote",
            )
        contracts.append((leg, c))

    worst_spread = max(_rel_spread(c) for _, c in contracts)
    if worst_spread > max_leg_spread_pct:
        return CostCheck(
            False, 0.0, 0.0, worst_spread,
            f"leg spread {worst_spread:.1%} of mid > cap {max_leg_spread_pct:.0%}",
            blocked_label="leg_spread",
        )

    net_vega = 0.0
    total_cost = 0.0
    for leg, c in contracts:
        sign = 1.0 if leg.side == "buy" else -1.0
        net_vega += sign * c.vega * leg.quantity
        total_cost += (
            _half_spread(c) * SHARES_PER_CONTRACT + per_contract_fee
        ) * leg.quantity

    vol_points = abs(forecast_vol - atm_iv) * (100.0 if VEGA_PER_PCT_POINT else 1.0)
    expected_edge = vol_points * abs(net_vega) * SHARES_PER_CONTRACT

    if expected_edge < cost_multiple * total_cost:
        return CostCheck(
            False, expected_edge, total_cost, worst_spread,
            f"edge ${expected_edge:.0f} < {cost_multiple:g}x cost ${total_cost:.0f}",
            blocked_label="cost_gate",
        )
    return CostCheck(True, expected_edge, total_cost, worst_spread, "")
