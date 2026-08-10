"""Level reconstruction and term projection for the h=1 deviation forecast.

The model predicts tomorrow's log-vol deviation dev_hat from the stock's
63-day baseline b_t. To compare against an option's ATM IV at D calendar
days to expiry, the 1-day forecast is projected along a GARCH-persistence
decay toward the baseline and averaged over the option's life — never
compare a 1-day forecast to a long-dated IV directly.

Unit convention: b_t and dev_hat are logs of DAILY vol, so var_k here are
trading-day variances; annualization is sqrt(252). Calendar DTE is
converted to trading days (x 252/365) before averaging so the result is a
Black-Scholes-comparable annualized vol (see the CALENDAR_DAYS_PER_YEAR
note in signals/signal_generator.py for the historical sqrt(T) bug this
avoids).
"""
from __future__ import annotations

import math

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365

PHI_MIN = 0.80
PHI_MAX = 0.99


def reconstruct_level(b_t: float, dev_hat: float) -> tuple[float, float]:
    """(daily_vol, annualized_vol) for day t+1: exp(b_t + dev_hat) and
    that times sqrt(252)."""
    daily = math.exp(b_t + dev_hat)
    return daily, daily * math.sqrt(TRADING_DAYS_PER_YEAR)


def project_term_vol(
    b_t: float,
    dev_hat: float,
    phi: float,
    dte_calendar: int,
) -> float:
    """Annualized vol forecast over the life of an option D calendar days out.

    var_1 = exp(2*(b_t + dev_hat)) decays toward var_base = exp(2*b_t) at
    rate phi (per-ticker GARCH persistence alpha+beta, clipped to
    [0.80, 0.99]):

        var_k     = var_base + (var_1 - var_base) * phi**(k-1),  k = 1..K
        forecast  = sqrt(mean(var_k)) * sqrt(252)

    with K = max(1, round(D * 252/365)) trading days.
    """
    if dte_calendar < 1:
        raise ValueError(f"dte_calendar must be >= 1, got {dte_calendar}")
    phi = min(max(phi, PHI_MIN), PHI_MAX)
    k_days = max(1, round(dte_calendar * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR))
    var_base = math.exp(2.0 * b_t)
    var_1 = math.exp(2.0 * (b_t + dev_hat))
    # mean of var_base + (var_1-var_base)*phi^(k-1) over k=1..K, closed form
    geo_sum = (1.0 - phi ** k_days) / (1.0 - phi)  # sum of phi^(k-1)
    mean_var = var_base + (var_1 - var_base) * geo_sum / k_days
    return math.sqrt(mean_var) * math.sqrt(TRADING_DAYS_PER_YEAR)
