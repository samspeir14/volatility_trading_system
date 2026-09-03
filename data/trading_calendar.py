"""US equity-market session calendar: weekdays minus NYSE holidays.

The exit manager's DTE windows count sessions, not calendar days, so a
Friday entry for a Monday expiry (or a Wednesday entry for the Friday after
Thanksgiving) is the 1-day trade it actually is. Weekday arithmetic alone
gets holidays wrong in the UNSAFE direction for time-left counts: it thinks
a session remains that doesn't. Hence the explicit table.

Extend US_MARKET_HOLIDAYS each autumn for the following year (NYSE publishes
the list ~2 years ahead). A date outside COVERED_YEARS falls back to weekday
arithmetic with a one-time warning.
"""
import logging
from datetime import date

logger = logging.getLogger(__name__)

# NYSE full-day closures. Observed dates where the holiday falls on a weekend.
US_MARKET_HOLIDAYS: frozenset[date] = frozenset({
    # 2026
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # Martin Luther King Jr. Day
    date(2026, 2, 16),   # Presidents' Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed; Jul 4 is a Saturday)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
    # 2027
    date(2027, 1, 1),    # New Year's Day
    date(2027, 1, 18),   # Martin Luther King Jr. Day
    date(2027, 2, 15),   # Presidents' Day
    date(2027, 3, 26),   # Good Friday
    date(2027, 5, 31),   # Memorial Day
    date(2027, 6, 18),   # Juneteenth (observed; Jun 19 is a Saturday)
    date(2027, 7, 5),    # Independence Day (observed; Jul 4 is a Sunday)
    date(2027, 9, 6),    # Labor Day
    date(2027, 11, 25),  # Thanksgiving
    date(2027, 12, 24),  # Christmas (observed; Dec 25 is a Saturday)
})

COVERED_YEARS: frozenset[int] = frozenset(d.year for d in US_MARKET_HOLIDAYS)
_warned_years: set[int] = set()


def is_trading_day(d: date) -> bool:
    """True on a regular NYSE session day (half days count as sessions)."""
    if d.weekday() >= 5:
        return False
    if d.year not in COVERED_YEARS and d.year not in _warned_years:
        _warned_years.add(d.year)
        logger.warning(
            "US_MARKET_HOLIDAYS has no entries for %d — counting weekdays only; "
            "extend data/trading_calendar.py", d.year,
        )
    return d not in US_MARKET_HOLIDAYS
