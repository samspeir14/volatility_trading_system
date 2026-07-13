"""Scheduled macro event dates (FOMC decisions, CPI releases).

Why this exists: for rate/index-linked products (TLT, SLV, SPY/QQQ/IWM/DIA,
XLF) an FOMC decision or CPI print IS their earnings — the one scheduled event
that dominates the distribution. The single-name earnings filter has no
analogue for them, so a premium seller holds short gamma straight through the
release. Scope is deliberately narrow: single-name equities are NOT gated on
macro dates — the 7-year VRP result was fat unconditionally *including* every
FOMC/CPI day, and macro releases are exactly the diversifiable ambient vol the
harvest exists to sell. Correlated-crash exposure across the book is bounded
by the portfolio wing-risk cap, not by skipping entries.

MAINTAINED BY HAND — extend once a year:
  * FOMC: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    (dates below are the DECISION day, the meeting's second day)
  * CPI:  https://www.bls.gov/schedule/news_release/cpi.htm

The calendar fails open past its horizon: when the table has no dates on or
after `today`, a single warning is logged per process and nothing is demoted.
"""
from __future__ import annotations

import logging
from datetime import date

logger = logging.getLogger(__name__)


# (date, label). Grouped by event type for hand-editing; MacroCalendar sorts.
MACRO_EVENTS: tuple[tuple[date, str], ...] = (
    # --- 2026 FOMC decision days ---
    (date(2026, 1, 28), "FOMC decision"),
    (date(2026, 3, 18), "FOMC decision"),
    (date(2026, 4, 29), "FOMC decision"),
    (date(2026, 6, 17), "FOMC decision"),
    (date(2026, 7, 29), "FOMC decision"),
    (date(2026, 9, 16), "FOMC decision"),
    (date(2026, 10, 28), "FOMC decision"),
    (date(2026, 12, 9), "FOMC decision"),
    # --- 2026 CPI releases (8:30 ET) — verified against the BLS schedule
    # 2026-07-13 (three independent mirrors; bls.gov blocks bots). Re-verify
    # when extending; release days drift a day or two month to month ---
    (date(2026, 1, 13), "CPI release"),
    (date(2026, 2, 11), "CPI release"),
    (date(2026, 3, 11), "CPI release"),
    (date(2026, 4, 10), "CPI release"),
    (date(2026, 5, 12), "CPI release"),
    (date(2026, 6, 10), "CPI release"),
    (date(2026, 7, 14), "CPI release"),
    (date(2026, 8, 12), "CPI release"),
    (date(2026, 9, 11), "CPI release"),
    (date(2026, 10, 14), "CPI release"),
    (date(2026, 11, 10), "CPI release"),
    (date(2026, 12, 10), "CPI release"),
)


class MacroCalendar:
    def __init__(self, events: tuple[tuple[date, str], ...] = MACRO_EVENTS):
        self._events = tuple(sorted(events))
        self._warned_stale = False

    def next_event_in_window(
        self, start: date, end: date,
    ) -> tuple[date, str] | None:
        """First event with start <= event_date <= end, or None. Fails open
        (with one warning per process) when the table has aged out entirely —
        a stale hand-maintained calendar must not halt trading silently."""
        if self._events and self._events[-1][0] < start:
            if not self._warned_stale:
                logger.warning(
                    "macro calendar has no dates on/after %s (last entry %s) — "
                    "macro filter is a no-op until data/macro_calendar.py is "
                    "extended",
                    start, self._events[-1][0],
                )
                self._warned_stale = True
            return None
        for event_date, label in self._events:
            if start <= event_date <= end:
                return (event_date, label)
        return None
