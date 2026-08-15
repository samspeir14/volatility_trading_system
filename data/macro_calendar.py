"""Scheduled macro release dates (FOMC, CPI, PPI, PCE, NFP), 2022-2026.

Two consumers with deliberately different scope:

1. TRADE GATING (MacroCalendar): rate/index-linked products only (TLT, SLV,
   SPY/QQQ/IWM/DIA, XLF) are gated on GATING_LABELS (FOMC + CPI) — an FOMC
   decision or CPI print IS their earnings. Single-name equities are NOT
   gated: the 7-year VRP result was fat unconditionally *including* every
   macro day, and macro releases are exactly the diversifiable ambient vol a
   premium seller is paid to hold.
2. MODEL FEATURES (events_by_label): the h=1 model reads all five series as
   event-day dummies for every ticker — the model prices per-name macro
   sensitivity instead of a blanket rule.

SOURCES (backfilled 2026-08-14; hand-verify when extending):
  * FOMC decision days (meeting's second day):
    federalreserve.gov/monetarypolicy/fomccalendars.htm. The 2025-08-22
    notation vote is excluded (not a scheduled decision).
  * CPI/PPI/NFP/PCE 2022-2026: the OMB/Census "Schedule of Release Dates for
    Principal Federal Economic Indicators" PDFs (census.gov/econcards,
    whitehouse.gov for CY2026), cross-checked against actuals on independent
    mirrors (inflationdata.com, tradingview, ebc.com, usinflationcalculator).
  * Q4-2025 shutdown overlays (scheduled != actual): Sep CPI released
    2025-10-24 (BLS reschedule notice); Oct CPI CANCELED, Nov CPI 2025-12-18;
    Sep NFP released 2025-11-20, Oct jobs report CANCELED, Nov NFP 2025-12-16
    (BLS via NBC/EBC); Sep PCE released 2025-12-05, Oct+Nov PCE combined into
    2026-01-22 (bea.gov). PPI Oct-Dec 2025 entries are OMITTED — post-shutdown
    actuals could not be verified. 2026-02-13 CPI was rescheduled from 02-11
    (usinflationcalculator); the table carries the actual.

The calendar fails open past its horizon: when the table has no dates on or
after `today`, a single warning is logged per process and nothing is demoted.
Extend once a year from the next OMB schedule PDF.
"""
from __future__ import annotations

import logging
from datetime import date

logger = logging.getLogger(__name__)


def _d(year: int, pairs: list[tuple[int, int]], label: str) -> list[tuple[date, str]]:
    return [(date(year, m, dd), label) for m, dd in pairs]


FOMC = "FOMC decision"
CPI = "CPI release"
PPI = "PPI release"
PCE = "PCE release"
NFP = "NFP release"

MACRO_RELEASE_HISTORY: tuple[tuple[date, str], ...] = tuple(
    # --- FOMC decision days ---
    _d(2022, [(1, 26), (3, 16), (5, 4), (6, 15), (7, 27), (9, 21), (11, 2), (12, 14)], FOMC)
    + _d(2023, [(2, 1), (3, 22), (5, 3), (6, 14), (7, 26), (9, 20), (11, 1), (12, 13)], FOMC)
    + _d(2024, [(1, 31), (3, 20), (5, 1), (6, 12), (7, 31), (9, 18), (11, 7), (12, 18)], FOMC)
    + _d(2025, [(1, 29), (3, 19), (5, 7), (6, 18), (7, 30), (9, 17), (10, 29), (12, 10)], FOMC)
    + _d(2026, [(1, 28), (3, 18), (4, 29), (6, 17), (7, 29), (9, 16), (10, 28), (12, 9)], FOMC)
    # --- CPI releases (8:30 ET) ---
    + _d(2022, [(1, 12), (2, 10), (3, 10), (4, 12), (5, 11), (6, 10), (7, 13), (8, 10), (9, 13), (10, 13), (11, 10), (12, 13)], CPI)
    + _d(2023, [(1, 12), (2, 14), (3, 14), (4, 12), (5, 10), (6, 13), (7, 12), (8, 10), (9, 13), (10, 12), (11, 14), (12, 12)], CPI)
    + _d(2024, [(1, 11), (2, 13), (3, 12), (4, 10), (5, 15), (6, 12), (7, 11), (8, 14), (9, 11), (10, 10), (11, 13), (12, 11)], CPI)
    # 2025: Sep-data release moved to 10-24 by the shutdown; Oct-data CANCELED;
    # Nov-data 12-18.
    + _d(2025, [(1, 15), (2, 12), (3, 12), (4, 10), (5, 13), (6, 11), (7, 15), (8, 12), (9, 11), (10, 24), (12, 18)], CPI)
    # 2026: 02-13 is the actual (rescheduled from the scheduled 02-11).
    + _d(2026, [(1, 13), (2, 13), (3, 11), (4, 10), (5, 12), (6, 10), (7, 14), (8, 12), (9, 11), (10, 14), (11, 10), (12, 10)], CPI)
    # --- PPI releases ---
    + _d(2022, [(1, 13), (2, 15), (3, 15), (4, 13), (5, 12), (6, 14), (7, 14), (8, 11), (9, 14), (10, 12), (11, 15), (12, 9)], PPI)
    + _d(2023, [(1, 18), (2, 16), (3, 15), (4, 13), (5, 11), (6, 14), (7, 13), (8, 11), (9, 14), (10, 11), (11, 15), (12, 13)], PPI)
    + _d(2024, [(1, 12), (2, 16), (3, 14), (4, 11), (5, 14), (6, 13), (7, 12), (8, 13), (9, 12), (10, 11), (11, 14), (12, 12)], PPI)
    # 2025: Oct-Dec entries omitted (shutdown; actuals unverified).
    + _d(2025, [(1, 14), (2, 13), (3, 13), (4, 11), (5, 15), (6, 12), (7, 16), (8, 14), (9, 10)], PPI)
    + _d(2026, [(1, 14), (2, 12), (3, 12), (4, 14), (5, 13), (6, 11), (7, 15), (8, 13), (9, 10), (10, 15), (11, 13), (12, 15)], PPI)
    # --- PCE releases (Personal Income and Outlays, BEA) ---
    + _d(2022, [(1, 28), (2, 25), (3, 31), (4, 29), (5, 27), (6, 30), (7, 29), (8, 26), (9, 30), (10, 28), (12, 1), (12, 23)], PCE)
    + _d(2023, [(1, 27), (2, 24), (3, 31), (4, 28), (5, 26), (6, 30), (7, 28), (8, 31), (9, 29), (10, 27), (11, 30), (12, 22)], PCE)
    + _d(2024, [(1, 26), (2, 29), (3, 29), (4, 26), (5, 31), (6, 28), (7, 26), (8, 30), (9, 27), (10, 31), (11, 27), (12, 20)], PCE)
    # 2025: Sep-data release came 12-05; Oct/Nov-data combined into 2026-01-22.
    + _d(2025, [(1, 31), (2, 28), (3, 28), (4, 30), (5, 30), (6, 27), (7, 31), (8, 29), (9, 26), (12, 5)], PCE)
    + _d(2026, [(1, 22), (2, 20), (3, 13), (4, 9), (4, 30), (5, 28), (6, 25), (7, 30), (8, 26), (9, 30), (10, 29), (11, 25), (12, 23)], PCE)
    # --- NFP releases (Employment Situation, 8:30 ET) ---
    + _d(2022, [(1, 7), (2, 4), (3, 4), (4, 1), (5, 6), (6, 3), (7, 8), (8, 5), (9, 2), (10, 7), (11, 4), (12, 2)], NFP)
    + _d(2023, [(1, 6), (2, 3), (3, 10), (4, 7), (5, 5), (6, 2), (7, 7), (8, 4), (9, 1), (10, 6), (11, 3), (12, 8)], NFP)
    + _d(2024, [(1, 5), (2, 2), (3, 8), (4, 5), (5, 3), (6, 7), (7, 5), (8, 2), (9, 6), (10, 4), (11, 1), (12, 6)], NFP)
    # 2025: Sep-data release came 11-20; Oct jobs report CANCELED; Nov-data 12-16.
    + _d(2025, [(1, 10), (2, 7), (3, 7), (4, 4), (5, 2), (6, 6), (7, 3), (8, 1), (9, 5), (11, 20), (12, 16)], NFP)
    + _d(2026, [(1, 9), (2, 6), (3, 6), (4, 3), (5, 8), (6, 5), (7, 2), (8, 7), (9, 4), (10, 2), (11, 6), (12, 4)], NFP)
)

# Trade-gating scope: deliberately narrow (see module docstring).
GATING_LABELS: tuple[str, ...] = (FOMC, CPI)

# Backward-compatible gating table consumed by MacroCalendar / signal gates.
MACRO_EVENTS: tuple[tuple[date, str], ...] = tuple(
    e for e in MACRO_RELEASE_HISTORY if e[1] in GATING_LABELS
)


def events_by_label() -> dict[str, frozenset[date]]:
    """Full event table for the feature pipeline: label -> set of dates."""
    out: dict[str, set[date]] = {}
    for event_date, label in MACRO_RELEASE_HISTORY:
        out.setdefault(label, set()).add(event_date)
    return {k: frozenset(v) for k, v in out.items()}


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
