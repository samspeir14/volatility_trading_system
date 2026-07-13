"""Regression tests for the IWM frozen-cache bug (2025-03-06 poison row).

Tradier's history endpoint can return placeholder rows with "NaN" fields.
float("NaN") parses to nan, sqlite binds nan as NULL, and the NOT NULL
constraint then aborts the WHOLE symbol's upsert — freezing its cache at the
last good date (IWM sat at 2025-03-05 for 16 months). Two independent fixes:
get_history skips malformed rows; fetch_and_cache isolates per-symbol
failures so one bad symbol can't abort the batch."""
import asyncio
import sys
import tempfile
from datetime import date
from pathlib import Path

from config import Settings
from data.async_client import AsyncTradierClient, Bar
from data.historical import HistoricalStore, fetch_and_cache


def _mk_settings() -> Settings:
    return Settings(
        api_key="fake", account_id="VA00000000",
        base_url="https://example.invalid/v1", env="sandbox",
    )


def _day(d, o=100.0, h=101.0, lo=99.0, c=100.5, v=1000):
    return {"date": d, "open": o, "high": h, "low": lo, "close": c, "volume": v}


def test_get_history_skips_malformed_rows():
    client = AsyncTradierClient(_mk_settings())
    payload = {"history": {"day": [
        _day("2025-03-04"),
        _day("2025-03-05"),
        # The exact live IWM poison row shape (string "NaN" fields)
        {"date": "2025-03-06", "open": "NaN", "high": "NaN", "low": "NaN",
         "close": 205.28, "volume": 39206266},
        {"date": "2025-03-07", "open": None, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 500},          # null field
        {"date": "2025-03-10", "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 500},          # missing key
        _day("2025-03-11"),
    ]}}

    async def fake_get(path, params=None):
        return payload

    client._get = fake_get
    bars = asyncio.run(client.get_history("IWM", date(2025, 3, 1), date(2025, 3, 15)))
    got_dates = [b.date.isoformat() for b in bars]
    assert got_dates == ["2025-03-04", "2025-03-05", "2025-03-11"], got_dates
    assert all(b.open == 100.0 for b in bars)
    print("get_history: NaN-string, null, and missing-key rows skipped; good rows kept")


def test_skipped_rows_no_longer_freeze_the_cache():
    """End-to-end heal: a symbol frozen before the poison row advances past it
    once malformed rows are skipped (the upsert no longer aborts)."""
    client = AsyncTradierClient(_mk_settings())
    payload = {"history": {"day": [
        {"date": "2025-03-06", "open": "NaN", "high": "NaN", "low": "NaN",
         "close": 205.28, "volume": 39206266},
        _day("2025-03-07"),
        _day("2025-03-10"),
    ]}}

    async def fake_get(path, params=None):
        return payload

    client._get = fake_get
    with tempfile.TemporaryDirectory() as tmp:
        store = HistoricalStore(Path(tmp) / "bars.db")
        # Cache frozen at 03-05 (the IWM state)
        store.upsert_bars("IWM", [Bar(date(2025, 3, 5), 100, 101, 99, 100.5, 1000)])
        assert store.latest_date("IWM") == date(2025, 3, 5)
        result = asyncio.run(fetch_and_cache(
            client, store, ["IWM"], lookback_years=1, today=date(2025, 3, 10),
        ))
        assert store.latest_date("IWM") == date(2025, 3, 10), (
            f"cache still frozen: {store.latest_date('IWM')}"
        )
        assert not result["IWM"].empty
        store.close()
    print("heal: frozen symbol advances past the poison row after the fix")


def test_fetch_and_cache_isolates_symbol_failure():
    """One symbol's fetch raising must not abort the batch: the good symbol
    still upserts, the bad one serves its (empty) cache, nothing propagates."""
    class _FakeClient:
        async def get_history(self, symbol, start, end, interval="daily"):
            if symbol == "BAD":
                raise RuntimeError("boom")
            return [Bar(date(2025, 3, 7), 100, 101, 99, 100.5, 1000)]

    with tempfile.TemporaryDirectory() as tmp:
        store = HistoricalStore(Path(tmp) / "bars.db")
        result = asyncio.run(fetch_and_cache(
            _FakeClient(), store, ["BAD", "GOOD"],
            lookback_years=1, today=date(2025, 3, 10),
        ))
        assert store.latest_date("GOOD") == date(2025, 3, 7), "good symbol must upsert"
        assert store.latest_date("BAD") is None
        assert result["BAD"].empty and not result["GOOD"].empty
        store.close()
    print("isolation: BAD raises, GOOD still cached, no exception propagates")


def main() -> int:
    test_get_history_skips_malformed_rows()
    test_skipped_rows_no_longer_freeze_the_cache()
    test_fetch_and_cache_isolates_symbol_failure()
    print("all history_bar_hygiene tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
