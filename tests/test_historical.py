import asyncio
import math
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_settings
from data import AsyncTradierClient, HistoricalStore, compute_log_returns, fetch_and_cache


def test_log_returns() -> None:
    series = pd.Series([100.0, 110.0, 99.0])
    rets = compute_log_returns(series)
    assert len(rets) == 2
    assert math.isclose(rets.iloc[0], math.log(110 / 100), rel_tol=1e-9)
    assert math.isclose(rets.iloc[1], math.log(99 / 110), rel_tol=1e-9)
    pd.testing.assert_series_equal(
        rets,
        np.log(series / series.shift(1)).dropna(),
        check_names=False,
    )
    print("log_returns: OK")


async def test_cache_lifecycle(db_path: Path) -> None:
    settings = load_settings()
    end = date.today()
    start = end - timedelta(days=30)

    # 1. Empty cache → fetch populates
    async with AsyncTradierClient(settings) as client:
        store1 = HistoricalStore(db_path)
        try:
            result = await fetch_and_cache(client, store1, ["AAPL"], lookback_years=0, today=end)
            df = result["AAPL"]
            assert not df.empty, "expected non-empty AAPL bars after first fetch"
            n_after_fetch = store1.row_count("AAPL")
            assert n_after_fetch >= 15, f"expected ≥15 rows after 30d fetch, got {n_after_fetch}"
            calls_after_fetch = client.rate_limiter.call_count
            assert calls_after_fetch >= 1, "expected ≥1 API call on cold fetch"
            print(f"cold fetch: {n_after_fetch} bars cached, {calls_after_fetch} API call(s)")
        finally:
            store1.close()

        # 2. Same range, same store reopened → no new API calls
        store2 = HistoricalStore(db_path)
        try:
            client.rate_limiter.reset_counter()
            # Use same end date to avoid Tradier even being asked for a 1-day delta
            result2 = await fetch_and_cache(client, store2, ["AAPL"], lookback_years=0, today=end)
            df2 = result2["AAPL"]
            assert len(df2) == n_after_fetch, "row count changed on reopen"
            calls_after_reopen = client.rate_limiter.call_count
            assert calls_after_reopen == 0, (
                f"expected 0 API calls on cache hit, got {calls_after_reopen}"
            )
            print(f"persistence-across-restart: {len(df2)} bars from disk, {calls_after_reopen} API call(s)")
        finally:
            store2.close()


async def main() -> int:
    settings = load_settings()
    if settings.env != "sandbox":
        print(f"refusing to run against env={settings.env!r}", file=sys.stderr)
        return 2

    test_log_returns()

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_cache.db"
        await test_cache_lifecycle(db_path)

    print("all historical tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
