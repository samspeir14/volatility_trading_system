import asyncio
import sys
import time
from datetime import date, timedelta

from config import Ticker, load_settings, load_watchlist
from data import AsyncTradierClient, MarketData


def _next_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


async def test_single_symbol(client: AsyncTradierClient) -> None:
    md = MarketData(client, [Ticker(symbol="AAPL", sector="tech")])
    result = await md.scan()
    assert "AAPL" in result.snapshots
    snap = result["AAPL"]
    assert snap.contracts, "expected ≥1 AAPL contract in window"
    assert snap.underlying.get("symbol") == "AAPL"
    for c in snap.contracts:
        assert c.iv >= 0, f"non-numeric IV on {c.symbol}: {c.iv}"
    print(f"AAPL single-scan: {len(snap.contracts)} contracts in window")


async def test_three_symbol(client: AsyncTradierClient) -> None:
    tickers = [
        Ticker(symbol="AAPL", sector="tech"),
        Ticker(symbol="SPY", sector="etf"),
        Ticker(symbol="QQQ", sector="etf"),
    ]
    md = MarketData(client, tickers)
    t0 = time.monotonic()
    result = await md.scan()
    elapsed = time.monotonic() - t0
    assert len(result) == 3
    for sym in ("AAPL", "SPY", "QQQ"):
        assert sym in result.snapshots, f"missing {sym}"
    print(f"3-symbol scan: {result.total_contracts} contracts in {elapsed*1000:.0f}ms")


async def test_full_watchlist(client: AsyncTradierClient) -> None:
    watchlist = load_watchlist()
    assert len(watchlist) == 20, f"expected 20 tickers, got {len(watchlist)}"
    md = MarketData(client, watchlist)

    t0 = time.monotonic()
    result = await md.scan()
    elapsed = time.monotonic() - t0

    assert len(result) == 20
    print(f"20-ticker full scan: {result.total_contracts} contracts in {elapsed:.2f}s "
          f"(rate-limiter calls={client.rate_limiter.call_count})")
    per_ticker_with_contracts = sum(1 for s in result if s.contracts)
    print(f"  tickers with ≥1 in-window contract: {per_ticker_with_contracts}/20")

    assert elapsed < 15.0, f"FAIL: full scan took {elapsed:.2f}s, must be <15s"
    print(f"PASS: scan completed under the 15s gate")


async def main() -> int:
    settings = load_settings()
    if settings.env != "sandbox":
        print(f"refusing to run against env={settings.env!r}", file=sys.stderr)
        return 2

    async with AsyncTradierClient(settings) as client:
        await test_single_symbol(client)
        client.rate_limiter.reset_counter()
        await test_three_symbol(client)
        client.rate_limiter.reset_counter()
        await test_full_watchlist(client)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
