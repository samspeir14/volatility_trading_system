import asyncio
import sys
import time

from config import load_settings
from data import AsyncTradierClient


async def run() -> int:
    settings = load_settings()
    if settings.env != "sandbox":
        print(f"refusing to run against env={settings.env!r}", file=sys.stderr)
        return 2

    symbols = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]

    async with AsyncTradierClient(settings) as client:
        t0 = time.monotonic()
        results = await asyncio.gather(*[client.get_quote(s) for s in symbols])
        elapsed = time.monotonic() - t0

        returned = {q["symbol"] for q in results}
        assert returned == set(symbols), f"missing/extra symbols: {returned} vs {set(symbols)}"
        for q in results:
            assert "last" in q, f"quote missing 'last': {q}"

        print(f"5 quotes concurrently in {elapsed*1000:.0f}ms; rate-limiter call_count={client.rate_limiter.call_count}")
        for q in results:
            print(f"  {q['symbol']:6s} last=${q['last']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
