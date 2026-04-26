import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from config import Ticker
from data.async_client import AsyncTradierClient, OptionContract


@dataclass(frozen=True)
class TickerSnapshot:
    symbol: str
    sector: str
    underlying: dict
    contracts: list[OptionContract]


@dataclass(frozen=True)
class ScanResult:
    fetched_at: datetime
    snapshots: dict[str, TickerSnapshot]

    def __getitem__(self, symbol: str) -> TickerSnapshot:
        return self.snapshots[symbol]

    def __len__(self) -> int:
        return len(self.snapshots)

    def __iter__(self):
        return iter(self.snapshots.values())

    @property
    def total_contracts(self) -> int:
        return sum(len(s.contracts) for s in self.snapshots.values())


class MarketData:
    def __init__(self, client: AsyncTradierClient, watchlist: list[Ticker]):
        self._client = client
        self._watchlist = watchlist
        self._exp_cache: dict[str, tuple[datetime, list[date]]] = {}
        self._exp_ttl = timedelta(hours=1)

    async def _get_expirations_cached(self, symbol: str) -> list[date]:
        now = datetime.now(timezone.utc)
        cached = self._exp_cache.get(symbol)
        if cached and (now - cached[0]) < self._exp_ttl:
            return cached[1]
        exps = await self._client.get_expirations(symbol)
        self._exp_cache[symbol] = (now, exps)
        return exps

    async def scan(
        self,
        expiration_window: tuple[int, int] = (14, 45),
        today: date | None = None,
    ) -> ScanResult:
        fetched_at = datetime.now(timezone.utc)
        ref = today or date.today()
        min_exp = ref + timedelta(days=expiration_window[0])
        max_exp = ref + timedelta(days=expiration_window[1])

        symbols = [t.symbol for t in self._watchlist]
        quotes_task = asyncio.create_task(self._client.get_quotes(symbols))

        async def fetch_for_ticker(ticker: Ticker) -> tuple[Ticker, list[OptionContract]]:
            exps = await self._get_expirations_cached(ticker.symbol)
            in_window = [e for e in exps if min_exp <= e <= max_exp]
            if not in_window:
                return ticker, []
            chains = await asyncio.gather(*[
                self._client.get_chain(ticker.symbol, e) for e in in_window
            ])
            return ticker, [c for chain in chains for c in chain]

        ticker_results = await asyncio.gather(*[fetch_for_ticker(t) for t in self._watchlist])
        quotes = await quotes_task

        snapshots = {
            t.symbol: TickerSnapshot(
                symbol=t.symbol,
                sector=t.sector,
                underlying=quotes.get(t.symbol, {}),
                contracts=contracts,
            )
            for t, contracts in ticker_results
        }
        return ScanResult(fetched_at=fetched_at, snapshots=snapshots)
