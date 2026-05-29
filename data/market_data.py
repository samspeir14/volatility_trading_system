import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from config import Ticker
from data.async_client import AsyncTradierClient, OptionContract

logger = logging.getLogger(__name__)


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
        self._sector_map = {t.symbol: t.sector for t in watchlist}
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

    async def fetch_missing_position_chains(
        self,
        scan: ScanResult,
        needed: dict[str, set[date]],
        today: date | None = None,
    ) -> ScanResult:
        """Augment `scan` with option chains for open-position expirations that
        the scan window didn't cover, returning a new ScanResult.

        Entries are only opened inside the scan window, but as a position ages
        its expiration drifts below the window's lower bound (e.g. DTE < 3 with a
        (3, 60) window). Those legs then vanish from the scan, so mark-to-market
        drops the position ("not all legs found in scan") and the
        expiration-proximity exit can never fire — exactly when gamma/theta risk
        peaks. This pulls just those expirations back in.

        Only expirations on/after `today` are fetched: anything already expired
        is the reconciler's job (and Tradier has no live chain for it). Targeted
        to one chain per missing (symbol, expiration) pair to keep API load low.
        """
        ref = today or date.today()
        to_fetch: list[tuple[str, date]] = []
        for symbol, exps in needed.items():
            snap = scan.snapshots.get(symbol)
            covered = {c.expiration for c in snap.contracts} if snap else set()
            for exp in exps:
                if exp < ref:
                    continue  # already expired — reconciler settles it
                if exp in covered:
                    continue  # already in the scan window
                to_fetch.append((symbol, exp))
        if not to_fetch:
            return scan

        chains = await asyncio.gather(*[
            self._client.get_chain(sym, exp) for sym, exp in to_fetch
        ])

        extra_by_symbol: dict[str, list[OptionContract]] = {}
        for (sym, _exp), chain in zip(to_fetch, chains):
            extra_by_symbol.setdefault(sym, []).extend(chain)

        new_snapshots = dict(scan.snapshots)
        for sym, extra in extra_by_symbol.items():
            existing = new_snapshots.get(sym)
            if existing is None:
                new_snapshots[sym] = TickerSnapshot(
                    symbol=sym,
                    sector=self._sector_map.get(sym, "unknown"),
                    underlying={},
                    contracts=list(extra),
                )
            else:
                new_snapshots[sym] = TickerSnapshot(
                    symbol=existing.symbol,
                    sector=existing.sector,
                    underlying=existing.underlying,
                    contracts=existing.contracts + extra,
                )
        logger.info(
            "augmented scan with %d below-window expiration chain(s): %s",
            len(to_fetch),
            ", ".join(f"{s}@{e.isoformat()}" for s, e in to_fetch),
        )
        return ScanResult(fetched_at=scan.fetched_at, snapshots=new_snapshots)
