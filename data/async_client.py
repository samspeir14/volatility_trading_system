import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import aiohttp

from config import Settings
from data.tradier_client import TradierAPIError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class OptionContract:
    symbol: str
    underlying: str
    expiration: date
    strike: float
    option_type: str
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float
    fetched_at: datetime


class RateLimiter:
    """Token bucket with per-minute refill, self-tuned from Tradier response headers."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._tokens = capacity
        self._refill_at = time.monotonic() + 60.0
        self._lock = asyncio.Lock()
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset_counter(self) -> None:
        self._call_count = 0

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                if now >= self._refill_at:
                    self._tokens = self._capacity
                    self._refill_at = now + 60.0
                if self._tokens > 0:
                    self._tokens -= 1
                    self._call_count += 1
                    return
                wait = self._refill_at - now
            logger.warning("Rate limiter exhausted; sleeping %.2fs", wait)
            await asyncio.sleep(wait + 0.05)

    def update_from_headers(self, headers) -> None:
        try:
            available = int(headers.get("X-Ratelimit-Available", -1))
            expiry_ms = int(headers.get("X-Ratelimit-Expiry", -1))
        except (TypeError, ValueError):
            return
        if available < 0 or expiry_ms < 0:
            return
        now_ms = time.time() * 1000
        if expiry_ms <= now_ms:
            return
        remaining_secs = (expiry_ms - now_ms) / 1000
        # Take the more conservative of (our local view, server view)
        if available < self._tokens:
            self._tokens = available
        self._refill_at = time.monotonic() + remaining_secs


class AsyncTradierClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._session: aiohttp.ClientSession | None = None
        self.rate_limiter = RateLimiter(settings.rate_limit_per_min)

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self._settings.api_key}",
                "Accept": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=self._settings.request_timeout),
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()
        self._session = None

    @property
    def account_id(self) -> str:
        return self._settings.account_id

    @property
    def env(self) -> str:
        return self._settings.env

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        if self._session is None:
            raise RuntimeError("AsyncTradierClient must be used as an async context manager")

        url = f"{self._settings.base_url}{path}"
        max_retries = self._settings.max_retries
        backoff = 0.5

        for attempt in range(1, max_retries + 1):
            await self.rate_limiter.acquire()

            try:
                async with self._session.get(url, params=params) as resp:
                    self.rate_limiter.update_from_headers(resp.headers)
                    logger.debug("Tradier GET %s -> %d", url, resp.status)

                    if resp.status == 429:
                        expiry_ms = int(resp.headers.get("X-Ratelimit-Expiry", "0") or 0)
                        wait = (expiry_ms - time.time() * 1000) / 1000 if expiry_ms else 5.0
                        wait = max(1.0, wait)
                        logger.warning("Tradier %s 429; sleeping %.2fs", path, wait)
                        await asyncio.sleep(wait + 0.1)
                        continue

                    if resp.status >= 500 and attempt < max_retries:
                        logger.warning("Tradier %s attempt %d/%d returned %d; retrying in %.1fs",
                                       path, attempt, max_retries, resp.status, backoff)
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                    if not resp.ok:
                        body = await resp.text()
                        raise TradierAPIError(resp.status, body, url)

                    return await resp.json()

            except aiohttp.ClientConnectionError as e:
                if attempt == max_retries:
                    raise
                logger.warning("Tradier %s attempt %d/%d connection error (%s); retrying in %.1fs",
                               path, attempt, max_retries, e.__class__.__name__, backoff)
                await asyncio.sleep(backoff)
                backoff *= 2

        raise RuntimeError("unreachable: retry loop exited without return")

    async def get_quotes(self, symbols: list[str]) -> dict[str, dict]:
        if not symbols:
            raise ValueError("symbols must be a non-empty list")
        data = await self._get("/markets/quotes", params={"symbols": ",".join(symbols)})
        quotes = data["quotes"]["quote"]
        if isinstance(quotes, list):
            return {q["symbol"]: q for q in quotes}
        return {quotes["symbol"]: quotes}

    async def get_quote(self, symbol: str) -> dict:
        return (await self.get_quotes([symbol]))[symbol]

    async def get_expirations(self, symbol: str, include_all_roots: bool = True) -> list[date]:
        params: dict[str, Any] = {"symbol": symbol}
        if include_all_roots:
            params["includeAllRoots"] = "true"
        data = await self._get("/markets/options/expirations", params=params)
        exps = data.get("expirations") or {}
        dates = exps.get("date") or []
        if isinstance(dates, str):
            dates = [dates]
        return [date.fromisoformat(d) for d in dates]

    async def get_chain(self, symbol: str, expiration: date, greeks: bool = True) -> list[OptionContract]:
        params = {
            "symbol": symbol,
            "expiration": expiration.isoformat(),
            "greeks": "true" if greeks else "false",
        }
        data = await self._get("/markets/options/chains", params=params)
        opts = data.get("options") or {}
        raw_list = opts.get("option") or []
        if isinstance(raw_list, dict):
            raw_list = [raw_list]

        fetched_at = datetime.now(timezone.utc)
        contracts = []
        for raw in raw_list:
            g = raw.get("greeks") or {}
            contracts.append(OptionContract(
                symbol=raw["symbol"],
                underlying=raw["underlying"],
                expiration=date.fromisoformat(raw["expiration_date"]),
                strike=float(raw["strike"]),
                option_type=raw["option_type"],
                bid=_to_float(raw.get("bid")),
                ask=_to_float(raw.get("ask")),
                last=_to_float(raw.get("last")),
                volume=int(raw.get("volume") or 0),
                open_interest=int(raw.get("open_interest") or 0),
                delta=_to_float(g.get("delta")),
                gamma=_to_float(g.get("gamma")),
                theta=_to_float(g.get("theta")),
                vega=_to_float(g.get("vega")),
                iv=_to_float(g.get("mid_iv")),
                fetched_at=fetched_at,
            ))
        return contracts

    async def get_history(self, symbol: str, start: date, end: date, interval: str = "daily") -> list[Bar]:
        params = {
            "symbol": symbol,
            "interval": interval,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        data = await self._get("/markets/history", params=params)
        hist = data.get("history") or {}
        days = hist.get("day") or []
        if isinstance(days, dict):
            days = [days]
        return [
            Bar(
                date=date.fromisoformat(d["date"]),
                open=float(d["open"]),
                high=float(d["high"]),
                low=float(d["low"]),
                close=float(d["close"]),
                volume=int(d["volume"]),
            )
            for d in days
        ]


def _to_float(value) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)
