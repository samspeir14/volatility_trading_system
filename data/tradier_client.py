import logging
import time
from typing import Any

import requests

from config import Settings

logger = logging.getLogger(__name__)


class TradierAPIError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"Tradier API {status} on {url}: {body[:300]}")
        self.status = status
        self.body = body
        self.url = url


class TradierClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {settings.api_key}",
            "Accept": "application/json",
        })

    @property
    def account_id(self) -> str:
        return self._settings.account_id

    @property
    def env(self) -> str:
        return self._settings.env

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        url = f"{self._settings.base_url}{path}"
        max_retries = self._settings.max_retries
        backoff = 0.5

        for attempt in range(1, max_retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=self._settings.request_timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == max_retries:
                    raise
                logger.warning("Tradier %s attempt %d/%d failed (%s); retrying in %.1fs",
                               path, attempt, max_retries, e.__class__.__name__, backoff)
                time.sleep(backoff)
                backoff *= 2
                continue

            logger.debug("Tradier GET %s -> %d", url, resp.status_code)

            if resp.status_code >= 500 and attempt < max_retries:
                logger.warning("Tradier %s attempt %d/%d returned %d; retrying in %.1fs",
                               path, attempt, max_retries, resp.status_code, backoff)
                time.sleep(backoff)
                backoff *= 2
                continue

            if not resp.ok:
                raise TradierAPIError(resp.status_code, resp.text, url)

            return resp.json()

        raise RuntimeError("unreachable: retry loop exited without return")

    def get_profile(self) -> dict:
        return self._get("/user/profile")["profile"]

    def get_balances(self, account_id: str | None = None) -> dict:
        acct = account_id or self._settings.account_id
        return self._get(f"/accounts/{acct}/balances")["balances"]

    def get_quote(self, symbol: str) -> dict:
        return self.get_quotes([symbol])[symbol]

    def get_quotes(self, symbols: list[str]) -> dict[str, dict]:
        if not symbols:
            raise ValueError("symbols must be a non-empty list")
        data = self._get("/markets/quotes", params={"symbols": ",".join(symbols)})
        quotes = data["quotes"]["quote"]
        if isinstance(quotes, list):
            return {q["symbol"]: q for q in quotes}
        return {quotes["symbol"]: quotes}
