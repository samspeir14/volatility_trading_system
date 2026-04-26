from dataclasses import dataclass
from pathlib import Path

import yaml

from config.settings import Settings, load_settings

WATCHLIST_PATH = Path(__file__).parent / "watchlist.yaml"

MARKET_INDICES: tuple[str, ...] = ("SPY", "VIX", "VIX9D", "VIX3M")


@dataclass(frozen=True)
class Ticker:
    symbol: str
    sector: str


def load_watchlist(path: Path = WATCHLIST_PATH) -> list[Ticker]:
    with path.open() as f:
        raw = yaml.safe_load(f)
    return [Ticker(symbol=t["symbol"], sector=t["sector"]) for t in raw["tickers"]]


__all__ = ["MARKET_INDICES", "Settings", "Ticker", "load_settings", "load_watchlist"]
