from dataclasses import dataclass
from pathlib import Path

import yaml

from config.settings import Settings, load_settings

WATCHLIST_PATH = Path(__file__).parent / "watchlist.yaml"
# Cheap-ticker universe for the small account profile: names priced so a
# 1-sigma-wing condor's max loss fits a ~$10k account's per-trade budget.
SMALL_WATCHLIST_PATH = Path(__file__).parent / "watchlist_small.yaml"

MARKET_INDICES: tuple[str, ...] = ("SPY", "VIX", "VIX9D", "VIX3M")


@dataclass(frozen=True)
class Ticker:
    symbol: str
    sector: str


def load_watchlist(path: Path = WATCHLIST_PATH) -> list[Ticker]:
    with path.open() as f:
        raw = yaml.safe_load(f)
    return [Ticker(symbol=t["symbol"], sector=t["sector"]) for t in raw["tickers"]]


__all__ = [
    "MARKET_INDICES", "SMALL_WATCHLIST_PATH", "Settings", "Ticker",
    "WATCHLIST_PATH", "load_settings", "load_watchlist",
]
