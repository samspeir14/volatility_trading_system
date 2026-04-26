import pandas as pd

from config import Ticker


def market_avg_rv(rv21_panel: pd.DataFrame) -> pd.Series:
    return rv21_panel.mean(axis=1)


def sector_avg_rv(rv21_panel: pd.DataFrame, watchlist: list[Ticker]) -> pd.DataFrame:
    sector_map = {t.symbol: t.sector for t in watchlist}
    by_sector: dict[str, pd.Series] = {}
    for sector in set(sector_map.values()):
        peers = [s for s, sec in sector_map.items() if sec == sector and s in rv21_panel.columns]
        if peers:
            by_sector[sector] = rv21_panel[peers].mean(axis=1)
    return pd.DataFrame(
        {s: by_sector[sector_map[s]] for s in rv21_panel.columns if sector_map.get(s) in by_sector}
    )


def rolling_corr_with_spy(returns_panel: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    if "SPY" not in returns_panel.columns:
        raise ValueError("returns_panel must contain SPY")
    spy = returns_panel["SPY"]
    return returns_panel.apply(lambda col: col.rolling(window).corr(spy))


def vix_features(vix: pd.Series, vix9d: pd.Series, vix3m: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "vix_level": vix,
        "vix9d_to_vix": vix9d / vix,
        "vix3m_to_vix": vix3m / vix,
    })
