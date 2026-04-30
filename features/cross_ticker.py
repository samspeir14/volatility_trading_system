import numpy as np
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


RATIO_FEATURE_COLUMNS: list[str] = [
    "rv21_vs_market", "rv21_vs_sector",
    "rv_5_21_ratio", "rv_10_63_ratio",
    "garch_vs_rv21", "ewma_94_97_ratio", "vix_vs_rv21_ann",
]


def add_ratio_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 7 ratio features that make implicit relationships
    explicit. Inputs come from the existing baseline feature columns
    (rv_5/10/21/63, ewma_vol_94/97, market_avg_rv21, sector_avg_rv21,
    garch_forecast_var, vix_level). Returns a new DataFrame with the same
    index as feature_df and the 7 ratio columns."""
    rv_21 = feature_df["rv_21"].replace(0, np.nan)
    rv_63 = feature_df["rv_63"].replace(0, np.nan)
    market_rv = feature_df["market_avg_rv21"].replace(0, np.nan)
    sector_rv = feature_df["sector_avg_rv21"].replace(0, np.nan)
    ewma_97 = feature_df["ewma_vol_97"].replace(0, np.nan)
    garch_var = feature_df["garch_forecast_var"].clip(lower=0.0)

    out = pd.DataFrame(index=feature_df.index)
    out["rv21_vs_market"] = feature_df["rv_21"] / market_rv
    out["rv21_vs_sector"] = feature_df["rv_21"] / sector_rv
    out["rv_5_21_ratio"] = feature_df["rv_5"] / rv_21
    out["rv_10_63_ratio"] = feature_df["rv_10"] / rv_63
    out["garch_vs_rv21"] = np.sqrt(garch_var) / rv_21
    out["ewma_94_97_ratio"] = feature_df["ewma_vol_94"] / ewma_97
    out["vix_vs_rv21_ann"] = feature_df["vix_level"] / (
        feature_df["rv_21"] * np.sqrt(252.0)
    ).replace(0, np.nan)

    return out.replace([np.inf, -np.inf], np.nan)
