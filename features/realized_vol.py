import numpy as np
import pandas as pd


def rolling_rv(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std(ddof=0)


def ewma_vol(returns: pd.Series, lam: float) -> pd.Series:
    var = (returns ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var)


def har_rv_components(returns: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "har_rv_daily": returns.abs(),
        "har_rv_weekly": rolling_rv(returns, 5),
        "har_rv_monthly": rolling_rv(returns, 21),
    })


def acf_squared_returns(returns: pd.Series, lag: int, window: int = 63) -> pd.Series:
    sq = returns ** 2
    return sq.rolling(window).corr(sq.shift(lag))
