import math
import sys

import numpy as np
import pandas as pd

from config import Ticker
from features.cross_ticker import (
    market_avg_rv,
    rolling_corr_with_spy,
    sector_avg_rv,
    vix_features,
)


def test_market_avg_rv():
    idx = pd.date_range("2024-01-01", periods=10)
    panel = pd.DataFrame({
        "AAPL": [0.1] * 10,
        "MSFT": [0.2] * 10,
        "GOOGL": [0.3] * 10,
    }, index=idx)
    avg = market_avg_rv(panel)
    np.testing.assert_allclose(avg.values, 0.2, atol=1e-10)
    print(f"market_avg_rv: avg = {avg.iloc[0]} (expected 0.2)")


def test_sector_avg_rv():
    watchlist = [
        Ticker("AAPL", "tech"),
        Ticker("MSFT", "tech"),
        Ticker("JPM", "financials"),
        Ticker("GS", "financials"),
    ]
    idx = pd.date_range("2024-01-01", periods=5)
    panel = pd.DataFrame({
        "AAPL": [0.1] * 5,
        "MSFT": [0.3] * 5,
        "JPM": [0.5] * 5,
        "GS": [0.7] * 5,
    }, index=idx)
    out = sector_avg_rv(panel, watchlist)
    # AAPL and MSFT both get tech avg = 0.2; JPM and GS both get financials avg = 0.6
    np.testing.assert_allclose(out["AAPL"].values, 0.2, atol=1e-10)
    np.testing.assert_allclose(out["MSFT"].values, 0.2, atol=1e-10)
    np.testing.assert_allclose(out["JPM"].values, 0.6, atol=1e-10)
    np.testing.assert_allclose(out["GS"].values, 0.6, atol=1e-10)
    print("sector_avg_rv: tech=0.2, financials=0.6")


def test_rolling_corr_with_spy_self_is_one():
    np.random.seed(0)
    idx = pd.date_range("2024-01-01", periods=200)
    spy_returns = pd.Series(np.random.normal(0, 0.01, 200), index=idx)
    panel = pd.DataFrame({"SPY": spy_returns, "OTHER": spy_returns * 2 + 0.0001})
    corr = rolling_corr_with_spy(panel, window=21)
    # SPY-with-SPY = 1.0 (after warm-up, where non-NaN)
    spy_self = corr["SPY"].dropna()
    assert (spy_self.round(10) == 1.0).all(), f"SPY-self corr should be 1: {spy_self.unique()[:3]}"
    # OTHER (linear transform of SPY) should also be ~1
    other = corr["OTHER"].dropna()
    assert (other.round(6) == 1.0).all(), f"linear-transform corr should be 1: {other.unique()[:3]}"
    print("rolling_corr_with_spy: SPY-self = 1.0, linear transform = 1.0")


def test_vix_features():
    idx = pd.date_range("2024-01-01", periods=5)
    vix = pd.Series([20.0, 22.0, 18.0, 25.0, 19.0], index=idx)
    vix9d = pd.Series([22.0, 24.0, 17.0, 30.0, 19.0], index=idx)
    vix3m = pd.Series([22.0, 22.0, 20.0, 24.0, 21.0], index=idx)
    out = vix_features(vix, vix9d, vix3m)
    assert list(out.columns) == ["vix_level", "vix9d_to_vix", "vix3m_to_vix"]
    assert math.isclose(out["vix9d_to_vix"].iloc[0], 22.0 / 20.0)
    assert math.isclose(out["vix3m_to_vix"].iloc[3], 24.0 / 25.0)
    print("vix_features: ratios computed correctly")


def main() -> int:
    test_market_avg_rv()
    test_sector_avg_rv()
    test_rolling_corr_with_spy_self_is_one()
    test_vix_features()
    print("all cross_ticker tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
