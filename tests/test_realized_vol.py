import math
import sys

import numpy as np
import pandas as pd

from features.realized_vol import (
    acf_squared_returns,
    ewma_vol,
    har_rv_components,
    rolling_rv,
)


def test_rolling_rv_recovers_known_std():
    np.random.seed(0)
    sigma = 0.01
    r = pd.Series(np.random.normal(0, sigma, 5000))
    rv = rolling_rv(r, 252).dropna()
    # Sample std should be close to true std for n=252; allow 15% tolerance
    assert abs(rv.mean() - sigma) / sigma < 0.15, f"rv mean {rv.mean():.6f} vs true {sigma}"
    print(f"rolling_rv: mean of 252-day RV = {rv.mean():.6f} (true σ = {sigma})")


def test_ewma_vol_matches_riskmetrics_recursion():
    np.random.seed(1)
    r = pd.Series(np.random.normal(0, 0.01, 1000))
    lam = 0.94
    out = ewma_vol(r, lam)
    # Manual recursion: σ²_t = (1-λ)·r²_t + λ·σ²_{t-1}, σ²_0 = r²_0
    var = r.iloc[0] ** 2
    manual = [math.sqrt(var)]
    for i in range(1, len(r)):
        var = (1 - lam) * r.iloc[i] ** 2 + lam * var
        manual.append(math.sqrt(var))
    np.testing.assert_allclose(out.values, manual, rtol=1e-9)
    print(f"ewma_vol: recursion matches RiskMetrics (n={len(r)})")


def test_har_rv_components_shape_and_values():
    r = pd.Series(np.random.normal(0, 0.01, 100), index=pd.date_range("2024-01-01", periods=100))
    har = har_rv_components(r)
    assert list(har.columns) == ["har_rv_daily", "har_rv_weekly", "har_rv_monthly"]
    # Daily component is just |return|
    np.testing.assert_array_equal(har["har_rv_daily"].values, r.abs().values)
    # Weekly = rolling 5-day std
    np.testing.assert_allclose(
        har["har_rv_weekly"].dropna().values,
        rolling_rv(r, 5).dropna().values,
    )
    print("har_rv_components: shape and daily/weekly identities OK")


def test_acf_squared_returns_iid_near_zero():
    np.random.seed(2)
    r = pd.Series(np.random.normal(0, 0.01, 500))
    a1 = acf_squared_returns(r, lag=1, window=63).dropna()
    # IID returns → squared returns also IID → ACF ≈ 0 on average
    assert abs(a1.mean()) < 0.1, f"IID ACF mean too far from 0: {a1.mean():.4f}"
    print(f"acf_squared_returns: mean ACF for IID series = {a1.mean():.4f}")


def main() -> int:
    test_rolling_rv_recovers_known_std()
    test_ewma_vol_matches_riskmetrics_recursion()
    test_har_rv_components_shape_and_values()
    test_acf_squared_returns_iid_near_zero()
    print("all realized_vol tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
