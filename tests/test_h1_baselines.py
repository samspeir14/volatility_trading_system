"""Tests for the h=1 deviation baselines and the HAR-RV model."""
import math
import sys

import numpy as np
import pandas as pd

from model.h1_baselines import ewma_deviation, persistence_deviation
from model.har_model import H1_HAR_FEATURES, HARRVPredictor


def _multi(symbol: str, values: np.ndarray, idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(
        values,
        index=pd.MultiIndex.from_product([[symbol], idx], names=["symbol", "date"]),
    )


def test_persistence_is_lv_minus_b():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    lv = _multi("AAPL", np.array([1.0, 2.0, 3.0, 4.0, 5.0]), idx)
    b = _multi("AAPL", np.array([0.5, 1.0, np.nan, 2.0, 2.5]), idx)
    dev = persistence_deviation(lv, b)
    assert len(dev) == 4  # NaN b row dropped
    assert math.isclose(float(dev.loc[("AAPL", idx[0])]), 0.5)
    assert math.isclose(float(dev.loc[("AAPL", idx[4])]), 2.5)
    print("persistence: dev = lv - b, NaN rows dropped")


def test_ewma_deviation_alignment():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=50, freq="B")
    returns = pd.Series(rng.normal(0, 0.01, 50), index=idx)
    b = _multi("AAPL", np.full(50, -4.5), idx)
    dev = ewma_deviation({"AAPL": returns}, b, lam=0.94)
    # spot-check the first value: ewma var at t0 = r0² (adjust=False seed)
    expected0 = math.log(abs(returns.iloc[0]) + 1e-8) - (-4.5)
    assert math.isclose(float(dev.iloc[0]), expected0, rel_tol=1e-9)
    assert dev.index.names == ["symbol", "date"]
    assert len(dev) == 50
    print("ewma: aligned to (symbol, date), first value matches formula")


def test_har_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    n = 2000
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    X = pd.DataFrame(
        {c: rng.normal(0, 1, n) for c in H1_HAR_FEATURES},
        index=pd.MultiIndex.from_product([["AAPL"], idx], names=["symbol", "date"]),
    )
    true_coefs = np.array([0.6, -0.3, 0.15])
    y = pd.Series(
        0.2 + X.to_numpy() @ true_coefs + rng.normal(0, 0.001, n),
        index=X.index,
    )
    model = HARRVPredictor()
    model.fit(X, y)
    assert math.isclose(model.intercept, 0.2, abs_tol=0.01)
    np.testing.assert_allclose(model.coefs, true_coefs, atol=0.01)
    preds = model.predict(X)
    resid = preds - y.to_numpy()
    assert float(np.abs(resid).mean()) < 0.01
    print(f"har_ols: recovered coefs {np.round(model.coefs, 3)} ≈ {true_coefs}")


def test_har_handles_nan_rows_and_save_load(tmp_path=None):
    import tempfile
    from pathlib import Path

    rng = np.random.default_rng(2)
    n = 200
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    X = pd.DataFrame(
        {c: rng.normal(0, 1, n) for c in H1_HAR_FEATURES},
        index=pd.MultiIndex.from_product([["MSFT"], idx], names=["symbol", "date"]),
    )
    y = pd.Series(X[H1_HAR_FEATURES[0]].to_numpy() * 0.5, index=X.index)
    X.iloc[5, 0] = np.nan  # NaN feature row must be excluded from the fit
    model = HARRVPredictor()
    model.fit(X, y)

    preds = model.predict(X)
    assert np.isnan(preds[5]), "NaN feature row should predict NaN"
    assert np.isfinite(preds[6])

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "har_h1_test.joblib"
        model.save(path)
        loaded = HARRVPredictor.load(path)
        np.testing.assert_allclose(loaded.predict(X)[6:], preds[6:])
        assert loaded.feature_columns == model.feature_columns
    print("har_nan_saveload: NaN rows excluded from fit, round-trips via joblib")


def main() -> int:
    test_persistence_is_lv_minus_b()
    test_ewma_deviation_alignment()
    test_har_recovers_known_coefficients()
    test_har_handles_nan_rows_and_save_load()
    print("all h1_baselines tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
