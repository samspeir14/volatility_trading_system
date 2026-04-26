import math
import sys

import numpy as np
import pandas as pd

from model.evaluation import per_horizon_metrics, regression_metrics


def test_basic_metrics():
    actual = pd.Series([1.0, 2.0, 3.0])
    predicted = pd.Series([1.5, 2.5, 3.5])
    m = regression_metrics(actual, predicted)
    assert m["n"] == 3
    assert math.isclose(m["rmse"], 0.5)
    assert math.isclose(m["mae"], 0.5)
    assert math.isclose(m["bias"], 0.5)
    assert m["r2"] < 1.0
    print(f"basic: rmse={m['rmse']:.3f} mae={m['mae']:.3f} r2={m['r2']:.3f}")


def test_perfect_prediction():
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(actual, actual)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert math.isclose(m["r2"], 1.0)
    assert m["bias"] == 0.0
    print(f"perfect: r2={m['r2']:.3f}")


def test_constant_mean_predictor():
    actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])  # mean of actual
    m = regression_metrics(actual, predicted)
    assert math.isclose(m["r2"], 0.0, abs_tol=1e-9), f"r2 should be 0 for mean predictor, got {m['r2']}"
    assert math.isclose(m["bias"], 0.0)
    print(f"constant mean: r2={m['r2']:.6f} bias={m['bias']:.6f}")


def test_nan_handling():
    actual = pd.Series([1.0, float("nan"), 3.0, 4.0, float("nan")])
    predicted = pd.Series([1.5, 2.5, float("nan"), 4.5, 5.5])
    m = regression_metrics(actual, predicted)
    # Only 2 valid pairs: (1, 1.5) and (4, 4.5)
    assert m["n"] == 2, f"expected n=2, got {m['n']}"
    assert math.isclose(m["rmse"], 0.5)
    print(f"nan handling: n={m['n']}")


def test_empty_returns_nan():
    m = regression_metrics(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert m["n"] == 0
    assert math.isnan(m["rmse"])
    assert math.isnan(m["r2"])
    print("empty: returns NaNs")


def test_per_horizon_metrics_shape():
    n = 50
    idx = pd.date_range("2024-01-01", periods=n)
    np.random.seed(0)
    df = pd.DataFrame({
        "pred_rv_5": np.random.uniform(0.01, 0.03, n),
        "actual_rv_5": np.random.uniform(0.01, 0.03, n),
        "pred_rv_10": np.random.uniform(0.01, 0.03, n),
        "actual_rv_10": np.random.uniform(0.01, 0.03, n),
        "pred_rv_21": np.random.uniform(0.01, 0.03, n),
        "actual_rv_21": np.random.uniform(0.01, 0.03, n),
    }, index=idx)
    out = per_horizon_metrics(df, horizons=(5, 10, 21))
    assert list(out.index) == [5, 10, 21]
    assert list(out.columns) == ["n", "rmse", "mae", "r2", "bias"]
    assert (out["n"] == n).all()
    print(f"per_horizon_metrics shape: {out.shape}")


def main() -> int:
    test_basic_metrics()
    test_perfect_prediction()
    test_constant_mean_predictor()
    test_nan_handling()
    test_empty_returns_nan()
    test_per_horizon_metrics_shape()
    print("all evaluation tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
