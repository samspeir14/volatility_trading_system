import math
import sys

import numpy as np
import pandas as pd

from model.evaluation import (
    lagged_rv_forecast,
    per_horizon_metrics,
    r2_vs_baseline,
    regression_metrics,
    within_ticker_r2,
)


def _panel(symbol_values: dict[str, list[float]]) -> pd.Series:
    parts = []
    for sym, vals in symbol_values.items():
        idx = pd.MultiIndex.from_product(
            [[sym], pd.date_range("2024-01-01", periods=len(vals))],
            names=["symbol", "date"],
        )
        parts.append(pd.Series(vals, index=idx))
    return pd.concat(parts)


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


def test_within_r2_level_only_model():
    # High-vol ticker oscillating around 0.50, low-vol around 0.15. A model
    # that only predicts each ticker's own mean has zero timing skill: pooled
    # R² is inflated by the level gap, within-ticker R² must be ~0.
    actual = _panel({
        "TSLA": [0.45, 0.55, 0.45, 0.55, 0.45, 0.55],
        "KO": [0.12, 0.18, 0.12, 0.18, 0.12, 0.18],
    })
    predicted = _panel({
        "TSLA": [0.50] * 6,
        "KO": [0.15] * 6,
    })
    pooled = regression_metrics(actual, predicted)["r2"]
    within = within_ticker_r2(actual, predicted)
    assert pooled > 0.9, f"pooled R² should be inflated by levels, got {pooled}"
    assert math.isclose(within, 0.0, abs_tol=1e-9), f"within R² should be 0, got {within}"
    print(f"level-only model: pooled={pooled:.3f} within={within:.3f}")


def test_within_r2_perfect_and_timing_skill():
    actual = _panel({
        "A": [0.10, 0.20, 0.15, 0.25],
        "B": [0.40, 0.50, 0.45, 0.55],
    })
    assert math.isclose(within_ticker_r2(actual, actual), 1.0)
    # Damped-toward-symbol-mean predictions retain timing information:
    # within R² should be strictly between 0 and 1.
    damped = actual.groupby(level="symbol").transform(
        lambda s: s.mean() + 0.5 * (s - s.mean())
    )
    within = within_ticker_r2(actual, damped)
    assert 0.0 < within < 1.0, f"expected 0 < within < 1, got {within}"
    print(f"timing skill: perfect=1.0 damped within={within:.3f}")


def test_r2_vs_baseline():
    actual = _panel({"A": [0.10, 0.20, 0.15, 0.25, 0.30]})
    baseline = _panel({"A": [0.12, 0.17, 0.18, 0.21, 0.26]})
    # Model identical to baseline ties it: skill = 0.
    assert math.isclose(r2_vs_baseline(actual, baseline, baseline), 0.0, abs_tol=1e-9)
    # Perfect model: skill = 1.
    assert math.isclose(r2_vs_baseline(actual, actual, baseline), 1.0)
    # Worse than baseline: negative.
    worse = baseline + 0.05
    assert r2_vs_baseline(actual, worse, baseline) < 0.0
    print("r2_vs_baseline: tie=0, perfect=1, worse<0")


def test_lagged_rv_forecast_values():
    returns = pd.Series(
        [0.01, -0.02, 0.03, -0.01, 0.02],
        index=pd.date_range("2024-01-01", periods=5),
    )
    out = lagged_rv_forecast({"A": returns}, horizon=2)
    assert list(out.index.names) == ["symbol", "date"]
    assert len(out) == 5
    # First horizon-1 rows are NaN (not enough trailing history).
    assert math.isnan(out.iloc[0])
    # At t=1: std([0.01, -0.02], ddof=0) = 0.015
    assert math.isclose(out.iloc[1], np.std([0.01, -0.02]), rel_tol=1e-12)
    # At t=4: std([-0.01, 0.02], ddof=0) = 0.015
    assert math.isclose(out.iloc[4], np.std([-0.01, 0.02]), rel_tol=1e-12)
    print(f"lagged_rv_forecast: n={len(out)} values check out")


def test_qlike_perfect_forecast_is_zero():
    from model.evaluation import qlike

    a = pd.Series([0.01, 0.04, 0.0002])
    assert math.isclose(qlike(a, a.copy()), 0.0, abs_tol=1e-15)
    print("qlike: perfect forecast scores 0")


def test_qlike_penalizes_underforecast_more():
    from model.evaluation import qlike

    a = pd.Series([0.02] * 10)
    under = qlike(a, a / 2.0)  # forecast half the variance
    over = qlike(a, a * 2.0)   # forecast double the variance
    assert under > over > 0.0
    print(f"qlike: under-forecast {under:.4f} > over-forecast {over:.4f}")


def test_qlike_drops_nan_and_nonpositive():
    from model.evaluation import qlike

    a = pd.Series([0.01, np.nan, 0.02, 0.03])
    f = pd.Series([0.01, 0.02, 0.0, 0.03])
    # row1 (NaN actual) and row2 (nonpositive forecast) dropped → perfect rows
    assert math.isclose(qlike(a, f), 0.0, abs_tol=1e-15)
    assert math.isnan(qlike(pd.Series(dtype=float), pd.Series(dtype=float)))
    print("qlike: NaN and nonpositive rows dropped; empty → NaN")


def test_per_ticker_r2_median():
    from model.evaluation import per_ticker_r2_median

    rng = np.random.default_rng(0)
    idx_a = pd.MultiIndex.from_product(
        [["A"], pd.date_range("2024-01-01", periods=50)], names=["symbol", "date"]
    )
    idx_b = pd.MultiIndex.from_product(
        [["B"], pd.date_range("2024-01-01", periods=50)], names=["symbol", "date"]
    )
    y_a = pd.Series(rng.normal(0, 1, 50), index=idx_a)
    y_b = pd.Series(rng.normal(0, 1, 50), index=idx_b)
    y = pd.concat([y_a, y_b])
    # A predicted perfectly (r2=1), B predicted by 0 (r2≈0) → median ≈ 0.5
    pred = pd.concat([y_a, pd.Series(0.0, index=idx_b)])
    med = per_ticker_r2_median(y, pred)
    per_b = 1.0 - float(((y_b - 0.0) ** 2).sum() / ((y_b - y_b.mean()) ** 2).sum())
    assert math.isclose(med, (1.0 + per_b) / 2.0, rel_tol=1e-9)
    print(f"per_ticker_r2_median: median of (1.0, {per_b:.4f}) = {med:.4f}")


def main() -> int:
    test_basic_metrics()
    test_perfect_prediction()
    test_constant_mean_predictor()
    test_nan_handling()
    test_empty_returns_nan()
    test_per_horizon_metrics_shape()
    test_within_r2_level_only_model()
    test_within_r2_perfect_and_timing_skill()
    test_r2_vs_baseline()
    test_lagged_rv_forecast_values()
    test_qlike_perfect_forecast_is_zero()
    test_qlike_penalizes_underforecast_more()
    test_qlike_drops_nan_and_nonpositive()
    test_per_ticker_r2_median()
    print("all evaluation tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
