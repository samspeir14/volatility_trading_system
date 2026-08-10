"""Tests for H1DeviationPredictor routing and the reconciliation logic."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from model.h1_predictor import H1DeviationPredictor
from model.har_model import H1_HAR_FEATURES, HARRVPredictor
from model.lightgbm_model import LightGBMVolPredictor
from model.training import DEFAULT_LGBM_HYPERPARAMS


def _fitted_har(slope: float = 0.5) -> HARRVPredictor:
    rng = np.random.default_rng(0)
    n = 100
    idx = pd.MultiIndex.from_product(
        [["A"], pd.date_range("2024-01-01", periods=n)], names=["symbol", "date"]
    )
    X = pd.DataFrame({c: rng.normal(0, 1, n) for c in H1_HAR_FEATURES}, index=idx)
    y = pd.Series(X[H1_HAR_FEATURES[0]] * slope, index=idx)
    model = HARRVPredictor()
    model.fit(X, y)
    return model


def _fitted_lgbm() -> LightGBMVolPredictor:
    rng = np.random.default_rng(1)
    n = 200
    idx = pd.MultiIndex.from_product(
        [["A"], pd.date_range("2024-01-01", periods=n)], names=["symbol", "date"]
    )
    X = pd.DataFrame({c: rng.normal(0, 1, n) for c in H1_HAR_FEATURES}, index=idx)
    y = pd.Series(rng.normal(0, 0.1, n), index=idx)
    model = LightGBMVolPredictor(horizon=1, hyperparams=DEFAULT_LGBM_HYPERPARAMS)
    model.fit(X, y)
    return model


def _one_row() -> pd.DataFrame:
    return pd.DataFrame(
        {c: [0.5] for c in H1_HAR_FEATURES},
        index=pd.MultiIndex.from_tuples([("A", pd.Timestamp("2025-01-02"))],
                                        names=["symbol", "date"]),
    )


def test_route_honored():
    har = _fitted_har()
    lgbm = _fitted_lgbm()
    p_lgbm = H1DeviationPredictor(lgbm, har, route="lgbm")
    p_har = H1DeviationPredictor(lgbm, har, route="har")
    assert p_lgbm.active_model == "lgbm"
    assert p_har.active_model == "har"
    row = _one_row()
    lgbm_val = float(lgbm.predict(row)[0])
    har_val = float(har.predict(row)[0])
    assert p_lgbm.predict_deviation(row) == lgbm_val
    assert p_har.predict_deviation(row) == har_val
    print("route: lgbm/har each served by the routed model")


def test_missing_lgbm_falls_back_to_har():
    p = H1DeviationPredictor(None, _fitted_har(), route="lgbm")
    assert p.active_model == "har"
    assert np.isfinite(p.predict_deviation(_one_row()))
    print("fallback: route=lgbm with no artifact degrades to HAR")


def test_invalid_route_rejected():
    try:
        H1DeviationPredictor(None, _fitted_har(), route="xgboost")
    except ValueError:
        print("invalid_route: rejected")
        return
    raise AssertionError("expected ValueError")


def _make_preds_long(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 300
    rows = []
    for model in ("lgbm", "har"):
        actual = rng.normal(0, 0.3, n)
        rows.append(pd.DataFrame({
            "symbol": np.repeat(["A", "B", "C"], n // 3),
            "date": pd.to_datetime(
                np.tile(pd.date_range("2025-01-01", periods=n // 3), 3)
            ),
            "model": model,
            "predicted_dev": actual + rng.normal(0, 0.1, n),
            "actual_dev": actual,
            "baseline_b": np.full(n, -4.6),
            "actual_lv": np.full(n, -4.6) + actual,
        }))
    return pd.concat(rows, ignore_index=True)


def test_reconciliation_pass_and_breach():
    from model.evaluation import h1_metrics_from_predictions
    from scripts.reconcile_h1_metrics import reconcile

    preds = _make_preds_long()
    metrics = h1_metrics_from_predictions(preds)
    payload = {
        "pipeline": "h1",
        "h1": {
            "deviation_r2_pooled": {m: v["dev_r2_pooled"] for m, v in metrics.items()},
            "deviation_r2_within": {m: v["dev_r2_within"] for m, v in metrics.items()},
            "deviation_r2_ticker_median": {
                m: v["dev_r2_ticker_median"] for m, v in metrics.items()
            },
            "qlike_level": {m: v["qlike_level"] for m, v in metrics.items()},
        },
    }
    assert reconcile(payload, preds) == []

    payload["h1"]["deviation_r2_pooled"]["lgbm"] += 1e-4  # drift beyond 1e-6
    problems = reconcile(payload, preds)
    assert len(problems) == 1 and "deviation_r2_pooled[lgbm]" in problems[0]
    print("reconcile: exact metrics pass, 1e-4 drift caught")


def main() -> int:
    test_route_honored()
    test_missing_lgbm_falls_back_to_har()
    test_invalid_route_rejected()
    test_reconciliation_pass_and_breach()
    print("all h1_predictor tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
