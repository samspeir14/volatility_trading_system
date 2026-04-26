import math
import sys

import numpy as np
import pandas as pd

from model.best_predictor import BestPredictor
from model.garch_baseline import GARCHBaseline
from model.training import DEFAULT_HYPERPARAMS
from model.xgboost_model import XGBoostVolPredictor


def _make_predictors() -> tuple[BestPredictor, GARCHBaseline, XGBoostVolPredictor]:
    garch = GARCHBaseline(refit_every=21, min_history=100)
    xgb_pred = XGBoostVolPredictor(horizon=21, hyperparams=DEFAULT_HYPERPARAMS)
    # Fit XGBoost on tiny synthetic data so .predict works
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
    y = pd.Series(rng.normal(0, 0.01, 100))
    xgb_pred.fit(X, y)
    return BestPredictor(garch, xgb_pred, horizon=21), garch, xgb_pred


def test_initial_state_routes_to_garch():
    bp, _, _ = _make_predictors()
    assert bp.active_model == "garch"
    print("initial_state: routes to GARCH")


def test_xgb_wins_when_higher_r2():
    bp, _, _ = _make_predictors()
    bp.update_from_eval(garch_r2=0.1, xgb_r2=0.2)
    assert bp.active_model == "xgboost", f"expected xgboost, got {bp.active_model}"
    print("xgb_wins: routes to XGBoost when R²_xgb > R²_garch")


def test_garch_wins_when_higher_r2():
    bp, _, _ = _make_predictors()
    bp.update_from_eval(garch_r2=0.2, xgb_r2=0.1)
    assert bp.active_model == "garch", f"expected garch, got {bp.active_model}"
    print("garch_wins: routes to GARCH when R²_garch > R²_xgb")


def test_xgb_nan_routes_to_garch():
    bp, _, _ = _make_predictors()
    bp.update_from_eval(garch_r2=-0.5, xgb_r2=float("nan"))
    assert bp.active_model == "garch", "NaN xgb R² must fall back to GARCH"
    print("xgb_nan: routes to GARCH when R²_xgb is NaN")


def test_flip_back_and_forth():
    bp, _, _ = _make_predictors()
    bp.update_from_eval(0.1, 0.2)
    assert bp.active_model == "xgboost"
    bp.update_from_eval(0.3, 0.1)
    assert bp.active_model == "garch"
    bp.update_from_eval(0.0, 0.05)
    assert bp.active_model == "xgboost"
    print("flip_back_and_forth: routing follows latest comparison")


def test_predict_xgb_route_requires_X_row():
    bp, _, _ = _make_predictors()
    bp.update_from_eval(garch_r2=-1.0, xgb_r2=0.5)
    assert bp.active_model == "xgboost"
    try:
        bp.predict_forward_rv(returns_history=pd.Series([0.01] * 50), X_row=None)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "X_row" in str(e)
    print("predict_xgb_route_requires_X_row: raises ValueError as expected")


def main() -> int:
    test_initial_state_routes_to_garch()
    test_xgb_wins_when_higher_r2()
    test_garch_wins_when_higher_r2()
    test_xgb_nan_routes_to_garch()
    test_flip_back_and_forth()
    test_predict_xgb_route_requires_X_row()
    print("all best_predictor tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
