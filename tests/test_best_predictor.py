import sys

import numpy as np
import pandas as pd

from model.best_predictor import BestPredictor
from model.garch_baseline import GARCHBaseline
from model.lightgbm_model import LightGBMVolPredictor
from model.training import DEFAULT_HYPERPARAMS, DEFAULT_LGBM_HYPERPARAMS
from model.xgboost_model import XGBoostVolPredictor


def _fit_xgb() -> XGBoostVolPredictor:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
    y = pd.Series(rng.normal(0, 0.01, 100))
    p = XGBoostVolPredictor(horizon=21, hyperparams=DEFAULT_HYPERPARAMS)
    p.fit(X, y)
    return p


def _fit_lgbm() -> LightGBMVolPredictor:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
    y = pd.Series(rng.normal(0, 0.01, 100))
    p = LightGBMVolPredictor(horizon=21, hyperparams=DEFAULT_LGBM_HYPERPARAMS)
    p.fit(X, y)
    return p


def _make_full() -> BestPredictor:
    return BestPredictor(
        lgbm=_fit_lgbm(), xgb=_fit_xgb(),
        garch=GARCHBaseline(refit_every=21, min_history=100),
        horizon=21,
    )


def test_default_routes_to_lgbm_when_available():
    bp = _make_full()
    assert bp.active_model == "lgbm"
    print("default: routes to LightGBM when artifact loaded")


def test_lgbm_wins_when_highest_r2():
    bp = _make_full()
    bp.update_from_eval(lgbm_r2=0.35, xgb_r2=0.30, garch_r2=-0.1)
    assert bp.active_model == "lgbm", f"expected lgbm, got {bp.active_model}"
    print("lgbm_wins: routes to LightGBM when its R² is highest")


def test_xgb_wins_when_lgbm_lower():
    bp = _make_full()
    bp.update_from_eval(lgbm_r2=0.20, xgb_r2=0.30, garch_r2=-0.1)
    assert bp.active_model == "xgboost", f"expected xgboost, got {bp.active_model}"
    print("xgb_wins: routes to XGBoost when its R² beats LightGBM and GARCH")


def test_garch_wins_when_all_others_negative():
    bp = _make_full()
    bp.update_from_eval(lgbm_r2=-0.5, xgb_r2=-0.3, garch_r2=0.1)
    assert bp.active_model == "garch", f"expected garch, got {bp.active_model}"
    print("garch_wins: routes to GARCH when others are worse")


def test_lgbm_nan_falls_through_to_xgb():
    bp = _make_full()
    bp.update_from_eval(lgbm_r2=float("nan"), xgb_r2=0.20, garch_r2=-0.1)
    assert bp.active_model == "xgboost", f"expected xgboost, got {bp.active_model}"
    print("lgbm_nan: falls through to XGBoost when LightGBM R² is NaN")


def test_all_nan_falls_to_garch():
    bp = _make_full()
    bp.update_from_eval(lgbm_r2=float("nan"), xgb_r2=float("nan"), garch_r2=float("nan"))
    assert bp.active_model == "garch", "all-NaN must fall back to GARCH"
    print("all_nan: falls back to GARCH")


def test_no_lgbm_artifact_uses_xgb():
    """If LGBM artifact missing on disk, BestPredictor still routes correctly."""
    bp = BestPredictor(
        lgbm=None, xgb=_fit_xgb(),
        garch=GARCHBaseline(refit_every=21, min_history=100),
        horizon=21,
    )
    assert bp.active_model == "xgboost", "default with no lgbm should be xgboost"
    bp.update_from_eval(lgbm_r2=0.99, xgb_r2=0.30, garch_r2=-0.1)
    assert bp.active_model == "xgboost", "missing lgbm cannot be selected even if R² claims highest"
    print("no_lgbm: ignores lgbm even when its claimed R² is highest")


def test_no_artifacts_garch_only():
    bp = BestPredictor(
        lgbm=None, xgb=None,
        garch=GARCHBaseline(refit_every=21, min_history=100),
        horizon=21,
    )
    assert bp.active_model == "garch"
    bp.update_from_eval(lgbm_r2=0.99, xgb_r2=0.99, garch_r2=-0.1)
    assert bp.active_model == "garch"
    print("no_artifacts: routes to GARCH and stays there")


def test_flip_logs_warning(caplog=None):
    """Flip from xgb to lgbm should log a WARNING."""
    import logging
    bp = _make_full()
    bp.update_from_eval(lgbm_r2=0.20, xgb_r2=0.30, garch_r2=-0.1)
    assert bp.active_model == "xgboost"
    # Now make lgbm best — should flip and log
    logger = logging.getLogger("model.best_predictor")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r)
    logger.addHandler(handler)
    try:
        bp.update_from_eval(lgbm_r2=0.40, xgb_r2=0.30, garch_r2=-0.1)
    finally:
        logger.removeHandler(handler)
    assert bp.active_model == "lgbm"
    assert any("flip" in r.getMessage().lower() and r.levelno == logging.WARNING
               for r in records), "flip should emit a WARNING"
    print("flip_logs_warning: WARNING emitted on model change")


def test_predict_lgbm_route_requires_X_row():
    bp = _make_full()
    bp.update_from_eval(lgbm_r2=0.5, xgb_r2=0.0, garch_r2=-1.0)
    assert bp.active_model == "lgbm"
    try:
        bp.predict_forward_rv(returns_history=pd.Series([0.01] * 50), X_row=None)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "X_row" in str(e)
    print("predict_lgbm_route_requires_X_row: raises ValueError as expected")


def main() -> int:
    test_default_routes_to_lgbm_when_available()
    test_lgbm_wins_when_highest_r2()
    test_xgb_wins_when_lgbm_lower()
    test_garch_wins_when_all_others_negative()
    test_lgbm_nan_falls_through_to_xgb()
    test_all_nan_falls_to_garch()
    test_no_lgbm_artifact_uses_xgb()
    test_no_artifacts_garch_only()
    test_flip_logs_warning()
    test_predict_lgbm_route_requires_X_row()
    print("all best_predictor tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
