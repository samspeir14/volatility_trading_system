import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from model.training import DEFAULT_HYPERPARAMS
from model.xgboost_model import XGBoostVolPredictor


def _make_data(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "c": rng.normal(0, 1, n),
    })
    # noisy linear target
    y = pd.Series(0.5 * X["a"] + 0.3 * X["b"] - 0.2 * X["c"] + rng.normal(0, 0.1, n))
    return X, y


def test_fit_then_predict():
    X, y = _make_data()
    pred = XGBoostVolPredictor(horizon=21, hyperparams=DEFAULT_HYPERPARAMS)
    pred.fit(X, y)
    preds = pred.predict(X)
    assert preds.shape == (len(X),), f"unexpected shape {preds.shape}"
    assert pred.feature_columns == ["a", "b", "c"]
    print(f"fit_then_predict: shape OK, columns recorded")


def test_feature_importance_nonzero():
    X, y = _make_data()
    pred = XGBoostVolPredictor(horizon=21, hyperparams=DEFAULT_HYPERPARAMS)
    pred.fit(X, y)
    imp = pred.feature_importance()
    assert (imp.sum() > 0), "feature importances should not all be zero"
    assert set(imp.index) == {"a", "b", "c"}
    print(f"feature_importance: top = {imp.idxmax()} ({imp.max():.3f})")


def test_save_load_roundtrip():
    X, y = _make_data()
    pred = XGBoostVolPredictor(horizon=10, hyperparams=DEFAULT_HYPERPARAMS)
    pred.fit(X, y)
    expected = pred.predict(X)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.joblib"
        pred.save(path)
        assert path.exists()
        loaded = XGBoostVolPredictor.load(path)
        assert loaded.horizon == 10
        assert loaded.hyperparams == DEFAULT_HYPERPARAMS
        assert loaded.feature_columns == ["a", "b", "c"]
        loaded_preds = loaded.predict(X)
        np.testing.assert_allclose(expected, loaded_preds, rtol=1e-12)
    print("save_load_roundtrip: predictions match to 1e-12 after reload")


def test_deterministic_with_seed():
    X, y = _make_data()
    p1 = XGBoostVolPredictor(horizon=21, hyperparams=DEFAULT_HYPERPARAMS)
    p1.fit(X, y)
    p2 = XGBoostVolPredictor(horizon=21, hyperparams=DEFAULT_HYPERPARAMS)
    p2.fit(X, y)
    np.testing.assert_allclose(p1.predict(X), p2.predict(X), rtol=1e-12)
    print("deterministic: identical predictions across two separate fits with same seed")


def test_predict_before_fit_raises():
    pred = XGBoostVolPredictor(horizon=21, hyperparams=DEFAULT_HYPERPARAMS)
    try:
        pred.predict(pd.DataFrame({"a": [1.0]}))
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "before fit" in str(e)
    print("predict_before_fit: raises RuntimeError as expected")


def main() -> int:
    test_fit_then_predict()
    test_feature_importance_nonzero()
    test_save_load_roundtrip()
    test_deterministic_with_seed()
    test_predict_before_fit_raises()
    print("all xgboost_model tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
