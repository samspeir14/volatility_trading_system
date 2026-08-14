import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from model.lightgbm_model import LightGBMVolPredictor
from model.training import DEFAULT_LGBM_HYPERPARAMS


def _make_data(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "c": rng.normal(0, 1, n),
    })
    y = pd.Series(0.5 * X["a"] + 0.3 * X["b"] - 0.2 * X["c"] + rng.normal(0, 0.1, n))
    return X, y


def test_fit_then_predict():
    X, y = _make_data()
    pred = LightGBMVolPredictor(horizon=21, hyperparams=DEFAULT_LGBM_HYPERPARAMS)
    pred.fit(X, y)
    preds = pred.predict(X)
    assert preds.shape == (len(X),), f"unexpected shape {preds.shape}"
    assert pred.feature_columns == ["a", "b", "c"]
    print("fit_then_predict: shape OK, columns recorded")


def test_feature_importance_nonzero():
    X, y = _make_data()
    pred = LightGBMVolPredictor(horizon=21, hyperparams=DEFAULT_LGBM_HYPERPARAMS)
    pred.fit(X, y)
    imp = pred.feature_importance()
    assert imp.sum() > 0, "feature importances should not all be zero"
    assert set(imp.index) == {"a", "b", "c"}
    print(f"feature_importance: top = {imp.idxmax()} ({imp.max():.3f})")


def test_save_load_roundtrip():
    X, y = _make_data()
    pred = LightGBMVolPredictor(horizon=10, hyperparams=DEFAULT_LGBM_HYPERPARAMS)
    pred.fit(X, y)
    expected = pred.predict(X)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.joblib"
        pred.save(path)
        assert path.exists()
        loaded = LightGBMVolPredictor.load(path)
        assert loaded.horizon == 10
        assert loaded.hyperparams == DEFAULT_LGBM_HYPERPARAMS
        assert loaded.feature_columns == ["a", "b", "c"]
        loaded_preds = loaded.predict(X)
        np.testing.assert_allclose(expected, loaded_preds, rtol=1e-9)
    print("save_load_roundtrip: predictions match to 1e-9 after reload")


def test_inverse_variance_weights():
    """Weighted fit runs, weights are per-ticker inverse variance (mean 1),
    the flag survives an artifact roundtrip, and a non-MultiIndex y is
    rejected loudly."""
    rng = np.random.default_rng(2)
    n = 120
    idx = pd.MultiIndex.from_product(
        [["CALM", "WILD"], pd.date_range("2024-01-01", periods=n // 2)],
        names=["symbol", "date"],
    )
    X = pd.DataFrame({c: rng.normal(0, 1, n) for c in ("a", "b", "c")}, index=idx)
    # WILD has 10x the deviation scale of CALM
    y = pd.Series(
        np.concatenate([rng.normal(0, 0.1, n // 2), rng.normal(0, 1.0, n // 2)]),
        index=idx,
    )
    w = LightGBMVolPredictor._inverse_variance_weights(y)
    assert abs(w.mean() - 1.0) < 1e-9
    assert w[: n // 2].mean() > w[n // 2 :].mean(), "CALM rows must outweigh WILD"

    pred = LightGBMVolPredictor(
        horizon=1, hyperparams=DEFAULT_LGBM_HYPERPARAMS,
        inverse_variance_weights=True,
    )
    pred.fit(X, y)
    assert pred.predict(X).shape == (n,)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ivw.joblib"
        pred.save(path)
        assert LightGBMVolPredictor.load(path).inverse_variance_weights is True

    flat_y = pd.Series(rng.normal(0, 1, 10))
    try:
        LightGBMVolPredictor._inverse_variance_weights(flat_y)
        raise AssertionError("expected ValueError for missing symbol level")
    except ValueError as e:
        assert "symbol" in str(e)
    print("inverse_variance_weights: weighting, roundtrip, and guard OK")


def test_deterministic_with_seed():
    X, y = _make_data()
    p1 = LightGBMVolPredictor(horizon=21, hyperparams=DEFAULT_LGBM_HYPERPARAMS)
    p1.fit(X, y)
    p2 = LightGBMVolPredictor(horizon=21, hyperparams=DEFAULT_LGBM_HYPERPARAMS)
    p2.fit(X, y)
    np.testing.assert_allclose(p1.predict(X), p2.predict(X), rtol=1e-9)
    print("deterministic: identical predictions across two separate fits with same seed")


def test_predict_before_fit_raises():
    pred = LightGBMVolPredictor(horizon=21, hyperparams=DEFAULT_LGBM_HYPERPARAMS)
    try:
        pred.predict(pd.DataFrame({"a": [1.0]}))
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "before fit" in str(e)
    print("predict_before_fit: raises RuntimeError as expected")


def test_load_rejects_wrong_model_type():
    """If someone tries to load an XGBoost artifact via LightGBMVolPredictor.load,
    the wrapper raises rather than silently producing wrong predictions."""
    import joblib
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wrong.joblib"
        joblib.dump({"model_type": "xgboost", "model": None,
                     "horizon": 5, "hyperparams": {}, "feature_columns": []}, path)
        try:
            LightGBMVolPredictor.load(path)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "model_type" in str(e)
    print("load_rejects_wrong_model_type: raises ValueError on xgb bundle")


def main() -> int:
    test_fit_then_predict()
    test_feature_importance_nonzero()
    test_save_load_roundtrip()
    test_inverse_variance_weights()
    test_deterministic_with_seed()
    test_predict_before_fit_raises()
    test_load_rejects_wrong_model_type()
    print("all lightgbm_model tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
