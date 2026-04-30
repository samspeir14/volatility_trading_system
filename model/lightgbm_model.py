from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


MODEL_TYPE = "lightgbm"


@dataclass
class LightGBMVolPredictor:
    horizon: int
    hyperparams: dict
    feature_columns: list[str] = field(default_factory=list)
    _model: lgb.LGBMRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_columns = list(X.columns)
        params = dict(self.hyperparams)
        # LightGBM requires bagging_freq>0 for subsample<1 to take effect.
        if params.get("subsample", 1.0) < 1.0 and "bagging_freq" not in params:
            params["bagging_freq"] = 1
        self._model = lgb.LGBMRegressor(
            objective="regression",
            random_state=0,
            verbose=-1,
            **params,
        )
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predict() called before fit()")
        if list(X.columns) != self.feature_columns:
            X = X[self.feature_columns]
        return self._model.predict(X)

    def feature_importance(self) -> pd.Series:
        """Gain-based importance, comparable across XGB/LGBM/CatBoost gain-equivalents."""
        if self._model is None:
            raise RuntimeError("feature_importance() called before fit()")
        booster = self._model.booster_
        gains = booster.feature_importance(importance_type="gain")
        return pd.Series(gains, index=self.feature_columns).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("save() called before fit()")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_type": MODEL_TYPE,
                "model": self._model,
                "horizon": self.horizon,
                "hyperparams": self.hyperparams,
                "feature_columns": self.feature_columns,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> LightGBMVolPredictor:
        bundle = joblib.load(path)
        if bundle.get("model_type") not in (MODEL_TYPE, None):
            raise ValueError(
                f"artifact at {path} has model_type={bundle.get('model_type')!r}, "
                f"expected {MODEL_TYPE!r}"
            )
        instance = cls(
            horizon=bundle["horizon"],
            hyperparams=bundle["hyperparams"],
            feature_columns=bundle["feature_columns"],
        )
        instance._model = bundle["model"]
        return instance
