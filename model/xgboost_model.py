from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class XGBoostVolPredictor:
    horizon: int
    hyperparams: dict
    feature_columns: list[str] = field(default_factory=list)
    _model: xgb.XGBRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_columns = list(X.columns)
        self._model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=0,
            **self.hyperparams,
        )
        self._model.fit(X, y, verbose=False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predict() called before fit()")
        if list(X.columns) != self.feature_columns:
            X = X[self.feature_columns]
        return self._model.predict(X)

    def feature_importance(self) -> pd.Series:
        if self._model is None:
            raise RuntimeError("feature_importance() called before fit()")
        return pd.Series(
            self._model.feature_importances_,
            index=self.feature_columns,
        ).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("save() called before fit()")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "horizon": self.horizon,
                "hyperparams": self.hyperparams,
                "feature_columns": self.feature_columns,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> XGBoostVolPredictor:
        bundle = joblib.load(path)
        instance = cls(
            horizon=bundle["horizon"],
            hyperparams=bundle["hyperparams"],
            feature_columns=bundle["feature_columns"],
        )
        instance._model = bundle["model"]
        return instance
