from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd


class VolPredictorProtocol(Protocol):
    horizon: int
    hyperparams: dict
    feature_columns: list[str]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def feature_importance(self) -> pd.Series: ...


LGBM_SPACE: dict[str, list] = {
    "max_depth": [3, 4, 5, 6, -1],
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 300, 500],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 5.0, 10.0],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_samples": [5, 10, 20],
}


CATBOOST_SPACE: dict[str, list] = {
    "depth": [4, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "iterations": [200, 500, 1000],
    "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
    "subsample": [0.6, 0.8, 1.0],
    "random_strength": [0.0, 0.5, 1.0],
}


DEFAULT_LGBM_HYPERPARAMS: dict = {
    "max_depth": -1, "num_leaves": 31, "learning_rate": 0.05,
    "n_estimators": 200, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
}


DEFAULT_CATBOOST_HYPERPARAMS: dict = {
    "depth": 6, "learning_rate": 0.05, "iterations": 500,
    "l2_leaf_reg": 3.0, "subsample": 0.8, "random_strength": 1.0,
}


@dataclass
class LightGBMVolPredictor:
    horizon: int
    hyperparams: dict
    feature_columns: list[str] = field(default_factory=list)
    _model: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import lightgbm as lgb

        self.feature_columns = list(X.columns)
        params = dict(self.hyperparams)
        # Bagging requires bagging_freq>0 in LightGBM; map sklearn-style names
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
        if self._model is None:
            raise RuntimeError("feature_importance() called before fit()")
        # Force gain-based to align with XGBoost default
        booster = self._model.booster_
        gains = booster.feature_importance(importance_type="gain")
        return pd.Series(gains, index=self.feature_columns).sort_values(ascending=False)


@dataclass
class CatBoostVolPredictor:
    horizon: int
    hyperparams: dict
    feature_columns: list[str] = field(default_factory=list)
    _model: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from catboost import CatBoostRegressor

        self.feature_columns = list(X.columns)
        # CatBoost cannot accept NaN in target; drop those rows defensively
        mask = y.notna()
        X_fit = X.loc[mask]
        y_fit = y.loc[mask]
        self._model = CatBoostRegressor(
            loss_function="RMSE",
            random_seed=0,
            verbose=False,
            allow_writing_files=False,
            **self.hyperparams,
        )
        # CatBoost handles NaN in features natively when allow_const_label and nan_mode default
        self._model.fit(X_fit, y_fit)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predict() called before fit()")
        if list(X.columns) != self.feature_columns:
            X = X[self.feature_columns]
        return self._model.predict(X)

    def feature_importance(self) -> pd.Series:
        if self._model is None:
            raise RuntimeError("feature_importance() called before fit()")
        # PredictionValuesChange is the closest analog to gain
        from catboost import EFstrType
        importances = self._model.get_feature_importance(type=EFstrType.PredictionValuesChange)
        return pd.Series(importances, index=self.feature_columns).sort_values(ascending=False)
