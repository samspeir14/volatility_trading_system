"""HAR-RV model on the h=1 deviation target.

Pooled OLS of tomorrow's log-vol deviation on the demeaned HAR components
(1/5/22-day trailing means of log GK vol, each minus the 63-day baseline
b_t). Demeaning both sides makes the pooled regression legitimate across
tickers, mirroring the LightGBM setup. This is the acceptance-gate
benchmark: the LightGBM route must beat it on out-of-sample within-ticker
deviation R².
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_TYPE = "har"

# Demeaned HAR components produced by FeaturePipeline (dev_gk is the 1-day
# component: log_gk_1 - b_t == har_dev_1).
H1_HAR_FEATURES: list[str] = ["dev_gk", "har_dev_5", "har_dev_22"]


@dataclass
class HARRVPredictor:
    feature_columns: list[str] = field(
        default_factory=lambda: list(H1_HAR_FEATURES)
    )
    intercept: float = 0.0
    coefs: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xs = X[self.feature_columns]
        mask = ~(Xs.isna().any(axis=1) | y.isna())
        Xv = Xs.loc[mask].to_numpy(dtype=float)
        yv = y.loc[mask].to_numpy(dtype=float)
        if len(yv) < len(self.feature_columns) + 1:
            raise ValueError(
                f"need >{len(self.feature_columns)} non-NaN rows, got {len(yv)}"
            )
        A = np.column_stack([np.ones(len(Xv)), Xv])
        sol, *_ = np.linalg.lstsq(A, yv, rcond=None)
        self.intercept = float(sol[0])
        self.coefs = sol[1:]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coefs is None:
            raise RuntimeError("predict() called before fit()")
        Xv = X[self.feature_columns].to_numpy(dtype=float)
        return self.intercept + Xv @ self.coefs

    def feature_importance(self) -> pd.Series:
        """|coefficient| per feature — API parity with the tree models."""
        if self.coefs is None:
            raise RuntimeError("feature_importance() called before fit()")
        return pd.Series(
            np.abs(self.coefs), index=self.feature_columns
        ).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        if self.coefs is None:
            raise RuntimeError("save() called before fit()")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_type": MODEL_TYPE,
                "feature_columns": self.feature_columns,
                "intercept": self.intercept,
                "coefs": self.coefs,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "HARRVPredictor":
        bundle = joblib.load(path)
        if bundle.get("model_type") != MODEL_TYPE:
            raise ValueError(
                f"artifact at {path} has model_type={bundle.get('model_type')!r}, "
                f"expected {MODEL_TYPE!r}"
            )
        instance = cls(feature_columns=bundle["feature_columns"])
        instance.intercept = bundle["intercept"]
        instance.coefs = bundle["coefs"]
        return instance
