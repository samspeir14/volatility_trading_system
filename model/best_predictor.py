from __future__ import annotations

import logging
import math

import pandas as pd

from model.garch_baseline import GARCHBaseline
from model.xgboost_model import XGBoostVolPredictor

logger = logging.getLogger(__name__)


class BestPredictor:
    """Routes between GARCH baseline and XGBoost based on most recent OOS R²
    comparison. Defaults to GARCH until XGBoost demonstrably beats it."""

    def __init__(
        self,
        garch: GARCHBaseline,
        xgb: XGBoostVolPredictor,
        horizon: int,
    ):
        self._garch = garch
        self._xgb = xgb
        self._horizon = horizon
        self._use_xgb = False

    def update_from_eval(self, garch_r2: float, xgb_r2: float) -> None:
        prev = self._use_xgb
        if math.isnan(xgb_r2):
            self._use_xgb = False
        else:
            self._use_xgb = xgb_r2 > garch_r2
        if prev != self._use_xgb:
            logger.warning(
                "BestPredictor flip: now using %s (garch_r2=%.3f, xgb_r2=%.3f)",
                "XGBoost" if self._use_xgb else "GARCH", garch_r2, xgb_r2,
            )

    def predict_forward_rv(
        self,
        returns_history: pd.Series,
        X_row: pd.DataFrame | None = None,
    ) -> float:
        if self._use_xgb:
            if X_row is None:
                raise ValueError("XGBoost route requires X_row (a single-row DataFrame of features)")
            return float(self._xgb.predict(X_row)[0])
        return self._garch.predict_forward_rv(returns_history, self._horizon)

    @property
    def active_model(self) -> str:
        return "xgboost" if self._use_xgb else "garch"
