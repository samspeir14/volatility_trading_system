from __future__ import annotations

import logging
import math

import pandas as pd

from model.garch_baseline import GARCHBaseline
from model.lightgbm_model import LightGBMVolPredictor
from model.xgboost_model import XGBoostVolPredictor

logger = logging.getLogger(__name__)


_VALID_MODELS = ("lgbm", "xgboost", "garch")


class BestPredictor:
    """Routes between LightGBM, XGBoost, and GARCH based on most recent OOS R²
    comparison. Priority order: LightGBM beats XGBoost beats GARCH.

    LightGBM and XGBoost are both optional — if a model artifact is missing
    on disk, pass None for that argument and the router will skip it. GARCH
    is always available because it has no artifact (fit happens online).
    """

    def __init__(
        self,
        lgbm: LightGBMVolPredictor | None,
        xgb: XGBoostVolPredictor | None,
        garch: GARCHBaseline,
        horizon: int,
    ):
        self._lgbm = lgbm
        self._xgb = xgb
        self._garch = garch
        self._horizon = horizon
        # Default: LGBM if available, else XGB if available, else GARCH.
        if lgbm is not None:
            self._active = "lgbm"
        elif xgb is not None:
            self._active = "xgboost"
        else:
            self._active = "garch"

    def update_from_eval(
        self,
        lgbm_r2: float = float("nan"),
        xgb_r2: float = float("nan"),
        garch_r2: float = float("nan"),
    ) -> None:
        """Pick the model with highest finite R² that we actually have loaded.
        Order: lgbm > xgb > garch when R²s tie. Models we don't have are
        excluded regardless of their reported R²."""
        prev = self._active
        candidates: list[tuple[str, float]] = []
        if self._lgbm is not None and not math.isnan(lgbm_r2):
            candidates.append(("lgbm", lgbm_r2))
        if self._xgb is not None and not math.isnan(xgb_r2):
            candidates.append(("xgboost", xgb_r2))
        # GARCH is always a valid fallback even if r2 is NaN (use sentinel).
        garch_score = garch_r2 if not math.isnan(garch_r2) else -float("inf")
        candidates.append(("garch", garch_score))

        # Pick highest score; ties broken by insertion order (lgbm > xgb > garch).
        best = max(candidates, key=lambda c: c[1])
        self._active = best[0]

        if prev != self._active:
            logger.warning(
                "BestPredictor flip h=%d: %s -> %s "
                "(lgbm_r2=%.3f, xgb_r2=%.3f, garch_r2=%.3f)",
                self._horizon, prev, self._active,
                lgbm_r2, xgb_r2, garch_r2,
            )

    def predict_forward_rv(
        self,
        returns_history: pd.Series,
        X_row: pd.DataFrame | None = None,
    ) -> float:
        if self._active == "lgbm":
            if X_row is None:
                raise ValueError("LightGBM route requires X_row (a single-row DataFrame of features)")
            return float(self._lgbm.predict(X_row)[0])
        if self._active == "xgboost":
            if X_row is None:
                raise ValueError("XGBoost route requires X_row (a single-row DataFrame of features)")
            return float(self._xgb.predict(X_row)[0])
        return self._garch.predict_forward_rv(returns_history, self._horizon)

    @property
    def active_model(self) -> str:
        return self._active
